#include "DXUT.h"
#include "PSVSM.hpp"
#include <cmath>
#include <limits>

//--------------------------------------------------------------------------------------
PSVSM::PSVSM(ID3D10Device* d3dDevice,
             ID3D10Effect* Effect,
             int Width, int Height,
             PostProcess* PostProcess,
             const DXGI_SAMPLE_DESC* SampleDesc,
             unsigned int NumSplits)
  : VSM(d3dDevice, Effect, Width, Height, PostProcess,
        DXGI_FORMAT_R32G32_FLOAT, SampleDesc, DXGI_FORMAT_R32G32_FLOAT, true)
  , m_NumSplits(NumSplits), m_CurSplit(0), m_SplitLambda(0.5f)
{
  HRESULT hr;

  // Setup effect
  m_EffectShadowMap = m_Effect->GetVariableByName("texShadow")->AsShaderResource();
  assert(m_EffectShadowMap && m_EffectShadowMap->IsValid());
  m_EffectSplits = m_Effect->GetVariableByName("g_Splits")->AsVector();
  assert(m_EffectSplits && m_EffectSplits->IsValid());
  m_EffectSplitMatrices = m_Effect->GetVariableByName("g_SplitMatrices")->AsMatrix();
  assert(m_EffectSplitMatrices && m_EffectSplitMatrices->IsValid());  
  m_EffectVisualizeSplits = m_Effect->GetVariableByName("g_VisualizeSplits")->AsScalar();
  assert(m_EffectVisualizeSplits && m_EffectVisualizeSplits->IsValid());

  // Optional displaying shadow map interface
  m_DisplayTechnique = m_Effect->GetTechniqueByName("DisplayShadowMap");
  m_EffectDisplayArrayIndex = m_Effect->GetVariableByName("g_DisplayArrayIndex")->AsScalar();

  // Effect initial values
  SetVisualizeSplits(false);


  // Create texture array and shader resource
  D3D10_TEXTURE2D_DESC texDesc;
  texDesc.Width              = m_Width;
  texDesc.Height             = m_Height;
  texDesc.MipLevels          = 0;                // Full chain
  texDesc.ArraySize          = m_NumSplits + 1;  // One extra as a temporary texture!
  texDesc.Format             = m_ShadowFormat;
  texDesc.SampleDesc.Count   = 1;                // After potential resolve
  texDesc.SampleDesc.Quality = 0;
  texDesc.Usage              = D3D10_USAGE_DEFAULT;
  texDesc.BindFlags          = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
  texDesc.CPUAccessFlags     = 0;
  texDesc.MiscFlags          = D3D10_RESOURCE_MISC_GENERATE_MIPS;

  V(d3dDevice->CreateTexture2D(&texDesc, NULL, &m_ShadowMapArray));
  // Update the description with the read number of mipmaps, etc.
  m_ShadowMapArray->GetDesc(&texDesc);

  // Create the shader-resource view (for all slices)
  D3D10_SHADER_RESOURCE_VIEW_DESC srDesc;
  srDesc.Format                         = texDesc.Format;
  srDesc.ViewDimension                  = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
  srDesc.Texture2DArray.MostDetailedMip = 0;
  srDesc.Texture2DArray.MipLevels       = texDesc.MipLevels;
  srDesc.Texture2DArray.FirstArraySlice = 0;
  srDesc.Texture2DArray.ArraySize       = m_NumSplits;   // Last item is temporary... remember

  V(d3dDevice->CreateShaderResourceView(m_ShadowMapArray, &srDesc,
                                        &m_ShadowMapArrayResource));

  // Save some info for later
  m_MipLevels = texDesc.MipLevels;

  // Create render target and shader resource views for each slice
  for (unsigned int i = 0; i < texDesc.ArraySize; ++i) {
    m_ShadowMapSlices.push_back(new RenderableTexture2D(d3dDevice, m_ShadowMapArray, i));
  }
}

//--------------------------------------------------------------------------------------
PSVSM::~PSVSM()
{
  for (std::size_t i = 0; i < m_ShadowMapSlices.size(); ++i) {
    SAFE_DELETE(m_ShadowMapSlices[i]);
  }
  SAFE_RELEASE(m_ShadowMapArrayResource);
  SAFE_RELEASE(m_ShadowMapArray);
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  PSVSM::BeginShadowMap(D3DXMATRIXA16& LightView,
                                              D3DXMATRIXA16& LightProj,
                                              const D3DXVECTOR3& LightPos,
                                              float LightLinFar,
                                              const CBaseCamera& Camera)
{
  // If this is the first time through this frame, compute split information
  if (m_CurSplit == 0) {
    const D3DXMATRIXA16& CameraView = *Camera.GetViewMatrix();
    const D3DXMATRIXA16& CameraProj = *Camera.GetProjMatrix();
    
    RecomputeSplitDistances(Camera.GetNearClip(), Camera.GetFarClip(), m_SplitLambda);
    RecomputeSplitMatrices(CameraView, CameraProj, LightView, LightProj);
  }

  // Use the projection matrix for this slice
  LightProj = m_SplitProj[m_CurSplit];

  // Set the current shadow map to render into
  m_ShadowMap = m_ShadowMapSlices[m_CurSplit];

  // Call parent
  return VSM::BeginShadowMap(LightView, LightProj, LightPos,
                             LightLinFar, Camera);
}

//--------------------------------------------------------------------------------------
bool PSVSM::EndShadowMap(ID3D10EffectTechnique* Technique)
{
  VSM::EndShadowMap(Technique);

  // Compute minimum filter width for this split
  D3DXVECTOR2 SplitMinFilterWidth = m_MinFilterWidth * m_SplitScales[m_CurSplit];

  // Blur if required
  if (SplitMinFilterWidth.x > 1.0f || SplitMinFilterWidth.y > 1.0f) {
    // Use the last array slice (which is extra - see constructor) for temporary
    BoxBlur(m_ShadowMap, m_ShadowMapSlices.back(), SplitMinFilterWidth);
  }

  // Done with m_ShadowMap (the current single slice)
  m_ShadowMap = 0;

  // Are we done rendering shadow maps?
  ++m_CurSplit;
  if (m_CurSplit >= m_NumSplits) {
    // Generate mipmaps for all array slices
    m_d3dDevice->GenerateMips(m_ShadowMapArrayResource);

    // All done
    m_CurSplit = 0;
    return false;
  } else {
    // More slices to render!
    return true;
  }
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  PSVSM::BeginShading()
{
  // Bind shadow texture array
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(m_ShadowMapArrayResource));

  return Filtering::BeginShading();
}

//--------------------------------------------------------------------------------------
void PSVSM::EndShading(ID3D10EffectTechnique* Technique)
{
  // Unbind shadow texture array
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(0));

  VSM::EndShading(Technique);
}

//--------------------------------------------------------------------------------------
void PSVSM::RecomputeSplitDistances(float SplitNear, float SplitFar,
                                    float Lambda)
{
  m_SplitDistances.resize(0);

  // Implements the "practical" split scheme from the PSSM paper
  float SplitRange = SplitFar - SplitNear;
  float SplitRatio = SplitFar / SplitNear;

  for (unsigned int i = 0; i < m_NumSplits; ++i) {
    float p = i / static_cast<float>(m_NumSplits);
    float LogSplit     = SplitNear * std::pow(SplitRatio, p);
    float UniformSplit = SplitNear + SplitRange * p;
    // Lerp between the two schemes
    float Split        = Lambda * (LogSplit - UniformSplit) + UniformSplit;
    m_SplitDistances.push_back(Split);
  }

  // Just for simplicty later, push the camera far plane as the last "split"
  m_SplitDistances.push_back(SplitFar);

  // Update the effect variable
  // NOTE: Single 4-tuple, so maximum of five splits right now
  float EffectSplits[4];
  float CameraSplitRatio = SplitFar / (SplitFar - SplitNear);

  unsigned int EffectSplitIndex;
  for (EffectSplitIndex = 1; EffectSplitIndex < (m_SplitDistances.size()-1);
       ++EffectSplitIndex) {
    // NOTE: Have to rescale the splits. The effect expects them to be in post-projection,
    // but pre-homogenious divide [0,...,far] range, relative to the REAL camera near
    // and far.
    float Value = (m_SplitDistances[EffectSplitIndex] - SplitNear) * CameraSplitRatio;
    EffectSplits[EffectSplitIndex-1] = Value;
  }
  // Fill in the rest with "a large number"
  float MaxFloat = std::numeric_limits<float>::max();
  for (; EffectSplitIndex <= 4; ++EffectSplitIndex) {
    EffectSplits[EffectSplitIndex-1] = MaxFloat;
  }
  
  HRESULT hr;
  V(m_EffectSplits->SetFloatVector(EffectSplits));
}

//--------------------------------------------------------------------------------------
void PSVSM::RecomputeSplitMatrices(const D3DXMATRIXA16& CameraView,
                                   const D3DXMATRIXA16& CameraProj,
                                   const D3DXMATRIXA16& LightView,
                                   const D3DXMATRIXA16& LightProj)
{
  m_SplitProj.resize(0);
  m_SplitViewProj.resize(0);
  m_SplitScales.resize(0);

  // Compute the minimum filter width in normalized device coordinates ([-1, 1])
  // NOTE: We compute *half* of this width here for modifying a region later.
  D3DXVECTOR2 HalfMinFilterWidthNDC;
  HalfMinFilterWidthNDC.x = m_MinFilterWidth / static_cast<float>(m_Width);
  HalfMinFilterWidthNDC.y = m_MinFilterWidth / static_cast<float>(m_Height);

  // Extract some useful camera-related information
  D3DXMATRIXA16 CameraViewInv;
  // NOTE: Ideally better to just construct a inverse view matrix from view parameters,
  // but CBaseCamera doesn't seem to provide this functionality...
  D3DXMatrixInverse(&CameraViewInv, 0, &CameraView);
  float XScaleInv = 1.0f / CameraProj._11;
  float YScaleInv = 1.0f / CameraProj._22;

  // Construct a matrix that transforms from camera view space into projected light space
  D3DXMATRIXA16 LightViewProj = LightView * LightProj;
  D3DXMATRIXA16 ViewToProjLightSpace = CameraViewInv * LightViewProj;

  D3DXVECTOR3 Corners[8];
  D3DXVECTOR4 CornersProj[8];
  for (unsigned int i = 0; i < m_NumSplits; ++i) {
    // Compute corners for this frustum
    // TODO: A bit of overlap?
    float Near = m_SplitDistances[i];
    float Far  = m_SplitDistances[i+1];

    // Near corners (in view space)
    float NX = XScaleInv * Near;
    float NY = YScaleInv * Near;
    Corners[0] = D3DXVECTOR3(-NX,  NY, Near);
    Corners[1] = D3DXVECTOR3( NX,  NY, Near);
    Corners[2] = D3DXVECTOR3(-NX, -NY, Near);
    Corners[3] = D3DXVECTOR3( NX, -NY, Near);
    // Far corners (in view space)
    float FX = XScaleInv * Far;
    float FY = YScaleInv * Far;
    Corners[4] = D3DXVECTOR3(-FX,  FY, Far);
    Corners[5] = D3DXVECTOR3( FX,  FY, Far);
    Corners[6] = D3DXVECTOR3(-FX, -FY, Far);
    Corners[7] = D3DXVECTOR3( FX, -FY, Far);

    // Transform corners into projected light space
    D3DXVec3TransformArray(CornersProj, sizeof(D3DXVECTOR4),
                           Corners, sizeof(D3DXVECTOR3),
                           &ViewToProjLightSpace, 8);
  
    // TODO: Adjust Near/Far and corresponding depth scaling
    D3DXVECTOR2 Min( 1,  1);
    D3DXVECTOR2 Max(-1, -1);
    for (unsigned int c = 0; c < 8; ++c) {
      // Homogenious divide x and y
      const D3DXVECTOR4& p = CornersProj[c];
      
      if (p.z < 0.0f) {
        // In front of near clipping plane! Be conservative...
        Min = D3DXVECTOR2(-1, -1);
        Max = D3DXVECTOR2( 1,  1);
        break;
      } else {
        D3DXVECTOR2 v(p.x, p.y);
        v *= 1.0f / p.w;
        // Update boundaries
        D3DXVec2Minimize(&Min, &Min, &v);
        D3DXVec2Maximize(&Max, &Max, &v);
      }
    }

    // Degenerate slice?
    D3DXVECTOR2 Dim = Max - Min;
    if (Max.x <= -1.0f || Max.y <= -1.0f || Min.x >= 1.0f || Min.y >= 1.0f ||
        Dim.x <= 0.0f || Dim.y <= 0.0f) {
      // TODO: Something better... (skip this slice)
      Min = D3DXVECTOR2(-1, -1);
      Max = D3DXVECTOR2( 1,  1);
    }

    // TODO: Clamp extreme magnifications, since they will cause gigantic blurs.
    // Not an issue if we were using PSSAVSM though (i.e. summed-area tables)

    // Expand region by minimum filter width in each dimension to make sure that
    // we can blur properly and get adjacent geometry. Arguably mipmapping will
    // still be wrong for extreme minifications, but that won't be noticable.
    Min -= HalfMinFilterWidthNDC;
    Max += HalfMinFilterWidthNDC;

    // Clamp to valid range
    Min.x = std::min(1.0f, std::max(-1.0f, Min.x));
    Min.y = std::min(1.0f, std::max(-1.0f, Min.y));
    Max.x = std::min(1.0f, std::max(-1.0f, Max.x));
    Max.y = std::min(1.0f, std::max(-1.0f, Max.y));

    // Compute scale and offset
    D3DXVECTOR2 Scale;
    Scale.x = 2.0f / (Max.x - Min.x);
    Scale.y = 2.0f / (Max.y - Min.y);
    D3DXVECTOR2 Offset;
    Offset.x = -0.5f * (Max.x + Min.x) * Scale.x;
    Offset.y = -0.5f * (Max.y + Min.y) * Scale.y;

    // Store scale factors for later use when blurring
    m_SplitScales.push_back(Scale);

    // Adjust projection matrix to "zoom in" on the target region
    D3DXMATRIX Zoom( Scale.x,     0.0f,  0.0f,   0.0f,
                        0.0f,  Scale.y,  0.0f,   0.0f,
                        0.0f,     0.0f,  1.0f,   0.0f,
                    Offset.x, Offset.y,  0.0f,   1.0f);

    // Compute new composite matrices and store
    D3DXMATRIX NewProj = LightProj * Zoom;
    D3DXMATRIX NewViewProj = LightView * NewProj;
    m_SplitProj.push_back(NewProj);
    m_SplitViewProj.push_back(NewViewProj);
  }

  // Update matrices in shader
  HRESULT hr;
  V(m_EffectSplitMatrices->SetMatrixArray(m_SplitViewProj[0], 0,
                                          static_cast<UINT>(m_SplitViewProj.size())));
}

//--------------------------------------------------------------------------------------
void PSVSM::DisplayShadowMap(const D3D10_VIEWPORT& Viewport)
{
  const unsigned int SplitSeparation = 25;

  // Bail if the effect doesn't have the right technique available
  if (!m_DisplayTechnique || !m_DisplayTechnique->IsValid() ||
      !m_EffectDisplayArrayIndex || !m_EffectDisplayArrayIndex->IsValid()) {
    return;
  }

  // Fit all of our slices into the given viewport (but maintain 1:1 aspect ratio)
  unsigned int Size = (Viewport.Width / m_NumSplits) - SplitSeparation;
  Size = static_cast<unsigned int>(std::min(Size, Viewport.Height));

  // Prepare for rendering
  D3D10_VIEWPORT SplitViewport(Viewport);
  SplitViewport.Width = Size;
  SplitViewport.Height = Size;
  m_PostProcess->SetupDrawState(m_d3dDevice);

  // Bind shadow texture array
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(m_ShadowMapArrayResource));

  for (unsigned int i = 0; i < m_NumSplits; ++i) {
    // Setup and draw
    V(m_EffectDisplayArrayIndex->SetInt(i));
    m_d3dDevice->RSSetViewports(1, &SplitViewport);
    m_PostProcess->FillScreen(m_d3dDevice, m_DisplayTechnique, false);

    // Move on
    SplitViewport.TopLeftX += Size + SplitSeparation;
  }

  // Unbind shadow texture array
  V(m_EffectShadowMap->SetResource(0));
  V(m_DisplayTechnique->GetPassByIndex(0)->Apply(0));
}

//--------------------------------------------------------------------------------------
void PSVSM::SetSplitLambda(float l)
{
  m_SplitLambda = l;
}

//--------------------------------------------------------------------------------------
void PSVSM::SetVisualizeSplits(bool v)
{
  HRESULT hr;
  V(m_EffectVisualizeSplits->SetBool(v));
}
