#include "DXUT.h"
#include "StandardSAT.hpp"

//--------------------------------------------------------------------------------------
StandardSAT::StandardSAT(ID3D10Device* d3dDevice,
                         ID3D10Effect* Effect,
                         int Width, int Height,
                         PostProcess* PostProcess,
                         const DXGI_SAMPLE_DESC* SampleDesc,
                         bool IntFormat,
                         bool DistributePrecision)
  : VSM(d3dDevice, Effect, Width, Height, PostProcess,
        DistributePrecision
          ? (IntFormat ? DXGI_FORMAT_R32G32B32A32_UINT : DXGI_FORMAT_R32G32B32A32_FLOAT)
          : (IntFormat ? DXGI_FORMAT_R32G32_UINT : DXGI_FORMAT_R32G32_FLOAT),
        SampleDesc,
        // Only use FP formats for multisampling
        // NOTE: Arguably distribute precision is a bit ill-defined here, but it doesn't
        // matter since we never need/use it with integer formats anyways.
        IntFormat ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_UNKNOWN,
        false)
  , m_SAT(0), m_MSAAResolvedShadowMap(0)
  , m_IntFormat(IntFormat), m_DistributePrecision(DistributePrecision)
  , m_GenerateSATRDTechnique(0)
{
  // Create two shadow maps (for ping-ponging) and put them in the cache
  AddFullTextures(d3dDevice, 2);

  // Setup for SAT generating
  InitGenerateSAT(4);

  // Techniques
  std::ostringstream oss;
  oss << "Depth";
  if (m_MSAAShadowMap && m_IntFormat) {
    oss << "MSAA";
  } else if (m_DistributePrecision) {
    oss << "Distribute";
  }
  m_DepthTechnique = m_Effect->GetTechniqueByName(oss.str().c_str());
  assert(m_DepthTechnique && m_DepthTechnique->IsValid());

  // Extra resources needed for doing MSAA with an integer SAT
  if (m_MSAAShadowMap && m_IntFormat) {
    // Resolve into a compatible texture...
    m_MSAAResolvedShadowMap = new RenderableTexture2D(d3dDevice, m_Width, m_Height,
                                                      false, m_MSAAShadowFormat);

    // Then convert to an int32 texture...
    m_FPToINTTechnique = m_Effect->GetTechniqueByName("ConvertToIntShadowMap");
    assert(m_FPToINTTechnique && m_FPToINTTechnique->IsValid());
  }

  // Setup effect
  m_EffectSATTexture = m_Effect->GetVariableByName("texSAT")->AsShaderResource();
  assert(m_EffectSATTexture && m_EffectSATTexture->IsValid());
}

//--------------------------------------------------------------------------------------
StandardSAT::~StandardSAT()
{
  SAFE_DELETE(m_MSAAResolvedShadowMap);
}

//--------------------------------------------------------------------------------------
bool StandardSAT::EndShadowMap(ID3D10EffectTechnique* Technique)
{
  // If we have a special resolve texture, use it
  if (m_MSAAResolvedShadowMap) {
    m_ShadowMap = m_MSAAResolvedShadowMap;
  }

  VSM::EndShadowMap(Technique);
 
  // If necessary, now convert to an int32 texture
  if (m_MSAAResolvedShadowMap) {
    RenderableTexture2D* IntShadowMap = m_FullTextures.Get();
    m_PostProcess->Begin(m_d3dDevice, m_Width, m_Height, IntShadowMap, m_ShadowMap);
    D3D10_RECT Region = {0, 0, m_Width, m_Height};
    m_PostProcess->Apply(m_FPToINTTechnique, Region);
    m_ShadowMap = m_PostProcess->End();
  }

  // Generate summed area table
  m_SAT = GenerateSATRecursiveDouble(m_ShadowMap, false);
  m_ShadowMap = 0;

  // Done
  return false;
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  StandardSAT::BeginShading()
{
  // Bind shadow texture
  HRESULT hr;
  V(m_EffectSATTexture->SetResource(m_SAT->GetShaderResource()));

  return VSM::BeginShading();
}

//--------------------------------------------------------------------------------------
void StandardSAT::EndShading(ID3D10EffectTechnique* Technique)
{
  // Unbind shadow texture
  HRESULT hr;
  V(m_EffectSATTexture->SetResource(0));

  VSM::EndShading(Technique);
}

//--------------------------------------------------------------------------------------
void StandardSAT::EndFrame()
{
  // Release SAT(s) back into the cache
  m_FullTextures.Add(m_SAT);
  m_SAT = 0;

  VSM::EndFrame();
}

//--------------------------------------------------------------------------------------
void StandardSAT::InitGenerateSAT(int RDSamplesPerPass)
{
  m_GenerateSATRDSamples = RDSamplesPerPass;

  // Grab technique interfaces
  std::ostringstream oss;
  oss << "GenerateSATRD" << m_GenerateSATRDSamples;
  m_GenerateSATRDTechnique = m_Effect->GetTechniqueByName(oss.str().c_str());
  assert(m_GenerateSATRDTechnique && m_GenerateSATRDTechnique->IsValid());

  // Any effect interface variables
  m_EffectSATPassOffset = m_Effect->GetVariableByName("g_SATPassOffset")->AsVector();
  assert(m_EffectSATPassOffset && m_EffectSATPassOffset->IsValid());
}

//--------------------------------------------------------------------------------------
RenderableTexture2D* StandardSAT::GenerateSATRecursiveDouble(RenderableTexture2D *Src,
                                                             bool MaintainSrc)
{
  // If not initialized, initialze with defaults
  if (!m_GenerateSATRDTechnique) {
    InitGenerateSAT();
  }

  HRESULT hr;

  // Grab a temporary texture
  RenderableTexture2D *Dest = m_FullTextures.Get();
  // If we have to maintain the source, grab another temporary texture
  RenderableTexture2D *Temp = MaintainSrc ? m_FullTextures.Get() : Src;

  m_PostProcess->Begin(m_d3dDevice, m_Width, m_Height, Dest, Src, Temp);

  // Horizontal pass
  for (int i = 1; i < m_Width; i *= m_GenerateSATRDSamples) {
    int PassOffset[4] = {i, 0, 0, 0};
    V(m_EffectSATPassOffset->SetIntVector(PassOffset));

    // We need to still propogate samples that were "just finished" from
    // the last pass. Alternately could copy the texture first, but probably
    // not worth it.
    int Done =  i / m_GenerateSATRDSamples;
    // Also if we're maintaining the source, the second pass will still need
    // to write the whole texture, since it wasn't ping-ponged in the first one.
    Done = (Done <= 1 && MaintainSrc) ? 0 : Done;

    D3D10_RECT Region = {Done, 0, m_Width, m_Height};
    m_PostProcess->Apply(m_GenerateSATRDTechnique, Region);
  }

  // Vertical pass
  for (int i = 1; i < m_Height; i *= m_GenerateSATRDSamples) {
    int PassOffset[4] = {0, i, 0, 0};
    V(m_EffectSATPassOffset->SetIntVector(PassOffset));
    int Done = i / m_GenerateSATRDSamples;
    D3D10_RECT Region = {0, Done, m_Width, m_Height};
    m_PostProcess->Apply(m_GenerateSATRDTechnique, Region);
  }

  RenderableTexture2D *Result = m_PostProcess->End();

  // Return any non-result textures to the available list
  m_FullTextures.Add(Dest);
  m_FullTextures.Add(Temp);
  m_FullTextures.Remove(Result);
  
  return Result;
}
