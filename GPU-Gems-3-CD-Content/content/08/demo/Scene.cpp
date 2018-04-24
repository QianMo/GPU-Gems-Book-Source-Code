#include "DXUT.h"
#include "Scene.hpp"

namespace {
  // Standard vertex format
  static const unsigned int DefaultMeshInputLayoutSize = 3;
  static const D3D10_INPUT_ELEMENT_DESC DefaultMeshInputLayout[DefaultMeshInputLayoutSize] =
  {
      {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0},
      {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D10_INPUT_PER_VERTEX_DATA, 0}
  };

  // "Do nothing" texture loader utility for use with SDKMESH
  void CALLBACK LoadTextureNull(ID3D10Device* pDev, char* szFileName,
                                ID3D10ShaderResourceView** ppRV, void* pContext)
  {
    // Seems to be a special flag to indicate an "error" loading the resource
    // Really bad cast though Microsoft...
    *ppRV = reinterpret_cast<ID3D10ShaderResourceView*>(ERROR_RESOURCE_VALUE);
  }
}


//--------------------------------------------------------------------------------------
Renderable::Renderable()
  : m_Effect(0), m_InputLayout(0)
{
}

//--------------------------------------------------------------------------------------
Renderable::~Renderable()
{
  SAFE_RELEASE(m_InputLayout);
}

//--------------------------------------------------------------------------------------
void Renderable::SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
{
  m_Effect = Effect;

  HRESULT hr;

  // Grab uniforms
  m_World = Effect->GetVariableByName("g_WorldMatrix")->AsMatrix();
  assert(m_World && m_World->IsValid());
  m_WorldView = Effect->GetVariableByName("g_WorldViewMatrix")->AsMatrix();
  assert(m_WorldView && m_WorldView->IsValid());
  m_WorldViewProj = Effect->GetVariableByName("g_WorldViewProjMatrix")->AsMatrix();
  assert(m_WorldViewProj && m_WorldViewProj->IsValid());

  // Create input layout (assume all techniques are created equal!)
  SAFE_RELEASE(m_InputLayout);

  // Try to use the "Shading" technique as it contains the most inputs
  ID3D10EffectTechnique* Technique = Effect->GetTechniqueByName("Shading");
  // If not, the first technique found
  if (!Technique || !Technique->IsValid()) {
    ID3D10EffectTechnique* Technique = Effect->GetTechniqueByIndex(0);
  }
  
  // Use the first pass
  ID3D10EffectPass *Pass = Technique->GetPassByIndex(0);
  
  D3D10_PASS_DESC PassDesc;
  V(Pass->GetDesc(&PassDesc));
  V(d3dDevice->CreateInputLayout(DefaultMeshInputLayout, DefaultMeshInputLayoutSize,
      PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_InputLayout));
}

//--------------------------------------------------------------------------------------
void Renderable::UpdateMatrices(const D3DXMATRIXA16& World,
                                const D3DXMATRIXA16& View,
                                const D3DXMATRIXA16& Proj) const
{
  D3DXMATRIXA16 W = World;
  D3DXMATRIXA16 WV = W * View;
  D3DXMATRIXA16 WVP = WV * Proj;

  HRESULT hr;
  V(m_World->SetMatrix(W));
  V(m_WorldView->SetMatrix(WV));
  V(m_WorldViewProj->SetMatrix(WVP));
}

//--------------------------------------------------------------------------------------
void Renderable::DrawSubset(ID3D10Device* d3dDevice,
                            CDXUTSDKMesh *Mesh, unsigned int Subset,
                            ID3D10EffectTechnique* RenderTechnique) const
{
  // Collect mesh information
  UINT Stride;
  UINT Offset;
  ID3D10Buffer* VertexBuffer;
  VertexBuffer = Mesh->GetVB10(0, 0);
  Stride = Mesh->GetVertexStride(0, 0);
  Offset = 0;

  const SDKMESH_SUBSET *MeshSubset = Mesh->GetSubset(0, Subset);
  D3D10_PRIMITIVE_TOPOLOGY PrimType = CDXUTSDKMesh::GetPrimitiveType10(
    static_cast<SDKMESH_PRIMITIVE_TYPE>(MeshSubset->PrimitiveType));

  // Setup input assembler
  d3dDevice->IASetInputLayout(m_InputLayout);
  d3dDevice->IASetPrimitiveTopology(PrimType);
  d3dDevice->IASetVertexBuffers(0, 1, &VertexBuffer, &Stride, &Offset);
  d3dDevice->IASetIndexBuffer(Mesh->GetIB10(0), Mesh->GetIBFormat10(0), 0);
  
  // Loop over passes
  D3D10_TECHNIQUE_DESC TechDesc;
  RenderTechnique->GetDesc(&TechDesc);
  for (unsigned int p = 0; p < TechDesc.Passes; ++p) {
    RenderTechnique->GetPassByIndex(p)->Apply(0);
    // Render the scene
    d3dDevice->DrawIndexed(static_cast<UINT>(MeshSubset->IndexCount),
                           static_cast<UINT>(MeshSubset->IndexStart),
                           static_cast<UINT>(MeshSubset->VertexStart));
  }
}

//--------------------------------------------------------------------------------------
CDXUTSDKMesh * Renderable::LoadMesh(ID3D10Device* d3dDevice, const std::wstring& File)
{
  HRESULT hr;
  std::wstring FileName = L"Media\\Meshes\\";
  FileName += File;

  // Use a custom loader to avoid trying to load any referenced textures
  SDKMESH_CALLBACKS10 LoaderCallbacks;
  ZeroMemory(&LoaderCallbacks, sizeof(SDKMESH_CALLBACKS10));
  LoaderCallbacks.pCreateTextureFromFile = LoadTextureNull;

  CDXUTSDKMesh *Mesh = new CDXUTSDKMesh();
  V(Mesh->Create(d3dDevice, FileName.c_str(), true, false, &LoaderCallbacks));
  return Mesh;
}

//--------------------------------------------------------------------------------------
ID3D10ShaderResourceView * Renderable::LoadTexture(ID3D10Device* d3dDevice, const std::wstring& File)
{
  HRESULT hr;
  std::wstring FileName = L"Media\\Textures\\";
  FileName += File;

  // Load texture
  ID3D10ShaderResourceView *Texture;
  V(D3DX10CreateShaderResourceViewFromFile(d3dDevice, FileName.c_str(), NULL, NULL, &Texture, NULL));
  
  return Texture;
}

//--------------------------------------------------------------------------------------





//--------------------------------------------------------------------------------------
void Scene::SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
{
  Renderable::SetEffect(d3dDevice, Effect);

  // Grab some additional common uniforms
  m_DiffuseTexture = m_Effect->GetVariableByName("texDiffuse")->AsShaderResource();
  assert(m_DiffuseTexture && m_DiffuseTexture->IsValid());
  m_AmbientIntensity = m_Effect->GetVariableByName("g_AmbientIntensity")->AsScalar();
  assert(m_AmbientIntensity && m_AmbientIntensity->IsValid());
  m_LightColor = m_Effect->GetVariableByName("g_LightColor")->AsVector();
  assert(m_LightColor && m_LightColor->IsValid());
  m_SpecularPower = m_Effect->GetVariableByName("g_SpecularPower")->AsScalar();
  assert(m_SpecularPower && m_SpecularPower->IsValid());
  m_SpecularColor = m_Effect->GetVariableByName("g_SpecularColor")->AsVector();
  assert(m_SpecularColor && m_SpecularColor->IsValid());
}
