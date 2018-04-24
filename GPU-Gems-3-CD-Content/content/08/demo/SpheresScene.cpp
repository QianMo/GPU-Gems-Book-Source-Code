#include "DXUT.h"
#include "SpheresScene.hpp"

//--------------------------------------------------------------------------------------
SpheresScene::SpheresScene(ID3D10Device* d3dDevice)
{
  // Spheres
  m_SpheresMesh = LoadMesh(d3dDevice, L"Spheres.sdkmesh");
  m_SpheresTexture = LoadTexture(d3dDevice, L"wood_hardwood_df_.dds");

  // Ground
  m_GroundMesh = LoadMesh(d3dDevice, L"GroundPlane.sdkmesh");
  m_GroundTexture = LoadTexture(d3dDevice, L"plastic_whiteleathery_df_.dds");
}

SpheresScene::~SpheresScene()
{
  SAFE_RELEASE(m_GroundTexture);
  SAFE_DELETE(m_GroundMesh);
  SAFE_RELEASE(m_SpheresTexture);
  SAFE_DELETE(m_SpheresMesh);
}

void SpheresScene::SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
{
  Scene::SetEffect(d3dDevice, Effect);

  HRESULT hr;

  // Set constants
  V(m_AmbientIntensity->SetFloat(0.0f));
  V(m_LightColor->SetFloatVector(D3DXVECTOR4(1.2f, 1.2f, 1.2f, 0.0f)));
  V(m_SpecularPower->SetFloat(20.0f));
  V(m_SpecularColor->SetFloatVector(D3DXVECTOR4(0.0f, 0.0f, 0.0f, 0.0f)));
}

void SpheresScene::Render(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj)
{
  HRESULT hr;

  UpdateMatrices(World, View, Proj);

  // Car
  {
    V(m_DiffuseTexture->SetResource(m_SpheresTexture));
    DrawSubset(d3dDevice, m_SpheresMesh, 0, RenderTechnique);
  }

  // Ground plane
  {
    V(m_DiffuseTexture->SetResource(m_GroundTexture));
    DrawSubset(d3dDevice, m_GroundMesh, 0, RenderTechnique);
  }
}


