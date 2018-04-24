#include "DXUT.h"
#include "CarScene.hpp"

//--------------------------------------------------------------------------------------
CarScene::CarScene(ID3D10Device* d3dDevice)
{
  // Car
  m_CarMesh = LoadMesh(d3dDevice, L"Car.sdkmesh");
  m_CarTextures[0] = LoadTexture(d3dDevice, L"Car1.dds");
  m_CarTextures[1] = LoadTexture(d3dDevice, L"Car2.dds");

  // Ground
  m_GroundMesh = LoadMesh(d3dDevice, L"GroundPlane.sdkmesh");
  m_GroundTexture = LoadTexture(d3dDevice, L"cement_asphaltscales-01_df_.dds");
}

CarScene::~CarScene()
{
  SAFE_RELEASE(m_GroundTexture);
  SAFE_DELETE(m_GroundMesh);
  SAFE_RELEASE(m_CarTextures[0]);
  SAFE_RELEASE(m_CarTextures[1]);
  SAFE_DELETE(m_CarMesh);
}

void CarScene::SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
{
  Scene::SetEffect(d3dDevice, Effect);

  HRESULT hr;

  // Set constants
  V(m_AmbientIntensity->SetFloat(0.05f));
  V(m_LightColor->SetFloatVector(D3DXVECTOR4(1.3f, 1.3f, 1.3f, 0.0f)));
  V(m_SpecularPower->SetFloat(30.0f));
  V(m_SpecularColor->SetFloatVector(D3DXVECTOR4(0.3f, 0.3f, 0.3f, 0.0f)));
}

void CarScene::Render(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj)
{
  HRESULT hr;

  // Global transform
  D3DXMATRIXA16 Scale;
  D3DXMatrixScaling(&Scale, 0.6f, 0.6f, 0.6f);
  D3DXMATRIXA16 W = Scale * World;

  UpdateMatrices(W, View, Proj);

  // Car
  for (int i = 0; i < 2; ++i) {
    V(m_DiffuseTexture->SetResource(m_CarTextures[i]));
    DrawSubset(d3dDevice, m_CarMesh, i, RenderTechnique);
  }

  // Ground plane
  {
    V(m_DiffuseTexture->SetResource(m_GroundTexture));
    DrawSubset(d3dDevice, m_GroundMesh, 0, RenderTechnique);
  }
}
