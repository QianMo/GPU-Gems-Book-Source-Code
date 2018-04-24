#include "DXUT.h"
#include "CommandoScene.hpp"

//--------------------------------------------------------------------------------------
CommandoScene::CommandoScene(ID3D10Device* d3dDevice)
{
  // Commando
  m_CommandoMesh = LoadMesh(d3dDevice, L"Commando.sdkmesh");
  m_CommandoTextures[0] = LoadTexture(d3dDevice, L"Commando_Body.dds");
  m_CommandoTextures[1] = LoadTexture(d3dDevice, L"Commando_Face.dds");
  m_CommandoTextures[2] = LoadTexture(d3dDevice, L"Commando_Gear.dds");

  // Ground
  m_GroundMesh = LoadMesh(d3dDevice, L"BumpyGround.sdkmesh");
  m_GroundTexture = LoadTexture(d3dDevice, L"dirt_grayrocky-mossy_df_.dds");
}

CommandoScene::~CommandoScene()
{
  SAFE_RELEASE(m_GroundTexture);
  SAFE_DELETE(m_GroundMesh);
  SAFE_RELEASE(m_CommandoTextures[0]);
  SAFE_RELEASE(m_CommandoTextures[1]);
  SAFE_RELEASE(m_CommandoTextures[2]);
  SAFE_DELETE(m_CommandoMesh);
}

void CommandoScene::SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
{
  Scene::SetEffect(d3dDevice, Effect);

  HRESULT hr;

  // Set constants
  V(m_AmbientIntensity->SetFloat(0.0f));
  V(m_LightColor->SetFloatVector(D3DXVECTOR4(1.1f, 0.9f, 0.6f, 0.0f)));
  V(m_SpecularPower->SetFloat(30.0f));
  V(m_SpecularColor->SetFloatVector(D3DXVECTOR4(0.0f, 0.0f, 0.0f, 0.0f)));
}

void CommandoScene::Render(ID3D10Device* d3dDevice,
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

  // Commando
  for (int i = 0; i < 3; ++i) {
    V(m_DiffuseTexture->SetResource(m_CommandoTextures[i]));
    DrawSubset(d3dDevice, m_CommandoMesh, i, RenderTechnique);
  }

  // Ground plane
  {
    V(m_DiffuseTexture->SetResource(m_GroundTexture));
    DrawSubset(d3dDevice, m_GroundMesh, 0, RenderTechnique);
  }
}

