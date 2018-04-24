#include "DXUT.h"
#include "LargeScene.hpp"

//--------------------------------------------------------------------------------------
LargeScene::LargeScene(ID3D10Device* d3dDevice)
{
  // Car
  m_CarMesh = LoadMesh(d3dDevice, L"Car.sdkmesh");
  m_CarTextures[0] = LoadTexture(d3dDevice, L"Car1.dds");
  m_CarTextures[1] = LoadTexture(d3dDevice, L"Car2.dds");

  // Commando
  m_CommandoMesh = LoadMesh(d3dDevice, L"Commando.sdkmesh");
  m_CommandoTextures[0] = LoadTexture(d3dDevice, L"Commando_Body.dds");
  m_CommandoTextures[1] = LoadTexture(d3dDevice, L"Commando_Face.dds");
  m_CommandoTextures[2] = LoadTexture(d3dDevice, L"Commando_Gear.dds");

  // Ground
  m_GroundMesh = LoadMesh(d3dDevice, L"GroundPlane.sdkmesh");
  m_GroundTexture = LoadTexture(d3dDevice, L"cement_salmonasphalt_df_.dds");
}

LargeScene::~LargeScene()
{
  SAFE_RELEASE(m_GroundTexture);
  SAFE_DELETE(m_GroundMesh);

  SAFE_RELEASE(m_CommandoTextures[0]);
  SAFE_RELEASE(m_CommandoTextures[1]);
  SAFE_RELEASE(m_CommandoTextures[2]);
  SAFE_DELETE(m_CommandoMesh);

  SAFE_RELEASE(m_CarTextures[0]);
  SAFE_RELEASE(m_CarTextures[1]);
  SAFE_DELETE(m_CarMesh);
}

void LargeScene::SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
{
  Scene::SetEffect(d3dDevice, Effect);

  HRESULT hr;

  // Set constants
  V(m_AmbientIntensity->SetFloat(0.2f));
  V(m_LightColor->SetFloatVector(D3DXVECTOR4(1.1f, 1.1f, 1.0f, 0.0f)));
  V(m_SpecularPower->SetFloat(30.0f));
  V(m_SpecularColor->SetFloatVector(D3DXVECTOR4(0, 0, 0, 0)));
}

void LargeScene::Render(ID3D10Device* d3dDevice,
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

  // Cars
  float Dist = 0.35f;

  for (int i = 0; i < 2; ++i) {
    V(m_DiffuseTexture->SetResource(m_CarTextures[i]));

    for (int y = -1; y <= 0; ++y) {
      for (int x = -3; x <= 3; ++x) {
        D3DXMATRIXA16 Translate;
        D3DXMatrixTranslation(&Translate, x * Dist, 0.0f, y * Dist);
        UpdateMatrices(Translate * W, View, Proj);

        DrawSubset(d3dDevice, m_CarMesh, i, RenderTechnique);
      }
    }
  }

  // Commandoes
  D3DXMatrixScaling(&Scale, 0.4f, 0.4f, 0.4f);

  for (int i = 0; i < 3; ++i) {
    V(m_DiffuseTexture->SetResource(m_CommandoTextures[i]));

    for (int x = -3; x <= 3; ++x) {
      D3DXMATRIXA16 Translate;
      float y = abs(x) % 2 == 0 ? -0.11f : -0.25f;
      D3DXMatrixTranslation(&Translate, x * Dist, 0.0f, y);
      UpdateMatrices(Scale * Translate * W, View, Proj);

      DrawSubset(d3dDevice, m_CommandoMesh, i, RenderTechnique);
    }
  }

  // Ground plane
  UpdateMatrices(W, View, Proj);
  {
    V(m_DiffuseTexture->SetResource(m_GroundTexture));
    DrawSubset(d3dDevice, m_GroundMesh, 0, RenderTechnique);
  }
}
