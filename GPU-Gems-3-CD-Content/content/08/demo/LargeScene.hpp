#pragma once

#include "Scene.hpp"

//--------------------------------------------------------------------------------------
class LargeScene : public Scene
{
public:
  LargeScene(ID3D10Device* d3dDevice);
  virtual ~LargeScene();

  virtual D3DXVECTOR3 GetCenter() const { return D3DXVECTOR3(0, 0, 0); }

  virtual float GetRadius() const { return 100.0f; }

  virtual void GetCameraDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const
  {
    Pos = D3DXVECTOR3(0.75f, 0.05f, -0.12f);
    Target = D3DXVECTOR3(0, 0, -0.09f);
  }

  virtual void GetLightDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const
  {
    Pos = D3DXVECTOR3(0.75f, 1.2f, 1.8f);
    Target = D3DXVECTOR3(0, 0, 0);
  }

  virtual void GetLightConstants(float &FOV, D3DXVECTOR2& DistFalloff) const
  {
    FOV = D3DX_PI * 0.35f;
    DistFalloff = D3DXVECTOR2(3.0f, 4.0f);
  }

  virtual void SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect);

  virtual void Render(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj);

private:
  CDXUTSDKMesh*             m_CarMesh;
  ID3D10ShaderResourceView* m_CarTextures[2];

  CDXUTSDKMesh*             m_CommandoMesh;
  ID3D10ShaderResourceView* m_CommandoTextures[3];

  CDXUTSDKMesh*             m_GroundMesh;
  ID3D10ShaderResourceView* m_GroundTexture;
};
