#pragma once

#include "Scene.hpp"

//--------------------------------------------------------------------------------------
class CommandoScene : public Scene
{
public:
  CommandoScene(ID3D10Device* d3dDevice);
  virtual ~CommandoScene();

  virtual D3DXVECTOR3 GetCenter() const { return D3DXVECTOR3(0, 0, 0); }

  virtual float GetRadius() const { return 25.0f; }

  virtual void GetCameraDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const
  {
    Pos = D3DXVECTOR3(0.3f, 0.3f, -0.3f);
    Target = D3DXVECTOR3(0, 0.1f, 0);
  }

  virtual void GetLightDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const
  {
    Pos = D3DXVECTOR3(1.0f, 0.5f, 0.2f);
    Target = D3DXVECTOR3(0, 0, 0);
  }

  virtual void SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect);

  virtual void Render(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj);

private:
  CDXUTSDKMesh*                m_CommandoMesh;
  ID3D10ShaderResourceView*    m_CommandoTextures[3];

  CDXUTSDKMesh*                m_GroundMesh;
  ID3D10ShaderResourceView*    m_GroundTexture;
};
