#pragma once

#include "Scene.hpp"

//--------------------------------------------------------------------------------------
class SpheresScene : public Scene
{
public:
  SpheresScene(ID3D10Device* d3dDevice);
  virtual ~SpheresScene();

  virtual D3DXVECTOR3 GetCenter() const { return D3DXVECTOR3(-0.03f, 0, 0.03f); }

  virtual float GetRadius() const { return 100.0f; }

  virtual void GetCameraDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const
  {
    Pos = D3DXVECTOR3(-0.18f, 0.1f, -0.18f);
    Target = D3DXVECTOR3(0.05f, 0, -0.05f);
  }

  virtual void GetLightDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const
  {
    Pos = D3DXVECTOR3(-0.2f, 0.4f, 0.3f);
    Target = D3DXVECTOR3(0, 0, 0);
  }

  virtual void SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect);

  virtual void Render(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj);

private:
  CDXUTSDKMesh*              m_SpheresMesh;
  ID3D10ShaderResourceView*  m_SpheresTexture;

  CDXUTSDKMesh*              m_GroundMesh;
  ID3D10ShaderResourceView*  m_GroundTexture;
};
