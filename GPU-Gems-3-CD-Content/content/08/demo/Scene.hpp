#pragma once

#include <string>
#include "SDKmesh.h"

//--------------------------------------------------------------------------------------
// NOTE: Renderables will be destroyed and recreated on device loss, so they can
// create/release resources in the constructors and destructors.
class Renderable
{
public:
  Renderable();
  virtual ~Renderable();

  // Set the current effect
  // NOTE: Do this only when it changes (i.e. infrequently)
  virtual void SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect);

  // Render the scene using the current effect, updating matrices as required
  virtual void Render(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj) = 0;

protected:
  // Handy utility to set world-matrix-related uniforms
  void UpdateMatrices(const D3DXMATRIXA16& World,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj) const;

  // Utility for drawing a mesh subset
  void DrawSubset(ID3D10Device* d3dDevice,
                  CDXUTSDKMesh *Mesh, unsigned int Subset,
                  ID3D10EffectTechnique* RenderTechnique) const;

  static CDXUTSDKMesh * LoadMesh(ID3D10Device* d3dDevice, const std::wstring& File);

  static ID3D10ShaderResourceView * LoadTexture(ID3D10Device* d3dDevice, const std::wstring& File);
  
  // Effect and shader constants
  ID3D10Effect*                 m_Effect;
  ID3D10EffectMatrixVariable*   m_World;
  ID3D10EffectMatrixVariable*   m_WorldView;
  ID3D10EffectMatrixVariable*   m_WorldViewProj;

  // Input layout
  ID3D10InputLayout*            m_InputLayout;
};

//--------------------------------------------------------------------------------------
class Scene : public Renderable
{
public:
  /// Bounding sphere for the scene
  virtual D3DXVECTOR3 GetCenter() const = 0;
  virtual float GetRadius() const = 0;

  /// Get the default camera position and target
  virtual void GetCameraDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const = 0;

  /// Get the default light position and target
  virtual void GetLightDefaults(D3DXVECTOR3& Pos, D3DXVECTOR3& Target) const = 0;

  /// Get light constants for this scene
  virtual void GetLightConstants(float &FOV, D3DXVECTOR2& DistFalloff) const
  {
    // Good defaults
    FOV = D3DX_PI * 0.2;
    DistFalloff = D3DXVECTOR2(2.2f, 2.5f);
  }

  virtual void SetEffect(ID3D10Device* d3dDevice, ID3D10Effect* Effect);

protected:
  // Handy for our scenes
  ID3D10EffectShaderResourceVariable* m_DiffuseTexture;
  ID3D10EffectScalarVariable*         m_AmbientIntensity;
  ID3D10EffectVectorVariable*         m_LightColor;
  ID3D10EffectScalarVariable*         m_SpecularPower;
  ID3D10EffectVectorVariable*         m_SpecularColor;
};
