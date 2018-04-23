///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Material.h
//  Desc : Simple material class
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "Texture.h"

class CMaterial
{
public:
  CMaterial()
  {
    Reset();
  }

  void SetDiffuse(const CColor &pDiffuse)
  {
    m_pDiffuse=pDiffuse;
  }

  void SetSpecular(const CColor &pSpecular)
  {
    m_pSpecular=pSpecular;
  }

  void SetAmbient(const CColor &pAmbient)
  {
    m_pAmbient=pAmbient;
  }

  void SetEmissive(const CColor &pEmissive)
  {
    m_pEmissive=pEmissive;
  }

  void SetSpecularLevel(float fSpecLevel)
  {
    m_fSpecLevel=fSpecLevel;
  }

  void SetReflectivity(float fReflectivity)
  {
    m_fReflectivity=fReflectivity;
  }

  void SetOpacity(float fOpacity)
  {
    m_fOpacity=fOpacity;
  }

  void SetLightSpeed(float fLightSpeed)
  {
    m_fLightSpeed=fLightSpeed;
  }

  void SetDoubleSided(bool bDoubleSided)
  {
    m_bDoubleSided=bDoubleSided;
  }

  const CColor &GetDiffuse() const
  {
    return m_pDiffuse;
  };

  const CColor &GetSpecular() const
  {
    return m_pSpecular;
  };

  const CColor &GetAmbient() const
  {
    return m_pAmbient;
  };

  const CColor &GetEmissive() const
  {
    return m_pEmissive;
  };

  const float GetSpecularLevel() const
  {
    return m_fSpecLevel;
  };

  float GetReflectivity() const
  {
    return m_fReflectivity;
  }

  float GetOpacity() const
  {
    return m_fOpacity;
  }

  float GetLightSpeed() const
  {
    return m_fLightSpeed;
  }

  bool GetDoubleSided() const
  {
    return m_bDoubleSided;
  }

  // reset material data
  void Reset()
  {
    m_pDiffuse.Set(1,1,1,1);
    m_pSpecular.Set(1,1,1,1);
    m_pAmbient.Set(1,1,1,1);
    m_pEmissive.Set(1,1,1,1);
    m_fSpecLevel=1.0f;
    m_fReflectivity=0.0f;
    m_fOpacity=1.0f;
    m_fLightSpeed=1.0f;
    m_bDoubleSided=0;
  };

  int SetDecalTex(const char *pDecal);
  int SetEnvMapTex(const char *pEnvMap);
  
  const CTexture *GetDecalTex() const
  {
    return &m_pTextureList[0];
  }

  const CTexture *GetEnvMapTex() const
  {
    return &m_pTextureList[1];
  }

private:
  // material properties
  CColor  m_pDiffuse, m_pSpecular, m_pAmbient, m_pEmissive;
  float   m_fSpecLevel, m_fReflectivity, m_fOpacity, m_fLightSpeed;
  bool    m_bDoubleSided;

  // decal and environment map
  CTexture m_pTextureList[2];
};
