///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Material.cpp
//  Desc : Simple material class
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Material.h"

int CMaterial::SetDecalTex(const char *pDecal)
{
  if(!pDecal)
  {
    return APP_ERR_INVALIDPARAM;
  }

  // Check if texture is same
  if(strcmp(pDecal, m_pTextureList[0].GetTextureFileName())==0)
  {
    return APP_OK;
  }
  m_pTextureList[0].Create(pDecal, 0);
  return APP_OK;
}

int CMaterial::SetEnvMapTex(const char *pEnvMap)
{
  if(!pEnvMap)
  {
    return APP_ERR_INVALIDPARAM;
  }

  // Check if texture is same
  if(strcmp(pEnvMap, m_pTextureList[1].GetTextureFileName())==0)
  {
    return APP_OK;
  }

  m_pTextureList[1].Create(pEnvMap, 1);
  return APP_OK;
}