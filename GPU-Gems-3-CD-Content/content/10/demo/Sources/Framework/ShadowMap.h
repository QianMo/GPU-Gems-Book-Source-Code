#pragma once

// Base class for shadow map texture wrappers
//
// ShadowMap_D3D9, ShadowMap_D3D10 and ShadowMap_OGL will derive from this
class ShadowMap
{
public:
  ShadowMap() { m_strInfo[0]=0; m_iSize=0; }
  virtual ~ShadowMap() {}

  // destroys the shadow map
  virtual void Destroy(void) = 0;

  // returns the amount of memory used by this class, in megabytes
  virtual int GetMemoryInMB(void) = 0;

  // returns stats
  inline char *GetInfoString(void) { return m_strInfo; }

  inline int GetSize(void) { return m_iSize; }

protected:
  char m_strInfo[1024];
  int m_iSize;
};