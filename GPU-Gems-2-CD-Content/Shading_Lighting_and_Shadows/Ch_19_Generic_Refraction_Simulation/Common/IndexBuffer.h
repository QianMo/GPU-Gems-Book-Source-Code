///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : IndexBuffer.h
//  Desc : Simple index buffer class
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CIndexBuffer
{
public:
  CIndexBuffer(): m_iCount(0), m_pIB(0)
  {
  }

  ~CIndexBuffer()
  {
    Release();
  }

  // Create an Index buffer
  int Create(int iCount, const ushort *pIndexList);
  // Release resources
  void Release();
  // Activate index buffer
  void Enable();

  // Manipulators
  IDirect3DIndexBuffer9 *Get()
  {
    return m_pIB; 
  }

  // Accessors
  const IDirect3DIndexBuffer9 *Get() const
  {
    return m_pIB; 
  }

  int GetCount() const
  {
    return m_iCount;
  }

private:
  int m_iCount;
  IDirect3DIndexBuffer9 *m_pIB;  
};