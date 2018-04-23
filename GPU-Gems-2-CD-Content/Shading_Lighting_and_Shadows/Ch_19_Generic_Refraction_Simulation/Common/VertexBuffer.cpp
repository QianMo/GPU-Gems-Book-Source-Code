///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : VertexBuffer.cpp
//  Desc : Simple vertex buffer class
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "VertexBuffer.h"
#include "D3dApp.h"

int CVertexBuffer::Create(int iCount, const float *pVertexList)
{
  if(!pVertexList || !iCount)
  {
    return APP_ERR_INVALIDPARAM;
  }

  //Make sure all data released, just in case client misuses
  Release();  

  // Set vertex data
  m_iCount=iCount;
  m_iSize=(3+3+3+2)*sizeof(float);
  m_iFormat= D3DFVF_XYZ|D3DFVF_TEXCOORDSIZE(0,2)|D3DFVF_TEXCOORDSIZE(1,3)|D3DFVF_TEXCOORDSIZE(2,3)|D3DFVF_TEX3; 

  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();
  
  if(FAILED(plD3DDevice->CreateVertexBuffer(m_iSize*m_iCount, D3DUSAGE_WRITEONLY, m_iFormat, D3DPOOL_MANAGED, &m_pVB, 0)))
  {
    return APP_ERR_INITFAIL;
  }

  // Copy data into vertex buffer
  float *pLockedVB=0;
  if(FAILED(m_pVB->Lock(0, 0, (VOID **)&pLockedVB, 0)))
  {
    return APP_ERR_INITFAIL;
  }
  memcpy(pLockedVB, pVertexList, m_iCount*m_iSize);
  m_pVB->Unlock();

  return APP_OK;;
}

void CVertexBuffer::Release()
{
  m_iFormat=0;
  m_iSize=0;
  m_iCount=0;
  SAFE_RELEASE(m_pVB)  
}

void CVertexBuffer::Enable()
{
  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();  
  plD3DDevice->SetStreamSource(0, m_pVB, 0, m_iSize);
}

