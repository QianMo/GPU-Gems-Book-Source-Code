///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : IndexBuffer.cpp
//  Desc : Simple index buffer class
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "IndexBuffer.h"
#include "D3dApp.h"

int CIndexBuffer::Create(int iCount, const ushort *pIndexList)
{
  if(!pIndexList || !iCount)
  {
    return APP_ERR_INVALIDPARAM;
  }

  //Make sure all data released, just in case client misuses
  Release();  

  // Set vertex data
  m_iCount=iCount;

  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();

  // Create index buffer  
  if(FAILED(plD3DDevice->CreateIndexBuffer(sizeof(ushort)*m_iCount, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &m_pIB, 0))) 
  {
    return APP_ERR_INITFAIL;
  }

  // Copy data into index buffer
  ushort  *pLockedIBuffer=0;
  if(FAILED(m_pIB->Lock(0, 0, (VOID **)&pLockedIBuffer, 0))) 
  {
    return APP_ERR_INITFAIL;
  }  
  memcpy(pLockedIBuffer, pIndexList, sizeof(ushort)*m_iCount);

  m_pIB->Unlock();

  return APP_OK;
}

void CIndexBuffer::Release()
{
  m_iCount=0;
  SAFE_RELEASE(m_pIB)
}

void CIndexBuffer::Enable()
{
  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();
  plD3DDevice->SetIndices(m_pIB);
}
