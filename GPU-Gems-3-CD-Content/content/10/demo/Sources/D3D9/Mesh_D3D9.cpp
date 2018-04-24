#include "../Framework/Common.h"
#include "Application_D3D9.h"
#include "../Framework/Mesh.h"
#include "Mesh_D3D9.h"

Mesh *CreateNewMesh(void)
{
  return new Mesh_D3D9();
}

Mesh_D3D9::Mesh_D3D9()
{
  m_pVertexBuffer = NULL;
  m_pVertexDeclaration = NULL;
  m_pIndexBuffer = NULL;
}

Mesh_D3D9::~Mesh_D3D9()
{
  if(m_pVertexBuffer != NULL)
  {
    m_pVertexBuffer->Release();
    m_pVertexBuffer = NULL;
  }

  if(m_pVertexDeclaration != NULL)
  {
    m_pVertexDeclaration->Release();
    m_pVertexDeclaration = NULL;
  }

  if(m_pIndexBuffer != NULL)
  {
    m_pIndexBuffer->Release();
    m_pIndexBuffer = NULL;
  }
}

bool Mesh_D3D9::CreateBuffers(void)
{
  HRESULT hr;

  // create vertex buffer
  //
  hr = GetApp()->GetDevice()->CreateVertexBuffer(m_iNumVertices * m_iVertexSize, D3DUSAGE_WRITEONLY, 0,
                                                 D3DPOOL_MANAGED, &m_pVertexBuffer, NULL);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating vertex buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  void *pData=NULL;

  // lock buffer
  //
  hr = m_pVertexBuffer->Lock(0, m_iNumVertices * m_iVertexSize, &pData, 0);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Locking vertex buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // write data
  //
  CopyMemory(pData, m_pVertices, m_iNumVertices * m_iVertexSize);

  // unlock buffer
  //
  hr = m_pVertexBuffer->Unlock();
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Unlocking vertex buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

///////////

  // create index buffer
  //
  hr = GetApp()->GetDevice()->CreateIndexBuffer(m_iNumTris * 3 * sizeof(unsigned short), D3DUSAGE_WRITEONLY,
                                                D3DFMT_INDEX16, D3DPOOL_MANAGED, &m_pIndexBuffer, NULL);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating index buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }


  // lock buffer
  //
  hr = m_pIndexBuffer->Lock(0, m_iNumTris * 3 * sizeof(unsigned short), &pData,0);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Locking index buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // write data
  //
  CopyMemory(pData, m_pIndices, m_iNumTris * 3 * sizeof(unsigned short));

  // unlock buffer
  //
  hr = m_pIndexBuffer->Unlock();
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Unlocking index buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

/////////

  D3DVERTEXELEMENT9 decl[] =
  {
    { 0, 0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
    { 0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
    { 0, 24, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
    D3DDECL_END()
  };

  hr = GetApp()->GetDevice()->CreateVertexDeclaration(decl, &m_pVertexDeclaration);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating vertex declaration failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  return true;
}

extern int g_iTrisPerFrame;

void Mesh_D3D9::Draw(void)
{
  GetApp()->GetDevice()->SetVertexDeclaration(m_pVertexDeclaration);
  GetApp()->GetDevice()->SetStreamSource(0, m_pVertexBuffer, 0, m_iVertexSize);
  GetApp()->GetDevice()->SetIndices(m_pIndexBuffer);
  GetApp()->GetDevice()->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, m_iNumVertices, 0, m_iNumTris);
  g_iTrisPerFrame += m_iNumTris;
}
