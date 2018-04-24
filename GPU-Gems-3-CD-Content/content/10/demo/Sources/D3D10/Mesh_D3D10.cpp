#include "../Framework/Common.h"
#include "../Framework/LWOFile.h"
#include "Application_D3D10.h"
#include "../Framework/Mesh.h"
#include "Mesh_D3D10.h"

Mesh *CreateNewMesh(void)
{
  return new Mesh_D3D10();
}

Mesh_D3D10::Mesh_D3D10()
{
  m_pVertexBuffer = NULL;
  m_pIndexBuffer = NULL;
  m_pVertexLayout = NULL;
}

Mesh_D3D10::~Mesh_D3D10()
{
  if(m_pVertexBuffer != NULL)
  {
    m_pVertexBuffer->Release();
    m_pVertexBuffer = NULL;
  }

  if(m_pIndexBuffer != NULL)
  {
    m_pIndexBuffer->Release();
    m_pIndexBuffer = NULL;
  }

  if(m_pVertexLayout != NULL)
  {
    m_pVertexLayout->Release();
    m_pVertexLayout = NULL;
  }
}

extern ID3D10EffectTechnique *g_pTechniqueShadows;
extern bool CreateShaders(void);

bool Mesh_D3D10::CreateBuffers(void)
{
  // Define the input layout
  D3D10_INPUT_ELEMENT_DESC layout[] =
  {
      { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },  
      { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 }, 
      { "COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D10_INPUT_PER_VERTEX_DATA, 0 }, 
  };
  unsigned int iElements = sizeof(layout)/sizeof(layout[0]);

  // create shaders if not created yet
  if(g_pTechniqueShadows == NULL)
  {
    if(!CreateShaders()) return false;
  }

  // Create the input layout
  HRESULT hr;
  D3D10_PASS_DESC PassDesc;
  hr = g_pTechniqueShadows->GetPassByIndex(0)->GetDesc(&PassDesc);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Getting technique pass description failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  hr = GetApp()->GetDevice()->CreateInputLayout(layout, iElements, PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_pVertexLayout);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating input layout failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create vertex buffer
  D3D10_BUFFER_DESC bd;
  bd.Usage = D3D10_USAGE_DEFAULT;
  bd.ByteWidth = m_iVertexSize * m_iNumVertices;
  bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
  bd.CPUAccessFlags = 0;
  bd.MiscFlags = 0;
  D3D10_SUBRESOURCE_DATA InitData;
  InitData.pSysMem = m_pVertices;
  hr = GetApp()->GetDevice()->CreateBuffer( &bd, &InitData, &m_pVertexBuffer );
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating vertex buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create index buffer
  bd.Usage = D3D10_USAGE_DEFAULT;
  bd.ByteWidth = sizeof(unsigned short) * m_iNumTris * 3;
  bd.BindFlags = D3D10_BIND_INDEX_BUFFER;
  bd.CPUAccessFlags = 0;
  bd.MiscFlags = 0;
  InitData.pSysMem = m_pIndices;
  hr = GetApp()->GetDevice()->CreateBuffer( &bd, &InitData, &m_pIndexBuffer );
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating index buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  return true;
}

extern int g_iTrisPerFrame;

void Mesh_D3D10::Draw(void)
{
  // Set vertex buffer
  unsigned int iOffset = 0;
  GetApp()->GetDevice()->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &m_iVertexSize, &iOffset);

  // Set index buffer
  GetApp()->GetDevice()->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);

  // Set primitive topology
  GetApp()->GetDevice()->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  // Set the input layout
  GetApp()->GetDevice()->IASetInputLayout(m_pVertexLayout);

  // Draw
  GetApp()->GetDevice()->DrawIndexed(m_iNumTris * 3, 0, 0);

  g_iTrisPerFrame += m_iNumTris;
}

void Mesh_D3D10::DrawInstanced(int iNumInstances)
{
  // Set vertex buffer
  unsigned int iOffset = 0;
  GetApp()->GetDevice()->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &m_iVertexSize, &iOffset);

  // Set index buffer
  GetApp()->GetDevice()->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);

  // Set primitive topology
  GetApp()->GetDevice()->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  // Set the input layout
  GetApp()->GetDevice()->IASetInputLayout(m_pVertexLayout);

  // Draw
  GetApp()->GetDevice()->DrawIndexedInstanced(m_iNumTris * 3, iNumInstances, 0, 0, 0);

  g_iTrisPerFrame += m_iNumTris;
}