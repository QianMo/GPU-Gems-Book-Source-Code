///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Mesh.cpp
//  Desc : Generic mesh class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Mesh.h"
#include "Material.h"
#include "ModelLoader.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"

int CSubMesh::Create(int iMaterialID, int iCount, const ushort *pIndexList)
{
  //Make sure all data released, just in case client misuses
  Release();

  m_iMaterialID=iMaterialID;
  m_iFaceCount=iCount;

  m_pIB=new CIndexBuffer;
  if(FAILED(m_pIB->Create(m_iFaceCount*3, pIndexList)))
  {
    return APP_ERR_INITFAIL;
  }

  m_pIndicesList=new ushort[m_iFaceCount*3];
  memcpy(m_pIndicesList, pIndexList, m_iFaceCount*3*sizeof(ushort));

  return APP_OK;
}

void CSubMesh::Release()
{
  m_iMaterialID=0;
  m_iFaceCount=0;
  SAFE_DELETE(m_pIB)
  SAFE_DELETE_ARRAY(m_pIndicesList)
}

int CBaseMesh::Create(const char *pFile)
{
  // Make sure all data released, just in case client misuses
  Release();

  if(!pFile)
  {
    return APP_ERR_INVALIDPARAM;
  }

  // Get file extention
  char *pExtension = strrchr(pFile, '.' );
  if(pExtension) 
  {
    pExtension++;
  }
  else
  {
    return APP_ERR_INVALIDPARAM;
  }

  // Convert to lower case
  _strlwr(pExtension);

  // Check is file type is supported
  if(stricmp(pExtension,"tds")==0)
  {
    // set complete filename
    char  pFileName[256];
    sprintf(pFileName,"%s%s", APP_DATAPATH_MODELS, pFile);

    CModelLoader pModel;
    if(pModel.LoadFromFile(pFileName)==0)
    {
      return APP_ERR_READFAIL;
    }

    // Load materials
    if((m_iMaterialsCount=pModel.GetMaterialCount())) 
    {
      // Create and copy material list
      m_pMaterialList=new CMaterial[m_iMaterialsCount];

      TSGMaterial *pMaterialList=pModel.GetMaterials();
      for(int m=0; m<m_iMaterialsCount; m++)
      {
        TSGMaterial *pTmpMat=&pMaterialList[m];

        // Copy material properties
        if(pTmpMat)
        {
          CMaterial *pMaterial=&m_pMaterialList[m];
          pMaterial->SetAmbient(CColor(pTmpMat->cAmbient[0], pTmpMat->cAmbient[1], pTmpMat->cAmbient[2], 1));
          pMaterial->SetDiffuse(CColor(pTmpMat->cDiffuse[0], pTmpMat->cDiffuse[1], pTmpMat->cDiffuse[2], 1));
          pMaterial->SetSpecular(CColor(pTmpMat->cSpecular[0], pTmpMat->cSpecular[1], pTmpMat->cSpecular[2], 1));
          pMaterial->SetEmissive(CColor(pTmpMat->cEmissive[0], pTmpMat->cEmissive[1], pTmpMat->cEmissive[2], 1));
          pMaterial->SetSpecularLevel(pTmpMat->fShininess);
          pMaterial->SetDoubleSided(0);
          pMaterial->SetOpacity(pTmpMat->fOpacity);          
          m_pMaterialList[m].SetDecalTex(pTmpMat->pTextureMap);                      
          m_pMaterialList[m].SetEnvMapTex(pTmpMat->pReflectionMap);     
        }
      }
    }

    // Load groups
    if((m_iSubMeshCount=pModel.GetMeshCount()))
    {
      // Allocate mem for mesh sections
      m_pSubMeshList=new CSubMesh[m_iSubMeshCount];
                  
      // Save mesh data
      TSGMesh *pMeshList=pModel.GetMeshes();
      for(int s=0; s<m_iSubMeshCount; s++)
      {
        m_pSubMeshList[s].Create(pMeshList[s].wMaterialID, pMeshList[s].dwFaceCount, pMeshList[s].pIndices);        
      }
            
      // Create system memory vertex buffer (position + texture coordinates + normal) and copy data into it      
      m_pVertsList=new SVertex[pModel.GetVertexCount()];

      TSGVertex *pVertexList=pModel.GetVertices();
      m_iVertexCount=pModel.GetVertexCount();

      for(int v=0;v<m_iVertexCount;v++)
      {        
        memcpy(&m_pVertsList[v].pos, &pVertexList[v].vPosition, 3*sizeof(float));
        memcpy(&m_pVertsList[v].normal, &pVertexList[v].vNormal, 3*sizeof(float));
        
        m_pVertsList[v].u=pVertexList[v].vTexCoord[0];
        m_pVertsList[v].v=1.0f-pVertexList[v].vTexCoord[1];        
      }

      // Generate tangent space
      ComputeTangentSpace();

      // Create mesh vertex buffer
      m_pVB=new CVertexBuffer;
      m_pVB->Create(pModel.GetVertexCount(), (float*)m_pVertsList);

      SAFE_DELETE_ARRAY(m_pVertsList)
    }
  }
  else 
  {
    OutputMsg("Error", "File format %s not supported", pExtension);
    return APP_ERR_NOTSUPPORTED;
  }

  return APP_OK;
}

void CBaseMesh::Release()
{  
  m_iMaterialsCount=0;        
  m_iSubMeshCount=0;
  m_iVertexCount=0;
  SAFE_DELETE_ARRAY(m_pMaterialList)
  SAFE_DELETE_ARRAY(m_pSubMeshList)
  SAFE_DELETE_ARRAY(m_pVertsList)
  SAFE_DELETE(m_pVB)
}

void CBaseMesh::ComputeTangentSpace()
{
  for(int m=0; m<m_iSubMeshCount; m++)
  {
    CSubMesh *pMesh=&m_pSubMeshList[m];
    ushort *pIndices=m_pSubMeshList[m].GetIndicesList();
    
    // vertex coordinates and triangle edges
    SVertex *pA, *pB, *pC;
    CVector3f pEdgeA, pEdgeB, pTangent;
    
    for(int t=0; t<pMesh->GetFaceCount(); t++)
    {
      int a=pIndices[t*3], b=pIndices[t*3+1], c=pIndices[t*3+2];
      pA=&m_pVertsList[a];
      pB=&m_pVertsList[b];
      pC=&m_pVertsList[c];

      // compute edges
      pEdgeA=pC->pos-pB->pos; 
      pEdgeB=pA->pos-pB->pos;

      // compute gradients
      float fDeltaU1=pC->u-pB->u;
      float fDeltaU2=pA->u-pB->u;

      // scale edges
      pEdgeA*=fDeltaU1;
      pEdgeB*=fDeltaU2;

      // compute tangent vector
      pTangent=pEdgeB-pEdgeA;
      pTangent.Normalize();

      // sum up tangent vectors
      pA->tangent+=pTangent;
      pB->tangent+=pTangent;            
      pC->tangent+=pTangent;
    }  
  }

  // normalize tangent vector values
  for(int v=0; v<m_iVertexCount; v++)
  {
    m_pVertsList[v].tangent.Normalize();        
  }
}