///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Mesh.h
//  Desc : Generic mesh class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CMaterial;
class CVertexStream;
class CVertexBuffer;
class CIndexBuffer;

class CSubMesh
{
public:
  CSubMesh(): m_iMaterialID(0), m_iFaceCount(0), m_pIB(0), m_pIndicesList(0)
  {

  }
  ~CSubMesh()
  {
    Release();
  }

  // Creators
  int Create(int iMaterialID, int iCount, const ushort *pIndexList);
  void Release();
  
  // Accessors
  int GetMaterialID() const
  {
    return m_iMaterialID;
  }

  int GetFaceCount() const
  {
    return m_iFaceCount;
  }

  const CIndexBuffer *GetIndicesBuffer() const
  {
    return m_pIB;
  }

  const ushort *GetIndexList() const
  {
    return m_pIndicesList;
  }

  // Manipulators
  CIndexBuffer *GetIndexBuffer()
  {
    return m_pIB;
  }
  
  ushort *GetIndicesList()
  {
    return m_pIndicesList;
  }

private:
  int m_iMaterialID, m_iFaceCount;
  CIndexBuffer *m_pIB;

  // tmp list
  ushort *m_pIndicesList;
};

class CBaseMesh
{
public:
  
  struct SVertex
  {
    SVertex()
    {
      pos.Set(0,0,0);
      normal.Set(0,0,0);
      tangent.Set(0,0,0);
      u=v=0;
    }

    CVector3f pos;
    float     u, v;
    CVector3f normal;
    CVector3f tangent;    
  };

  CBaseMesh(): m_iMaterialsCount(0), m_pMaterialList(0), m_pVB(0), m_pSubMeshList(0), 
    m_iSubMeshCount(0), m_pVertsList(0), m_iVertexCount(0)
  {

  }

  ~CBaseMesh()
  {
    Release();
  }

  // Load mesh from file
  int Create(const char *pFile);
  // Free resources
  void Release();

  // Return materials count
  int GetMaterialsCount() const
  {
    return m_iMaterialsCount;
  }  

  // Return mesh material
  const CMaterial *GetMaterialList() const
  {
    return m_pMaterialList;
  }

  // Accessors

  int GetSubMeshCount() const
  {
    return m_iSubMeshCount;
  }

  const CSubMesh *GetSubMeshes() const
  {
    return m_pSubMeshList;
  }

  const CVertexBuffer *GetVB() const
  {
    return m_pVB;
  }

  // Manipulators

  CSubMesh *GetSubMeshes()
  {
    return m_pSubMeshList;
  }

  CVertexBuffer *GetVB()
  {
    return m_pVB;
  }
  
private:
  
  int m_iMaterialsCount;
  CMaterial *m_pMaterialList;
  
  int m_iSubMeshCount, m_iVertexCount;
  // Mesh is contained in a single vertex buffer
  // and Sub-meshes are divided among multiple index buffers
  CVertexBuffer *m_pVB;  
  SVertex *m_pVertsList;

  CSubMesh *m_pSubMeshList;
  

  // Compute model tangent space basis
  void ComputeTangentSpace();
};


