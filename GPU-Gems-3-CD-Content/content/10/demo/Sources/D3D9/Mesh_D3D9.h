#pragma once

class Mesh_D3D9 : public Mesh
{
public:
  Mesh_D3D9();
  virtual ~Mesh_D3D9();

  virtual bool CreateBuffers(void);
  virtual void Draw(void);

public:
  LPDIRECT3DVERTEXBUFFER9 m_pVertexBuffer;
  LPDIRECT3DVERTEXDECLARATION9 m_pVertexDeclaration;
  LPDIRECT3DINDEXBUFFER9 m_pIndexBuffer;
};

// Creates a new mesh and returns it
extern Mesh *CreateNewMesh(void);