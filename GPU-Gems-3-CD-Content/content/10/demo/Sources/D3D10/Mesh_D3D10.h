#pragma once

class Mesh_D3D10 : public Mesh
{
public:
  Mesh_D3D10();
  virtual ~Mesh_D3D10();

  virtual bool CreateBuffers(void);
  virtual void Draw(void);
  virtual void DrawInstanced(int iNumInstances);

public:
  ID3D10Buffer *m_pVertexBuffer;
  ID3D10Buffer *m_pIndexBuffer;
  ID3D10InputLayout *m_pVertexLayout;
};

// Creates a new mesh and returns it
extern Mesh *CreateNewMesh(void);