#pragma once

class Mesh_OGL : public Mesh
{
public:
  Mesh_OGL();
  virtual ~Mesh_OGL();

  virtual bool CreateBuffers(void);
  virtual void Draw(void);
  virtual void DrawInstanced(int iNumInstances);

public:
  GLuint m_iVertexBuffer;
  GLuint m_iIndexBuffer;
};

// Creates a new mesh and returns it
extern Mesh *CreateNewMesh(void);