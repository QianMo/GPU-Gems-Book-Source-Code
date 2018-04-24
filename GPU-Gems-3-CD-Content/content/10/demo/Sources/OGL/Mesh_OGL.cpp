#include "../Framework/Common.h"
#include "Application_OGL.h"
#include "../Framework/Mesh.h"
#include "Mesh_OGL.h"

Mesh *CreateNewMesh(void)
{
  return new Mesh_OGL();
}

Mesh_OGL::Mesh_OGL()
{
  m_iVertexBuffer = 0;
  m_iIndexBuffer = 0;
}

Mesh_OGL::~Mesh_OGL()
{
  if(m_iVertexBuffer != 0)
  {
    glDeleteBuffersARB(1, &m_iVertexBuffer);
    m_iVertexBuffer = 0;
  }
  if(m_iIndexBuffer != 0)
  {
    glDeleteBuffersARB(1, &m_iIndexBuffer);
    m_iIndexBuffer = 0;
  }
}

bool Mesh_OGL::CreateBuffers(void)
{
  // vertex buffer
  //
  glGenBuffersARB(1, &m_iVertexBuffer);
  if(m_iVertexBuffer == 0)
  {
    MessageBox(NULL, TEXT("Could not create vertex buffer!"), TEXT("Error!"), MB_OK);
    return false;
  }
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_iVertexBuffer);
  glBufferDataARB(GL_ARRAY_BUFFER_ARB, m_iNumVertices*m_iVertexSize, m_pVertices, GL_STATIC_DRAW_ARB);

  // index buffer
  //
  glGenBuffersARB(1, &m_iIndexBuffer);
  if(m_iIndexBuffer == 0)
  {
    MessageBox(NULL, TEXT("Could not create index buffer!"), TEXT("Error!"), MB_OK);
    return false;
  }
  glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_iIndexBuffer);
  glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_iNumTris*3*sizeof(unsigned short), m_pIndices, GL_STATIC_DRAW_ARB);

  return true;
}

extern int g_iTrisPerFrame;

void Mesh_OGL::Draw(void)
{
  char *pPointer = NULL;

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_iVertexBuffer);
  glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_iIndexBuffer);
	
  glVertexPointer(3, GL_FLOAT, m_iVertexSize, pPointer);
  glNormalPointer(GL_FLOAT, m_iVertexSize, pPointer+sizeof(float)*3);
	glColorPointer(3, GL_FLOAT, m_iVertexSize, pPointer+sizeof(float)*6);

  glDrawElements(GL_TRIANGLES, m_iNumTris*3, GL_UNSIGNED_SHORT, 0);
  
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
  g_iTrisPerFrame += m_iNumTris;
}

void Mesh_OGL::DrawInstanced(int iNumInstances)
{
  char *pPointer = NULL;

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_iVertexBuffer);
  glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, m_iIndexBuffer);
	
  glVertexPointer(3, GL_FLOAT, m_iVertexSize, pPointer);
  glNormalPointer(GL_FLOAT, m_iVertexSize, pPointer+sizeof(float)*3);
	glColorPointer(3, GL_FLOAT, m_iVertexSize, pPointer+sizeof(float)*6);

  glDrawElementsInstancedEXT(GL_TRIANGLES, m_iNumTris*3, GL_UNSIGNED_SHORT, 0, iNumInstances);
  
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
  g_iTrisPerFrame += m_iNumTris;
}