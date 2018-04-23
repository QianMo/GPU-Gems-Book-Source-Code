/* ----------------------------------------------------------

Octree Textures on the GPU - source code - GPU Gems 2 release
                                                   2004-11-21

Updates on http://www.aracknea.net/octreetex
--
(c) 2004 Sylvain Lefebvre - all rights reserved
--
The source code is provided 'as it is', without any warranties. 
Use at your own risk. The use of any part of the source code in a
commercial or non commercial product without explicit authorisation
from the author is forbidden. Use for research and educational
purposes is allowed and encouraged, provided that a short notice
acknowledges the author's work.
---------------------------------------------------------- */
// --------------------------------------------------------------
#ifdef WIN32
#include <windows.h>
#endif

#include <cmath>
// --------------------------------------------------------------
#include "CGL3DSMesh.h"
#include "CCoreException.h"
// --------------------------------------------------------------
#include <glux.h>
//      ---
GLUX_REQUIRE(GL_ARB_vertex_program);
GLUX_REQUIRE(GL_ARB_vertex_buffer_object);
// --------------------------------------------------------------
#define BUFFER_OFFSET(o) (((char *)NULL) + o)
// --------------------------------------------------------------
CGL3DSMesh::CGL3DSMesh(const char *file3ds,const char *order)
{

  // ==================
  // load 3ds file
  m_Model=lib3ds_file_load(file3ds);
  if (NULL == m_Model) 
    throw CCoreException("Cannot open file %s",file3ds);
  // normalize model and init m_iNbPts, m_iNbFaces  
  if (order == NULL)
    order="xyz";
  normalize(m_Model,order);
  std::cerr << "Model " << file3ds << " " 
    << m_iNbPts << " points, " 
    << m_iNbFaces << " triangles." << std::endl;

  // ==================
  // create vertex and index buffers

  int nbtotpts=m_iNbPts;
  GLfloat *buffer=new GLfloat[nbtotpts*(3+2+3)];
  unsigned int *indices=new unsigned int[m_iNbFaces*3];

  m_iPtsOffs  = 0;
  m_iUVsOffs  = 3*nbtotpts;
  m_iNrmsOffs = 3*nbtotpts + 2*nbtotpts;

  GLfloat *pts  = buffer+m_iPtsOffs;
  GLfloat *uvs  = buffer+m_iUVsOffs;
  GLfloat *nrms = buffer+m_iNrmsOffs;

  srand(1234);
  int pi=0;
  int fi=0;

  Lib3dsMesh *m=NULL;

  // for each model face
  int ptsoff=pi;
  int n;
  for (n=0,m=m_Model->meshes; m; m=m->next,n++) 
  {
    for (int f=0; f<(int)m->faces; f++) 
    {
      indices[fi*3  ]=m->faceL[f].points[0]+ptsoff;
      indices[fi*3+1]=m->faceL[f].points[1]+ptsoff;
      indices[fi*3+2]=m->faceL[f].points[2]+ptsoff;
      fi++;
      // normals
      for (int k=0;k<3;k++)
      {
        nrms[indices[(fi-1)*3+k]*3  ]=m_Normals[n][f*3+k][order[0]-'x'];
        nrms[indices[(fi-1)*3+k]*3+1]=m_Normals[n][f*3+k][order[1]-'x'];
        nrms[indices[(fi-1)*3+k]*3+2]=m_Normals[n][f*3+k][order[2]-'x'];
      }
    }
    ptsoff+=m->points;
  }

  // for each model point
  for (m=m_Model->meshes; m; m=m->next) 
  {
    for (int p=0; p<(int)m->points; p++) 
    {
      CVertex pt=CVertex(m->pointL[p].pos[0],
        m->pointL[p].pos[1],
        m->pointL[p].pos[2]);
      pts[pi*3  ]=pt.x();
      pts[pi*3+1]=pt.y();
      pts[pi*3+2]=pt.z();
      if (m->texelL != NULL)
      {
        uvs[pi*2  ]=m->texelL[p][0];
        uvs[pi*2+1]=m->texelL[p][1];
      }
      else
      {
        uvs[pi*2  ]=pt.x();
        uvs[pi*2+1]=pt.y();	
      }
      pi++;
    }
  }

  // create display list
  create_list(m_iNbFaces,buffer,indices);
  create_uv_list(m_iNbFaces,buffer,indices);
  create_atlas_list(m_iNbFaces,buffer,indices);

  // create triangles list
  create_tris(m_iNbFaces,buffer,indices);

  delete [](buffer);
  delete [](indices);

}
// --------------------------------------------------------
void CGL3DSMesh::draw()
{
  glPushAttrib(GL_ENABLE_BIT);

  glDisable(GL_CULL_FACE);

  glCallList(m_DisplayList);

  glPopAttrib();
}
// --------------------------------------------------------
void CGL3DSMesh::draw_uv()
{
  glPushAttrib(GL_ENABLE_BIT);

  glDisable(GL_CULL_FACE);

  glCallList(m_UVDisplayList);

  glPopAttrib();
}
// --------------------------------------------------------
void CGL3DSMesh::draw_atlas()
{
  glPushAttrib(GL_ENABLE_BIT);

  glDisable(GL_CULL_FACE);

  glCallList(m_AtlasDisplayList);

  glPopAttrib();
}
// --------------------------------------------------------
void CGL3DSMesh::create_list(int nbf,
                             GLfloat *buffer,
                             unsigned int *indices)
{
  m_DisplayList=glGenLists(1);
  glNewList(m_DisplayList,GL_COMPILE);
  glBegin(GL_TRIANGLES);
  GLfloat *pts=buffer+m_iPtsOffs;
  GLfloat *nrms=buffer+m_iNrmsOffs;
  for (int f=0;f<nbf;f++)
  {
    int i0=indices[f*3+0];
    int i1=indices[f*3+1];
    int i2=indices[f*3+2];

    glNormal3fv(&nrms[i0*3]);
    glTexCoord3fv(&pts[i0*3]);
    glVertex3fv(&pts[i0*3]);

    glNormal3fv(&nrms[i1*3]);
    glTexCoord3fv(&pts[i1*3]);
    glVertex3fv(&pts[i1*3]);

    glNormal3fv(&nrms[i2*3]);
    glTexCoord3fv(&pts[i2*3]);
    glVertex3fv(&pts[i2*3]);
  }
  glEnd();
  glEndList();
}
// --------------------------------------------------------
void CGL3DSMesh::create_uv_list(int nbf,
                                GLfloat *buffer,
                                unsigned int *indices)
{
  m_UVDisplayList=glGenLists(1);
  glNewList(m_UVDisplayList,GL_COMPILE);
  glBegin(GL_TRIANGLES);
  GLfloat *pts=buffer+m_iPtsOffs;
  GLfloat *uvs=buffer+m_iUVsOffs;
  GLfloat *nrms=buffer+m_iNrmsOffs;
  for (int f=0;f<nbf;f++)
  {
    int i0=indices[f*3+0];
    int i1=indices[f*3+1];
    int i2=indices[f*3+2];

    glNormal3fv(&nrms[i0*3]);
    glTexCoord2fv(&uvs[i0*2]);
    glVertex3fv(&pts[i0*3]);

    glNormal3fv(&nrms[i1*3]);
    glTexCoord2fv(&uvs[i1*2]);
    glVertex3fv(&pts[i1*3]);

    glNormal3fv(&nrms[i2*3]);
    glTexCoord2fv(&uvs[i2*2]);
    glVertex3fv(&pts[i2*3]);
  }
  glEnd();
  glEndList();
}
// --------------------------------------------------------
void CGL3DSMesh::create_atlas_list(int nbf,
                                   GLfloat *buffer,
                                   unsigned int *indices)
{
  m_AtlasDisplayList=glGenLists(1);
  glNewList(m_AtlasDisplayList,GL_COMPILE);
  glBegin(GL_TRIANGLES);
  GLfloat *pts=buffer+m_iPtsOffs;
  GLfloat *uvs=buffer+m_iUVsOffs;
  glNormal3f(0,0,1.0f);
  for (int f=0;f<nbf;f++)
  {
    int i0=indices[f*3+0];
    int i1=indices[f*3+1];
    int i2=indices[f*3+2];

    glTexCoord3fv(&pts[i0*3]);
    glVertex2fv(&uvs[i0*2]);

    glTexCoord3fv(&pts[i1*3]);
    glVertex2fv(&uvs[i1*2]);

    glTexCoord3fv(&pts[i2*3]);
    glVertex2fv(&uvs[i2*2]);
  }
  glEnd();
  glEndList();
}
// --------------------------------------------------------
void CGL3DSMesh::create_tris(int nbf,
                             GLfloat *buffer,
                             unsigned int *indices)
{
  GLfloat *pts=buffer+m_iPtsOffs;
  GLfloat *nrms=buffer+m_iNrmsOffs;
  for (int i=0;i<m_iNbPts;i++)
  {
    mesh3ds_vertex v;
    v.v=CVertex(pts[i*3],pts[i*3+1],pts[i*3+2]);
    v.n=CVertex(nrms[i*3],nrms[i*3+1],nrms[i*3+2]);
    v.t=v.v;

    m_Vertices.push_back(v);
  }
  for (int f=0;f<nbf;f++)
  {

    mesh3ds_tri tri;
    tri.i=indices[f*3+0];
    tri.j=indices[f*3+1];
    tri.k=indices[f*3+2];

    m_Tris.push_back(tri);
  }
}
// --------------------------------------------------------
void CGL3DSMesh::normalize(Lib3dsFile *f,const char *order)
{
  Lib3dsMesh  *m;
  unsigned     i;

  m_iNbPts=0;
  m_iNbFaces=0;

  double minx=999999.0,maxx=-999999.0;
  double miny=999999.0,maxy=-999999.0;
  double minz=999999.0,maxz=-999999.0;

  int nbmesh=0;
  for (m=f->meshes; m; m=m->next) 
  {
    m_iNbPts+=m->points;
    m_iNbFaces+=m->faces;
    for (i=0; i<m->points; i++) 
    {
      double x=m->pointL[i].pos[order[0]-'x'];
      double y=m->pointL[i].pos[order[1]-'x'];
      double z=m->pointL[i].pos[order[2]-'x'];
      if (x > maxx) maxx=x;
      if (x < minx) minx=x;
      if (y > maxy) maxy=y;
      if (y < miny) miny=y;
      if (z > maxz) maxz=z;
      if (z < minz) minz=z;
    }
    nbmesh++;
  }
  m_Normals=new Lib3dsVector*[nbmesh];
  int n;
  double maxd=max((maxx-minx),max((maxy-miny),(maxz-minz)));
  for (n=0,m=f->meshes; m; m=m->next,n++) 
  {
    for (i=0; i<m->points; i++) 
    {
      double x=m->pointL[i].pos[order[0]-'x'];
      double y=m->pointL[i].pos[order[1]-'x'];
      double z=m->pointL[i].pos[order[2]-'x'];
      m->pointL[i].pos[0]=(vertex_real)((x-minx)/maxd);
      m->pointL[i].pos[1]=(vertex_real)((y-miny)/maxd);
      m->pointL[i].pos[2]=(vertex_real)((z-minz)/maxd);
    }
    m_Normals[n]=new Lib3dsVector[(3*sizeof(Lib3dsVector)*m->faces)];
    lib3ds_mesh_calculate_normals(m,m_Normals[n]);
  }
  m_Center=CVertex((vertex_real)(((maxx+minx)/2.0f-minx)/maxd),
                   (vertex_real)(((maxy+miny)/2.0f-miny)/maxd),
                   (vertex_real)(((maxz+minz)/2.0f-minz)/maxd));
}
// --------------------------------------------------------------
