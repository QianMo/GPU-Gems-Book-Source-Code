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
// 2003-07-15
//
// This really is an ugly piece of code - DO NOT USE IT in real apps !!!
//
// --------------------------------------------------------------
// --------------------------------------------------------------
#ifndef __C3DSMesh__
#define __C3DSMesh__
// --------------------------------------------------------------
#include <list>
#include <vector>
// --------------------------------------------------------------
#include <lib3ds/file.h>
#include <lib3ds/mesh.h>
#include <lib3ds/vector.h>
#include <lib3ds/material.h>
// --------------------------------------------------------------
#include "CVertex.h"
#include "CPolygon.h"
//----------------------------------------------------------
typedef struct mesh3ds_vertex
{
  CVertex v;
  CVertex t;
  CVertex n;
  CVertex uv;
}mesh3ds_vertex;
//----------------------------------------------------------
typedef struct mesh3ds_tri
{
  int     i,j,k;
}mesh3ds_tri;
// --------------------------------------------------------------
class CGL3DSMesh
{
protected:

  int         m_iPtsOffs;
  int         m_iNrmsOffs;
  int         m_iUVsOffs;
  GLuint      m_DisplayList;
  GLuint      m_UVDisplayList;
  GLuint      m_AtlasDisplayList;
  CVertex     m_Center;

  Lib3dsFile    *m_Model;
  Lib3dsVector **m_Normals;
  int            m_iNbPts;
  int            m_iNbFaces;

  std::vector<mesh3ds_vertex> m_Vertices;
  std::vector<mesh3ds_tri>    m_Tris;

  // model tools
  void normalize(Lib3dsFile *f,const char *order);
  void create_list(int nbf,
    GLfloat *buffer,
    unsigned int *indices);
  void create_tris(int nbf,
    GLfloat *buffer,
    unsigned int *indices);
  void create_uv_list(int nbf,
    GLfloat *buffer,
    unsigned int *indices);
  void create_atlas_list(int nbf,
    GLfloat *buffer,
    unsigned int *indices);

public:

  CGL3DSMesh(const char *file3ds,const char *order=NULL);

  void draw();
  void draw_atlas();
  void draw_uv();

  void polygons(std::list<CPolygon> &_polys)
  {
    for (std::vector<mesh3ds_tri>::iterator I=m_Tris.begin();
      I!=m_Tris.end();I++)
    {
      CPolygon p(m_Vertices[(*I).i].t,
        m_Vertices[(*I).j].t,
        m_Vertices[(*I).k].t);
      _polys.push_back(p);
    }
  }

  CVertex center() const {return (m_Center);}
};
// --------------------------------------------------------------
#endif
// --------------------------------------------------------------
