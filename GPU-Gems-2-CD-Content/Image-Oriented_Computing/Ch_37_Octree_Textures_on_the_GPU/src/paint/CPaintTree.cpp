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
//--------------------------------------------------------

#ifdef WIN32
#  include <windows.h>
#endif

#include <cmath>
#include "CPaintTree.h"
#include "common.h"
#include "CTexture.h"

//--------------------------------------------------------

#include <iostream>
using namespace std;

//--------------------------------------------------------

#include <GL/gl.h>
#include <glux.h>

GLUX_REQUIRE(GL_EXT_texture3D);

//--------------------------------------------------------

CPaintTree::CPaintTree(std::list<CPolygon>& polys,
		       const char *fp_prg,
		       const char *vp_prg)
{
  // hardware tree
  m_HrdwTree=new CHrdwTree(PAINT_NODE_SIZE,
			   PAINT_POOL_SIZE,
			   PAINT_POOL_SIZE,
			   PAINT_POOL_SIZE,
         PAINT_MAX_DEPTH,
			   fp_prg,vp_prg);

  // create list of pointers
  std::list<CPolygon *> ptr_lst;
  for (std::list<CPolygon>::iterator P=polys.begin();P!=polys.end();P++)
    ptr_lst.push_back(&(*P));

  // root
  m_Root=new CPaintNode(this,
			NULL,
			m_HrdwTree->getRoot(),
			0,
			COBox());
  m_Root->subdivide(ptr_lst,min(PAINT_DEFAULT_DEPTH,PAINT_MAX_DEPTH));
  
  m_HrdwTree->commit();
}

//--------------------------------------------------------

CPaintTree::CPaintTree(const char *fname,
           std::list<CPolygon>& polys,
           const char *fp_prg,
	         const char *vp_prg)
{
  // hardware tree
  m_HrdwTree=new CHrdwTree(PAINT_NODE_SIZE,
			   PAINT_POOL_SIZE,
			   PAINT_POOL_SIZE,
			   PAINT_POOL_SIZE,
         PAINT_MAX_DEPTH,
			   fp_prg,vp_prg);

  // create list of pointers
  std::list<CPolygon *> ptr_lst;
  for (std::list<CPolygon>::iterator P=polys.begin();P!=polys.end();P++)
    ptr_lst.push_back(&(*P));

  // root
  m_Root=new CPaintNode(this,
			NULL,
			m_HrdwTree->getRoot(),
			0,
			COBox());

  load(fname,ptr_lst);

  m_HrdwTree->commit();
}

//--------------------------------------------------------

CPaintTree::~CPaintTree()
{
  delete (m_HrdwTree);
}

//--------------------------------------------------------

void CPaintTree::commit()
{
  m_HrdwTree->commit();
}

//--------------------------------------------------------

void CPaintTree::paint(const CVertex& pos,double radius,double opacity,
		       unsigned char r,unsigned char g,unsigned char b)
{
  m_Root->paint(pos,radius,opacity,r,g,b);
}

//--------------------------------------------------------

void CPaintTree::refine(const CVertex& pos,double radius,
			int depth,
			std::list<CPolygon>& polys)
{
  std::list<CPolygon *> ptr_lst;
  for (std::list<CPolygon>::iterator P=polys.begin();P!=polys.end();P++)
    ptr_lst.push_back(&(*P));
  m_Root->refine(pos,radius,depth,ptr_lst);
}

//--------------------------------------------------------

void CPaintTree::report()
{
  int d=m_HrdwTree->depth();

  cerr << endl;
  cerr <<   "  Node size             = " << PAINT_NODE_SIZE << endl;
  cerr <<   "  Indirection pool size = " 
       << PAINT_POOL_SIZE << 'x' 
       << PAINT_POOL_SIZE << 'x' 
       << PAINT_POOL_SIZE
       << endl;
  cerr << endl;
  cerr <<   "  Tree depth            = " << d << endl;
  cerr <<   "  Tree nb nodes         = " << m_HrdwTree->nbNodes() << endl;
  
  int allocated=m_HrdwTree->allocatedMemory();
  if (allocated >= 1024*1024)
    cerr << "  Allocated memory = " << allocated/(1024.0*1024.0) << " Mo" << endl;
  else if (allocated >= 1024)
    cerr << "  Allocated memory = " << allocated/(1024.0) << " Ko" << endl;
  else
    cerr << "  Allocated memory = " << allocated << " bytes" << endl;

  int used=m_HrdwTree->usedMemory();
  if (used >= 1024*1024)
    cerr << "  Used memory      = " << used/(1024.0*1024.0) << " Mo" << endl;
  else if (used >= 1024)
    cerr << "  Used memory      = " << used/(1024.0) << " Ko" << endl;
  else
    cerr << "  Used memory      = " << used << " bytes" << endl;
  cerr << endl << endl;

  int min=m_HrdwTree->minMemory();
  if (min >= 1024*1024)
    cerr << "  Min memory      = " << min/(1024.0*1024.0) << " Mo" << endl;
  else if (min >= 1024)
    cerr << "  Min memory      = " << min/(1024.0) << " Ko" << endl;
  else
    cerr << "  Min memory      = " << min << " bytes" << endl;
}

//--------------------------------------------------------

void CPaintTree::save(const char *fname) const
{
  FILE *fout=fopen(fname,"wb");
  if (fout == NULL)
  {
    cerr << "CPaintTree::save - cannot save to '" << fname << '\'' << endl;
    return;
  }
  m_Root->save(fout);
  fclose(fout);
  cerr << "Octree saved in file " << fname << endl;
}

//--------------------------------------------------------

void CPaintTree::load(const char *fname,const std::list<CPolygon *>& polys)
{
  FILE *fin=fopen(fname,"rb");
  if (fin == NULL)
  {
    cerr << "CPaintTree::load - cannot open '" << fname << '\'' << endl;
    return;
  }
  cerr << "Reading octree from file " << fname << " ... ";
  m_Root->load(fin,polys);
  fclose(fin);
  cerr << "done.";
}

//--------------------------------------------------------
