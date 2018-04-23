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

//--------------------------------------------------------

#include "CLibOctreeGPUException.h"
#include "CPaintNode.h"
#include "CPaintTree.h"
#include "CTexture.h"

#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace std;

//--------------------------------------------------------

CPaintNode::CPaintNode(CPaintTree *tree,
                       CPaintNode *parent,
                       CHrdwTree::hrdw_node *twin,
                       int depth,
                       const COBox& b)
{
  m_Tree=tree;
  m_HrdwTwin=twin;
  m_iDepth=depth;
  m_Box=b;
  m_Parent=parent;
  m_bVisited=false;
}

//--------------------------------------------------------

/**
Compute the bounding box of a given children.

  Two boxes are returned: the real bounding box, and an enlarged
  bounding box in order to create the tree for correct linear 
  interpolation
*/
std::pair<COBox,COBox> CPaintNode::childBox(int i,int j,int k)
{
  static const CVertex local_size(1.0/PAINT_NODE_SIZE,
                                  1.0/PAINT_NODE_SIZE,
                                  1.0/PAINT_NODE_SIZE);

  double di=(double)i/(double)(PAINT_NODE_SIZE);
  double dj=(double)j/(double)(PAINT_NODE_SIZE);
  double dk=(double)k/(double)(PAINT_NODE_SIZE);
  CVertex local_pos((vertex_real)di,(vertex_real)dj,(vertex_real)dk);

  COBox box=m_Box.sub(local_pos,local_size);

  // enlarge box for linear interpolation
  COBox bx;
  float sz=1.0/(float)(1<<PAINT_MAX_DEPTH);
  CVertex u=(box.p100()-box.p000());
  CVertex v=(box.p010()-box.p000());
  CVertex w=(box.p001()-box.p000());
  CVertex nu=u;
  nu.normalize();
  CVertex nv=v;
  nv.normalize();
  CVertex nw=w;
  nw.normalize();
  u += nu*sz;
  v += nv*sz;
  w += nw*sz;
  bx=COBox(box.p000()-nu*sz-nv*sz-nw*sz,u,v,w);

  return (make_pair(box,bx));
}

//--------------------------------------------------------

/**
Recursively subdivide the nodes (build the tree).
  Nodes are subdivided until maximum depth is reached
  or the node does not contain any part of the mesh.
*/
void CPaintNode::subdivide(const std::list<CPolygon *>& polys,
                           int           maxdepth,
                           bool          trackleafcolor,
                           unsigned char r,
                           unsigned char g,
                           unsigned char b)
{
  m_Polys=polys;
  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)  
      {
        std::list<CPolygon *> cutpolys;
        std::pair<COBox,COBox> boxes=childBox(i,j,k);

        if (m_iDepth < maxdepth-1)
        { // if max depth not reached, test against geometry only
          for (std::list<CPolygon *>::const_iterator P=polys.begin();
            P!=polys.end();P++)
          {
            CPolygon r;
            boxes.second.cut(*(*P),r);
            if (!r.empty())
            {
              cutpolys.push_back((*P));
            }
          }
        }
        CPaintNode *child=NULL;
        if (!cutpolys.empty()) 
        { // child contains geometry
          try
          {
            unsigned char lr=r,lg=g,lb=b;
            if (trackleafcolor)
            {
              m_HrdwTwin->getLeafColor(i,j,k,lr,lg,lb);
            }
            CHrdwTree::hrdw_node *hchild=m_HrdwTwin->createChild(i,j,k); // create a child
            child=new CPaintNode(m_Tree,
              this,
              hchild,
              m_iDepth+1,
              boxes.first);
            // recurs
            child->subdivide(cutpolys,maxdepth,false,lr,lg,lb);
          }
          catch (...)
          {
            child=NULL;
            m_HrdwTwin->setLeafColor(i,j,k,
              255,
              0,
              0);
          }
        }
        else 
        {
          // child is a leaf
          m_HrdwTwin->setLeafColor(i,j,k,
            r,g,b);
        }
        m_Childs[i
          +j*PAINT_NODE_SIZE 
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE]=child;
      }
    }
  }
}

//--------------------------------------------------------

CPaintNode::~CPaintNode()
{
  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)  
      {
        if (m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE] != NULL)
          delete (m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE]);
      }
    }
  }
}

//--------------------------------------------------------

/**
Draw the tree structure, for visualization purpose only.
*/
void CPaintNode::draw_structure(bool draw_empty)
{
  m_Box.draw_box_line();
  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)
      {
        CPaintNode *child=m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE];

        if (child != NULL)
          child->draw_structure(draw_empty);
      }
    }
  }
  glLineWidth(1.0);
}

//--------------------------------------------------------

/**
To be called when drawing start (a stroke begins).
  This mechanism is used to prevent iterative brushes (like
  refinment brush) to be applied multiple times at a same 
  location during a single stroke.
*/
void CPaintNode::startDrawing()
{
  // clear visited flag
  m_bVisited=false;

  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)
      {
        CPaintNode *child=m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE];

        if (child != NULL)
          child->startDrawing();
      }
    }
  }
}

//--------------------------------------------------------

/**
Apply paint on the mesh.
*/
void CPaintNode::paint(const CVertex& pos,double radius,double opacity,
                       unsigned char r,unsigned char g,unsigned char b)
{
  if (m_bVisited)
    return;

  bool leaf=true;

  // recursively paint children
  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)
      {
        CPaintNode *child=m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE];

        if (child != NULL)
        {
          leaf=false;
          double d =(pos - child->center()).length();
          double sz=1.0/(pow((double)PAINT_NODE_SIZE,(double)child->m_iDepth));
          // if child within brush, recurs
          if (d < radius + sz*1.41)
            child->paint(pos,radius,opacity,r,g,b);
        }
      }
    }
  }  

  // if current node is a leaf, apply colors
  if (leaf)
  {
    for (int i=0;i<PAINT_NODE_SIZE;i++)
    {
      for (int j=0;j<PAINT_NODE_SIZE;j++)
      {
        for (int k=0;k<PAINT_NODE_SIZE;k++)
        {
          // compute brush effect
          unsigned char pr,pg,pb;
          m_HrdwTwin->getLeafColor(i,j,k,
            pr,pg,pb);

          double sz        = 1.0/(pow((double)PAINT_NODE_SIZE,(double)m_iDepth+1));
          CVertex  local   = pos - (center() + (vertex_real)sz*CVertex((vertex_real)i,(vertex_real)j,(vertex_real)k));
          double d         = (double)local.length();
          CVertex nrmlocal = local / (float)radius;

          double falloff = (1.0 - max(0.0,min(1.0,(d - radius*0.5)/(0.5*radius))));
          
		  // uncomment for procedural brush pattern
		  //double pattern = (0.7+0.3*sin(5.0*pos.x()+nrmlocal.x()*6.28))
          //  *(0.7+0.3*sin(3.0*pos.y()+nrmlocal.y()*6.28))
          //  *(0.7+0.3*sin(1.0*pos.z()+nrmlocal.z()*6.28));
          
		  double pattern = 1.0;
          
		  falloff       *= opacity*pattern;
          double dr      = falloff * (double)r + (1.0-falloff) * (double)pr;
          double dg      = falloff * (double)g + (1.0-falloff) * (double)pg;
          double db      = falloff * (double)b + (1.0-falloff) * (double)pb;

          // set leaf color
          m_HrdwTwin->setLeafColor(i,j,k,
            (unsigned char)dr,
            (unsigned char)dg,
            (unsigned char)db);

        }
      }
    }
  }

}

//--------------------------------------------------------

/**
Apply refinment brush.
  Add one level to tree depth within the brush.
*/
void CPaintNode::refine(const CVertex& pos,double radius,
                        int depth,
                        const std::list<CPolygon *>& polys)
{
  if (m_bVisited)
    return;

  bool leaf=true;

  if (m_iDepth == PAINT_MAX_DEPTH)
    return;

  // recursively apply
  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)  
      {

        // examine child
        CPaintNode *child=m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE];

        if (child != NULL)
        {
          double d =(pos - child->center()).length();
          double sz=1.0/(pow((double)PAINT_NODE_SIZE,(double)child->m_iDepth));
          if (d < radius + sz*1.41)
          {		
            // recurs
            child->refine(pos,radius,depth,m_Polys);
          }
          leaf=false;
        }
      }
    }
  }

  // if current node is a leaf, subdivide it
  if (leaf)
  {
    if (m_iDepth < PAINT_MAX_DEPTH-1)
    {
      subdivide(m_Polys,min(m_iDepth+2,PAINT_MAX_DEPTH),true);
      m_bVisited=true;
    }
  }

}

//--------------------------------------------------------

/**
Draw a subset of the tree
  This is usefull on slow computers to paint on only
  a small part of the geometry.
*/
void CPaintNode::drawSubset(const CVertex& pos,double radius,
                            int depth) const
{
  if (m_iDepth == depth)
  {
    for (std::list<CPolygon *>::const_iterator P=m_Polys.begin();P != m_Polys.end();P++)
      (*P)->gl();
  }
  else
  {
    for (int i=0;i<PAINT_NODE_SIZE;i++)
    {
      for (int j=0;j<PAINT_NODE_SIZE;j++)
      {
        for (int k=0;k<PAINT_NODE_SIZE;k++)  
        {

          // examine child
          CPaintNode *child=m_Childs[i
            +j*PAINT_NODE_SIZE
            +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE];

          if (child != NULL)
          {
            double d =(pos - child->center()).length();
            double sz=1.0/(pow((double)PAINT_NODE_SIZE,(double)child->m_iDepth));
            if (d < radius + sz*1.41)
            {
              // recurs
              child->drawSubset(pos,radius,depth);
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------------------

/**
Save octree texture
*/
void CPaintNode::save(FILE *fout) const
{
  static char id_leaf ='L';
  static char id_node ='N';

  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)  
      {
        CPaintNode *child=m_Childs[i
          +j*PAINT_NODE_SIZE
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE];
        if (child == NULL)
        {
          fwrite(&id_leaf,1,1,fout);
          unsigned char r,g,b;
          m_HrdwTwin->getLeafColor(i,j,k,r,g,b);
          fwrite(&r,1,1,fout);
          fwrite(&g,1,1,fout);
          fwrite(&b,1,1,fout);
        }
        else
        {
          fwrite(&id_node,1,1,fout);
          child->save(fout);
        }
      }
    }
  }
}

//--------------------------------------------------------

/**
Load octree texture
*/
void CPaintNode::load(FILE *fin,
                      const std::list<CPolygon *>& polys)
{
  char          c;
  unsigned char r,g,b;

  m_Polys=polys;
  for (int i=0;i<PAINT_NODE_SIZE;i++)
  {
    for (int j=0;j<PAINT_NODE_SIZE;j++)
    {
      for (int k=0;k<PAINT_NODE_SIZE;k++)  
      {
        CPaintNode *child=NULL;

        fread(&c,1,1,fin);
        if (c == 'L')
        {
          fread(&r,1,1,fin);
          fread(&g,1,1,fin);
          fread(&b,1,1,fin);

          m_HrdwTwin->setLeafColor(i,j,k,r,g,b);
        }
        else if (c == 'N')
        {
          // cut polygons
          std::list<CPolygon *> cutpolys;
          std::pair<COBox,COBox> boxes=childBox(i,j,k);

          for (std::list<CPolygon *>::const_iterator P=polys.begin();
            P!=polys.end();P++)
          {
            CPolygon r;
            boxes.second.cut(*(*P),r);
            if (!r.empty())
            {
              cutpolys.push_back((*P));
            }
          }

          try
          {
            CHrdwTree::hrdw_node *hchild=m_HrdwTwin->createChild(i,j,k); // create a child
            child=new CPaintNode(m_Tree,
              this,
              hchild,
              m_iDepth+1,
              boxes.first);
            child->load(fin,cutpolys);
          }
          catch (...)
          {
            child=NULL;
            m_HrdwTwin->setLeafColor(i,j,k,
              255,
              0,
              0);
          }
        }
        else
        {
          cerr << "[WARNING] CPaintNode::load - unknown node content '" << c << '\'' << endl;
        }
        m_Childs[i
          +j*PAINT_NODE_SIZE 
          +k*PAINT_NODE_SIZE*PAINT_NODE_SIZE]=child;
      }
    }
  }
}

//--------------------------------------------------------

