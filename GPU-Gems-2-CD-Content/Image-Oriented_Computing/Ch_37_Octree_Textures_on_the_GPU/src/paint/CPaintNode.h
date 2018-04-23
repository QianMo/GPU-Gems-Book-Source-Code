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

#ifndef __PAINTNODE__
#define __PAINTNODE__

//--------------------------------------------------------

#define PAINT_NODE_SIZE         2
#define PAINT_MAX_DEPTH        10
#define PAINT_DEFAULT_DEPTH     8
#define PAINT_POOL_SIZE       128

//--------------------------------------------------------

#include "CVertex.h"
#include "CBox.h"
#include "CHrdwTree.h"

#include <list>
#include <set>

//--------------------------------------------------------

class CPaintTree;
class CTexture;

//--------------------------------------------------------

class CPaintNode
{
 
protected:

  // children
  CPaintNode            *m_Childs[PAINT_NODE_SIZE*PAINT_NODE_SIZE*PAINT_NODE_SIZE];
  // parent
  CPaintNode            *m_Parent;
  // owner
  CPaintTree            *m_Tree;
  // gpu twin node
  CHrdwTree::hrdw_node  *m_HrdwTwin;
  // node's depth
  int                    m_iDepth;
  // bounding box
  COBox                  m_Box;
  // intersected polygons
  std::list<CPolygon *>  m_Polys;
  // visited flag
  bool                   m_bVisited;

  // Compute the bounding box of a given children.
  std::pair<COBox,COBox> childBox(int i,int j,int k);

public:
  
  CPaintNode(CPaintTree     *, // owner
	     CPaintNode           *, // parent
	     CHrdwTree::hrdw_node *, // gpu twin node
	     int   depth,            // node's depth
	     const COBox&);          // bounding box
  ~CPaintNode();

  // Apply paint on the mesh.
  void paint(const CVertex& pos,double radius,double opacity,
	     unsigned char r,unsigned char g,unsigned char b);

  // Apply refinment brush.
  void refine(const CVertex& pos,double radius,
	      int depth,
	      const std::list<CPolygon *>& polys);

  // Recursively subdivide the nodes (build the tree).
  void subdivide(const std::list<CPolygon *>& polys,
		 int maxdeph,
		 bool trackleafcolor=false,
		 unsigned char cr=255,
		 unsigned char cg=255,
		 unsigned char cb=255);

  // Draw a subset of the tree.
  void drawSubset(const CVertex& pos,double radius,int depth) const;

  // To be called when drawing start (a stroke begins).
  void startDrawing();

  // Save octree to file
  void save(FILE *fout) const;
  // Load octree from file
  void load(FILE *fin,const std::list<CPolygon *>& polys);

  // Draw the tree structure, for visualization purpose only.
  void draw_structure(bool draw_empty);

  const COBox& box() const {return (m_Box);}
  CVertex      center() const {return (m_Box.center());}
};

//--------------------------------------------------------

#endif

//--------------------------------------------------------
