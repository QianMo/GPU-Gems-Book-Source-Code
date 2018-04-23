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

#ifndef __PAINTTREE__
#define __PAINTTREE__

//--------------------------------------------------------

#include "CVertex.h"
#include "CBox.h"
#include "CPaintNode.h"

//--------------------------------------------------------

class CHrdwSpriteManager;
class CTexture;

//--------------------------------------------------------

class CPaintTree
{

protected:

  // tree root
  CPaintNode       *m_Root;
  // hardware tree
  CHrdwTree        *m_HrdwTree;

public:

   // Create a new tree.
   CPaintTree(std::list<CPolygon>& polys,
	      const char *fp_prg,
	      const char *vp_prg="liboctreegpu/vp_tree.cg");

   // Load from file.
   CPaintTree(const char *fname,
        std::list<CPolygon>& polys,
	      const char *fp_prg,
	      const char *vp_prg="liboctreegpu/vp_tree.cg");

  ~CPaintTree();

  // Commit changes (update tree on GPU).
  void commit();
  // Report tree properties.
  void report();

  // Bind octree texture.
  void bind()   {m_HrdwTree->bind();}
  // Unbind octree texture.
  void unbind() {m_HrdwTree->unbind();}
  // Update object's tranform (modelview): to be called before drawing objects
  void setCgTransform()       {m_HrdwTree->setCgTransform();}

  // Apply paint to the tree.
  void paint(const CVertex& pos,double radius,double opacity,
	     unsigned char r,unsigned char g,unsigned char b);

  // Apply refinment brush.
  void refine(const CVertex& pos,double radius,
	      int depth,
	      std::list<CPolygon>& polys);

  // Draw a subset of the tree.
  void drawSubset(const CVertex& pos,double radius,int depth) const {m_Root->drawSubset(pos,radius,depth);}

  // To be called when drawing start (a stroke begins).
  void startDrawing() {m_Root->startDrawing();}

  // Save octree to file.
  void save(const char *) const;
  // Load octree from file.
  void load(const char *,const std::list<CPolygon *>& polys);

  // Change programs
  void changePrograms(const char *fp,const char *vp="liboctreegpu/vp_tree.cg") const {m_HrdwTree->loadPrograms(fp,vp);m_HrdwTree->commit();}

  CHrdwTree      *getHrdwTree() {return (m_HrdwTree);}
  CPaintNode     *getRoot()     {return (m_Root);}
  
};

//--------------------------------------------------------

#endif

//--------------------------------------------------------
