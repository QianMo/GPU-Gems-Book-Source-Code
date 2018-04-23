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
// --------------------------------------------------------

#ifndef __HRDWTREE__
#define __HRDWTREE__

// --------------------------------------------------------

#include "config_hrdwtree.h"
#include "CLibOctreeGPUException.h"

// --------------------------------------------------------

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include <list>
#include <iostream>

// -------------------------------------------------------- 

class CHrdwTree
{

public:
  
  /// Node in tree
  class hrdw_node
  {
  protected:
    // owner
    CHrdwTree     *m_Tree;
    // parent
    hrdw_node     *m_Parent;
    // depth of this node
    int            m_iDepth;
    // pos in level grid
    int            m_iGi;
    int            m_iGj;
    int            m_iGk;
    // pos in indirection grid
    // -> indirection node index
    int            m_iBi;
    int            m_iBj;
    int            m_iBk;
    // -> texture space (x,y,z) coords
    int            m_iX;
    int            m_iY;
    int            m_iZ;
    // -> resolution
    int            m_iW;
    int            m_iH;
    int            m_iD;
    // local memory data
    unsigned char *m_Data;
    // local memory data for updates
    unsigned char *m_DataUpdate;
    // children
    hrdw_node    **m_Children;
    // update flags
    bool           m_bNeedUpdate;
    bool           m_bChildrenNeedUpdate;
    int            m_UpdateMin[3];
    int            m_UpdateMax[3];
    unsigned int   m_iBytesUpdated;

    void       includeUpdate(int li,int lj,int lk);
    void       resetUpdate();
    void       setCellColor(int li,int lj,int lk,
			    unsigned char r,
			    unsigned char g,
			    unsigned char b,
			    unsigned char a);

  public:

    hrdw_node(CHrdwTree *,
	      int depth,
	      int gi,int gj,int gk,
	      int bi,int bj,int bk);
    ~hrdw_node();

    /// commit changes
    void       commit();
    /// create a child
    hrdw_node *createChild(int li,int lj,int lk);
    /// delete a child
    void       deleteChild(int li,int lj,int lk);
    /// delete a leaf
    void       deleteLeaf(int li,int lj,int lk);
    /// retrieve child's pointer
    hrdw_node *getChild(int li,int lj,int lk);
    /// set leaf color
    void       setLeafColor(int li,int lj,int lk,
			    unsigned char r,
			    unsigned char g,
			    unsigned char b);
    /// get leaf color
    void       getLeafColor(int li,int lj,int lk,
			    unsigned char& _r,
			    unsigned char& _g,
			    unsigned char& _b);
    
    // set node's parent
    void         setParent(hrdw_node *p) {m_Parent=p;}
    
    unsigned int bytesUpdated() const;
    int          nbNodes() const;
    int          depth() const;
    void         setChildrenNeedUpdate();

    // retrieve pos in indirection pool
    int bi() const {return (m_iBi);}
    int bj() const {return (m_iBj);}
    int bk() const {return (m_iBk);}
    // retrieve pos in level grid
    int gi() const {return (m_iGi);}
    int gj() const {return (m_iGj);}
    int gk() const {return (m_iGk);}

    hrdw_node *parent() const {return (m_Parent);}
    CHrdwTree *tree()   const {return (m_Tree);}
  };

protected:

  struct free_box_nfo
  {
    int bi,bj,bk;
  };

protected:

  // maximum tree depth
  int      m_iTreeMaxDepth;
  // number of indirection grids in indirection pool (S)
  int      m_NbGrids[3];
  // indirection grid size (N)
  int      m_iGridRes;
  // indirection pool 3d texture id
  GLuint   m_uiIndirPool;

  int                     m_iNextFreeI;
  int                     m_iNextFreeJ;
  int                     m_iNextFreeK;
  std::list<free_box_nfo> m_ReleasedGrids;
  
  void       allocIndirPool();

  hrdw_node *allocNode_aux(int depth,int gi,int gj,int gk);

  // Cg
  
  // vertex program
  CGprogram       m_cgVertexProg;
  CGparameter     m_cgViewProj;
  CGparameter     m_cgView;
  CGparameter     m_cgITView;

  // fragment program
  CGprogram       m_cgFragmentProg;
  CGparameter     m_cgBoxRes;
  CGparameter     m_cgRefTexCellSize;
  CGparameter     m_cgRefTex;
  CGparameter     m_cgSriteTex;
  CGparameter     m_cgTransform;
  CGparameter     m_cgLevelCellSize;

  // leaf selection (picking)
  CGprogram       m_cgLeafSelectFragmentProg;
  CGparameter     m_cgLeafSelectRefTex;
  
  // Hrdw tree root
  CHrdwTree::hrdw_node *m_Root;

public:

  CHrdwTree(int boxres,
      	    int NbGridsu,int NbGridsv,int NbGridsw,
            int maxdepth,
	          const char *fp="liboctreegpu/fp_tree.cg",
      	    const char *vp="liboctreegpu/vp_tree.cg");

  virtual ~CHrdwTree();

  // Compute the offset to be stored in node's indirection grids
  void computeOffset(int bi,int bj,int bk,
		     int gi,int gj,int gk,
		     int gridsizeu,
		     int gridsizev,
		     int gridsizew,
		     unsigned char& _offset_u,
		     unsigned char& _offset_v,
		     unsigned char& _offset_w);
  
  /// Commit changes
  void       commit();
  /// Erase the full tree
  void       reset();
  
  /// Bind the tree as texture
  virtual void       bind();
  /// Unbind
  virtual void       unbind();

  /// Bind as texture for tree leaf selection
  virtual void       bind_leafselect();
  /// Unbind leaf selection
  virtual void       unbind_leafselect();

  // Load Cg programs
  void loadPrograms(const char *fp,const char *vp);

  /// Memory size helpers

  unsigned int blockMemorySize() const;
  unsigned int allocatedMemory() const;
  unsigned int usedMemory() const;
  unsigned int minMemory() const;
  unsigned int bytesUpdated() const {return (m_Root->bytesUpdated());}
  unsigned int nbGrids(int b) const {return (m_NbGrids[b]);}
  
  // Return number of nodes in the tree
  int          nbNodes() const;
  // Return current tree depth
  int          depth() const;

  // Allocates a new tree node in indirection pool
  hrdw_node *allocNode(int depth,int gi,int gj,int gk);
  // Erase a tree node
  void       releaseNode(hrdw_node** );
  // Return indirection pool 3D texture id
  GLuint     getIndirPool() const        {return (m_uiIndirPool);}
  // Return indirection grid's resolution (N)
  int        getGridRes() const          {return (m_iGridRes);}
  // Return level 'depth' grid size 
  int        getGridSize(int depth) const;
  // Return tree root
  hrdw_node *getRoot()   const           {return (m_Root);}

  void         setCgTransform();

};

// --------------------------------------------------------

#endif

// --------------------------------------------------------
