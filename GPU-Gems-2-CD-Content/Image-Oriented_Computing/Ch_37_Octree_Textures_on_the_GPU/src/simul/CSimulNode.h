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

#ifndef __SIMULNODE__
#define __SIMULNODE__

//--------------------------------------------------------

#define SIMUL_NODE_SIZE 2

//--------------------------------------------------------

#include "CVertex.h"
#include "CBox.h"
#include "CHrdwTree.h"

#include <list>
#include <set>

//--------------------------------------------------------

class CSimulTree;
class CTexture;

//--------------------------------------------------------

class CSimulNode
{
 
protected:

  CSimulNode   *m_Childs[SIMUL_NODE_SIZE*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE];

  CHrdwTree::hrdw_node *m_HrdwTwin;
  unsigned char         m_iDepth;
  //COBox                 m_Box;
  CVertex               m_Center;

  bool                  m_bIsLeaf;

  unsigned short        m_uiI;
  unsigned short        m_uiJ;
  unsigned short        m_uiK;

  short                 m_iU;
  short                 m_iV;

  CVertex               m_Normal;

  float                 m_fArea;

  CSimulTree           *m_Tree;

public:
  // node
  CSimulNode(CSimulTree *,
	     const std::list<CPolygon *>& polys,
	     CHrdwTree::hrdw_node *,
	     short depth,
	     short ni,
	     short nj,
	     short nk,
	     const COBox&);
  // leaf
  CSimulNode(CSimulTree *,
	     short depth,
	     short ni,
	     short nj,
	     short nk,
	     const COBox&);
  ~CSimulNode();

  void draw_structure(bool draw_empty);
  void draw_node();

  int  depth();

  CSimulNode *getNode(int i,int j,int k,int d);

  int  getU() const {return (m_iU);}
  int  getV() const {return (m_iV);}
  int  getI() const {return (m_uiI);}
  int  getJ() const {return (m_uiJ);}
  int  getK() const {return (m_uiK);}
  int  getDepth() const {return (m_iDepth);}

  const CVertex& getCenter() const {return (m_Center);}
  const CVertex& getNormal() const {return (m_Normal);}
  float          getArea()   const {return (m_fArea);}

};

//--------------------------------------------------------

#endif

//--------------------------------------------------------
