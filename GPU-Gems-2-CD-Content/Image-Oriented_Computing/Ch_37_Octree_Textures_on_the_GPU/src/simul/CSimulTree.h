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

#ifndef __SIMULTREE__
#define __SIMULTREE__

//--------------------------------------------------------

#include "CVertex.h"
#include "CBox.h"
#include "CHrdwSimulTree.h"
#include "CSimulNode.h"
#include "pbuffer.h"

#include "config.h"

//--------------------------------------------------------

#include <vector>
#include <map>
using namespace std;

//--------------------------------------------------------

class CTexture;

//--------------------------------------------------------

#define CELL(i,j,k) ((i)+(j)*3+(k)*3*3)

//--------------------------------------------------------

#define SIMULTREE_AVAILABLE_TEX 15

//--------------------------------------------------------

class cell_nfo
{
public:
  double value;
  bool   used;

  CSimulNode *write_node;
  CSimulNode *read_node;
  int read_u;
  int read_v;
  int write_u;
  int write_v;
  
  bool operator < (const cell_nfo& c) const
    {
      return (value > c.value);
    };

  class cmp
  {
  public:
    bool operator ()(const cell_nfo *c0,const cell_nfo *c1) const
      {
	return (c0->value > c1->value);
      }
  };
};

//----------

#ifdef SIMULTREE_8BIT
typedef unsigned char pix_type;
#else
typedef GLfloat pix_type;
#endif

//----------

class CSimulTree
{

protected:

  // tree root
  CSimulNode       *m_Root;
  // hardware tree
  CHrdwSimulTree   *m_HrdwTree;

  // registration
  int              m_iNextFreeU;
  int              m_iNextFreeV;
  CSimulNode      *m_Leaves[SIMULTREE_NODE_POOL_SIZE_U*SIMULTREE_NODE_POOL_SIZE_V];

  // textures
  GLuint           m_Densities;
  GLuint           m_LeaveCenters;
  GLuint           m_NeighboursTex[SIMULTREE_AVAILABLE_TEX];
  pix_type        *m_NeighboursTexData[SIMULTREE_AVAILABLE_TEX];
  // flow graph
  short            m_Neighbours[27][2*SIMULTREE_NODE_POOL_SIZE_U*SIMULTREE_NODE_POOL_SIZE_V];
  CSimulNode      *m_NeighboursPtr[27][SIMULTREE_NODE_POOL_SIZE_U*SIMULTREE_NODE_POOL_SIZE_V];
  vector<cell_nfo *> m_Sources[SIMULTREE_NODE_POOL_SIZE_U*SIMULTREE_NODE_POOL_SIZE_V];
  vector<cell_nfo *> m_Dests[SIMULTREE_NODE_POOL_SIZE_U*SIMULTREE_NODE_POOL_SIZE_V];
  map<pair<pair<int,int>,int>,CSimulNode *> m_IJK2Neighbours;
  // program for simulation
  CGprogram       m_cgFragmentProg;
  CGparameter     m_cgDensityIn;
  CGparameter     m_cgN[SIMULTREE_AVAILABLE_TEX];

  CGprogram       m_cgFragmentProgUpd;
  CGparameter     m_cgUpdDensity;
  CGparameter     m_cgUpdCenters;
  CGparameter     m_cgUpdCoords;
  CGparameter     m_cgUpdValue;
  CGparameter     m_cgUpdRadius;

  PBuffer        *m_PBuffer;

  void           allocTextures();
  void           initDensityTex();
  void           loadCg();
  void           buildNeighboursTable();
  void           buildNeighboursTable_aux(int ni,int nj,int nk);
  void           buildNeighboursTex();
  unsigned char *buildNeighboursTex_aux(int ni,int nj,int nk);
  CSimulNode    *getNode(int i,int j,int k,int d);

public:

  CSimulTree(std::list<CPolygon>& polys,const char *fp_prg);
  ~CSimulTree();

  void commit();
  void report();

  void bind()   {m_HrdwTree->bind(m_Densities);}
  void unbind() {m_HrdwTree->unbind();}

  void setCgTransform()       {m_HrdwTree->setCgTransform();}

  void registerNode(CSimulNode *n,
		    unsigned int i,unsigned int j,unsigned int k,int d,
		    short& _u,short& _v);

  void simulstep();
  void addDensity(CVertex p,double v,double r);
  void clear();

  void draw_nodes(int u,int v);

  GLuint getDensity() const {return (m_Densities);}

  CHrdwTree      *getHrdwTree() {return (m_HrdwTree);}
  //CSimulNode     *getRoot()     {return (m_Root);}
  void            draw_structure(bool b);
  CVertex         getLeafCenter(int i,int j) const 
    {return (m_Leaves[i+j*SIMULTREE_NODE_POOL_SIZE_U]->getCenter());}
};

//--------------------------------------------------------

#endif

//--------------------------------------------------------
