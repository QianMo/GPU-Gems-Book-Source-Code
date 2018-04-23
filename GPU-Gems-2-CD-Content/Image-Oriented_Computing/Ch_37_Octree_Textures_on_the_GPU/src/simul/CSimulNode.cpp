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

#include <cmath>

#include "CSimulNode.h"
#include "CSimulTree.h"
#include "common.h"
#include "noise.h"
#include "CTexture.h"

#include <set>
#include <algorithm>

//--------------------------------------------------------

CSimulNode::CSimulNode(CSimulTree *tree,
                       const std::list<CPolygon *>& polys,
                       CHrdwTree::hrdw_node *twin,
                       short depth,
                       short ni,
                       short nj,
                       short nk,
                       const COBox& b)
{
  m_Tree=tree;
  m_HrdwTwin=twin;
  m_iDepth=(unsigned char)depth;
  m_uiI=ni;
  m_uiJ=nj;
  m_uiK=nk;
  m_Center=b.center();
  m_bIsLeaf=false;
  m_iU=-1;
  m_iV=-1;
  CVertex local_size(1.0/SIMUL_NODE_SIZE,
    1.0/SIMUL_NODE_SIZE,
    1.0/SIMUL_NODE_SIZE);
  CVertex nrm;
  float area;
  for (int i=0;i<SIMUL_NODE_SIZE;i++)
  {
    float di=(float)i/(float)(SIMUL_NODE_SIZE);
    for (int j=0;j<SIMUL_NODE_SIZE;j++)
    {
      float dj=(float)j/(float)(SIMUL_NODE_SIZE);
      for (int k=0;k<SIMUL_NODE_SIZE;k++)  
      {
        float                 dk=(float)k/(float)(SIMUL_NODE_SIZE);
        CVertex               local_pos(di,dj,dk);
        COBox                 box=b.sub(local_pos,local_size);
        std::list<CPolygon *> cutpolys;

        // cut polys
        nrm=CVertex(0,0,0);
        area=0.0;
        int nb=0;
        for (std::list<CPolygon *>::const_iterator P=polys.begin();
          P!=polys.end();P++)
        {
          CPolygon r;
          box.cut(*(*P),r);
          if (!r.empty())
          {
            nrm+=r.normal();
            area+=(float)r.area();
            nb++;
            cutpolys.push_back(*P);
          }
        }
        if (nrm.norme() > CVertex::getEpsilon())
          nrm.normalize();
        // create child
        CSimulNode *child=NULL;
        if (!cutpolys.empty() && m_iDepth < SIMULTREE_MAX_DEPTH-1)
        {
          // go down
          try
          {
            m_HrdwTwin->createChild(i,j,k);
            CHrdwTree::hrdw_node *hchild=m_HrdwTwin->getChild(i,j,k);
            if (m_iDepth > 7)
            {
              cutpolys.clear();
              child=new CSimulNode(tree,
                polys,
                hchild,
                m_iDepth+1,
                ni*SIMUL_NODE_SIZE+i,
                nj*SIMUL_NODE_SIZE+j,
                nk*SIMUL_NODE_SIZE+k,
                box);

            }
            else
            {
              child=new CSimulNode(tree,
                cutpolys,
                hchild,
                m_iDepth+1,
                ni*SIMUL_NODE_SIZE+i,
                nj*SIMUL_NODE_SIZE+j,
                nk*SIMUL_NODE_SIZE+k,
                box);
            }
            delete (child);
            child=NULL;
          }
          catch (...)
          {
            child=NULL;
            m_HrdwTwin->setLeafColor(i,j,k,
              255,
              255,
              255);
          }
        }
        else
        {
          // stop here
          if (m_iDepth == SIMULTREE_MAX_DEPTH-1 && !cutpolys.empty())
          {
            // this is a leaf
            child=new CSimulNode(tree,
              m_iDepth+1,
              ni*SIMUL_NODE_SIZE+i,
              nj*SIMUL_NODE_SIZE+j,
              nk*SIMUL_NODE_SIZE+k,
              box);
            child->m_Normal=nrm;
            child->m_fArea=area;
            unsigned char r=CPU_ENCODE_INDEX8_R(child->getU(),child->getV());
            unsigned char g=CPU_ENCODE_INDEX8_G(child->getU(),child->getV());
            unsigned char b=CPU_ENCODE_INDEX8_B(child->getU(),child->getV());
            m_HrdwTwin->setLeafColor(i,j,k,
              r,g,b);
          }
          else
          {
            // no more data (pending leaf)
            child=NULL;
            m_HrdwTwin->setLeafColor(i,j,k,
              255,
              255,
              255);
          }
        }
        // set child
        m_Childs[i
          +j*SIMUL_NODE_SIZE 
          +k*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE]=child;
      }
    }
  }
}

//--------------------------------------------------------

CSimulNode::CSimulNode(CSimulTree *tree,
                       short depth,
                       short ni,
                       short nj,
                       short nk,
                       const COBox& b)
{
  m_Tree=tree;
  m_HrdwTwin=NULL;
  m_iDepth=(unsigned char)depth;
  m_uiI=ni;
  m_uiJ=nj;
  m_uiK=nk;
  m_Center=b.center();
  m_bIsLeaf=true;
  m_iU=-1;
  m_iV=-1;
  memset(m_Childs,0,
    SIMUL_NODE_SIZE*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE*sizeof(CSimulNode*));
  m_Tree->registerNode(this,
    m_uiI,m_uiJ,m_uiK,m_iDepth,
    m_iU,m_iV);
  /*
  cerr << "(" << ni << "," << nj << "," << nk << ")" << " " 
  << m_iU << "," << m_iV << endl;
  */
}

//--------------------------------------------------------

CSimulNode::~CSimulNode()
{
  if (!m_bIsLeaf)
  {
    for (int i=0;i<SIMUL_NODE_SIZE;i++)
    {
      for (int j=0;j<SIMUL_NODE_SIZE;j++)
      {
        for (int k=0;k<SIMUL_NODE_SIZE;k++)  
        {
          CSimulNode *child=m_Childs[i
            +j*SIMUL_NODE_SIZE
            +k*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE];
          if (child != NULL)
          {
            if (!child->m_bIsLeaf)
              delete (child);
          }
        }
      }
    }
  }
}

//--------------------------------------------------------

void CSimulNode::draw_structure(bool draw_empty)
{
  glColor3d(1,1,1);
  if (draw_empty)
  {
    CVertex u=CVertex(1,0,0)/(vertex_real)pow((double)SIMUL_NODE_SIZE,(double)m_iDepth);
    CVertex v=CVertex(0,1,0)/(vertex_real)pow((double)SIMUL_NODE_SIZE,(double)m_iDepth);
    CVertex w=CVertex(0,0,1)/(vertex_real)pow((double)SIMUL_NODE_SIZE,(double)m_iDepth);
    COBox box(m_Center-u*0.5-v*0.5-w*0.5,u,v,w);
    box.draw_box_line();
  }
  for (int i=0;i<SIMUL_NODE_SIZE;i++)
  {
    for (int j=0;j<SIMUL_NODE_SIZE;j++)
    {
      for (int k=0;k<SIMUL_NODE_SIZE;k++)
      {
        CSimulNode *child=m_Childs[i
          +j*SIMUL_NODE_SIZE
          +k*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE];

        if (child != NULL)
          child->draw_structure(draw_empty);
        else if (m_iDepth == SIMULTREE_MAX_DEPTH && !draw_empty)
        {
          glBegin(GL_LINES);
          glColor3d(1,0,0);
          getCenter().gl();
          glColor3d(0,0,1);
          (getCenter()+m_Normal*0.05f).gl();
          glEnd();
        }
      }
    }
  }
  glLineWidth(1.0);
}

//--------------------------------------------------------

void CSimulNode::draw_node()
{
  CVertex u=CVertex(1,0,0)/(vertex_real)pow((double)SIMUL_NODE_SIZE,(double)m_iDepth);
  CVertex v=CVertex(0,1,0)/(vertex_real)pow((double)SIMUL_NODE_SIZE,(double)m_iDepth);
  CVertex w=CVertex(0,0,1)/(vertex_real)pow((double)SIMUL_NODE_SIZE,(double)m_iDepth);
  COBox box(m_Center-u*0.5-v*0.5-w*0.5,u,v,w);

  box.draw_box_fill();
  glColor4d(1,1,1,1);
  box.draw_box_line();
}

//--------------------------------------------------------

CSimulNode *CSimulNode::getNode(int i,
                                int j,
                                int k,
                                int d)
{
  /*
  if (m_iDepth == 0)
  cerr << "------------------------" << endl;
  cerr << i << ',' << j << ',' << k << ' ' << d << " -> " 
  << m_uiI << ',' << m_uiJ << ',' << m_uiK << ' ' << m_iDepth << endl;
  */
  if (m_iDepth > d)
    throw CCoreException("CSimulNode::getNode - wrong node selection (depth) !");
  else if (m_iDepth == d)
  {
    if (i != m_uiI || j != m_uiJ || k != m_uiK || !m_bIsLeaf)
    {
      if (m_bIsLeaf)
        throw CCoreException("CSimulNode::getNode - wrong node selection (i,j,k) !");
      else
        throw CCoreException("CSimulNode::getNode - wrong node selection (not a leaf) !");
    }
    else
      return (this);
  }
  else
  {
    int delta=d-m_iDepth;
    int li=i;
    int lj=j;
    int lk=k;
    for (;delta>1;delta--)
    {
      li/=SIMUL_NODE_SIZE;
      lj/=SIMUL_NODE_SIZE;
      lk/=SIMUL_NODE_SIZE;
    }
    int ci=(li-(int)m_uiI*SIMUL_NODE_SIZE);
    int cj=(lj-(int)m_uiJ*SIMUL_NODE_SIZE);
    int ck=(lk-(int)m_uiK*SIMUL_NODE_SIZE);
    if (ci < 0 || ci >= SIMUL_NODE_SIZE)
      return (NULL);
    if (cj < 0 || cj >= SIMUL_NODE_SIZE)
      return (NULL);
    if (ck < 0 || ck >= SIMUL_NODE_SIZE)
      return (NULL);
    CSimulNode *child=m_Childs[ci
      +cj*SIMUL_NODE_SIZE
      +ck*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE];
    if (child != NULL)
      return (child->getNode(i,j,k,d));
    else
      return (NULL);
  }
}

//--------------------------------------------------------

int CSimulNode::depth()
{
  int l=1;

  if (m_bIsLeaf)
    return (0);
  else
  {
    for (int i=0;i<SIMUL_NODE_SIZE;i++)
    {
      for (int j=0;j<SIMUL_NODE_SIZE;j++)
      {
        for (int k=0;k<SIMUL_NODE_SIZE;k++)
        {
          CSimulNode *cell=m_Childs[i
            +j*SIMUL_NODE_SIZE
            +k*SIMUL_NODE_SIZE*SIMUL_NODE_SIZE];
          l=max(l,1+cell->depth());
        }
      }
    }
  }
  return (l);
}

//--------------------------------------------------------
