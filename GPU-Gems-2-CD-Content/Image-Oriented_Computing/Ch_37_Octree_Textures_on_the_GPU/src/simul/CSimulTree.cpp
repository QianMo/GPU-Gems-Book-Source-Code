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

#include <math.h>

#include "CSimulTree.h"
#include "common.h"
#include "CTexture.h"
#include "cg_load.h"
#include "CProfiler.h"
#include "noise.h"

//--------------------------------------------------------

#include <iostream>
#include <set>
#include <list>
#include <vector>
#include <algorithm>

using namespace std;

//--------------------------------------------------------

#include <GL/gl.h>
#include <GL/glu.h>
#include <glux.h>

GLUX_REQUIRE(GL_EXT_texture3D);

//--------------------------------------------------------

#ifdef SIMULTREE_8BIT
#  define TEXTURE_TARGET          GL_TEXTURE_2D
#  define TEXTURE_INTERNAL_RGB    GL_RGB
#  define TEXTURE_INTERNAL_RGBA   GL_RGBA
#  define TEXTURE_EXTERNAL        GL_UNSIGNED_BYTE
#  define TEXTURE_FLOAT(v)        (unsigned char)(max(min((v)*255.0,255),0))
#else
#  define TEXTURE_TARGET          GL_TEXTURE_RECTANGLE_NV
#  define TEXTURE_INTERNAL_RGB    GL_FLOAT_RGB16_NV
#  define TEXTURE_INTERNAL_RGBA   GL_FLOAT_RGBA16_NV
#  define TEXTURE_EXTERNAL        GL_FLOAT
#  define TEXTURE_FLOAT(v)        (v)
#endif

//--------------------------------------------------------

MyNoise      g_Noise;

//--------------------------------------------------------

CSimulTree::CSimulTree(std::list<CPolygon>& polys,
                       const char *fp_prg)
{
  cerr << "sizeof(CSimulTree) = " << sizeof(CSimulTree)/(1024*1024) << " Mb" << endl;
  cerr << "sizeof(CSimulNode) = " << sizeof(CSimulNode) << " bytes" << endl;

  // indir stack texture  
  CHECK_GLERROR("CSimulTree::CSimulTree - (before)");

  m_iNextFreeU=0;
  m_iNextFreeV=0;
  memset(m_Leaves,0,sizeof(CSimulNode *)*SIMULTREE_NODE_POOL_SIZE_U*SIMULTREE_NODE_POOL_SIZE_V);
  // hardware tree
  m_HrdwTree=new CHrdwSimulTree(SIMUL_NODE_SIZE,
    128,
    128,
    128,
    fp_prg);
  // init pbuffer
#ifdef SIMULTREE_8BIT
  m_PBuffer=new PBuffer("rgb alpha");
#else
  m_PBuffer=new PBuffer("float=16 alpha");
#endif
  m_PBuffer->Initialize(SIMULTREE_NODE_POOL_SIZE_U,SIMULTREE_NODE_POOL_SIZE_V,false,true);

  // create list of pointers
  std::list<CPolygon *> ptr_lst;
  for (std::list<CPolygon>::iterator P=polys.begin();P!=polys.end();P++)
    ptr_lst.push_back(&(*P));

  // root
  m_Root=new CSimulNode(this,
    ptr_lst,
    m_HrdwTree->getRoot(),
    0,
    0,0,0,
    COBox());
  delete (m_Root);
  m_Root=NULL;

  buildNeighboursTable();
  double start=PROFILER.getTime();
  buildNeighboursTex();
  cerr << "buildNeighboursTex: " << (int)(PROFILER.getTime()-start) << " ms." << endl;
  allocTextures();

  loadCg();

  initDensityTex();

  cerr << "NextFreeV = " << m_iNextFreeV << endl;

  // commit tree
  m_HrdwTree->commit();

}

//--------------------------------------------------------

CSimulTree::~CSimulTree()
{
  delete (m_HrdwTree);
  delete (m_PBuffer);
}

//--------------------------------------------------------

void CSimulTree::loadCg()
{
  // =============================================
  // load fragment program
  m_cgFragmentProg=cg_loadFragmentProgram("simul/fp_simul.cg");
  m_cgDensityIn=cgGetNamedParameter(m_cgFragmentProg,"DensityIn");
  for (int t=0;t<SIMULTREE_AVAILABLE_TEX;t++)
  {
    static char str[4];
    sprintf(str,"N%d",t);
    m_cgN[t]=cgGetNamedParameter(m_cgFragmentProg,str);
  }
  for (int t=0;t<SIMULTREE_AVAILABLE_TEX;t++)
    cgGLSetTextureParameter(m_cgN[t],m_NeighboursTex[t]);
  cgGLSetTextureParameter(m_cgDensityIn,m_Densities);

  // =============================================
  // load fragment program for updates
  m_cgFragmentProgUpd=cg_loadFragmentProgram("simul/fp_simul_upd.cg");
  m_cgUpdDensity=cgGetNamedParameter(m_cgFragmentProgUpd,"UpdDensity");
  cgGLSetTextureParameter(m_cgUpdDensity,m_Densities);
  m_cgUpdCenters=cgGetNamedParameter(m_cgFragmentProgUpd,"UpdCenters");
  cgGLSetTextureParameter(m_cgUpdCenters,m_LeaveCenters);
  m_cgUpdCoords=cgGetNamedParameter(m_cgFragmentProgUpd,"UpdCoords");
  m_cgUpdValue=cgGetNamedParameter(m_cgFragmentProgUpd,"UpdValue");
  m_cgUpdRadius=cgGetNamedParameter(m_cgFragmentProgUpd,"UpdRadius");

}

//--------------------------------------------------------

void CSimulTree::allocTextures()
{
  pix_type *data;

  // neighbours tex
  glGenTextures(SIMULTREE_AVAILABLE_TEX,m_NeighboursTex);
  for (int t=0;t<SIMULTREE_AVAILABLE_TEX;t++)
  {
    data=m_NeighboursTexData[t];
    glBindTexture(TEXTURE_TARGET,m_NeighboursTex[t]);
    glTexImage2D(TEXTURE_TARGET,0,
      TEXTURE_INTERNAL_RGBA,
      SIMULTREE_NODE_POOL_SIZE_U,SIMULTREE_NODE_POOL_SIZE_V,0,
      GL_RGBA,TEXTURE_EXTERNAL,data);
    glTexParameteri(TEXTURE_TARGET,
      GL_TEXTURE_MAG_FILTER,
      GL_NEAREST);
    glTexParameteri(TEXTURE_TARGET,
      GL_TEXTURE_MIN_FILTER,
      GL_NEAREST);
    delete [](data);
    m_NeighboursTexData[t]=NULL;
  }

  // density tex
  glGenTextures(1,&m_Densities);
  glBindTexture(TEXTURE_TARGET,m_Densities);
  glTexImage2D(TEXTURE_TARGET,0,
    TEXTURE_INTERNAL_RGBA,
    SIMULTREE_NODE_POOL_SIZE_U,
    SIMULTREE_NODE_POOL_SIZE_V,0,
    GL_RGBA,TEXTURE_EXTERNAL,NULL);
  glTexParameteri(TEXTURE_TARGET,
    GL_TEXTURE_MAG_FILTER,
    GL_NEAREST);
  glTexParameteri(TEXTURE_TARGET,
    GL_TEXTURE_MIN_FILTER,
    GL_NEAREST);

  // leave centers
  glGenTextures(1,&m_LeaveCenters);
  glBindTexture(TEXTURE_TARGET,m_LeaveCenters);
  pix_type *dbldata=new pix_type[SIMULTREE_NODE_POOL_SIZE_U
    *SIMULTREE_NODE_POOL_SIZE_V
    *3];
  memset(dbldata,0,
    SIMULTREE_NODE_POOL_SIZE_U
    *SIMULTREE_NODE_POOL_SIZE_V
    *3
    *sizeof(pix_type));
  // store centers
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      if (m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U] != NULL)
      {
        CVertex c=m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U]->getCenter();
        dbldata[(u+v*SIMULTREE_NODE_POOL_SIZE_U)*3  ]=TEXTURE_FLOAT(c.x());
        dbldata[(u+v*SIMULTREE_NODE_POOL_SIZE_U)*3+1]=TEXTURE_FLOAT(c.y());
        dbldata[(u+v*SIMULTREE_NODE_POOL_SIZE_U)*3+2]=TEXTURE_FLOAT(c.z());
      }
    }
  }
  glTexImage2D(TEXTURE_TARGET,0,
    TEXTURE_INTERNAL_RGB,
    SIMULTREE_NODE_POOL_SIZE_U,
    SIMULTREE_NODE_POOL_SIZE_V,0,
    GL_RGB,TEXTURE_EXTERNAL,dbldata);
  delete [](dbldata);
  glTexParameteri(TEXTURE_TARGET,
    GL_TEXTURE_MAG_FILTER,
    GL_NEAREST);
  glTexParameteri(TEXTURE_TARGET,
    GL_TEXTURE_MIN_FILTER,
    GL_NEAREST);

}

//--------------------------------------------------------

void CSimulTree::addDensity(CVertex p,double v,double r)
{
  m_PBuffer->Activate();

  glViewport(0,0,SIMULTREE_NODE_POOL_SIZE_U,SIMULTREE_NODE_POOL_SIZE_V);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0,1.0,1.0,0.0);
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();  

  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);
  glColor3d(1,1,1);

  cgGLSetParameter3f(m_cgUpdCoords,p.x(),p.y(),p.z());
  cgGLSetParameter1f(m_cgUpdValue,(float)v);
  cgGLSetParameter1f(m_cgUpdRadius,max((float)r,1.0f/(float)pow((double)SIMUL_NODE_SIZE,
    (double)SIMULTREE_MAX_DEPTH)));

  cgGLEnableProfile(g_cgFragmentProfile);
  cgGLBindProgram(m_cgFragmentProgUpd);
  cgGLEnableTextureParameter(m_cgUpdDensity);
  cgGLEnableTextureParameter(m_cgUpdCenters);

  glBegin(GL_QUADS);
  glTexCoord2d(0,1);
  glVertex2i(0,0);
  glTexCoord2d(0,0);
  glVertex2i(0,1);
  glTexCoord2d(1,0);
  glVertex2i(1,1);
  glTexCoord2d(1,1);
  glVertex2i(1,0);
  glEnd();

  cgGLDisableTextureParameter(m_cgUpdDensity);
  cgGLDisableTextureParameter(m_cgUpdCenters);

  cgGLDisableProfile(g_cgFragmentProfile);

  // copy to texture
  glBindTexture(TEXTURE_TARGET,m_Densities);
  /*
  glCopyTexImage2D(TEXTURE_TARGET,0,
  TEXTURE_INTERNAL_RGBA,
  0,0,SIMULTREE_NODE_POOL_SIZE,SIMULTREE_NODE_POOL_SIZE,
  0);
  */
  glCopyTexSubImage2D(TEXTURE_TARGET,0,0,0,0,0,
    SIMULTREE_NODE_POOL_SIZE_U,
    SIMULTREE_NODE_POOL_SIZE_V);

  m_PBuffer->Deactivate();

}

//--------------------------------------------------------

void CSimulTree::clear()
{
  // clear texture
  glFinish();
  initDensityTex();
}

//--------------------------------------------------------

void CSimulTree::simulstep()
{
  int vp[4];

  m_PBuffer->Activate();

  glGetIntegerv(GL_VIEWPORT,vp);
  glViewport(0,0,SIMULTREE_NODE_POOL_SIZE_U,SIMULTREE_NODE_POOL_SIZE_V);
  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0,1.0,1.0,0.0);
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();  

  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);
  glColor3d(1,1,1);

  cgGLEnableTextureParameter(m_cgDensityIn);
  for (int t=0;t<SIMULTREE_AVAILABLE_TEX;t++)
    cgGLEnableTextureParameter(m_cgN[t]);

  cgGLEnableProfile(g_cgFragmentProfile);
  cgGLBindProgram(m_cgFragmentProg);  

  // render only relevant part
  double h=(m_iNextFreeV)/(double)(SIMULTREE_NODE_POOL_SIZE_V);

  glBegin(GL_QUADS);
  glTexCoord2d(0,h);
  glVertex2d(0,  1.0-h);

  glTexCoord2d(0,0.0);
  glVertex2d(0,  1.0);

  glTexCoord2d(1,0.0);
  glVertex2d(1,  1.0);

  glTexCoord2d(1,h);
  glVertex2d(1,  1.0-h);
  glEnd();

  cgGLDisableTextureParameter(m_cgDensityIn);
  for (int t=0;t<SIMULTREE_AVAILABLE_TEX;t++)
    cgGLDisableTextureParameter(m_cgN[t]);

  cgGLDisableProfile(g_cgFragmentProfile);

  // copy to texture
  glBindTexture(TEXTURE_TARGET,m_Densities);

  glCopyTexSubImage2D(TEXTURE_TARGET,0,0,0,
    0,0,
    SIMULTREE_NODE_POOL_SIZE_U,
    //		      SIMULTREE_NODE_POOL_SIZE);		      
    m_iNextFreeV);

  /*
  glCopyTexImage2D(TEXTURE_TARGET,0,
  TEXTURE_INTERNAL_RGB,
  0,0,SIMULTREE_NODE_POOL_SIZE,SIMULTREE_NODE_POOL_SIZE,
  0);
  */
  //    glViewport(vp[0],vp[1],vp[2],vp[3]);
  m_PBuffer->Deactivate();

}

//--------------------------------------------------------

void CSimulTree::buildNeighboursTable()
{
  for (int i=0;i<3;i++)
    for (int j=0;j<3;j++)
      for (int k=0;k<3;k++)
        buildNeighboursTable_aux(i,j,k);
}

//--------------------------------------------------------

CSimulNode *CSimulTree::getNode(int i,int j,int k,int d)
{
  pair<pair<int,int>,int> key;
  key.first.first =i;
  key.first.second=j;
  key.second      =k;

  map<pair<pair<int,int>,int>,CSimulNode *>::iterator F=m_IJK2Neighbours.find(key);
  if (F!=m_IJK2Neighbours.end())
    return ((*F).second);
  else
    return (NULL);
  //  return (m_Root->getNode(i-1,j-1,k-1,d));
}

//--------------------------------------------------------

void CSimulTree::buildNeighboursTable_aux(int ni,int nj,int nk)
{
  memset(&(m_Neighbours[CELL(ni,nj,nk)][0]),0,
    SIMULTREE_NODE_POOL_SIZE_U
    *SIMULTREE_NODE_POOL_SIZE_V
    *2*sizeof(short));
  memset(&(m_NeighboursPtr[CELL(ni,nj,nk)][0]),0,
    SIMULTREE_NODE_POOL_SIZE_U
    *SIMULTREE_NODE_POOL_SIZE_V
    *sizeof(CSimulNode *));

  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      // get child
      CSimulNode *child=m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U];

      // if child exists
      if (child != NULL)
      {
        // get neigbour
        CSimulNode *neigh=NULL;

        int i=child->getI();
        int j=child->getJ();
        int k=child->getK();
        int d=child->getDepth();

        switch (CELL(ni,nj,nk))
        {
        case CELL(0,0,0) : neigh=getNode(i-1,j-1,k-1,d); break;
        case CELL(0,0,1) : neigh=getNode(i-1,j-1,k  ,d); break;
        case CELL(0,0,2) : neigh=getNode(i-1,j-1,k+1,d); break;
        case CELL(0,1,0) : neigh=getNode(i-1,j  ,k-1,d); break;
        case CELL(0,1,1) : neigh=getNode(i-1,j  ,k  ,d); break;
        case CELL(0,1,2) : neigh=getNode(i-1,j  ,k+1,d); break;
        case CELL(0,2,0) : neigh=getNode(i-1,j+1,k-1,d); break;
        case CELL(0,2,1) : neigh=getNode(i-1,j+1,k  ,d); break;
        case CELL(0,2,2) : neigh=getNode(i-1,j+1,k+1,d); break;
        case CELL(1,0,0) : neigh=getNode(i  ,j-1,k-1,d); break;
        case CELL(1,0,1) : neigh=getNode(i  ,j-1,k  ,d); break;
        case CELL(1,0,2) : neigh=getNode(i  ,j-1,k+1,d); break;
        case CELL(1,1,0) : neigh=getNode(i  ,j  ,k-1,d); break;
        case CELL(1,1,1) : neigh=NULL; break;//neigh=getNode(i  ,j  ,k  ,d); break;
        case CELL(1,1,2) : neigh=getNode(i  ,j  ,k+1,d); break;
        case CELL(1,2,0) : neigh=getNode(i  ,j+1,k-1,d); break;
        case CELL(1,2,1) : neigh=getNode(i  ,j+1,k  ,d); break;
        case CELL(1,2,2) : neigh=getNode(i  ,j+1,k+1,d); break;
        case CELL(2,0,0) : neigh=getNode(i+1,j-1,k-1,d); break;
        case CELL(2,0,1) : neigh=getNode(i+1,j-1,k  ,d); break;
        case CELL(2,0,2) : neigh=getNode(i+1,j-1,k+1,d); break;
        case CELL(2,1,0) : neigh=getNode(i+1,j  ,k-1,d); break;
        case CELL(2,1,1) : neigh=getNode(i+1,j  ,k  ,d); break;
        case CELL(2,1,2) : neigh=getNode(i+1,j  ,k+1,d); break;
        case CELL(2,2,0) : neigh=getNode(i+1,j+1,k-1,d); break;
        case CELL(2,2,1) : neigh=getNode(i+1,j+1,k  ,d); break;
        case CELL(2,2,2) : neigh=getNode(i+1,j+1,k+1,d); break;
        default:
          throw CCoreException("CSimulTree::buildNeighboursTable - wrong parameter (%d,%d,%d) !",ni,nj,nk);
        }
        m_NeighboursPtr[CELL(ni,nj,nk)][u+v*SIMULTREE_NODE_POOL_SIZE_U]=neigh;
        if (neigh != NULL)
        {
          int nu=neigh->getU();
          int nv=neigh->getV();
          // store neighbour index
          m_Neighbours[CELL(ni,nj,nk)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2  ]=nu;
          m_Neighbours[CELL(ni,nj,nk)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2+1]=nv;
        }
        else
        {
          m_Neighbours[CELL(ni,nj,nk)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2  ]=255;
          m_Neighbours[CELL(ni,nj,nk)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2+1]=255;
        }
      }
      else
      {
        m_Neighbours[CELL(ni,nj,nk)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2  ]=255;
        m_Neighbours[CELL(ni,nj,nk)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2+1]=255;
      }
    }
  }
}

//--------------------------------------------------------

void randomize(vector<cell_nfo *>& v)
{
  for (int i=0;i<(int)v.size();i++)
  {
    int j=rand() % v.size();
    cell_nfo *tmp=v[i];
    v[i]=v[j];
    v[j]=tmp;
  }
}

//--------------------------------------------------------

void CSimulTree::buildNeighboursTex()
{
  static const CVertex z(0,-1,0);

  // alloc and clear all neighbours textures
  for (int i=0;i<SIMULTREE_AVAILABLE_TEX;i++)
  {
    m_NeighboursTexData[i]=new pix_type[SIMULTREE_NODE_POOL_SIZE_U
      *SIMULTREE_NODE_POOL_SIZE_V
      *4];
    memset(m_NeighboursTexData[i],0,
      SIMULTREE_NODE_POOL_SIZE_U
      *SIMULTREE_NODE_POOL_SIZE_V
      *4*sizeof(pix_type));
  }

  // for each child in latest level
  int nbn_min=27,nbn_max=0;
  for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
  {
    for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
    {
      // get child
      CSimulNode *leaf=m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U];
      //cerr << "---" << endl;
      if (leaf != NULL)
      {
        int nbn=0;
        double minval=99999.0;
        double maxval=0.0;
        // distrubute content over neighbours
        for (int i=0;i<3;i++)
        {
          for (int j=0;j<3;j++)
          {
            for (int k=0;k<3;k++)
            {
              // present ?
              CSimulNode *neigh=m_NeighboursPtr[CELL(i,j,k)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)];
              if (neigh != NULL)// && neigh != leaf)
              {
                nbn++;

                cell_nfo *nfo=new cell_nfo;

                // read from this leaf ...
                nfo->read_u=u;
                nfo->read_v=v;
                nfo->read_node=leaf;
                // ... and distrbute over neighbours
                nfo->write_u=m_Neighbours[CELL(i,j,k)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2  ];
                nfo->write_v=m_Neighbours[CELL(i,j,k)][(u+v*SIMULTREE_NODE_POOL_SIZE_U)*2+1];
                nfo->write_node=neigh;
                // compute value
                double val=0.0;

                // distance between centers
                CVertex c=leaf->getCenter();
                CVertex p=neigh->getCenter();
                CVertex d=(p-c);
                if (d.norme() > CVertex::getEpsilon())
                  d.normalize();

                CVertex opt=(leaf->getNormal().cross(z)).cross(leaf->getNormal());

                //val=4.0+3.0*neigh->getNormal().dot(leaf->getNormal());
                val=4.0+3.0*(0.5+0.5*neigh->getNormal().dot(leaf->getNormal()))*d.dot(opt); // for following z

                //cerr << '(' << i << ',' << j << ',' << k << ')' <<  val << endl;
                nfo->value=val;
                nfo->used=true;
                if (val > 0.0)
                {
                  minval=min(val,minval);
                  maxval=max(val,maxval);
                  // destinations are neighbours
                  m_Dests[nfo->write_u+nfo->write_v*SIMULTREE_NODE_POOL_SIZE_U].push_back(nfo);
                  // current leaf is read
                  m_Sources[nfo->read_u+nfo->read_v*SIMULTREE_NODE_POOL_SIZE_U].push_back(nfo);
                }
                else
                  delete (nfo);
              } // neigh != NULL
            } // k
          } // j
        } // i
        if (nbn > nbn_max)
          nbn_max=nbn;
        if (nbn < nbn_min)
          nbn_min=nbn;

        // enhance gap between values
        for (vector<cell_nfo *>::iterator C=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].begin();
          C!=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].end();C++)
        {
          if (maxval - minval > 0)
          {
            double v=((*C)->value - minval)/(maxval-minval);
            (*C)->value=pow(v,3.0);
            //cerr << (*C)->value << endl;
          }
        }

      } // leaf != NULL
    } // u
  } // v
  cerr << "min/max number of neighbours: " << nbn_min << " / " << nbn_max << endl;

# define LIMIT SIMULTREE_AVAILABLE_TEX

  // randomize m_Dests and m_Sources
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      randomize(m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U]);
      randomize(m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U]);
    }
  }

  // parse m_Dests and keep only the available number of neighbours
  for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
  {
    for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
    {
      if (m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].size() > LIMIT)
      {
        // sort by value
        sort(m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].begin(),
          m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].end(),
          cell_nfo::cmp());
        // suppress lower values ...
        //double low=m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE][LIMIT]->value;
        for (int n=LIMIT;n<(int)m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].size();n++)
        {
          //cerr << m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE][n]->value << endl;
          //  if (m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE][n]->value <= low)
          m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U][n]->used=false;
        }
      }
    }
  }

  // normalize outputs (m_Sources)
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      double l=0.0;
      for (vector<cell_nfo *>::iterator C=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].begin();
        C!=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].end();C++)
      {
        if ((*C)->used)
          l+=(*C)->value;
      }
      for (vector<cell_nfo *>::iterator C=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].begin();
        C!=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].end();C++)
      {
        if ((*C)->used)
          (*C)->value=(*C)->value/l;
      }
    }
  }

  // update texture data
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      int n=0;
      for (vector<cell_nfo *>::iterator C=m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].begin();
        C!=m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].end()
        && n < LIMIT
        ;C++)
      {
        if ((*C)->used)
        {
          m_NeighboursTexData[n][((*C)->write_u+(*C)->write_v*SIMULTREE_NODE_POOL_SIZE_U)*4  ]
          =(pix_type)(*C)->read_u;
          m_NeighboursTexData[n][((*C)->write_u+(*C)->write_v*SIMULTREE_NODE_POOL_SIZE_U)*4+1]
          =(pix_type)(*C)->read_v;
          m_NeighboursTexData[n][((*C)->write_u+(*C)->write_v*SIMULTREE_NODE_POOL_SIZE_U)*4+2]
          =(pix_type)TEXTURE_FLOAT((*C)->value);
          m_NeighboursTexData[n][((*C)->write_u+(*C)->write_v*SIMULTREE_NODE_POOL_SIZE_U)*4+3]
          =(pix_type)TEXTURE_FLOAT(1.0);
          n++;
        }
      }
    }
  }
}

//--------------------------------------------------------

void CSimulTree::registerNode(CSimulNode *n,
                              unsigned int i,unsigned int j,unsigned int k,int d,
                              short& _u,short& _v)
{
  if (m_iNextFreeV < SIMULTREE_NODE_POOL_SIZE_V)
  {
    _u=m_iNextFreeU;
    _v=m_iNextFreeV;
    m_Leaves[_u+_v*SIMULTREE_NODE_POOL_SIZE_U]=n;
    pair<pair<int,int>,int> key;
    key.first.first =n->getI();
    key.first.second=n->getJ();
    key.second      =n->getK();
    m_IJK2Neighbours[key]=n;
    if (m_iNextFreeU < SIMULTREE_NODE_POOL_SIZE_U-2) // -1 produces wrong neighbors
    {
      m_iNextFreeU++;
    }
    else
    {
      m_iNextFreeU=0;
      m_iNextFreeV++;
    }
  }
  else
    throw CCoreException("CSimulTree::registerNode - out of memory ! (%d,%d)",m_iNextFreeU,m_iNextFreeV);
}

//--------------------------------------------------------

void CSimulTree::initDensityTex()
{
  static const CVertex z(0,-1,0);

  pix_type *dbldata=new pix_type[SIMULTREE_NODE_POOL_SIZE_U
    *SIMULTREE_NODE_POOL_SIZE_V
    *4];
  memset(dbldata,0,
    SIMULTREE_NODE_POOL_SIZE_U
    *SIMULTREE_NODE_POOL_SIZE_V
    *4
    *sizeof(pix_type));

  // compute min and max area
  double area_min=0.0,area_max=0.0;
  bool area_init=false;
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      CSimulNode *l=m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U];
      if (l != NULL)
      {
        if (!area_init)
        {
          area_init=true;
          area_min=area_max=l->getArea();
        }
        else
        {
          if (l->getArea() > area_max) area_max=l->getArea();
          if (l->getArea() < area_min) area_min=l->getArea();
        }
      }
    }
  }
  cerr << area_min << '/' << area_max << endl;
  // compute capacities
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      // compute leaf capacity
      double capacity=0.0;
      double threshold=0.0;
      if (m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U] != NULL)
      {
        int nb=0;
        for (int i=0;i<(int)m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U].size();i++)
          if (m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE_U][i]->used)
            nb++;
        if (nb > 0)
        {
          // area threshold
          //double a=(m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U]->getArea())/(area_max);
          // random
          //double rnd=1.0;//0.2+0.8*drand48();
          // capacity
          capacity=0.5+0.5*(m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U]->getNormal().cross(z)).norme();
          //a; 
          //*((1.0-rnd)+rnd*(m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U]->getNormal().cross(z)).norme());
          threshold=0.8*
            fabs(g_Noise.GetNoiseAt(8.0*m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U]->getCenter()));
        }
      }
      // store in density table G channel
      dbldata[(u+v*SIMULTREE_NODE_POOL_SIZE_U)*4+1]=(pix_type)TEXTURE_FLOAT(capacity);
      dbldata[(u+v*SIMULTREE_NODE_POOL_SIZE_U)*4+3]=(pix_type)TEXTURE_FLOAT(threshold);
    }
  }

  glBindTexture(TEXTURE_TARGET,m_Densities);
  glTexImage2D(TEXTURE_TARGET,0,
    TEXTURE_INTERNAL_RGBA,
    SIMULTREE_NODE_POOL_SIZE_U,
    SIMULTREE_NODE_POOL_SIZE_V,0,
    GL_RGBA,TEXTURE_EXTERNAL,dbldata);
  delete [](dbldata);
}

//--------------------------------------------------------

void CSimulTree::draw_nodes(int u,int v)
{
  glColor4d(0,0,1,0.2);
  m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U]->draw_node();
  /*
  for (int i=0;i<27;i++)
  {
  glColor4d(0,1,0,0.4);
  CSimulNode *n=m_NeighboursPtr[i][u+v*SIMULTREE_NODE_POOL_SIZE];
  if (n != NULL)
  n->draw_node();  
  }
  */

  for (int j=0;j<(int)m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U].size();j++)
  {
    CSimulNode *n=m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U][j]->read_node;
    if (n != NULL)
    {
      if (m_Dests[u+v*SIMULTREE_NODE_POOL_SIZE_U][j]->used)
        glColor4d(0,1,0,0.8);
      else
        glColor4d(1,0,0,0.5);	    
      n->draw_node();
    }
    else
      throw CCoreException("draw_nodes");
  }

  /*
  for (int j=0;j<m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE].size();j++)
  {
  CSimulNode *n=m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE][j]->write_node;
  if (n != NULL)
  {
  if (m_Sources[u+v*SIMULTREE_NODE_POOL_SIZE][j]->used)
  glColor4d(0,1,0,0.8);
  else
  glColor4d(1,0,0,0.5);	    
  n->draw_node();
  }
  }
  */
}

//--------------------------------------------------------

void CSimulTree::commit()
{
  m_HrdwTree->commit();
}

//--------------------------------------------------------

void CSimulTree::draw_structure(bool b)
{
  for (int u=0;u<SIMULTREE_NODE_POOL_SIZE_U;u++)
  {
    for (int v=0;v<SIMULTREE_NODE_POOL_SIZE_V;v++)
    {
      CSimulNode *l=m_Leaves[u+v*SIMULTREE_NODE_POOL_SIZE_U];
      if (l != NULL)
        l->draw_structure(b);
    }
  }
}

//--------------------------------------------------------

void CSimulTree::report()
{
  int d=m_HrdwTree->depth();

  cerr << endl;
  cerr <<   "  Node size             = " << SIMUL_NODE_SIZE << endl;
  cerr <<   "  Indirection pool size = " 
    << m_HrdwTree->nbGrids(0) << 'x' 
    << m_HrdwTree->nbGrids(1) << 'x' 
    << m_HrdwTree->nbGrids(2)
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
