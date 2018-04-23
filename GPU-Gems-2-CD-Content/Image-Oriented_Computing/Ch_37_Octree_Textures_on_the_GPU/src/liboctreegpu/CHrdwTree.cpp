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
#ifdef WIN32
#  include <windows.h>
#endif

// -------------------------------------------------------- 

#include <GL/gl.h>
#include <Cg/cg.h>
#include "utils.h"

// -------------------------------------------------------- 

#include <glux.h>
GLUX_REQUIRE(GL_EXT_texture3D);

// -------------------------------------------------------- 

#include "CHrdwTree.h"
#include "cg_load.h"

// -------------------------------------------------------- 

#include <iostream>
using namespace std;

// -------------------------------------------------------- 

CHrdwTree::CHrdwTree(int boxres,
                     int NbGridsu,int NbGridsv,int NbGridsw,
                     int maxdepth,
                     const char *fp,
                     const char *vp)
                     : 
m_iGridRes(boxres),
m_iNextFreeI(0),
m_iNextFreeJ(0),
m_iNextFreeK(0),
m_iTreeMaxDepth(maxdepth)
{

  m_NbGrids[0]=NbGridsu;
  m_NbGrids[1]=NbGridsv;
  m_NbGrids[2]=NbGridsw;

  allocIndirPool();

  m_Root=allocNode(0,0,0,0);

  commit();

  loadPrograms(fp,vp);
}

// -------------------------------------------------------- 

CHrdwTree::~CHrdwTree() 
{
  glDeleteTextures(1,&m_uiIndirPool);
  delete (m_Root);
}

// -------------------------------------------------------- 

void CHrdwTree::loadPrograms(const char *fp,const char *vp)
{

  // =============================================
  // load vertex program
  if (glGetError())
    cerr << "CHrdwTree::CHrdwTree - GL Error (0)" << endl;

  m_cgVertexProg=cg_loadVertexProgram(vp);
  m_cgViewProj =cgGetNamedParameter(m_cgVertexProg,"ViewProj");
  m_cgView     =cgGetNamedParameter(m_cgVertexProg,"View");
  m_cgITView   =cgGetNamedParameter(m_cgVertexProg,"ITView");

  if (glGetError())
    cerr << "CHrdwTree::CHrdwTree - GL Error (1)" << endl;

  // =============================================
  // load fragment program

  char   *argv[5];
  char    strdepth[32];
  char   *str0="-DNV40";
  char   *str1="-unroll";
  char   *str2="none";

  sprintf(strdepth,"-DHRDWTREE_MAX_DEPTH=%d",m_iTreeMaxDepth);
  argv[0]=strdepth;
  argv[1]=NULL;
#ifdef CG_V1_2
  if (g_cgFragmentProfile == CG_PROFILE_FP40)
  {
    cerr << "[Cg] CHrwdTree: compiling with \"-DNV40 -unroll none\"" << endl;
    argv[1]=str0;
    argv[2]=str1;
    argv[3]=str2;
    argv[4]=NULL;
  }
#endif

  m_cgFragmentProg  =cg_loadFragmentProgram(fp,(const char **)argv);
  m_cgRefTex        =cgGetNamedParameter(m_cgFragmentProg,"RefTex");
  m_cgRefTexCellSize=cgGetNamedParameter(m_cgFragmentProg,"reftex_cell_size");
  m_cgBoxRes        =cgGetNamedParameter(m_cgFragmentProg,"boxres");
  m_cgTransform     =cgGetNamedParameter(m_cgFragmentProg,"transform");
  m_cgLevelCellSize =cgGetNamedParameter(m_cgFragmentProg,"level_cell_size");

  cgGLSetTextureParameter(m_cgRefTex,m_uiIndirPool);
  float cs[3];
  for (int k=0;k<3;k++)
    cs[k]=1.0f/((float)m_NbGrids[k]);
  cgGLSetParameter3fv(m_cgRefTexCellSize,cs);
  float bres=(float)m_iGridRes;
  cgGLSetParameter1fv(m_cgBoxRes,&bres);
  if (m_cgLevelCellSize != NULL)
  {
    float ni_exp_l=(float)m_iGridRes;
    for (int i=0;i<m_iTreeMaxDepth-1;i++)
      ni_exp_l*=m_iGridRes;
    float l[3]={1.0f/ni_exp_l,1.0f/ni_exp_l,1.0f/ni_exp_l};
    cgGLSetParameter3fv(m_cgLevelCellSize,l);
  }
  if (glGetError())
    cerr << "CHrdwTree::CHrdwTree - GL Error (2)" << endl;

  CGparameter p;

  // =============================================
  // load fragment program for leaf selection
  m_cgLeafSelectFragmentProg=cg_loadFragmentProgram("liboctreegpu/fp_color_tree.cg",(const char **)argv);

  m_cgLeafSelectRefTex=cgGetNamedParameter(m_cgLeafSelectFragmentProg,"RefTex");
  cgGLSetTextureParameter(m_cgLeafSelectRefTex,m_uiIndirPool);
  p=cgGetNamedParameter(m_cgLeafSelectFragmentProg,"reftex_cell_size");
  cgGLSetParameter3fv(p,cs);
  p=cgGetNamedParameter(m_cgLeafSelectFragmentProg,"boxres");
  cgGLSetParameter1fv(p,&bres);

  if (glGetError())
    cerr << "CHrdwTree::CHrdwTree - GL Error (3)" << endl;
}


// -------------------------------------------------------- 

/**
Allocate indirection pool

*/
void CHrdwTree::allocIndirPool()
{

  if (glGetError())
    cerr << "CHrdwTree::allocRefTex - GL Error (before)" << endl;

  glGenTextures(1,&m_uiIndirPool);
  glBindTexture(GL_TEXTURE_3D_EXT,m_uiIndirPool);

  unsigned char *data=new unsigned char[m_NbGrids[0]*m_iGridRes*m_NbGrids[1]*m_iGridRes*m_NbGrids[2]*m_iGridRes*4];
  memset(data,0,
    m_NbGrids[0]*m_iGridRes
    *m_NbGrids[1]*m_iGridRes
    *m_NbGrids[2]*m_iGridRes*4);
  int size=m_NbGrids[0]*m_iGridRes*m_NbGrids[1]*m_iGridRes*m_NbGrids[2]*m_iGridRes*4;
  cerr << "Allocating indirection pool: " << (int)size/(1024*1024) << " Mo" << endl;

  glTexImage3DEXT(GL_TEXTURE_3D_EXT,0,GL_RGBA,
    m_NbGrids[0]*m_iGridRes,
    m_NbGrids[1]*m_iGridRes,
    m_NbGrids[2]*m_iGridRes,
    0,GL_RGBA,GL_UNSIGNED_BYTE,data);
  delete [](data);

  glTexParameteri(GL_TEXTURE_3D_EXT,
    GL_TEXTURE_MAG_FILTER,
    GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D_EXT,
    GL_TEXTURE_MIN_FILTER,
    GL_NEAREST);

  if (glGetError())
    cerr << "CHrdwTree::allocRefTex - GL Error (after)" << endl;
}

// -------------------------------------------------------- 

CHrdwTree::hrdw_node *CHrdwTree::allocNode(int depth,
                                           int gi,
                                           int gj,
                                           int gk)
{
  /*
  cerr << "CHrdwTree::allocNode(" << depth
  << ',' << gi
  << ',' << gj
  << ',' << gk << ')' << endl;
  */
  if (m_iNextFreeK < m_NbGrids[2])
  {
    int ni=m_iNextFreeI;
    int nj=m_iNextFreeJ;
    int nk=m_iNextFreeK;
    if (m_iNextFreeI < m_NbGrids[0]-1)
      m_iNextFreeI++;
    else
    {
      if (m_iNextFreeJ < m_NbGrids[1]-1)
      {
        m_iNextFreeI=0;
        m_iNextFreeJ++;
      }
      else
      {
        m_iNextFreeI=0;
        m_iNextFreeJ=0;
        m_iNextFreeK++;
      }
    }
    /*
    cerr << " alloc node (level " << depth << ") - memory occupation = " 
    << (m_iNextFreeI
    +m_iNextFreeJ*m_NbGrids[0]
    +m_iNextFreeK*m_NbGrids[0]*m_NbGrids[1])
    *100/(m_NbGrids[0]*m_NbGrids[1]*m_NbGrids[2]) 
    << " (" << (m_iNextFreeI
    +m_iNextFreeJ*m_NbGrids[0]
    +m_iNextFreeK*m_NbGrids[0]*m_NbGrids[1]) << " blocks)"
    << endl;
    */
    // return next free box
    return (new CHrdwTree::hrdw_node(this,
      depth,
      gi,gj,gk,
      ni,nj,nk));
  }
  else
    return (allocNode_aux(depth,gi,gj,gk));
}

// -------------------------------------------------------- 

CHrdwTree::hrdw_node *CHrdwTree::allocNode_aux(int depth,
                                               int gi,
                                               int gj,
                                               int gk)
{
  if (m_ReleasedGrids.empty())
    throw CLibOctreeGPUException("CHrdwTree::allocBox_aux - out of memory !");
  // take first
  int bi=m_ReleasedGrids.front().bi;
  int bj=m_ReleasedGrids.front().bj;
  int bk=m_ReleasedGrids.front().bk;
  m_ReleasedGrids.pop_front();
  return (new CHrdwTree::hrdw_node(this,depth,
    gi,gj,gk,
    bi,bj,bk));
}

// -------------------------------------------------------- 

void CHrdwTree::releaseNode(hrdw_node **pp_box)
{
  free_box_nfo nfo;
  nfo.bi=(*pp_box)->bi();
  nfo.bj=(*pp_box)->bj();
  nfo.bk=(*pp_box)->bk();
  m_ReleasedGrids.push_back(nfo);

  delete (*pp_box);
  *pp_box=NULL;
}

// -------------------------------------------------------- 

void CHrdwTree::computeOffset(int bi,int bj,int bk, // index of the node within the indirection pool
                              int gi,int gj,int gk,             // index of the node within the level grid 
                              int gridsizeu,
                              int gridsizev,
                              int gridsizew,
                              unsigned char& _offset_u, // computed offset
                              unsigned char& _offset_v,
                              unsigned char& _offset_w)
{
  int     ti,tj,tk;
  int     di,dj,dk;

  gi %= gridsizeu;
  gj %= gridsizev;
  gk %= gridsizew;
  // compute offset
  ti  = gi % m_NbGrids[0];
  tj  = gj % m_NbGrids[1];
  tk  = gk % m_NbGrids[2];
  di  = (bi-ti);
  dj  = (bj-tj);
  dk  = (bk-tk);
  // write to grid
  _offset_u=(unsigned char)(di & 255);
  _offset_v=(unsigned char)(dj & 255);
  _offset_w=(unsigned char)(dk & 255);
}

// -------------------------------------------------------- 

int CHrdwTree::getGridSize(int depth) const
{
  int n=1; // depth = 0 => gridsize = 1
  for (int i=0;i<depth;i++)
    n*=m_iGridRes;
  return (n);
}

// -------------------------------------------------------- 

void CHrdwTree::setCgTransform()
{
  cgGLSetStateMatrixParameter(m_cgViewProj,
    CG_GL_MODELVIEW_PROJECTION_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(m_cgView,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(m_cgITView,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_INVERSE_TRANSPOSE);

  if (m_cgTransform != NULL)
    cgGLSetStateMatrixParameter(m_cgTransform,
    CG_GL_TEXTURE_MATRIX,
    CG_GL_MATRIX_IDENTITY);

}

// -------------------------------------------------------- 

void CHrdwTree::bind()
{
  // -> enable profile
  cgGLEnableProfile(g_cgVertexProfile);
  // -> bind program
  cgGLBindProgram(m_cgVertexProg);  
  // -> enable profile
  cgGLEnableProfile(g_cgFragmentProfile);
  // -> bind program
  cgGLBindProgram(m_cgFragmentProg);  
  // -> enable reference texture
  cgGLEnableTextureParameter(m_cgRefTex);
}

// -------------------------------------------------------- 

void CHrdwTree::unbind()
{
  // -> disable reference texture
  cgGLDisableTextureParameter(m_cgRefTex);
  // -> disable profiles
  cgGLDisableProfile(g_cgVertexProfile);
  cgGLDisableProfile(g_cgFragmentProfile);
}

// -------------------------------------------------------- 

void CHrdwTree::reset()
{
  m_iNextFreeI=0;
  m_iNextFreeJ=0;
  m_iNextFreeK=0;
  m_ReleasedGrids.clear();
  delete (m_Root);
  m_Root=allocNode(0,0,0,0);
}

// -------------------------------------------------------- 

void CHrdwTree::commit()
{
  glBindTexture(GL_TEXTURE_3D_EXT,m_uiIndirPool);
  m_Root->commit();
}

// -------------------------------------------------------- 

void CHrdwTree::bind_leafselect()
{
  // -> enable profile
  cgGLEnableProfile(g_cgFragmentProfile);
  // -> bind program
  cgGLBindProgram(m_cgLeafSelectFragmentProg);  
  // -> enable reference texture
  cgGLEnableTextureParameter(m_cgLeafSelectRefTex);
}

// -------------------------------------------------------- 

void CHrdwTree::unbind_leafselect()
{
  // -> disable reference texture
  cgGLDisableTextureParameter(m_cgLeafSelectRefTex);
  // -> disable profiles
  cgGLDisableProfile(g_cgFragmentProfile);
}


// -------------------------------------------------------- 

unsigned int CHrdwTree::blockMemorySize() const
{
  return (m_iGridRes*m_iGridRes*m_iGridRes*4);
}

// -------------------------------------------------------- 

unsigned int CHrdwTree::allocatedMemory() const
{
  return (m_NbGrids[0]*m_NbGrids[1]*m_NbGrids[2]*blockMemorySize());
}

// -------------------------------------------------------- 

unsigned int CHrdwTree::usedMemory() const
{
  return ((m_iNextFreeI
    +m_iNextFreeJ*m_NbGrids[0]
    +m_iNextFreeK*m_NbGrids[0]*m_NbGrids[1])
      *blockMemorySize());
}

// -------------------------------------------------------- 

unsigned int CHrdwTree::minMemory() const
{
  int mini=next_puiss2(m_iNextFreeI);
  int minj=next_puiss2(m_iNextFreeJ);
  int mink=next_puiss2(m_iNextFreeK);

  return ((mini
    +minj*m_NbGrids[0]
    +mink*m_NbGrids[0]*m_NbGrids[1])
      *blockMemorySize());  
}

// -------------------------------------------------------- 

int          CHrdwTree::nbNodes() const
{
  return (m_Root->nbNodes());
}

// -------------------------------------------------------- 

int          CHrdwTree::depth() const
{
  return (m_Root->depth());
}

// -------------------------------------------------------- 
// -------------------------------------------------------- 
// -------------------------------------------------------- 
// -------------------------------------------------------- 

CHrdwTree::hrdw_node::hrdw_node(
                                CHrdwTree *tree,
                                int depth,
                                int gi,int gj,int gk,
                                int bi,int bj,int bk)
                                : m_Tree(tree), 
                                m_iDepth(depth),
                                m_iGi(gi), m_iGj(gj), m_iGk(gk),
                                m_iBi(bi), m_iBj(bj), m_iBk(bk)
{
  m_iW=m_Tree->getGridRes();
  m_iH=m_Tree->getGridRes();
  m_iD=m_Tree->getGridRes();
  m_iX=bi*m_Tree->getGridRes();
  m_iY=bj*m_Tree->getGridRes();
  m_iZ=bk*m_Tree->getGridRes();
  /*
  cerr << "new hrdw_node " << endl
  << " g     = "
  << '(' << gi << ',' << gj << ',' << gk << ')' << endl
  << " b     = "
  << '(' << bi << ',' << bj << ',' << bk << ')' << endl
  << " x,y,z = "
  << '(' << m_iX << ',' << m_iY << ',' << m_iZ << ')' << endl
  << " sz    = "
  << '(' << m_iW << ',' << m_iH << ',' << m_iD << ')' << endl;
  */  
  m_iBytesUpdated=0;
  m_Parent=NULL;
  m_Data=new unsigned char[m_iW*m_iH*m_iD*4];
  m_DataUpdate=new unsigned char[m_iW*m_iH*m_iD*4];
  memset(m_Data,0,m_iW*m_iH*m_iD*4*sizeof(unsigned char));
  m_Children=new hrdw_node *[m_iW*m_iH*m_iD];
  resetUpdate();
  m_bChildrenNeedUpdate=false;
  for (int i=0;i<m_iW;i++)
    for (int j=0;j<m_iH;j++)
      for (int k=0;k<m_iD;k++)
      {
        m_Children[i+j*m_iW+k*m_iH*m_iW]=NULL;
        /* // checker board
        setLeafColor(i,j,k,
        (i+j+k)&1?255:0,
        (i+j+k)&1?255:0,
        (i+j+k)&1?255:0);
        */
        // FIXME: empty case ...
        setCellColor(i,j,k,
          255,255,255,
          HRDWTREE_EMPTY_ALPHA);

      }
}

// -------------------------------------------------------- 

CHrdwTree::hrdw_node::~hrdw_node()
{
  for (int i=0;i<m_iW;i++)
    for (int j=0;j<m_iH;j++)
      for (int k=0;k<m_iD;k++)
        if (m_Children[i+j*m_iW+k*m_iH*m_iW] != NULL)
          m_Tree->releaseNode(&m_Children[i+j*m_iW+k*m_iH*m_iW]);
  delete [](m_Children);
  delete [](m_Data);
  delete [](m_DataUpdate);
}

// -------------------------------------------------------- 

void       CHrdwTree::hrdw_node::resetUpdate()
{
  m_bNeedUpdate=false;

  m_UpdateMax[0]=0;
  m_UpdateMax[1]=0;
  m_UpdateMax[2]=0;
  m_UpdateMin[0]=m_iW-1;
  m_UpdateMin[1]=m_iH-1;
  m_UpdateMin[2]=m_iD-1;
}

// -------------------------------------------------------- 

void       CHrdwTree::hrdw_node::includeUpdate(int li,int lj,int lk)
{
  m_bNeedUpdate=true;
  if (li > m_UpdateMax[0]) m_UpdateMax[0]=li;
  if (li < m_UpdateMin[0]) m_UpdateMin[0]=li;
  if (lj > m_UpdateMax[1]) m_UpdateMax[1]=lj;
  if (lj < m_UpdateMin[1]) m_UpdateMin[1]=lj;
  if (lk > m_UpdateMax[2]) m_UpdateMax[2]=lk;
  if (lk < m_UpdateMin[2]) m_UpdateMin[2]=lk;
  if (m_Parent != NULL)
    m_Parent->setChildrenNeedUpdate();
}

// -------------------------------------------------------- 

void      CHrdwTree::hrdw_node::setChildrenNeedUpdate()
{
  m_bChildrenNeedUpdate=true;
  if (m_Parent != NULL)
    m_Parent->setChildrenNeedUpdate();
}

// -------------------------------------------------------- 

CHrdwTree::hrdw_node *CHrdwTree::hrdw_node::createChild(int li,int lj,int lk)
{
  int gridsize_next=m_Tree->getGridSize(m_iDepth+1);
  int gi=m_iGi*m_Tree->getGridRes()+li;
  int gj=m_iGj*m_Tree->getGridRes()+lj;
  int gk=m_iGk*m_Tree->getGridRes()+lk;
  hrdw_node *child=m_Tree->allocNode(m_iDepth+1,gi,gj,gk);
  m_Children[li+lj*m_iW+lk*m_iH*m_iW]=child;
  child->setParent(this);
  unsigned char offset_u,offset_v,offset_w;
  m_Tree->computeOffset(child->bi(),child->bj(),child->bk(),
    gi,gj,gk,
    gridsize_next,gridsize_next,gridsize_next,
    offset_u,offset_v,offset_w);
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+0]=offset_u;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+1]=offset_v;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+2]=offset_w;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+3]=HRDWTREE_NODE_ALPHA;
  includeUpdate(li,lj,lk);
  return (child);
}

// -------------------------------------------------------- 

void CHrdwTree::hrdw_node::deleteChild(int li,int lj,int lk)
{
  if (m_Children[li+lj*m_iW+lk*m_iH*m_iW] == NULL)
    throw CLibOctreeGPUException("CHrdwTree::hrdw_node::deleteChild - %d,%d,%d is not a node !",li,lj,lk);
  m_Tree->releaseNode(&m_Children[li+lj*m_iW+lk*m_iH*m_iW]);
  setCellColor(li,lj,lk,
    255,255,255,HRDWTREE_EMPTY_ALPHA);
}

// -------------------------------------------------------- 

void CHrdwTree::hrdw_node::deleteLeaf(int li,int lj,int lk)
{
  if (m_Children[li+lj*m_iW+lk*m_iH*m_iW] != NULL)
    throw CLibOctreeGPUException("CHrdwTree::hrdw_node::deleteLeaf - %d,%d,%d is not a leaf !",li,lj,lk);
  setCellColor(li,lj,lk,
    255,255,255,HRDWTREE_EMPTY_ALPHA);
}

// -------------------------------------------------------- 

CHrdwTree::hrdw_node *CHrdwTree::hrdw_node::getChild(int li,int lj,int lk)
{
  return (m_Children[li+lj*m_iW+lk*m_iH*m_iW]);
}

// -------------------------------------------------------- 

void CHrdwTree::hrdw_node::setLeafColor(int li,int lj,int lk,
                                        unsigned char r,
                                        unsigned char g,
                                        unsigned char b)
{
  if (m_Children[li+lj*m_iW+lk*m_iH*m_iW] != NULL)
    throw CLibOctreeGPUException("CHrdwTree::hrdw_node::setLeafColor - %d,%d,%d is not a leaf !",li,lj,lk);
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+0]=r;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+1]=g;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+2]=b;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+3]=HRDWTREE_LEAF_ALPHA;
  includeUpdate(li,lj,lk);
}

// -------------------------------------------------------- 

void CHrdwTree::hrdw_node::getLeafColor(int li,int lj,int lk,
                                        unsigned char& _r,
                                        unsigned char& _g,
                                        unsigned char& _b)
{
  _r=m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+0];
  _g=m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+1];
  _b=m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+2];
}

// -------------------------------------------------------- 

void CHrdwTree::hrdw_node::setCellColor(int li,int lj,int lk,
                                        unsigned char r,
                                        unsigned char g,
                                        unsigned char b,
                                        unsigned char a)
{
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+0]=r;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+1]=g;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+2]=b;
  m_Data[(li+lj*m_iW+lk*m_iH*m_iW)*4+3]=a;
  includeUpdate(li,lj,lk);
}

// -------------------------------------------------------- 

void CHrdwTree::hrdw_node::commit()
{

  if (glGetError())
    cerr << "CHrdwTree::hrdw_node - GL Error (before)" << endl;

  if (m_bChildrenNeedUpdate)
  {
    for (int i=0;i<m_iW;i++)
      for (int j=0;j<m_iH;j++)
        for (int k=0;k<m_iD;k++)
          if (m_Children[i+j*m_iW+k*m_iH*m_iW] != NULL)  
            m_Children[i+j*m_iW+k*m_iH*m_iW]->commit();
    m_bChildrenNeedUpdate=false;
  }

  if (!m_bNeedUpdate)
    return;

  // TODO FIXME - it seems that updating non power of 2 sizes is slow
  //              force update of full node indirection grid
  m_UpdateMin[0]=0;
  m_UpdateMax[0]=m_iW-1;
  m_UpdateMin[1]=0;
  m_UpdateMax[1]=m_iH-1;
  m_UpdateMin[2]=0;
  m_UpdateMax[2]=m_iD-1;

  // copy to data update
  int w=m_UpdateMax[0]-m_UpdateMin[0]+1;
  int h=m_UpdateMax[1]-m_UpdateMin[1]+1;
  int d=m_UpdateMax[2]-m_UpdateMin[2]+1;
  for (int i=m_UpdateMin[0];i<=m_UpdateMax[0];i++)
    for (int j=m_UpdateMin[1];j<=m_UpdateMax[1];j++)
      for (int k=m_UpdateMin[2];k<=m_UpdateMax[2];k++)
        for (int c=0;c<4;c++)
          m_DataUpdate[((i-m_UpdateMin[0])+
          (j-m_UpdateMin[1])*w+
          (k-m_UpdateMin[2])*w*h)*4+c]
          =m_Data[   (i+j*m_iW+k*m_iH*m_iW)*4+c];

          // send data
          glTexSubImage3DEXT(GL_TEXTURE_3D_EXT,0,
            m_iX+m_UpdateMin[0],
            m_iY+m_UpdateMin[1],
            m_iZ+m_UpdateMin[2],
            w,h,d,
            GL_RGBA,GL_UNSIGNED_BYTE,
            m_DataUpdate);

          /*  
          cerr << "commit - " 
          << '[' << m_UpdateMin[0] << ',' << m_UpdateMax[0] << ']'
          << '[' << m_UpdateMin[1] << ',' << m_UpdateMax[1] << ']'
          << '[' << m_UpdateMin[2] << ',' << m_UpdateMax[2] << ']'
          << endl;
          */

          resetUpdate();
          m_iBytesUpdated=w*h*d*4;

          if (glGetError())
            cerr << "CHrdwTree::hrdw_node - GL Error (after)" << endl;
}

// -------------------------------------------------------- 

unsigned int CHrdwTree::hrdw_node::bytesUpdated() const
{
  unsigned int b=m_iBytesUpdated;

  for (int i=0;i<m_iW;i++)
    for (int j=0;j<m_iH;j++)
      for (int k=0;k<m_iD;k++)
        if (m_Children[i+j*m_iW+k*m_iH*m_iW] != NULL)
          b+=m_Children[i+j*m_iW+k*m_iH*m_iW]->bytesUpdated();
  return (b);
}

// -------------------------------------------------------- 

int CHrdwTree::hrdw_node::nbNodes() const
{
  int n=1;
  for (int i=0;i<m_iW;i++)
    for (int j=0;j<m_iH;j++)
      for (int k=0;k<m_iD;k++)
        if (m_Children[i+j*m_iW+k*m_iH*m_iW] != NULL)
          n+=m_Children[i+j*m_iW+k*m_iH*m_iW]->nbNodes();
  return (n);
}

// -------------------------------------------------------- 

int CHrdwTree::hrdw_node::depth() const
{
  int dmax=0;
  for (int i=0;i<m_iW;i++)
    for (int j=0;j<m_iH;j++)
      for (int k=0;k<m_iD;k++)
        if (m_Children[i+j*m_iW+k*m_iH*m_iW] != NULL)
          dmax=max(m_Children[i+j*m_iW+k*m_iH*m_iW]->depth(),dmax);
  return (dmax+1);
}

// -------------------------------------------------------- 
