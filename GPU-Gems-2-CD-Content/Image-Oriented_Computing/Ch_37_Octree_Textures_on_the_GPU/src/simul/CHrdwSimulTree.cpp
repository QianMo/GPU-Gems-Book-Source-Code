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

#include <math.h>

#include "CHrdwSimulTree.h"
#include "common.h"
#include "config.h"

#include <glux.h>

#include <CTexture.h>
#include <gltex.h>

using namespace std;

// ----------------------------------------------------------------

CHrdwSimulTree::CHrdwSimulTree(int boxres,
			       int NbGridsu,int NbGridsv,int NbGridsw,
			       const char *fp)
  : CHrdwTree(boxres,NbGridsu,NbGridsv,NbGridsw,SIMULTREE_MAX_DEPTH,fp)
{
  CHECK_GLERROR("CHrdwSimulTree::CHrdwSimulTree - 0 ");
  
  m_cgDensity=cgGetNamedParameter(m_cgFragmentProg,"Density");

  CHECK_GLERROR("CHrdwSimulTree::CHrdwSimulTree - 1 ");
}

// ----------------------------------------------------------------

CHrdwSimulTree::~CHrdwSimulTree()
{

}

// ----------------------------------------------------------------

void CHrdwSimulTree::bind(GLuint densid)
{
  CHECK_GLERROR("CHrdwSimulTree::bind - 0 ");
  // bind parent
  CHrdwTree::bind();
  // -> enable density texture
  cgGLSetTextureParameter(m_cgDensity,densid);
  cgGLEnableTextureParameter(m_cgDensity);

  CHECK_GLERROR("CHrdwSimulTree::bind - 1 ");
}

// ----------------------------------------------------------------

void CHrdwSimulTree::unbind()
{
  CHECK_GLERROR("CHrdwSimulTree::unbind - 0 ");
  // -> disable density texture
  cgGLDisableTextureParameter(m_cgDensity);
  // unbind parent
  CHrdwTree::unbind();

  CHECK_GLERROR("CHrdwSimulTree::unbind - 1 ");
}

// ----------------------------------------------------------------
