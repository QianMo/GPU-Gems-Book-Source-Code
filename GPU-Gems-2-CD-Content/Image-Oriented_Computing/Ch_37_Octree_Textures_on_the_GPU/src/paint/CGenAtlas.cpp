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

#include "CGenAtlas.h"
#include "pbuffer.h"

#include <GL/gl.h>
#include <GL/glu.h>

#include "utils.h"

#include <iostream>

// --------------------------------------------------------

#define CHECK_GLERROR(m) if (glGetError()) std::cerr << std::endl << "ERROR: OpenGL - " << m << "\n\tline " << __LINE__ << " file " << __FILE__ << std::endl << std::endl;

// --------------------------------------------------------

CGenAtlas::CGenAtlas(int sz)
{
  m_iSize     = sz;
  m_iSizeLog2 = puiss2(m_iSize);

  // create pbufer
  m_Buffer=new PBuffer("rgb alpha");
  m_Buffer->Initialize(m_iSize,m_iSize,false,true);

  // create texture
  glGenTextures(1,&m_uiTex);
  glBindTexture(GL_TEXTURE_2D,m_uiTex);

  glTexImage2D(GL_TEXTURE_2D,0,
	       GL_RGBA,
	       m_iSize,m_iSize,0,
	       GL_RGBA,GL_UNSIGNED_BYTE,NULL);

  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MAG_FILTER,
		  GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MIN_FILTER,
		  GL_LINEAR_MIPMAP_LINEAR);

  // by default extrapolate
  m_bExtrapolate=true;

  // load Cg programs
  char        *argv[3];
  static char  str0[64],str1[64];

  sprintf(str0,"-DTEX_SIZE=%d",m_iSize);
  sprintf(str1,"-DTEX_SIZE_LOG2=%d",m_iSizeLog2);

  argv[0]=str0;
  argv[1]=str1;
  argv[2]=NULL;

  m_cgFPGenDMAP     = cg_loadFragmentProgram("genatlas/fp_gen_atlas.cg",(const char **)argv);
  m_cgDMAPTex       = cgGetNamedParameter(m_cgFPGenDMAP,"Tex");

}

// --------------------------------------------------------

CGenAtlas::~CGenAtlas()
{
  delete (m_Buffer);
}

// --------------------------------------------------------

void CGenAtlas::begin()
{
  m_Buffer->Activate();

  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0.0,1.0,0.0,1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

}

// --------------------------------------------------------

void CGenAtlas::end()
{
  glBindTexture(GL_TEXTURE_2D,m_uiTex);

  glTexParameteri(GL_TEXTURE_2D,GL_GENERATE_MIPMAP_SGIS,GL_TRUE);

  glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,
    0,0,m_iSize,m_iSize);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  m_Buffer->Deactivate();

  if (m_bExtrapolate)
    genDMap();

  glBindTexture(GL_TEXTURE_2D,m_uiTex);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_LOD_SGIS,0);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LOD_SGIS,m_iSizeLog2);
  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MAG_FILTER,
		  GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MIN_FILTER,
		  GL_LINEAR_MIPMAP_LINEAR);
}

// --------------------------------------------------------

void  CGenAtlas::bind()
{
  glBindTexture(GL_TEXTURE_2D,m_uiTex); 
}

// --------------------------------------------------------

void CGenAtlas::quad()
{
  glBegin(GL_QUADS);
  glTexCoord2d(0,0);
  glVertex2i(0,0);
  glTexCoord2d(0,1);
  glVertex2i(0,1);
  glTexCoord2d(1,1);
  glVertex2i(1,1);
  glTexCoord2d(1,0);
  glVertex2i(1,0);
  glEnd();
}

// --------------------------------------------------------

void CGenAtlas::genDMap()
{
  CHECK_GLERROR("gen_Atlas - entry");

  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);

  m_Buffer->Activate();
  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);
  glColor4d(1,1,1,1);
  glBindTexture(GL_TEXTURE_2D,m_uiTex);

  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MAG_FILTER,
		  GL_LINEAR);

  glTexParameteri(GL_TEXTURE_2D,
		  GL_TEXTURE_MIN_FILTER,
		  GL_LINEAR_MIPMAP_NEAREST);

  cgGLEnableProfile(g_cgFragmentProfile);
  cgGLBindProgram(m_cgFPGenDMAP);
  cgGLSetTextureParameter(m_cgDMAPTex,m_uiTex);
  cgGLEnableTextureParameter(m_cgDMAPTex);

  glViewport(0,0,
	     ((int)m_iSize),
	     ((int)m_iSize));

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0,1.0,0.0,1.0);
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();      
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  quad();

  cgGLDisableProfile(g_cgFragmentProfile);
  cgGLDisableTextureParameter(m_cgDMAPTex);

  glBindTexture(GL_TEXTURE_2D,m_uiTex);

  glCopyTexImage2D(GL_TEXTURE_2D,0,
		   GL_RGBA,
		   0,0,
		   m_iSize,
		   m_iSize,
		   0);

  m_Buffer->Deactivate();

  CHECK_GLERROR("gen_Atlas - exit");
}

// --------------------------------------------------------
