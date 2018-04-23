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

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cg_load.h"
#include "OSAdapter.h"
#include "CLibOctreeGPUException.h"

#include <iostream>
using namespace std;

/* -------------------------------------------------------- */

CGprofile g_cgVertexProfile   = CG_PROFILE_VP30;
CGprofile g_cgFragmentProfile = CG_PROFILE_FP30;
char      g_szCgPath[1024];
char      g_szBuffer[1024];

CGcontext g_cgContext=NULL;

/* -------------------------------------------------------- */

static bool fileExists(const char *fname)
{
  FILE *f=NULL;
  f=fopen(fname,"r");
  if (f == NULL)
    return (false);
  else
  {
    fclose(f);
    return (true);
  }
}

/* -------------------------------------------------------- */

void cg_init()
{
  g_cgContext=cgCreateContext();

  g_cgVertexProfile=cgGLGetLatestProfile(CG_GL_VERTEX);
  cgGLSetOptimalOptions(g_cgVertexProfile);

  g_cgFragmentProfile=cgGLGetLatestProfile(CG_GL_FRAGMENT);
  cgGLSetOptimalOptions(g_cgFragmentProfile);

  if (g_cgFragmentProfile == CG_PROFILE_FP20)
    cerr << "[Cg] fragment profile FP30 selected" << endl;
  else if (g_cgFragmentProfile == CG_PROFILE_FP30)
    cerr << "[Cg] fragment profile FP30 selected" << endl;
#ifdef CG_V1_2
  else if (g_cgFragmentProfile == CG_PROFILE_FP40)
  {
    cerr << "[Cg] fragment profile FP40 selected" << endl;
    cerr << "     -> switching back to FP30 (update cg_load.cpp to test FP40 profile)" << endl;
    g_cgFragmentProfile=CG_PROFILE_FP30;
  }
#endif
  else if (g_cgFragmentProfile == CG_PROFILE_ARBFP1)
    cerr << "[Cg] fragment profile ARB selected" << endl;

  g_szCgPath[0]='\0';
}

/* -------------------------------------------------------- */

void cg_set_path(const char *p)
{
  strncpy(g_szCgPath,p,1024);
  cerr << "Cg path set to: " << p << endl;
}

/* -------------------------------------------------------- */

CGprogram cg_loadVertexProgram(const char *prg,const char **args)
{
  CGprogram vp;
  
  if (glGetError())
    cerr << "[Cg] cg_loadVertexProgram - GL Error (0)" << prg << endl;

  cerr << "[Cg] loading " << prg << " ... ";

  sprintf(g_szBuffer,"%s/%s",g_szCgPath,prg);
  OSAdapter::convertName(g_szBuffer);
  if (!fileExists(g_szBuffer))
    throw CLibOctreeGPUException("\n[Cg] cg_loadVertexProgram - File %s does not exist",g_szBuffer);
  vp = cgCreateProgramFromFile(g_cgContext, CG_SOURCE, 
			       g_szBuffer, g_cgVertexProfile, "main", args);
  if (glGetError())
    cerr << endl <<  "[Cg] cg_loadVertexProgram - GL Error (1)" << prg << endl;

  if (NULL == vp)
    throw CLibOctreeGPUException("\n[Cg] cg_loadVertexProgram - Unable to compile %s\nYour system does not support required hardware capabilities ... (>= NV30)",g_szBuffer);

  cgGLLoadProgram(vp);

  if (glGetError())
    cerr << endl <<  "[Cg] cg_loadVertexProgram - GL Error (2)" << prg << endl;

  cerr << "done." << endl;

  return (vp);
}

/* -------------------------------------------------------- */

CGprogram cg_loadFragmentProgram(const char *prg,const char **args)
{
  CGprogram fp;

  cerr << "[Cg] loading " << prg << " ... ";
  
  if (glGetError())
    cerr << endl << "[Cg] cg_loadFragmentProgram - GL Error (0) " << prg << endl;

  sprintf(g_szBuffer,"%s/%s",g_szCgPath,prg);
  OSAdapter::convertName(g_szBuffer);
  if (!fileExists(g_szBuffer))
    throw CLibOctreeGPUException("\n[Cg] cg_loadFragmentProgram - File %s does not exist",g_szBuffer);
  fp = cgCreateProgramFromFile(g_cgContext, CG_SOURCE, 
			       g_szBuffer, g_cgFragmentProfile, "main", args);

  if (glGetError())
    cerr << endl <<  "[Cg] cg_loadFragmentProgram - GL Error (1) " << prg << endl;

  if (NULL == fp)
    throw CLibOctreeGPUException("\n[Cg] cg_loadFragmentProgram - Unable to compile %s\nYour system does not support required hardware capabilities ... (>= NV30)",g_szBuffer);

  cgGLLoadProgram(fp);

  if (glGetError())
    cerr << endl <<  "[Cg] cg_loadFragmentProgram - GL Error (2) " << prg << endl;

  cerr << "done." << endl;

  return (fp);
}

/* -------------------------------------------------------- */
