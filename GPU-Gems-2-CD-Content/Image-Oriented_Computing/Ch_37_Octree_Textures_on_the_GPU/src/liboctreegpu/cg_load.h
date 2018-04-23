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
#ifndef __CG_LOAD__
#define __CG_LOAD__

#include <Cg/cg.h>
#include <Cg/cgGL.h>

void      cg_set_path(const char *);
void      cg_init();
CGprogram cg_loadVertexProgram(const char *,const char **args=NULL);
CGprogram cg_loadFragmentProgram(const char *,const char **args=NULL);

extern CGprofile g_cgVertexProfile;
extern CGprofile g_cgFragmentProfile;
extern CGcontext g_cgContext;

#endif
