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
#include "CProfiler.h"
#include "CFont.h"
#include <GL/gl.h>
#include "help.h"

#define HELP_STRING_TIME 2000
#define HELP_NB_LINES    5

extern CFont *g_Font;
int           g_iNbHelpStrings=-1;
extern char   g_HelpStrings[HELP_MAX_STRINGS][HELP_MAX_STRING_LENGTH];

static int gs_iCurrentHelpString=0;
static int gs_iLastHelpStringTime=-1;

void drawHelp()
{
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  if (g_iNbHelpStrings < 0)
  {
    for (g_iNbHelpStrings=0;(g_iNbHelpStrings < 1024) && (g_HelpStrings[g_iNbHelpStrings][0] != '\0');g_iNbHelpStrings++)
    {}
    std::cerr << std::endl << "       ---= HELP =---" << std::endl;
    for (int i=0;i<g_iNbHelpStrings;i++)
    {
      std::cerr << g_HelpStrings[i] << std::endl;
    }
  }
  if (gs_iLastHelpStringTime < 0)
    gs_iLastHelpStringTime=(int)PROFILER.getRealTime();

  double sz=0.03;
  for (int i=0;i<=HELP_NB_LINES;i++)
  {
    int c=((-HELP_NB_LINES+i+(int)((PROFILER.getRealTime()-gs_iLastHelpStringTime)/(double)HELP_STRING_TIME))) % g_iNbHelpStrings;
    if (c >= 0)
    {
      double tr=((PROFILER.getRealTime()-gs_iLastHelpStringTime)/(double)HELP_STRING_TIME);
      tr=tr-(int)tr;
      if (i == 0)
        glColor4d(1.0,1.0,1.0,1.0-tr);
      else
        glColor4d(1.0,1.0,1.0,1);
      tr*=sz;
      g_Font->printString(0.0,1.0-sz*HELP_NB_LINES+i*sz-tr,sz,g_HelpStrings[c]);
    }
  }
  glPopAttrib();
}
