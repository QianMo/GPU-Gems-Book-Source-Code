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
#include "config.h"

#include "help.h"

char g_HelpStrings[HELP_MAX_STRINGS][HELP_MAX_STRING_LENGTH]={
  " --------------------------------",
  " Painting on                     ",
  "          unparameterized meshes ",
  " --------------------------------",
  "             GPU Gems 2 release  ",
  " --------------------------------",
  "  [q]     quit",
  "  [o/p]   zoom in/out",
  "  [+/-]   change brush size",
  "  [SPACE] center view on brush",
  "  [c]     change color",
  "  [l/k]   change brush opacity",
  "  [r]     refinement brush",
  "  [i]     enable linear interp.",
  "    (done at max refinement depth)",
  "  [s]     save texture (out.octree)",
  "     (load on command line)",
  "  [t]     show tree structure",
  " --------------------------------",
  "Sylvain.Lefebvre@laposte.net     ",
  "     (c) 2004 all rights reserved",
  " --------------------------------",
  "\0",
};
