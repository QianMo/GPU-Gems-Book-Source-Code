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
#ifndef __SPRITE_EXCEPT__
#define __SPRITE_EXCEPT__

#include <stdarg.h>
#include <stdio.h>

class CLibOctreeGPUException
{
protected:
  char m_szMsg[512];
public:

  CLibOctreeGPUException(){m_szMsg[0]='\0';}

  CLibOctreeGPUException(char *msg,...)
  {
    va_list args;
    va_start(args, msg);

    vsprintf(m_szMsg,msg,args);
    fprintf(stderr,"[ERROR] ");
    vfprintf(stderr,msg,args);
    fprintf(stderr,"\n");

    va_end(args);
  }

  const char *getMsg() const {return (m_szMsg);}
};

#endif
