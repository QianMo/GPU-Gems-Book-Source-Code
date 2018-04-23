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
/**

   2004-09-13 Sylvain Lefebvre
 */

// --------------------------------------------------------

#ifndef __GENATLAS__
#define __GENATLAS__

class PBuffer;

#include <stdio.h>
#include "cg_load.h"

// --------------------------------------------------------

class CGenAtlas
{
private:

  PBuffer        *m_Buffer;
  unsigned int    m_uiTex;
  int             m_iSize;
  int             m_iSizeLog2;
  bool            m_bExtrapolate;

  CGprogram   m_cgFPGenDMAP;
  CGparameter m_cgDMAPTex;

  void quad();
  void genMIPmap();
  void genDMap();

public:

   CGenAtlas(int sz);
  ~CGenAtlas();
  
  void begin();
  void end();
  void bind();

  void setExtrapolate(bool b) {m_bExtrapolate=b;}

  int nbLevels() const {return (m_iSizeLog2);}
};

// --------------------------------------------------------

#endif

// --------------------------------------------------------
