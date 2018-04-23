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
#ifndef __HRDWSIMULTREE__
#define __HRDWSIMULTREE__

#include "CHrdwTree.h"

class CTexSpriteTree;
class CHrdwSimulManager;
class CCubeMap;

class CHrdwSimulTree : public CHrdwTree
{
protected:

  CGparameter m_cgDensity;

public:

  CHrdwSimulTree(int boxres,
    int nbboxesu,int nbboxesv,int nbboxesw,
    const char *fp);
  ~CHrdwSimulTree();

  virtual void bind(GLuint);
  virtual void unbind(); 

};

#endif
