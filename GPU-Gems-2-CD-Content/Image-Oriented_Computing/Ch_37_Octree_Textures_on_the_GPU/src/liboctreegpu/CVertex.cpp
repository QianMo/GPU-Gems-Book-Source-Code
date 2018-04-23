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
// ---------------------------------------------------------------

#ifdef WIN32
#include <windows.h>
#endif

// ---------------------------------------------------------------

#include <math.h>
#include "CVertex.h"

// ---------------------------------------------------------------

vertex_real CVertex::s_dEpsilon=0.0000000001f;

// ---------------------------------------------------------------

vertex_real	CVertex::dist(const CVertex	&p) const
{
  const vertex_real	&tmp1 = m_dX - p.m_dX;
  const vertex_real	&tmp2 = m_dY - p.m_dY;
  const vertex_real	&tmp3 = m_dZ - p.m_dZ;

  return (vertex_real)sqrt(tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3);
}

// ---------------------------------------------------------------

vertex_real	CVertex::norme() const
{
  return (vertex_real)sqrt(m_dX*m_dX+m_dY*m_dY+m_dZ*m_dZ);
}

// ---------------------------------------------------------------
