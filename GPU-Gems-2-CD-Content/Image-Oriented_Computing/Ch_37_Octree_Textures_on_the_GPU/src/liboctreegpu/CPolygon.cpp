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
//--------------------------------------------------------

#ifdef WIN32
#  include <windows.h>
#endif

//--------------------------------------------------------

#include "CPolygon.h"

//--------------------------------------------------------

#define ON_EPSILON 0.1*CVertex::getEpsilon()

//--------------------------------------------------------

using namespace std;

//--------------------------------------------------------

int CPolygon::cut(const CPlane& p,
		  CPolygon& _front,
		  CPolygon& _back) const
{
  static float          dists[MAX_PTS+1];
  int                   ndists;
  static int            sides[MAX_PTS+1];
  int                   nsides;
  int                   counts[3];
  float                 dot;
  int                   numpt = m_iNbPts;
  int                   i;
  CVertex               pcur,p1,p2,nouv,pi;
      
  // get ready
  ndists=0;
  nsides=0;  
  // init resulting polys
  _back.m_iNbPts=0;
  _front.m_iNbPts=0;
  // compute side of each points
  counts[0] = counts[1] = counts[2] = 0;
  for (i=0 ; i<numpt ; i++)
  {
    pcur = m_Pts[i];
    dot = (float)p.distance(pcur);
    dists[i] = dot;
    if (dot > ON_EPSILON)
      sides[i] = SIDE_FRONT;
    else if (dot < -ON_EPSILON)
      sides[i] = SIDE_BACK;
    else
      sides[i] = SIDE_ON;
    counts[sides[i]]++;
  }
  sides[i] = sides[0];
  dists[i] = dists[0];
  if (counts[SIDE_ON] > 2)
  {
    _front.m_iNbPts=m_iNbPts;
    for (int i=0;i<m_iNbPts;i++)
      _front.m_Pts[i]=m_Pts[i];
    _back.m_iNbPts=m_iNbPts;
    for (int i=0;i<m_iNbPts;i++)
      _back.m_Pts[i]=m_Pts[i];
    return (ON);
  }
  if (counts[SIDE_FRONT] == 0)
  {
    _back.m_iNbPts=m_iNbPts;
    for (int i=0;i<m_iNbPts;i++)
      _back.m_Pts[i]=m_Pts[i];
    return (BACK);
  }
  if (counts[SIDE_BACK] == 0)
  {
    _front.m_iNbPts=m_iNbPts;
    for (int i=0;i<m_iNbPts;i++)
      _front.m_Pts[i]=m_Pts[i];
    return (FRONT);
  }
  // compute new vertices
  for (i=0 ; i<numpt ; i++)
  {
    pi = m_Pts[i];
    if (sides[i] == SIDE_ON)
    {
      _front.add(pi);
      _back.add(pi);
      continue;
    }
    if (sides[i] == SIDE_FRONT)
    {
      _front.add(pi);
    }
    else
    {
      _back.add(pi);
    }
    if (sides[i+1] == SIDE_ON || sides[i+1] == sides[i])
      continue;
    // compute cut vertex
    p1 = m_Pts[i];
    p2 = m_Pts[(i+1)%numpt];
    dot = dists[i] / (dists[i]-dists[i+1]);
    nouv = p1 + ((p2 - p1) * dot);
    _front.add(nouv);
    _back.add(nouv);
  }
  return (CUT); 
}

//--------------------------------------------------------
