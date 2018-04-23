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
#ifndef __POLYGON__
#define __POLYGON__

#include <vector>

#include "CVertex.h"
#include "assert.h"

#define MAX_PTS    10
#define ON          0
#define FRONT       1
#define BACK        2
#define CUT        -1
#define SIDE_ON     0
#define SIDE_FRONT  1
#define SIDE_BACK   2

class CPlane
{
public:
  double  m_D;
  CVertex m_N;

  CPlane(const CVertex& p,const CVertex& n)
    {
      m_N=n;
      m_N.unit();
      m_D=n.dot(p);
    }
  CPlane(const CVertex& p0,
	 const CVertex& p1,
	 const CVertex& p2)
    {
      m_N=((p1-p0).cross(p2-p0)).unit();
      m_D=m_N.dot(p0);
    }
  double distance(const CVertex& p) const
    {
      return (p.dot(m_N) - m_D);
    }
};

class CPolygon
{
public:
  CVertex    m_Pts[MAX_PTS];
  int        m_iNbPts;
 
  CPolygon() {m_iNbPts=0;}
  CPolygon(const CVertex& p0,const CVertex& p1,const CVertex& p2)
    {
      m_iNbPts=0;
      add(p0);
      add(p1);
      add(p2);
   }
  CPolygon(const CVertex& p0,const CVertex& p1,const CVertex& p2,const CVertex& p3)
    {
      m_iNbPts=0;
      add(p0);
      add(p1);
      add(p2);
      add(p3);
    }

  void add(const CVertex& p)
    {
      assert(m_iNbPts < MAX_PTS);
      m_Pts[m_iNbPts++]=p;
    }
 
  void gl()
    {
	  CVertex n=normal();
      glBegin(GL_TRIANGLE_FAN);
	  glNormal3f(n.x(),n.y(),n.z());
      for (int p=0;p<(int)m_iNbPts;p++)
      {
		glTexCoord3f(m_Pts[p].x(),m_Pts[p].y(),m_Pts[p].z());
		m_Pts[p].gl();
      }
      glEnd();
    }

  CVertex normal()
    {
      if (m_iNbPts > 2)
      {
	try
	{
	  CVertex& p0=m_Pts[0];
	  CVertex& p1=m_Pts[1];
	  CVertex& p2=m_Pts[2];
	  CVertex n=((p1-p0).cross(p2-p0));
	  if (n.norme() > CVertex::getEpsilon())
	    n.normalize();
	  return (n);
	}
	catch (...)
	{}
      }
      return (CVertex(0,0,0));
    }

  CVertex center()
    {
      CVertex ctr(0,0,0);
      if (m_iNbPts < 1)
	return (ctr);
      for (int i=0;i<(int)m_iNbPts;i++)
	ctr+=m_Pts[i];
      return (ctr / (vertex_real) m_iNbPts);
    } 

  double area() const
    {
      if (m_iNbPts < 3)
	return (0.0);
      double a=0.0;
      for (int i=1;i<(int)m_iNbPts-1;i++)
      {
	CVertex u=(m_Pts[i  ]-m_Pts[0]);
	CVertex v=(m_Pts[i+1]-m_Pts[0]);
	a+=(u.cross(v)).norme();
      }
      return (a);
    }

  int cut(const CPlane& p,
	  CPolygon& _front,
	  CPolygon& _back) const;

  bool empty() const {return (m_iNbPts < 3);}
};

#endif
