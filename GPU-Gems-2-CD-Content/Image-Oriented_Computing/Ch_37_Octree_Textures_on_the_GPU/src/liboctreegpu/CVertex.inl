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
// -------------------------------------------------------------------------
#include "CVertex.h"
#include <GL/gl.h>
#include <cmath>
// -------------------------------------------------------------------------
#ifndef ABS
  #define ABS(x) ((x)>0?(x):(-(x)))
#endif /* ABS */
// -------------------------------------------------------------------------
inline CVertex::CVertex()
{
  m_dX = m_dY = m_dZ = 0.0;
}
// -------------------------------------------------------------------------
inline CVertex::CVertex(const vertex_real	&x_,
		 const vertex_real	&y_,
		 const vertex_real	&z_)
{
  m_dX = x_;
  m_dY = y_;
  m_dZ = z_;
}
// -------------------------------------------------------------------------
inline CVertex::CVertex(const vertex_real	&x_,
		 const vertex_real	&y_)
{
  m_dX = x_;
  m_dY = y_;
  m_dZ = 0.0;
}
// -------------------------------------------------------------------------
inline vertex_real	CVertex::dot(const CVertex &a) const
{
  return (a.m_dX * m_dX + a.m_dY * m_dY + a.m_dZ * m_dZ);
}
// -------------------------------------------------------------------------
inline CVertex	operator+(const CVertex	&a,const CVertex &b)
{
  return CVertex(a.m_dX + b.m_dX, a.m_dY + b.m_dY, a.m_dZ + b.m_dZ);
}
// -------------------------------------------------------------------------
inline CVertex	operator-(const CVertex	&a,
			  const CVertex	&b)
{
  return CVertex(a.m_dX - b.m_dX, a.m_dY - b.m_dY, a.m_dZ - b.m_dZ);
}
// -------------------------------------------------------------------------
inline CVertex	operator-(const CVertex &p)
{
  return CVertex(-p.m_dX, -p.m_dY, -p.m_dZ);
}
// -------------------------------------------------------------------------
inline CVertex	operator*(const vertex_real& d,
			  const CVertex	&p)
{
  return CVertex(d * p.m_dX, d * p.m_dY, d * p.m_dZ);
}
// -------------------------------------------------------------------------
inline CVertex	operator*(const CVertex	&p,
			  const vertex_real&	d)
{
  return (d * p);
}
// -------------------------------------------------------------------------
inline CVertex	 operator*(const vertex_real m[4][4], const CVertex& b)
{
  return (CVertex(m[0][0]*b.x()+m[1][0]*b.y()+m[2][0]*b.z()+m[3][0],
		  m[0][1]*b.x()+m[1][1]*b.y()+m[2][1]*b.z()+m[3][1],
		  m[0][2]*b.x()+m[1][2]*b.y()+m[2][2]*b.z()+m[3][2]));
}
// -------------------------------------------------------------------------
inline CVertex	 operator*(const vertex_real *m, const CVertex& b)
{
  return (CVertex(m[0]*b.x()+m[4]*b.y()+m[8]*b.z()+m[12],
		  m[1]*b.x()+m[5]*b.y()+m[9]*b.z()+m[13],
		  m[2]*b.x()+m[6]*b.y()+m[10]*b.z()+m[14]));
}
// -------------------------------------------------------------------------
inline CVertex	 CVertex::normMult(const vertex_real m[4][4])
{
  return (CVertex(m[0][0]*x()+m[1][0]*y()+m[2][0]*z(),
		  m[0][1]*x()+m[1][1]*y()+m[2][1]*z(),
		  m[0][2]*x()+m[1][2]*y()+m[2][2]*z()));  
}
// -------------------------------------------------------------------------
inline CVertex	 CVertex::normMult(const vertex_real *m)
{
  return (CVertex(m[0]*x()+m[4]*y()+m[8]*z(),
		  m[1]*x()+m[5]*y()+m[9]*z(),
		  m[2]*x()+m[6]*y()+m[10]*z()));  
}
// -------------------------------------------------------------------------
inline CVertex	operator/(const CVertex	&p,
			  const vertex_real&	d)
{
  return CVertex(p.m_dX / d, p.m_dY / d, p.m_dZ / d);
}
// -------------------------------------------------------------------------
inline vertex_real	operator*(const CVertex &a,
			  const CVertex &b)
{
  return (a.dot(b));
}
// -------------------------------------------------------------------------
inline int operator==(const CVertex &a, const CVertex &b)
{
  return (ABS(b.m_dX-a.m_dX) < CVertex::getEpsilon() 
       && ABS(b.m_dY-a.m_dY) < CVertex::getEpsilon() 
       && ABS(b.m_dZ-a.m_dZ) < CVertex::getEpsilon());
}
// -------------------------------------------------------------------------
inline CVertex	CVertex::vect(const CVertex	&p) const
{
  return CVertex(m_dY * p.m_dZ - p.m_dY * m_dZ, m_dZ * p.m_dX - p.m_dZ * m_dX, m_dX * p.m_dY - p.m_dX * m_dY);
}
// -------------------------------------------------------------------------
inline CVertex	CVertex::cross(const CVertex	&p) const
{
  return (vect(p));
}
// -------------------------------------------------------------------------
inline vertex_real CVertex::getEpsilon()
{
  return (s_dEpsilon);
}
// -------------------------------------------------------------------------
inline void CVertex::setEpsilon(const vertex_real& e)
{
  if (e > 0.0) 
    s_dEpsilon=e; 
  else 
    throw CLibOctreeGPUException("CVertex: epsilon should be greater than 0.0 in CVertex::setEpsilon");
}
// -------------------------------------------------------------------------
inline void    CVertex::gl() const
{
  glVertex3d(m_dX,m_dY,m_dZ);
}
// -------------------------------------------------------------------------
inline vertex_real& CVertex::operator[](int i)
{
  switch (i)
  {
    case 0: return (m_dX);
    case 1: return (m_dY);
    case 2: return (m_dZ);
    default: throw CLibOctreeGPUException("CVertex: index out of range in CVertex::[]");
  }
}
// -------------------------------------------------------------------------
inline const vertex_real& CVertex::operator[](int i) const
{
  switch (i)
  {
    case 0: return (m_dX);
    case 1: return (m_dY);
    case 2: return (m_dZ);
    default: throw CLibOctreeGPUException("CVertex: index out of range in CVertex::[]");
  }
}
// -------------------------------------------------------------------------
inline void CVertex::axis(CVertex& u,CVertex& v) const
{
  u.set(-y(),x(),0.0); 
  if (u.norme() < CVertex::getEpsilon())
  {
    u.set(z(),0.0,-x());
    if (u.norme() < CVertex::getEpsilon())
      throw CLibOctreeGPUException("CVertex: can't create axis");
  }
  u.unit();
  v=vect(u);
  v.unit();
}
// -------------------------------------------------------------------------
inline void CVertex::operator+=(const CVertex& a)
{
  *this = *this + a; 
}
// -------------------------------------------------------------------------
inline void CVertex::operator*=(const vertex_real& d)
{
  *this = (*this) * d;
}
// -------------------------------------------------------------------------
inline void CVertex::operator/=(const vertex_real& d)
{
  *this = *this / d;
}
// ---------------------------------------------------------------
inline vertex_real CVertex::normalize()
{
  vertex_real tmp = (vertex_real)sqrt(m_dX * m_dX + m_dY * m_dY + m_dZ * m_dZ);

  if (tmp < CVertex::s_dEpsilon)
    throw CLibOctreeGPUException("CVertex: cannot normalize zero length vector in CVertex::unit");
  m_dX /= tmp;
  m_dY /= tmp;
  m_dZ /= tmp;
  return (tmp);
}
// -------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& s,const CVertex& v)
{
  return (s << "(" << v.x() << "," << v.y() << "," << v.z() << ")");
}
// -------------------------------------------------------------------------
inline std::istream& operator>>(std::istream& s,CVertex& v)
{
  int    i;
  vertex_real d[3];
  char   c;

  s >> c;
  if (c == '(')
    {
      for (i=0;i<3;i++)
	{
	  s >> d[i] >> c;
	  if (i < 2 && c != ',')
	    s.clear(std::ios::badbit);
	  else if (i == 2 && c != ')')
	    s.clear(std::ios::badbit);
	}
    }
  else
    {
      s.clear(std::ios::badbit);
    }
  if (s)
    v.set(d[0],d[1],d[2]);
  else
    throw CLibOctreeGPUException("CVertex::>> syntax error !");
  return (s);
}
// -------------------------------------------------------------------------
