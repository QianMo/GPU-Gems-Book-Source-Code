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
// CVertex class - define vertex and advanced operations
//
// History: 10/03/2001 ... creation
//
// Author : Sylvain Lefebvre - sylvain.lefebvre@imag.fr 
//                           - http://www.aracknea.net
// -------------------------------------------------------------------------
#ifndef	__CVERTEX_H_
#define	__CVERTEX_H_
// -------------------------------------------------------------------------
typedef float vertex_real;
// -------------------------------------------------------------------------
#include <iostream>
#include "CLibOctreeGPUException.h"
// -------------------------------------------------------------------------
#define BIPRODUCT(a,b,c) (( ((b)-(a)).vect((c)-(a)) ).norme())
// -------------------------------------------------------------------------
class	CVertex
{
 public:
/*
	 class equal
	 {
	 public:
		 bool operator()(const CVertex& v1,const CVertex& v2)
		 {
			 return (v1 == v2);
		 }
	 };
*/
 protected:
  static vertex_real s_dEpsilon;
  vertex_real	m_dX, m_dY, m_dZ;
 public:
  CVertex();
  CVertex(const vertex_real& x_, const vertex_real& y_, const vertex_real& z_);
  CVertex(const vertex_real& x_, const vertex_real& y_);
  
  vertex_real		 dist(const CVertex& p) const;
  vertex_real		 norme() const;
  vertex_real		 length() const {return (norme());}
  inline CVertex&	 unit() {normalize(); return (*this);}
  vertex_real                 normalize();
  static inline vertex_real   getEpsilon();
  static inline void     setEpsilon(const vertex_real& e);
  inline CVertex	 normMult(const vertex_real m[4][4]);
  inline CVertex	 normMult(const vertex_real *m);
  inline CVertex	 vect(const CVertex& p) const;
  inline CVertex	 cross(const CVertex& p) const;
  inline vertex_real	         dot(const CVertex& p) const;
  inline void            axis(CVertex& u,CVertex& v) const;
  inline void            gl() const;
  inline void            operator+=(const CVertex& a);
  inline void	         operator*=(const vertex_real& d);
  inline void	         operator/=(const vertex_real& d);

  inline vertex_real x() const {return (m_dX);}
  inline vertex_real y() const {return (m_dY);}
  inline vertex_real z() const {return (m_dZ);}
  inline void   setX(vertex_real d){m_dX=d;}
  inline void   setY(vertex_real d){m_dY=d;}
  inline void   setZ(vertex_real d){m_dZ=d;}
  inline void   set(vertex_real x,vertex_real y,vertex_real z){m_dX=x;m_dY=y;m_dZ=z;}
  inline vertex_real& operator[](int i);
  inline const vertex_real& operator[](int i) const;

  friend inline int	 operator==(const CVertex& a, const CVertex& b);
  friend inline std::ostream& operator<<(std::ostream& s,const CVertex& v);
  friend inline std::istream& operator>>(std::istream& s,CVertex& v);

  friend inline CVertex	 operator+(const CVertex& a, const CVertex& b);
  friend inline CVertex	 operator-(const CVertex& a, const CVertex& b);
  friend inline CVertex	 operator-(const CVertex& p);
  friend inline CVertex	 operator*(const vertex_real& d, const CVertex& b);
  friend inline CVertex	 operator*(const CVertex& b, const vertex_real& d);
  friend inline CVertex	 operator*(const vertex_real m[4][4], const CVertex& b);
  friend inline CVertex	 operator*(const vertex_real *m, const CVertex& b);
  friend inline CVertex	 operator/(const CVertex& b, const vertex_real& d);
  friend inline vertex_real	 operator*(const CVertex& a, const CVertex& b);
};
// -------------------------------------------------------------------------
#include "CVertex.inl"
// -------------------------------------------------------------------------
#endif /*__CVERTEX_H_*/
// -------------------------------------------------------------------------
