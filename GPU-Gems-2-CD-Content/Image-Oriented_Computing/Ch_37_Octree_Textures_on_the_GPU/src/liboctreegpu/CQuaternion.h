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
//
// class CQuaternion.
// 
// Unit quaternion used to model a rotation in 3D.
// The CQuaternion is modeled using (w, x, y, z)
// i.e. cos(t/2), ax.sin(t/2), ay.sin(t/2), az.sin(t/2)
// where {ax,ay,az} is a rotation axis with norm=1
// and t is the CQuaternion angle
//
// Author : Sylvain Lefebvre from various sources (Francois Faure, WildMagick)
// History: 15/11/2001 ... created
//          25/08/2002 ... fix rotation bug
//          06/09/2002 ... add squad 
// -------------------------------------------------------------------------
#ifndef __CQUATERNION__
#define __CQUATERNION__
// -------------------------------------------------------------------------
#include <cmath>
#include <iostream>
#include "CVertex.h"
// -------------------------------------------------------------------------
#ifndef M_PI
#  define M_PI						3.14159265359
#endif
// -------------------------------------------------------------------------
using namespace std;
// -------------------------------------------------------------------------
class CQuaternion
{
 private:
  static vertex_real s_dEpsilon;
 public:
  //! Default constructor: does nothing
  CQuaternion();
  
  //! Components w, x, y, z
  CQuaternion(const vertex_real w, const vertex_real x, const vertex_real y, const vertex_real z);
  
  //! Axis and angle of rotation
  CQuaternion(const CVertex& axis, const vertex_real angle);
  
  //! Constructor with normalized axis, cos & sin of half angle
  CQuaternion(const CVertex&, const vertex_real, const vertex_real);

  //! Copy
  CQuaternion(const CQuaternion& );
  
  //! From matrix
  CQuaternion(vertex_real mat[4][4]); // FIXME: a tester .........

  //! From three vector (orthonormal frame)
  CQuaternion(const CVertex& u,const CVertex& v,const CVertex& w); // FIXME: a tester .........

  //! Destructor: does nothing
  ~CQuaternion();

  //! Get/Set w component in w, x, y, z
  const vertex_real& w() const;
  void  setW(vertex_real d) {m_dW=d;}

  //! Get x component in w, x, y, z
  const vertex_real& x() const;
  void  setX(vertex_real d) {m_dX=d;}
  
  //! Get y component in w, x, y, z
  const vertex_real& y() const;
  void  setY(vertex_real d) {m_dY=d;}
  
  //! Get z component in w, x, y, z
  const vertex_real& z() const;
  void  setZ(vertex_real d) {m_dZ=d;}
  
  //! Set components w, x, y, z
  CQuaternion& setWXYZ(const vertex_real w, const vertex_real x, const vertex_real y, const vertex_real z);
  
  //! Quaternion from matrix
  void fromMatrix(vertex_real mat[4][4]);

  /*! Set axis, angle.
      Axis represented by 3 vertex_real.
      Axis needs not be normalized. */
  void fromAxisAngle(const vertex_real x, const vertex_real y, const vertex_real z, const vertex_real angle);
  
  //! set quaternion from axis and angle
  void fromAxisAngle(const CVertex& axis, const vertex_real angle);
  //! get quaternion as axis and angle
  void toAxisAngle (vertex_real& rfAngle, CVertex& rkAxis) const;
  
  //! set a quaternion to the rotation of axis v and half angle ha
  void setAxisAndHalfAngle(CVertex ve, const vertex_real sin_ha, const vertex_real cos_ha);
  
  //! Inverse rotation
  CQuaternion inverse() const ;

  CQuaternion exp () const;
  CQuaternion log () const;

  //! slerp - Spherical interpolation
  static CQuaternion slerp(vertex_real t,const CQuaternion& from,const CQuaternion& to);

  //! intermediate for spherical quadratic interpolation
  static void squad_intermediate(const CQuaternion& rkQ0,
				 const CQuaternion& rkQ1,
				 const CQuaternion& rkQ2,
				 CQuaternion& rka);
  
  //! spherical quadratic interpolation
  static CQuaternion squad(vertex_real fT, 
			   const CQuaternion& rkP,
			   const CQuaternion& rkA, 
			   const CQuaternion& rkB,
			   const CQuaternion& rkQ);

  //! Returns the image of v
  CVertex image(const CVertex& v) const ;

  //! Returns the vector whose v is image.
  CVertex source(const CVertex& v) const ;

  //! Normalize
  CQuaternion& normalize();

  //! Identity (rotation with null angle)
  static const CQuaternion identity();

  /*! Write the rotation matrix m corresponding to this quaternion
      Assumes that the quaternion is normalized */
  void getRotMatrix(vertex_real m[3][3]) const;
  
  /*! Write the rotation matrix m corresponding to this quaternion, in OpenGL format;
    Assumes that the quaternion is normalized */
  void getOpenGLRotMatrix(vertex_real *m) const;
  void getOpenGLRotMatrix(vertex_real  m[4][4]) const;

  //! dot product
  vertex_real dot(const CQuaternion& q) const {return (x()*q.x()+y()*q.y()+z()*q.z()+w()*q.w());}

  //! Norm
  vertex_real norm() const;
  
  //! Assignement operator
  CQuaternion& operator = (const CQuaternion& );
  
  //! From vector assignement operator
  CQuaternion &operator = (const CVertex& v) ;
  
  //! Product with a quaternion
  CQuaternion operator * (const CQuaternion& ) const;
  
  //! Apply rotation to a vector
  CVertex operator * (const CVertex& ) const;
  
  //! Scalar
  CQuaternion operator * (vertex_real s) const {return CQuaternion(s*w(),s*x(),s*y(),s*z());}

  //! write to stream
  inline friend ostream& operator << (ostream& , const CQuaternion& );

  //! read from stream
  inline friend istream& operator >> (istream& , CQuaternion& );

  CQuaternion operator + (const CQuaternion& q) {return CQuaternion(w()+q.w(),x()+q.x(),y()+q.y(),z()+q.z());}

  CQuaternion operator - (const CQuaternion& q) {return CQuaternion(w()-q.w(),x()-q.x(),y()-q.y(),z()-q.z());}

  CQuaternion operator - () const {return CQuaternion(-w(),-x(),-y(),-z());}

private:
  
  union
  {
    vertex_real v[4];   // v[0] = w, v[1] = x, ...
    struct 
    {
      vertex_real m_dW,m_dX,m_dY,m_dZ;
    };
  };
  
};
// -------------------------------------------------------------------------
//! Apply rotation to a vector
CVertex operator* (const CVertex&, const CQuaternion&);
// -------------------------------------------------------------------------
//! scalar * CQuaternion
inline CQuaternion operator * (vertex_real s,const CQuaternion& q) {return (q*s);}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion()
{
  v[0]=1.0;
  v[1]=0.0; 
  v[2]=0.0; 
  v[3]=0.0;
}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion(const vertex_real w, const vertex_real x, const vertex_real y, const vertex_real z)
{
  v[0]=w; 
  v[1]=x; 
  v[2]=y; 
  v[3]=z;
}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion(const CVertex& axis, const vertex_real angle)
{
  fromAxisAngle(axis, angle); 
}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion(const CVertex &ve, 
				const vertex_real sin_ha,
				const vertex_real cos_ha)
{
  *this = ve*sin_ha ;
  v[0] = cos_ha ;
}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion(const CQuaternion& q)
{
  v[0]=q.v[0]; 
  v[1]=q.v[1]; 
  v[2]=q.v[2]; 
  v[3]=q.v[3];
}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion(const CVertex& _u,const CVertex& _v,const CVertex& _w)
{
  int i;
  static vertex_real m[4][4];

  for (i=0;i<3;i++)
  {
    m[0][i]=_u[i];
    m[1][i]=_v[i];
    m[2][i]=_w[i];
  }
  fromMatrix(m);
}
// -------------------------------------------------------------------------
inline CQuaternion::CQuaternion(vertex_real mat[4][4])
{
  fromMatrix(mat);
}
// -------------------------------------------------------------------------
inline void CQuaternion::fromMatrix(vertex_real mat[4][4])
{
  vertex_real t,s;
    
  t=1.0f + mat[0][0] + mat[1][1] + mat[2][2];
  if (t > s_dEpsilon)
  {
    s = 0.5f / (vertex_real)sqrt(t);
    v[1] = ( mat[2][1] - mat[1][2] ) * s;
    v[2] = ( mat[0][2] - mat[2][0] ) * s;
    v[3] = ( mat[1][0] - mat[0][1] ) * s;
    v[0] = 0.25f / s;
  }
  else if ( mat[0][0] > mat[1][1] && mat[0][0] > mat[2][2] )  
  {
    s  = (vertex_real)sqrt( 1.0f + mat[0][0] - mat[1][1] - mat[2][2] ) * 2;
    v[1] = 0.5f / s;
    v[2] = (mat[1][0] + mat[0][1] ) / s;
    v[3] = (mat[2][0] + mat[0][2] ) / s;
    v[0] = (mat[2][1] + mat[1][2] ) / s;
    
  } 
  else if ( mat[1][1] > mat[2][2] ) 
  { 
    s  = (vertex_real)sqrt( 1.0f + mat[1][1] - mat[0][0] - mat[2][2] ) * 2;
    v[1] = (mat[1][0] + mat[0][1] ) / s;
    v[2] = 0.5f / s;
    v[3] = (mat[2][1] + mat[1][2] ) / s;
    v[0] = (mat[2][0] + mat[0][2] ) / s;
    
  } 
  else 
  { 
    s  = (vertex_real)sqrt( 1.0f + mat[2][2] - mat[0][0] - mat[1][1] ) * 2;
    v[1] = (mat[2][0] + mat[0][2] ) / s;
    v[2] = (mat[2][1] + mat[1][2] ) / s;
    v[3] = 0.5f / s;
    v[0] = (mat[1][0] + mat[0][1] ) / s;
  }
}
// -------------------------------------------------------------------------
inline CQuaternion::~CQuaternion() {}
// -------------------------------------------------------------------------
inline const vertex_real& CQuaternion::w() const
{
  return v[0];
}
// -------------------------------------------------------------------------
inline const vertex_real&CQuaternion::x() const
{
  return v[1];
}
// -------------------------------------------------------------------------
inline const vertex_real&CQuaternion::y() const
{
  return v[2];
}
// -------------------------------------------------------------------------
inline const vertex_real&CQuaternion::z() const
{
  return v[3];
}
// -------------------------------------------------------------------------
inline void CQuaternion::fromAxisAngle(const CVertex& ax, const vertex_real an)
{
  v[0] = (vertex_real)cos(an/2);
  
  CVertex axn = ax;
  axn.normalize();
  vertex_real tmp = (vertex_real)sin(an/2);
  
  v[1] = axn.x()*tmp;
  v[2] = axn.y()*tmp;
  v[3] = axn.z()*tmp;
}
// -------------------------------------------------------------------------
inline void CQuaternion::fromAxisAngle(const vertex_real x, const vertex_real y, const vertex_real z, const vertex_real an)
{
  v[0] = (vertex_real)cos(an/2);
  
  CVertex axn(x, y, z);
  axn.normalize();
  vertex_real tmp = (vertex_real)sin(an/2);
  
  v[1] = axn.x()*tmp;
  v[2] = axn.y()*tmp;
  v[3] = axn.z()*tmp;
}
// -------------------------------------------------------------------------
inline void CQuaternion::toAxisAngle(vertex_real& rfAngle, CVertex& rkAxis) const
{
	// The quaternion representing the rotation is
	//   q = cos(A/2)+sin(A/2)*(x*i+y*j+z*k)

    vertex_real fSqrLength = x()*x()+y()*y()+z()*z();
    if ( fSqrLength > 0.0f )
    {
        rfAngle = 2.0f*(vertex_real)acos(m_dW);
        vertex_real fInvLength = 1.0f/(vertex_real)sqrt(fSqrLength);
        rkAxis=CVertex( x()*fInvLength,
		                    y()*fInvLength,
			                  z()*fInvLength);
    }
    else
    {
        // angle is 0 (mod 2*pi), so any axis will do
        rfAngle = 0.0f;
        rkAxis = CVertex(1.0,0.0,0.0);
    }
}
// -------------------------------------------------------------------------
inline void CQuaternion::setAxisAndHalfAngle(CVertex ve,const vertex_real sin_ha,const vertex_real cos_ha) 
{
  ve.normalize();
  *this = ve*sin_ha ;
  v[0] = cos_ha;
}
// -------------------------------------------------------------------------
inline CQuaternion& CQuaternion::setWXYZ(const vertex_real w, const vertex_real x, const vertex_real y, const vertex_real z)
{
  m_dW=w; 
  m_dX=x; 
  m_dY=y; 
  m_dZ=z;
  return *this;
}
// -------------------------------------------------------------------------
inline CQuaternion& CQuaternion::operator=(const CQuaternion& q)
{
  v[0] = q.v[0];
  v[1] = q.v[1];
  v[2] = q.v[2];
  v[3] = q.v[3];  
  return *this;
}
// -------------------------------------------------------------------------
inline CQuaternion& CQuaternion::operator=(const CVertex& ve) 
{
  v[1] = ve.x(); 
  v[2] = ve.y();
  v[3] = ve.z();
  return (*this);
}
// -------------------------------------------------------------------------
inline CQuaternion CQuaternion::operator*(const CQuaternion& q) const
{
    return CQuaternion(w()*q.w()-x()*q.x()-y()*q.y()-z()*q.z(),
		       w()*q.x()+x()*q.w()+y()*q.z()-z()*q.y(),
		       w()*q.y()+y()*q.w()+z()*q.x()-x()*q.z(),
		       w()*q.z()+z()*q.w()+x()*q.y()-y()*q.x());
}
// -------------------------------------------------------------------------
inline CQuaternion CQuaternion::inverse() const
{
  return CQuaternion(v[0], -v[1], -v[2], -v[3]);
}
// -------------------------------------------------------------------------
inline CQuaternion CQuaternion::slerp(vertex_real t,const CQuaternion& from,const CQuaternion& to)
{
  vertex_real fCos = from.dot(to);
  vertex_real fAngle = (vertex_real)acos(fCos);
  
  if (fabs(fAngle) < s_dEpsilon)
    return (from);
  
  vertex_real fSin    = (vertex_real)sin(fAngle);
  vertex_real fInvSin = 1.0f/fSin;
  vertex_real fCoeff0 = (vertex_real)sin((1.0-t)*fAngle)*fInvSin;
  vertex_real fCoeff1 = (vertex_real)sin(t*fAngle)*fInvSin;
  return ((fCoeff0*from) + (fCoeff1*to));
}
// -------------------------------------------------------------------------
inline CVertex CQuaternion::operator*(const CVertex& p) const 
{
  vertex_real r[4];

  r[0] = -p.x()*v[1] - p.y()*v[2] - p.z()*v[3];
  r[1] =  p.x()*v[0] + p.y()*v[3] - p.z()*v[2];
  r[2] = -p.x()*v[3] + p.y()*v[0] + p.z()*v[1];
  r[3] =  p.x()*v[2] - p.y()*v[1] + p.z()*v[0];
  
  return CVertex(v[0]*r[1] - v[1]*r[0] - v[2]*r[3] + v[3]*r[2],
		 v[0]*r[2] + v[1]*r[3] - v[2]*r[0] - v[3]*r[1],
		 v[0]*r[3] - v[1]*r[2] + v[2]*r[1] - v[3]*r[0]);
}
// -------------------------------------------------------------------------
inline CVertex operator*(const CVertex& p, const CQuaternion& q )
{
  return (q*p);
}
// -------------------------------------------------------------------------
inline vertex_real CQuaternion::norm() const
{
  return (vertex_real)sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] );
}
// -------------------------------------------------------------------------
inline CQuaternion& CQuaternion::normalize()
{
  vertex_real nrm = (*this).norm();
  vertex_real tmp = 1.0f/nrm;
  
  if (tmp < CVertex::getEpsilon())
    throw CLibOctreeGPUException("CQuaternion::normalize - degenerated quaternion !");
  v[0]*=tmp; v[1]*=tmp; v[2]*=tmp; v[3]*=tmp;

  return *this;
}
// -------------------------------------------------------------------------
inline const CQuaternion CQuaternion::identity()
{
  return CQuaternion(1.0f, 0.0, 0.0, 0.0);
}
// -------------------------------------------------------------------------
inline void CQuaternion::getRotMatrix(vertex_real m[3][3]) const
{
  vertex_real qw = w();
  vertex_real qx = x();
  vertex_real qy = y();
  vertex_real qz = z();
  
  vertex_real qww = qw*qw;
  vertex_real qwx = qw*qx;
  vertex_real qwy = qw*qy;
  vertex_real qwz = qw*qz;
  
  vertex_real qxx = qx*qx;
  vertex_real qxy = qx*qy;
  vertex_real qxz = qx*qz;
  
  vertex_real qyy = qy*qy;
  vertex_real qyz = qy*qz;
  
  vertex_real qzz = qz*qz;
  
  m[0][0] = qww+qxx - (qyy + qzz);
  m[0][1] =       2.0f*(qxy - qwz);
  m[0][2] =       2.0f*(qxz + qwy);
  
  m[1][0] =       2.0f*(qxy + qwz);
  m[1][1] = qww+qyy - (qzz + qxx);
  m[1][2] =       2.0f*(qyz - qwx);
  
  m[2][0] =       2.0f*(qxz - qwy);
  m[2][1] =       2.0f*(qyz + qwx);
  m[2][2] = qww+qzz - (qyy + qxx);
}
// -------------------------------------------------------------------------
inline void CQuaternion::getOpenGLRotMatrix(vertex_real *m) const
{
  vertex_real qw = w();
  vertex_real qx = x();
  vertex_real qy = y();
  vertex_real qz = z();
  
  vertex_real qwx = qw*qx;
  vertex_real qwy = qw*qy;
  vertex_real qwz = qw*qz;
  
  vertex_real qxx = qx*qx;
  vertex_real qxy = qx*qy;
  vertex_real qxz = qx*qz;
  
  vertex_real qyy = qy*qy;
  vertex_real qyz = qy*qz;
  
  vertex_real qzz = qz*qz;
  
  m[0 ] = 1.0f - 2.0f*(qyy + qzz);
  m[1 ] =        2.0f*(qxy - qwz);
  m[2 ] =        2.0f*(qxz + qwy);
  m[3 ] = 0.0f;
  
  m[4 ] =        2.0f*(qxy + qwz);
  m[5 ] = 1.0f - 2.0f*(qzz + qxx);
  m[6 ] =        2.0f*(qyz - qwx);
  m[7 ] = 0.0f;
  
  m[8 ] =        2.0f*(qxz - qwy);
  m[9 ] =        2.0f*(qyz + qwx);
  m[10] = 1.0f - 2.0f*(qyy + qxx);
  m[11] = 0.0;
  
  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = 0.0;
  m[15] = 1.0f;
}
// -------------------------------------------------------------------------
inline void CQuaternion::getOpenGLRotMatrix(vertex_real  m[4][4]) const
{
  vertex_real qw = w();
  vertex_real qx = x();
  vertex_real qy = y();
  vertex_real qz = z();
  
  vertex_real qwx = qw*qx;
  vertex_real qwy = qw*qy;
  vertex_real qwz = qw*qz;
  
  vertex_real qxx = qx*qx;
  vertex_real qxy = qx*qy;
  vertex_real qxz = qx*qz;
  
  vertex_real qyy = qy*qy;
  vertex_real qyz = qy*qz;
  
  vertex_real qzz = qz*qz;
  
  m[0][0] = 1.0f - 2.0f*(qyy + qzz);
  m[0][1] =        2.0f*(qxy - qwz);
  m[0][2] =        2.0f*(qxz + qwy);
  m[0][3] = 0.0;
  
  m[1][0] =        2.0f*(qxy + qwz);
  m[1][1] = 1.0f - 2.0f*(qzz + qxx);
  m[1][2] =        2.0f*(qyz - qwx);
  m[1][3] = 0.0;
  
  m[2][0] =        2.0f*(qxz - qwy);
  m[2][1] =        2.0f*(qyz + qwx);
  m[2][2] = 1.0f - 2.0f*(qyy + qxx);
  m[2][3] = 0.0;
  
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = 1.0f;
}
// -------------------------------------------------------------------------
inline CVertex CQuaternion::image(const CVertex& ve) const 
{
  return ((*this)*ve);
}
// -------------------------------------------------------------------------
inline CVertex CQuaternion::source(const CVertex& ve) const 
{
  CQuaternion q;

  q= (*this).inverse();
  return (q*ve);
}
//----------------------------------------------------------------------------
inline CQuaternion CQuaternion::exp () const
{
    // If q = A*(x*i+y*j+z*k) where (x,y,z) is unit length, then
    // exp(q) = cos(A)+sin(A)*(x*i+y*j+z*k).  If sin(A) is near zero,
    // use exp(q) = cos(A)+A*(x*i+y*j+z*k) since A/sin(A) has limit 1.

    vertex_real fAngle = (vertex_real)sqrt(x()*x()+y()*y()+z()*z());
    vertex_real fSin = (vertex_real)sin(fAngle);

    CQuaternion kResult;
    kResult.setW((vertex_real)cos(fAngle));

    if ( fabs(fSin) >= s_dEpsilon )
    {
        vertex_real fCoeff = fSin/fAngle;
        kResult = CVertex(fCoeff*x(),
			                    fCoeff*y(),
			                    fCoeff*z());
    }
    else
    {
      kResult = CVertex(x(),y(),z());
    }
    
    return kResult;
}
//----------------------------------------------------------------------------
inline CQuaternion CQuaternion::log () const
{
  // If q = cos(A)+sin(A)*(x*i+y*j+z*k) where (x,y,z) is unit length, then
  // log(q) = A*(x*i+y*j+z*k).  If sin(A) is near zero, use log(q) =
  // sin(A)*(x*i+y*j+z*k) since sin(A)/A has limit 1.

  CQuaternion kResult;
  kResult.setW(0.0);

  if ( fabs(w()) < 1.0 )
  {
    vertex_real fAngle = (vertex_real)acos(w());
    vertex_real fSin = (vertex_real)sin(fAngle);
    if ( fabs(fSin) >= s_dEpsilon )
    {
      vertex_real fCoeff = fAngle/fSin;
      kResult = CVertex(fCoeff*x(),
			fCoeff*y(),
			fCoeff*z());
      return kResult;
    }
  }
    
  kResult = CVertex(x(),y(),z());

  return kResult;
}
//----------------------------------------------------------------------------
inline void CQuaternion::squad_intermediate (const CQuaternion& rkQ0,
					     const CQuaternion& rkQ1, 
					     const CQuaternion& rkQ2, 
					     CQuaternion& rkA)
{
  CQuaternion kQ1inv = rkQ1.inverse();
  CQuaternion rkP2 = kQ1inv*rkQ2;
  CQuaternion rkP0 = kQ1inv*rkQ0;
  CQuaternion kArg = (-0.25)*(rkP2.log()+rkP0.log());

  rkA = rkQ1*(kArg.exp());
}
//----------------------------------------------------------------------------
inline CQuaternion CQuaternion::squad (vertex_real fT, 
				const CQuaternion& rkP,
				const CQuaternion& rkA, 
				const CQuaternion& rkB, 
				const CQuaternion& rkQ)
{
  vertex_real fSlerpT = 2.0f*fT*(1.0f-fT);
  CQuaternion kSlerpP = slerp(fT,rkP,rkQ);
  CQuaternion kSlerpQ = slerp(fT,rkA,rkB);
  return slerp(fSlerpT,kSlerpP,kSlerpQ);
}
//----------------------------------------------------------------------------
inline ostream& operator<<(ostream& s,const CQuaternion& q)
{
  return (s << "(" << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << ")");
}
// -------------------------------------------------------------------------
inline istream& operator>>(istream& s,CQuaternion& q)
{
  int    i;
  vertex_real d[4];
  char   c;

  s >> c;
  if (c == '(')
    {
      for (i=0;i<4;i++)
	{
	  s >> d[i] >> c;
	  if (i < 3 && c != ',')
	    s.clear(ios::badbit);
	  else if (i == 3 && c != ')')
	    s.clear(ios::badbit);
	}
    }
  else
    {
      s.clear(ios::badbit);
    }
  if (s)
    q.setWXYZ(d[3],d[0],d[1],d[2]);
  else
    throw CLibOctreeGPUException("CQuaternion::>> syntax error !");
  return (s);
}
// -------------------------------------------------------------------------
#endif
// -------------------------------------------------------------------------
