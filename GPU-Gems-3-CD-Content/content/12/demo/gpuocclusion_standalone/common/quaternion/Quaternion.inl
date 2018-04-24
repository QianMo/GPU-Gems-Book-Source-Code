/*! \file Quaternion.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Quaternion.h.
 */

#include "Quaternion.h"
#include "math.h"
#include <assert.h>

Quaternion
  ::Quaternion(void)
    :Parent()
{
  ;
} // end Quaternion::Quaternion()

Quaternion
  ::Quaternion(const float4 &v)
    :Parent(v)
{
  ;
} // end Quaternion::Quaternion()

Quaternion
  ::Quaternion(const float angle, const float3 &axis)
{
  mElements[3] = cosf(angle/2.0f);
  float s = sinf(angle/2.0f);
  float3 unit = axis.normalize();
  mElements[0] = s * unit[0];
  mElements[1] = s * unit[1];
  mElements[2] = s * unit[2];
} // end Quaternion::Quaternion()

float4x4 Quaternion
  ::getRotationMatrix(void) const
{
  const Quaternion &q = *this;

  float4x4 result;
  float n=norm2();
  float s=n>0?2.0f/n:0.0f;
  float xs=q[0]*s;  float ys=q[1]*s;  float zs=q[2]*s;
  float wx=q[3]*xs; float wy=q[3]*ys; float wz=q[3]*zs;
  float xx=q[0]*xs; float xy=q[0]*ys; float xz=q[0]*zs;	
  float yy=q[1]*ys; float yz=q[1]*zs; float zz=q[2]*zs;	

  result(0,0) = 1.0f - (yy + zz);
  result(1,0) = xy - wz;
  result(2,0) = xz + wy;
  result(3,0) = 0.0f;
  
  result(0,1) = xy + wz;
  result(1,1) = 1.0f - (xx+ zz);
  result(2,1) = yz - wx;
  result(3,1) = 0.0f;
  
  result(0,2) = xz - wy;
  result(1,2) = yz + wx;
  result(2,2) = 1.0f - (xx + yy);
  result(3,2) = 0.0f;
  
  result(0,3) = 0.0f;
  result(1,3) = 0.0f;
  result(2,3) = 0.0f;
  result(3,3) = 1.0f;
  return result;
} // end Quaternion::getRotationMatrix()

Quaternion Quaternion
  ::operator*(const Quaternion &rhs) const
{
  const Quaternion &lhs = *this;
  float3 vLhs(lhs[0], lhs[1], lhs[2]);
  float3 vRhs(rhs[0], rhs[1], rhs[2]);

  float ts=lhs[3]*rhs[3]-vLhs.dot(vRhs);

  float3 tv=lhs[3]*vRhs + rhs[3]*vLhs + vLhs.cross(vRhs);

  Quaternion result;
  result[0] = tv[0];
  result[1] = tv[1];
  result[2] = tv[2];
  result[3] = ts;
  return result;
} // end Quaternion::operator*()

Quaternion &Quaternion
  ::operator*=(const Quaternion &rhs)
{
  const Quaternion &lhs = *this;
  float3 vLhs(lhs[0], lhs[1], lhs[2]);
  float3 vRhs(rhs[0], rhs[1], rhs[2]);

  float ts=lhs[3]*rhs[3]-vLhs.dot(vRhs);

  float3 tv=lhs[3]*vRhs + rhs[3]*vLhs + vLhs.cross(vRhs);
  mElements[0] = tv[0];
  mElements[1] = tv[1];
  mElements[2] = tv[2];
  mElements[3] = ts;
  return *this;
} // end Quaternion::operator*=()

