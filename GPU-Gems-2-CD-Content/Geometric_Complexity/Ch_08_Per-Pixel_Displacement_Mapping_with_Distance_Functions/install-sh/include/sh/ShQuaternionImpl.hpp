// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
#include "ShQuaternion.hpp"

namespace SH {
template<ShBindingType B, typename T>
ShQuaternion<B, T>::ShQuaternion() 
{
  if (B == SH_TEMP) 
    {
      m_data = ShVector4f(1.0, 0.0, 0.0, 0.0);
      //m_data.setUnit(true);
    }
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>::ShQuaternion(const ShQuaternion<B2, T>& other)
  : m_data(other.getVector())
{
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>::ShQuaternion(const ShVector<4, B2, T>& values)
  : m_data(values)
{
}

template<ShBindingType B, typename T>
template<ShBindingType B2, ShBindingType B3>   
ShQuaternion<B, T>::ShQuaternion(const ShAttrib<1, B2, T>& angle, 
                                 const ShVector<3, B3, T>& axis)
{
  m_data(0) = cos(angle/2.0);
  m_data(1,2,3) = SH::normalize(axis);
  m_data(1,2,3) *= sin(angle/2.0);
  //m_data.setUnit(true);
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>::ShQuaternion(const ShMatrix<4, 4, B2, T>& mat)
{
  ShAttrib1f trace = 1.0 + mat[0](0) + mat[1](1) + mat[2](2);
  trace = (trace >= 0.0)*trace + (trace < 0.0)*0.0;
  ShAttrib1f c0 = (trace > 0.001);
  ShAttrib1f c1 = ((mat[0](0) > mat[1](1))*(mat[0](0) > mat[2](2)));
  ShAttrib1f c2 = ( mat[1](1) > mat[2](2) );
  ShVector4f res0, res1, res2, res3;
  ShAttrib1f S0 = sqrt(trace) * 2.0;
  S0 += (S0 == 0.0)*1.0;
  
  res0(0) = 0.25 * S0;
  res0(1) = (mat[1](2) - mat[2](1)) / S0;
  res0(2) = (mat[2](0) - mat[0](2)) / S0;
  res0(3) = (mat[0](1) - mat[1](0)) / S0;
  
  trace = 1.0 + mat[0](0) - mat[1](1) - mat[2](2);
  trace = (trace >= 0.0)*trace + (trace < 0.0)*0.0;
  ShAttrib1f S1 = sqrt(trace) * 2.0;
  S1 += (S1 == 0.0)*1.0;
  
  res1(0) = (mat[2](1) - mat[1](2)) / S1;
  res1(1) = 0.25 * S1;
  res1(2) = (mat[0](1) + mat[1](0)) / S1;
  res1(3) = (mat[2](0) + mat[0](2)) / S1;
  
  trace = 1.0 - mat[0](0) + mat[1](1) - mat[2](2);
  trace = (trace >= 0.0)*trace + (trace < 0.0)*0.0;
  ShAttrib1f S2 = sqrt(trace) * 2.0;
  S2 += (S2 == 0.0)*1.0;
  
  res2(0) = (mat[2](0) - mat[0](2)) / S2;
  res2(1) = (mat[0](1) + mat[1](0)) / S2;
  res2(2) = 0.25 * S2;
  res2(3) = (mat[1](2) + mat[2](1)) / S2;
  
  trace = 1.0 - mat[0](0) - mat[1](1) + mat[2](2);
  trace = (trace >= 0.0)*trace + (trace < 0.0)*0.0;
  ShAttrib1f S3 = sqrt(trace) * 2.0;
  S3 += (S3 == 0.0)*1.0;
  
  res3(0) = (mat[1](0) - mat[0](1)) / S3;
  res3(1) = (mat[2](0) + mat[0](2)) / S3;
  res3(2) = (mat[1](2) + mat[2](1)) / S3;
  res3(3) = 0.25 * S3;
  
  m_data = c0*res0 + 
    (c0 == 0.0)*(c1*res1 + (c1 == 0.0)*(c2*res2 + (c2 == 0.0)*res3));
  //m_data.setUnit(true);
}

template<ShBindingType B, typename T>
std::ostream& operator<<(std::ostream& out, const ShQuaternion<B, T>& q)
{
  float vals[4];
  q.m_data.getValues(vals);
  out << "ShQuaternion: " << vals[0] << " " << vals[1] << " " << vals[2] 
      << " " << vals[3];
  return out;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator=(const ShQuaternion<B2, T>& other) 
{
  m_data = other.getVector();
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator+=(const ShQuaternion<B2, T>& right) 
{
  m_data += right.getVector();
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator-=(const ShQuaternion<B2, T>& right) 
{
  m_data -= right.getVector();
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator*=(const ShQuaternion<B2, T>& right) 
{
  ShVector4f result;
  ShVector4f rightData = right.getVector();
  result(0) = 
    m_data(0)*rightData(0) - SH::dot(m_data(1,2,3), rightData(1,2,3));
  result(1,2,3) = 
    m_data(0)*rightData(1,2,3) + rightData(0)*m_data(1,2,3) + 
    cross(m_data(1,2,3), rightData(1,2,3));

  //result.setUnit(m_data.isUnit() && rightData.isUnit());
  m_data = result;
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator*=(const ShAttrib<1, B2, T>& right) 
{
  m_data = m_data*right;
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator*=(const ShVector<3, B2, T>& right) 
{
  ShVector4f v;
  v(0) = 0.0;
  v(1,2,3) = right;
  //v.setUnit(right.isUnit());
  *this *= ShQuaternionf(v);
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<B, T>& 
ShQuaternion<B, T>::operator*=(const ShNormal<3, B2, T>& right) 
{
  ShVector4f v;
  v(0) = 0.0;
  v(1,2,3) = right;
  //v.setUnit(right.isUnit());
  *this *= ShQuaternionf(v);
  return *this;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShAttrib<1, SH_TEMP, T> 
ShQuaternion<B, T>::dot(const ShQuaternion<B2, T>& q) const 
{
  return SH::dot(m_data, q.getVector());
}

template<ShBindingType B, typename T>
ShQuaternion<SH_TEMP, T> ShQuaternion<B, T>::conjugate() const 
{
  ShVector4f conjData;
  conjData(0) = m_data(0);
  conjData(1, 2, 3) = -m_data(1, 2, 3);
  //conjData.setUnit(m_data.isUnit());

  return ShQuaternion<SH_TEMP>(conjData);
}

template<ShBindingType B, typename T>
ShQuaternion<SH_TEMP, T> ShQuaternion<B, T>::inverse() const 
{
  //  if (m_data.isUnit()) {
  //    return conjugate();
  //  } else {
  ShAttrib1f norm = SH::dot(m_data, m_data); 
  return conjugate() * (1.0 / norm);
  //  }
}

template<ShBindingType B, typename T>
ShMatrix<4, 4, SH_TEMP, T> ShQuaternion<B, T>::getMatrix() const
{
  SH::ShMatrix4x4f m;
  ShAttrib4f x = m_data(1,1,1,1) * m_data(1,2,3,0);
  ShAttrib4f y = m_data(2,2,2,2) * m_data(0,2,3,0);
  ShAttrib4f z = m_data(3,3,3,3) * m_data(0,0,3,0);

  m[0](0) = 1 - 2 * (y(1) + z(2));
  m[1](0) = 2 * (x(1) - z(3));
  m[2](0) = 2 * (x(2) + y(3));

  m[0](1) = 2 * (x(2) + z(3));
  m[1](1) = 1 - 2 * (x(0) + z(2));
  m[2](1) = 2 * (y(2) - x(3));

  m[0](2) = 2 * (x(2) - y(3));
  m[1](2) = 2 * (y(2) + x(3));
  m[2](2) = 1 - 2 * (x(0) + y(1));

  return m;
}

template<ShBindingType B, typename T>
ShVector<4, SH_TEMP, T> ShQuaternion<B, T>::getVector() const
{
  return m_data;
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
ShQuaternion<B, T>::operator+(const ShQuaternion<B2, T>& q)
{
  ShQuaternion<B, T> r = *this;
  return (r += q);
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
ShQuaternion<B, T>::operator-(const ShQuaternion<B2, T>& q)
{
  ShQuaternion<B, T> r = *this;
  return (r -= q);
}
  
template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
ShQuaternion<B, T>::operator*(const ShQuaternion<B2, T>& q)
{
  ShQuaternion<B, T> r = *this;
  return (r *= q);
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
ShQuaternion<B, T>::operator*(const ShAttrib<1, B2, T>& c)
{
  ShQuaternion<B, T> r = *this;
  return (r *= c);
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
ShQuaternion<B, T>::operator*(const ShVector<3, B2, T>& v)
{
  ShQuaternion<B, T> r = *this;
  return (r *= v);
}

template<ShBindingType B, typename T>
template<ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
ShQuaternion<B, T>::operator*(const ShNormal<3, B2, T>& v)
{
  ShQuaternion<B, T> r = *this;
  return (r *= v);
}

template<ShBindingType B, typename T>
void ShQuaternion<B, T>::normalize()
{
  m_data = SH::normalize(m_data);
}

template<ShBindingType B, typename T>
void ShQuaternion<B, T>::setUnit(bool flag)
{
  //m_data.setUnit(flag);
}

template<ShBindingType B, typename T>
void ShQuaternion<B, T>::getValues(HostType values[]) const
{
  m_data.getValues(values);
}

template<ShBindingType B, typename T, ShBindingType B2>
ShQuaternion<SH_TEMP, T> 
operator*(const ShAttrib<1, B2, T>& c, const ShQuaternion<B, T>& q)
{
  ShQuaternion<B, T> r = q;
  return (r *= c);
}

template<ShBindingType B1, ShBindingType B2, typename T>
extern ShQuaternion<SH_TEMP, T>
slerp(const ShQuaternion<B1, T>& q1, const ShQuaternion<B2, T>& q2, 
      const ShAttrib1f& t)
{
  //TODO::q1 and q2 must be unit quaternions, we cannot call normalize here
  //since it's not a const function.
  //TODO: when cosTheta is 1 or -1, we need to fallback to linear interpolation
  //not sure how to implement this efficiently yet
  ShAttrib<1, SH_TEMP, T> cosTheta = q1.dot(q2);
  ShAttrib<1, SH_TEMP, T> sinTheta = sqrt(1.0 - cosTheta*cosTheta);
  
  ShQuaternion<B2, T> q2prime = (cosTheta >= 0.0)*q2 - (cosTheta < 0.0)*q2;
  ShAttrib<1, SH_TEMP, T> theta = asin(sinTheta);

  return (sin((1.0 - t)*theta)/sinTheta)*q1 + (sin(t*theta)/sinTheta)*q2prime;
}

template<ShBindingType B, typename T>
std::string ShQuaternion<B, T>::name() const
{
  return m_data.name();
}

template<ShBindingType B, typename T>
void ShQuaternion<B, T>::name(const std::string& name)
{
  m_data.name(name);
}


}
