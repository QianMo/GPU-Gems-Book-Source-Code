/*! \file Vector.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Vector.h.
 */

#include "Vector.h"
//#include <boost/static_assert.hpp>
#include <math.h>
#include <limits>

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(void)
{
  ;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  template<typename CopyFromType>
    Vector<Scalar, N>
      ::Vector(const CopyFromType &v)
{
  for(int i = 0; i < N; ++i)
    mElements[i] = v[i];
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Scalar v)
{
  for(size_t i = 0; i < N; ++i)
    mElements[i] = v;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Scalar *v)
{
  memcpy(mElements, v, N*sizeof(Scalar));
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1)
{
  //BOOST_STATIC_ASSERT(N == 2);
  mElements[0] = v0;
  mElements[1] = v1;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1,
             const Scalar &v2)
{
  //BOOST_STATIC_ASSERT(N == 3);
  mElements[0] = v0;
  mElements[1] = v1;
  mElements[2] = v2;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1,
             const Scalar &v2,
             const Scalar &v3)
{
  //BOOST_STATIC_ASSERT(N == 4);
  mElements[0] = v0;
  mElements[1] = v1;
  mElements[2] = v2;
  mElements[3] = v3;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1,
             const Scalar &v2,
             const Scalar &v3,
             const Scalar &v4)
{
  //BOOST_STATIC_ASSERT(N == 5);
  mElements[0] = v0;
  mElements[1] = v1;
  mElements[2] = v2;
  mElements[3] = v3;
  mElements[4] = v4;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::Vector(const Vector<Scalar,N-1> &v,
             const Scalar &s)
{
  int i = 0;
  for(; i < N-1; ++i)
    mElements[i] = v[i];

  mElements[i] = s;
} // end Vector::Vector()

template<typename Scalar, unsigned int N>
  Scalar &Vector<Scalar,N>
    ::operator[](int i)
{
  return mElements[i];
} // end Vector::operator[]()

template<typename Scalar, unsigned int N>
  const Scalar &Vector<Scalar,N>
   ::operator[](int i) const
{
  return mElements[i];
} // end Vector::operator[]()

template<typename Scalar, unsigned int N>
  Scalar &Vector<Scalar,N>
    ::operator[](const unsigned int i)
{
  return mElements[i];
} // end Vector::operator[]()

template<typename Scalar, unsigned int N>
  const Scalar &Vector<Scalar,N>
   ::operator[](const unsigned int i) const
{
  return mElements[i];
} // end Vector::operator[]()

template<typename Scalar, unsigned int N>
  Vector<Scalar,N>
    ::operator Scalar * ()
{
  return mElements;
} // end Vector::operator Scalar * ()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    ::operator const Scalar * () const
{
  return mElements;
} // end Vector::operator Scalar * ()

template<typename Scalar, unsigned int N>
  Vector<Scalar,N> &Vector<Scalar, N>
    ::operator*=(const Scalar &s)
{
  for(unsigned int i = 0; i < N; ++i)
  {
    mElements[i] *= s;
  } // end for i

  return *this;
} // end Vector::operator*=()

template<typename Scalar, unsigned int N>
  Vector<Scalar,N> Vector<Scalar, N>
    ::operator*(const Scalar &rhs) const
{
  This result;
  for(unsigned int i = 0; i < N; ++i)
  {
    result[i] = mElements[i] * rhs;
  } // end for i

  return result;
} // end Vector::operator*()

template<typename Scalar, unsigned int N>
  Vector<Scalar,N> Vector<Scalar, N>
    ::operator*(const This &rhs) const
{
  This result;
  for(unsigned int i = 0; i < N; ++i)
  {
    result[i] = mElements[i] * rhs[i];
  } // end for i

  return result;
} // end Vector::operator*()

template<typename Scalar, unsigned int N>
  Vector<Scalar,N> &Vector<Scalar, N>
    ::operator*=(const This &rhs)
{
  for(unsigned int i = 0; i < N; ++i)
  {
    mElements[i] *= rhs[i];
  } // end for i

  return *this;
} // end Vector::operator*=()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::dot(const This &rhs) const
{
  Scalar result = 0;
  Scalar temp;
  for(unsigned int i = 0; i < N; ++i)
  {
    //result += this->operator[](i) * rhs.operator[](i);
    temp = this->operator[](i) * rhs.operator[](i);
    result += temp;
  } // end for i

  return result;
} // end Vector::dot()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::absDot(const This &rhs) const
{
  return fabs(dot(rhs));
} // end Vector::absDot()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::posDot(const This &rhs) const
{
  return std::max(Scalar(0), dot(rhs));
} // end Vector::posDot()

template<typename Scalar, unsigned int N>
  Vector<Scalar,N> Vector<Scalar, N>
    ::cross(const This &rhs) const
{
  //BOOST_STATIC_ASSERT(N == 3);
  This result;
  const This &lhs = *this;

  This subtractMe(lhs[2]*rhs[1], lhs[0]*rhs[2], lhs[1]*rhs[0]);

  result[0] = (lhs[1] * rhs[2]);
  result[0] -= subtractMe[0];
  result[1] = (lhs[2] * rhs[0]);
  result[1] -= subtractMe[1];
  result[2] = (lhs[0] * rhs[1]);
  result[2] -= subtractMe[2];

  return result;
} // end Vector::cross()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator+(const This &rhs) const
{
  This result;
  for(int i = 0; i < N; ++i)
  {
    result[i] = this->operator[](i) + rhs.operator[](i);
  } // end for i

  return result;
} // end Vector::operator+()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator+=(const This &rhs)
{
  for(int i = 0; i < N; ++i)
  {
    this->operator[](i) += rhs.operator[](i);
  } // end for i

  return *this;
} // end Vector::operator+()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
   ::operator/(const Scalar &rhs) const
{
  This result;
  for(int i = 0; i < N; ++i)
  {
    result[i] = this->operator[](i) / rhs;
  } // end for i

  return result;
} // end Vector::operator/()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator/=(const This &rhs)
{
  This &result = *this;
  for(int i = 0; i < N; ++i)
  {
    result[i] = this->operator[](i) / rhs[i];
  } // end for i

  return result;
} // end Vector::operator/=()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator/(const This &rhs) const
{
  This result = *this;
  return result /= rhs;
} // end Vector::operator/=()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator-(void) const
{
  This result;
  for(int i = 0; i < N; ++i)
  {
    result[i] = -(*this)[i];
  } // for i

  return result;
} // end Vector::operator-()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator-(const This &rhs) const
{
  This result;
  for(int i = 0; i < N; ++i)
  {
    result[i] = (*this)[i] - rhs[i];
  } // end for i

  return result;
} // end Vector::operator-()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::norm(void) const
{
  return Sqrt<Scalar>()(norm2());
} // end Vector::norm()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::length(void) const
{
  return norm();
} // end Vector::length()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::sum(void) const
{
  Scalar result = Scalar(0);
  for(int i = 0; i < N; ++i)
  {
    result += (*this)[i];
  } // end for i

  return result;
} // end Vector::sum()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::product(void) const
{
  Scalar result = Scalar(1);
  for(int i = 0; i < N; ++i)
  {
    result *= (*this)[i];
  } // end for i

  return result;
} // end Vector::product()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::norm2(void) const
{
  Scalar result = 0.0;
  for(int i = 0; i < N; ++i)
  {
    result += (*this)[i] * (*this)[i];
  } // end for i

  return result;
} // end Vector::norm2()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::normalize(void) const
{
  return (*this) / norm();
} // end Vector::normalize()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator/=(const Scalar &rhs)
{
  for(int i = 0; i < N; ++i)
  {
    (*this)[i] /= rhs;
  } // end for i

  return *this;
} // end for Vector::operator/=()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator-=(const This &rhs)
{
  for(int i = 0; i < N; ++i)
  {
    (*this)[i] -= rhs[i];
  } // end for i

  return *this;
} // end Vector::operator-=()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N>
    operator*(const Scalar &lhs, const Vector<Scalar,N> &rhs)
{
  Vector<Scalar, N> result;
  for(int i = 0; i < N; ++i)
  {
    result[i] = lhs * rhs[i];
  } // end for i

  return result;
} // end operator*()

template<typename Scalar, unsigned int N>
  std::ostream &operator<<(std::ostream &os,
                           const Vector<Scalar,N> &v)
{
  int i = 0;
  for(; i < N - 1; ++i)
  {
    os << v[i] << " ";
  } // end for i

  os << v[i];

  return os;
} // end operator<<()

template<typename Scalar, unsigned int N>
  std::istream &operator>>(std::istream &is,
                           Vector<Scalar,N> &v)
{
  for(int i = 0; i < N; ++i)
  {
    is >> v[i];
  } // end for i

  return is;
} // end operator>>()

template<typename Scalar, unsigned int N>
  unsigned int Vector<Scalar,N>
    ::numElements(void)
{
  return N;
} // end Vector::numElements()

template<typename Scalar, unsigned int N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::orthogonalVector(void) const
{
  //BOOST_STATIC_ASSERT(N == 3);

  int i = -1, j = -1, k = -1;

  // choose the minimal element
  if(fabs(mElements[0]) <= fabs(mElements[1]))
  {
    if(fabs(mElements[0]) <= fabs(mElements[2]))
    {
      // x is min
      k = 0;
      i = 1;
      j = 2;
    } // end if
    else
    {
      // z is min
      k = 2;
      i = 0;
      j = 1;
    } // end else
  } // end if
  else if(fabs(mElements[1]) <= fabs(mElements[2]))
  {
    // y is min
    k = 1;
    i = 0;
    j = 2;
  } // end else if
  else
  {
    // z is min
    k = 2;
    i = 0;
    j = 1;
  } // end else

  // supposing that y was the min, result would look like:
  // result = (z / sqrt(1.0 - y*y), 0, -x / sqrt(1.0 - y*y))
  Scalar denom = Sqrt<Scalar>()(Identity<Scalar>()()- mElements[k]*mElements[k]);

  This result;
  result[i] =  mElements[j] / denom;
  result[j] = -mElements[i] / denom;
  result[k] = 0.0;

  return result;
} // end Vector::orthogonalVector()

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::maxElement(void) const
{
  Scalar result = -std::numeric_limits<Scalar>::infinity();
  for(size_t i = 0; i < N; ++i)
  {
    result = mElements[i] > result ? mElements[i] : result;
  } // end for

  return result;
} // end Vector::maxElement()

template<typename Scalar, unsigned int N>
  const Scalar Vector<Scalar,N>::mOneOverN = static_cast<Scalar>(1.0) / N;

template<typename Scalar, unsigned int N>
  Scalar Vector<Scalar, N>
    ::mean(void) const
{
  Scalar result = 0;
  for(size_t i = 0; i < N; ++i)
  {
    result += mElements[i];
  } // end for

  return result * mOneOverN;
} // end Vector::mean()

template<typename Scalar, unsigned int N>
  typename Vector<Scalar, N>::This Vector<Scalar, N>
    ::reflect(const This &v) const
{
  This result = Scalar(2) * this->dot(v)*(*this) - v; 
  return result;
} // end Vector::reflect()

