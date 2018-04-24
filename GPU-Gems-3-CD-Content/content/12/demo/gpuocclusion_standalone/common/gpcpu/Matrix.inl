/*! \file Matrix.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Matrix.h.
 */

#include "Matrix.h"
#include <algorithm>

template<typename Scalar, unsigned int N, unsigned int M>
  Scalar &Matrix<Scalar,N,M>
    ::operator()(const unsigned int i,
                 const unsigned int j)
{
  return mElements[i*M + j];
} // end Matrix::operator()()

template<typename Scalar, unsigned int N, unsigned int M>
  const Scalar &Matrix<Scalar,N,M>
    ::operator()(const unsigned int i,
                 const unsigned int j) const
{
  return mElements[i*M + j];
} // end Matrix::operator()()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M>
    ::operator const Scalar * () const
{
  return mElements;
} // end Matrix::operator const Scalar * ()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M>
   ::operator Scalar * ()
{
  return mElements;
} // end Matrix::operator Scalar * ()

template<typename Scalar, unsigned int N, unsigned int M>
  template<unsigned int P>
    Matrix<Scalar,N,P> Matrix<Scalar,N,M>
      ::operator*(const Matrix<Scalar,M,P> &rhs) const
{
  Matrix<Scalar,N,P> result;
  const Matrix<Scalar,N,M> &lhs = *this;

  Scalar temp;
  // iterate over elements of the result:
  for(unsigned int i = 0;
      i < N;
      ++i)
  {
    for(unsigned int j = 0;
        j < P;
        ++j)
    {
      // dot the ith row of lhs
      // with the jth column of rhs
      temp = 0;
      for(unsigned int k = 0;
          k < M;
          ++k)
      {
        temp += lhs(i,k) * rhs(k,j);
      } // end for k

      result(i,j) = temp;
    } // end for j
  } // end for i

  return result;
} // end Matrix::operator*()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,M,N> Matrix<Scalar,N,M>
    ::transpose(void) const
{
  Matrix<Scalar,M,N> result;

  for(unsigned int i = 0;
      i < M;
      ++i)
  {
    for(unsigned int j = 0;
        j < N;
        ++j)
    {
      result(i,j) = operator()(j,i);
    } // end for j
  } // end for i

  return result;
} // end Matrix::transpose()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M> &Matrix<Scalar,N,M>
    ::operator*=(const Scalar &s)
{
  for(unsigned int i = 0; i < N*M; ++i)
  {
    mElements[i] *= s;
  } // end for i

  return *this;
} // end Matrix::operator*=()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M> Matrix<Scalar,N,M>
    ::inverse(void) const
{
  return adjoint() / determinant();
} // end Matrix::inverse()

template<typename Scalar, unsigned int N, unsigned int M>
  Scalar Matrix<Scalar,N,M>
    ::minor(const int r0, const int r1, const int r2,
            const int c0, const int c1, const int c2) const
{
  const This &A = *this;
  return A(r0,c0) * (A(r1,c1) * A(r2,c2) - A(r2,c1) * A(r1,c2)) -
	 A(r0,c1) * (A(r1,c0) * A(r2,c2) - A(r2,c0) * A(r1,c2)) +
	 A(r0,c2) * (A(r1,c0) * A(r2,c1) - A(r2,c0) * A(r1,c1));
} // end Matrix::minor()

template<typename Scalar, unsigned int N, unsigned int M>
  Scalar Matrix<Scalar,N,M>
    ::minor(const int r0, const int r1,
            const int c0, const int c1) const
{
  const This &A = *this;
  return A(r0,c0) * A(r1,c1) - A(r1,c0) * A(r0,c1);
} // end Matrix::minor()

template<typename Scalar, unsigned int N, unsigned int M>
  Scalar Matrix<Scalar,N,M>
    ::determinant(void) const
{
  return Determinant<Scalar,N>()(*this);
} // end Matrix::determinant()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M> Matrix<Scalar,N,M>
    ::adjoint(void) const
{
  return Adjoint<Scalar,N,M>()(*this);
} // end Matrix::adjoint()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar, N, M> Matrix<Scalar,N,M>
    ::identity(void)
{
  This I;

  std::fill(&I.mElements[0], &I.mElements[N*M], static_cast<Scalar>(0.0));
  for(unsigned int i = 0; i < N; ++i)
  {
    I(i,i) = static_cast<Scalar>(1.0);
  } // end for i

  return I;
} // end Matrix::identity()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar, N, M> &Matrix<Scalar,N,M>
    ::operator/=(const Scalar &rhs)
{
  Scalar recip = static_cast<Scalar>(1.0) / rhs;

  return operator*=(recip);
} // end Matrix::operator/=()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M> Matrix<Scalar,N,M>
    ::operator/(const Scalar &rhs) const
{
  This result = *this;
  return result/=rhs;
} // end Matrix::operator/()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M>
    ::Matrix(void)
{
  ;
} // end Matrix::Matrix()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M>
    ::Matrix(const Scalar m00, const Scalar m01, const Scalar m02,
             const Scalar m10, const Scalar m11, const Scalar m12,
             const Scalar m20, const Scalar m21, const Scalar m22)
{
  (*this)(0,0) = m00; (*this)(0,1) = m01; (*this)(0,2) = m02;
  (*this)(1,0) = m10; (*this)(1,1) = m11; (*this)(1,2) = m12;
  (*this)(2,0) = m20; (*this)(2,1) = m21; (*this)(2,2) = m22;
} // end Matrix::Matrix()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M>
    ::Matrix(const Scalar m00, const Scalar m01, const Scalar m02, const Scalar m03,
             const Scalar m10, const Scalar m11, const Scalar m12, const Scalar m13,
             const Scalar m20, const Scalar m21, const Scalar m22, const Scalar m23)
{
  (*this)(0,0) = m00; (*this)(0,1) = m01; (*this)(0,2) = m02; (*this)(0,3) = m03;
  (*this)(1,0) = m10; (*this)(1,1) = m11; (*this)(1,2) = m12; (*this)(1,3) = m13;
  (*this)(2,0) = m20; (*this)(2,1) = m21; (*this)(2,2) = m22; (*this)(2,3) = m23;
} // end Matrix::Matrix()

template<typename Scalar, unsigned int N, unsigned int M>
  Matrix<Scalar,N,M>
    ::Matrix(const Scalar m00, const Scalar m01, const Scalar m02, const Scalar m03,
             const Scalar m10, const Scalar m11, const Scalar m12, const Scalar m13,
             const Scalar m20, const Scalar m21, const Scalar m22, const Scalar m23,
             const Scalar m30, const Scalar m31, const Scalar m32, const Scalar m33)
{
  (*this)(0,0) = m00; (*this)(0,1) = m01; (*this)(0,2) = m02; (*this)(0,3) = m03;
  (*this)(1,0) = m10; (*this)(1,1) = m11; (*this)(1,2) = m12; (*this)(1,3) = m13;
  (*this)(2,0) = m20; (*this)(2,1) = m21; (*this)(2,2) = m22; (*this)(2,3) = m23;
  (*this)(3,0) = m30; (*this)(3,1) = m31; (*this)(3,2) = m32; (*this)(3,3) = m33;
} // end Matrix::Matrix()

template<typename Scalar, unsigned int N, unsigned int M>
  Vector<Scalar,N> Matrix<Scalar,N,M>
    ::operator*(const Vector<Scalar,M> &rhs) const
{
  Vector<Scalar,N> result(Scalar(0));

  for(unsigned int j = 0; j < N; ++j)
  {
    for(unsigned int i = 0; i < M; ++i)
    {
      result[j] += (*this)(j,i) * rhs[i];
    } // end for i
  } // end for j

  return result;
} // end Vector::operator*()

