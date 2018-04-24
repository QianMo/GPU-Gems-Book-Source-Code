/*! \file Matrix.h
 *  \author Jared Hoberock
 *  \brief Defines the interface for a matrix type
 *         templatized on type and dimension.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "Vector.h"

template<typename Scalar, unsigned int N, unsigned int M>
  class Matrix
{
  public:
    /*! \typedef This
     *  \brief Shorthand.
     */
    typedef Matrix<Scalar,N,M> This;

    /*! \fn Matrix
     *  \brief Null constructor does nothing.
     */
    inline Matrix(void);

    /*! \fn Matrix
     *  \brief Constructor initializes the upper left 3x3 subblock
     *         of this Matrix.
     */
    inline Matrix(const Scalar m00, const Scalar m01, const Scalar m02,
                  const Scalar m10, const Scalar m11, const Scalar m12,
                  const Scalar m20, const Scalar m21, const Scalar m22);

    /*! \fn Matrix
     *  \brief Constructor initializes the upper left 3x4 subblock
     *         of this Matrix.
     */
    inline Matrix(const Scalar m00, const Scalar m01, const Scalar m02, const Scalar m03,
                  const Scalar m10, const Scalar m11, const Scalar m12, const Scalar m13,
                  const Scalar m20, const Scalar m21, const Scalar m22, const Scalar m23);

    /*! \fn Matrix
     *  \brief Constructor initializes the upper left 4x4 subblock
     *         of this Matrix.
     */
    inline Matrix(const Scalar m00, const Scalar m01, const Scalar m02, const Scalar m03,
                  const Scalar m10, const Scalar m11, const Scalar m12, const Scalar m13,
                  const Scalar m20, const Scalar m21, const Scalar m22, const Scalar m23,
                  const Scalar m30, const Scalar m31, const Scalar m32, const Scalar m33);

    /*! \fn operator()
     *  \brief This method provides access to i,jth element.
     *  \param i Which row to select.
     *  \param j Which column to select.
     *  \return A reference to the i,jth element.
     */
    inline Scalar &operator()(const unsigned int i,
                              const unsigned int j);

    /*! \fn operator()
     *  \brief This method provides const access to i,jth element.
     *  \param i Which row to select.
     *  \param j Which column to select.
     *  \return A reference to the i,jth element.
     */
    inline const Scalar &operator()(const unsigned int i,
                                    const unsigned int j) const;

    /*! \fn operator const Scalar * ()
     *  \brief Cast to const Scalar * operator.
     *  \return Returns &mElements[0].
     */
    inline operator const Scalar *(void) const;

    /*! \fn operator Scalar *
     *  \brief Cast to Scalar * operator.
     *  \return Returns &mElements[0].
     */
    inline operator Scalar *(void);

    /*! \fn operator *
     *  \brief Matrix-Matrix multiplication.
     *  \param rhs The right hand side of the multiplication.
     *  \return (*this) * rhs
     */
    template<unsigned int P>
      inline Matrix<Scalar,N,P> operator*(const Matrix<Scalar,M,P> &rhs) const;

    /*! \fn operator/
     *  \brief Scalar division.
     *  \param rhs The right hand side of the operation.
     *  \return (*this) / rhs
     */
    This operator/(const Scalar &rhs) const;

    /*! \fn operator/=
     *  \brief Scalar divide equal.
     *  \param rhs The right hand side of the operation.
     *  \return *this
     */
    This &operator/=(const Scalar &rhs);

    /*! \fn transpose()
     *  \brief This method returns the transpose of this Matrix.
     *  \return The transpose of this Matrix.
     */
    inline Matrix<Scalar,M,N> transpose(void) const;

    /*! \fn inverse()
     *  \brief This method computes the inverse of this Matrix
     *         when N == M and N is small.
     *  \return The inverse of this Matrix.
     */
    inline This inverse(void) const;

    /*! \fn adjoint()
     *  \brief This method computes the adjoint of this Matrix
     *         when N == M and N is small.
     *  \return The adjoint of this Matrix.
     */
    inline This adjoint(void) const;

    /*! \fn operator*=
     *  \brief Scalar times equal.
     *  \param s The Scalar to multiply by.
     *  \return *this.
     */
    inline This &operator*=(const Scalar &s);

    /*! \fn identity
     *  \brief This static method returns an identity matrix.
     *  \return I
     */
    static This identity(void);

    /*! \fn minor
     *  \brief Helper function for adjoint when N is 3.
     */
    Scalar minor(const int r0, const int r1,
                 const int c0, const int c1) const;

    /*! \fn minor
     *  \brief Helper function for adjoint when N is 4.
     */
    Scalar minor(const int r0, const int r1, const int r2,
                 const int c0, const int c1, const int c2) const;

    /*! \typedef Array
     *  \brief Shorthand.
     */
    typedef Scalar Array[M*N];

    /*! This method provides access to the Array.
     *  \return mElements;
     *  FIXME: move this to the .inl.
     */
    inline Array &getData(void)
    {
      return mElements;
    } // end getArray()

    /*! This method multiplies a Vector by this Matrix.
     *  \param rhs The Vector to multiply by.
     *  \return (*this) * rhs
     */
    Vector<Scalar,N> operator*(const Vector<Scalar,M> &rhs) const;

  private:
    /*! \fn determinant
     *  \brief Helper function for determinant when N is small.
     *  \return det(*this)
     */
    Scalar determinant(void) const;

    Array mElements;
}; // end class Matrix

template<typename Scalar, unsigned int N>
  struct Determinant
{
};

template<typename Scalar>
  struct Determinant<Scalar,2>
{
  Scalar
    operator()(const Matrix<Scalar,2,2> &A) const
  {
    return A(0,0)*A(1,1) - A(0,1)*A(1,0);
  }
};

template<typename Scalar>
  struct Determinant<Scalar,3>
{
  Scalar
    operator()(const Matrix<Scalar,3,3> &A) const
  {
    return A(0,0) * A.minor(1, 2, 1, 2) -
           A(0,1) * A.minor(1, 2, 0, 2) +
           A(0,2) * A.minor(1, 2, 0, 1);
  }
};

template<typename Scalar>
  struct Determinant<Scalar,4>
{
  Scalar
    operator()(const Matrix<Scalar,4,4> &A) const
  {
    return A(0,0) * A.minor(1, 2, 3, 1, 2, 3) -
           A(0,1) * A.minor(1, 2, 3, 0, 2, 3) +
           A(0,2) * A.minor(1, 2, 3, 0, 1, 3) -
           A(0,3) * A.minor(1, 2, 3, 0, 1, 2);
  }
};

/*! XXX Hack: C++ doesn't allow partial specialization of
 *      template methods.  We work around it here.
 */
template<typename Scalar, unsigned int N, unsigned int M>
  struct Adjoint
{
};

template<typename Scalar>
  struct Adjoint<Scalar,2,2>
{
  Matrix<Scalar,2,2>
    operator()(const Matrix<Scalar,2,2> &A) const
  {
    Matrix<Scalar,2,2> result;
    result(0,0) =  A(1,1);
    result(0,1) = -A(0,1);
    result(1,0) = -A(1,0);
    result(1,1) =  A(0,0);
    return result;
  } // end operator()()
};

template<typename Scalar>
  struct Adjoint<Scalar,3,3>
{
  Matrix<Scalar,3,3>
    operator()(const Matrix<Scalar,3,3> &A) const
  {
    Matrix<Scalar,3,3> result;
    
    result(0,0) =  A.minor(1, 2, 1, 2);
    result(0,1) = -A.minor(0, 2, 1, 2);
    result(0,2) =  A.minor(0, 1, 1, 2);
    result(1,0) = -A.minor(1, 2, 0, 2);
    result(1,1) =  A.minor(0, 2, 0, 2);
    result(1,2) = -A.minor(0, 1, 0, 2);
    result(2,0) =  A.minor(1, 2, 0, 1);
    result(2,1) = -A.minor(0, 2, 0, 1);
    result(2,2) =  A.minor(0, 1, 0, 1);

    return result;
  } // end operator()()
};

template<typename Scalar>
  struct Adjoint<Scalar,4,4>
{
  Matrix<Scalar,4,4>
    operator()(const Matrix<Scalar,4,4> &A) const
  {
    Matrix<Scalar,4,4> result;

    result(0,0) =  A.minor(1, 2, 3, 1, 2, 3);
    result(0,1) = -A.minor(0, 2, 3, 1, 2, 3);
    result(0,2) =  A.minor(0, 1, 3, 1, 2, 3);
    result(0,3) = -A.minor(0, 1, 2, 1, 2, 3);
    result(1,0) = -A.minor(1, 2, 3, 0, 2, 3);
    result(1,1) =  A.minor(0, 2, 3, 0, 2, 3);
    result(1,2) = -A.minor(0, 1, 3, 0, 2, 3);
    result(1,3) =  A.minor(0, 1, 2, 0, 2, 3);
    result(2,0) =  A.minor(1, 2, 3, 0, 1, 3);
    result(2,1) = -A.minor(0, 2, 3, 0, 1, 3);
    result(2,2) =  A.minor(0, 1, 3, 0, 1, 3);
    result(2,3) = -A.minor(0, 1, 2, 0, 1, 3);
    result(3,0) = -A.minor(1, 2, 3, 0, 1, 2);
    result(3,1) =  A.minor(0, 2, 3, 0, 1, 2);
    result(3,2) = -A.minor(0, 1, 3, 0, 1, 2);
    result(3,3) =  A.minor(0, 1, 2, 0, 1, 2);

    return result;
  } // end operator()()
};

#include "Matrix.inl"

#endif // MATRIX_H

