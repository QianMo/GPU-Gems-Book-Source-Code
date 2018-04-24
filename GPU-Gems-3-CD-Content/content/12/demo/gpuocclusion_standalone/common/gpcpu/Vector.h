/*! \file Vector.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a mathematical
 *         vector class.
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <math.h>

template<typename S, unsigned int N>
  class Vector
{
  public:
    /*! \typedef This
     *  \brief Shorthand.
     */
    typedef Vector<S,N> This;

    /*! \typedef Scalar
     *  \brief Shorthand.
     */
    typedef S Scalar;

    /*! \fn Vector
     *  \brief Null constructor does nothing.
     */
    inline Vector(void);

    /*! \fn Vector
     *  \brief Templated universal copy constructor
     *         accepts anything that can be indexed.
     *  \param v Something to copy from.
     */
    template<typename CopyFromType>
      inline Vector(const CopyFromType &v);

    /*! \fn Vector
     *  \brief Constructor takes a const pointer to
     *         an N-length array of Scalars.
     *  \param v A pointer to an N-length array of
     *           Scalars to copy from.
     */
    inline Vector(const Scalar *v);

    /*! \fn Vector
     *  \brief This method sets every element of this Vector
     *         to the given value.
     *  \param v The fill value.
     */
    inline Vector(const Scalar v);

    /*! \fn Vector
     *  \brief Special constructor for 2-vectors.
     *  \param s0 The first element.
     *  \param s1 The second element.
     */
    inline Vector(const Scalar &v0,
                  const Scalar &v1);

    /*! \fn Vector
     *  \brief Special constructor for 3-vectors.
     *  \param s0 The first element.
     *  \param s1 The second element.
     *  \param s2 The third element.
     */
    inline Vector(const Scalar &v0,
                  const Scalar &v1,
                  const Scalar &v2);

    /*! \fn Vector
     *  \brief Special constructor for 4-vectors.
     *  \param s0 The first element.
     *  \param s1 The second element.
     *  \param s2 The third element.
     *  \param s3 The fourth element.
     */
    inline Vector(const Scalar &v0,
                  const Scalar &v1,
                  const Scalar &v2,
                  const Scalar &v3);

    /*! \fn Vector
     *  \brief Special constructor for 5-vectors.
     *  \param s0 The first element.
     *  \param s1 The second element.
     *  \param s2 The third element.
     *  \param s3 The fourth element.
     *  \param s4 The fifth element.
     */
    inline Vector(const Scalar &v0,
                  const Scalar &v1,
                  const Scalar &v2,
                  const Scalar &v3,
                  const Scalar &v4);

    /*! \fn Vector
     *  \brief Constructor accepts a smaller Vector
     *         and a final Scalar.
     *  \param v The first N-1 Scalars.
     *  \param s The Nth Scalar.
     */
    inline Vector(const Vector<Scalar,N-1> &v,
                  const Scalar &s);

    /*! \fn operator Scalar *
     *  \brief Cast to Scalar * operator.
     *  \return Returns &mElements[0].
     */
    inline operator Scalar *(void);

    /*! \fn operator const Scalar * ()
     *  \brief Cast to const Scalar * operator.
     *  \return Returns &mElements[0].
     */
    inline operator const Scalar *(void) const;

    /*! \fn operator[]
     *  \brief This method provides access to the ith element.
     *  \param i Which element to select.
     *  \return A reference to the ith element.
     */
    inline Scalar &operator[](int i);

    /*! \fn operator[]
     *  \brief This method provides access to the ith element.
     *  \param i Which element to select.
     *  \return A reference to the ith element.
     */
    inline const Scalar &operator[](int i) const;

    /*! \fn operator[]
     *  \brief This method provides access to the ith element.
     *  \param i Which element to select.
     *  \return A reference to the ith element.
     */
    inline Scalar &operator[](const unsigned int i);

    /*! \fn operator[]
     *  \brief This method provides access to the ith element.
     *  \param i Which element to select.
     *  \return A reference to the ith element.
     */
    inline const Scalar &operator[](const unsigned int i) const;

    /*! \fn operator+
     *  \brief Addition operator.
     *  \return Returns (*this) + rhs
     */
    inline This operator+(const This &rhs) const;

    /*! \fn operator+=
     *  \brief Plus equal operator.
     *  \param rhs The right hand side of the relation.
     *  \return *this
     */
    inline This &operator+=(const This &rhs);

    /*! \fn operator*=
     *  \brief Scalar times equal.
     *  \param s The Scalar to multiply by.
     *  \return *this.
     */
    inline This &operator*=(const Scalar &s);

    /*! \fn operator/
     *  \brief Scalar divide.
     *  \param rhs The Scalar to divide by.
     *  \return (*this) / s
     */
    inline This operator/(const Scalar &rhs) const;

    /*! \fn operator/=
     *  \brief Scalar divide equal.
     *  \param rhs The right hand side of the operation.
     *  \return (*this)
     */
    inline This &operator/=(const Scalar &rhs);

    /*! \fn operator/=
     *  \brief Vector component-wise divide equal.
     *  \param rhs The vector to divide by.
     *  \return (*this) / rhs
     */
    inline This &operator/=(const Vector &rhs);

    /*! \fn operator/
     *  \brief Vector component-wise divide.
     *  \param rhs The vector to divide by.
     *  \return (*this) / rhs
     */
    inline This operator/(const Vector &rhs) const;

    /*! \fn operator*
     *  \brief Vector component-wise mutliply
     *  \param rhs The vector to multiply by.
     *  \return (*this) * rhs
     */
    inline This operator*(const This &rhs) const;

    /*! \fn operator*
     *  \brief Scalar multiply.
     *  \param rhs The Scalar to multiply by.
     *  \return (*this) * rhs
     */
    inline This operator*(const Scalar &rhs) const;

    /*! \fn operator*
     *  \brief Vector component-wise mutliply equal.
     *  \param rhs The vector to multiply by.
     *  \return *this. 
     */
    inline This &operator*=(const This &rhs);

    /*! \fn dot
     *  \brief Dot product
     *  \param rhs The vector to dot by.
     *  \return (*this) dot rhs
     */
    inline Scalar dot(const This &rhs) const;

    /*! \fn absDot
     *  \brief Absolute value of dot product.
     *  \param rhs The vector to dot by.
     *  \return |(*this) dot rhs)|
     */
    inline Scalar absDot(const This &rhs) const;

    /*! \fn posDot
     *  \brief Dot product where negative values
     *         are clamped to 0.
     *  \param rhs The vector to dot by.
     *  \return max(0, (*this) dot rhs)
     */
    inline Scalar posDot(const This &rhs) const;

    /*! \fn cross
     *  \brief Cross product
     *  \param rhs The rhs of the operation.
     *  \return (*this) cross rhs
     *  \note This in only implemented for 3-vectors.
     */
    inline This cross(const This &rhs) const;

    /*! \fn operator-
     *  \brief Unary negation.
     *  \return -(*this)
     */
    inline This operator-(void) const;

    /*! \fn operator-
     *  \brief Binary minus.
     *  \param rhs The right hand side of the operation.
     *  \return (*this) - rhs
     */
    inline This operator-(const This &rhs) const;

    /*! \fn operator-=
     *  \brief Decrement equal.
     *  \param rhs The right hand side of the operation.
     *  \return *this
     */
    inline This &operator-=(const This &rhs);

    /*! \fn norm
     *  \brief This method returns the 2-norm of this
     *         Vector.
     *  \return As above.
     */
    inline Scalar norm(void) const;

    /*! \fn length
     *  \brief Alias for norm().
     *  \return norm().
     */
    inline Scalar length(void) const;

    /*! \fn norm2
     *  \brief This method returns the squared 2-norm of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar norm2(void) const;

    /*! \fn sum
     *  \brief This method returns the sum of elements of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar sum(void) const;

    /*! \fn product
     *  \brief This method returns the product of elements of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar product(void) const;

    /*! \fn max
     *  \brief This method returns the maximum element of this
     *         Vector.
     *  \return The maximum element of this Vector.
     */
    inline Scalar maxElement(void) const;

    /*! \fn mean
     *  \brief This method returns the mean of the elements of
     *         this Vector.
     *  \return The mean of this Vector's elements.
     */
    inline Scalar mean(void) const;

    /*! \fn normalize
     *  \brief This method returns a normalized version
     *         of this Vector.
     *  \return (*this) / this->norm()
     */
    inline This normalize(void) const;

    /*! This method returns the dimension of this Vector.
     *  \return N
     */
    inline static unsigned int numElements(void);

    /*! This method returns a Vector orthogonal to this
     *  Vector
     *  \return A Vector v such that this->dot(v) is small.
     *  \note This method is only valid for 3-Vectors.
     */
    This orthogonalVector(void) const;

    /*! This method reflects this Vector about another.
     *  \param v The Vector about which to reflect.
     *  \return The reflection of this Vector about v.
     */
    This reflect(const Vector &v) const;

  protected:
    Scalar mElements[N];

    const static Scalar mOneOverN;
}; // end class Vector

/*! \fn operator*
 *  \brief Scalar left-multiply.
 *  \param lhs The left hand side of the operation.
 *  \param rhs The right hand side of the operation.
 *  \return lhs * rhs
 */
template<typename Scalar, unsigned int N>
  Vector<Scalar, N> operator*(const Scalar &lhs, const Vector<Scalar,N> &rhs);

/*! \fn operator<<
 *  \brief Output to ostream operator.
 *  \param os The ostream to put to.
 *  \param v The Vector to output.
 *  \return os
 */
template<typename Scalar, unsigned int N>
  std::ostream &operator<<(std::ostream &os, const Vector<Scalar,N> &v);

/*! \fn operator>>
 *  \brief Input from istream operator.
 *  \param is The istream to input from.
 *  \param v The Vector to input.
 *  \return is
 */
template<typename Scalar, unsigned int N>
  std::istream &operator>>(std::istream &is, Vector<Scalar,N> &v);

/*! \class Identity
 *  \brief This is a templated functor which
 *         returns the identity of a Scalar type.
 */
template<typename Scalar>
  class Identity
{
}; // end Identity

template<>
  class Identity<float>
{
  public:
    float operator()(void) const
    {
      return 1.0f;
    } // end operator()()
}; // end Identity

template<>
  class Identity<double>
{
  public:
    double operator()(void) const
    {
      return 1.0;
    } // end operator()()
}; // end Identity

/*! \class Sqrt
 *  \brief This is a templated functor which
 *         returns the square root of a scalar.
 */
template<typename Scalar>
  class Sqrt
{
}; // end class Sqrt

template<>
  class Sqrt<float>
{
  public:
    float operator()(const float &f) const
    {
      return sqrtf(f);
    } // end operator()()
}; // end class Sqrt

template<>
  class Sqrt<double>
{
  public:
    double operator()(const double &d) const
    {
      return sqrt(d);
    } // end operator()()
}; // end class Sqrt

#include "Vector.inl"

typedef Vector<int, 2> int2;
typedef Vector<int, 3> int3;
typedef Vector<int, 4> int4;

typedef Vector<unsigned int, 2> uint2;
typedef Vector<unsigned int, 3> uint3;
typedef Vector<unsigned int, 4> uint4;

typedef Vector<unsigned char, 2> uchar2;
typedef Vector<unsigned char, 3> uchar3;
typedef Vector<unsigned char, 4> uchar4;

/*! \typedef float2
 *  \brief Shorthand.
 */
typedef Vector<float, 2> float2;

/*! \typedef float3
 *  \brief Shorthand.
 */
typedef Vector<float, 3> float3;

/*! \typedef float4
 *  \brief Shorthand.
 */
typedef Vector<float, 4> float4;

/*! \typedef float5
 *  \brief Shorthand.
 */
typedef Vector<float, 5> float5;

/*! \typedef double2
 *  \brief Shorthand.
 */
typedef Vector<double, 2> double2;

/*! \typedef double3
 *  \brief Shorthand.
 */
typedef Vector<double, 3> double3;

/*! \typedef double4
 *  \brief Shorthand.
 */
typedef Vector<double, 4> double4;

/*! \typedef double5
 *  \brief Shorthand.
 */
typedef Vector<double, 5> double5;

#endif // VECTOR_H

