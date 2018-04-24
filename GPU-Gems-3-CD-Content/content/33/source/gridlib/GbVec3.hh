/*----------------------------------------------------------------------
|
| $Id: GbVec3.hh,v 1.16 2003/03/06 17:01:52 prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBVEC3_HH
#define  GBVEC3_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbMath.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief Simple 3D point with arithmetic operators.
  \author Peter Kipfer
  $Revision: 1.16 $
  $Date: 2003/03/06 17:01:52 $
  \note Original Source: Vis3D class from the Vision project (c) 1993 Philipp Slusallek

  \note This is a version modified for GPUGems from the original gridlib file

  This class represents a simple point in 3D space. It has access methods
  to treat it like a vector, comparison operators that support fuzzy logic,
  arithmetic operators for all standard operations, operators for
  scalar and vector product and some special operators for building an
  orthogonal basis.

  \warning This class template is intended to be parametrized with
           floating point data types. Look for the special treatment of
           integral numbers in the operators. 

  \sa GbVec2 GbVec3H GbVec3i GbVec4 GbVecN
 */
template <class T>
class GRIDLIB_API GbVec3
{
public:
  /** @name Construction and Destruction */
  //@{

  //! Constructor for uninitialized vector
  GbVec3();
  //! Constructor for initialized vector
  explicit GbVec3(T s);
  //! Constructor for explicitly initialized vector
  GbVec3(T x, T y, T z);
  //! Constructor for explicitly initialized vector
  GbVec3(const T p[3]);
  //! Constructor for explicitly initialized vector
  GbVec3(const GbVec3<T> &p);

  //! Destructor
  ~GbVec3();

  //@}

  /** @name Binary operators */
  //@{

  //! Equality test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator==(const GbVec3<T>& p) const;
  //! Inequality test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator!=(const GbVec3<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator< (const GbVec3<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator<=(const GbVec3<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator> (const GbVec3<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator>=(const GbVec3<T>& p) const;
  
  //! Componentwise addition
  INLINE GbVec3<T>& operator+=(const GbVec3<T>& p);
  //! Addition of a scalar to each component
  INLINE GbVec3<T>& operator+=(const T& s);
  //! Componentwise subtraction
  INLINE GbVec3<T>& operator-=(const GbVec3<T>& p);
  //! Subtraction of a scalar from each component
  INLINE GbVec3<T>& operator-=(const T& s);
  //! Componentwise multiplication
  INLINE GbVec3<T>& operator*=(const GbVec3<T>& p);
  //! Scaling with integral value
  INLINE GbVec3<T>& operator*=(const int& s);
  //! Scaling with floating point value
  INLINE GbVec3<T>& operator*=(const T& s);
  //! Componentwise division
  INLINE GbVec3<T>& operator/=(const GbVec3<T>& p);
  //! Inverse scaling with integral value
  INLINE GbVec3<T>& operator/=(const int& s);
  //! Inverse scaling with floating point value
  INLINE GbVec3<T>& operator/=(const T& s);

  //! Componentwise addition
  INLINE GbVec3<T>  operator+(const GbVec3<T>& p) const;
  //! Addition of a scalar to each component
  INLINE GbVec3<T>  operator+(const T& p) const;
  //! Componentwise subtraction
  INLINE GbVec3<T>  operator-(const GbVec3<T>& p) const;
  //! Subtraction of a scalar from each component
  INLINE GbVec3<T>  operator-(const T& p) const;
  //! Componentwise multiplication
  INLINE GbVec3<T>  operator*(const GbVec3<T>& p) const;
  //! Scaling with integral value
  INLINE GbVec3<T>  operator*(const int& s) const;
  //! Scaling with floating point value
  INLINE GbVec3<T>  operator*(const T& s) const;
  //! Componentwise division
  INLINE GbVec3<T>  operator/(const GbVec3<T>& p) const;
  //! Inverse scaling with integral value
  INLINE GbVec3<T>  operator/(const int& s) const;
  //! Inverse scaling with floating point value
  INLINE GbVec3<T>  operator/(const T& s) const;

  //@}

  //! Scaling with floating point value as a prefix
  //friend GRIDLIB_API GbVec3<T>  operator* GB_TEXPORT (T s, const GbVec3<T>& a);

  //! Vector inversion
  INLINE GbVec3<T>  operator-() const;

  /** @name Vector products */
  //@{
  
  //! Vector product (cross product)
  INLINE GbVec3<T>  operator^(const GbVec3<T>& a) const;
  //! Vector product (cross product)
  INLINE GbVec3<T>  cross(const GbVec3<T> &a) const;
  //! Vector product (cross product) with normalization
  INLINE GbVec3<T>  unitCross(const GbVec3<T>& a) const;

  //! Scalar product (dot product)
  INLINE        T operator|(const GbVec3<T> &a) const;
  //! Scalar product (dot product)
  INLINE        T dot(const GbVec3<T> &a) const;

  //@}

  /** @name Access methods */
  //@{

  //! Component access operator
  INLINE T &        operator[] (int i);
  //! Component access operator
  INLINE const T &  operator[] (int i) const;

  //! Get a pointer to vector component storage
  INLINE const T * getVec3() const;
  //! Fill component values into arguments
  INLINE void      getVec3(T &x, T &y, T &z) const;
  
  //! Direct manipulation
  INLINE void setX(T x);
  //! Direct manipulation
  INLINE void setY(T y);
  //! Direct manipulation
  INLINE void setZ(T z);
  //! Direct manipulation
  INLINE void setVec3(const T f[]);
  //! Direct manipulation
  INLINE void setVec3(T x,T y,T z);

  //@}

  /** @name Vector length */
  //@{

  //! Get vector length
  INLINE T    getNorm() const;
  //! Get squared vector length
  INLINE T    getSquareNorm() const;
  //! Normalize the vector
  INLINE T    normalize(T tolerance = std::numeric_limits<T>::epsilon());
  //! Get a normalized copy of the vector
  INLINE GbVec3<T> getNormalized(T tolerance = std::numeric_limits<T>::epsilon()) const;

  //@}

  /** @name Measurements */
  //@{

  //! Get the determinant of a matrix consisting of the given row vectors
  INLINE static T det(const GbVec3<T> &a, const GbVec3<T> &b, const GbVec3<T> &c);
  //! Compute the angle formed by the two vectors and the origin
  INLINE static T angle(const GbVec3<T> &a, const GbVec3<T> &b);

  //@}

  /** @name Vector basis methods */
  //@{

  //! Get a vector that is orthonormal to self
  INLINE GbVec3<T>	  getOrthogonalVector() const;
  //! Project vector into a plane normal to the given vector
  INLINE const GbVec3<T>& projectNormalTo(const GbVec3<T> &v);

  //@}

  //! Perform Gram-Schmidt orthonormalization
  INLINE static void orthonormalize(GbVec3<T> v[/*3*/]);
  //! Build an orthonormal basis of vectors
  INLINE static void generateOrthonormalBasis(GbVec3<T>& u, GbVec3<T>& v, GbVec3<T>& w, GbBool unitLenW = true);

  /** @name Methods for rendering */
  //@{

  //! Get a vector that is reflected on the plane with the given normal
  INLINE GbVec3<T>	  getReflectedAt(const GbVec3<T>& n) const;
  //! Get a vector that is refracted on the plane with the given normal and refraction coefficient
  INLINE GbVec3<T>	  getRefractedAt(const GbVec3<T>& n, T index, GbBool& total_reflection) const;

  //@}

  /** @name Input and Output */
  //@{

  //! Output the vector in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbVec3<T>&);
  //! Read the vector in specific ASCII format
  //friend GRIDLIB_API std::istream& operator>> GB_TEXPORT (std::istream&, GbVec3<T>&);

  //@}

  /** @name Predefined vectors */
  //@{

  //! Predefined zero vector
  static const GbVec3<T> ZERO;
  //! Predefined vector in X direction
  static const GbVec3<T> UNIT_X;
  //! Predefined vector in Y direction
  static const GbVec3<T> UNIT_Y;
  //! Predefined vector in Z direction
  static const GbVec3<T> UNIT_Z;

  //@}

  //! To control exactness of comparison operators set this to > 0
  static T EPSILON;

private:
  //! Private data storage
  T xyz[3];
};

/*!
  \param s The scalar to scale vector \a a with
  \param a The vector to be scaled
  \return The scaled vector
 */
template<class T>
GRIDLIB_API
GbVec3<T>
operator* (T s, const GbVec3<T>& a)
{
  T x = s*a[0];
  T y = s*a[1];
  T z = s*a[2];
  return GbVec3<T>(x,y,z);
}

/*!
  \param s The stream to write to
  \param v The vector to be written
  \return The filled stream
*/
template<class T>
GRIDLIB_API
std::ostream&
operator<<(std::ostream& s, const GbVec3<T>& v)
{
  s<<"["<<v[0]<<", "<<v[1]<<", "<<v[2]<<"]";
  return s;
}

/*!
  \param s The stream to read from
  \param v The vector that will be filled with the values read
  \return The emptied stream
*/
template<class T>
GRIDLIB_API
std::istream&
operator>>(std::istream& s, GbVec3<T>& v)
{
  char c;
  char dummy[3];
  T x,y,z;

  s>>c>>x>>dummy>>y>>dummy>>z>>c;
  v.setVec3(x,y,z);
  return s;
}

#ifndef OUTLINE

#include "GbVec3.in"
#include "GbVec3.T" 

#else

INSTANTIATE( GbVec3<float> );
INSTANTIATE( GbVec3<double> ); 
INSTANTIATEF( GbVec3<float> operator*(float s, const GbVec3<float>& a) );
INSTANTIATEF( GbVec3<double> operator*(double s, const GbVec3<double>& a) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbVec3<float>&) ); 
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbVec3<double>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbVec3<float>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbVec3<double>&) );

#endif  // OUTLINE

#endif  // GBVEC3_HH
