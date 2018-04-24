/*----------------------------------------------------------------------
|
| $Id: GbVec4.hh,v 1.7 2003/03/06 17:01:52 prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBVEC4_HH
#define  GBVEC4_HH

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
  \brief Simple 4D point with arithmetic operators.
  \author Peter Kipfer
  $Revision: 1.7 $
  $Date: 2003/03/06 17:01:52 $

  \note This is a version modified for GPUGems from the original gridlib file

  This class represents a simple point in 4D space. It has access methods
  to treat it like a vector, comparison operators that support fuzzy logic and
  arithmetic operators for all standard operations.

  \warning This class template is intended to be parametrized with
           floating point data types. Look for the special treatment of
           integral numbers in the operators. 

  \sa GbVec2 GbVec3H GbVec3i GbVec3 GbVecN
 */
template <class T>
class GRIDLIB_API GbVec4
{
public:
  /** @name Construction and Destruction */
  //@{

  //! Constructor for uninitialized vector
  GbVec4();
  //! Constructor for initialized vector
  explicit GbVec4(T s);
  //! Constructor for explicitly initialized vector
  GbVec4(T a, T b, T c, T d);
  //! Constructor for explicitly initialized vector
  GbVec4(const T p[4]);
  //! Constructor for explicitly initialized vector
  GbVec4(const GbVec4<T> &p);

  //! Destructor
  ~GbVec4();

  //@}

  /** @name Binary operators */
  //@{

  //! Equality test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator==(const GbVec4<T>& p) const;
  //! Inequality test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator!=(const GbVec4<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator< (const GbVec4<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator<=(const GbVec4<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator> (const GbVec4<T>& p) const;
  //! Ordering test (supports fuzzy arithmetic when EPSILON > 0)
  INLINE GbBool operator>=(const GbVec4<T>& p) const;

  //! Componentwise addition
  INLINE GbVec4<T>& operator+=(const GbVec4<T>& p);
  //! Addition of a scalar to each component
  INLINE GbVec4<T>& operator+=(const T& s);
  //! Componentwise subtraction
  INLINE GbVec4<T>& operator-=(const GbVec4<T>& p);
  //! Subtraction of a scalar from each component
  INLINE GbVec4<T>& operator-=(const T& s);
  //! Componentwise multiplication
  INLINE GbVec4<T>& operator*=(const GbVec4<T>& p);
  //! Scaling with integral value
  INLINE GbVec4<T>& operator*=(const int& s);
  //! Scaling with floating point value
  INLINE GbVec4<T>& operator*=(const T& s);
  //! Componentwise division
  INLINE GbVec4<T>& operator/=(const GbVec4<T>& p);
  //! Inverse scaling with integral value
  INLINE GbVec4<T>& operator/=(const int& s);
  //! Inverse scaling with floating point value
  INLINE GbVec4<T>& operator/=(const T& s);

  //! Componentwise addition
  INLINE GbVec4<T>  operator+(const GbVec4<T>& p) const;
  //! Addition of a scalar to each component
  INLINE GbVec4<T>  operator+(const T& p) const;
  //! Componentwise subtraction
  INLINE GbVec4<T>  operator-(const GbVec4<T>& p) const;
  //! Subtraction of a scalar from each component
  INLINE GbVec4<T>  operator-(const T& p) const;
  //! Componentwise multiplication
  INLINE GbVec4<T>  operator*(const GbVec4<T>& p) const;
  //! Scaling with integral value
  INLINE GbVec4<T>  operator*(const int& s) const;
  //! Scaling with floating point value
  INLINE GbVec4<T>  operator*(const T& s) const;
  //! Componentwise division
  INLINE GbVec4<T>  operator/(const GbVec4<T>& p) const;
  //! Inverse scaling with integral value
  INLINE GbVec4<T>  operator/(const int& s) const;
  //! Inverse scaling with floating point value
  INLINE GbVec4<T>  operator/(const T& s) const;

  //@}

  //! Scaling with floating point value as a prefix
  //friend GRIDLIB_API GbVec4<T>  operator* GB_TEXPORT (T s, const GbVec4<T>& a);

  //! Vector inversion
  INLINE GbVec4<T>  operator-() const;

  /** @name Vector products */
  //@{
  
  //! Scalar product (dot product)
  INLINE        T operator|(const GbVec4<T> &a) const;
  //! Scalar product (dot product)
  INLINE        T dot(const GbVec4<T> &a) const;

  //@}

  /** @name Access methods */
  //@{

  //! Component access operator
  INLINE T &        operator[] (int i);
  //! Component access operator
  INLINE const T &  operator[] (int i) const;

  //! Get a pointer to vector component storage
  INLINE const T * getVec4() const;
  //! Fill component values into arguments
  INLINE void      getVec4(T &a, T &b, T &c, T &d) const;
  
  //! Direct manipulation
  INLINE void setA(T a);
  //! Direct manipulation
  INLINE void setB(T b);
  //! Direct manipulation
  INLINE void setC(T c);
  //! Direct manipulation
  INLINE void setD(T d);
  //! Direct manipulation
  INLINE void setVec4(const T f[]);
  //! Direct manipulation
  INLINE void setVec4(T a,T b,T c,T d);

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
  INLINE GbVec4<T> getNormalized(T tolerance = std::numeric_limits<T>::epsilon()) const;

  //@}

  /** @name Measurements */
  //@{

  //! Compute the angle formed by the two vectors and the origin
  INLINE static T angle(const GbVec4<T> &a, const GbVec4<T> &b);

  //@}

  /** @name Methods for rendering */
  //@{

  //! Project vector into a plane normal to the given vector
  INLINE const GbVec4<T>& projectNormalTo(const GbVec4<T> &v);
  //! Get a vector that is reflected on the plane with the given normal
  INLINE GbVec4<T>	  getReflectedAt(const GbVec4<T>& n) const;
  //! Get a vector that is refracted on the plane with the given normal and refraction coefficient
  INLINE GbVec4<T>	  getRefractedAt(const GbVec4<T>& n, T index, GbBool& total_reflection) const;
  
  //@}

  /** @name Input and Output */
  //@{

  //! Output the vector in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbVec4<T>&);
  //! Read the vector in specific ASCII format
  //friend GRIDLIB_API std::istream& operator>> GB_TEXPORT (std::istream&, GbVec4<T>&);

  //@}

  /** @name Predefined vectors */
  //@{

  //! Predefined zero vector
  static const GbVec4<T> ZERO;
  //! Predefined vector in first dimension
  static const GbVec4<T> UNIT_A;
  //! Predefined vector in second dimension
  static const GbVec4<T> UNIT_B;
  //! Predefined vector in third dimension
  static const GbVec4<T> UNIT_C;
  //! Predefined vector in fourth dimension
  static const GbVec4<T> UNIT_D;

  //@}

  //! To control exactness of comparison operators set this to > 0
  static T EPSILON;

private:
  //! Private data storage
  T xyz[4];
};

/*!
  \param s The scalar to scale vector \a v with
  \param v The vector to be scaled
  \return The scaled vector
 */
template<class T>
GbVec4<T>
operator* (T s, const GbVec4<T>& v)
{
  T a = s*v[0];
  T b = s*v[1];
  T c = s*v[2];
  T d = s*v[3];
  return GbVec4<T>(a,b,c,d);
}

/*!
  \param s The stream to write to
  \param v The vector to be written
  \return The filled stream
*/
template<class T>
std::ostream&
operator<<(std::ostream& s, const GbVec4<T>& v)
{
  s<<"["<<v[0]<<", "<<v[1]<<", "<<v[2]<<", "<<v[3]<<"]";
  return s;
}

/*!
  \param s The stream to read from
  \param v The vector that will be filled with the values read
  \return The emptied stream
*/
template<class T>
std::istream&
operator>>(std::istream& s, GbVec4<T>& v)
{
  char ch;
  char dummy[3];
  T a,b,c,d;

  s>>ch>>a>>dummy>>b>>dummy>>c>>dummy>>d>>ch;
  v.setVec4(a,b,c,d);
  return s;
}

#ifndef OUTLINE

#include "GbVec4.in"
#include "GbVec4.T"

#else

INSTANTIATE( GbVec4<float> );
INSTANTIATE( GbVec4<double> );
INSTANTIATEF( GbVec4<float> operator*(float s, const GbVec4<float>& a) );
INSTANTIATEF( GbVec4<double> operator*(double s, const GbVec4<double>& a) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbVec4<float>&) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbVec4<double>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbVec4<float>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbVec4<double>&) );

#endif // OUTLINE

#endif  // GBVEC4_HH
