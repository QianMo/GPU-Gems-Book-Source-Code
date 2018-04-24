/*----------------------------------------------------------------------
|
| $Id: GbVec3i.hh,v 1.5 2003/03/06 17:01:52 prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBVEC3I_HH
#define  GBVEC3I_HH

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
  \brief Simple 3D point with arithmetic operators for integral numbers.
  \author Peter Kipfer
  $Revision: 1.5 $
  $Date: 2003/03/06 17:01:52 $

  \note This is a version modified for GPUGems from the original gridlib file

  This class represents a simple point in 3D space. It has access methods
  to treat it like a vector, comparison operators and
  arithmetic operators for all standard operations.

  \warning This class template is intended to be parametrized with
           integral data types. Look for the special treatment of
           floating point numbers in the operators. 

  \sa GbVec2 GbVec3H GbVec3 GbVec4 GbVecN
 */
template <class T>
class GRIDLIB_API GbVec3i
{
public:
  /** @name Construction and Destruction */
  //@{

  //! Constructor for uninitialized vector
  GbVec3i();
  //! Constructor for initialized vector
  explicit GbVec3i(T s);
  //! Constructor for explicitly initialized vector
  GbVec3i(T x, T y, T z);
  //! Constructor for explicitly initialized vector
  GbVec3i(const T p[3]);
  //! Constructor for explicitly initialized vector
  GbVec3i(const GbVec3i<T> &p);

  //! Destructor
  ~GbVec3i();

  //@}

  /** @name Binary operators */
  //@{

  //! Equality test
  INLINE GbBool operator==(const GbVec3i<T>& p) const;
  //! Inequality test
  INLINE GbBool operator!=(const GbVec3i<T>& p) const;
  //! Ordering test
  INLINE GbBool operator< (const GbVec3i<T>& p) const;
  //! Ordering test
  INLINE GbBool operator<=(const GbVec3i<T>& p) const;
  //! Ordering test
  INLINE GbBool operator> (const GbVec3i<T>& p) const;
  //! Ordering test
  INLINE GbBool operator>=(const GbVec3i<T>& p) const;
  
  //! Componentwise addition
  INLINE GbVec3i<T>& operator+=(const GbVec3i<T>& p);
  //! Addition of a scalar to each component
  INLINE GbVec3i<T>& operator+=(const T& s);
  //! Componentwise subtraction
  INLINE GbVec3i<T>& operator-=(const GbVec3i<T>& p);
  //! Subtraction of a scalar from each component
  INLINE GbVec3i<T>& operator-=(const T& s);
  //! Componentwise multiplication
  INLINE GbVec3i<T>& operator*=(const GbVec3i<T>& p);
  //! Scaling with floating point value
  INLINE GbVec3i<T>& operator*=(const float& s);
  //! Scaling with integral value
  INLINE GbVec3i<T>& operator*=(const T& s);
  //! Componentwise division
  INLINE GbVec3i<T>& operator/=(const GbVec3i<T>& p);
  //! Inverse scaling with floating point value
  INLINE GbVec3i<T>& operator/=(const float& s);
  //! Inverse scaling with integral value
  INLINE GbVec3i<T>& operator/=(const T& s);

  //! Componentwise addition
  INLINE GbVec3i<T>  operator+(const GbVec3i<T>& p) const;
  //! Addition of a scalar to each component
  INLINE GbVec3i<T>  operator+(const T& p) const;
  //! Componentwise subtraction
  INLINE GbVec3i<T>  operator-(const GbVec3i<T>& p) const;
  //! Subtraction of a scalar from each component
  INLINE GbVec3i<T>  operator-(const T& p) const;
  //! Componentwise multiplication
  INLINE GbVec3i<T>  operator*(const GbVec3i<T>& p) const;
  //! Scaling with floating point value
  INLINE GbVec3i<T>  operator*(const float& s) const;
  //! Scaling with integral value
  INLINE GbVec3i<T>  operator*(const T& s) const;
  //! Componentwise division
  INLINE GbVec3i<T>  operator/(const GbVec3i<T>& p) const;
  //! Inverse scaling with floating point value
  INLINE GbVec3i<T>  operator/(const float& s) const;
  //! Inverse scaling with integral value
  INLINE GbVec3i<T>  operator/(const T& s) const;

  //@}

  //! Scaling with integral value as a prefix
  //friend GRIDLIB_API GbVec3i<T>  operator* GB_TEXPORT (T s, const GbVec3i<T>& a);

  //! Vector inversion
  INLINE GbVec3i<T>  operator-() const;

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

  /** @name Input and Output */
  //@{

  //! Output the vector in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbVec3i<T>&);
  //! Read the vector in specific ASCII format
  //friend GRIDLIB_API std::istream& operator>> GB_TEXPORT (std::istream&, GbVec3i<T>&);

  //@}

  /** @name Predefined vectors */
  //@{

  //! Predefined zero vector
  static const GbVec3i<T> ZERO;
  //! Predefined vector in X direction
  static const GbVec3i<T> UNIT_X;
  //! Predefined vector in Y direction
  static const GbVec3i<T> UNIT_Y;
  //! Predefined vector in Z direction
  static const GbVec3i<T> UNIT_Z;

  //@}

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
GbVec3i<T>
operator* (T s, const GbVec3i<T>& a)
{
  T x = s*a[0];
  T y = s*a[1];
  T z = s*a[2];
  return GbVec3i<T>(x,y,z);
}

/*!
  \param s The stream to write to
  \param v The vector to be written
  \return The filled stream
*/
template<class T>
GRIDLIB_API
std::ostream&
operator<<(std::ostream& s, const GbVec3i<T>& v)
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
operator>>(std::istream& s, GbVec3i<T>& v)
{
  char c;
  char dummy[3];
  T x,y,z;

  s>>c>>x>>dummy>>y>>dummy>>z>>c;
  v.setVec3(x,y,z);
  return s;
}

#ifndef OUTLINE

#include "GbVec3i.in"
#include "GbVec3i.T"

#else

INSTANTIATE( GbVec3i<int> );
INSTANTIATEF( GbVec3i<int> operator*(int s, const GbVec3i<int>& a)  );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbVec3i<int>&)  );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbVec3i<int>&) );

#endif  // OUTLINE

#endif  // GBVEC3I_HH
