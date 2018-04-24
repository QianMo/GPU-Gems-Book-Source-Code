/*----------------------------------------------------------------------
|
| $Id: GbMatrix4.hh,v 1.7 2005/10/21 09:44:52 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBMATRIX4_HH
#define  GBMATRIX4_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbVec3.hh"
#include "GbVec4.hh"
#include "GbMath.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief Simple 4x4 matrix class with arithmetic operators.
  \author Peter Kipfer
  $Revision: 1.7 $
  $Date: 2005/10/21 09:44:52 $

  \note This is a version modified for GPUGems from the original gridlib file

  Simple 4x4 Matrix class with standard arithmetic operations.

  \warning This class template is intended to be parametrized with
           floating point data types. Look for the special treatment of
           integral numbers in the operators. 

  \sa GbMatrix2 GbMatrix3 GbMatrixN
*/
template <class T>
class GRIDLIB_API GbMatrix4
{
public:
  /** @name Construction and Destruction */
  //@{

  //! Constructor for uninitialized matrix
  GbMatrix4 ();
  //! Constructor for explicitly initialized matrix
  GbMatrix4 (const T entry[4][4]);
  //! Constructor for explicitly initialized matrix
  GbMatrix4 (const T entry[16]);
  //! Copy Constructor
  GbMatrix4 (const GbMatrix4<T>& m);
  //! Construct a coordinate frame matrix
  GbMatrix4 (const GbVec3<T>& origin, const GbVec3<T>& x,
	     const GbVec3<T>& y, const GbVec3<T>& z);
  //! Constructor for explicitly initialized matrix
  GbMatrix4 (T entry00, T entry01, T entry02, T entry03,
	     T entry10, T entry11, T entry12, T entry13,
	     T entry20, T entry21, T entry22, T entry23,
	     T entry30, T entry31, T entry32, T entry33);
  //! Constructor for initialized matrix
  explicit GbMatrix4 (T s);

  //! Destructor
  ~GbMatrix4();

  //@}

  /** @name Access methods */
  //@{

  //! Matrix entry access, allows use of construct mat[r][c]
  INLINE T* operator[] (int row) const;
  //! Get a column vector
  INLINE GbVec4<T> getColumn (int col) const;

  //@}

  /** @name Binary operators */
  //@{

  //! Equality test
  INLINE GbBool        operator== (const GbMatrix4<T>& m) const;
  //! Inequality test
  INLINE GbBool        operator!= (const GbMatrix4<T>& m) const;
  //! Componentwise addition
  INLINE GbMatrix4<T>& operator+= (const GbMatrix4<T>& m);
  //! Componentwise subtraction
  INLINE GbMatrix4<T>& operator-= (const GbMatrix4<T>& m);
  //! Matrix multiplication
  INLINE GbMatrix4<T>& operator*= (const GbMatrix4<T>& m);

  //! Componentwise addition
  INLINE GbMatrix4<T> operator+ (const GbMatrix4<T>& m) const;
  //! Componentwise subtraction
  INLINE GbMatrix4<T> operator- (const GbMatrix4<T>& m) const;
  //! Matrix multiplication
  INLINE GbMatrix4<T> operator* (const GbMatrix4<T>& m) const;
  //! Componentwise negation
  INLINE GbMatrix4<T> operator- () const;

  //! Self * vector [4x4 * 4x1 = 4x1]
  INLINE GbVec4<T> operator* (const GbVec4<T>& v) const;
  //! Transform a 3D vector by self
  INLINE GbVec3<T> transform (const GbVec3<T>& v, T w=1.0) const;

  //@}

  /** @name Matrix inversion */
  //@{

  //! Get the transpose of self
  INLINE GbMatrix4<T> transpose () const;
  //! Get the inverse of self
  GbBool inverse (GbMatrix4<T>& inv, T tolerance = std::numeric_limits<T>::epsilon()) const;
  //! Get the inverse of self
  GbMatrix4<T> inverse (T tolerance = std::numeric_limits<T>::epsilon()) const;
  //! Get the determinant of self
  T determinant () const;
  //! Get the determinant of the affine part of self
  T affineDeterminant() const;

  //@}

  /** @name Predefined matrices */
  //@{

  //! Predefined zero matrix
  static const GbMatrix4<T> ZERO;
  //! Predefined identity matrix
  static const GbMatrix4<T> IDENTITY;

  //@}

  /** @name Input and Output */
  //@{

  //! Output the matrix in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbMatrix4<T>&);
  //! Read the matrix in specific ASCII format
  //friend GRIDLIB_API std::istream& operator>> GB_TEXPORT (std::istream&, GbMatrix4<T>&);

  //@}

private:
  //! Compute a 2x2 subdeterminant
  INLINE T det2(const T a11, const T a12, const T a21, const T a22) const;
  //! Compute a 3x3 subdeterminant
  INLINE T det3(const T a11, const T a12, const T a13, 
		const T a21, const T a22, const T a23,
		const T a31, const T a32, const T a33) const;
  //! Swap two vectors
  INLINE void swap(T a[4], T b[4]) const;

  //! Storage for the matrix elements
  T entry_[4][4];
};

/*!
  \param s The stream to write to
  \param v The matrix to be written
  \return The filled stream
*/
template<class T>
std::ostream&
operator<<(std::ostream& s, const GbMatrix4<T>& v)
{
  s<<"[("<<v[0][0]<<", "<<v[1][0]<<", "<<v[2][0]<<", "<<v[3][0]<<") ";
  s<<"("<<v[0][1]<<", "<<v[1][1]<<", "<<v[2][1]<<", "<<v[3][1]<<") ";
  s<<"("<<v[0][2]<<", "<<v[1][2]<<", "<<v[2][2]<<", "<<v[3][2]<<") ";
  s<<"("<<v[0][3]<<", "<<v[1][3]<<", "<<v[2][3]<<", "<<v[3][3]<<")]";
  return s;
}

/*!
  \param s The stream to read from
  \param v The matrix that will be filled with the values read
  \return The emptied stream
*/
template<class T>
std::istream&
operator>>(std::istream& s, GbMatrix4<T>& v)
{
  char c;
  char dummy[3];
  T x,y,z,w;
  
  s>>c>>c>>x>>dummy>>y>>dummy>>z>>dummy>>w>>c>>c;
  v[0][0]=x; v[1][0]=y; v[2][0]=z; v[3][0]=w;
  s>>c>>x>>dummy>>y>>dummy>>z>>dummy>>w>>c>>c;
  v[0][1]=x; v[1][1]=y; v[2][1]=z; v[3][1]=w;
  s>>c>>x>>dummy>>y>>dummy>>z>>dummy>>w>>c>>c;
  v[0][2]=x; v[1][2]=y; v[2][2]=z; v[3][2]=w;
  s>>c>>x>>dummy>>y>>dummy>>z>>dummy>>w>>c>>c;
  v[0][3]=x; v[1][3]=y; v[2][3]=z; v[3][3]=w;
  return s;
}

#ifndef OUTLINE

#include "GbMatrix4.in"
#include "GbMatrix4.T"

#else

INSTANTIATE( GbMatrix4<float> );
INSTANTIATE( GbMatrix4<double> );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbMatrix4<float>&) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbMatrix4<double>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbMatrix4<float>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbMatrix4<double>&) );

#endif

#endif  // GBMATRIX4_HH
