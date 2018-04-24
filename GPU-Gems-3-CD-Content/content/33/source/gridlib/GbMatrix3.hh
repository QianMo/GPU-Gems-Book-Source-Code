/*----------------------------------------------------------------------
|
| $Id: GbMatrix3.hh,v 1.9 2003/03/06 17:01:52 prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBMATRIX3_HH
#define  GBMATRIX3_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbVec3.hh"
#include "GbMath.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief Simple 3x3 matrix class with arithmetic operators.
  \author Peter Kipfer
  $Revision: 1.9 $
  $Date: 2003/03/06 17:01:52 $
  \note Original Source: VisMatrix3x3 from the Vision project (c) 1995 Swen Campagna

  \note This is a version modified for GPUGems from the original gridlib file

  Simple 3x3 Matrix class.
  The (x,y,z) coordinate system is assumed to be right-handed.
  Coordinate axis rotation matrices are of the form

  \f[ RX = \begin{pmatrix}   1 &    0     &    0 \\
                             0 &  \cos(t) & -\sin(t) \\
                             0 &  \sin(t) &  \cos(t)
	   \end{pmatrix}
  \f]
  where \f$ t > 0 \f$ indicates a counterclockwise rotation in the yz-plane

  \f[ RY = \begin{pmatrix} \cos(t) & 0 & \sin(t) \\
                              0    & 1 &   0 \\
			  -\sin(t) & 0 & \cos(t)
	   \end{pmatrix}
  \f]
  where \f$ t > 0 \f$ indicates a counterclockwise rotation in the zx-plane

  \f[ RZ = \begin{pmatrix} \cos(t) & -\sin(t) & 0 \\
                           \sin(t) &  \cos(t) & 0 \\
			       0   &    0     & 1
           \end{pmatrix}
  \f]
  where \f$ t > 0 \f$ indicates a counterclockwise rotation in the xy-plane.

  \warning This class template is intended to be parametrized with
           floating point data types. Look for the special treatment of
           integral numbers in the operators. 

  \sa GbMatrix2 GbMatrix4 GbMatrixN
*/
template <class T>
class GRIDLIB_API GbMatrix3
{
public:
  /** @name Construction and Destruction */
  //@{

  //! Constructor for uninitialized matrix
  GbMatrix3();
  //! Constructor for explicitly initialized matrix
  GbMatrix3(const T entry[3][3]);
  //! Constructor for explicitly initialized matrix
  GbMatrix3(const T entry[9]);
  //! Copy Constructor
  GbMatrix3(const GbMatrix3<T>& m);
  //! Constructor for explicitly initialized matrix
  GbMatrix3(T entry00, T entry01, T entry02,
	    T entry10, T entry11, T entry12,
	    T entry20, T entry21, T entry22);
  //! Constructor for initialized matrix
  explicit GbMatrix3(T s);

  //! Destructor
  ~GbMatrix3();

  //@}

  /** @name Access methods */
  //@{

  //! Matrix entry access, allows use of construct mat[r][c]
  INLINE T* operator[] (int row) const;
  //! Get a column vector
  INLINE GbVec3<T> getColumn (int col) const;

  //@}

  /** @name Binary operators */
  //@{

  //! Equality test
  INLINE GbBool        operator== (const GbMatrix3<T>& m) const;
  //! Inequality test
  INLINE GbBool        operator!= (const GbMatrix3<T>& m) const;
  //! Componentwise addition
  INLINE GbMatrix3<T>& operator+= (const GbMatrix3<T>& m);
  //! Componentwise subtraction
  INLINE GbMatrix3<T>& operator-= (const GbMatrix3<T>& m);
  //! Matrix multiplication
  INLINE GbMatrix3<T>& operator*= (const GbMatrix3<T>& m);

  //! Componentwise addition
  INLINE GbMatrix3<T> operator+ (const GbMatrix3<T>& m) const;
  //! Componentwise subtraction
  INLINE GbMatrix3<T> operator- (const GbMatrix3<T>& m) const;
  //! Matrix multiplication
  INLINE GbMatrix3<T> operator* (const GbMatrix3<T>& m) const;
  //! Componentwise negation
  INLINE GbMatrix3<T> operator- () const;

  //! Self * vector [3x3 * 3x1 = 3x1]
  INLINE GbVec3<T> operator* (const GbVec3<T>& v) const;

  //! Scaling with a floating point value
  INLINE GbMatrix3<T> operator* (const T& s) const;
  //! Scaling with an integral value
  INLINE GbMatrix3<T> operator* (const int& s) const;

  //@}

  //! Vector * matrix [1x3 * 3x3 = 1x3]
  //friend GRIDLIB_API GbVec3<T> operator* GB_TEXPORT (const GbVec3<T>& v, const GbMatrix3<T>& m);

  //! Scalar * matrix
  //friend GRIDLIB_API GbMatrix3<T> operator* GB_TEXPORT (const T& s, const GbMatrix3<T>& m);

  /** @name Matrix inversion */
  //@{

  //! Multiply the transpose of self with the given matrix
  INLINE GbMatrix3<T> transposeTimes (const GbMatrix3<T>& rkM) const;
  //! Multiply self with the transpose of the given matrix
  INLINE GbMatrix3<T> timesTranspose (const GbMatrix3<T>& rkM) const;

  //! Get the transpose of self
  INLINE GbMatrix3<T> transpose () const;
  //! Get the inverse of self
  GbBool inverse (GbMatrix3<T>& inv, T tolerance = std::numeric_limits<T>::epsilon()) const;
  //! Get the inverse of self
  GbMatrix3<T> inverse (T tolerance = std::numeric_limits<T>::epsilon()) const;
  //! Get the determinant of self
  INLINE T determinant () const;

  //@}

  //! Compute the tensor \a product of two vectors \a u and \a v
  static void tensorProduct (const GbVec3<T>& u, const GbVec3<T>& v, GbMatrix3<T>& product);

  /** @name Predefined matrices */
  //@{

  //! Epsilon for tuning numerical stability
  static const T EPSILON;
  //! Predefined zero matrix
  static const GbMatrix3<T> ZERO;
  //! Predefined identity matrix
  static const GbMatrix3<T> IDENTITY;

  //@}

  /** @name Input and Output */
  //@{

  //! Output the matrix in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbMatrix3<T>&);
  //! Read the matrix in specific ASCII format
  //friend GRIDLIB_API std::istream& operator>> GB_TEXPORT (std::istream&, GbMatrix3<T>&);

  //@}

private:

  //! Storage for the matrix elements
  T entry_[3][3];
};

/*!
  \param s The floating point scaling factor
  \param m The matrix to scale
  \return The scaled matrix

  This method returns a matrix that is \a m scaled componentwise by \a s
*/
template<class T>
GRIDLIB_API
GbMatrix3<T>
operator* (const T& s, const GbMatrix3<T>& m)
{
  T kProd[9];
  int k=0;

  for (int iRow = 0; iRow < 3; ++iRow) {
    for (int iCol = 0; iCol < 3; ++iCol)
      kProd[k++] = s*m.entry_[iRow][iCol];
  }
  return GbMatrix3<T>(kProd);
}

/*!
  \param v The 1x3 vector to multiply
  \param m The 3x3 matrix to multiply
  \return The 1x3 multiplication result

  This method returns a vector that is matrix \a m multiplied with \a v from the left.
 */
template<class T>
GRIDLIB_API
GbVec3<T> 
operator* (const GbVec3<T>& v, const GbMatrix3<T>& m)
{
  T kProd[3];
  for (int iRow = 0; iRow < 3; ++iRow) {
    kProd[iRow] = v[0]*m[0][iRow] + v[1]*m[1][iRow] + v[2]*m[2][iRow];
  }
  return GbVec3<T>(kProd);
}

/*!
  \param s The stream to write to
  \param v The matrix to be written
  \return The filled stream
*/
template<class T>
GRIDLIB_API
std::ostream&
operator<<(std::ostream& s, const GbMatrix3<T>& v)
{
  s<<"[("<<v[0][0]<<", "<<v[1][0]<<", "<<v[2][0]<<") ";
  s<<"("<<v[0][1]<<", "<<v[1][1]<<", "<<v[2][1]<<") ";
  s<<"("<<v[0][2]<<", "<<v[1][2]<<", "<<v[2][2]<<")]";
  return s;
}

/*!
  \param s The stream to read from
  \param v The matrix that will be filled with the values read
  \return The emptied stream
*/
template<class T>
GRIDLIB_API
std::istream&
operator>>(std::istream& s, GbMatrix3<T>& v)
{
  char c;
  char dummy[3];
  T x,y,z;
  
  s>>c>>c>>x>>dummy>>y>>dummy>>z>>c>>c;
  v[0][0]=x; v[1][0]=y; v[2][0]=z;
  s>>c>>x>>dummy>>y>>dummy>>z>>c>>c;
  v[0][1]=x; v[1][1]=y; v[2][1]=z;
  s>>c>>x>>dummy>>y>>dummy>>z>>c>>c;
  v[0][2]=x; v[1][2]=y; v[2][2]=z;
  return s;
}

#ifndef OUTLINE

#include "GbMatrix3.in"
#include "GbMatrix3.T"

#else // OUTLINE

INSTANTIATE( GbMatrix3<float> );
INSTANTIATE( GbMatrix3<double> );
INSTANTIATEF( GbVec3<float>  operator* (const GbVec3<float>& v, const GbMatrix3<float>& m) );
INSTANTIATEF( GbVec3<double> operator* (const GbVec3<double>& v, const GbMatrix3<double>& m) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbMatrix3<float>&) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbMatrix3<double>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbMatrix3<float>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbMatrix3<double>&) );

#endif  // OUTLINE

#endif  // GBMATRIX3_HH
