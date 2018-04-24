/*----------------------------------------------------------------------
|
| $Id: GbQuaternion.hh,v 1.9 2004/11/08 11:04:03 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBQUATERNION_HH
#define  GBQUATERNION_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbMatrix3.hh"
#include "GbMath.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief Quaternion class with arithmetic operators.
  \author Peter Kipfer
  $Revision: 1.9 $
  $Date: 2004/11/08 11:04:03 $
  \note Original Source: Magic Software, Inc.

  \note This is a version modified for GPUGems from the original gridlib file

  This class represents a quaternion. A quaternion is \f$ q = w + x*i + y*j + z*k \f$ 
  where \f$ (w,x,y,z) \f$ is not necessarily a unit length vector in 4D.
  This class has all standard arithmetic capabilities,
  methods for conversion to other representations and spherical linear/cubic 
  interpolation methods.

  For further technical documentation of the internal workings of this
  class consult the section \ref TECHQUATERNION
*/
template <class T>
class GRIDLIB_API GbQuaternion
{
public:
  /** @name Construction and Destruction */
  //@{

  //! Constructor for uninitialized quaternion
  GbQuaternion();
  //! Constructor for explicitly initialized vector
  GbQuaternion(const T q[4]);
  //! Constructor for explicitly initialized vector
  GbQuaternion(T fW, T fX, T fY, T fZ);
  //! Constructor for explicitly initialized vector
  GbQuaternion(const GbQuaternion<T>& rkQ);
    
  //! Quaternion for the input rotation matrix
  GbQuaternion(const GbMatrix3<T>& rkRot);

  //! Quaternion for the rotation of the axis-angle pair
  GbQuaternion(const GbVec3<T>& rkAxis, T fAngle);

  //! Quaternion for the rotation matrix with specified columns
  GbQuaternion(const GbVec3<T> akRotColumn[3]);

  //! Destructor
  ~GbQuaternion();

  //@}

  /** @name Access methods */
  //@{

  //! Get the W scalar component
  INLINE T W() const;
  //! Get the XYZ vector component
  INLINE GbVec3<T> XYZ() const;
  //! Set the quaternion to the provided component values
  INLINE void set(T w, T x, T y, T z);

  //@}

  /** @name Representation conversion methods */
  //@{

  //! Build a quaternion from the provided rotation matrix
  void fromRotationMatrix (const GbMatrix3<T>& kRot);
  //! Build a rotation matrix in the provided storage from the quaternion
  void toRotationMatrix (GbMatrix3<T>& kRot) const;
  //! Build a quaternion from the provided column vectors of a rotation matrix
  void fromRotationMatrix (const GbVec3<T> akRotColumn[3]);
  //! Build the colum vectors of a rotation matrix in the provided storage from the quaternion
  void toRotationMatrix (GbVec3<T> akRotColumn[3]) const;
  //! Build a quaternion from the  angle-axis representation
  void fromAxisAngle (const GbVec3<T>& rkAxis,const T& rfAngle);
  //! Build a angle-axis representation from the quaternion
  void toAxisAngle (GbVec3<T>& rkAxis,T& rfAngle) const;
  //! Build a quaternion from the provided rotation axes (vector basis)
  void fromAxes (const GbVec3<T>* akAxis);
  //! Build a vector basis (rotation axes) from the quaternion
  void toAxes (GbVec3<T>* akAxis) const;

  //@}

  /** @name Assignment and comparison */
  //@{

  //! Copy the provided quaternion
  INLINE GbQuaternion<T>& operator= (const GbQuaternion<T>& rkQ);
  INLINE GbBool operator== (const GbQuaternion<T>& rkQ) const;
  INLINE GbBool operator!= (const GbQuaternion<T>& rkQ) const;
  INLINE GbBool operator<  (const GbQuaternion<T>& rkQ) const;
  INLINE GbBool operator<= (const GbQuaternion<T>& rkQ) const;
  INLINE GbBool operator>  (const GbQuaternion<T>& rkQ) const;
  INLINE GbBool operator>= (const GbQuaternion<T>& rkQ) const;

  //@}

  /** @name Arithmetic operations */
  //@{

  //! Add two quaternions
  INLINE GbQuaternion<T> operator+ (const GbQuaternion<T>& rkQ) const;
  //! Subtract two quaternions
  INLINE GbQuaternion<T> operator- (const GbQuaternion<T>& rkQ) const;
  //! Multiply two quaternions
  INLINE GbQuaternion<T> operator* (const GbQuaternion<T>& rkQ) const;
  //! Scale the quaternion
  INLINE GbQuaternion<T> operator* (T fScalar) const;
  //! Scale the quaternion
  INLINE GbQuaternion<T> operator/ (T fScalar) const;
  //! Invert the quaternion
  INLINE GbQuaternion<T> operator- () const;
  //! Scale the quaternion with prefix scalar value
  //friend GRIDLIB_API GbQuaternion<T> operator* GB_TEXPORT (T fScalar, const GbQuaternion<T>& rkQ);

  //@}

  /** @name Arithmetic updates */
  //@{

  INLINE GbQuaternion<T>& operator+= (const GbQuaternion<T>& rkQ);
  INLINE GbQuaternion<T>& operator-= (const GbQuaternion<T>& rkQ);
  INLINE GbQuaternion<T>& operator*= (T fScalar);
  INLINE GbQuaternion<T>& operator/= (T fScalar);

  //@}

  /** @name Functions */
  //@{

  //! Compute the scalar product (dot product)
  INLINE T dot (const GbQuaternion<T>& rkQ) const;
  //! Get the norm of the quaternion (=squared length)
  INLINE T norm () const;
  //! Get the inverse quaternion
  INLINE GbQuaternion<T> inverse () const;
  //! Get the inverse of a unit quaternion
  INLINE GbQuaternion<T> unitInverse () const;
  //! Get the exponential of self
  GbQuaternion<T> exp () const;
  //! Get the logarithmic of self
  GbQuaternion<T> log () const;

  //@}

  //! Rotation of a vector by a quaternion
  GbVec3<T> operator* (const GbVec3<T>& rkVector) const;

  /** @name Interpolation */
  //@{

  //! Spherical linear interpolation
  static GbQuaternion<T> slerp (T fT, const GbQuaternion<T>& rkP, const GbQuaternion<T>& rkQ);

  //! Spherical linear interpolation with extra spins
  static GbQuaternion<T> slerpExtraSpins (T fT, const GbQuaternion<T>& rkP, 
					  const GbQuaternion<T>& rkQ, int iExtraSpins);

  //! Setup for spherical cubic interpolation
  static GbQuaternion<T> intermediate (const GbQuaternion<T>& rkQ0, const GbQuaternion<T>& rkQ1, const GbQuaternion<T>& rkQ2);

  //! Spherical cubic interpolation
  static GbQuaternion<T> squad (T fT, const GbQuaternion<T>& rkP, const GbQuaternion<T>& rkA, 
				const GbQuaternion<T>& rkB, const GbQuaternion<T>& rkQ);

  //! Rotate vector to other vector and return rotation quaternion
  static GbQuaternion<T> align (const GbVec3<T>& rkV1, const GbVec3<T>& rkV2);

  //! Decompose rotation
  void decomposeTwistTimesNoTwist (const GbVec3<T>& rkAxis, GbQuaternion<T>& rkTwist, GbQuaternion<T>& rkNoTwist);

  //! Decompose rotation
  void decomposeNoTwistTimesTwist (const GbVec3<T>& rkAxis, GbQuaternion<T>& rkTwist, GbQuaternion<T>& rkNoTwist);

  //@}

  /** @name Predefined quaternions */
  //@{

  //! Predefined zero quaternion
  static const GbQuaternion<T> ZERO;
  //! Predefined identity quaternion
  static const GbQuaternion<T> IDENTITY;

  //! Cutoff for sine values near zero	
  static const T EPSILON;

  //@}

  /** @name Input and Output */
  //@{

  //! Output the quaternion in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbQuaternion<T>&);
  //! Read the quaternion in specific ASCII format
  //friend GRIDLIB_API std::istream& operator>> GB_TEXPORT (std::istream&, GbQuaternion<T>&);

  //@}

protected:
  // Support for comparisons
  int compareArrays (const GbQuaternion<T>& rkQ) const;

  // Data storage
  T w_, x_, y_, z_;
};

/*!
  \param fScalar The scalar to scale quaternion \a rkQ with
  \param rkQ The quaternion to be scaled
  \return The scaled quaternion
 */
template <class T>
GbQuaternion<T> 
operator* (T fScalar, const GbQuaternion<T>& rkQ)
{
  GbVec3<T> v = rkQ.XYZ();
  return GbQuaternion<T>(fScalar*rkQ.W(),fScalar*v[0],fScalar*v[1],fScalar*v[2]);
}

/*!
  \param s The stream to write to
  \param q The quaternion to be written
  \return The filled stream
*/
template<class T>
std::ostream&
operator<<(std::ostream& s, const GbQuaternion<T>& q)
{
  GbVec3<T> v = q.XYZ();
  s<<"("<<q.W()<<", "<<v[0]<<", "<<v[1]<<", "<<v[2]<<")";
  return s;
}

/*!
  \param s The stream to read from
  \param q The quaternion that will be filled with the values read
  \return The emptied stream
*/
template<class T>
std::istream&
operator>>(std::istream& s, GbQuaternion<T>& q)
{
  char c;
  char dummy[3];
  T w,x,y,z;
  
  s>>c>>w>>dummy>>x>>dummy>>y>>dummy>>z>>c;
  q.set(w,x,y,z);
  return s;
}

#ifndef OUTLINE

#include "GbQuaternion.in"
#include "GbQuaternion.T"

#else

INSTANTIATE( GbQuaternion<float> );
INSTANTIATE( GbQuaternion<double> );
INSTANTIATEF( GbQuaternion<float>  operator* (float fScalar, const GbQuaternion<float>& rkQ) );
INSTANTIATEF( GbQuaternion<double> operator* (double fScalar, const GbQuaternion<double>& rkQ) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbQuaternion<float>&) );
INSTANTIATEF( std::ostream& operator<<(std::ostream&, const GbQuaternion<double>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbQuaternion<float>&) );
INSTANTIATEF( std::istream& operator>>(std::istream&, GbQuaternion<double>&) );

#endif // OUTLINE

#endif  // GBQUATERNION_HH
