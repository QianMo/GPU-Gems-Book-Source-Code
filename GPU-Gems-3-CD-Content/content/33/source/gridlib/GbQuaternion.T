// -*- C++ -*- automatisch in C++mode wechseln (emacs)

/*!
  Construct a quaternion. The data storage is not initialized. This enables 
  fast construction if the compiler is able to optimize it. Use one of
  the other constructors if you want an initialized quaternion, or assign one
  of the predefined unit quaternions.
*/
template <class T>
GbQuaternion<T>::GbQuaternion()
{
}

/*!
  \param q Array of four values for the components

  Construct a quaternion. The components are explicitly initialized with
  the given values.
*/
template <class T>
GbQuaternion<T>::GbQuaternion(const T q[4])
{
  w_ = q[0];
  x_ = q[1];
  y_ = q[2];
  z_ = q[3];
}

/*!
  \param fW W component
  \param fX X component
  \param fY Y component
  \param fZ Z component

  Construct a quaternion. The components are explicitly initialized with
  the given values.
*/
template <class T>
GbQuaternion<T>::GbQuaternion(T fW, T fX, T fY, T fZ)
{
  w_ = fW;
  x_ = fX;
  y_ = fY;
  z_ = fZ;
}

/*!
  \param rkQ The quaternion to copy

  Construct a quaternion. The components are copied from the \a rkQ quaternion.
*/
template <class T>
GbQuaternion<T>::GbQuaternion(const GbQuaternion<T>& rkQ)
{
  w_ = rkQ.w_;
  x_ = rkQ.x_;
  y_ = rkQ.y_;
  z_ = rkQ.z_;
}

/*!
  \param rkRot A rotation matrix

  Construct a quaternion from a rotation matrix.
*/
template <class T>
GbQuaternion<T>::GbQuaternion(const GbMatrix3<T>& rkRot)
{
    fromRotationMatrix(rkRot);
}

/*!
  \param rkAxis The rotation axis
  \param fAngle The rotation angle

  Construct a quaternion as a rotation about \a fAngle around axis \a rkAxis.
*/
template <class T>
GbQuaternion<T>::GbQuaternion(const GbVec3<T>& rkAxis, T fAngle)
{
    fromAxisAngle(rkAxis,fAngle);
}

/*!
  \param akRotColumn Three column of the rotation matrix

  Construct a quaternion for the rotation matrix with specified columns.
*/
template <class T>
GbQuaternion<T>::GbQuaternion(const GbVec3<T> akRotColumn[3])
{
    fromRotationMatrix(akRotColumn);
}

/*!
  Destruct a quaternion. Nothing unusual to pay attention to here. Has
  no side effects.
*/
template <class T>
GbQuaternion<T>::~GbQuaternion()
{
}

/*!
  \param kRot The rotation matrix

  This method converts the given rotation matrix to quaternion representation
  and stores the resulting quaternion with self.

  \internal Algorithm from Ken Shoemake's article in 1987 SIGGRAPH course notes
            "Quaternion Calculus and Fast Animation".
*/
template <class T>
void 
GbQuaternion<T>::fromRotationMatrix (const GbMatrix3<T>& kRot)
{
  T fTrace = kRot[0][0]+kRot[1][1]+kRot[2][2];
  T fRoot;

  if ( fTrace > T(0.0) ) {
    // |w| > 1/2, may as well choose w > 1/2
    fRoot = GbMath<T>::Sqrt(fTrace + T(1.0));  // 2w
    w_ = T(0.5)*fRoot;
    fRoot = T(0.5)/fRoot;  // 1/(4w)
    x_ = (kRot[2][1]-kRot[1][2])*fRoot;
    y_ = (kRot[0][2]-kRot[2][0])*fRoot;
    z_ = (kRot[1][0]-kRot[0][1])*fRoot;
  }
  else {
    // |w| <= 1/2
    static int s_iNext[3] = { 1, 2, 0 };
    int i = 0;
    if ( kRot[1][1] > kRot[0][0] )
      i = 1;
    if ( kRot[2][2] > kRot[i][i] )
      i = 2;
    int j = s_iNext[i];
    int k = s_iNext[j];

    fRoot = GbMath<T>::Sqrt(kRot[i][i]-kRot[j][j]-kRot[k][k] + T(1.0));
    T* apkQuat[3] = { &x_, &y_, &z_ };
    *apkQuat[i] = T(0.5)*fRoot;
    fRoot = T(0.5)/fRoot;
    w_ = (kRot[k][j]-kRot[j][k])*fRoot;
    *apkQuat[j] = (kRot[j][i]+kRot[i][j])*fRoot;
    *apkQuat[k] = (kRot[k][i]+kRot[i][k])*fRoot;
  }
}

/*!
  \param kRot The rotation matrix

  This method converts self to rotation matrix representation and stores
  the matrix in the provided \a kRot storage.
*/
template <class T>
void 
GbQuaternion<T>::toRotationMatrix (GbMatrix3<T>& kRot) const
{
  T fTx  = T(2.0)*x_;
  T fTy  = T(2.0)*y_;
  T fTz  = T(2.0)*z_;
  T fTwx = fTx*w_;
  T fTwy = fTy*w_;
  T fTwz = fTz*w_;
  T fTxx = fTx*x_;
  T fTxy = fTy*x_;
  T fTxz = fTz*x_;
  T fTyy = fTy*y_;
  T fTyz = fTz*y_;
  T fTzz = fTz*z_;
  
  kRot[0][0] = T(1.0)-(fTyy+fTzz);
  kRot[0][1] = fTxy-fTwz;
  kRot[0][2] = fTxz+fTwy;
  kRot[1][0] = fTxy+fTwz;
  kRot[1][1] = T(1.0)-(fTxx+fTzz);
  kRot[1][2] = fTyz-fTwx;
  kRot[2][0] = fTxz-fTwy;
  kRot[2][1] = fTyz+fTwx;
  kRot[2][2] = T(1.0)-(fTxx+fTyy);
}

/*!
  \param akRotColumn Three column vectors of the rotation matrix
  
  This method builds self from the given column vectors of a rotation matrix.
*/
template <class T>
void 
GbQuaternion<T>::fromRotationMatrix (const GbVec3<T> akRotColumn[3])
{
    GbMatrix3<T> kRot;
    for (int iCol = 0; iCol < 3; iCol++) {
        kRot[0][iCol] = akRotColumn[iCol][0];
        kRot[1][iCol] = akRotColumn[iCol][1];
        kRot[2][iCol] = akRotColumn[iCol][2];
    }
    fromRotationMatrix(kRot);
}

/*!
  \param akRotColumn Three column vectors of the rotation matrix

  This method computes a rotation matrix from self and stors the column vectors
  of the matrix in \a akRotColumn.
*/
template <class T>
void 
GbQuaternion<T>::toRotationMatrix (GbVec3<T> akRotColumn[3]) const
{
    GbMatrix3<T> kRot;
    toRotationMatrix(kRot);
    for (int iCol = 0; iCol < 3; iCol++) {
        akRotColumn[iCol][0] = kRot[0][iCol];
        akRotColumn[iCol][1] = kRot[1][iCol];
        akRotColumn[iCol][2] = kRot[2][iCol];
    }
}

/*!
  \param rfAngle The rotation angle
  \param rkAxis The rotation axis

  This method converts the given axis-angle representation into a quaternion
  and stores the result with self.

  \warning Assertion: axis is unit length

  \internal The quaternion representing the rotation is
            \f$ q = \cos(A/2)+\sin(A/2)*(x*i+y*j+z*k) \f$
*/
template <class T>
void 
GbQuaternion<T>::fromAxisAngle (const GbVec3<T>& rkAxis,const T& rfAngle)
{
  T fHalfAngle = T(0.5)*rfAngle;
  T fSin = GbMath<T>::Sin(fHalfAngle);
  w_ = GbMath<T>::Cos(fHalfAngle);
  x_ = fSin*rkAxis[0];
  y_ = fSin*rkAxis[1];
  z_ = fSin*rkAxis[2];
}

/*!
  \param rfAngle The rotation angle
  \param rkAxis The rotation axis

  This method converts self to axis-angle representation
  and stores the result in the provided storage.

  \internal The quaternion representing the rotation is
            \f$ q = \cos(A/2)+\sin(A/2)*(x*i+y*j+z*k) \f$
*/
template <class T>
void 
GbQuaternion<T>::toAxisAngle (GbVec3<T>& rkAxis,T& rfAngle) const
{
  T fSqrLength = x_*x_+y_*y_+z_*z_;
  if ( fSqrLength > T(0.0) ) {
    rfAngle = T(2.0)*GbMath<T>::ACos(w_);
    T fInvLength = T(1.0)/GbMath<T>::Sqrt(fSqrLength);
    rkAxis[0] = x_*fInvLength;
    rkAxis[1] = y_*fInvLength;
    rkAxis[2] = z_*fInvLength;
  }
  else {
    // angle is 0 (mod 2*pi), so any axis will do
    rfAngle = T(0.0);
    rkAxis[0] = T(1.0);
    rkAxis[1] = T(0.0);
    rkAxis[2] = T(0.0);
  }
}

/*!
  \param akAxis Array of basis vectors

  This method creates a quaternion from the provided vector basis
  by viewing the vectors as columns of a rotation matrix. The
  result is stored with self.
*/
template <class T>
void 
GbQuaternion<T>::fromAxes (const GbVec3<T>* akAxis)
{
  GbMatrix3<T> kRot;

  for (int iCol = 0; iCol < 3; iCol++) {
    kRot[0][iCol] = akAxis[iCol][0];
    kRot[1][iCol] = akAxis[iCol][1];
    kRot[2][iCol] = akAxis[iCol][2];
  }

  fromRotationMatrix(kRot);
}

/*!
  \param akAxis Array for vector basis

  This method converts self into a vector basis. The vectors are
  considered to be the columns of a rotation matrix. The vector basis 
  is stored in the provided array.
*/
template <class T>
void 
GbQuaternion<T>::toAxes (GbVec3<T>* akAxis) const
{
  GbMatrix3<T> kRot;

  toRotationMatrix(kRot);

  for (int iCol = 0; iCol < 3; iCol++) {
    akAxis[iCol][0] = kRot[0][iCol];
    akAxis[iCol][1] = kRot[1][iCol];
    akAxis[iCol][2] = kRot[2][iCol];
  }
}

/*!
  \return The exponential of self

  \internal If \f$ q = A*(x*i+y*j+z*k) \f$ where (x,y,z) is unit length, then
            \f$ \exp(q) = \cos(A)+\sin(A)*(x*i+y*j+z*k) \f$ . If \f$ \sin(A) \f$ is near zero,
            use \f$ \exp(q) = \cos(A)+A*(x*i+y*j+z*k) \f$ since \f$ A/\sin(A) \f$ has limit 1.
*/
template <class T>
GbQuaternion<T> 
GbQuaternion<T>::exp () const
{
  T fAngle = GbMath<T>::Sqrt(x_*x_+y_*y_+z_*z_);
  T fSin = GbMath<T>::Sin(fAngle);

  if ( GbMath<T>::Abs(fSin) >= GbQuaternion<T>::EPSILON ) {
    T fCoeff = fSin/fAngle;
    return GbQuaternion<T>(GbMath<T>::Cos(fAngle),
			   fCoeff*x_,
			   fCoeff*y_,
			   fCoeff*z_);
  }
  return GbQuaternion<T>(GbMath<T>::Cos(fAngle),
			 x_,
			 y_,
			 z_);
}

/*!
  \return The logarithmic of self

  \internal If \f$ q = \cos(A)+\sin(A)*(x*i+y*j+z*k) \f$ where (x,y,z) is unit length, then
            \f$ \log(q) = A*(x*i+y*j+z*k) \f$ . If \f$ \sin(A) \f$ is near zero, use 
	    \f$ \log(q) = \sin(A)*(x*i+y*j+z*k) \f$ since \f$ \sin(A)/A \f$ has limit 1.
*/
template <class T>
GbQuaternion<T> 
GbQuaternion<T>::log () const
{
  if ( GbMath<T>::Abs(w_) < T(1.0) ) {
    T fAngle = GbMath<T>::ACos(w_);
    T fSin = GbMath<T>::Sin(fAngle);
    if ( GbMath<T>::Abs(fSin) >= GbQuaternion<T>::EPSILON ) {
      T fCoeff = fAngle/fSin;
      return GbQuaternion<T>(T(0.0),
			     fCoeff*x_,
			     fCoeff*y_,
			     fCoeff*z_);
    }
  }
  return GbQuaternion<T>(T(0.0),
			 x_,
			 y_,
			 z_);
}

/*!
  \param rkVector The vector in question
  \return The modified vector

  Use this method to transform a 3D vector \a rkVector by the
  self quaternion and return the transformed vector. As the quaternion
  can be viewed as a specific rotation, this is an alternative to a
  multiplication with a rotation matrix.

  \internal Given a vector \f$ u = (x_0,y_0,z_0) \f$ and a unit length quaternion
  \f$ q = <w,x,y,z> \f$ , the vector \f$ v = (x_1,y_1,z_1) \f$ which represents the
  rotation of \p u by \p q is \f$ v = q*u*q^{-1} \f$ where * indicates quaternion
  multiplication and where \p u is treated as the quaternion \f$ <0,x_0,y_0,z_0> \f$ .
  Note that \f$ q^{-1} = <w,-x,-y,-z> \f$ , so no real work is required to
  invert \p q. Now
  \f[ q*u*q^{-1} = q*<0,x_0,y_0,z_0>*q^{-1}
                 = q*(x_0*i+y_0*j+z_0*k)*q^{-1}
		 = x_0*(q*i*q^{-1})+y_0*(q*j*q^{-1})+z_0*(q*k*q^{-1}) \f]
  As 3-vectors, \f$ q*i*q^{-1}, q*j*q^{-1}, 2*k*q^{-1} \f$ are the columns
  of the rotation matrix computed in GbQuaternion::toRotationMatrix.
  The vector \p v is obtained as the product of that rotation matrix with
  vector \p u .  As such, the quaternion representation of a rotation
  matrix requires less space than the matrix and more time to compute
  the rotated vector.  Typical space-time tradeoff...
*/
template <class T>
GbVec3<T> 
GbQuaternion<T>::operator* (const GbVec3<T>& rkVector) const
{
  GbMatrix3<T> kRot;
  toRotationMatrix(kRot);
  return kRot*rkVector;
}

/*!
  \param fT The time parameter
  \param rkP The start quaternion
  \param rkQ The end quaternion
  \return The interpolated quaternion

  This method performs a spherical linear interpolation between the start and end
  quaternion according to the \a fT parameter and returns the result.
*/
template <class T>
GbQuaternion<T> 
GbQuaternion<T>::slerp (T fT, const GbQuaternion<T>& rkP, const GbQuaternion<T>& rkQ)
{
  T fCos = rkP.dot(rkQ);
  T fAngle = GbMath<T>::ACos(fCos);

  if ( GbMath<T>::Abs(fAngle) < GbQuaternion<T>::EPSILON )
    return rkP;

  T fSin = GbMath<T>::Sin(fAngle);
  T fInvSin = T(1.0)/fSin;
  T fCoeff0 = GbMath<T>::Sin((T(1.0)-fT)*fAngle)*fInvSin;
  T fCoeff1 = GbMath<T>::Sin(fT*fAngle)*fInvSin;
  return fCoeff0*rkP + fCoeff1*rkQ;
}

/*!
  \param fT The time parameter
  \param rkP The start quaternion
  \param rkQ The end quaternion
  \param iExtraSpins Number of additional complete revolutions
  \return The interpolated quaternion

  This method performs a spherical linear interpolation between the start and end
  quaternion according to the \a fT parameter and returns the result. The interpolated
  path takes \a iExtraSpins full rotations.
*/
template <class T>
GbQuaternion<T> 
GbQuaternion<T>::slerpExtraSpins (T fT, const GbQuaternion<T>& rkP, const GbQuaternion<T>& rkQ, int iExtraSpins)
{
  T fCos = rkP.dot(rkQ);
  T fAngle = GbMath<T>::ACos(fCos);

  if ( GbMath<T>::Abs(fAngle) < GbQuaternion<T>::EPSILON )
    return rkP;

  T fSin = GbMath<T>::Sin(fAngle);
  T fPhase = GbMath<T>::PI*iExtraSpins*fT;
  T fInvSin = T(1.0)/fSin;
  T fCoeff0 = GbMath<T>::Sin((T(1.0)-fT)*fAngle - fPhase)*fInvSin;
  T fCoeff1 = GbMath<T>::Sin(fT*fAngle + fPhase)*fInvSin;
  return fCoeff0*rkP + fCoeff1*rkQ;
}

/*!
  \param rkQ0 The first quaternion
  \param rkQ1 The second quaternion
  \param rkQ2 The third quaternion

  This method performs the first step for the spherical cubic interpolation.
  
  \warning Assertion: \a rkQ0 , \a rkQ1 , \a rkQ2 are unit-length quaternions
*/
template <class T>
GbQuaternion<T>
GbQuaternion<T>::intermediate (const GbQuaternion<T>& rkQ0, const GbQuaternion<T>& rkQ1, const GbQuaternion<T>& rkQ2)
{
    GbQuaternion<T> kQ1Inv = rkQ1.unitInverse();
    GbQuaternion<T> kP0 = kQ1Inv*rkQ0;
    GbQuaternion<T> kP2 = kQ1Inv*rkQ2;
    GbQuaternion<T> kArg = T(-0.25)*(kP0.log()+kP2.log());
    return GbQuaternion<T>(rkQ1*kArg.exp());
}

/*!
  \param fT The time parameter
  \param rkP The first quaternion
  \param rkA The first interpolated quaternion
  \param rkB The second interpolated quaternion
  \param rkQ The second quaternion
  \return The interpolated quaternion

  This method performs the spherical cubic interpolation and
  returns the resulting quaternion for the time parameter \a fT . The two interpolated
  quaternions necessary can be obtained from the intermediate() method.
*/
template <class T>
GbQuaternion<T> 
GbQuaternion<T>::squad (T fT, const GbQuaternion<T>& rkP, const GbQuaternion<T>& rkA,
			const GbQuaternion<T>& rkB, const GbQuaternion<T>& rkQ)
{
  T fSlerpT = T(2.0)*fT*(T(1.0)-fT);
  GbQuaternion<T> kSlerpP = slerp(fT,rkP,rkQ);
  GbQuaternion<T> kSlerpQ = slerp(fT,rkA,rkB);
  return slerp(fSlerpT,kSlerpP,kSlerpQ);
}

/*!
  \param rkV1 The first vector
  \param rkV2 The second vector
  \return The quaternion describing the rotation

  Compute a quaternion that rotates unit-length vector \a rkV1 to unit-length
  vector \a rkV2.  The rotation is about the axis perpendicular to both \a rkV1 and
  \a rkV2, with angle of that between \a rkV1 and \a rkV2.  If \a rkV1 and \a rkV2 are parallel,
  any axis of rotation will do, such as the permutation \f$ (z_2,x_2,y_2) \f$ , where
  \f$ rkV2 = (x_2,y_2,z_2) \f$.

  \internal If V1 and V2 are not parallel, the axis of rotation is the unit-length
  vector U = Cross(V1,V2)/Length(Cross(V1,V2)).  The angle of rotation,
  A, is the angle between V1 and V2.  The quaternion for the rotation is
  q = cos(A/2) + sin(A/2)*(ux*i+uy*j+uz*k) where U = (ux,uy,uz).
  -# Rather than extract A = acos(Dot(V1,V2)), multiply by 1/2, then
     compute sin(A/2) and cos(A/2), we reduce the computational costs by
     computing the bisector B = (V1+V2)/Length(V1+V2), so cos(A/2) =
     Dot(V1,B).
  -# The rotation axis is U = Cross(V1,B)/Length(Cross(V1,B)), but
     Length(Cross(V1,B)) = Length(V1)*Length(B)*sin(A/2) = sin(A/2), in
     which case sin(A/2)*(ux*i+uy*j+uz*k) = (cx*i+cy*j+cz*k) where
     C = Cross(V1,B).
  .
  If V1 and V2 are parallel, or nearly parallel as far as the floating
  point calculations are concerned, the calculation of B will produce
  the zero vector: GbVec3::normalize checks for closeness to zero and
  returns the zero vector accordingly.  Thus, we test for parallelism
  by checking if cos(A/2) is zero.  The test for exactly zero is usually
  not recommend for floating point arithmetic, but the implementation of
  GbVec3::normalize guarantees the comparison is robust.
*/
template <class T>
GbQuaternion<T> 
GbQuaternion<T>::align (const GbVec3<T>& rkV1, const GbVec3<T>& rkV2)
{
    GbVec3<T> kBisector = rkV1 + rkV2;
    kBisector.normalize();

    T fCosHalfAngle = rkV1.dot(kBisector);
    GbVec3<T> kCross;

    if ( fCosHalfAngle != T(0.0) )
        kCross = rkV1.cross(kBisector);
    else
        kCross = rkV1.unitCross(GbVec3<T>(rkV2[2],rkV2[0],rkV2[1]));

    return GbQuaternion<T>(fCosHalfAngle,kCross[0],kCross[1],kCross[2]);
}

/*!
  \param rkAxis The rotation axis
  \param rkTwist The twisted component
  \param rkNoTwist The twist-free component

  Decompose self into \f$ self = q_{twist} * q_{notwist} \f$ .
  If V1 is the input axis and V2 is the rotation of V1 by
  self, \f$ q_{notwist} \f$ represents the rotation about the axis perpendicular to
  V1 and V2, and \f$ q_{twist} \f$ is a rotation about V1.
  
  \sa GbQuaternion<T>::align
*/
template <class T>
void 
GbQuaternion<T>::decomposeTwistTimesNoTwist (const GbVec3<T>& rkAxis, GbQuaternion<T>& rkTwist, GbQuaternion<T>& rkNoTwist)
{
    GbVec3<T> kRotatedAxis = (*this)*rkAxis;
    rkNoTwist = align(rkAxis,kRotatedAxis);
    rkTwist = (*this)*rkNoTwist.unitInverse();
}

/*!
  \param rkAxis The rotation axis
  \param rkTwist The twisted component
  \param rkNoTwist The twist-free component
  
  Decompose self into \f$ self = q_{notwist} * q_{twist} \f$ .
  If V1 is the input axis and V2 is the rotation of V1 by self,
  \f$ q_{notwist} \f$ represents the rotation about the axis perpendicular to
  V1 and V2, and \f$ q_{twist} \f$ is a rotation about V1.
  
  \sa GbQuaternion<T>::align
*/
template <class T>
void 
GbQuaternion<T>::decomposeNoTwistTimesTwist (const GbVec3<T>& rkAxis, GbQuaternion<T>& rkTwist, GbQuaternion<T>& rkNoTwist)
{
    GbVec3<T> kRotatedAxis = (*this)*rkAxis;
    rkNoTwist = align(rkAxis,kRotatedAxis);
    rkTwist = rkNoTwist.unitInverse()*(*this);
}
