// -*- C++ -*- automatisch in C++mode wechseln (emacs)

/*!
  Construct a 3x3 matrix. The data storage is not initialized. This enables 
  fast construction if the compiler is able to optimize it. Use one of
  the other constructors if you want an initialized matrix, or assign one
  of the predefined matrices.
*/
template <class T>
GbMatrix3<T>::GbMatrix3()
{
}

/*!
  \param entry 2D array of values for the matrix entries

  Construct a 3x3 matrix. The entries are explicitly initialized with
  the given values.
*/
template <class T>
GbMatrix3<T>::GbMatrix3(const T entry[3][3])
{
  memcpy(entry_,entry,9*sizeof(T));
}

/*!
  \param entry Array of values for the matrix entries

  Construct a 3x3 matrix. The entries are explicitly initialized with
  the given values.
*/
template <class T>
GbMatrix3<T>::GbMatrix3(const T entry[9])
{
  memcpy(entry_,entry,9*sizeof(T));
}

/*!
  \param m Matrix to copy

  Construct a 3x3 matrix by copying the entries from matrix \a m
*/
template <class T>
GbMatrix3<T>::GbMatrix3(const GbMatrix3<T>& m)
{
  entry_[0][0] = m[0][0];
  entry_[0][1] = m[0][1];
  entry_[0][2] = m[0][2];
  entry_[1][0] = m[1][0];
  entry_[1][1] = m[1][1];
  entry_[1][2] = m[1][2];
  entry_[2][0] = m[2][0];
  entry_[2][1] = m[2][1];
  entry_[2][2] = m[2][2];
}

/*!
  \param entry00 Floating point value
  \param entry01 Floating point value
  \param entry02 Floating point value
  \param entry10 Floating point value
  \param entry11 Floating point value
  \param entry12 Floating point value
  \param entry20 Floating point value
  \param entry21 Floating point value
  \param entry22 Floating point value

  Construct a 3x3 matrix. The entries are explicitly initialized with
  the given values.
*/
template <class T>
GbMatrix3<T>::GbMatrix3(T entry00, T entry01, T entry02,
			T entry10, T entry11, T entry12, 
			T entry20, T entry21, T entry22)
{
  entry_[0][0] = entry00; entry_[0][1] = entry01; entry_[0][2] = entry02;
  entry_[1][0] = entry10; entry_[1][1] = entry11; entry_[1][2] = entry12;
  entry_[2][0] = entry20; entry_[2][1] = entry21; entry_[2][2] = entry22;
}

/*!
  \param s Floating point value

  Construct a 3x3 matrix. All entries are initialized with \a s
*/
template <class T>
GbMatrix3<T>::GbMatrix3(T s)
{
  entry_[0][0] = entry_[0][1] = entry_[0][2] =
  entry_[1][0] = entry_[1][1] = entry_[1][2] =
  entry_[2][0] = entry_[2][1] = entry_[2][2] = s;
}

/*!
  Destruct a 3x3 matrix. Nothing unusual to pay attention to here. Has
  no side effects.
*/
template <class T>
GbMatrix3<T>::~GbMatrix3()
{
}

/*!
  \param inv The inverted matrix
  \param tolerance Epsilon value for determining success of inversion
  \return true if inversion was successfull

  This method computes the inverse of self using cofactors.
  This is about 8 times faster than the Numerical Recipes code which 
  uses Gaussian elimination. You can
  provide a \a tolerance that is compared to the determinant of the
  inversion result. If the determinant is smaller than \a tolerance ,
  the inversion is considered to have failed and false is returned.
  If the inversion is successfull, true is returned. 
  The inverse matrix is return in \a inv . 
*/
template <class T>
GbBool 
GbMatrix3<T>::inverse (GbMatrix3<T>& inv, T tolerance) const
{
  inv[0][0] = entry_[1][1]*entry_[2][2] - entry_[1][2]*entry_[2][1];
  inv[0][1] = entry_[0][2]*entry_[2][1] - entry_[0][1]*entry_[2][2];
  inv[0][2] = entry_[0][1]*entry_[1][2] - entry_[0][2]*entry_[1][1];
  inv[1][0] = entry_[1][2]*entry_[2][0] - entry_[1][0]*entry_[2][2];
  inv[1][1] = entry_[0][0]*entry_[2][2] - entry_[0][2]*entry_[2][0];
  inv[1][2] = entry_[0][2]*entry_[1][0] - entry_[0][0]*entry_[1][2];
  inv[2][0] = entry_[1][0]*entry_[2][1] - entry_[1][1]*entry_[2][0];
  inv[2][1] = entry_[0][1]*entry_[2][0] - entry_[0][0]*entry_[2][1];
  inv[2][2] = entry_[0][0]*entry_[1][1] - entry_[0][1]*entry_[1][0];

  T fDet = entry_[0][0]*inv[0][0] +
           entry_[0][1]*inv[1][0] +
           entry_[0][2]*inv[2][0];

  if ( GbMath<T>::Abs(fDet) <= tolerance ) return false;

  T fInvDet = T(1.0)/fDet;
  for (int iRow = 0; iRow < 3; ++iRow) {
    for (int iCol = 0; iCol < 3; ++iCol)
      inv[iRow][iCol] *= fInvDet;
  }

  return true;
}

/*!
  \param tolerance Epsilon value for determining success of inversion
  \return The inverted matrix

  This method computes the inverse of self.
  You can provide a \a tolerance that is compared to the determinant of the
  inversion result. If the determinant is smaller than \a tolerance ,
  the inversion is considered to have failed and the ZERO matrix is returned.
  If the inversion is successfull, the inverse matrix is returned. 
*/
template <class T>
GbMatrix3<T> 
GbMatrix3<T>::inverse (T tolerance) const
{
  GbMatrix3<T> kInverse = GbMatrix3<T>::ZERO;
  inverse(kInverse,tolerance);
  return kInverse;
}

/*!
  \param rkU First vector
  \param rkV Second vector
  \param rkProduct The resulting tensor product

  This method computes the tensor product of the two vectors and returns the
  result in \a rkProduct .
*/
template <class T>
void 
GbMatrix3<T>::tensorProduct (const GbVec3<T>& rkU, const GbVec3<T>& rkV, GbMatrix3<T>& rkProduct)
{
  for (int iRow = 0; iRow < 3; ++iRow) {
    for (int iCol = 0; iCol < 3; ++iCol)
      rkProduct[iRow][iCol] = rkU[iRow]*rkV[iCol];
  }
}

