// -*- C++ -*- automatisch in C++mode wechseln (emacs)

/*!
  Construct a 4x4 matrix. The data storage is not initialized. This enables 
  fast construction if the compiler is able to optimize it. Use one of
  the other constructors if you want an initialized matrix, or assign one
  of the predefined matrices.
*/
template <class T>
GbMatrix4<T>::GbMatrix4()
{
}

/*!
  \param entry 2D array of values for the matrix entries

  Construct a 4x4 matrix. The entries are explicitly initialized with
  the given values.
*/
template <class T>
GbMatrix4<T>::GbMatrix4(const T entry[4][4])
{
  memcpy(entry_,entry,16*sizeof(T));
}

/*!
  \param entry Array of values for the matrix entries

  Construct a 4x4 matrix. The entries are explicitly initialized with
  the given values.
*/
template <class T>
GbMatrix4<T>::GbMatrix4(const T entry[16])
{
  memcpy(entry_,entry,16*sizeof(T));
}

/*!
  \param m Matrix to copy

  Construct a 4x4 matrix by copying the entries from matrix \a m
*/
template <class T>
GbMatrix4<T>::GbMatrix4(const GbMatrix4<T>& m)
{
  entry_[0][0] = m[0][0];
  entry_[0][1] = m[0][1];
  entry_[0][2] = m[0][2];
  entry_[0][3] = m[0][3];
  entry_[1][0] = m[1][0];
  entry_[1][1] = m[1][1];
  entry_[1][2] = m[1][2];
  entry_[1][3] = m[1][3];
  entry_[2][0] = m[2][0];
  entry_[2][1] = m[2][1];
  entry_[2][2] = m[2][2];
  entry_[2][3] = m[2][3];
  entry_[3][0] = m[3][0];
  entry_[3][1] = m[3][1];
  entry_[3][2] = m[3][2];
  entry_[3][3] = m[3][3];
}

/*!
  \param origin The coordinate frame origin
  \param x The X direction vector
  \param y The Y direction vector
  \param z The Z direction vector

  Construct a 4x4 transformation matrix to a reference frame as
  specified in the input parameters. The origin of the frame is at \a origin ,
  while the unit vectors in \a x , \a y , and \a z will be transformed to the respective
  input vectors. No checking is performed, whether the resulting matrix is
  singular or others.
*/
template <class T>
GbMatrix4<T>::GbMatrix4(const GbVec3<T>& origin, const GbVec3<T>& x,
			const GbVec3<T>& y, const GbVec3<T>& z)
{
  entry_[0][0]= x[0];
  entry_[0][1]= y[0];
  entry_[0][2]= z[0];
  entry_[0][3]= origin[0];
  entry_[1][0]= x[1];
  entry_[1][1]= y[1];
  entry_[1][2]= z[1];
  entry_[1][3]= origin[1];
  entry_[2][0]= x[2];
  entry_[2][1]= y[2];
  entry_[2][2]= z[2];
  entry_[2][3]= origin[2];
  entry_[3][0]= T(0.0);
  entry_[3][1]= T(0.0);
  entry_[3][2]= T(0.0);
  entry_[3][3]= T(1.0);
}

/*!
  \param s Floating point value

  Construct a 4x4 matrix. All entries are initialized with \a s
*/
template <class T>
GbMatrix4<T>::GbMatrix4(T s)
{
  entry_[0][0] = entry_[0][1] = entry_[0][2] = entry_[0][3] =
  entry_[1][0] = entry_[1][1] = entry_[1][2] = entry_[1][3] =
  entry_[2][0] = entry_[2][1] = entry_[2][2] = entry_[2][3] =
  entry_[3][0] = entry_[3][1] = entry_[3][2] = entry_[3][3] = s;
}

/*!
  \param entry00 Floating point value
  \param entry01 Floating point value
  \param entry02 Floating point value
  \param entry03 Floating point value
  \param entry10 Floating point value
  \param entry11 Floating point value
  \param entry12 Floating point value
  \param entry13 Floating point value
  \param entry20 Floating point value
  \param entry21 Floating point value
  \param entry22 Floating point value
  \param entry23 Floating point value
  \param entry30 Floating point value
  \param entry31 Floating point value
  \param entry32 Floating point value
  \param entry33 Floating point value

  Construct a 4x4 matrix. The entries are explicitly initialized with
  the given values.
*/
template <class T>
GbMatrix4<T>::GbMatrix4(T entry00, T entry01, T entry02, T entry03,
			T entry10, T entry11, T entry12, T entry13,
			T entry20, T entry21, T entry22, T entry23,
			T entry30, T entry31, T entry32, T entry33)
{
  entry_[0][0] = entry00; entry_[0][1] = entry01; entry_[0][2] = entry02; entry_[0][3] = entry03; 
  entry_[1][0] = entry10; entry_[1][1] = entry11; entry_[1][2] = entry12; entry_[1][3] = entry13; 
  entry_[2][0] = entry20; entry_[2][1] = entry21; entry_[2][2] = entry22; entry_[2][3] = entry23; 
  entry_[3][0] = entry30; entry_[3][1] = entry31; entry_[3][2] = entry32; entry_[3][3] = entry33; 
}

/*!
  Destruct a 4x4 matrix. Nothing unusual to pay attention to here. Has
  no side effects.
*/
template <class T>
GbMatrix4<T>::~GbMatrix4()
{
}

/*!
  \return The determinant value

  This method computes and returns the determinant of self.
*/
template <class T>
T
GbMatrix4<T>::determinant() const
{
  return  entry_[0][0] * det3( entry_[1][1],entry_[1][2],entry_[1][3],
			       entry_[2][1],entry_[2][2],entry_[2][3],
			       entry_[3][1],entry_[3][2],entry_[3][3] ) -
	  entry_[1][0] * det3( entry_[0][1],entry_[0][2],entry_[0][3],
			       entry_[2][1],entry_[2][2],entry_[2][3],
			       entry_[3][1],entry_[3][2],entry_[3][3] ) +
	  entry_[2][0] * det3( entry_[0][1],entry_[0][2],entry_[0][3],
			       entry_[1][1],entry_[1][2],entry_[1][3],
			       entry_[3][1],entry_[3][2],entry_[3][3] ) -
	  entry_[3][0] * det3( entry_[0][1],entry_[0][2],entry_[0][3],
			       entry_[1][1],entry_[1][2],entry_[1][3],
			       entry_[2][1],entry_[2][2],entry_[2][3] ) ;
}

/*!
  \return The determinant value

  This method computes and returns the determinant of 
  the upper left 3x3 submatrix (the linear/affine part of the transform).
*/
template <class T>
T
GbMatrix4<T>::affineDeterminant() const
{
  return det3( entry_[0][0],entry_[0][1],entry_[0][2],
	       entry_[1][0],entry_[1][1],entry_[1][2],
	       entry_[2][0],entry_[2][1],entry_[2][2] ) ;
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
GbMatrix4<T>
GbMatrix4<T>::inverse(T tolerance) const
{
  GbMatrix4<T> kInverse = GbMatrix4<T>::ZERO;
  inverse(kInverse,tolerance);
  return kInverse;
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

  \internal Code is adopted from Graphics Gems IV using Gauss-Jordan elimination
            with partial pivoting.
*/
template <class T>
GbBool
GbMatrix4<T>::inverse(GbMatrix4<T>& inv, T tolerance) const
{
  GbBool singular = false;
  GbMatrix4<T> a(*this);    // As a evolves from original mat into identity
  GbMatrix4<T> b(GbMatrix4::IDENTITY); // b evolves from identity into inverse(a)
  int i, j, i1;

  // Loop over cols of a from left to right, eliminating above and below diag
  for (j=0; !singular && j<4; ++j) {
    // Find largest pivot in column j among rows j..3
    i1 = j;                 // Row with largest pivot candidate
    for (i=j+1; i<4; ++i)
      if (GbMath<T>::Abs(a.entry_[i][j]) > GbMath<T>::Abs(a.entry_[i1][j]))
        i1 = i;

    // Swap rows i1 and j in a and b to put pivot on diagonal
    swap(a.entry_[i1], a.entry_[j]);
    swap(b.entry_[i1], b.entry_[j]);

    // Scale row j to have a unit diagonal
    if (GbMath<T>::Abs(a.entry_[j][j]) > tolerance) {
      // Scale row by diagomal element
      // b.v[j] /= a.v[j].n[j];
      // a.v[j] /= a.v[j].n[j];
      T tmp= T(1.0)/a.entry_[j][j];
      b.entry_[j][0] *= tmp;
      b.entry_[j][1] *= tmp;
      b.entry_[j][2] *= tmp;
      b.entry_[j][3] *= tmp;
      a.entry_[j][0] *= tmp;
      a.entry_[j][1] *= tmp;
      a.entry_[j][2] *= tmp;
      a.entry_[j][3] *= tmp;

      // Eliminate off-diagonal elems in col j of a, doing identical ops to b
      for (i=0; i<4; ++i) {
	if (i!=j) {
	  // subtract rows
	  // b.v[i] -= a.v[i].n[j]*b.v[j];
	  // a.v[i] -= a.v[i].n[j]*a.v[j];
	  T tmp1= a.entry_[i][j];
	  b.entry_[i][0] -= tmp1*b.entry_[j][0];
	  b.entry_[i][1] -= tmp1*b.entry_[j][1];
	  b.entry_[i][2] -= tmp1*b.entry_[j][2];
	  b.entry_[i][3] -= tmp1*b.entry_[j][3];
	  a.entry_[i][0] -= tmp1*a.entry_[j][0];
	  a.entry_[i][1] -= tmp1*a.entry_[j][1];
	  a.entry_[i][2] -= tmp1*a.entry_[j][2];
	  a.entry_[i][3] -= tmp1*a.entry_[j][3];
	}
      }
    }
    else
      singular = true;
  }

  if (singular) 
    inv=GbMatrix4<T>::ZERO;
  else
    inv=b;
  
  return !singular;
}

