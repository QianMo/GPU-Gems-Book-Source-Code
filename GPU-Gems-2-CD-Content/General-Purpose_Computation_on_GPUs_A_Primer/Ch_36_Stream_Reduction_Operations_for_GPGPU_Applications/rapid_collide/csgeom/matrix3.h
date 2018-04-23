/*
    Copyright (C) 1998,1999,2000 by Jorrit Tyberghein
    Largely rewritten by Ivan Avramovic <ivan@avramovic.com>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#ifndef __CS_MATRIX3_H__
#define __CS_MATRIX3_H__

#ifndef __CS_CSSYSDEFS_H__
//#error "cssysdef.h must be included in EVERY source file!"
#endif



struct Quaternion;

/**
 * A 3x3 matrix.
 */
class csMatrix3
{
public:
  float m11, m12, m13;
  float m21, m22, m23;
  float m31, m32, m33;

public:
  /// Construct a matrix, initialized to be the identity.
  csMatrix3 ()
      : m11(1), m12(0), m13(0),
	m21(0), m22(1), m23(0),
	m31(0), m32(0), m33(1)
  {}

  /// Construct a matrix and initialize it.
  csMatrix3 (float am11, float am12, float am13,
             float am21, float am22, float am23,
             float am31, float am32, float am33)
      : m11(am11), m12(am12), m13(am13),
	m21(am21), m22(am22), m23(am23),
	m31(am31), m32(am32), m33(am33)
  {}

  /// Construct a matrix with a quaternion.
  explicit csMatrix3 (const Quaternion &quat) { Set (quat); }

  /// Get the first row of this matrix as a vector.
  inline csVector3 Row1() const { return csVector3 (m11,m12,m13); }

  /// Get the second row of this matrix as a vector.
  inline csVector3 Row2() const { return csVector3 (m21,m22,m23); }

  /// Get the third row of this matrix as a vector.
  inline csVector3 Row3() const { return csVector3 (m31,m32,m33); }

  /// Get the first column of this matrix as a vector.
  inline csVector3 Col1() const { return csVector3 (m11,m21,m31); }

  /// Get the second column of this matrix as a vector.
  inline csVector3 Col2() const { return csVector3 (m12,m22,m32); }

  /// Get the third column of this matrix as a vector.
  inline csVector3 Col3() const { return csVector3 (m13,m23,m33); }

  /// Set matrix values.
  inline void Set (float m11, float m12, float m13,
                   float m21, float m22, float m23,
                   float m31, float m32, float m33)
  {
    csMatrix3::m11 = m11; csMatrix3::m12 = m12; csMatrix3::m13 = m13;
    csMatrix3::m21 = m21; csMatrix3::m22 = m22; csMatrix3::m23 = m23;
    csMatrix3::m31 = m31; csMatrix3::m32 = m32; csMatrix3::m33 = m33;
  }

  /// Initialize matrix with a quaternion.
  void Set (const Quaternion &quat);

  /// Add another matrix to this matrix.
  csMatrix3& operator+= (const csMatrix3& m);

  /// Subtract another matrix from this matrix.
  csMatrix3& operator-= (const csMatrix3& m);

  /// Multiply another matrix with this matrix.
  csMatrix3& operator*= (const csMatrix3& m);

  /// Multiply this matrix with a scalar.
  csMatrix3& operator*= (float s);

  /// Divide this matrix by a scalar.
  csMatrix3& operator/= (float s);

  /// Unary + operator.
  inline csMatrix3 operator+ () const { return *this; }
  /// Unary - operator.
  inline csMatrix3 operator- () const
  {
    return csMatrix3(-m11,-m12,-m13,
                     -m21,-m22,-m23,
                    -m31,-m32,-m33);
  }

  /// Transpose this matrix.
  void Transpose ();

  /// Return the transpose of this matrix.
  csMatrix3 GetTranspose () const;

  /// Return the inverse of this matrix.
  inline csMatrix3 GetInverse () const
  {
    csMatrix3 C(
             (m22*m33 - m23*m32), -(m12*m33 - m13*m32),  (m12*m23 - m13*m22),
            -(m21*m33 - m23*m31),  (m11*m33 - m13*m31), -(m11*m23 - m13*m21),
             (m21*m32 - m22*m31), -(m11*m32 - m12*m31),  (m11*m22 - m12*m21) );
    float s = (float)1./(m11*C.m11 + m12*C.m21 + m13*C.m31);

    C *= s;

    return C;
  }

  /// Invert this matrix.
  void Invert() { *this = GetInverse (); }

  /// Compute the determinant of this matrix.
  float Determinant () const;

  /// Set this matrix to the identity matrix.
  void Identity ();

  /// Check if the matrix is identity
  bool IsIdentity () const;

  /// Add two matricies.
  friend csMatrix3 operator+ (const csMatrix3& m1, const csMatrix3& m2);
  /// Subtract two matricies.
  friend csMatrix3 operator- (const csMatrix3& m1, const csMatrix3& m2);
  /// Multiply two matricies.
  friend csMatrix3 operator* (const csMatrix3& m1, const csMatrix3& m2);

  /// Multiply a vector by a matrix (transform it).
  inline friend csVector3 operator* (const csMatrix3& m, const csVector3& v)
  {
    return csVector3 (m.m11*v.x + m.m12*v.y + m.m13*v.z,
                      m.m21*v.x + m.m22*v.y + m.m23*v.z,
                      m.m31*v.x + m.m32*v.y + m.m33*v.z);
  }

  /// Multiply a matrix and a scalar.
  friend csMatrix3 operator* (const csMatrix3& m, float f);
  /// Multiply a matrix and a scalar.
  friend csMatrix3 operator* (float f, const csMatrix3& m);
  /// Divide a matrix by a scalar.
  friend csMatrix3 operator/ (const csMatrix3& m, float f);
  /// Check if two matricies are equal.
  friend bool operator== (const csMatrix3& m1, const csMatrix3& m2);
  /// Check if two matricies are not equal.
  friend bool operator!= (const csMatrix3& m1, const csMatrix3& m2);
  /// Test if each component of a matrix is less than a small epsilon value.
  friend bool operator< (const csMatrix3& m, float f);
  /// Test if each component of a matrix is greater than a small epsilon value.
  friend bool operator> (float f, const csMatrix3& m);
};

/// An instance of csMatrix3 that is initialized as a rotation about X
class csXRotMatrix3 : public csMatrix3
{
public:
  /**
   * Return a rotation matrix around the X axis.
   * 'angle' is given in radians.
   */
  csXRotMatrix3 (float angle);
};

/// An instance of csMatrix3 that is initialized as a rotation about Y
class csYRotMatrix3 : public csMatrix3
{
public:
  /**
   * Return a rotation matrix around the Y axis.
   * 'angle' is given in radians.
   */
  csYRotMatrix3 (float angle);
};

/// An instance of csMatrix3 that is initialized as a rotation about Z
class csZRotMatrix3 : public csMatrix3
{
public:
  /**
   * Return a rotation matrix around the Z axis.
   * 'angle' is given in radians.
   */
  csZRotMatrix3 (float angle);
};

/// An instance of csMatrix3 that is initialized to scale the X dimension
class csXScaleMatrix3 : public csMatrix3
{
public:
  /**
   * Return a matrix which scales in the X dimension.
   */
  csXScaleMatrix3 (float scaler) : csMatrix3(scaler, 0, 0, 0, 1, 0, 0, 0, 1) {}
};

/// An instance of csMatrix3 that is initialized to scale the Y dimension
class csYScaleMatrix3 : public csMatrix3
{
public:
  /**
   * Return a matrix which scales in the Y dimension.
   */
  csYScaleMatrix3 (float scaler) : csMatrix3(1, 0, 0, 0, scaler, 0, 0, 0, 1) {}
};

/// An instance of csMatrix3 that is initialized to scale the Z dimension
class csZScaleMatrix3 : public csMatrix3
{
public:
  /**
   * Return a matrix which scales in the Z dimension.
   */
  csZScaleMatrix3 (float scaler) : csMatrix3(1, 0, 0, 0, 1, 0, 0, 0, scaler) {}
};

#endif // __CS_MATRIX3_H__
