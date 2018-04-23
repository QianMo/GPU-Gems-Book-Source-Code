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

#ifndef __CS_VECTOR3_H__
#define __CS_VECTOR3_H__

#ifndef __CS_CSSYSDEFS_H__
//#error "cssysdef.h must be included in EVERY source file!"
#endif

//#include "csgeom/math3d_d.h"

/**
 * A 3D vector.
 */
/*
inline float ABS(float x) {
  return x>0?x:-x;
}
*/
#include <brook/brook.hpp>
class csVector3
{
public:
  /// The X component of the vector
  float x;
  /// The Y component of the vector
  float y;
  /// The Z component of the vector
  float z;

  /**
   * Make a new vector. The vector is not
   * initialized. This makes the code slightly faster as
   * csVector3 objects are used a lot.
   */
  csVector3 () {}

  /**
   * Make a new initialized vector.
   * Creates a new vector and initializes it to m*<1,1,1>.  To create
   * a vector initialized to the zero vector, use csVector3(0)
   */
  csVector3 (float m) : x(m), y(m), z(m) {}

  /// Make a new vector and initialize with the given values.
  csVector3 (float ix, float iy, float iz = 0) : x(ix), y(iy), z(iz) {}

  /// Copy Constructor.
  csVector3 (const csVector3& v) : x(v.x), y(v.y), z(v.z) {}
  csVector3 (const float3& v) : x(v.x), y(v.y), z(v.z) {}
  /// Conversion from double precision vector to single.
  /// Add two vectors.
  inline csVector3 operator+ (const csVector3& v2) const
  { return csVector3(x+v2.x, y+v2.y, z+v2.z); }

  /// Subtract two vectors.
  inline csVector3 operator- (const csVector3& v2) const
  { return csVector3(x-v2.x, y-v2.y, z-v2.z); }

  /// Subtract two vectors of differing type, cast to double.

  /// Subtract two vectors of differing type, cast to double.

  /// Take the dot product of two vectors.
  inline friend float operator* (const csVector3& v1, const csVector3& v2)
  { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

  /// Take the cross product of two vectors.
  inline friend csVector3 operator% (const csVector3& v1, const csVector3& v2)
  {
    return csVector3 (v1.y*v2.z-v1.z*v2.y,
                      v1.z*v2.x-v1.x*v2.z,
                      v1.x*v2.y-v1.y*v2.x);
  }

  /// Take cross product of two vectors and put result in this vector.
  void Cross (const csVector3 & px, const csVector3 & py)
  {
    x = px.y*py.z - px.z*py.y;
    y = px.z*py.x - px.x*py.z;
    z = px.x*py.y - px.y*py.x;
  }

  /// Multiply a vector and a scalar.
  inline friend csVector3 operator* (const csVector3& v, float f)
  { return csVector3(v.x*f, v.y*f, v.z*f); }

  /// Multiply a vector and a scalar.
  inline friend csVector3 operator* (float f, const csVector3& v)
  { return csVector3(v.x*f, v.y*f, v.z*f); }


  /// Multiply a vector and a scalar int.
  inline friend csVector3 operator* (const csVector3& v, int f)
  { return v * (float)f; }

  /// Multiply a vector and a scalar int.
  inline friend csVector3 operator* (int f, const csVector3& v)
  { return v * (float)f; }

  /// Divide a vector by a scalar.
  inline friend csVector3 operator/ (const csVector3& v, float f)
  { f = 1.0f/f; return csVector3(v.x*f, v.y*f, v.z*f); }


  /// Divide a vector by a scalar int.
  inline friend csVector3 operator/ (const csVector3& v, int f)
  { return v / (float)f; }

  /// Check if two vectors are equal.
  inline friend bool operator== (const csVector3& v1, const csVector3& v2)
  { return v1.x==v2.x && v1.y==v2.y && v1.z==v2.z; }

  /// Check if two vectors are not equal.
  inline friend bool operator!= (const csVector3& v1, const csVector3& v2)
  { return v1.x!=v2.x || v1.y!=v2.y || v1.z!=v2.z; }

  /// Project one vector onto another.
  inline friend csVector3 operator>> (const csVector3& v1, const csVector3& v2)
  { return v2*(v1*v2)/(v2*v2); }

  /// Project one vector onto another.
  inline friend csVector3 operator<< (const csVector3& v1, const csVector3& v2)
  { return v1*(v1*v2)/(v1*v1); }

  /// Test if each component of a vector is less than a small epsilon value.
  inline friend bool operator< (const csVector3& v, float f)
  { return ABS(v.x)<f && ABS(v.y)<f && ABS(v.z)<f; }

  /// Test if each component of a vector is less than a small epsilon value.
  inline friend bool operator> (float f, const csVector3& v)
  { return ABS(v.x)<f && ABS(v.y)<f && ABS(v.z)<f; }

  /// Returns n-th component of the vector.
  inline float operator[] (int n) const { return !n?x:n&1?y:z; }

  /// Returns n-th component of the vector.
  inline float & operator[] (int n) { return !n?x:n&1?y:z; }

  /// Add another vector to this vector.
  inline csVector3& operator+= (const csVector3& v)
  {
    x += v.x;
    y += v.y;
    z += v.z;

    return *this;
  }

  /// Subtract another vector from this vector.
  inline csVector3& operator-= (const csVector3& v)
  {
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return *this;
  }

  /// Multiply this vector by a scalar.
  inline csVector3& operator*= (float f)
  { x *= f; y *= f; z *= f; return *this; }

  /// Divide this vector by a scalar.
  inline csVector3& operator/= (float f)
  { f = 1.0f / f; x *= f; y *= f; z *= f; return *this; }

  /// Unary + operator.
  inline csVector3 operator+ () const { return *this; }

  /// Unary - operator.
  inline csVector3 operator- () const { return csVector3(-x,-y,-z); }

  /// Set the value of this vector.
  inline void Set (float sx, float sy, float sz) { x = sx; y = sy; z = sz; }

  /// Set the value of this vector.
  inline void Set (const csVector3& v) { x = v.x; y = v.y; z = v.z; }

  /// Returns the norm of this vector.
  float Norm () const;

  /// Return the squared norm (magnitude) of this vector.
  float SquaredNorm () const
  { return x * x + y * y + z * z; }

  /**
   * Returns the unit vector in the direction of this vector.
   * Attempting to normalize a zero-vector will result in a divide by
   * zero error.  This is as it should be... fix the calling code.
   */
  csVector3 Unit () const { return (*this)/(this->Norm()); }

  /// Returns the norm (magnitude) of a vector.
  inline static float Norm (const csVector3& v) { return v.Norm(); }

  /// Normalizes a vector to a unit vector.
  inline static csVector3 Unit (const csVector3& v) { return v.Unit(); }

  /// Scale this vector to length = 1.0;
  void Normalize ();

  /// Query if the vector is zero
  inline bool IsZero () const
  { return (x == 0) && (y == 0) && (z == 0); }
};

#endif // __CS_VECTOR3_H__
