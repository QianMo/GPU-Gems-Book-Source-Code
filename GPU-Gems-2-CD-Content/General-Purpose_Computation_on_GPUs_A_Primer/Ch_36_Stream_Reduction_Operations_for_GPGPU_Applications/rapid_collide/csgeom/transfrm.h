/*
    Copyright (C) 1998-2001 by Jorrit Tyberghein
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

#ifndef __CS_TRANSFORM_H__
#define __CS_TRANSFORM_H__

#include "csgeom/matrix3.h"

class csReversibleTransform;

/**
 * A class which defines a transformation from one coordinate system to
 * another. The two coordinate systems are refered to as 'other'
 * and 'this'. The transform defines a transformation from 'other'
 * to 'this'.
 */
class csTransform
{
protected:
  /// Transformation matrix from 'other' space to 'this' space.
  csMatrix3 m_o2t;
  /// Location of the origin for 'this' space.
  csVector3 v_o2t;

public:
  /**
   * Initialize with the identity transformation.
   */
  csTransform () : m_o2t (), v_o2t (0, 0, 0) {}

  /**
   * Initialize with the given transformation. The transformation
   * is given as a 3x3 matrix and a vector. The transformation
   * is defined to mean T=M*(O-V) with T the vector in 'this' space,
   * O the vector in 'other' space, M the transformation matrix and
   * V the transformation vector.
   */
  csTransform (const csMatrix3& other2this, const csVector3& origin_pos) :
  	m_o2t (other2this), v_o2t (origin_pos) {}

  /**
   * Get 'other' to 'this' transformation matrix. This is the 3x3
   * matrix M from the transform equation T=M*(O-V).
   */
  inline const csMatrix3& GetO2T () const { return m_o2t; }

  /**
   * Get 'world' to 'this' translation. This is the vector V
   * from the transform equation T=M*(O-V).
   */
  inline const csVector3& GetO2TTranslation () const { return v_o2t; }

  /**
   * Get origin of transformed coordinate system.
   */
  inline const csVector3& GetOrigin () const { return v_o2t; }

  /**
   * Set 'other' to 'this' transformation matrix.
   * this is the 3x3 matrix M from the transform equation T=M*(O-V).
   */
  virtual void SetO2T (const csMatrix3& m) { m_o2t = m; }

  /**
   * Set 'world' to 'this' translation. This is the vector V
   * from the transform equation T=M*(O-V).
   */
  virtual void SetO2TTranslation (const csVector3& v) { v_o2t = v; }

  /**
   * Set origin of transformed coordinate system.
   */
  inline void SetOrigin (const csVector3& v) { SetO2TTranslation (v); }

  /**
   * Move the 'other' to 'this' translation by a specified amount.
   */
  inline void Translate (const csVector3& v) { SetO2TTranslation (v_o2t + v); }

  /**
   * Transform vector in 'other' space v to a vector in 'this' space.
   * This is the basic transform function.
   */
  inline csVector3 Other2This (const csVector3& v) const
  {
    return m_o2t * (v - v_o2t);
  }

  /**
   * Convert vector v in 'other' space to a vector in 'this' space.
   * Use the origin of 'other' space.
   */
  csVector3 Other2ThisRelative (const csVector3& v) const
  { return m_o2t * v; }

  /**
   * Convert a plane in 'other' space to 'this' space.
   */
  csPlane3 Other2This (const csPlane3& p) const;

  /**
   * Convert a plane in 'other' space to 'this' space.
   * This version ignores translation.
   */
  csPlane3 Other2ThisRelative (const csPlane3& p) const;

  /**
   * Convert a plane in 'other' space to 'this' space. This is an optimized
   * version for which a point on the new plane is known (point). The result
   * is stored in 'result'.
   */
  void Other2This (const csPlane3& p, const csVector3& point,
  	csPlane3& result) const;

  /**
   * Apply a transformation to a 3D vector.
   */
  friend csVector3 operator* (const csVector3& v, const csTransform& t);

  /// Apply a transformation to a 3D vector.
  friend csVector3 operator* (const csTransform& t, const csVector3& v);
  /// Apply a transformation to a 3D vector.
  friend csVector3& operator*= (csVector3& v, const csTransform& t);
  /// Apply a transformation to a Plane.
  friend csPlane3 operator* (const csPlane3& p, const csTransform& t);
  /// Apply a transformation to a Plane.
  friend csPlane3 operator* (const csTransform& t, const csPlane3& p);
  /// Apply a transformation to a Plane.
  friend csPlane3& operator*= (csPlane3& p, const csTransform& t);

  /// Multiply a matrix with the transformation matrix.
  friend csMatrix3 operator* (const csMatrix3& m, const csTransform& t);
  /// Multiply a matrix with the transformation matrix.
  friend csMatrix3 operator* (const csTransform& t, const csMatrix3& m);
  /// Multiply a matrix with the transformation matrix.
  friend csMatrix3& operator*= (csMatrix3& m, const csTransform& t);
  /// Combine two transforms, rightmost first.
  friend csTransform operator* (const csTransform& t1,
                              const csReversibleTransform& t2);

  /**
   * Return a transform that represents a mirroring across a plane.
   * This function will return a csTransform which represents a reflection
   * across the plane pl.
   */
  static csTransform GetReflect (const csPlane3& pl);
};

/**
 * A class which defines a reversible transformation from one coordinate
 * system to another by maintaining an inverse transformation matrix.
 * This version is similar to csTransform (in fact, it is a sub-class)
 * but it is more efficient if you plan to do inverse transformations
 * often.
 */
class csReversibleTransform : public csTransform
{
protected:
  /// Inverse transformation matrix ('this' to 'other' space).
  csMatrix3 m_t2o;
  
  /**
   * Initialize transform with both transform matrix and inverse tranform.
   */
  csReversibleTransform (const csMatrix3& o2t, const csMatrix3& t2o, 
    const csVector3& pos) : csTransform (o2t,pos), m_t2o (t2o) {}

public:
  /**
   * Initialize with the identity transformation.
   */
  csReversibleTransform () : csTransform (), m_t2o () {}

  ///Daniel: I think that this matrix is in row major order...
  ///but it appears to look for the matrix that takes from world to object
  ///hence it is the transpose... transpose of transpose is identty...which is what I pass in below
  csReversibleTransform (const float *m):
		  csTransform (
			  csMatrix3 (m[0],m[3],m[6],
						 m[1],m[4],m[7],
						 m[2],m[5],m[8]).GetInverse(),
			  csVector3 (m[12],m[13],m[14])),
		  m_t2o(m[0],m[3],m[6],
				m[1],m[4],m[7],
				m[2],m[5],m[8]) {
  }
  /**
   * Initialize with the given transformation. The transformation
   * is given as a 3x3 matrix and a vector. The transformation
   * is defined to mean T=M*(O-V) with T the vector in 'this' space,
   * O the vector in 'other' space, M the transformation matrix and
   * V the transformation vector.
   */
  csReversibleTransform (const csMatrix3& o2t, const csVector3& pos) :
    csTransform (o2t,pos) { m_t2o = m_o2t.GetInverse (); }

  /**
   * Initialize with the given transformation.
   */
  csReversibleTransform (const csTransform& t) :
    csTransform (t) { m_t2o = m_o2t.GetInverse (); }

  csReversibleTransform (const csReversibleTransform& t) :
    csTransform (t) { m_t2o = t.m_t2o; }

   /**
   * Get 'this' to 'other' transformation matrix.
   */
  inline const csMatrix3& GetT2O () const { return m_t2o; }

  /**
   * Get 'this' to 'other' translation.
   */
  inline csVector3 GetT2OTranslation () const { return -m_o2t*v_o2t; }

  /**
   * Get the inverse of this transform.
   */
  csReversibleTransform GetInverse () const 
  { return csReversibleTransform (m_t2o, m_o2t, -m_o2t*v_o2t); } 

  /**
   * Set 'other' to 'this' transformation matrix.
   */
  virtual void SetO2T (const csMatrix3& m) 
  { m_o2t = m;  m_t2o = m_o2t.GetInverse (); }

  /**
   * Set 'this' to 'other' transformation matrix.
   */
  virtual void SetT2O (const csMatrix3& m) 
  { m_t2o = m;  m_o2t = m_t2o.GetInverse (); }

  /**
   * Convert vector v in 'this' space to 'other' space.
   * This is the basic inverse transform operation.
   */
  csVector3 This2Other (const csVector3& v) const
  { return v_o2t + m_t2o * v; }

  /**
   * Convert vector v in 'this' space to a vector in 'other' space,
   * relative to local origin.
   */
  inline csVector3 This2OtherRelative (const csVector3& v) const
  { return m_t2o * v; }

  /**
   * Convert a plane in 'this' space to 'other' space.
   */
  csPlane3 This2Other (const csPlane3& p) const;

  /**
   * Convert a plane in 'this' space to 'other' space.
   * This version ignores translation.
   */
  csPlane3 This2OtherRelative (const csPlane3& p) const;

  /**
   * Convert a plane in 'this' space to 'other' space. This is an optimized
   * version for which a point on the new plane is known (point). The result
   * is stored in 'result'.
   */
  void This2Other (const csPlane3& p, const csVector3& point,
  	csPlane3& result) const;


  /**
   * Rotate the transform by the angle (radians) around the given vector,
   * in other coordinates.
   * Note: this function rotates the transform, not the coordinate system.
   */
  void RotateOther (const csVector3& v, float angle);

  /**
   * Rotate the transform by the angle (radians) around the given vector,
   * in these coordinates.
   * Note: this function rotates the tranform, not the coordinate system.
   */
  void RotateThis (const csVector3& v, float angle);

  /**
   * Use the given transformation matrix, in other space,
   * to reorient the transformation.
   * Note: this function rotates the transformation, not the coordinate system.
   */
  void RotateOther (const csMatrix3& m) { SetT2O (m * m_t2o); }

  /**
   * Use the given transformation matrix, in this space,
   * to reorient the transformation.
   * Note: this function rotates the transformation, not the coordinate system.
   */
  void RotateThis (const csMatrix3& m) { SetT2O (m_t2o * m); }

  /**
   * Let this transform look at the given (x,y,z) point, using up as
   * the up-vector. 'v' should be given relative to the position
   * of the origin of this transform.
   */
  void LookAt (const csVector3& v, const csVector3& up);

  /// Reverse a transformation on a 3D vector.
  friend csVector3 operator/ (const csVector3& v, const csReversibleTransform& t); 
  /// Reverse a transformation on a 3D vector.
  friend csVector3& operator/= (csVector3& v, const csReversibleTransform& t); 
  /// Reverse a transformation on a Plane.
  friend csPlane3 operator/ (const csPlane3& p, const csReversibleTransform& t);
  /// Reverse a transformation on a Plane.
  friend csPlane3& operator/= (csPlane3& p, const csReversibleTransform& t);
  /// Combine two transforms, with the rightmost being applied first.
  friend csReversibleTransform& operator*= (csReversibleTransform& t1,
                                          const csReversibleTransform& t2)
  {
    t1.v_o2t = t2.m_t2o*t1.v_o2t;
    t1.v_o2t += t2.v_o2t;
    t1.m_o2t *= t2.m_o2t;
    t1.m_t2o *= t1.m_t2o;
    return t1;
  }
  /// Combine two transforms, with the rightmost being applied first.
  friend csReversibleTransform operator* (const csReversibleTransform& t1,
                                        const csReversibleTransform& t2)
  {
    return csReversibleTransform (t1.m_o2t*t2.m_o2t, t2.m_t2o*t1.m_t2o, 
                             t2.v_o2t + t2.m_t2o*t1.v_o2t); 
  }
  /// Combine two transforms, with the rightmost being applied first.
  friend csTransform operator* (const csTransform& t1, 
                              const csReversibleTransform& t2);
  /// Combine two transforms, reversing t2 then applying t1.
  friend csReversibleTransform& operator/= (csReversibleTransform& t1,
                                          const csReversibleTransform& t2);
  /// Combine two transforms, reversing t2 then applying t1.
  friend csReversibleTransform operator/ (const csReversibleTransform& t1,
                                        const csReversibleTransform& t2);
};

/**
 * A class which defines a reversible transformation from one coordinate
 * system to another by maintaining an inverse transformation matrix.
 * This is a variant which only works on orthonormal transformations (like
 * the camera transformation) and is consequently much more optimal.
 */
class csOrthoTransform : public csReversibleTransform
{
public:
  /**
   * Initialize with the identity transformation.
   */
  csOrthoTransform () : csReversibleTransform () {}

  /**
   * Initialize with the given transformation.
   */
  csOrthoTransform (const csMatrix3& o2t, const csVector3& pos) :
    csReversibleTransform (o2t, o2t.GetTranspose (), pos) { }

  /**
   * Initialize with the given transformation.
   */
  csOrthoTransform (const csTransform& t) :
    csReversibleTransform (t.GetO2T (), t.GetO2T ().GetTranspose (), t.GetO2TTranslation ()) { }

  /**
   * Set 'other' to 'this' transformation matrix.
   */
  virtual void SetO2T (const csMatrix3& m) 
  { m_o2t = m;  m_t2o = m_o2t.GetTranspose (); }

  /**
   * Set 'this' to 'other' transformation matrix.
   */
  virtual void SetT2O (const csMatrix3& m) 
  { m_t2o = m;  m_o2t = m_t2o.GetTranspose (); }
};

#endif // __CS_TRANSFORM_H__
