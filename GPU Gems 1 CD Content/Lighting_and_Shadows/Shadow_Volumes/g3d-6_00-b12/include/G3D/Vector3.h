/**
  @file Vector3.h
 
  3D vector class
 
  @maintainer Morgan McGuire, matrix@graphics3d.com

  @cite Portions based on Dave Eberly's Magic Software Library at http://www.magic-software.com
 
  @created 2001-06-02
  @edited  2004-01-09
  Copyright 2000-2004, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_VECTOR3_H
#define G3D_VECTOR3_H

#include "G3D/g3dmath.h"
#include "G3D/Vector2.h"
#include <string>

namespace G3D {

class Vector2;    
class Vector3;
class Vector4;

/**
  <B>Swizzles</B>
 Vector classes have swizzle operators, e.g. <CODE>v.xy()</CODE>, that
 allow selection of arbitrary sub-fields.  These cannot be used as write 
 masks.  Examples

  <PRE>
Vector3 v(1, 2, 3);
Vector3 j;
Vector2 b;

b = v.xz();
j = b.xx();
</PRE>


  <B>Warning</B>

 Do not subclass-- this implementation makes assumptions about the
 memory layout.
 */
class Vector3 {    
public:
    // construction
    Vector3();
    Vector3(class BinaryInput& b);
    Vector3(float _x, float _y, float _z);
    Vector3(float coordinate[3]);
    Vector3(const Vector3& rkVector);
    Vector3(const class Vector3int16& v);

    void serialize(class BinaryOutput& b) const;
    void deserialize(class BinaryInput& b);
    
    // coordinates
    float x, y, z;

    // access vector V as V[0] = V.x, V[1] = V.y, V[2] = V.z
    //
    // WARNING.  These member functions rely on
    // (1) Vector3 not having virtual functions
    // (2) the data packed in a 3*sizeof(float) memory block
    float& operator[] (int i) const; 
    operator float* ();
    operator const float* () const;

    enum Axis {X_AXIS=0, Y_AXIS=1, Z_AXIS=2, DETECT_AXIS=-1};

    /**
     Returns the largest dimension.  Particularly convenient for determining
     which plane to project a triangle onto for point-in-polygon tests.
     */
    Axis primaryAxis() const;

    // assignment and comparison
    Vector3& operator= (const Vector3& rkVector);
    bool operator== (const Vector3& rkVector) const;
    bool operator!= (const Vector3& rkVector) const;
    unsigned int hashCode() const;
    bool fuzzyEq(const Vector3& other) const;
    bool fuzzyNe(const Vector3& other) const;

    /** Returns true if this vector has finite length. */
    bool isFinite() const;

    /** Returns true if this vector has length ~= 0 */
    bool isZero() const;

    /** Returns true if this vector has length ~= 1 */
    bool isUnit() const;
    
    // arithmetic operations
    Vector3 operator+ (const Vector3& rkVector) const;
    Vector3 operator- (const Vector3& rkVector) const;
    Vector3 operator* (float fScalar) const;
    Vector3 operator/ (float fScalar) const;
    Vector3 operator* (const Vector3& rkVecto) const;
    Vector3 operator/ (const Vector3& rkVecto) const;
    Vector3 operator- () const;
    friend Vector3 operator* (float fScalar, const Vector3& rkVector);

    // arithmetic updates
    Vector3& operator+= (const Vector3& rkVector);
    Vector3& operator-= (const Vector3& rkVector);
    Vector3& operator*= (float fScalar);
    Vector3& operator/= (float fScalar);
    Vector3& operator*= (const Vector3& rkVector);
    Vector3& operator/= (const Vector3& rkVector);

    float length () const;
    
    /**
     The result is a nan vector if the length is almost zero.
     */
    Vector3 direction() const;

    /**
     Potentially less accurate but faster than direction().
     Only works if System::hasSSE is true.
     */
    Vector3 fastDirection() const;

    /**
     Reflect this vector about the (not necessarily unit) normal.
     Note that if used for a collision or ray reflection you
     must negate the resulting vector to get a direction pointing
     <I>away</I> from the collision.

     <PRE>
       V'    N      V
                 
         r   ^   -,
          \  |  /
            \|/
     </PRE>

     See also Vector3::reflectionDirection
     */
    Vector3 reflectAbout(const Vector3& normal) const;

    /**
     Returns -reflectAbout(normal).  The length is 1. 
     <PRE>
       V'    N       V
                 
         r   ^    /
          \  |  /
            \|'-
     </PRE>
     */
    Vector3 reflectionDirection(const Vector3& normal) const;
    
    /**
     Returns the direction of a refracted ray,
     where iExit is the index of refraction for the
     previous material and iEnter is the index of refraction
     for the new material.  Like Vector3::reflectionDirection,
     the result has length 1 and is 
     pointed <I>away</I> from the intersection.

     Returns Vector3::ZERO in the case of total internal refraction.

     <PRE>
              N      V
                  
              ^    /
              |  /
              |'-
          __--
     V'<--
     </PRE>
     */
    Vector3 refractionDirection(const Vector3& normal,
        double iExit, double iEnter) const;

    inline Vector3 unit() const {
        return direction();
    }

    inline Vector3 fastUnit() const {
        return fastDirection();
    }

    float squaredLength () const;
    float dot (const Vector3& rkVector) const;
    float unitize (float fTolerance = 1e-06);
    Vector3 cross (const Vector3& rkVector) const;
    Vector3 unitCross (const Vector3& rkVector) const;

    Vector3 min(const Vector3 &v) const;
    Vector3 max(const Vector3 &v) const;

    std::string toString() const;

    inline Vector3 clamp(const Vector3& low, const Vector3& high) const {
        return Vector3(
            G3D::clamp(x, low.x, high.x),
            G3D::clamp(y, low.y, high.y),
            G3D::clamp(z, low.z, high.z));
    }

    inline Vector3 clamp(double low, double high) const {
        return Vector3(
            G3D::clamp(x, low, high),
            G3D::clamp(y, low, high),
            G3D::clamp(z, low, high));
    }

    /**
     Linear interpolation
     */
    inline Vector3 lerp(const Vector3& v, double alpha) const {
        return (*this) + (v - *this) * alpha; 
    }

    /** Gram-Schmidt orthonormalization. */
    static void orthonormalize (Vector3 akVector[3]);

    /** Random unit vector, uniformly distributed */
    static Vector3 random();

    // Input W must be initialize to a nonzero vector, output is {U,V,W}
    // an orthonormal basis.  A hint is provided about whether or not W
    // is already unit length.
    static void generateOrthonormalBasis (Vector3& rkU, Vector3& rkV,
                                          Vector3& rkW, bool bUnitLengthW = true);

    // special points
    static const Vector3 ZERO;
    static const Vector3 ZERO3;
    static const Vector3 UNIT_X;
    static const Vector3 UNIT_Y;
    static const Vector3 UNIT_Z;
    static const Vector3 INF3;
    static const Vector3 NAN3;

    // 2-char swizzles

    Vector2 xx() const;
    Vector2 yx() const;
    Vector2 zx() const;
    Vector2 xy() const;
    Vector2 yy() const;
    Vector2 zy() const;
    Vector2 xz() const;
    Vector2 yz() const;
    Vector2 zz() const;

    // 3-char swizzles

    Vector3 xxx() const;
    Vector3 yxx() const;
    Vector3 zxx() const;
    Vector3 xyx() const;
    Vector3 yyx() const;
    Vector3 zyx() const;
    Vector3 xzx() const;
    Vector3 yzx() const;
    Vector3 zzx() const;
    Vector3 xxy() const;
    Vector3 yxy() const;
    Vector3 zxy() const;
    Vector3 xyy() const;
    Vector3 yyy() const;
    Vector3 zyy() const;
    Vector3 xzy() const;
    Vector3 yzy() const;
    Vector3 zzy() const;
    Vector3 xxz() const;
    Vector3 yxz() const;
    Vector3 zxz() const;
    Vector3 xyz() const;
    Vector3 yyz() const;
    Vector3 zyz() const;
    Vector3 xzz() const;
    Vector3 yzz() const;
    Vector3 zzz() const;

    // 4-char swizzles

    Vector4 xxxx() const;
    Vector4 yxxx() const;
    Vector4 zxxx() const;
    Vector4 xyxx() const;
    Vector4 yyxx() const;
    Vector4 zyxx() const;
    Vector4 xzxx() const;
    Vector4 yzxx() const;
    Vector4 zzxx() const;
    Vector4 xxyx() const;
    Vector4 yxyx() const;
    Vector4 zxyx() const;
    Vector4 xyyx() const;
    Vector4 yyyx() const;
    Vector4 zyyx() const;
    Vector4 xzyx() const;
    Vector4 yzyx() const;
    Vector4 zzyx() const;
    Vector4 xxzx() const;
    Vector4 yxzx() const;
    Vector4 zxzx() const;
    Vector4 xyzx() const;
    Vector4 yyzx() const;
    Vector4 zyzx() const;
    Vector4 xzzx() const;
    Vector4 yzzx() const;
    Vector4 zzzx() const;
    Vector4 xxxy() const;
    Vector4 yxxy() const;
    Vector4 zxxy() const;
    Vector4 xyxy() const;
    Vector4 yyxy() const;
    Vector4 zyxy() const;
    Vector4 xzxy() const;
    Vector4 yzxy() const;
    Vector4 zzxy() const;
    Vector4 xxyy() const;
    Vector4 yxyy() const;
    Vector4 zxyy() const;
    Vector4 xyyy() const;
    Vector4 yyyy() const;
    Vector4 zyyy() const;
    Vector4 xzyy() const;
    Vector4 yzyy() const;
    Vector4 zzyy() const;
    Vector4 xxzy() const;
    Vector4 yxzy() const;
    Vector4 zxzy() const;
    Vector4 xyzy() const;
    Vector4 yyzy() const;
    Vector4 zyzy() const;
    Vector4 xzzy() const;
    Vector4 yzzy() const;
    Vector4 zzzy() const;
    Vector4 xxxz() const;
    Vector4 yxxz() const;
    Vector4 zxxz() const;
    Vector4 xyxz() const;
    Vector4 yyxz() const;
    Vector4 zyxz() const;
    Vector4 xzxz() const;
    Vector4 yzxz() const;
    Vector4 zzxz() const;
    Vector4 xxyz() const;
    Vector4 yxyz() const;
    Vector4 zxyz() const;
    Vector4 xyyz() const;
    Vector4 yyyz() const;
    Vector4 zyyz() const;
    Vector4 xzyz() const;
    Vector4 yzyz() const;
    Vector4 zzyz() const;
    Vector4 xxzz() const;
    Vector4 yxzz() const;
    Vector4 zxzz() const;
    Vector4 xyzz() const;
    Vector4 yyzz() const;
    Vector4 zyzz() const;
    Vector4 xzzz() const;
    Vector4 yzzz() const;
    Vector4 zzzz() const;



};

}

unsigned int hashCode(const G3D::Vector3& v);

#include "Vector3.inl"

#endif
