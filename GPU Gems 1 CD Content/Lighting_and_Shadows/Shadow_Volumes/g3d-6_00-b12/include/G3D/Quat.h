/**
  @file Quat.h
 
  Quaternion
  
  @maintainer Morgan McGuire, matrix@graphics3d.com
  
  @created 2002-01-23
  @edited  2003-09-28
 */

#ifndef G3D_QUAT_H
#define G3D_QUAT_H

#include "G3D/g3dmath.h"
#include "G3D/Vector3.h"
#include "G3D/Matrix3.h"
#include <string>

namespace G3D {

/**
  Unit quaternions are used in computer graphics to represent
  rotation about an axis.  Any 3x3 rotation matrix can
  be stored as a quaternion.

  Do not subclass
 */
class Quat {
private:
    /**
     q = [sin(angle / 2) * axis, cos(angle / 2)]
    
     In Watt & Watt's notation, s = w, v = (x, y, z)
     */
    float x, y, z, w;

    Quat(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}

public:

    /**
     Initializes to a zero degree rotation.
     */
    Quat() : x(0), y(0), z(0), w(1) {}

    Quat(
        const Vector3&      axis,
        double              angle);

    Quat(
        const Matrix3& rot);

    void toAxisAngle(
        Vector3&            axis,
        double&             angle) const;

    Matrix3 toRotationMatrix() const;

    void toRotationMatrix(
        Matrix3&            rot) const;

    
    /**
     Computes the linear interpolation of this to
     other at time alpha.
     */
    Quat lerp(
        const Quat&         other,
        double              alpha) const;

    /**
     Raise this quaternion to a power.  For a rotation, this is
     the effect of rotating x times as much as the original
     quaterion.
     */
    inline Quat pow(double x) const;

    /**
     Quaternion multiplication (composition of rotations).
     Note that this does not commute.
     */
    Quat operator*(const Quat& other) const;

    /**
     Quaternion magnitude (sum squares; no sqrt).
     */
    inline float magnitude() const;


private:

    
    Quat operator- (const Quat& other) const;
    double dot(const Quat& other) const;

    /**
      This is private because I don't want to make the order of elements
      publicly visible.
     */
    // access quaternion as q[0] = q.x, q[1] = q.y, q[2] = q.z, q[3] = q.w
    //
    // WARNING.  These member functions rely on
    // (1) Quat not having virtual functions
    // (2) the data packed in a 4*sizeof(float) memory block
    float& operator[] (int i) const;
    operator float* ();
    operator const float* () const;


};

}

#include "Quat.inl"


#endif

