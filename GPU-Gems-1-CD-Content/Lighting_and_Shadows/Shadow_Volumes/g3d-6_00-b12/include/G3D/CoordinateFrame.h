/**
 @file CoordinateFrame.h

 @maintainer Morgan McGuire, matrix@graphics3d.com
 
 @created 2001-03-04
 @edited  2004-01-10

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
*/

#ifndef G3D_COORDINATEFRAME_H
#define G3D_COORDINATEFRAME_H

#include "G3D/Vector3.h"
#include "G3D/Vector4.h"
#include "G3D/Ray.h"
#include "G3D/Matrix3.h"
#include "G3D/Array.h"
#include <math.h>
#include <string>
#include <stdio.h>
#include <cstdarg>
#include <assert.h>

namespace G3D {

/**
 An RT transformation.
 */
class CoordinateFrame {

public:

    /**
     The direction an object "looks" relative to its own axes.
     */
    static const float				zLookDirection;

    /**
     Takes object space points to world space.
     */
    Matrix3							rotation;

    /**
     Takes object space points to world space.
     */
    Vector3							translation;

    /**
     Initializes to the identity coordinate frame.
     */
    inline CoordinateFrame() : 
        rotation(Matrix3::IDENTITY), translation(Vector3::ZERO) {
    }

	CoordinateFrame(const Vector3& _translation) :
        rotation(Matrix3::IDENTITY), translation(_translation) {
	}

    CoordinateFrame(const Matrix3 &rotation, const Vector3 &translation) :
        rotation(rotation), translation(translation) {
    }

    CoordinateFrame(class BinaryInput& b);

    void deserialize(class BinaryInput& b);
    void serialize(class BinaryOutput& b) const;

    CoordinateFrame(const CoordinateFrame &other) :
        rotation(other.rotation), translation(other.translation) {}

    /**
      Computes the inverse of this coordinate frame.
     */
    inline CoordinateFrame inverse() const {
        CoordinateFrame out;
        out.rotation = rotation.transpose();
        out.translation = out.rotation * -translation;
        return out;
    }

    virtual ~CoordinateFrame() {}

    class Matrix4 toMatrix4() const;

    /**
     Produces an XML serialization of this coordinate frame.
     */
    std::string toXML() const;


    /*
     Returns the heading as an angle in radians, where
    north is 0 and west is PI/2
     */
    inline float getHeading() const {
        Vector3 look = rotation.getColumn(2);
        float angle = (float) atan2( -look.z, look.x);
        return angle;
    }

    /*
     Takes the coordinate frame into object space.
     this->inverse() * c
     */
    inline CoordinateFrame toObjectSpace(const CoordinateFrame& c) const {
        return this->inverse() * c;
    }

    inline Vector4 toObjectSpace(const Vector4& v) const {
        return this->inverse().toWorldSpace(v);
    }

    inline Vector4 toWorldSpace(const Vector4& v) const {
        return Vector4(rotation * Vector3(v.x, v.y, v.z) + translation * v.w, v.w);
    }

    /**
     Transforms the point into world space.
     */
    inline Vector3 pointToWorldSpace(const Vector3& v) const {
        return rotation * v + translation;
    }

    /**
     Transforms the point into object space.
     */
    inline Vector3 pointToObjectSpace(const Vector3& v) const {
        return this->inverse().pointToWorldSpace(v);
    }

    /**
     Transforms the vector into world space (no translation).
     */
    inline Vector3 vectorToWorldSpace(const Vector3& v) const {
        return rotation * v;
    }

    inline Vector3 normalToWorldSpace(const Vector3& v) const {
        return rotation * v;
    }

    Ray toObjectSpace(const Ray& r) const;
    Ray toWorldSpace(const Ray& r) const;

    /**
     Transforms the vector into object space (no translation).
     */
    inline Vector3 vectorToObjectSpace(const Vector3 &v) const {
        return rotation.transpose() * v;
    }

    inline Vector3 normalToObjectSpace(const Vector3 &v) const {
        return rotation.transpose() * v;
    }

    void pointToWorldSpace(const Array<Vector3>& v, Array<Vector3>& vout) const;

    void normalToWorldSpace(const Array<Vector3>& v, Array<Vector3>& vout) const;

    void vectorToWorldSpace(const Array<Vector3>& v, Array<Vector3>& vout) const;

    void pointToObjectSpace(const Array<Vector3>& v, Array<Vector3>& vout) const;

    void normalToObjectSpace(const Array<Vector3>& v, Array<Vector3>& vout) const;

    void vectorToObjectSpace(const Array<Vector3>& v, Array<Vector3>& vout) const;

    class Box toWorldSpace(const class AABox& b) const;

    class Box toWorldSpace(const class Box& b) const;

    class Plane toWorldSpace(const class Plane& p) const;

    class Sphere toWorldSpace(const class Sphere& b) const;

    class Triangle toWorldSpace(const class Triangle& t) const;

    class Box toObjectSpace(const AABox& b) const;

    class Box toObjectSpace(const Box& b) const;

    class Plane toObjectSpace(const Plane& p) const;
 
    class Sphere toObjectSpace(const Sphere& b) const;

    Triangle toObjectSpace(const Triangle& t) const;

    CoordinateFrame operator*(const CoordinateFrame &other) const {
        return CoordinateFrame(rotation * other.rotation,
                               pointToWorldSpace(other.translation));
    }

    void lookAt(const Vector3& target);

    void lookAt(
        const Vector3&  target,
        Vector3         up);

	inline Vector3 getLookVector() const {
		return rotation.getColumn(2) * zLookDirection;
	}

    /**
     If a viewer looks along the look vector, this is the viewer's "left"
     */
    inline Vector3 getLeftVector() const {
		return -rotation.getColumn(0);
	}

    inline Vector3 getRightVector() const {
		return rotation.getColumn(0);
	}

    /**
     Uses Quat.lerp to interpolate between two coordinate frames.
     */
    CoordinateFrame lerp(
        const CoordinateFrame&  other,
        double                  alpha) const;

};

} // namespace

#endif
