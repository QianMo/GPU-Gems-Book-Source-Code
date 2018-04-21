/**
 @file Ray.h
 
 Ray class
 
 @maintainer Morgan McGuire, matrix@graphics3d.com
 
 @created 2002-07-12
 @edited  2004-01-12
 */

#ifndef G3D_RAY_H
#define G3D_RAY_H

#include "G3D/Vector3.h"

namespace G3D {

/**
 A 3D Ray.
 */
class Ray {
private:
    Ray(const Vector3& origin, const Vector3& direction) {
        this->origin    = origin;
        this->direction = direction;
    }

public:
    Vector3         origin;

    /**
     Not unit length
     */
    Vector3         direction;

    Ray() : origin(Vector3::ZERO3), direction(Vector3::ZERO3) {}

	Ray(class BinaryInput& b);
	void serialize(class BinaryOutput& b) const;
	void deserialize(class BinaryInput& b);

    virtual ~Ray() {}

    /**
     Creates a Ray from a origin and a (nonzero) direction.
     */
    static Ray fromOriginAndDirection(const Vector3& point, const Vector3& direction) {
        return Ray(point, direction);
    }

    Ray unit() const {
        return Ray(origin, direction.unit());
    }

    /**
     Returns the closest point on the Ray to point.
     */
    Vector3 closestPoint(const Vector3& point) const {
        double t = direction.dot(point - this->origin);
        if (t < 0) {
            return this->origin;
        } else {
            return this->origin + direction * t;
        }
    }

    /**
     Returns the closest distance between point and the Ray
     */
    double distance(const Vector3& point) const {
        return (closestPoint(point) - this->origin).length();
    }

    /**
     Returns the point where the Ray and plane intersect.  If there
     is no intersection, returns a point at infinity.
     */
    Vector3 intersection(const class Plane& plane) const;

    /**
     Returns the distance until intersection with the (solid) sphere.
     Will be 0 if inside the sphere, inf if there is no intersection.

     The ray direction is <B>not</B> normalized.  If the ray direction
     has unit length, the distance from the origin to intersection
     is equal to the time.  If the direction does not have unit length,
     the distance = time * direction.length().

     See also G3D::CollisionDetection.
     */
    float intersectionTime(const class Sphere& sphere) const;

    float intersectionTime(const class Plane& plane) const;

    float intersectionTime(const class Box& box) const;

    float intersectionTime(const class Triangle& triangle) const;

    /**
     Ray-triangle intersection
     */
    float intersectionTime(
        const Vector3& v0,
        const Vector3& v1,
        const Vector3& v2) const;

};

}// namespace


#endif
