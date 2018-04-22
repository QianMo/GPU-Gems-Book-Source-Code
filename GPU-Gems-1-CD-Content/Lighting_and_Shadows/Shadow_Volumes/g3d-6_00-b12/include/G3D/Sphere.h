/**
 @file Sphere.h
 
 Sphere class
 
 @maintainer Morgan McGuire, matrix@graphics3d.com
 
 @created 2001-06-02
 @edited  2004-01-11
 */

#ifndef G3D_SPHERE_H
#define G3D_SPHERE_H

#include "G3D/Vector3.h"

namespace G3D {

/**
 Sphere.
 */
class Sphere {
public:
    Vector3          center;
    float            radius;

    Sphere() {
        center = Vector3::ZERO;
        radius = 0;
    }

	Sphere(class BinaryInput& b);
	void serialize(class BinaryOutput& b) const;
	void deserialize(class BinaryInput& b);

    Sphere(
        const Vector3&  center,
        double          radius) {

        this->center = center;
        this->radius = radius;
    }

    virtual ~Sphere() {}

    /**
     Returns true if point is less than or equal to radius away from
     the center.
     */
    bool contains(const Vector3& point) const;

    /**
     Returns true if this sphere is culled by the provided set of 
     planes.  The sphere is culled if there exists at least one plane
     whose halfspace the entire sphere is not in.
     */
    bool culledBy(
        const class Plane*  plane,
        int                 numPlanes) const;

    virtual std::string toString() const;

    double volume() const;

    double surfaceArea() const;

    /**
     Uniformly distributed on the surface.
     */
    Vector3 randomSurfacePoint() const;

    /**
     Uniformly distributed on the interior (includes surface)
     */
    Vector3 randomInteriorPoint() const;

    void getBounds(class AABox& out) const;
};

} // namespace

#endif
