/**
  @file AABox.h
 
  Axis-aligned box class
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2004-01-10
  @edited  2004-01-10

  Copyright 2000-2004, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_AABOX_H
#define G3D_AABOX_H

#include "G3D/Vector3.h"
#include "G3D/debug.h"

namespace G3D {

/**
 An axis-aligned box.
 */
class AABox {
private:

    Vector3  lo;
    Vector3  hi;

public:

    /** Does not initialize the fields */
    inline AABox() {}

    /**
     Constructs a zero-area AABox at v.
     */
    inline AABox(const Vector3& v) {
        lo = hi = v;
    }

    inline AABox(const Vector3& low, const Vector3& high) {
        lo = low.min(high);
        hi = high.max(low);
    }

	void serialize(class BinaryOutput& b) const;

	void deserialize(class BinaryInput& b);

    inline const Vector3& low() const {
        return lo;
    }

    inline const Vector3& high() const {
        return hi;
    }

    /**
      Returns the centroid of the box.
     */
    inline Vector3 center() const {
        return (lo + hi) * 0.5;
    }

    /**
     Distance from corner(0) to the next corner along axis a.
     */
    inline double extent(int a) const {
        debugAssert(a < 3);
        return hi[a] - lo[a];
    }

    inline Vector3 extent() const {
        return hi - lo;
    }

    /**
     See Box::culledBy
     */
    bool culledBy(
        const class Plane*  plane,
        int                 numPlanes) const;

    inline bool contains(
        const Vector3&      point) const {
        return
            (point.x >= lo.x) &&
            (point.y >= lo.y) &&
            (point.y >= lo.y) &&
            (point.x <= hi.x) &&
            (point.y <= hi.y) &&
            (point.z <= hi.z);
    }

    inline double surfaceArea() const {
        Vector3 diag = hi - lo;
        return 2 * (diag.x * diag.y + diag.y * diag.z + diag.x * diag.z);
    }

    inline double volume() const {
        Vector3 diag = hi - lo;
        return diag.x * diag.y * diag.z;
    }

    Vector3 randomInteriorPoint() const;

    Vector3 randomSurfacePoint() const;

    class Box toBox() const;

    /** Returns true if there is any overlap */
    bool intersects(const AABox& other) const;

    inline unsigned int hashCode() const {
        return lo.hashCode() + hi.hashCode();
    }

    inline bool operator==(const G3D::AABox& b) {
        return (lo == b.lo) && (hi == b.hi);
    }

    void getBounds(AABox& out) const {
        out = *this;
    }
};

}

/**
 Hashing function for use with Table.
 */
inline unsigned int hashCode(const G3D::AABox& b) {
	return b.hashCode();
}


#endif
