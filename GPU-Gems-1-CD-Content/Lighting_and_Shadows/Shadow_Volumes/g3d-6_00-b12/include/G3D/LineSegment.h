/**
 @file LineSegment.h
  
 @maintainer Morgan McGuire, matrix@graphics3d.com
 
 @created 2003-02-08
 @edited  2003-02-08
 */

#ifndef G3D_LINESEGMENT_H
#define G3D_LINESEGMENT_H

#include "G3D/Vector3.h"

namespace G3D {

/**
 An finite segment of an infinite 3D line.
 */
class LineSegment {
protected:

    Vector3             _point;

    /** Not normalized */
    Vector3             direction;

    LineSegment(const Vector3& __point, const Vector3& _direction) : _point(__point), direction(_direction) {
    }

public:

	LineSegment(class BinaryInput& b);

	void serialize(class BinaryOutput& b) const;

	void deserialize(class BinaryInput& b);

    virtual ~LineSegment() {}

    /**
     * Constructs a line from two (not equal) points.
     */
    static LineSegment fromTwoPoints(const Vector3 &point1, const Vector3 &point2) {
        return LineSegment(point1, point2 - point1);
    }


    /** Returns end point 0 or 1 */
    Vector3 endPoint(int i) const;

    /**
     * Returns the closest point on the line segment to point.
     */
    Vector3 closestPoint(const Vector3 &point) const;

    /**
     Returns the distance between point and the line
     */
    double distance(const Vector3& p) const {
        return (closestPoint(p) - p).length();
    }

    double distanceSquared(const Vector3& p) const {
        return (closestPoint(p) - p).squaredLength();
    }

    /** Returns true if some part of this segment is inside the sphere */
    bool intersectsSolidSphere(const class Sphere& s) const;

    Vector3 randomPoint() const;

};

} // namespace


#endif
