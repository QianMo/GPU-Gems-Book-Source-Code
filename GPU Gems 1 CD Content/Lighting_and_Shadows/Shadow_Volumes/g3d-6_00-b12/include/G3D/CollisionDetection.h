/**
  @file CollisionDetection.h
  

  Moving collision detection for simple primitives.

  @author Morgan McGuire, matrix@graphics3d.com
  @cite Spherical collision based on Paul Nettle's ftp://ftp.3dmaileffects.com/pub/FluidStudios/CollisionDetection/Fluid_Studios_Generic_Collision_Detection_for_Games_Using_Ellipsoids.pdf and comments by Max McGuire.  Ray-sphere intersection by Eric Haines.  Thanks to Max McGuire of Iron Lore for various bug fixes.

  @created 2001-11-19
  @edited  2004-01-04

  Copyright 2000-2003, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_COLLISIONDETECTION_H
#define G3D_COLLISIONDETECTION_H

#include "G3D/Vector3.h"
#include "G3D/Plane.h"
#include "G3D/Box.h"
#include "G3D/Triangle.h"
#include "G3D/Array.h"

namespace G3D {


/**
  Collision detection primitives and tools for building
  higher order collision detection schemes.

  These routines provide <I>moving</I> collision detection.
  This allows you to detect collisions that occur during a period
  of time, as opposed to the intersection of two static bodies.
  The routines detect collisions between static primitives and moving
  spheres or points.  Because you can choose your reference frame,
  however, you can assign a velocity to a primitive by subtracting
  that same velocity from the sphere or point.

  The Magic Software Library provides sophisticated static collision
  detection if that is needed.

  Collisions are detected for single-sided objects only.  That is,
  no collision is detected when <I>leaving</I> a primitive or passing
  through a plane or triangle opposite the normal.

  For a sphere, the collision location returned is the point in world
  space where the surface of the sphere and the fixed object meet.
  It is <B>not</B> not the position of the center of the sphere at
  the time of the collision.

  The collision normal returned is the surface normal to the fixed
  object at the collision location.
 */
class CollisionDetection {
private:

	/** Provides the default argument for the location parameter */
	static Vector3 ignore;

    // Static class!
    CollisionDetection() {}
    virtual ~CollisionDetection() {}

public:

    /**
     Returns the penetration depth (negative if there is no penetration)
     of the two spheres and a series of contact points.  The normal
     returned points <B>away</B> from the object A, although it may
     represent a perpendicular to either the faces of object B or object A
     depending on their relative orientations.
     */
    static float penetrationDepthForFixedSphereFixedSphere(
        const class Sphere&     sphereA,
        const Sphere&           sphereB,
        Array<Vector3>&         contactPoints,
        Vector3&                outNormalA = ignore);
    
    /**
     @cite Adapted from Jim Arvo's method in Graphics Gems
     TODO: See also http://www.win.tue.nl/~gino/solid/gdc2001depth.pdf
     */
    static float penetrationDepthForFixedSphereFixedBox(
        const Sphere&   sphere,
        const Box&      box,
        Array<Vector3>& contactPoints,
        Vector3&        outNormal);
                                                                 
    static float penetrationDepthForFixedSphereFixedPlane(
        const Sphere&           sphereA,
        const class Plane&      planeB,
        Array<Vector3>&         contactPoints,
        Vector3&                outNormalA = ignore);

    static float penetrationDepthForFixedBoxFixedPlane(
        const Box&          box,
        const Plane&        plane,
        Array<Vector3>&     contactPoints,
        Vector3&            outNormalA = ignore);
    
    /**
     Returns the amount of time until the point intersects the plane 
     (the plane is one sided; the point can only hit the side the 
     normal faces out of).  The return value is infReal if no 
     collision will occur, zero if the point is already in the plane.
     
     To perform a two sided collision, call twice, once for each direction
     of the plane normal.

     @param location The location of the collision is point + velocity * time.
      This is returned in location, if a collision occurs.  Otherwise location
      is the infinite vector.
     */
    static float collisionTimeForMovingPointFixedPlane(
        const Vector3&			point,
        const Vector3&			velocity,
        const class Plane&		plane,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    static float collisionTimeForMovingPointFixedTriangle(
        const Vector3&			point,
        const Vector3&			velocity,
        const Triangle&       triangle,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);


    static float collisionTimeForMovingPointFixedSphere(
        const Vector3&			point,
        const Vector3&			velocity,
        const class Sphere&		sphere,                                                       
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    static float collisionTimeForMovingPointFixedBox(
        const Vector3&			point,
        const Vector3&			velocity,
        const class  Box&		box,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

	/**
	  The 4 vertices are assumed to form a rectangle.
	 */
    static float collisionTimeForMovingPointFixedRectangle(
        const Vector3&			point,
        const Vector3&			velocity,
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&			v2,
        const Vector3&			v3,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

	static float collisionTimeForMovingPointFixedCapsule(
		const Vector3&		    point,
		const Vector3&		    velocity,
		const class Capsule&	capsule,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    /////////////////////////

    static float collisionTimeForMovingSphereFixedPlane(
        const class Sphere&		sphere,
        const Vector3&	    	velocity,
        const class Plane&		plane,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    static float collisionTimeForMovingSphereFixedTriangle(
        const class Sphere&		sphere,
        const Vector3&		    velocity,
        const Triangle&       triangle,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    static float collisionTimeForMovingSphereFixedRectangle(
        const class Sphere&		sphere,
        const Vector3&	    	velocity,
        const Vector3&	    	v0,
        const Vector3&	    	v1,
        const Vector3&	    	v2,
        const Vector3&	    	v3,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    static float collisionTimeForMovingSphereFixedBox(
        const class Sphere&		sphere,
        const Vector3&		    velocity,
        const class Box&		box,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    /**
      This won't detect a collision if the sphere is already interpenetrating the fixed sphere.
    */
	static float collisionTimeForMovingSphereFixedSphere(
		const class Sphere&		sphere,
		const Vector3&		    velocity,
		const class Sphere&	    fixedSphere,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    /**
      This won't detect a collision if the sphere is already interpenetrating the capsule
    */
	static float collisionTimeForMovingSphereFixedCapsule(
		const class Sphere&		sphere,
		const Vector3&		    velocity,
		const class Capsule&	capsule,
        Vector3&				outLocation,
        Vector3&                outNormal = ignore);

    /**
     Returns the direction the sphere should bounce if it collided with
     an object at collisionTime and collisionLocation.
     */
    static Vector3 bounceDirection(
        const class Sphere&		sphere,
        const Vector3&			velocity,
        const float				collisionTime,
        const Vector3&			collisionLocation,
        const Vector3&          collisionNormal);

    /**
     Returns the direction the sphere should slide if it collided with
     an object at collisionTime and collisionLocation and will hug the surface.
     */
    static Vector3 slideDirection(
        const class Sphere&		sphere,
        const Vector3&			velocity,
        const float				collisionTime,
        const Vector3&			collisionLocation);

    /**
     Returns the point on the line segment closest to the input point.
     */
    static Vector3 closestPointOnLineSegment(
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&			point);

    /**
     This is the fast version of the function.

     @param v0 The starting vertex of the line segment
     @param v1 The ending vertex of the line segment
     @param edgeLength The length of the segment
     @param edgeDirection The direction of the segment (unit length)
     @param point The point in question.
     */
    static Vector3 closestPointOnLineSegment(
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&          edgeDirection,
        double                  edgeLength,
        const Vector3&			point);

    /**
     Returns the point on the perimeter of the triangle closest to the input point and 
     returns it in location.  The return value is the distance to that point.
     */
    static Vector3 closestPointToTrianglePerimeter(
        const Vector3&			v0, 
        const Vector3&			v1,
        const Vector3&			v2,
        const Vector3&			point);

    static Vector3 closestPointToTrianglePerimeter(
        const Vector3           v[3],
        const Vector3           edgeDirection[3],
        const double            edgeLength[3],
        const Vector3&			point);

    /**
     Returns true if a point <B>known to be in the plane of a triangle</B> is inside of its bounds.
     */
    static bool isPointInsideTriangle(
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&			v2,
        const Vector3&			normal,
        const Vector3&			point,
        Vector3::Axis  primaryAxis = Vector3::DETECT_AXIS);

    
    /**
     Returns true if any part of the sphere is inside the box
     during the time period (inf means "ever").  Useful for
     performing bounding-box collision detection.
     */
    static bool movingSpherePassesThroughFixedBox(
        const Sphere&           sphere,
        const Vector3&          velocity,
        const Box&              box,
        double                  timeLimit = inf);

    static bool movingSpherePassesThroughFixedSphere(
        const Sphere&           sphere,
        const Vector3&          velocity,
        const Sphere&           fixedSphere,
        double                  timeLimit = inf);

    static bool fixedSolidSphereIntersectsFixedSolidSphere(
        const Sphere&           sphere1,
        const Sphere&           sphere2);

    static bool fixedSolidSphereIntersectsFixedSolidBox(
        const Sphere&           sphere,
        const Box&              box);

    /**
     Returns true if a point in the plane of a triangle is inside of its bounds.
     */
    static bool isPointInsideRectangle(
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&			v2,
        const Vector3&			v3,
        const Vector3&			normal,
        const Vector3&			point);

    /**
     Returns the point on the perimeter of the rectangle closest to the input point and 
     returns it in location.  The return value is the distance to that point.
     */
    static Vector3 closestPointToRectanglePerimeter(
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&			v2,
        const Vector3&			v3,
        const Vector3&			point);

    static Vector3 closestPointToRectangle(
        const Vector3&			v0,
        const Vector3&			v1,
        const Vector3&			v2,
        const Vector3&			v3,
        const Vector3&			point);
};

} // namespace

#endif
