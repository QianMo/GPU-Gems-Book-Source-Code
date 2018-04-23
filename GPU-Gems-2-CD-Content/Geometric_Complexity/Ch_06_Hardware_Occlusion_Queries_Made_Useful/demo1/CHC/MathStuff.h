//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

/*
a few simple mathematical routines; not the most efficient ones, but easy to understand.
*/

#ifndef MathStuffH
#define MathStuffH

#include <Mathematic/Vector3.h>
#include <Mathematic/Matrix4.h>
#include "DataTypes.h"

//min and max are the two extreme points of an AABB containing all the points
extern void calcCubicHull(V3& min, V3& max, const VecPoint& ps);
// mulHomogenPoint each point of VecPoint
extern void transformVecPoint(VecPoint&, const M4&);
// transformVecPoint each VecPoint of Object
extern void transformObject(Object&, const M4&);

//calculates the six polygons defining an view frustum
extern void calcViewFrustObject(Object&, const Vector3x8&);
//the given object is clipped by the given AABox; the object is assumed closed
//and is closed after the clipping
extern void clipObjectByAABox(Object&, const AABox&);
//extrudes the object into -lightDir and clippes by the AABox the defining points are returned
extern void includeObjectLightVolume(VecPoint& points, const Object&,
	const V3& lightDir, const AABox& sceneAABox);
//calculates the ViewFrustum Object	clippes this Object By the sceneAABox and
//extrudes the object into -lightDir and clippes by the AABox the defining points are returned
extern void calcFocusedLightVolumePoints(VecPoint& points, const M4& invEyeProjView,
	const V3& lightDir, const AABox& sceneAABox);

#endif
