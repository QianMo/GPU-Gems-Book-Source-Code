//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

/*
a few simple mathematical routines; not the most efficient ones, but easy to understand.
*/

#ifndef MathStuffH
#define MathStuffH

#include "DataTypes.h"

//PI
extern const double PI;
//PI/2
extern const double PI_2;
//PI/180
extern const double PI_180;

//vector = (0,0,0)
extern const Vector3 ZERO;
//vector = (1,0,0)
extern const Vector3 UNIT_X;
//vector = (0,1,0)
extern const Vector3 UNIT_Y;
//vector = (0,0,1)
extern const Vector3 UNIT_Z;

//[1,0,0,0]
//[0,1,0,0]
//[0,0,1,0]
//[0,0,0,1]
extern const Matrix4x4 IDENTITY;

//absolut value of double
extern double absDouble(const double);
//signum of double
extern double signDouble(const double);
//clamp value in between min and max borders
extern void clamp(double* value, const double min, const double max);
//result = pos+t*dir
extern void linCombVector3(Vector3 result, const Vector3 pos, const Vector3 dir, const double t);
extern double dot(const Vector3, const Vector3);
extern void cross(Vector3, const Vector3, const Vector3);
extern void normalize(Vector3);
extern double squaredLength(const Vector3);

//[x,0,0,0]
//[0,y,0,0]
//[0,0,z,0]
//[0,0,0,1]
extern void makeScaleMtx(Matrix4x4 output, const double x, const double y, const double z);
//rotation around x axis
extern void makeRotationXMtx(Matrix4x4 output, const double rad);
//rotation around y axis
extern void makeRotationYMtx(Matrix4x4 output, const double rad);
//rotation around z axis
extern void makeRotationZMtx(Matrix4x4 output, const double rad);
//translation
extern void makeTranslationMtx(Matrix4x4 output, Vector3 translation);

//output = a*b (matrix product)
extern void mult(Matrix4x4 output, const Matrix4x4 a, const Matrix4x4 b);
//output = i^(-1)
extern void invert(Matrix4x4 output, const Matrix4x4 i);
//output = look from position:pos into direction:dir with up-vector:up
extern void look(Matrix4x4 output, const Vector3 pos, const Vector3 dir, const Vector3 up);
//make a scaleTranslate matrix that includes the two values vMin and vMax
extern void scaleTranslateToFit(Matrix4x4 output, const Vector3 vMin, const Vector3 vMax);
//output is initialized with the same result as glPerspective vFovy in degrees
extern void perspectiveDeg(Matrix4x4 output, const double vFovy, const double vAspect,
	const double vNearDis, const double vFarDis);

//calc matrix-vector product; input has assumed homogenous component w = 1
//before the output is  written homogen division is performed (w = 1)
extern void mulHomogenPoint(Vector3 output, const Matrix4x4 m, const Vector3 v);
//min and max are the two extreme points of an AABB containing all the points
extern void calcCubicHull(Vector3 min, Vector3 max, const Vector3* ps, const int size);
//calculates the world coordinates of the view frustum corner points
//input matrix is the (eyeProj*eyeView)^(-1) matrix
extern void calcViewFrustumWorldCoord(Vector3x8, const Matrix4x4);
// mulHomogenPoint each point of VecPoint
extern void transformVecPoint(struct VecPoint* , const Matrix4x4);
// transformVecPoint each VecPoint of Object
extern void transformObject(struct Object*, const Matrix4x4);
//min and max are the two extreme points of an AABB containing all the points of the object
extern void calcObjectCubicHull(Vector3 min, Vector3 max, const struct Object);

//calculates the six polygons defining an view frustum
extern void calcViewFrustObject(struct Object*, const Vector3x8);
//the given object is clipped by the given AABox; the object is assumed closed
//and is closed after the clipping
extern void clipObjectByAABox(struct Object*, const struct AABox);
//extrudes the object into -lightDir and clippes by the AABox the defining points are returned
extern void includeObjectLightVolume(struct VecPoint* points, const struct Object,
	const Vector3 lightDir, const struct AABox sceneAABox);
//calculates the ViewFrustum Object	clippes this Object By the sceneAABox and
//extrudes the object into -lightDir and clippes by the AABox the defining points are returned
extern void calcFocusedLightVolumePoints(struct VecPoint* points,const Matrix4x4 invEyeProjView,
	const Vector3 lightDir,const struct AABox sceneAABox);
// calculates the index of the nearest vertex in the AAB vertex set
// obtainened by calcAABoxPoints according to the normal vector of a clip plane
extern int calcAABNearestVertexIdx(Vector3 clipPlaneNormal);
// calculates the index of the farthest vertex in the AAB vertex set
// obtainened by calcAABoxPoints according to the normal vector of a clip plane
extern int calcAABFarthestVertexIdx(Vector3 clipPlaneNormal);

extern double pointPlaneDistance(const struct Plane A, const Vector3 p);

extern int pointBeforePlane(const struct Plane A, const Vector3 p);
// calculates the view frustum planes of the frustum defined by the eyeProjView matrix
extern void calcViewFrustumPlanes(struct VecPlane* planes, const Matrix4x4 eyeProjView);
// calculates the np-indices of an aab for a set of 6 clipping planes
// the indices are stored in the form (n0, p0, n1, p1, ..., n11, p11) in the array vertexIdx 
extern void calcAABNPVertexIndices(int *vertexIndices, const struct VecPlane planes);
// combines two AABs, stores result in a
extern void combineAABoxes(struct AABox *a, const struct AABox b);
// calculates volume of this aab
extern double calcAABoxVolume(const struct AABox aab);
// calculates surface of this aab
double calcAABoxSurface(const struct AABox aab);
// clips box aab by box enclosed
extern void clipAABoxByAABox(struct AABox *aab, const struct AABox enclosed);
// computes world position center of an aab
extern void calcAABoxCenter(Vector3 vec, const struct AABox aab);

extern void rotateVectorX(Vector3 v, double rad);
extern void rotateVectorY(Vector3 v, double rad);
extern void rotateVectorZ(Vector3 v, double rad);
// rotates vector around axis(must be unit vector)
extern void rotateVector(Vector3 result, const Vector3 p, const float rad, const Vector3 axis);
// generates a normal to vector v
extern void vectorNormal(Vector3 result, const Vector3 v);
#endif
