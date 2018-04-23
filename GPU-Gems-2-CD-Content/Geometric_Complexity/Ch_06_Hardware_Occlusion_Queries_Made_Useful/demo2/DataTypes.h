//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef DATATYPES_H
#define DATATYPES_H

typedef double Vector3[3];

typedef Vector3 Vector3x8[8];

//Plane defined through normal vector n and distance to origin d
struct Plane {
	Vector3 n;
	double d;
};

//a dynamic array of planes
struct VecPlane {
	struct Plane* plane;
	int size;
};

//a dynamic array 3d points
struct VecPoint {
	Vector3* points;
	int size;
};

//a dynamic array of point list each point list is a polygon
struct Object {
	struct VecPoint* poly;
	int size;
};

//Axis-Aligned Bounding Box defined through the two extreme points
struct AABox {
	Vector3 min;
	Vector3 max;
};

//4x4 matrix
typedef double Matrix4x4[16];

//initialisation objects (because we have no constructors)
extern const struct Object OBJECT_NULL;
extern const struct VecPoint VECPOINT_NULL;
extern const struct VecPlane VECPLANE_NULL;

//copy the 3 values of the input vector into the output vector
extern void copyVector3(Vector3, const Vector3);
//copy the 3 values of the input vector into the output vector
extern void copyVector3Values(Vector3, const double, const double, const double);
//copy the 3 values of the substraction of the two input vectors into the output vector
extern void diffVector3(Vector3, const Vector3, const Vector3);
//copy the 3 values of the addition of the two input vectors into the output vector
extern void addVector3(Vector3 result, const Vector3, const Vector3);

//copy the 16 values of the input matrix into the output matrix
extern void copyMatrix(Matrix4x4, const Matrix4x4);

extern void vecPointSetSize(struct VecPoint*, const int);
extern void emptyVecPoint(struct VecPoint*);
extern void swapVecPoint(struct VecPoint*, struct VecPoint*);
extern void copyVecPoint(struct VecPoint*, const struct VecPoint);
extern void append2VecPoint(struct VecPoint*, const Vector3);

extern void emptyVecPlane(struct VecPlane*);

extern void objectSetSize(struct Object*, const int);
extern void emptyObject(struct Object*);
extern void copyObject(struct Object*, const struct Object);
//makes 1 VecPoint out of all the VecPoints of an object
extern void convObject2VecPoint(struct VecPoint* points, const struct Object);

//calculates the 8 corner points of the given AABox
extern void calcAABoxPoints(Vector3x8 points, const struct AABox b);
//calculates the 6 planes of the given AABox
extern void calcAABoxPlanes(struct VecPlane*, const struct AABox);
// allocates size planes and sets size variable
extern void planesSetSize(struct VecPlane* p, const int size);
// access plane per index
extern double getPlaneCoeff(struct Plane* plane, int i);
// set plane coefficienr per index
extern void setPlaneCoeff(double coeff, struct Plane* plane, int i);

#endif // DATATYPES
