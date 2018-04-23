//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef DataTypesH
#define DataTypesH

#include <vector>
#include <Types/DynamicArray.h>
#include <Types/StaticArray.h>
#include <Mathematic/Vector3.h>
#include <Mathematic/Matrix4.h>
#include <Mathematic/Geometry/Line.h>
#include <Mathematic/Geometry/Plane.h>
#include <Mathematic/Geometry/AABox.h>
#include <Mathematic/Geometry/Frustum.h>

typedef Math::Vector3d V3;

typedef Math::Geometry::Line<double> Line;

typedef StaticArray<V3,8> Vector3x8;

//a dynamic array of planes
typedef Math::Geometry::Plane<double> Plane;

//a dynamic array 3d points
typedef DynamicArray<V3> VecPoint;

//a dynamic array of point list each point list is a polygon
typedef std::vector<VecPoint> Object;

//Axis-Aligned Bounding Box defined through the two extreme points
typedef Math::Geometry::AABox<double> AABox;

typedef Math::Geometry::Frustum<float> Frustum;

//4x4 matrix
typedef Math::Matrix4d M4;

//makes 1 VecPoint out of all the VecPoints of an object
extern void convObject2VecPoint(VecPoint& points, const Object&);

#endif
