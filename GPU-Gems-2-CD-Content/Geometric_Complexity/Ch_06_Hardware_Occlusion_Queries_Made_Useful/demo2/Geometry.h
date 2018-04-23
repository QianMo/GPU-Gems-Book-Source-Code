#ifndef NO_PRAGMA_ONCE
#pragma once
#endif

#ifndef GEOMETRY_H
#define GEOMETRY_H

extern "C"
{
	#include "MathStuff.h"
}

/** 
	Represents a drawable geometry. It can draw simple objects (teapot,
	torus, ...). It also contains the object transformation and 
	the material of the objects. An AAB is generated as bounding volume of the object.
*/
class Geometry
{
public:
	Geometry();
	//! generate geometry with specified transformations and of specified object type (e.g. teapot)
	Geometry(Vector3 translation, float xRot, float yRot, float zRot, 
			 float scale, int objectType);
	//! renders this geometry
	void Render();
	
	//! sets rotations in degree : executed in the order x, y, and z rotation
	void SetRotations(float xRot, float yRot, float zRot);
	//! sets the rotation matrix
	void SetTranslation(Vector3 translation);
	//! a uniform scale
	void SetScale(float scale);
	//! returns current scale
	float GetScale();
	//! returns translation
	void GetTranslation(Vector3 translation);
	//! returns rotation
	void GetRotations(float &xRot, float &yRot, float &zRot);

	// --- material settings
	void SetAmbientColor(float ambientR,   float ambientG,  float ambientB);
	void SetDiffuseColor(float diffuseR,   float diffuseG,  float diffuseB);
	void SetSpecularColor(float specularR, float specularG, float specularB);

	//! returns boudning box of this geometry
	const AABox& GetBoundingVolume();
		
	//! set frame when geometry was last rendered. important for rendering each geometry only once.
	void SetLastRendered(int lastRendered);
	//! returns frame when geometry was last rendered
	int GetLastRendered();

	//! sets the object type to one of teapot, torus, sphere
	void SetObjectType(int type);

	enum{TEAPOT, TORUS, SPHERE, NUM_OBJECTS};
	
	static int sDisplayList[NUM_OBJECTS];
	
	//! cleans static members
	static void CleanUp();
	//! returns the triangle count of the specified object type
	static int CountTriangles(int objectType);
	//! initialises display lists
	static void ResetLists(); 

protected:
	//! generates the display list this a object
	void GenerateList();

	//! calculates accumulated transformation matrix
	void CalcTransform();
	//! applies tranformations
	void Transform();

	// some statistics
	//! returns the triangle number of the torus (called by CountTriangles)
	static int CountTorusTriangles();
	//! returns the triangle number of the sphere (called by CountTriangles)
	static int CountSphereTriangles();
	//! returns the triangle number of the teapot (called by CountTriangles)
	static int CountTeapotTriangles();

	//! creates a torus with specified radii, with number of rings and ring	subdivision = precision.
	static void CreateTorus(float innerRadius, float outerRadius, int precision);

	//! calculates the bounding volume of this geometry
	void CalcBoundingVolume();
	// the size of the bounging box is calculated using the radius of the sphere
	void CalcSphereBoundingVolume();
	// the standard bounding volume calculation having an array of vertices
	void CalcBoundingVolume(float *vertices, const int num_vertices);

	//! Updates transformation and bounding volume
	void Recalculate();

	// initialises static members (display lists, torus)

	static bool Init();

	// drawing routines
	static void RenderTeapot();
	static void RenderSphere();
	static void RenderTorus();

	// transformations
	float mXRotation;
	float mYRotation;
	float mZRotation;
	float mScale;
	Vector3 mTranslation;
	
	//! accumulated transform matrix 
	Matrix4x4 mTransform;

	// material
	float mAmbientColor[3];
	float mDiffuseColor[3];
	float mSpecularColor[3];
	//! the bounding box of the geometry
	AABox mBoundingBox;
	
	//! type of the rendered object (currently one of teapot, sphere, torus)
	int mObjectType;
	
	//! last rendered frame
	int mLastRendered;

	// static members
	static bool sIsInitialised;
	static float const sphere_radius;

	static int num_torus_indices;
	static int num_torus_vertices;
	static int num_torus_normals;

	static float *torus_vertices;
	static float *torus_normals;
	static int *torus_indices;

	static const int torus_precision;
	static const int sphere_precision;
	static const float torus_inner_radius;
	static const float torus_outer_radius;
};


#endif // GEOMETRY_H