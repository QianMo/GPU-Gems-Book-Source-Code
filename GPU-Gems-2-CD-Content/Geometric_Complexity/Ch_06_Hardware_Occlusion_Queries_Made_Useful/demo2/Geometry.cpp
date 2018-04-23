#include "Geometry.h"
#include "glInterface.h"
#include "stdio.h"
#include "teapot.h"
#include <math.h>

#define NO_LIST    -1
#define STRIP_END  -1

int Geometry::num_torus_indices;
int Geometry::num_torus_vertices;
int Geometry::num_torus_normals;

const int Geometry::torus_precision = 24;
const int Geometry::sphere_precision = 24;

float *Geometry::torus_vertices;
float *Geometry::torus_normals;
int *Geometry::torus_indices;

// choose size so the objects appear approximately the same size
const float Geometry::sphere_radius = 0.1f;
const float Geometry::torus_inner_radius = 0.05;
const float Geometry::torus_outer_radius = 0.07;

int Geometry::sDisplayList[NUM_OBJECTS];

bool Geometry::sIsInitialised = Init();

Geometry::Geometry(): 
mXRotation(0), mYRotation(0), mZRotation(0), mScale(1.0), mObjectType(TEAPOT)
{
	copyVector3Values(mTranslation, 0, 0, 0);
	
	SetAmbientColor(0.1745, 0.01175, 0.01175);
	SetDiffuseColor(0.61424, 0.04136, 0.04136);
	SetSpecularColor(0.727811, 0.626959, 0.626959);
	
	Recalculate();
	GenerateList();
}

Geometry::Geometry(Vector3 translation, float xRot, float yRot, float zRot, 
				   float scale, int objectType):
mXRotation(xRot), mYRotation(yRot), mZRotation(zRot), mScale(scale), mObjectType(objectType)
{
	copyVector3(mTranslation, translation);

	SetAmbientColor(0.1745, 0.01175, 0.01175);
	SetDiffuseColor(0.61424, 0.04136, 0.04136);
	SetSpecularColor(0.727811, 0.626959, 0.626959);

	Recalculate();
	GenerateList();
}

void Geometry::ResetLists()
{
	for(int i = 0; i < NUM_OBJECTS; i++)
	{
        if(sDisplayList[i] == NO_LIST)
			glDeleteLists(sDisplayList[i], 1);

		sDisplayList[i] = NO_LIST;
	}
}


void Geometry::CleanUp()
{
	ResetLists();

	delete [] torus_vertices;
	delete [] torus_normals;
	delete [] torus_indices;
}


bool Geometry::Init()
{
	for(int i=0; i < NUM_OBJECTS; i++)
		sDisplayList[i] = NO_LIST;

	CreateTorus(torus_inner_radius, torus_outer_radius, torus_precision);

	return true;
}

void Geometry::GenerateList()
{
	if(sDisplayList[mObjectType] == NO_LIST)
	{
		sDisplayList[mObjectType] = glGenLists(1);
		glNewList(sDisplayList[mObjectType], GL_COMPILE);

		switch(mObjectType)
		{
			case TEAPOT:
				RenderTeapot();
				break;
			case TORUS:
				RenderTorus();
				break;
			case SPHERE:	
				RenderSphere();
				break;
			
			default:
				break;
		}
		glEndList();
	}
}

void Geometry::Render()
{
	glMaterialfv(GL_FRONT, GL_AMBIENT, mAmbientColor);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mDiffuseColor);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mSpecularColor);
	
	glPushMatrix();
	
	Transform();
	
	glCallList(sDisplayList[mObjectType]);

	glPopMatrix();
}


//! sets rotations around the three axis: executed in the order specified here
void Geometry::SetRotations(float xRot, float yRot, float zRot)
{
	mXRotation = xRot;
	mYRotation = yRot;
	mZRotation = zRot;

	Recalculate();
}


void Geometry::SetTranslation(Vector3 translation)
{
	copyVector3(mTranslation, translation);

	Recalculate();
}


void Geometry::SetScale(float scale)
{
	mScale = scale;

    Recalculate();
}


void Geometry::SetAmbientColor(float ambientR, float ambientG, float ambientB)
{
	mAmbientColor[0] = ambientR;
	mAmbientColor[1] = ambientG;
	mAmbientColor[2] = ambientB;
}


void Geometry::SetDiffuseColor(float diffuseR, float diffuseG, float diffuseB)
{
	mDiffuseColor[0] = diffuseR;
	mDiffuseColor[1] = diffuseG;
	mDiffuseColor[2] = diffuseB;
}


void Geometry::SetSpecularColor(float specularR, float specularG, float specularB)
{
	mSpecularColor[0] = specularR;
	mSpecularColor[1] = specularG;
	mSpecularColor[2] = specularB;
}


float Geometry::GetScale()
{
	return mScale;
}


void Geometry::GetTranslation(Vector3 translation)
{
	copyVector3(translation, mTranslation);
}


void Geometry::GetRotations(float &xRot, float &yRot, float &zRot)
{
	xRot = mXRotation;
	yRot = mYRotation;
	zRot = mZRotation;
}

	
const AABox &Geometry::GetBoundingVolume()
{
	return mBoundingBox;
}


void Geometry::SetLastRendered(int lastRendered)
{
	mLastRendered = lastRendered;
}


int Geometry::GetLastRendered()
{
	return mLastRendered;
}


void Geometry::SetObjectType(int type)
{
	mObjectType = type;
	GenerateList();
}


void Geometry::Recalculate()
{
	CalcTransform();
	CalcBoundingVolume();
}


void Geometry::RenderTeapot()
{
	int i = 0;

	while(i < num_teapot_indices)
	{
		glBegin(GL_TRIANGLE_STRIP);
			while(teapot_indices[i] != STRIP_END)
			{	
				int index = teapot_indices[i] * 3;
				
				glNormal3fv(teapot_normals + index);
				glVertex3fv(teapot_vertices + index);
				
				i++;		
			}
		glEnd();
		
		i++; // skip strip end flag
	}
}

void Geometry::CalcBoundingVolume()
{
    switch(mObjectType)
	{
		case TORUS:
			CalcBoundingVolume(torus_vertices, num_torus_vertices);
			break;
		case SPHERE:
			CalcSphereBoundingVolume();
			break;
		case TEAPOT:
			CalcBoundingVolume(teapot_vertices, num_teapot_vertices);
			break;
		default:
			break;
	}
}


void Geometry::CalcBoundingVolume(float *vertices, const int num_vertices)
{
	Vector3 *transformedPoints = new Vector3[num_vertices];
	
	for(int i = 0; i < num_vertices; i++)
	{
		Vector3 currentVtx = {vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]};

		copyVector3(transformedPoints[i], currentVtx);
	
		mulHomogenPoint(transformedPoints[i], mTransform, currentVtx);
	}

	calcCubicHull(mBoundingBox.min, mBoundingBox.max, transformedPoints, num_vertices);

	delete [] transformedPoints;
}


void Geometry::CalcSphereBoundingVolume()
{
	float len = mScale * sphere_radius;
	Vector3 size = {len, len, len};
					
	diffVector3(mBoundingBox.min, mTranslation, size);
	addVector3(mBoundingBox.max, mTranslation, size);
}


void Geometry::RenderTorus()
{
	glNormalPointer(GL_FLOAT, sizeof(float), torus_normals);
	glEnableClientState(GL_NORMAL_ARRAY);

	glVertexPointer(3, GL_FLOAT, sizeof(float), torus_vertices);
	glEnableClientState(GL_VERTEX_ARRAY);

    glDrawElements(GL_TRIANGLES, num_torus_indices, GL_UNSIGNED_INT, torus_indices);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}


void Geometry::CreateTorus(float innerRadius, float outerRadius, int precision)
{
	num_torus_vertices = num_torus_normals = (precision + 1) * (precision + 1);
	num_torus_indices = 2 * precision * precision * 3;

	torus_vertices = new float[num_torus_vertices * 3];
	torus_normals = new float[num_torus_normals * 3];
	torus_indices = new int[num_torus_indices];

	
	for(int i=0; i<precision+1; i++)
	{
		int index = i * 3;

		Vector3 currentVtx = {innerRadius, 0.0f, 0.0f};
		rotateVectorZ(currentVtx, ((float)i * 2.0 * PI) / (float)precision);

		currentVtx[0] += outerRadius;

		// the normal is the cross product of the two tangent 
		// vectors in this vertex
		Vector3 tangS = {0.0f, 0.0f, -1.0f};
		Vector3 tangT = {0, -1, 0};

		rotateVectorZ(tangT, ((float)i * 2.0 * PI) / (float)precision);
      	Vector3 currentNorm;
		cross(currentNorm, tangT, tangS);

		torus_normals[index + 0] = currentNorm[0];
		torus_normals[index + 1] = currentNorm[1];
		torus_normals[index + 2] = currentNorm[2];
		
		torus_vertices[index + 0] = currentVtx[0];
		torus_vertices[index + 1] = currentVtx[1];
		torus_vertices[index + 2] = currentVtx[2];
	}

	for(int i_rng=1; i_rng<precision+1; ++i_rng)
	{
		for(int i=0; i<precision+1; ++i)
		{
			int index = 3 * (i_rng * (precision + 1) + i);

			Vector3 currentVtx = {torus_vertices[i*3], torus_vertices[i*3+1], torus_vertices[i*3+2]};
			
			rotateVectorY(currentVtx, ((float)i_rng * 2.0 * PI) / (float)precision);
						
			Vector3 currentNorm = {torus_normals[i*3], torus_normals[i*3+1], torus_normals[i*3+2]};
			
			rotateVectorY(currentNorm, ((float)i_rng * 2.0 * PI) / (float)precision);

			torus_normals[index + 0] = currentNorm[0];
			torus_normals[index + 1] = currentNorm[1];
			torus_normals[index + 2] = currentNorm[2];
		
			torus_vertices[index + 0] = currentVtx[0];
			torus_vertices[index + 1] = currentVtx[1];
			torus_vertices[index + 2] = currentVtx[2];
		}
	}


	for(i_rng=0; i_rng<precision; ++i_rng)
	{
		for(i=0; i<precision; ++i)
		{
			int index = ((i_rng * precision + i) * 2) * 3;

			torus_indices[index+0] = (i_rng     * (precision+1) + i)*3;
			torus_indices[index+1] = ((i_rng+1) * (precision+1) + i)*3;
			torus_indices[index+2] = (i_rng     * (precision+1) + i + 1)*3;

			index = ((i_rng * precision + i) * 2 + 1) * 3;

			torus_indices[index+0] = (i_rng     * (precision+1) + i + 1)*3;
			torus_indices[index+1] = ((i_rng+1) * (precision+1) + i)*3;
			torus_indices[index+2] = ((i_rng+1) * (precision+1) + i + 1)*3;
		}
	}
}

// counts the triangles of the teapot by
// traversing through all the triangle strips.
int Geometry::CountTeapotTriangles()
{
	int result = 0;
	int i = 0;

	// n - 2 triangles are drawn for a strip
	while(i < num_teapot_indices)
	{
		while(teapot_indices[i] != STRIP_END)
		{	
			result ++;;
			i++;		
		}
		result -= 2;
		i++; // skip STRIP_END
	}

	return result;
}


int Geometry::CountTorusTriangles()
{
	return 2 * torus_precision * torus_precision;
}


int Geometry::CountSphereTriangles()
{
	return sphere_precision * sphere_precision;
}


int Geometry::CountTriangles(int objectType)
{
	int result = 0;

	switch(objectType)
	{
		case TEAPOT:
			result = CountTeapotTriangles();
			break;
		case TORUS:
			result = CountTorusTriangles();
			break;
		case SPHERE:	
			result = CountSphereTriangles();
			break;
		default:
			break;
	}

	return result;
}


void Geometry::Transform()
{
	glMultMatrixd(mTransform);
}


void Geometry::CalcTransform()
{
	Matrix4x4 scale;
	Matrix4x4 xrot;
	Matrix4x4 yrot;
	Matrix4x4 zrot;
	Matrix4x4 transl;

	makeScaleMtx(scale, mScale, mScale, mScale);
	makeTranslationMtx(transl, mTranslation);
	makeRotationXMtx(xrot, mXRotation * PI_180);
	makeRotationYMtx(yrot, mYRotation * PI_180);
	makeRotationZMtx(zrot, mZRotation * PI_180);

	copyMatrix(mTransform, IDENTITY);
	mult(mTransform, mTransform, transl);
	mult(mTransform, mTransform, xrot);
	mult(mTransform, mTransform, yrot);
	mult(mTransform, mTransform, zrot);
	mult(mTransform, mTransform, scale);
}

/**
   renders a sphere with sphere_precision subdivisions.
   note: works only for even sphere_precision
*/
void Geometry::RenderSphere()
{
	Vector3 vertex;

	// this algorithm renders the triangles clockwise
	glFrontFace(GL_CW);

	for (int j = 0; j < sphere_precision/2; ++j) 
	{
		double alpha = j * PI * 2.0 / (double)sphere_precision - PI_2;
		double beta = (j + 1) * PI * 2.0 / (double)sphere_precision - PI_2;

		glBegin(GL_TRIANGLE_STRIP);

		for (int i = 0; i <= sphere_precision; ++i) 
		{
			double gamma = i * PI * 2.0 / (double)sphere_precision;
	
			vertex[0] = sphere_radius * cos(beta) * cos(gamma);
			vertex[1] = sphere_radius * sin(beta);
			vertex[2] = sphere_radius * cos(beta) * sin(gamma);

			glNormal3dv(vertex);
			glVertex3dv(vertex);

			vertex[0] = sphere_radius * cos(alpha) * cos(gamma);
			vertex[1] = sphere_radius * sin(alpha);
			vertex[2] = sphere_radius * cos(alpha) * sin(gamma);

		    glNormal3dv(vertex);
			glVertex3dv(vertex);

		}
		glEnd();
	}
	glFrontFace(GL_CCW);
}