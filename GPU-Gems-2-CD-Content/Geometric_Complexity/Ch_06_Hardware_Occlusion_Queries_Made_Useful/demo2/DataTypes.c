//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include <memory.h>
#include <stdlib.h>
#include <malloc.h>
#include "DataTypes.h"
#include "MathStuff.h"

const struct Object OBJECT_NULL = { 0, 0 };
const struct VecPoint VECPOINT_NULL = { 0, 0 };
const struct VecPlane VECPLANE_NULL = { 0, 0 };

void copyVector3(Vector3 result, const Vector3 input) {
	result[0] = input[0]; 
	result[1] = input[1];
	result[2] = input[2];
}

void addVector3(Vector3 result, const Vector3 a, const Vector3 b) {
	int i;
	for(i = 0; i < 3; i++) {
		result[i] = a[i]+b[i];
	}
}

void diffVector3(Vector3 result, const Vector3 a, const Vector3 b) {
	int i;
	for(i = 0; i < 3; i++) {
		result[i] = a[i]-b[i];
	}
}

void copyVector3Values(Vector3 result, const double x, const double y, const double z) {
	result[0] = x; 
	result[1] = y;
	result[2] = z;
}

void copyMatrix(Matrix4x4 result, const Matrix4x4 input) {
	if(input != result) {
		memcpy(result,input,4*4*sizeof(double));
	}
}

void planesSetSize(struct VecPlane* p, const int size) {
	if(0 != p) {
		if(size == p->size) {
			return;
		}
		p->plane = (struct Plane*)realloc(p->plane,size*sizeof(struct Plane));
		p->size = size;
	}
}

void emptyVecPlane(struct VecPlane* p) {
	planesSetSize(p,0);
}

void vecPointSetSize(struct VecPoint* poly, const int size) {
	if(0 != poly) {
		if(size == poly->size) {
			return;
		}
		poly->points = (Vector3*)realloc(poly->points,size*sizeof(Vector3));
		poly->size = size;
	}
}

void emptyVecPoint(struct VecPoint* poly) {
	vecPointSetSize(poly,0);
}

void append2VecPoint(struct VecPoint* poly, const Vector3 p) {
	if(0 != poly) {
		int size = poly->size;
		vecPointSetSize(poly,size+1);
		copyVector3(poly->points[size],p);
	}
}

void copyVecPoint(struct VecPoint* poly, const struct VecPoint poly2) {
	if(0 != poly) {
		int i;
		vecPointSetSize(poly,poly2.size);
		for(i= 0; i < poly2.size; i++) {
			copyVector3(poly->points[i],poly2.points[i]);
		}
	}
}

void swapVecPoint(struct VecPoint* poly1, struct VecPoint* poly2) {
	if(0 != poly1 && 0 != poly2 && poly1 != poly2) {
		{
			Vector3* points = poly1->points;
			poly1->points = poly2->points;
			poly2->points = points;
		}
		{
			int size = poly1->size;
			poly1->size = poly2->size;
			poly2->size = size;
		}
	}
}

void objectSetSize(struct Object* obj, const int size) {
	if(0 != obj) {
		int i;
		if(size == obj->size) {
			return;
		}
		//dispose if shrinking
		for(i = size; i < obj->size; i++) {
			emptyVecPoint( &(obj->poly[i]) ); 
		}
		//allocate new place
		obj->poly = (struct VecPoint*)realloc(obj->poly,size*sizeof(struct VecPoint));
		//initialize new place
		for(i = obj->size; i < size; i++) {
			obj->poly[i] = VECPOINT_NULL;
		}
		obj->size = size;
	}
}

void emptyObject(struct Object* obj) {
	objectSetSize(obj,0);
}

void copyObject(struct Object* obj, const struct Object objIn) {
	if(0 != obj) {
		int i;
		objectSetSize(obj,objIn.size);
		for(i = 0; i < objIn.size; i++) {
			copyVecPoint( &(obj->poly[i]), objIn.poly[i]);
		}
	}
}

void append2Object(struct Object* obj, const struct VecPoint poly) {
	if(0 != obj) {
		int size = obj->size;
		objectSetSize(obj,size+1);
		copyVecPoint( &(obj->poly[size]) ,poly);
	}
}


void convObject2VecPoint(struct VecPoint* points,const struct Object obj) {
	if(0 != points) {
		int i, j;
		emptyVecPoint(points);
		for(i = 0; i < obj.size; i++) {
			struct VecPoint* p = &(obj.poly[i]);
			for(j = 0; j < p->size; j++) {
				append2VecPoint(points,p->points[j]);
			}
		}
	}
}

void calcAABoxPoints(Vector3x8 points, const struct AABox b) {
    //generate 8 corners of the box
	copyVector3Values(points[0],b.min[0],b.min[1],b.min[2]);//     7+------+6
	copyVector3Values(points[1],b.max[0],b.min[1],b.min[2]);//     /|     /|
	copyVector3Values(points[2],b.max[0],b.max[1],b.min[2]);//    / |    / |
	copyVector3Values(points[3],b.min[0],b.max[1],b.min[2]);//   / 4+---/--+5
	copyVector3Values(points[4],b.min[0],b.min[1],b.max[2]);// 3+------+2 /    y   z
	copyVector3Values(points[5],b.max[0],b.min[1],b.max[2]);//  | /    | /     |  /
	copyVector3Values(points[6],b.max[0],b.max[1],b.max[2]);//  |/     |/      |/
	copyVector3Values(points[7],b.min[0],b.max[1],b.max[2]);// 0+------+1      *---x
}

void calcAABoxPlanes(struct VecPlane* planes, const struct AABox b) {
	if(0 != planes) {
		struct Plane *p;
		planesSetSize(planes,6);
		
		//bottom plane
		p = &(planes->plane[0]);
		copyVector3Values(p->n,0,-1,0);
		p->d = absDouble(b.min[1]);

		//top plane
		p = &(planes->plane[1]);
		copyVector3Values(p->n,0,1,0);
		p->d = absDouble(b.max[1]);

		//left plane
		p = &(planes->plane[2]);
		copyVector3Values(p->n,-1,0,0);
		p->d = absDouble(b.min[0]);

		//right plane
		p = &(planes->plane[3]);
		copyVector3Values(p->n,1,0,0);
		p->d = absDouble(b.max[0]);

		//back plane
		p = &(planes->plane[4]);
		copyVector3Values(p->n,0,0,-1);
		p->d = absDouble(b.min[2]);

		//front plane
		p = &(planes->plane[5]);
		copyVector3Values(p->n,0,0,1);
		p->d = absDouble(b.max[2]);

	}
}

double getPlaneCoeff(struct Plane* plane, int i)
{
	if(i < 3) return plane->n[i];
	return plane->d;
}

void setPlaneCoeff(double coeff, struct Plane* plane, int i)
{
	if(i < 3) 
		plane->n[i] = coeff;
	else 
		plane->d = coeff;
}