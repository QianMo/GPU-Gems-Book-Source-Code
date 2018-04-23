//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include <memory.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include "MathStuff.h"

#ifndef M_PI
#define M_PI (double)3.14159265358979323846
#endif

const double DOUBLE_MAX = DBL_MAX;
const double PI = M_PI;
const double PI_2 = M_PI/2;
const double PI_180 = M_PI/180;
const Vector3 ZERO = {0.0, 0.0, 0.0};
const Vector3 UNIT_X = {1.0, 0.0, 0.0};
const Vector3 UNIT_Y = {0.0, 1.0, 0.0};
const Vector3 UNIT_Z = {0.0, 0.0, 1.0};

const Matrix4x4 IDENTITY = {1.0, 0.0, 0.0, 0.0, 
							0.0, 1.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 1.0};

double maximum(const double a, const double b) {
	return ( (a > b)? (a):(b) );

}

void clamp(double* value, const double min, const double max) {
	if( (*value) > max) {
		(*value) = max;
	}
	else {
		if( (*value) < min) {
			(*value) = min;
		}
	}
}

double absDouble(const double a) {
	return ( (a < 0.0)? (-a):(a) );
}

double signDouble(const double a) {
	return ( (a < 0.0)? (-1.0f): ( (a == 0.0)? (0.0f):(1.0f) ) );
}

double coTan(const double vIn) {
	return (double)-tan(vIn+PI_2);
}

double relativeEpsilon(const double a, const double epsilon) {
	double relEpsilon = maximum(absDouble(a*epsilon),epsilon);
	return relEpsilon;
}

int alike(const double a, const double b, const double epsilon) {
	if(a == b) {
		return 1;
	}
	{
		double relEps = relativeEpsilon(a,epsilon);
		return (a-relEps <= b) && (b <= a+relEps);
	}
}



int alikeVector3(const Vector3 a, const Vector3 b, const double epsilon) {
	return 
		alike(a[0],b[0],epsilon) &&
		alike(a[1],b[1],epsilon) &&
		alike(a[2],b[2],epsilon);
}

void linCombVector3(Vector3 result, const Vector3 pos, const Vector3 dir, const double t) {
	int i;
	for(i = 0; i < 3; i++) {
		result[i] = pos[i]+t*dir[i];
	}
}

double squaredLength(const Vector3 vec) {
	int i;
	double tmp = 0.0;
	for(i = 0; i < 3; i++) {
		tmp += vec[i]*vec[i];
	}
	return tmp;
}

void normalize(Vector3 vec) {
	int i;
	const double len = (double)(1.0/sqrt(squaredLength(vec)));
	for(i = 0; i < 3; i++) {
		vec[i] *= len;
	}
}

void copyNormalize(Vector3 vec, const Vector3 input) {
	copyVector3(vec,input),
	normalize(vec);
}

/*	| a1 a2 |
	| b1 b2 | calculate the determinent of a 2x2 matrix*/
double det2x2(const double a1, const double a2,
			const double b1, const double b2) {
	return a1*b2 - b1*a2;
}

/*	| a1 a2 a3 |
	| b1 b2 b3 |
	| c1 c2 c3 | calculate the determinent of a 3x3 matrix*/
double det3x3(const double a1, const double a2, const double a3,
			  const double b1, const double b2, const double b3,
			  const double c1, const double c2, const double c3) {
	return a1*det2x2(b2,b3,c2,c3) - b1*det2x2(a2,a3,c2,c3) +
			c1*det2x2(a2,a3,b2,b3);
}
void cross(Vector3 result, const Vector3 a, const Vector3 b) {
	result[0] =  det2x2(a[1],b[1],a[2],b[2]);
	result[1] = -det2x2(a[0],b[0],a[2],b[2]);
	result[2] =  det2x2(a[0],b[0],a[1],b[1]);
}

double dot(const Vector3 a, const Vector3 b) {
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

void look(Matrix4x4 output, const Vector3 pos, const Vector3 dir, const Vector3 up) {
	Vector3 dirN;
	Vector3 upN;
	Vector3 lftN;

	cross(lftN,dir,up);
	normalize(lftN);

	cross(upN,lftN,dir);
	normalize(upN);
	copyNormalize(dirN,dir);

	output[ 0] = lftN[0];
	output[ 1] = upN[0];
	output[ 2] = -dirN[0];
	output[ 3] = 0.0;

	output[ 4] = lftN[1];
	output[ 5] = upN[1];
	output[ 6] = -dirN[1];
	output[ 7] = 0.0;

	output[ 8] = lftN[2];
	output[ 9] = upN[2];
	output[10] = -dirN[2];
	output[11] = 0.0;

	output[12] = -dot(lftN,pos);
	output[13] = -dot(upN,pos);
	output[14] = dot(dirN,pos);
	output[15] = 1.0;
}

void makeScaleMtx(Matrix4x4 output, const double x, const double y, const double z) {
	output[ 0] = x;
	output[ 1] = 0.0;
	output[ 2] = 0.0;
	output[ 3] = 0.0;

	output[ 4] = 0.0;
	output[ 5] = y;
	output[ 6] = 0.0;
	output[ 7] = 0.0;

	output[ 8] = 0.0;
	output[ 9] = 0.0;
	output[10] = z;
	output[11] = 0.0;

	output[12] = 0.0;
	output[13] = 0.0;
	output[14] = 0.0;
	output[15] = 1.0;
}

void makeTranslationMtx(Matrix4x4 output, Vector3 translation) {
	output[ 0] = 1.0;
	output[ 1] = 0.0;
	output[ 2] = 0.0;
	output[ 3] = 0;//translation[0];

	output[ 4] = 0.0;
	output[ 5] = 1.0;
	output[ 6] = 0.0;
	output[ 7] = 0;//translation[1];

	output[ 8] = 0.0;
	output[ 9] = 0.0;
	output[10] = 1.0;
	output[11] = 0;//translation[2];

	output[12] = translation[0];//0.0;
	output[13] = translation[1];//0.0;
	output[14] = translation[2];//0.0;
	output[15] = 1.0;
}

void makeRotationZMtx(Matrix4x4 output, const double rad)
{
	output[ 0] = cos(rad);
	output[ 1] = sin(rad);
	output[ 2] = 0.0;
	output[ 3] = 0.0;

	output[ 4] = -sin(rad);
	output[ 5] = cos(rad);
	output[ 6] = 0.0;
	output[ 7] = 0.0;

	output[ 8] = 0.0;
	output[ 9] = 0.0;
	output[10] = 1.0;
	output[11] = 0.0;

	output[12] = 0.0;
	output[13] = 0.0;
	output[14] = 0.0;
	output[15] = 1.0;
}

void makeRotationXMtx(Matrix4x4 output, const double rad)
{
	output[ 0] = 1.0;
	output[ 1] = 0.0;
	output[ 2] = 0.0;
	output[ 3] = 0.0;

	output[ 4] = 0.0;
	output[ 5] = cos(rad);
	output[ 6] = sin(rad);
	output[ 7] = 0.0;

	output[ 8] = 0.0;
	output[ 9] = -sin(rad);
	output[10] = cos(rad);
	output[11] = 0.0;

	output[12] = 0.0;
	output[13] = 0.0;
	output[14] = 0.0;
	output[15] = 1.0;
}

void makeRotationYMtx(Matrix4x4 output, const double rad)
{
	output[ 0] = cos(rad);
	output[ 1] = 0.0;
	output[ 2] = -sin(rad);
	output[ 3] = 0.0;

	output[ 4] = 0.0;
	output[ 5] = 1.0;
	output[ 6] = 0.0;
	output[ 7] = 0.0;

	output[ 8] = sin(rad);
	output[ 9] = 0.0;
	output[10] = cos(rad);
	output[11] = 0.0;

	output[12] = 0.0;
	output[13] = 0.0;
	output[14] = 0.0;
	output[15] = 1.0;
}

void scaleTranslateToFit(Matrix4x4 output, const Vector3 vMin, const Vector3 vMax) {
	output[ 0] = 2/(vMax[0]-vMin[0]);
	output[ 4] = 0;
	output[ 8] = 0;
	output[12] = -(vMax[0]+vMin[0])/(vMax[0]-vMin[0]);

	output[ 1] = 0;
	output[ 5] = 2/(vMax[1]-vMin[1]);
	output[ 9] = 0;
	output[13] = -(vMax[1]+vMin[1])/(vMax[1]-vMin[1]);

	output[ 2] = 0;
	output[ 6] = 0;
	output[10] = 2/(vMax[2]-vMin[2]);
	output[14] = -(vMax[2]+vMin[2])/(vMax[2]-vMin[2]);

	output[ 3] = 0;
	output[ 7] = 0;
	output[11] = 0;
	output[15] = 1;
}

void perspectiveRad(Matrix4x4 output, const double vFovy, const double vAspect,
						  const double vNearDis, const double vFarDis) {
	const double f = coTan(vFovy/2);
	const double dif = 1/(vNearDis-vFarDis);

	output[ 0] = f/vAspect;
	output[ 4] = 0;
	output[ 8] = 0;
	output[12] = 0;

	output[ 1] = 0;
	output[ 5] = f;
	output[ 9] = 0;
	output[13] = 0;

	output[ 2] = 0;
	output[ 6] = 0;
	output[10] = (vFarDis+vNearDis)*dif;
	output[14] = 2*vFarDis*vNearDis*dif;

	output[ 3] = 0;
	output[ 7] = 0;
	output[11] = -1;
	output[15] = 0;
}

void perspectiveDeg(Matrix4x4 output, const double vFovy, const double vAspect,
					const double vNearDis, const double vFarDis) {
	perspectiveRad(output,vFovy*PI_180,vAspect,vNearDis,vFarDis);
}

void invert(Matrix4x4 output, const Matrix4x4 i) {
	double a11 =  det3x3(i[5],i[6],i[7],i[9],i[10],i[11],i[13],i[14],i[15]);
	double a21 = -det3x3(i[1],i[2],i[3],i[9],i[10],i[11],i[13],i[14],i[15]);
	double a31 =  det3x3(i[1],i[2],i[3],i[5],i[6],i[7],i[13],i[14],i[15]);
	double a41 = -det3x3(i[1],i[2],i[3],i[5],i[6],i[7],i[9],i[10],i[11]);

	double a12 = -det3x3(i[4],i[6],i[7],i[8],i[10],i[11],i[12],i[14],i[15]);
	double a22 =  det3x3(i[0],i[2],i[3],i[8],i[10],i[11],i[12],i[14],i[15]);
	double a32 = -det3x3(i[0],i[2],i[3],i[4],i[6],i[7],i[12],i[14],i[15]);
	double a42 =  det3x3(i[0],i[2],i[3],i[4],i[6],i[7],i[8],i[10],i[11]);

	double a13 =  det3x3(i[4],i[5],i[7],i[8],i[9],i[11],i[12],i[13],i[15]);
	double a23 = -det3x3(i[0],i[1],i[3],i[8],i[9],i[11],i[12],i[13],i[15]);
	double a33 =  det3x3(i[0],i[1],i[3],i[4],i[5],i[7],i[12],i[13],i[15]);
	double a43 = -det3x3(i[0],i[1],i[3],i[4],i[5],i[7],i[8],i[9],i[11]);

	double a14 = -det3x3(i[4],i[5],i[6],i[8],i[9],i[10],i[12],i[13],i[14]);
	double a24 =  det3x3(i[0],i[1],i[2],i[8],i[9],i[10],i[12],i[13],i[14]);
	double a34 = -det3x3(i[0],i[1],i[2],i[4],i[5],i[6],i[12],i[13],i[14]);
	double a44 =  det3x3(i[0],i[1],i[2],i[4],i[5],i[6],i[8],i[9],i[10]);

	double det = (i[0]*a11) + (i[4]*a21) + (i[8]*a31) + (i[12]*a41);
	double oodet = 1/det;

	output[ 0] = a11*oodet;
	output[ 1] = a21*oodet;
	output[ 2] = a31*oodet;
	output[ 3] = a41*oodet;

	output[ 4] = a12*oodet;
	output[ 5] = a22*oodet;
	output[ 6] = a32*oodet;
	output[ 7] = a42*oodet;

	output[ 8] = a13*oodet;
	output[ 9] = a23*oodet;
	output[10] = a33*oodet;
	output[11] = a43*oodet;

	output[12] = a14*oodet;
	output[13] = a24*oodet;
	output[14] = a34*oodet;
	output[15] = a44*oodet;
}

void multUnSave(Matrix4x4 output, const Matrix4x4 a, const Matrix4x4 b) {
	const int SIZE = 4;
	int iCol;
	for(iCol = 0; iCol < SIZE; iCol++) {
		const int cID = iCol*SIZE;
		int iRow;
		for(iRow = 0; iRow < SIZE; iRow++) {
			const int id = iRow+cID;
			int k;
			output[id] = a[iRow]*b[cID];
			for(k = 1; k < SIZE; k++) {
				output[id] += a[iRow+k*SIZE]*b[k+cID];
			}
		}
	}
}

void mult(Matrix4x4 output, const Matrix4x4 a, const Matrix4x4 b) {
	if(a == output) {
		Matrix4x4 tmpA;
		copyMatrix(tmpA,a);
		if(b == output) {
			multUnSave(output,tmpA,tmpA);
		}
		else {
			multUnSave(output,tmpA,b);
		}
	}
	else {
		if(b == output) {
			Matrix4x4 tmpB;
			copyMatrix(tmpB,b);
			multUnSave(output,a,tmpB);
		}
		else {
			multUnSave(output,a,b);
		}
	}
}

void mulHomogenPoint(Vector3 output, const Matrix4x4 m, const Vector3 v) {
	//if v == output -> overwriting problems -> so store in temp
	double x = m[0]*v[0] + m[4]*v[1] + m[ 8]*v[2] + m[12];
	double y = m[1]*v[0] + m[5]*v[1] + m[ 9]*v[2] + m[13];
	double z = m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14];
	double w = m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15];

	output[0] = x/w;
	output[1] = y/w;
	output[2] = z/w;
}

void appendToCubicHull(Vector3 min, Vector3 max, const Vector3 v) {
	int j;
	for(j = 0; j < 3; j++) {
		if(v[j] < min[j]) {
			min[j] = v[j];
		}
		else if(v[j] > max[j]) {
			max[j] = v[j];
		}
	}
}

void calcCubicHull(Vector3 min, Vector3 max, const Vector3* ps, const int size) {
	if(size > 0) {
		int i;
		copyVector3(min,ps[0]);
		copyVector3(max,ps[0]);
		for(i = 1; i < size; i++) {
			appendToCubicHull(min,max,ps[i]);
		}
	}
}

void calcViewFrustumWorldCoord(Vector3x8 points, const Matrix4x4 invEyeProjView) {
	const struct AABox box = {
		{-1, -1, -1},
		{1, 1, 1}
	};
	int i;
	calcAABoxPoints(points,box); //calc unit cube corner points
	for(i = 0; i < 8; i++) {
		mulHomogenPoint(points[i],invEyeProjView,points[i]); //camera to world frame
	}
	//viewFrustumWorldCoord[0] == near-bottom-left
	//viewFrustumWorldCoord[1] == near-bottom-right
	//viewFrustumWorldCoord[2] == near-top-right
	//viewFrustumWorldCoord[3] == near-top-left
	//viewFrustumWorldCoord[4] == far-bottom-left
	//viewFrustumWorldCoord[5] == far-bottom-right
	//viewFrustumWorldCoord[6] == far-top-right
	//viewFrustumWorldCoord[7] == far-top-left
}

void calcViewFrustObject(struct Object* obj, const Vector3x8 p) {
	int i;
	Vector3* ps;
	objectSetSize(obj,6);
	for(i = 0; i < 6; i++) {
		vecPointSetSize(&obj->poly[i],4);
	}
	//near poly ccw
	ps = obj->poly[0].points;
	for(i = 0; i < 4; i++) {
		copyVector3(ps[i],p[i]);
	}
	//far poly ccw
	ps = obj->poly[1].points;
	for(i = 4; i < 8; i++) {
		copyVector3(ps[i-4],p[11-i]);
	}
	//left poly ccw
	ps = obj->poly[2].points;
	copyVector3(ps[0],p[0]);
	copyVector3(ps[1],p[3]);
	copyVector3(ps[2],p[7]);
	copyVector3(ps[3],p[4]);
	//right poly ccw
	ps = obj->poly[3].points;
	copyVector3(ps[0],p[1]);
	copyVector3(ps[1],p[5]);
	copyVector3(ps[2],p[6]);
	copyVector3(ps[3],p[2]);
	//bottom poly ccw
	ps = obj->poly[4].points;
	copyVector3(ps[0],p[4]);
	copyVector3(ps[1],p[5]);
	copyVector3(ps[2],p[1]);
	copyVector3(ps[3],p[0]);
	//top poly ccw
	ps = obj->poly[5].points;
	copyVector3(ps[0],p[6]);
	copyVector3(ps[1],p[7]);
	copyVector3(ps[2],p[3]);
	copyVector3(ps[3],p[2]);
}

void transformVecPoint(struct VecPoint* poly, const Matrix4x4 xForm) {
	if(0 != poly) {
		int i;
		for(i = 0; i < poly->size; i++) {
			mulHomogenPoint(poly->points[i],xForm,poly->points[i]);
		}
	}
}

void transformObject(struct Object* obj, const Matrix4x4 xForm) {
	if(0 != obj) {
		int i;
		for(i = 0; i < obj->size; i++) {
			transformVecPoint( &(obj->poly[i]) ,xForm);
		}
	}
}

void calcObjectCubicHull(Vector3 min, Vector3 max, const struct Object obj) {
	if(0 < obj.size) {
		int i;
		calcCubicHull(min,max,obj.poly[0].points,obj.poly[0].size);
		for(i = 1; i < obj.size; i++) {
			struct VecPoint* p = &(obj.poly[i]);
			int j;
			for(j = 0; j < p->size; j++) {
				appendToCubicHull(min,max,p->points[j]);
			}
		}
	}
}

int findSamePointInVecPoint(const struct VecPoint poly, const Vector3 p) {
	int i;
	for(i = 0; i < poly.size; i++) {
		if(alikeVector3(poly.points[i],p,(double)0.001)) {
			return i;
		}
	}
	return -1;
}

int findSamePointInObjectAndSwapWithLast(struct Object *inter, const Vector3 p) {
	int i;
	if(0 == inter) {
		return -1;
	}
	if(1 > inter->size) {
		return -1;
	}
	for(i = inter->size; i > 0; i--) {
		struct VecPoint* poly = &(inter->poly[i-1]);
		if(2 == poly->size) {
			const int nr = findSamePointInVecPoint(*poly,p);
			if(0 <= nr) {
				//swap with last
				swapVecPoint(poly, &(inter->poly[inter->size-1]) );
				return nr;
			}
		}
	}
	return -1;
}

void appendIntersectionVecPoint(struct Object* obj, struct Object* inter) {
	struct VecPoint *polyOut;
	struct VecPoint *polyIn;
	int size = obj->size;
	int i;
	//you need at least 3 sides for a polygon
	if(3 > inter->size) {
		return;
	}
	//compact inter: remove poly.size != 2 from end on forward
	for(i = inter->size; 0 < i; i--) {
		if(2 == inter->poly[i-1].size) {
			break;
		}
	}
	objectSetSize(inter,i);
	//you need at least 3 sides for a polygon
	if(3 > inter->size) {
		return;
	}
	//make place for one additional polygon in obj
	objectSetSize(obj,size+1);
	polyOut = &(obj->poly[size]);
	//we have line segments in each poly of inter 
	//take last linesegment as first and second point
	polyIn = &(inter->poly[inter->size-1]);
	append2VecPoint(polyOut,polyIn->points[0]);
	append2VecPoint(polyOut,polyIn->points[1]);
	//remove last poly from inter, because it is already saved
	objectSetSize(inter,inter->size-1);

	//iterate over inter until their is no line segment left
	while(0 < inter->size) {
		//pointer on last point to compare
		Vector3 *lastPt = &(polyOut->points[polyOut->size-1]);
		//find same point in inter to continue polygon
		const int nr = findSamePointInObjectAndSwapWithLast(inter,*lastPt);
		if(0 <= nr) {
			//last line segment
			polyIn = &(inter->poly[inter->size-1]);
			//get the other point in this polygon and save into polyOut
			append2VecPoint(polyOut,polyIn->points[(nr+1)%2]);
		}
		//remove last poly from inter, because it is already saved or degenerated
		objectSetSize(inter,inter->size-1);
	}
	//last point can be deleted, because he is the same as the first (closes polygon)
	vecPointSetSize(polyOut,polyOut->size-1);
}

double pointPlaneDistance(const struct Plane A, const Vector3 p) {
	return dot(A.n,p)+(A.d);
}

int pointBeforePlane(const struct Plane A, const Vector3 p) {
	return pointPlaneDistance(A,p) > 0.0;
}

int intersectPlaneEdge(Vector3 output, const struct Plane A, const Vector3 a, const Vector3 b) {
	Vector3 diff;
	double t;
	diffVector3(diff,b,a);
	t = dot(A.n,diff);
	if(0.0 == t)
		return 0;
	t = (A.d-dot(A.n,a))/t;
	if(t < 0.0 || 1.0 < t)
		return 0;
	linCombVector3(output,a,diff,t);
	return 1;
}

void clipVecPointByPlane(const struct VecPoint poly, const struct Plane A, struct VecPoint* polyOut, struct VecPoint* interPts) {
	int i;
	int *outside = 0;
	if(poly.size < 3 || 0 == polyOut) {
		return;
	}
	outside = (int*)realloc(outside,sizeof(int)*poly.size);
	//for each point
	for(i = 0; i < poly.size; i++) {
		outside[i] = pointBeforePlane(A,poly.points[i]);
	}
	for(i = 0; i < poly.size; i++) {
		int idNext = (i+1) % poly.size;
		//both outside -> save none
		if(outside[i] && outside[idNext]) {
			continue;
		}
		//outside to inside -> calc intersection save intersection and save i+1
		if(outside[i]) {
			Vector3 inter;
			if(intersectPlaneEdge(inter,A,poly.points[i],poly.points[idNext])) {
				append2VecPoint(polyOut,inter);
				if(0 != interPts) {
					append2VecPoint(interPts,inter);
				}
			}
			append2VecPoint(polyOut,poly.points[idNext]);
			continue;
		}
		//inside to outside -> calc intersection save intersection
		if(outside[idNext]) {
			Vector3 inter;
			if(intersectPlaneEdge(inter,A,poly.points[i],poly.points[idNext])) {
				append2VecPoint(polyOut,inter);
				if(0 != interPts) {
					append2VecPoint(interPts,inter);
				}
			}
			continue;
		}
		//both inside -> save point i+1
		append2VecPoint(polyOut,poly.points[idNext]);
	}
	outside = (int*)realloc(outside,0);
}

void clipObjectByPlane(const struct Object obj, const struct Plane A, struct Object* objOut) {
	int i;
	struct Object inter = OBJECT_NULL;
	struct Object objIn = OBJECT_NULL;
	if(0 == obj.size || 0 == objOut)
		return;
	if(obj.poly == objOut->poly) {
		//need to copy object if input and output are the same
		copyObject(&objIn,obj);
	}
	else {
		objIn = obj;
	}
	emptyObject(objOut);

	for(i = 0; i < objIn.size; i++)
	{
		int size = objOut->size;
		objectSetSize(objOut,size+1);
		objectSetSize(&inter,size+1);
		clipVecPointByPlane(objIn.poly[i],A, &(objOut->poly[size]) , &(inter.poly[size]));
		//if whole poly was clipped away -> delete empty poly
		if(0 == objOut->poly[size].size) {
			objectSetSize(objOut,size);
			objectSetSize(&inter,size);
		}
	}
	//add a polygon of all intersection points with plane to close the object
	appendIntersectionVecPoint(objOut,&inter);
	emptyObject(&inter);
	emptyObject(&objIn);
}


void clipObjectByAABox(struct Object* obj, const struct AABox box) {
	struct VecPlane planes = VECPLANE_NULL;
	int i;
	if(0 == obj) {
		return;
	}
	
	calcAABoxPlanes(&planes,box);
	for(i = 0; i < planes.size; i++) {
		clipObjectByPlane(*obj,planes.plane[i],obj);
	}

	emptyVecPlane(&planes);
}

int clipTest(const double p, const double q, double* u1, double* u2) {
	// Return value is 'true' if line segment intersects the current test
	// plane.  Otherwise 'false' is returned in which case the line segment
	// is entirely clipped.
	if(0 == u1 || 0 == u2) {
		return 0;
	}
	if(p < 0.0) {
		double r = q/p;
		if(r > (*u2)) {
			return 0;
		}
		else {
			if(r > (*u1)) {
				(*u1) = r;
			}
			return 1;
		}
	}
	else {
		if(p > 0.0)	{
			double r = q/p;
			if(r < (*u1)) {
				return 0;
			}
			else {
				if(r < (*u2)) {
					(*u2) = r;
				}
				return 1;
			}
		}
		else {
			return q >= 0.0;
		}
	}
}

int intersectionLineAABox(Vector3 v, const Vector3 p, const Vector3 dir, const struct AABox b) {
	double t1 = 0.0;
	double t2 = DOUBLE_MAX;
	int intersect =
		clipTest(-dir[2],p[2]-b.min[2],&t1,&t2) && clipTest(dir[2],b.max[2]-p[2],&t1,&t2) &&
		clipTest(-dir[1],p[1]-b.min[1],&t1,&t2) && clipTest(dir[1],b.max[1]-p[1],&t1,&t2) &&
		clipTest(-dir[0],p[0]-b.min[0],&t1,&t2) && clipTest(dir[0],b.max[0]-p[0],&t1,&t2);
	if(!intersect) {
		return 0;
	}
	intersect = 0;
	if(0 <= t1) {
		linCombVector3(v,p,dir,t1);
		intersect = 1;
	}
	if(0 <= t2) {
		linCombVector3(v,p,dir,t2);
		intersect = 1;
	}
	return intersect;
}


void includeObjectLightVolume(struct VecPoint* points, const struct Object obj,
							  const Vector3 lightDir, const struct AABox sceneAABox) {
	if(0 != points) {
		int i, size;
		Vector3 ld;
		copyVector3Values(ld,-lightDir[0],-lightDir[1],-lightDir[2]);
		convObject2VecPoint(points,obj);
		size = points->size;
		//for each point add the point on the ray in -lightDir
		//intersected with the sceneAABox
		for(i = 0; i < size; i++) {
			Vector3 pt;
			if(intersectionLineAABox(pt,points->points[i],ld,sceneAABox) ) {
				append2VecPoint(points,pt);
			}
		}
	}
}

void calcFocusedLightVolumePoints(struct VecPoint* points, const Matrix4x4 invEyeProjView,
								  const Vector3 lightDir,const struct AABox sceneAABox) {
	Vector3x8 pts;
	struct Object obj = OBJECT_NULL;

	calcViewFrustumWorldCoord(pts,invEyeProjView);
	calcViewFrustObject(&obj, pts);
	clipObjectByAABox(&obj,sceneAABox);
	includeObjectLightVolume(points,obj,lightDir,sceneAABox);
	emptyObject(&obj);
}

void calcViewFrustumPlanes(struct VecPlane* planes, const Matrix4x4 eyeProjView) {
	if(0 != planes) {
		int i;
		
		planesSetSize(planes, 6);
	
		//---- extract the plane equations
		for (i = 0; i < 4; i++)	{
			setPlaneCoeff(eyeProjView[i * 4 + 3] - eyeProjView[i * 4 + 0], planes->plane + 0, i);   // right plane
			setPlaneCoeff(eyeProjView[i * 4 + 3] + eyeProjView[i * 4 + 0], planes->plane + 1, i);	// left plane
			setPlaneCoeff(eyeProjView[i * 4 + 3] + eyeProjView[i * 4 + 1], planes->plane + 2, i);	// bottom plane
			setPlaneCoeff(eyeProjView[i * 4 + 3] - eyeProjView[i * 4 + 1], planes->plane + 3, i);	// top plane
			setPlaneCoeff(eyeProjView[i * 4 + 3] - eyeProjView[i * 4 + 2], planes->plane + 4, i);	// far plane
			setPlaneCoeff(eyeProjView[i * 4 + 3] + eyeProjView[i * 4 + 2], planes->plane + 5, i);	// near plane
		}

		//---- normalize the coefficients
		for (i = 0; i < 6; i++)	{
			float invLength = 1;
			float length = (float)sqrt(squaredLength(planes->plane[i].n));

			if(length) invLength	= 1.0f / length;

			planes->plane[i].n[0] *= invLength;
			planes->plane[i].n[1] *= invLength;
			planes->plane[i].n[2] *= invLength;
			
			planes->plane[i].d *= invLength;
		}
	}
}

int calcAABNearestVertexIdx(Vector3 clipPlaneNormal) {
	int	index;

	//     7+------+6
	//     /|     /|
	//    / |    / |
	//   / 4+---/--+5
	// 3+------+2 /    y   z
	//  | /    | /     |  /
	//  |/     |/      |/
	// 0+------+1      *---x
	if (clipPlaneNormal[0] <= 0.0f) {
		index	= (clipPlaneNormal[1] <= 0.0f) ? 0 : 3;
	}
	else {
		index	= (clipPlaneNormal[1] <= 0.0f) ? 1 : 2;
	}

	return (clipPlaneNormal[2] <= 0.0f) ? index : 4 + index;
}

int calcAABFarthestVertexIdx(Vector3 clipPlaneNormal) {
	int index;

	if (clipPlaneNormal[0] > 0.0f) {
		index	= (clipPlaneNormal[1] > 0.0f) ? 0 : 3;
	}
	else {
		index	= (clipPlaneNormal[1] > 0.0f) ? 1 : 2;
	}

	return (clipPlaneNormal[2] > 0.0f) ? index : 4 + index;
}

void calcAABNPVertexIndices(int *vertexIndices, const struct VecPlane planes) {
	int i;
	for (i = 0; i < 6; i++)	
	{
		vertexIndices[i * 2 + 0] = calcAABNearestVertexIdx(planes.plane[i].n);	// n-vertex
		vertexIndices[i * 2 + 1] = calcAABFarthestVertexIdx(planes.plane[i].n); // p-vertex
	}
}

void combineAABoxes(struct AABox *a, const struct AABox b) {
	appendToCubicHull(a->min, a->max, b.min);
	appendToCubicHull(a->min, a->max, b.max);
}

double calcAABoxVolume(const struct AABox aab)
{
	return (aab.max[0] - aab.min[0]) * 
		   (aab.max[1] - aab.min[1]) * 
		   (aab.max[2] - aab.min[2]);
}


void calcAABoxCenter(Vector3 vec, const struct AABox aab)
{
	diffVector3(vec, aab.max, aab.min);
	vec[0] *= 0.5f; vec[1] *= 0.5f; vec[2] *= 0.5f;
	
	addVector3(vec, aab.min, vec);
}


double calcAABoxSurface(const struct AABox aab)
{
	return 2 * (aab.max[0] - aab.min[0]) * (aab.max[1] - aab.min[1]) +
		   2 * (aab.max[0] - aab.min[0]) * (aab.max[2] - aab.min[2]) +
		   2 * (aab.max[1] - aab.min[1]) * (aab.max[2] - aab.min[2]);
}

void clipAABoxByAABox(struct AABox *aab, const struct AABox enclosed)
{
	int i;
	for(i=0; i < 3; i++)
	{
		if(aab->min[i] < enclosed.min[i])
			aab->min[i] = enclosed.min[i];
		if(aab->max[i] > enclosed.max[i])
			aab->max[i] = enclosed.max[i];
	}
}


void rotateVectorZ(Vector3 v, double rad)
{
	Vector3 temp;

	temp[0] = v[0]*cos(rad) - v[1]*sin(rad);
	temp[1] = v[0]*sin(rad) + v[1]*cos(rad);
	temp[2] = v[2];

	copyVector3(v, temp);
}


void rotateVectorX(Vector3 v, double rad)
{
	Vector3 temp;
	
	temp[0] = v[0];
	temp[1] = v[1]*cos(rad) - v[2]*sin(rad);
	temp[2] = v[1]*sin(rad) + v[2]*cos(rad);
	
	copyVector3(v, temp);
}


void rotateVectorY(Vector3 v, double rad)
{
	Vector3 temp;

	temp[0] = v[2]*sin(rad) + v[0]*cos(rad);
	temp[1] = v[1]; 
	temp[2] = v[2]*cos(rad) - v[0]*sin(rad);

	copyVector3(v, temp);
}



void rotateVector(Vector3 result, const Vector3 p, const float rad, const Vector3 axis)
{
	int i;
	// using quaternions (vector and scalar part)
	float s = (float)cos(rad * 0.5);
	float h = (float)sin(rad * 0.5);
	
	Vector3 v = {axis[0] * h, axis[1]* h, axis[2] * h};
	Vector3 v_cross_p;
	Vector3 v_cross_v_cross_p;

	cross(v_cross_p, v, p);
	cross(v_cross_v_cross_p, v, v_cross_p);

	for(i=0; i<3; i++)
	{
		result[i] = s * s * p[i] + v_cross_p[i] * (float)dot(v, p) + 
					2.0f * s * v_cross_p[i] + v_cross_v_cross_p[i];
	}

}


void vectorNormal(Vector3 result, const Vector3 v)
{
	result[0] = result[1] = result[2] = 0.0f;
	
	// to avoid 0-vector: assure that at least one component is not 0
	if((v[0] != 0.0f) || (v[1] != 0.0f))
	{
		result[0] = -v[1];
		result[1] = v[0];
	}	
	else if(v[0] != 0.0f) 
	{
		result[0] = -v[2];
		result[2] = v[0];
	}
	else
	{
		result[1] = -v[2];
		result[2] = v[1];
	}
}