//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include <float.h>
#include <Mathematic/Mathematic.h>
#include <Mathematic/MathTools.h>
#include "MathStuff.h"

const double DOUBLE_MAX = DBL_MAX;

void appendToCubicHull(V3& min, V3& max, const V3& v) {
	for(unsigned j = 0; j < 3; j++) {
		if(v[j] < min[j]) {
			min[j] = v[j];
		}
		else if(v[j] > max[j]) {
			max[j] = v[j];
		}
	}
}

void calcCubicHull(V3& min, V3& max, const VecPoint& ps) {
	if(ps.size() > 0) {
		min = ps[0];
		max = ps[0];
		for(unsigned i = 1; i < ps.size(); i++) {
			appendToCubicHull(min,max,ps[i]);
		}
	}
}

void calcViewFrustObject(Object& obj, const Vector3x8& p) {
	obj.resize(6);
	for(unsigned i = 0; i < 6; i++) {
		obj[i].resize(4);
	}
	//near poly ccw
	for(unsigned i = 0; i < 4; i++) {
		obj[0][i] = p[i];
	}
	//far poly ccw
	for(unsigned i = 4; i < 8; i++) {
		obj[1][i-4] = p[11-i];
	}
	//left poly ccw
	obj[2][0] = p[0];
	obj[2][1] = p[3];
	obj[2][2] = p[7];
	obj[2][3] = p[4];
	//right poly ccw
	obj[3][0] = p[1];
	obj[3][1] = p[5];
	obj[3][2] = p[6];
	obj[3][3] = p[2];
	//bottom poly ccw
	obj[4][0] = p[4];
	obj[4][1] = p[5];
	obj[4][2] = p[1];
	obj[4][3] = p[0];
	//top poly ccw
	obj[5][0] = p[6];
	obj[5][1] = p[7];
	obj[5][2] = p[3];
	obj[5][3] = p[2];
}

void transformVecPoint(VecPoint& poly, const M4& xForm) {
	for(unsigned i = 0; i < poly.size(); i++) {
		poly[i] = xForm.mulHomogenPoint(poly[i]);
	}
}

void transformObject(Object& obj, const M4& xForm) {
	for(unsigned i = 0; i < obj.size(); i++) {
		transformVecPoint(obj[i],xForm);
	}
}

int findSamePointInVecPoint(const VecPoint& poly, const V3& p) {
	for(unsigned i = 0; i < poly.size(); i++) {
		if(poly[i].alike(p,(double)0.001)) {
			return i;
		}
	}
	return -1;
}

int findSamePointInObjectAndSwapWithLast(Object& inter, const V3& p) {
	if(1 > inter.size()) {
		return -1;
	}
	for(unsigned i = inter.size(); i > 0; i--) {
		VecPoint& poly = inter[i-1];
		if(2 == poly.size()) {
			const int nr = findSamePointInVecPoint(poly,p);
			if(0 <= nr) {
				//swap with last
				poly.swap(inter[inter.size()-1]);
				return nr;
			}
		}
	}
	return -1;
}

void appendIntersectionVecPoint(Object& obj, Object& inter) {
	const unsigned size = obj.size();
	//you need at least 3 sides for a polygon
	if(3 > inter.size()) {
		return;
	}
	//compact inter: remove poly.size != 2 from end on forward
	for(unsigned i = inter.size(); 0 < i; i--) {
		if(2 == inter[i-1].size()) {
			break;
		}
	}
	inter.resize(i);
	//you need at least 3 sides for a polygon
	if(3 > inter.size()) {
		return;
	}
	//make place for one additional polygon in obj
	obj.resize(size+1);
	VecPoint& polyOut = obj[size];
	//we have line segments in each poly of inter 
	//take last linesegment as first and second point
	const VecPoint& polyIn = inter[inter.size()-1];
	polyOut.push_back(polyIn[0]);
	polyOut.push_back(polyIn[1]);
	//remove last poly from inter, because it is already saved
	inter.resize(inter.size()-1);

	//iterate over inter until their is no line segment left
	while(0 < inter.size()) {
		//pointer on last point to compare
		const V3 &lastPt = polyOut[polyOut.size()-1];
		//find same point in inter to continue polygon
		const int nr = findSamePointInObjectAndSwapWithLast(inter,lastPt);
		if(0 <= nr) {
			//last line segment
			const VecPoint& polyIn = inter[inter.size()-1];
			//get the other point in this polygon and save into polyOut
			polyOut.push_back(polyIn[(nr+1)%2]);
		}
		//remove last poly from inter, because it is already saved or degenerated
		inter.resize(inter.size()-1);
	}
	//last point can be deleted, because he is the same as the first (closes polygon)
	polyOut.resize(polyOut.size()-1);
}

bool intersectPlaneEdge(V3& output, const Plane& A, const V3& a, const V3& b) {
	const Line l(a,b-a);
	double t;
	if(A.intersection(l,t)) {
		//-> calculate intersection point
		output = l.getPoint(t);
		return true;
	}
	else {
		return false;
	}
}

void clipVecPointByPlane(const VecPoint poly, const Plane& A, VecPoint& polyOut, VecPoint& interPts) {
	bool *outside = 0;
	if(poly.size() < 3) {
		return;
	}
	outside = (bool*)realloc(outside,sizeof(bool)*poly.size());
	//for each point
	for(unsigned i = 0; i < poly.size(); i++) {
		outside[i] = A.planeBehindPoint(poly[i]);
	}
	for(unsigned i = 0; i < poly.size(); i++) {
		const unsigned idNext = (i+1) % poly.size();
		//both outside -> save none
		if(outside[i] && outside[idNext]) {
			continue;
		}
		//outside to inside -> calc intersection save intersection and save i+1
		if(outside[i]) {
			V3 inter;
			if(intersectPlaneEdge(inter,A,poly[i],poly[idNext])) {
				polyOut.push_back(inter);
				interPts.push_back(inter);
			}
			polyOut.push_back(poly[idNext]);
			continue;
		}
		//inside to outside -> calc intersection save intersection
		if(outside[idNext]) {
			V3 inter;
			if(intersectPlaneEdge(inter,A,poly[i],poly[idNext])) {
				polyOut.push_back(inter);
				interPts.push_back(inter);
			}
			continue;
		}
		//both inside -> save point i+1
		polyOut.push_back(poly[idNext]);
	}
	outside = (bool*)realloc(outside,0);
}

void clipObjectByPlane(const Object& obj, const Plane& A, Object& objOut) {
	Object inter;
	Object objIn(obj);
	if(0 == obj.size())
		return;
	objOut.clear();

	for(unsigned i = 0; i < objIn.size(); i++) {
		const unsigned size = objOut.size();
		objOut.resize(size+1);
		inter.resize(size+1);
		clipVecPointByPlane(objIn[i],A,objOut[size],inter[size]);
		//if whole poly was clipped away -> delete empty poly
		if(0 == objOut[size].size()) {
			objOut.resize(size);
			inter.resize(size);
		}
	}
	//add a polygon of all intersection points with plane to close the object
	appendIntersectionVecPoint(objOut,inter);
}


void clipObjectByAABox(Object& obj, const AABox& box) {
	StaticArray<Plane,6> planes;
	box.calcAABoxPlanes(planes);
	for(unsigned i = 0; i < planes.size(); i++) {
		clipObjectByPlane(obj,planes[i],obj);
	}
}

void includeObjectLightVolume(VecPoint& points, const Object& obj,
							  const V3& lightDir, const AABox& sceneAABox) {
	V3 ld(-lightDir[0],-lightDir[1],-lightDir[2]);
	convObject2VecPoint(points,obj);
	const unsigned size = points.size();
	//for each point add the point on the ray in -lightDir
	//intersected with the sceneAABox
	for(unsigned i = 0; i < size; i++) {
		V3 pt;
		if(sceneAABox.intersectWithLine(pt,Line(points[i],ld)) ) {
			points.push_back(pt);
		}
	}
}

void calcFocusedLightVolumePoints(VecPoint& points,const M4& invEyeProjView,
								  const V3& lightDir,const AABox& sceneAABox) {
	Vector3x8 pts;
	Object obj;

	Math::calcViewFrustumWorldCoord(pts,invEyeProjView);
	calcViewFrustObject(obj,pts);
	clipObjectByAABox(obj,sceneAABox);
	includeObjectLightVolume(points,obj,lightDir,sceneAABox);
}

