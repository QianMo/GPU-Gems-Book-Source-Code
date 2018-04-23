//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef AABoxH
#define AABoxH
//---------------------------------------------------------------------------
#pragma warning(disable:4786)
#include <Types/Array.h>
#include "../Mathematic.h"
//#include "Mathematic/MathTools.h"
#include "../Vector3.h"
#include "../Matrix4.h"
//#include "../VecVector.h"
#include "Plane.h"
#include "Line.h"
//---------------------------------------------------------------------------
namespace Math {
namespace Geometry {

//Axis-Aligned Bounding Box defined through the two extreme points
template<class REAL = float>
struct AABox {
	typedef Vector3<REAL> V3;
protected:
	typedef Plane<REAL> Plane;
	typedef AABox<REAL> B;
	V3 min, max;

//	AABox() { }

	inline void correctMinMax() {
		for(unsigned i = 0; i < min.size(); i++) {
			if(min[i] > max[i]) {
				std::swap(min[i],max[i]);
			}
		}
	}
    enum Axe { X = 0, Y = 1, Z = 2 };

	inline void expandToContain(const unsigned axe, const REAL& value) {
		if(value > max[axe]) {
			max[axe] = value;
		}
		else {
			if(value < min[axe]) {
				min[axe] = value;
			}
		}
	}

public:
	AABox(const AABox& box): min(box.getMin()), max(box.getMax()) { }
	AABox(const V3& vA, const V3& vB): min(vA), max(vB) { correctMinMax(); }

/*	AABox(const Matrix4<REAL>& mtxTransform, const AABox& box) {
		//simple approach:
		Vector3x8 p; 
		box.computeVerticesLeftHanded(p);
		M = m = mtxTransform.mulHomogenPoint(p[0]); //attention not with AABox::ZERO
		for(uint i = 1; i < 8; i++) {
			expandToContain(mtxTransform.mulHomogenPoint(p[i]));
		}

		//	the following doesn't work
		//	m = mtxTransform.mulHomogen(box.getm());
		//	M = mtxTransform.mulHomogen(box.getM());
		//	correctmM();
	}

*/	AABox(const Matrix4<REAL>& mtxTransform, const Array<V3>& p) {
		if(0 == p.size()) {
			throw GeometryException("empty point list in AABox constructor");
		}
		min = (max = mtxTransform.mulHomogenPoint(p[0])); //attention not with AABox::ZERO
		for(unsigned i = 1; i < p.size(); i++) {
			expandToContain(mtxTransform.mulHomogenPoint(p[i]));
		}
	}

	inline const V3& getMin() const { return min; }
	inline const V3& getMax() const { return max; }
	inline V3 getCenter() const {	return getMin()+getExtents(); }
	inline V3 getExtents() const { return (getMax()-getMin())*0.5; }
	inline REAL volume() const { return (getMax()-getMin()).abs().product(); }

	inline void expandToContain(const V3& v) {
		for(unsigned j = 0; j < v.size(); j++) {
			expandToContain(j,v[j]);
		}
	}

	inline AABox& operator+=(const AABox& a) {
		expandToContain(a.getMin());
		expandToContain(a.getMax());
		return *this;
	}

	inline AABox operator+(const AABox& a) const {
		B result(*this);
		result.expandToContain(a.getMin());
		result.expandToContain(a.getMax());
		return result;
	}

	template<class T> AABox<T> convert2() const {
		return AABox<T>(getMin().convert2<T>(),getMax().convert2<T>());
	}

	//calculates the 8 corner points of the AABox
	void computeVerticesLeftHanded(Array<V3>& v) const {
		if(8 > v.size()) {
			throw GeometryException("AABox::computeVerticesLeftHanded with Array smaller 8 called");
		}
		const V3& m = getMin();
		const V3& M = getMax();
		//generate 8 corners of the bbox
		v[0] = V3(m[0],m[1],m[2]); //     7+------+6
		v[1] = V3(M[0],m[1],m[2]); //     /|     /|
		v[2] = V3(M[0],M[1],m[2]); //    / |    / |
		v[3] = V3(m[0],M[1],m[2]); //   / 4+---/--+5
		v[4] = V3(m[0],m[1],M[2]); // 3+------+2 /    y   z
		v[5] = V3(M[0],m[1],M[2]); //  | /    | /     |  /
		v[6] = V3(M[0],M[1],M[2]); //  |/     |/      |/
		v[7] = V3(m[0],M[1],M[2]); // 0+------+1      *---x
	}

	//calculates the 8 corner points of the AABox
	void computeVerticesRightHanded(Array<V3>& v) const {
		if(8 > v.size()) {
			throw GeometryException("AABox::computeVerticesRightHanded with Array smaller 8 called");
		}
		const V3& m = getMin();
		const V3& M = getMax();
		//generate 8 corners of the bbox
		v[0] = V3(m[0],m[1],M[2]); //     7+------+6
		v[1] = V3(M[0],m[1],M[2]); //     /|     /|
		v[2] = V3(M[0],M[1],M[2]); //    / |    / |
		v[3] = V3(m[0],M[1],M[2]); //   / 4+---/--+5
		v[4] = V3(m[0],m[1],m[2]); // 3+------+2 /    y  -z
		v[5] = V3(M[0],m[1],m[2]); //  | /    | /     |  /
		v[6] = V3(M[0],M[1],m[2]); //  |/     |/      |/
		v[7] = V3(m[0],M[1],m[2]); // 0+------+1      *---x
	}

	//calculates the 6 planes of the given AABox
	void calcAABoxPlanes(Array<Plane>& planes) const {
		const V3& min = getMin();
		const V3& max = getMax();
		if(6 > planes.size()) {
			throw GeometryException("AABox::calcAABoxPlanes with Array smaller 6 called");
		}
		//bottom plane
		planes[0] = Plane(0,-1,0,Math::abs(min[1]));
		//top plane
		planes[1] = Plane(0,1,0,Math::abs(max[1]));
		//left plane
		planes[2] = Plane(-1,0,0,Math::abs(min[0]));
		//right plane
		planes[3] = Plane(1,0,0,Math::abs(max[0]));
		//back plane
		planes[4] = Plane(0,0,-1,Math::abs(min[2]));
		//front plane
		planes[5] = Plane(0,0,1,Math::abs(max[2]));
	}

	bool intersectWithLine(V3& v, const Line<REAL>& l) const {
		const V3& p = l.p;
		const V3& dir = l.dir;
		const V3& min = getMin();
		const V3& max = getMax();
		double t1 = 0.0;
		double t2 = DOUBLE_MAX;
		bool intersect =
			clipTest(-dir[2],p[2]-min[2],t1,t2) && clipTest(dir[2],max[2]-p[2],t1,t2) &&
			clipTest(-dir[1],p[1]-min[1],t1,t2) && clipTest(dir[1],max[1]-p[1],t1,t2) &&
			clipTest(-dir[0],p[0]-min[0],t1,t2) && clipTest(dir[0],max[0]-p[0],t1,t2);
		if(!intersect) {
			return false;
		}
		intersect = false;
		if(0 <= t1) {
			v = p;
			v += t1*dir;
			intersect = true;
		}
		if(0 <= t2) {
			v = p;
			v += t2*dir;
			intersect = true;
		}
		return intersect;
	}

	const PlaneSide getPlaneSide(const Plane& A) const {
		const V3& n = A.getN();
		const V3 min = getMin();
		const V3 max = getMax();
		V3 vMin, vMax;
		//extend: the for could be cached in table for speedup, but the 3 if remain
		for(unsigned i = 0; i < V3::size(); i++) {
			if(n[i] >= 0.0) {
				vMin[i] = min[i];
				vMax[i] = max[i];
			}
			else {
				vMin[i] = max[i];
				vMax[i] = min[i];
			}
		}

		if(A.distance(vMin) > 0.0) {
			return BEFORE;
		}
		if(A.distance(vMax) < 0.0) {
			return BEHIND;
		}
		return CROSS;
	}

	REAL squaredDistance(const V3& p) const {
		const V3 e(getExtents());

		// compute coordinates of point in box coordinate system	
		V3 s(p-getCenter());
		for(unsigned i = 0; i < V3::size(); i++) {
			if(s[i] <= -e[i]) {
				s[i] += e[i];
			}
			else {
				if(s[i] < e[i]) {
					s[i] = REAL(0.0);
				}
				else {
					s[i] -= e[i];
				}
			}	
		}
		return s.dot(s);
	}
	inline REAL distance(const V3& p) const { return Math::sqrt(squaredDistance(p)); }
	inline AABox copy() const { return AABox(*this); }

	inline AABox& scale(const REAL& factor) {
		min *= factor;
		max *= factor;
		return *this;
	}

	inline AABox& scaleCenter(const REAL& factor) {
		V3 extents = getExtents();
		const V3 center = getMin()+extents;
		extents *= factor;
		min = center-extents;
		max = center+extents;
		return *this;
	}

	inline AABox& translate(const V3& translation) {
		min += translation;
		max += translation;
		return *this;
	}
//AABox
};


//namespace
}
//namespace
}
#endif
