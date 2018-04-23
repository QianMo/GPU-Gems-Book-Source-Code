//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef PlaneH
#define PlaneH
#include "../Mathematic.h"
#include "../Vector3.h"
#include "Line.h"
//---------------------------------------------------------------------------
namespace Math {
namespace Geometry {

enum PlaneSide { BEFORE, BEHIND, CROSS };

//Plane defined through normal vector n and signed distance to origin d
template<class REAL = float>
class Plane {
protected:
	typedef Vector3<REAL> V3;
	V3 n;
	REAL d;
public:
	// construction
	Plane(const V3& vN, const REAL& vD): 
		n(vN), d(vD) { hessNorm(); }

	Plane(const REAL& vA, const REAL& vB, const REAL& vC, const REAL& vD): 
		n(vA,vB,vC), d(vD) { hessNorm(); }

	Plane(const REAL abcd[4]): 
		n(abcd[0],abcd[1],abcd[2]), d(abcd[3]) { hessNorm(); }

	Plane(const V3& a, const V3& b, const V3& c) {
		n = V3(b-a).cross(c-a);
		n.unitize();
		d = n.dot(a);
	}

	Plane(const V3& p, const V3& vN) {	
		n = vN;
		n.unitize();
		d = n.dot(p);
	}

	inline bool operator==(const Plane&) const { return p.n == n && p.d == d; }
	inline bool operator!= (const Plane&) const { return !operator==(p); }

	inline const V3& getN() const { return n; }
	inline const REAL& getD() const { return d; } 
	// d has different meanings in different math books!!
	// because sometimes a plane is defined as [n dot p + d = 0] or [n dot p = d]
	// so better avoid code which depends on d
	inline void hessNorm() {
		REAL t = n.squaredLength();
		if(t != 1.0f) {
			t = Math::sqrt(t);
			n /= t;
			d /= t;
		}
	}

	inline const bool intersection(const Line<REAL>& l, REAL& t) const {
		REAL a = n.dot(l.dir);
		if(Math::abs(a) < Math::Const<REAL>::near_epsilon()) {
			return false;
		}
		t = (d-n.dot(l.p))/a;
		return true;
	}

	inline const bool intersectionWithEdge(const V3 a, const V3 b, REAL& t) {
		V3 diff = b-a;
		REAl t = n.dot(diff);
		if(0.0 == t) //edge and plane are parallel
			return false;
		t = (A.getD()-A.getN().dot(a))/t;
		if(t < 0.0 || 1.0 < t)
			return false; //intersection is outside of edge
		
		//-> plane and edge intersect
		return true;
	}

	REAL distance(const V3& p) const {
		return distance(p[0],p[1],p[2]);
	}

	REAL distance(const REAL& x, const REAL& y, const REAL& z) const {
		return n[0]*x+n[1]*y+n[2]*z-d;
	}

	inline const bool planeBehindPoint(const V3& p) const {	return distance(p) > 0.0;	} //plane behind point
	inline const bool planeInFrontPoint(const V3& p) const { return distance(p) < 0.0; } //plane in front of point
	
	inline const PlaneSide getPlaneSide(const V3& p) const {
		const Real dist = distance(p);
		if(dist < 0.0) {
			return BEHIND;
		}
		else {
			if(dist > 0.0) {
				return BEFORE;
			}
			else {
				return CROSS;
			}
		}
	}

};

//namespace
}
//namespace
}
#endif