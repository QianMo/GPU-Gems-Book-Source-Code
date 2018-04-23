//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef LineH
#define LineH
#include <Mathematic/Vector3.h>
//---------------------------------------------------------------------------
namespace Math {
namespace Geometry {

template<class REAL = float>
struct Line {
	typedef REAL IntersectionResult;
	typedef Vector3<REAL> V3;
	V3 p;
	V3 dir;

	// construction
	//Line() { /* no initialization -> performance */ }
	Line(const V3& vP, const V3& vDirection): p(vP), dir(vDirection) { }

	bool operator==(const Line& l) const { return (p == l.p) && (dir == l.dir); }
	bool operator!= (const Line& l) const { return !operator==(l); }

	void normalize() { dir.normalize(); }

	V3 getPoint(const REAL& t) const {
		return Vector<REAL,3>(p+dir*t);
	}
};

//namespace
}
//namespace
}
#endif