//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef Vector3H
#define Vector3H
//---------------------------------------------------------------------------
#include "Mathematic.h"
#include "Vector.h"

namespace Math {

//typedef double Vec3[3]; //todo: for engine conversion only

template<class REAL = float>
class Vector3: public Vector<REAL,3> {
protected:
	typedef Vector<REAL,3> V3;
	void cross(V3& v, const V3& a, const V3& b) {
		v[0] =  det2x2(a[1],b[1],a[2],b[2]);
		v[1] = -det2x2(a[0],b[0],a[2],b[2]);
		v[2] =  det2x2(a[0],b[0],a[1],b[1]);
	}
public:
	Vector3() { /* no initialization -> performance */ }
	Vector3(const REAL p[3]): V3(p) { } 
	Vector3(const V3& v): V3(v) { }
	Vector3(const Tuppel<REAL,3>& v): V3(v) { }
	Vector3(const REAL x, const REAL y, const REAL z) { 
		array[0] = x;
		array[1] = y;
		array[2] = z;
	}

	V3& cross(const V3& a, const V3& b) {
		cross(*this,a,b);
		return *this;
	}

	V3& unitCross(const V3& a, const V3& b) {
		cross(a,b);
		normalize();
		return *this;
	}

	//Vec3& tovec3() const { return array; }
	static const Vector3 ZERO;
	static const Vector3 ONE;
	static const Vector3 UNIT_X;
	static const Vector3 UNIT_Y;
	static const Vector3 UNIT_Z;
};

typedef Vector3<float> Vector3f;
typedef Vector3<double> Vector3d;

//namespace
};
#endif