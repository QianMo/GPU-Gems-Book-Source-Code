//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef Vector4H
#define Vector4H
//---------------------------------------------------------------------------
#include "Vector.h"
#include "Vector3.h"

namespace Math {

template<class REAL = float>
class Vector4: public Vector<REAL,4> {
protected:
	typedef Vector<REAL,4> V4;
	void init(const REAL x, const REAL y, const REAL z, const REAL w) { 
		array[0] = x;
		array[1] = y;
		array[2] = z;
		array[3] = w;
	}
public:
	Vector4() { /* no initialization -> performance */ }
	Vector4(const REAL x, const REAL y, const REAL z, const REAL w) { init(x,y,z,w); }
	Vector4(const REAL p[4]): V4(p) { } 
	Vector4(const V4& v): V4(v) { }
	Vector4(const Tuppel<REAL,4>& v): V4(v) { }
	Vector4(const Vector3<REAL>& v, const REAL& r) { init(v[0],v[1],v[2],r); }

	Vector4& makeHomogen() { operator/=(array[4]); return *this; }
	
//	Vector4& mult(const Matrix4&, const Vector4&);
	Vector3<REAL> toVector3() const { return Vector3<REAL>(x,y,z); }

	static const Vector4 ZERO;
	static const Vector4 ONE;
	static const Vector4 UNIT_X;
	static const Vector4 UNIT_Y;
	static const Vector4 UNIT_Z;
	static const Vector4 UNIT_W;
};

/*Vector4& Vector4::mult(const Matrix4& m, const Vector4& v) {
	x = m.a11*v.x + m.a12*v.y + m.a13*v.z + m.a14*v.w;
	y = m.a21*v.x + m.a22*v.y + m.a23*v.z + m.a24*v.w;
	z = m.a31*v.x + m.a32*v.y + m.a33*v.z + m.a34*v.w;
	w = m.a41*v.x + m.a42*v.y + m.a43*v.z + m.a44*v.w;
	return *this;
}

inline Vector4 operator*(const Matrix4& m, const Vector4& v) {
	return Vector4().mult(m,v);
}
*/

typedef Vector4<float> Vector4f;
typedef Vector4<double> Vector4d;

//namespace
};
#endif