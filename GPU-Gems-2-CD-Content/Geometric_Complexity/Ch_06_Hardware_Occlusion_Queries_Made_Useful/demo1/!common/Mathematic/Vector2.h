//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef Vector2H
#define Vector2H
//---------------------------------------------------------------------------
#include "Mathematic.h"
#include "Vector.h"

namespace Math {

template<class REAL = float>
class Vector2: public Vector<REAL,2> {
public:
	Vector2() { /* no initialization -> performance */ }
	Vector2(const REAL p[2]): Vector<REAL,2>(p) { } 
	Vector2(Vector<REAL,2>& v): Vector<REAL,2>(v) { }
	Vector2(const REAL x, const REAL y) { 
		array[0] = x;
		array[1] = y;
	}
};

typedef Vector2<float> Vector2f;
typedef Vector2<double> Vector2d;

//namespace
}
#endif