//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef VectorH
#define VectorH
//---------------------------------------------------------------------------
#include "Tuppel.h"

namespace Math {

template<class REAL = float, const unsigned SIZE = 3>
struct Vector : public Tuppel<REAL,SIZE> {
	typedef Tuppel<REAL,SIZE> T;
	// construction
	Vector() { /* no initialization -> performance */ }
	Vector(const T& v) : T(v) { }
	Vector(const Vector& v): T(v) { }
	Vector(const REAL p[SIZE]): T(p) { }

	Vector& invert() { 
		for(unsigned i = 0; i < SIZE; i++)
			array[i] = -array[i];
		return *this; 
	}

	REAL dot(const Vector& v) const {
		REAL result = array[0]*v[0];
		for(unsigned i = 1; i < SIZE; i++)
			result += array[i]*v[i];
		return result;
	}

	REAL squaredLength() const { return dot(*this); }
	REAL length() const { return sqrt(squaredLength()); }
	Vector& normalize() { operator/=(length()); return *this; }

	REAL unitize(const REAL& rTolerance = 1e-06) {
		Real rLength = length();
		if(rLength > rTolerance) {
			operator/=(rLength);
		}
		else {
			rLength = 0.0;
		}
		return rLength;
	}
};

//namespace
};
#endif