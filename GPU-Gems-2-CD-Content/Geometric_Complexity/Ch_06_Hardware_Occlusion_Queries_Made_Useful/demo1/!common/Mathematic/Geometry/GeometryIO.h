//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef GeometryIOH
#define GeometryIOH

#include <iostream>
#include <Base/TypeIO.h>
#include "../MathIO.h"
#include "Line.h"
#include "Plane.h"
#include "AABox.h"
#include "Frustum.h"
//---------------------------------------------------------------------------
namespace Math {
namespace Geometry {

template<class REAL> 
inline std::ostream& operator<<(std::ostream& s, const Line<REAL>& input) {
	return s << "(p:" << input.p << " dir:" << input.dir << ')';
}

template<class REAL> 
inline std::ostream& operator<<(std::ostream& s, const Plane<REAL>& input) {
	return s << "(n:" << input.getN() << " d:" << input.getD() << ')';
}

template<class REAL> 
inline std::ostream& operator<<(std::ostream& s, const AABox<REAL>& input) {
	typedef std::pair<Vector3<REAL>,Vector3<REAL> > MinMax;
	return Tools::operator<<(s,MinMax(input.getMin(),input.getMax()));
}

template<class REAL> 
inline std::istream& operator>>(std::istream& s, AABox<REAL>& output) {
	typedef std::pair<Vector3<REAL>,Vector3<REAL> > MinMax;
	MinMax m;
	Tools::operator>>(s,m);
	if(s.good()) {
		output = AABox<REAL>(m.first,m.second);
	}
	return s;
}

template<class REAL> 
inline std::ostream& operator<<(std::ostream& s, const Frustum<REAL>& input) {
	return Tools::tuppelOut(s,input.getVecPlanes());
}

//namespace
}
//namespace
}
#endif
