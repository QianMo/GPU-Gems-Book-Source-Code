//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef MathIOH
#define MathIOH

#include "Vector.h"
#include <iostream>
#include <Base/StreamTools.h>
//---------------------------------------------------------------------------
namespace Math {

template<class REAL, unsigned SIZE> 
std::ostream& operator<<(std::ostream& s, const Tuppel<REAL,SIZE>& input) {
	return Tools::tuppelOut(s,input);
}

template<class REAL, unsigned SIZE> 
std::istream& operator>>(std::istream& s, Tuppel<REAL,SIZE>& output) {
	Tools::tuppelIn(s,output);
	return s;
}

//namespace
};
#endif
