//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef TypeIOH
#define TypeIOH
//---------------------------------------------------------------------------
#pragma warning(disable:4786)  // Disable warning message
#include <vector>
#include "StreamTools.h"
#include "../Types/Array.h"
//---------------------------------------------------------------------------
template<class T> inline std::ostream& operator<<(std::ostream& s, const Array<T>& input) {
    return Tools::tuppelOut(s,input);
}

template<class T> inline std::ostream& operator<<(std::ostream& s, const std::vector<T>& input) {
    return Tools::tuppelOut(s,input);
}

namespace Tools {

template<class A, class B> inline std::ostream& operator<<(std::ostream& s, const std::pair<A,B>& input) {
	return s << '<' << input.first << ' ' << input.second << '>';
}

template<class A, class B> inline std::istream& operator>>(std::istream& s, std::pair<A,B>& output) {
    if(s.good()) {
		eatToChar(s,'<');
		if(s.fail()) {
            return s;
        }
		s >> output.first;
		s >> output.second;
		Tools::eatToChar(s,'>');
    }
	return s;
}

//namespace
};

#endif
