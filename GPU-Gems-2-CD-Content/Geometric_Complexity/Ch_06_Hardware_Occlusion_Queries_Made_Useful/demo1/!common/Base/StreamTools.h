//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.

#ifndef StreamToolsH
#define StreamToolsH
//---------------------------------------------------------------------------
#pragma warning(disable:4786)  // Disable warning message
#include <iostream>
//---------------------------------------------------------------------------
namespace Tools {

template<class T>
inline std::ostream& tuppelOut(std::ostream& s, const T& input, const char trenner = ' ', 
								const char open = '(', const char close = ')') {
	if(input.size() > 0) {
		s << open;
		uint ct = 0;
		for(; ct < input.size()-1; ct++) {
			s << input[ct] << trenner;		
		}
		s << input[ct];		
		s << close;
	}
	return s;
}

inline void parseError(std::istream& s) {
	s.setstate(std::ios_base::failbit);
}

inline void parseErrorCritical(std::istream& s) {
	s.setstate(std::ios_base::badbit);
}

inline void eatToChar(std::istream& s, const char ende) {
    if(s.good()) {
		char c;
		while(s.get(c)) {
			if(!isspace(c)) {
				if(ende == c) {
					return;
				}
				s.putback(c);
				parseError(s);
				return;
			}
		}
    }
}

template<class T> inline void tuppelIn(std::istream& s, T& output,
										const char open = '(', const char close = ')') {
    if(s.good()) {
        eatToChar(s,open);
		if(s.fail()) {
            return;
        }
        uint ct = 0;
        while(ct < output.size() && s >> output[ct]) {
            ct++;
        }
        if(ct != output.size()) {
            parseErrorCritical(s);
        }
        eatToChar(s,close);
    }
}

//namespace
}
#endif
