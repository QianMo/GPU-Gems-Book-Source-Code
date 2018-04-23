//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef StringToolsH
#define StringToolsH
//---------------------------------------------------------------------------
#pragma warning(disable:4786)  // Disable warning message
#include <sstream>
#include <string>
//---------------------------------------------------------------------------
namespace Tools {

inline std::string lTrim(const std::string& in) {
	std::string::const_iterator p = in.begin();
    while(p != in.end() && isspace(*p)) {
        p++;
    }
    std::string out(p,in.end());
    return out;
}

inline std::string rTrim(const std::string& in) {
	std::string::const_iterator p = in.end();
    while(p != in.begin() && isspace(*p)) {
        p--;
    }
    std::string out(in.begin(),p);
    return out;
}


inline std::string toUpper(const std::string& in) {
    std::string::const_iterator p = in.begin();
    std::string out;
    while(p != in.end()) {
		
		out += std::string::value_type(toupper(*p));
        p++;
    }
    return out;
}


template<class T> inline std::string toString(const T& input) {
	std::ostringstream converter;
	converter << input;
	return converter.str();
}

template<class T> inline std::string toString(const T& input, const unsigned int precicion) {
	std::ostringstream converter;
	converter.precision(precicion);
	converter << input;
	return converter.str();
}

template<class T> inline void stringTo(const std::string& input, T& out) {
	std::istringstream converter(input);
	converter >> out;
}

//namespace
}
#endif
