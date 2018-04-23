//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef PersistentH
#define PersistentH
//---------------------------------------------------------------------------
#pragma warning(disable:4786)
#include <iostream>
#include "Base.h"
//---------------------------------------------------------------------------
class Persistent : public Base {
public:
	//todo: make them abstract
	virtual std::ostream& put(std::ostream& s) const { return s; }
	virtual std::istream& get(std::istream& s) { return s; }
};

inline std::ostream& operator<<(std::ostream& s, const Persistent& p) {
	return p.put(s);
}

inline std::istream& operator>>(std::istream& s, Persistent& p) {
	return p.get(s);
}

#endif
