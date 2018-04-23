//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef StaticArrayH
#define StaticArrayH
#include "Array.h"
//---------------------------------------------------------------------------
template<class T, const unsigned int SIZE> class StaticArray : public Array<T> {
public:
	StaticArray() { setDataSize(SIZE); }

	StaticArray(const StaticArray<T,SIZE>& dyn): DynamicArray(SIZE) { 
		copyData(dyn);
	}

	StaticArray<T,SIZE>& operator=(const StaticArray<T,SIZE>& dyn) {
		copyData(dyn);
		return *this;
	}

	virtual const unsigned int size() const { return SIZE; }

};

#endif


