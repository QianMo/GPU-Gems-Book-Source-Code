//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef DynamicArrayH
#define DynamicArrayH
#include "Array.h"
//---------------------------------------------------------------------------
template<class T> class DynamicArray : public Array<T> {
private:
	unsigned count;

public:
	DynamicArray(): count(0) { }
	DynamicArray(const unsigned cnt): count(0) { resize(cnt); }

	DynamicArray(const DynamicArray<T>& dyn): count(0) { 
		//if a(a) is handled by copyData
		resize(dyn.count);
		copyData(dyn);
	}

	DynamicArray<T>& operator=(const DynamicArray<T>& dyn) {
		//if a = a is handled by copyData
		resize(dyn.count);
		copyData(dyn);
		return *this;
	}

	void resize(const unsigned cnt) {
        if(cnt != count) {
			setDataSize(cnt);
			count = cnt;
		}
    }

	void clear() { resize(0); }
	void swap(DynamicArray<T>& a) {
		if(a.data == data) 
			return;
		unsigned temp = a.size();
		a.count = count;
		count = temp;
		
		T* tempData = a.data;
		a.data = data;
		data = tempData;
	}

	virtual const unsigned size() const { return count; }

    void append(const T& v) {
        resize(size()+1);
        //todo: potentieller fehler bei objekten  mit operator= ungleich einfachem kopieren
		data[count-1] = v;
    }
	inline void push_back(const T& v) { append(v); }
};

#endif


