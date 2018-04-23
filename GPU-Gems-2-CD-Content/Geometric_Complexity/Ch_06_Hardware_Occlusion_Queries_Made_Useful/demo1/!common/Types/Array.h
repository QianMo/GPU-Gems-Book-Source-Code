//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef ArrayH
#define ArrayH
//---------------------------------------------------------------------------
template<class T> class Array {
private:
	Array(const Array<T>& dyn) { }
	Array<T>& operator=(const Array<T>& dyn) { return *this; }

protected:
	T* data;

	void setDataSize(const unsigned int cnt) {
		data = (T*)realloc(data,cnt*sizeof(T));
    }
	
	//copies data but checks not if source holds enough bytes for reading!!
	void copyData(const Array<T>& dyn) { 
		if(data != dyn.data) {
			memcpy(data,dyn.data,size()*sizeof(T));
		}
	}

public:
	Array(): data(0) { }
	virtual ~Array() { setDataSize(0); }

	T* operator&() { return data; }
	const T* operator&() const { return data; }

	/* this const T* operator& is very important for const References
		because the compile generates a default, which returns garbage
	*/
//	T& operator&() { if(size() > 0) { return *data; } }

	T& operator[](const unsigned int id) { return data[id]; }
	const T& operator[](const unsigned int id) const { return data[id]; }

	const bool operator==(const Array<T>& dyn2) const {
		if(dyn2.size() == size()) {
			return 0 == memcmp(data,dyn2.data,size()*sizeof(T));
		}
		return false;
	}

	const bool operator!=(const Array<T>& dyn2) const {
		return !operator==(dyn2);
	}

	virtual const unsigned int size() const = 0;
	const bool empty() const { return size() == 0; }

	const T& last() const { return data[size()-1]; }
	T& last() { return data[size()-1]; }

    void fillWith(const T& v) {
        for(unsigned int i = 0; i < size(); i++) {
            data[i] = v;
        }
    }

};

#endif


