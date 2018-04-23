//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.

#ifndef SmartPointerH
#define SmartPointerH
#pragma warning(disable:4786)  // Disable warning message for too long debug strings
template<class T> class SmartPointer {
	T *pT;
	void saveNew() { if(pT != 0) pT->newRef(); }
	void saveDel() { if(pT != 0) pT->delRef(); }
public:
	SmartPointer(T *_pT = 0) : pT(_pT) { saveNew(); }
    SmartPointer(const SmartPointer<T>& _smartP) : pT(_smartP.pT) { saveNew(); }
	~SmartPointer() { saveDel(); }

	operator T*() const { return pT; }
	T& operator*() const { return *pT; }
	T* operator->() const { return pT; }

	SmartPointer<T>& operator=(const SmartPointer<T>& smartP) {
		return operator=(smartP.pT);
	}

	SmartPointer<T>& operator=(T* _pT) {
		if(pT != _pT) {
			saveDel();
			pT = _pT;
			saveNew();
		}
		return (*this);
	}

    bool operator==(T*) const;
    bool operator!=(T*) const;
    bool operator==(const SmartPointer<T>&) const;
    bool operator!=(const SmartPointer<T>&) const;

    template<class T2> operator SmartPointer<T2>() { return SmartPointer<T2>(pT); }
};

template<class T>
inline bool SmartPointer<T>::operator==(T *_pT) const {
    return (pT == _pT);
}

template<class T>
inline bool SmartPointer<T>::operator!=(T *_pT) const {
    return !operator==(_pT);
}

template<class T>
inline bool SmartPointer<T>::operator==(const SmartPointer& _pT) const {
    return operator==(_pT.pT);
}

template<class T>
inline bool SmartPointer<T>::operator!= (const SmartPointer& _pT) const {
    return !operator==(_pT);
}

#endif
