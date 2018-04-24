#ifndef __COM_PTR_H
#define __COM_PTR_H

#include "Cfg.h"
// from ATL/MFC

template <class T>
class com_ptr
{
public:
	typedef T _PtrClass;

	typedef void (com_ptr::*unspecified_bool_type)();

	com_ptr() throw()
	{
		p = NULL;
	}
	com_ptr(int nNull) throw()
	{
		assert(nNull == 0);
		(void)nNull;
		p = NULL;
	}
	com_ptr(T* lp) throw()
	{
		p = lp;
		if (p != NULL)
			p->AddRef();
	}
	com_ptr( const com_ptr<T>& lp ) throw()
	{
		p = lp;
		if (p != NULL)
			p->AddRef();
	}
	T* operator=(T* lp) throw()
	{
		if( lp != NULL )
			lp->AddRef();
		if( p )
			p->Release();
		p = lp;

		return lp;
	}

	const com_ptr<T>& operator=(const com_ptr<T>& lp) throw()
	{
		if( lp.p != NULL )
			lp.p->AddRef();
		if( p )
			p->Release();

		p = lp.p;
		return *this;
	}

	~com_ptr() throw()
	{
		if (p)
			p->Release();
	}

	operator T*() const throw()
	{
		return p;
	}

	T& operator*() const throw()
	{
		assert(p!=NULL);
		return *p;
	}
	//The assert on operator& usually indicates a bug.  If this is really
	//what is needed, however, take the address of the p member explicitly.
	T** operator&() throw()
	{
		assert(p==NULL);
		return &p;
	}

	T* get() const throw() {
		return p;
	}

	T** assignGet() throw() {
		release();
		return &p;
	}

	T* operator->() const throw()
	{
		assert(p!=NULL);
		return p;
	}

	bool operator!() const throw()
	{
		return (p == NULL);
	}

	bool operator<(T* pT) const throw()
	{
		return p < pT;
	}

	bool operator==(T* pT) const throw()
	{
		return p == pT;
	}

	// Release the interface and set to NULL
	void release() throw()
	{
		T* pTemp = p;
		if (pTemp)
		{
			p = NULL;
			pTemp->Release();
		}
	}

	// Compare two objects for equivalence
	bool isEqualObject(IUnknown* pOther) throw()
	{
		if (p == NULL && pOther == NULL)
			return true;	// They are both NULL objects

		if (p == NULL || pOther == NULL)
			return false;	// One is NULL the other is not

		com_ptr<IUnknown> punk1;
		com_ptr<IUnknown> punk2;
		p->QueryInterface(__uuidof(IUnknown), (void**)&punk1);
		pOther->QueryInterface(__uuidof(IUnknown), (void**)&punk2);
		return punk1 == punk2;
	}
	// Attach to an existing interface (does not AddRef)
	void attach(T* p2) throw()
	{
		if (p)
			p->Release();
		p = p2;
	}
	// Detach the interface (does not Release)
	T* detach() throw()
	{
		T* pt = p;
		p = NULL;
		return pt;
	}
	HRESULT copyTo(T** ppT) throw()
	{
		assert(ppT != NULL);
		if (ppT == NULL)
			return E_POINTER;
		*ppT = p;
		if (p)
			p->AddRef();
		return S_OK;
	}
	HRESULT CoCreateInstance(REFCLSID rclsid, LPUNKNOWN pUnkOuter = NULL, DWORD dwClsContext = CLSCTX_ALL) throw()
	{
		assert(p == NULL);
		return ::CoCreateInstance(rclsid, pUnkOuter, dwClsContext, __uuidof(T), (void**)&p);
	}
	HRESULT CoCreateInstance(LPCOLESTR szProgID, LPUNKNOWN pUnkOuter = NULL, DWORD dwClsContext = CLSCTX_ALL) throw()
	{
		CLSID clsid;
		HRESULT hr = CLSIDFromProgID(szProgID, &clsid);
		assert(p == NULL);
		if (SUCCEEDED(hr))
			hr = ::CoCreateInstance(clsid, pUnkOuter, dwClsContext, __uuidof(T), (void**)&p);
		return hr;
	}
	template <class Q>
	HRESULT QueryInterface(Q** pp) const throw()
	{
		assert(pp != NULL);
		return p->QueryInterface(__uuidof(Q), (void**)pp);
	}
	T* p;
};

#endif
