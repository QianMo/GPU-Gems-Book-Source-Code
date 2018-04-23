//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.

#ifndef BaseH
#define BaseH
//#define __DMS_DEBUG_BASE__ //needs DebugBase.cpp
//---------------------------------------------------------------------------
#include "ShortTypes.h"
//---------------------------------------------------------------------------
class Base {
private:
	#ifdef __DMS_DEBUG_BASE__
		static uint objectCount;
	#endif
	uint cRefs;

/*protected:
	#ifdef __DMS_DEBUG_BASE__
		virtual ~Base();
	#endif
	#ifndef __DMS_DEBUG_BASE__
		virtual ~Base() { }
	#endif*/
public:
	#ifdef __DMS_DEBUG_BASE__
		Base();
		virtual ~Base();
		void newRef();
		void delRef();
		static cuint getObjectCount() { return objectCount; }
	#endif

	#ifndef __DMS_DEBUG_BASE__
		Base() : cRefs(0) { }
		virtual ~Base() { }
		void newRef() { ++cRefs; }
		void delRef() { 
			if (--cRefs == 0) {	
				delete this; 
			} 
		}
	#endif

	cuint getRefCount() const { return cRefs; }
};

#endif
