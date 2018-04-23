//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef InvalidIndexExceptionH
#define InvalidIndexExceptionH
//---------------------------------------------------------------------------
#include "BaseException.h"
#include "../../Base/StringTools.h"
//---------------------------------------------------------------------------
struct InvalidIndexException : public BaseException {
	InvalidIndexException(int id) { 	
		msg = "Invalid Index:";
		msg += Tools::toString(id);
	}
};

#endif
