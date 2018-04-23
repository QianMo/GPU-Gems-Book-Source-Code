/************************************************************
 *															*
 * decr     : abstract base class for all vectors			*
 * version  : 1.01											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 25.11.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#pragma once




#include "clClass.h"


class clVector : public clClass {
	public:
		clVector(void)	{}
		~clVector(void) {};

		virtual TCHAR* toString(int iOutputCount=0) = NULL;
		virtual TCHAR* toShortString(int iOutputCount=0) = NULL;

		virtual int getSize()=NULL;

		virtual void clear() = NULL;

};
