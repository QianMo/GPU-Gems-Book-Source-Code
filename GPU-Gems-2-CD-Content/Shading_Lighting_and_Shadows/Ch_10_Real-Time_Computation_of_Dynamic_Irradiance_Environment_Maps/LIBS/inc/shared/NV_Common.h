/*--------------------------------------------------------------------- NVMH5 -|----------------------
Path:  Sdk\Libs\inc\shared\
File:  NV_Common.h

Copyright NVIDIA Corporation 2003
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS* AND NVIDIA AND
AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA
OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER
INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THIS
SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.


Comments:
Useful macros, etc.

Define these to output messages on delete and release:
#define NVMSG_SAFE_ARRAY_DELETE
#define NVMSG_SAFE_DELETE
#define NVMSG_SAFE_RELEASE

FDebug and FMsg are defined in NV_Error.h.  These functions take a variable argument list, similar to
that for fprintf(), and output the string via OutputDebugString.  FDebug() outputs only for DEBUG builds.
FMsg() outputs text for bot DEBUG and RELEASE builds.


-------------------------------------------------------------------------------|--------------------*/

#pragma once

#ifndef H_NVCOMMONHEADER_H
#define H_NVCOMMONHEADER_H

#define INLINE __forceinline

#ifndef MULTI_THREAD
#include <windows.h>
#endif

#include <assert.h> 
#include "shared\NV_Error.h"


typedef unsigned short		USHORT;
typedef unsigned short		ushort;


//----------------------------------------------------------------
// Preferred macros

#ifndef MSG_IF
#define MSG_IF( expr, msg )								\
{														\
	if( expr ) { FMsg(msg);  }							\
}
#endif

#ifndef MSG_AND_RET_IF
#define MSG_AND_RET_IF( expr, msg )						\
{														\
	if( expr ) { FMsg(msg); return; }				\
}
#endif

#ifndef MSG_AND_RET_VAL_IF
#define MSG_AND_RET_VAL_IF( expr, msg, retval )					\
{																\
	if( expr )	{ FMsg(msg); return( retval );	}		\
}
#endif


// Causes a debug assertion if expr is true
#ifndef MSG_AND_BREAK_IF
#define MSG_AND_BREAK_IF( expr, msg )						\
{															\
	if( expr )	{ FMsg(msg); assert(false); }			\
}
#endif

#ifndef MSG_BREAK_AND_RET_IF
#define MSG_BREAK_AND_RET_IF( expr, msg )						\
{																\
	if( expr )	{ FMsg(msg); assert(false); return;	}	\
}
#endif

#ifndef MSG_BREAK_AND_RET_VAL_IF
#define MSG_BREAK_AND_RET_VAL_IF( expr, msg, retval )					\
{																		\
	if( expr )	{ FMsg(msg); assert(false); return(retval);	}	\
}
#endif

// Causes a debug assertion if expr is true
#ifndef BREAK_IF
#define BREAK_IF( expr )								\
{														\
	if( expr )	{ assert(false);	}					\
}
#endif

#ifndef BREAK_AND_RET_IF
#define BREAK_AND_RET_IF( expr )						\
{														\
	if( expr )	{ assert(false); return;	}			\
}
#endif

#ifndef BREAK_AND_RET_VAL_IF
#define BREAK_AND_RET_VAL_IF( expr, retval )			\
{														\
	if( expr )	{ assert( false ); return( retval ); }	\
}
#endif

#ifndef RET_IF
#define RET_IF( expr )									\
{														\
	if( expr )	{ return; }								\
}
#endif

#ifndef RET_VAL_IF
#define RET_VAL_IF( expr, retval )						\
{														\
	if( expr )	{ return( retval );	}					\
}
#endif


// Macro for checking if a variable is NULL
// This outputs the variable name and returns E_FAIL if the variable == NULL.
// Example:
//   pMyPointer = NULL;
//   MSGVARNAME_AND_FAIL_IF_NULL( pMyPointer );
// outputs:
//   "pMyPointer == NULL" and returns E_FAIL
#ifndef MSGVARNAME_AND_FAIL_IF_NULL
#define MSGVARNAME_AND_FAIL_IF_NULL( var )												\
{																						\
	if( (var) == NULL )  { FMsg( TEXT(#var) TEXT(" == NULL\n")); return( E_FAIL ); }	\
}
#endif

//--------------------------------------------------------------
// Below are macros that can be expressed with the above macros
// These are included for convenience.

#ifndef FAIL_IF_NULL
#define FAIL_IF_NULL( x )								\
{														\
	if( (x) == NULL )	{ return( E_FAIL ); }			\
}
#endif

#ifndef MSG_AND_FAIL_IF_NULL
#define MSG_AND_FAIL_IF_NULL( p, msg )							\
{																\
	if( (p) == NULL )  { FMsg(msg); return( E_FAIL ); }	\
}
#endif

#ifndef RET_IF_NULL
#define RET_IF_NULL( p )						\
{												\
	if( (p) == NULL ) { return; }				\
}
#endif

#ifndef RET_VAL_IF_NULL
#define RET_VAL_IF_NULL( p, retval )					\
{														\
	if( (p) == NULL )	{ return( retval );	}			\
}
#endif

#ifndef RET_NULL_IF_NULL
#define RET_NULL_IF_NULL(p)								\
{														\
	if( (p) == NULL )	{ return( NULL ); }				\
}
#endif

#ifndef BREAK_AND_RET_VAL_IF_FAILED
#define BREAK_AND_RET_VAL_IF_FAILED( hr )						\
{																\
	if( FAILED(hr) )	{ assert(false); return( hr ); }		\
}
#endif

#ifndef RET_VAL_IF_FAILED
#define RET_VAL_IF_FAILED( hr )							\
{														\
	if( FAILED( hr ) )	{ return( hr );	}				\
}
#endif


//-------------------------------------------------------------------------------------------

// delete a pointer allocated with new <type>[num], and set the pointer to NULL.
#ifndef SAFE_DELETE_ARRAY
	#ifdef	NVMSG_SAFE_ARRAY_DELETE
	#define SAFE_DELETE_ARRAY(p)											\
	{																		\
		FMsg(TEXT("SAFE_DELETE_ARRAY: %35s = 0x%8.8X\n"), TEXT(#p), p );	\
		if( p != NULL )														\
		{																	\
			delete [] (p);													\
			p = NULL;														\
		}																	\
	}
	#else
	#define SAFE_DELETE_ARRAY(p)										\
	{																	\
		if( p != NULL )													\
		{																\
			delete [] (p);												\
			p = NULL;													\
		}																\
	}
	#endif
#endif


#ifndef SAFE_ARRAY_DELETE
	#ifdef	NVMSG_SAFE_ARRAY_DELETE
	#define SAFE_ARRAY_DELETE(p)										\
	{																	\
		FMsg(TEXT("SAFE_ARRAY_DELETE: %35s = 0x%8.8X\n"), TEXT(#p), p );\
		if( p != NULL )													\
		{																\
			delete [] (p);												\
			p = NULL;													\
		}																\
	}
	#else
	#define SAFE_ARRAY_DELETE(p)								\
	{															\
		if( p != NULL )											\
		{														\
			delete [] (p);										\
			p = NULL;											\
		}														\
	}
	#endif
#endif


// deletes all pointers in a vector of pointers, and clears the vector
#ifndef SAFE_VECTOR_DELETE
#define SAFE_VECTOR_DELETE( v )									\
{																\
	for( UINT svecdel = 0; svecdel < (v).size(); svecdel++ )	\
	{	if( (v).at( svecdel ) != NULL )							\
		{	delete( (v).at( svecdel ));							\
			(v).at( svecdel ) = NULL;							\
		}														\
	}															\
	(v).clear();												\
}
#endif


#ifndef SAFE_DELETE
	#ifdef	NVMSG_SAFE_DELETE
	#define SAFE_DELETE( p )											\
	{																	\
		FMsg(TEXT("SAFE_DELETE: %35s = 0x%8.8X\n"), TEXT(#p), p );		\
		if( (p) != NULL)												\
		{																\
			delete(p);													\
			(p) = NULL;													\
		}																\
	}
#else
	#define SAFE_DELETE( p )									\
	{															\
		if( (p) != NULL )										\
		{														\
			delete(p);											\
			(p) = NULL;											\
		}														\
	}
#endif
#endif


// Releases all pointers in a vector of pointers, and clears the vector
#ifndef SAFE_VECTOR_RELEASE
#define SAFE_VECTOR_RELEASE( v )								\
{																\
	for( UINT svecrel = 0; svecrel < (v).size(); svecrel++ )	\
	{	if( (v).at( svecrel ) != NULL )							\
		{	(v).at( svecrel )->Release();						\
			(v).at( svecrel ) = NULL;							\
		}														\
	}															\
	(v).clear();												\
}
#endif

// Calls the Release() member of the object pointed to, and sets the pointer to NULL
#ifndef SAFE_RELEASE
	#ifdef	NVMSG_SAFE_RELEASE
	#define SAFE_RELEASE(p)												\
	{																	\
		FMsg(TEXT("SAFE_RELEASE: %35s = 0x%8.8X\n"), TEXT(#p), p );			\
		if( (p) != NULL )										\
		{														\
			(p)->Release();										\
			(p) = NULL;											\
		}														\
	}
#else
	#define SAFE_RELEASE(p)										\
	{															\
		if( (p) != NULL )										\
		{														\
			(p)->Release();										\
			(p) = NULL;											\
		}														\
	}
#endif
#endif


#ifndef SAFE_ADDREF
	#ifdef	NVMSG_SAFE_ADDREF
	#define SAFE_ADDREF(p)										\
	{															\
		FMsg(TEXT("SAFE_ADDREF: %35s = 0x%8.8X\n"), TEXT(#p), p );			\
		if( (p) != NULL )										\
		{														\
			(p)->AddRef();										\
		}														\
	}
#else
	#define SAFE_ADDREF(p)										\
	{															\
		if( (p) != NULL )										\
		{														\
			(p)->AddRef();										\
		}														\
	}
#endif
#endif


///////////////////////////////////////////////////////////////

#ifndef CHECK_BOUNDS
#define CHECK_BOUNDS( v, n, x )					\
	if( (v < n) || (v > x) )					\
	{	FDebug("Variable out of bounds!\n");	\
		assert( false ); return;				\
	}
#endif


#ifndef CHECK_BOUNDS_NULL
#define CHECK_BOUNDS_NULL( v, n, x )			\
	if( (v < n) || (v > x) )					\
	{	FDebug("Variable out of bounds!\n");	\
		assert( false ); return(NULL);			\
	}
#endif

#ifndef CHECK_BOUNDS_HR
#define CHECK_BOUNDS_HR( v, n, x )				\
	if( (v < n) || (v > x) )					\
	{	FDebug("Variable out of bounds!\n");	\
		assert( false ); return(E_FAIL);		\
	}
#endif

///////////////////////////////////////////////////////////////

#define ifnot(x)  if (!(x))
#define until(x) while(!(x))
#define ever          (;;)
#define wait        do {}
#define nothing     {}

///////////////////////////////////////////////////////////////

// Macro to make sure that a handle is allocated.
// If in_handle is NULL, an object is created at ptr.
//		handle will point to ptr; 
// If in_handle is not NULL, but it's pointer is NULL, then
//		an object is created at (*in_handle)
//		ptr is set to same value as (*in_handle)
//		handle will point to *in_handle
// class - the type of object to create
// init  - text of an Initialize() function called if object is created
// 

#ifndef GUARANTEE_ALLOCATED
#define GUARANTEE_ALLOCATED( in_handle, handle, ptr, classname, init )	\
{													\
	bool alloc;										\
	alloc = false;									\
	bool setptr;									\
	setptr = false;									\
	HRESULT hr;										\
	handle = in_handle;								\
	if( handle == NULL )							\
	{												\
		handle = & ptr;								\
		alloc = true;								\
	}												\
	else if( *handle == NULL )						\
	{												\
		alloc = true;								\
		setptr = true;								\
	}												\
	if( alloc == true )								\
	{												\
		(*handle) = new classname;					\
		FAIL_IF_NULL( (*handle) );					\
		hr = (*handle)->init;						\
		BREAK_AND_RET_VAL_IF_FAILED( hr );			\
	}												\
	if( setptr == true )							\
		ptr = *handle;								\
}
#endif

#ifndef FREE_GUARANTEED_ALLOC
#define FREE_GUARANTEED_ALLOC( handle, ptr )	\
{												\
	if( handle != NULL )						\
	{											\
		if( ptr == *handle )					\
		{										\
			SAFE_DELETE( (*handle) );			\
			ptr = NULL;							\
		}										\
		handle = NULL;							\
	}											\
}
#endif


//----------------------------------------------------------------
// Below are older deprecated macros.  Please use the macros above instead of these.


#ifndef	ASSERT_AND_RET_IF_FAILED
#define ASSERT_AND_RET_IF_FAILED(hres)	\
{										\
	if( FAILED(hres) )					\
	{									\
		assert( false );				\
		return( hres );					\
	}									\
}
#endif

//----------------------------------------------------------------

#endif				// H_NVCOMMONHEADER_H

