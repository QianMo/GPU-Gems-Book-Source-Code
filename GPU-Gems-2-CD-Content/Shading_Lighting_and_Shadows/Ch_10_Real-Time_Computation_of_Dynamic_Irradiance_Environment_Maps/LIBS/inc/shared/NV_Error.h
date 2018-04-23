/*--------------------------------------------------------------------- NVMH5 -|----------------------
Path:  Sdk\Libs\inc\shared\
File:  NV_Error.h

Copyright NVIDIA Corporation 2003
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS* AND NVIDIA AND
AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA
OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER
INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THIS
SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.


Comments:
No longer requires separate .cpp file.  Functions defined in the .h with the
magic of the static keyword.

For UNICODE, both TCHAR and char * flavors are defined, so that old code not
using TCHAR args to the functions (ie, not using the TEXT() macro) do not
need to change.

-------------------------------------------------------------------------------|--------------------*/

#pragma warning( disable : 4505 )		// unreferenced local function has been removed

#ifndef	H_NV_ERROR_H
#define	H_NV_ERROR_H

#pragma warning( disable : 4995 )		// old string functions marked as #pragma deprecated

#include    <stdlib.h>          // for exit()
#include    <stdio.h>
#include    <windows.h>
#include	<tchar.h>

#if _MSC_VER >= 1000
	#pragma once
#endif // _MSC_VER >= 1000

//---------------------------------------------------------------

#ifndef OUTPUT_POINTER
	#define OUTPUT_POINTER(p) { FMsg("%35s = %8.8x\n", #p, p ); }
#endif

#ifndef NULLCHECK
	#define NULLCHECK(q, msg,quit) {if(q==NULL) { DoError(msg, quit); }}
#endif

#ifndef IFNULLRET
	#define IFNULLRET(q, msg)	   {if(q==NULL) { FDebug(msg); return;}}
#endif

#ifndef FAILRET
	#define FAILRET(hres, msg) {if(FAILED(hres)){FDebug("*** %s   HRESULT: %d\n",msg, hres);return hres;}}
#endif

#ifndef HRESCHECK
	#define HRESCHECK(q, msg)	 {if(FAILED(q)) { FDebug(msg); return;}}
#endif

#ifndef NULLASSERT
	#define NULLASSERT( q, msg,quit )   {if(q==NULL) { FDebug(msg); assert(false); if(quit) exit(0); }}
#endif

/////////////////////////////////////////////////////////////////

static void OkMsgBox( TCHAR * szCaption, TCHAR * szFormat, ... )
{
	TCHAR buffer[256];
    va_list args;
    va_start( args, szFormat );
	_vsntprintf( buffer, 256, szFormat, args );
    va_end( args );
	buffer[256-1] = '\0';			// terminate in case of overflow
	MessageBox( NULL, buffer, szCaption, MB_OK );
}


#ifdef _DEBUG
	static void FDebug ( TCHAR * szFormat, ... )
	{
		// It does not work to call FMsg( szFormat ).  The variable arg list will be incorrect
		static TCHAR buffer[2048];
		va_list args;
		va_start( args, szFormat );
		_vsntprintf( buffer, 2048, szFormat, args );
		va_end( args );
		buffer[2048-1] = '\0';			// terminate in case of overflow
		OutputDebugString( buffer );
		Sleep( 2 );		// OutputDebugString sometimes misses lines if
						//  called too rapidly in succession.
	}
	#ifdef UNICODE
	static void FDebug( char * szFormat, ... )
	{
		static char buffer[2048];
		va_list args;
		va_start( args, szFormat );
		_vsnprintf( buffer, 2048, szFormat, args );
		va_end( args );
		buffer[2048-1] = '\0';			// terminate in case of overflow

		#ifdef UNICODE
			int nLen = MultiByteToWideChar( CP_ACP, 0, buffer, -1, NULL, NULL );
			LPWSTR lpszW = new WCHAR[ nLen ];
			if( lpszW != NULL )
			{
				MultiByteToWideChar( CP_ACP, 0, buffer, -1, lpszW, nLen );
				OutputDebugString( lpszW );
				delete lpszW;
				lpszW = NULL;
			}
		#else
			OutputDebugString( buffer );
		#endif
		Sleep( 2 );		// OutputDebugString sometimes misses lines if
						//  called too rapidly in succession.
	}
	#endif

	#pragma warning( disable : 4100 )	// unreferenced formal parameter
	inline static void NullFunc( TCHAR * szFormat, ... ) {}
	#ifdef UNICODE
		inline static void NullFunc( char * szFormat, ... ) {}
	#endif
	#pragma warning( default : 4100 )

	#if 0
		#define WMDIAG(str) { OutputDebugString(str); }
	#else
		#define WMDIAG(str) {}
	#endif
#else
	inline static void FDebug( TCHAR * szFormat, ... )		{ szFormat; }
	#ifdef UNICODE
		inline static void FDebug( char * szFormat, ... )		{ szFormat; }
	#endif
	inline static void NullFunc( char * szFormat, ... )		{ szFormat; }
	#define WMDIAG(str) {}
#endif

static void FMsg( TCHAR * szFormat, ... )
{	
	static TCHAR buffer[2048];
	va_list args;
	va_start( args, szFormat );
    _vsntprintf( buffer, 2048, szFormat, args );
	va_end( args );
	buffer[2048-1] = '\0';			// terminate in case of overflow
	OutputDebugString( buffer );
	Sleep( 2 );		// OutputDebugString sometimes misses lines if
					//  called too rapidly in succession.
}

static void FMsgW( WCHAR * wszFormat, ... )
{
	WCHAR wbuff[2048];
	va_list args;
	va_start( args, wszFormat );
    _vsnwprintf( wbuff, 2048, wszFormat, args );
    va_end( args );
	wbuff[2048-1] = '\0';				// terminate in case of overflow
	OutputDebugStringW( wbuff );
	Sleep( 2 );		// OutputDebugString sometimes misses lines if
					//  called too rapidly in succession.
}


#ifdef UNICODE
	// You must make sure that the variable arg list is also char * and NOT WCHAR *
	// This function allows FMsg("") in old non-UNICODE builds to work without requiring FMsg(TEXT(""))
	static void FMsg( char * szFormat, ... )
	{
		static char buffer[2048];
		va_list args;
		va_start( args, szFormat );
		_vsnprintf( buffer, 2048, szFormat, args );
		va_end( args );
		buffer[2048-1] = '\0';			// terminate in case of overflow
	#ifdef UNICODE
		int nLen = MultiByteToWideChar( CP_ACP, 0, buffer, -1, NULL, NULL );
		LPWSTR lpszW = new WCHAR[ nLen ];
		if( lpszW != NULL )
		{
			MultiByteToWideChar( CP_ACP, 0, buffer, -1, lpszW, nLen );
			OutputDebugString( lpszW );
			delete lpszW;
			lpszW = NULL;
		}
	#else
		OutputDebugString( buffer );
	#endif
		Sleep( 2 );		// OutputDebugString sometimes misses lines if
						//  called too rapidly in succession.
	}
#endif

#endif
