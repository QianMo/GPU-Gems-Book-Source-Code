/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)

#ifndef UNISTD_H
#define UNISTD_H


// Compatibility Layer

#ifdef WINNT

#ifndef NOMINMAX
#define NOMINMAX
#endif

#define Polygon dxPolygon
#include <windows.h>
#undef Polygon
#include <io.h>



#define strerror_r(a, b, c)    strerror(a)

#define vsnprintf _vsnprintf
#define snprintf _snprintf

#define usleep(a) Sleep((a)/1000)

#include <process.h>
#define execvp(cmd,argv) _spawnvp(_P_NOWAIT,cmd,argv)

// define a user socket event message
#define WM_SOCKET_NOTIFY (WM_USER + 1)

// Standard strtok is not thread-safe.  POSIX.1c defines reentrant 
// version strtok_r.  Windows lacks this, but it seems that they somehow
// made ordinary strtok reentrant.
// this flavor of macro copied from pthreads library so they play well together
#if !defined(__MINGW32__)
#define strtok_r( _s, _sep, _lasts ) \
	( *(_lasts) = strtok( (_s), (_sep) ) )
#endif /* !__MINGW32__ */

#if !defined(localtime_r)
#define localtime_r( _clock, _result ) \
	( *(_result) = *localtime( (_clock) ), \
	  (_result) )
#endif /* !defined */

#define	R_OK	4

#endif /* WINNT */

#endif /* UNISTD_H */
