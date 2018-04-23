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
/////////////////////////////////////////////////////////////////////////////


#ifndef DASSERT_H
#define DASSERT_H


#ifdef ASSERT
#  error "ASSERT macro defined twice"
#  undef ASSERT
#endif
#ifdef DASSERT
#  error "DASSERT macro defined twice"
#  undef DASSERT
#endif


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

// ABORT is a platform-independent way to quit the application.  
//
// Under Linux, we write the abort message to stderr and call abort().
// Under Windows, we bring up the debugger in debug mode and write the
// error message to the debugger's output window.  In release mode, we
// display the error message in a dialog box with an OK button, this
// will cause scripts to block.

#ifdef WINNT
#  ifdef DEBUG
#    define ABORT(msg) OutputDebugString (msg), DebugBreak(), abort()
#  else
#    define ABORT(msg) FatalAppExit(0, msg), abort()
#  endif
#else
#  define ABORT(msg) fprintf (stderr, "%s", msg), abort()
#endif




// ASSERT is a macro to test assertions.  It does so unconditionally,
// and if the condition is not met, it will print a scary error message
// and terminate.

#define ASSERT(x)                                                        \
    if (!(x)) {                                                          \
        char buf[4096];                                                  \
        snprintf (buf, 4096, "Error: Assertion failed, \"%s\", line %d\n"\
                 "\tProbable bug in software.  Alert tech support.\n",   \
                 __FILE__, __LINE__);                                    \
        ABORT (buf);                                                     \
    }                                                                    \
    else   // This 'else' exists to catch the user's following semicolon


// DASSERT is like assert, except that it only happens in debug mode.
// DASSERTs disappear in production compiles.

#ifdef DEBUG
#  define DASSERT(x) ASSERT(x)
#else
#  define DASSERT(x)
#endif



#endif /* !defined(DASSERT_H) */
