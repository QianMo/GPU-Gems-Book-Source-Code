/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/

/* CUda UTility Library */

#ifndef _ERRORCHECKER_H_
#define _ERRORCHECKER_H_

// includes, system
#include <string>
#include <sstream>

// includes, project
#include <exception.h>

// typedefs
typedef unsigned int GLuint;

//! Class providing the handler / tester functions for errors as static members
class ErrorChecker 
{
public:

    //! Check for OpenGL errors (including GLSLang).
    static void checkErrorGL(const char* file = "-" , const int line = -1);

    //! Check if a condition is true.
    //! @note In prinicple has the same functionality as assert but allows 
    //!       much better control this version prints an error and terminates
    //!       the program, no exception is thrown.
    inline static void condition( bool val, const char* file, const int line);
};

// functions, inlined

// includes, system
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
//! Check if a condition is true.
//! @note In prinicple has the same functionality as assert but allows much
//!       better control this version prints an error and terminates the 
//!       program, no exception is thrown.
////////////////////////////////////////////////////////////////////////////////
/* static */ inline void
ErrorChecker::condition( bool val, const char* file, const int line) 
{
    if ( ! val) 
    {
        std::ostringstream os;
        os << "Condition failed: " << file << " in line " << line;
        RUNTIME_EXCEPTION( os.str() );
    }
}

#endif // _ERRORCHECKER_H_

