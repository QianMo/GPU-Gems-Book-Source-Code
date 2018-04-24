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

// includes, file
#include <error_checker.h>

// includes, system
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#  include <stdlib.h>
#  undef min
#  undef max
#endif

// includes, system
#include <iostream>
#include <sstream>
#include <cassert>

// includes, graphics
#include <GL/gl.h>
#include <GL/glu.h>

//includes, project
#include <exception.h>

////////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL errors (including GLSLang).
//! @return true if no error was pending, else fals
////////////////////////////////////////////////////////////////////////////////
/* static */ void
ErrorChecker::checkErrorGL(const char* file, const int line) 
{
    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) 
    {
        std::ostringstream s( std::string("GL Error:\n"));
        s << "GL Error in file '" << file << "' in line " << line << " :\n";
        s << gluErrorString( gl_error);

        RUNTIME_EXCEPTION(  s.str());
    }
}


