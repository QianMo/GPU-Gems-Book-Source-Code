#ifndef _GLUT_GLUT_H_
#define _GLUT_GLUT_H_

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2003, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
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
///////////////////////////////////////////////////////////////////////////

#include <map>
#include <GL/glut.h>

namespace Glut
{

//-------------------------------------------------------------
//
//    A simple C++ wrapper around glut for 2D applications.
//
//    All windows run in double-buffered RGBA mode with no Z.
//
//-------------------------------------------------------------

class Window;

class Glut
{
  public:

    //-----------------------------------------------------------------
    // Use this method to obtain a handle for the global glut context.
    // It always returns the same Glut instance.
    //-----------------------------------------------------------------

    static Glut & theGlutContext ();


    //---------------------------------------------------------------
    // Add a window.  Each window has its own keyboard handler,
    // idle handler and display handler.
    //---------------------------------------------------------------

    void addWindow (Window * win);


    //--------------------------------------------------------------------
    // Call the main glut loop and start drawing windows.  Never returns.
    //--------------------------------------------------------------------

    void run () const;


  private:

    //-----------------------------------------------------------------
    // These global callback handlers identify the current window and
    // route the callback request to the proper window.
    //-----------------------------------------------------------------

    static void kbHandler (unsigned char key, int x, int y);
    static void idleHandler ();
    static void displayHandler ();

    Glut ();
    ~Glut ();
    
    std::map<int, Window *> _win;
};

} // namespace Glut

#endif // _GLUT_GLUT_H_
