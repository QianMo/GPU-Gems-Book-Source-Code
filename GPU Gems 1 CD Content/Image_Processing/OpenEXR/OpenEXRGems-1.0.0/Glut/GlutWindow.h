#ifndef _GLUT_WINDOW_H_
#define _GLUT_WINDOW_H_

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

#include <string>

#include <ImathVec.h>

namespace Glut
{

//------------------------------------------------------------------
//
//    A C++ wrapper around a glut window.
//
//    This is a virtual class.  Glut clients can build their own
//    display windows on this class.
//
//------------------------------------------------------------------

class Window
{
  public:

    //------------------------------------------------------------
    // Constructor.
    //
    // dim is the window dimensions, name is the window name.
    //------------------------------------------------------------

    Window (const Imath::V2i & dim, const std::string & name);

    virtual ~Window ();


    //------------------------------------------------------------------
    // The name of the window, used by Glut to give the window a title.
    //------------------------------------------------------------------

    const char * name () const throw ();


    //-------------------------
    // The window dimensions.
    //-------------------------

    const Imath::V2i & dim () const throw ();


    //--------------------------------------------------------------
    // Window-specific initialization.  Glut::addWindow calls this
    // method after it has created a window and an OpenGL context
    // for this Window object.
    //--------------------------------------------------------------

    virtual void init () = 0;


    //------------------------------------------------------------------
    // GLUT display, keyboard, and idle functions.  These are
    // called by Glut when this Window object is the current GLUT
    // window.
    //------------------------------------------------------------------

    virtual void display () = 0;
    virtual void keyboard (unsigned char key, int x, int y) = 0;
    virtual void idle () = 0;


  protected:

    const Imath::V2i  _dim;
    const std::string _name;


  private:

    //------------------
    // Not implemented.
    //------------------

    Window ();
};

//-----------------
// Inline methods.
//-----------------

inline
Window::Window (const Imath::V2i & dim, const std::string & name)
    : _dim (dim),
      _name (name)
{
}

inline
Window::~Window ()
{
}

inline const char *
Window::name () const throw ()
{
    return _name.c_str ();
}

inline const Imath::V2i &
Window::dim () const throw ()
{
    return _dim;
}

} // namespace Glut

#endif // _GLUT_WINDOW_H_
