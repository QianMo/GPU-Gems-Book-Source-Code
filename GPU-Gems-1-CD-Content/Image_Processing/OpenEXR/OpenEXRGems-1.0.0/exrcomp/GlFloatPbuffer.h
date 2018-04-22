#ifndef _GL_FLOAT_PBUFFER_H_
#define _GL_FLOAT_PBUFFER_H_

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

#include <GL/gl.h>
#include <GL/glx.h>

#include <ImathVec.h>

//-------------------------------------------------------------------
//
//    A C++ wrapper around OpenGL floating-point pbuffers.  It
//    creates a pbuffer with no stencil and no Z, so it's really
//    only useful for 2D rendering.  
//
//    Based on the simple_float_pbuffer OpenGL demo in NVIDIA's Cg
//    SDK.
//
//    Currently only works with glx.
//
//-------------------------------------------------------------------

class GlFloatPbuffer
{
  public:

    //--------------------------------------------------------------
    // Constructor.
    //
    // Creates a 32-bit floating point pbuffer with the specified
    // dimensions.
    //
    // Throws an Iex::BaseExc exception if the pbuffer can't be
    // created.  If this happens, the object can't be used, but
    // it can safely be deleted.
    //--------------------------------------------------------------

    GlFloatPbuffer (const Imath::V2i & dim);

    virtual ~GlFloatPbuffer ();


    //---------------------------
    // The pbuffer's dimensions.
    //---------------------------

    const Imath::V2i & dim () const throw ();


    //--------------------------------------------------------------
    // Make the pbuffer the current GL context and drawable.
    // Subsequent OpenGL calls will apply to this pbuffer.
    //--------------------------------------------------------------

    virtual void       activate ();


    //-----------------------------------------------------------
    // Restore the GL context and drawable that were in effect
    // prior to the last activate () call.
    //-----------------------------------------------------------

    virtual void       deactivate ();

  protected:

    const Imath::V2i _dim;

    Display *        _display;
    GLXPbuffer       _pbuffer;
    GLXContext       _context;

    bool             _validContext, _validPbuffer;

    Display *        _prevDisplay;
    GLXPbuffer       _prevDrawable;
    GLXContext       _prevContext;
};

//-----------------
// Inline methods.
//-----------------

inline const Imath::V2i &
GlFloatPbuffer::dim () const throw ()
{
    return _dim;
}

#endif // _GL_FLOAT_PBUFFER_H_
