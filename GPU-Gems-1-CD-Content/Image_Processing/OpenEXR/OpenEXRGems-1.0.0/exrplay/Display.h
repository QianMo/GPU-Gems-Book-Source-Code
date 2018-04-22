#ifndef _EXRPLAY_DISPLAY_H_
#define _EXRPLAY_DISPLAY_H_

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

#include <vector>
#include <string>
#include <inttypes.h>

#include <ImfRgbaFile.h>
#include <GlutGlut.h>
#include <GlutWindow.h>
#include <Cg/cgGL.h>

namespace ExrPlay
{

//-------------------------------------------------------------------------
//
//    This display window applies a gamma ramp and an exposure setting
//    to a sequence of OpenEXR images and plays them back as fast as it
//    can.
//
//-------------------------------------------------------------------------

class Display : public Glut::Window
{
  public:

    //-----------------------------------------------------------------
    // Constructor.
    //
    // frames is the list of OpenEXR frames to be played back.  dim is
    // the dimensions of a frame (all frames must have the same 
    // dimensions).
    //
    // The display window will use the Nvidia Pixel Data Range GL
    // extension if specified.
    //
    // Use the built-in shader, rather than loading one from a file
    // named "Display.cg", if specified.
    //
    // The window name is optional.
    //-----------------------------------------------------------------

    Display (const std::vector<Imf::Rgba *> & frames, 
	     const Imath::V2i & dim,
	     float rate,
	     bool useBuiltin = false,
	     bool usePdr = false,
	     const std::string & name = "Display");


    virtual ~Display ();

    virtual void         init ();

    virtual void         display ();
    virtual void         keyboard (unsigned char key, int x, int y);
    virtual void         idle ();

    virtual const char * theProgram () const;


  protected:

    //-------------------
    // Handle Cg errors.
    //-------------------

    static void cgErrorCallback ();


    //----------------------
    // Check for GL errors.
    //----------------------

    virtual void checkGlErrors (const char * where) const;


    const std::vector<Imf::Rgba *> &             _frames;
    int64_t                                      _period;
    bool                                         _nextFrame;
    bool                                         _rateLimit;
    bool                                         _pause;
    Imf::Rgba *                                  _pdrMem;
    bool                                         _usePdr;

    GLuint                                       _imageTexId;
    CGprogram                                    _cgprog;
    std::string                                  _cgProgramName;


    //---------------------
    // Display parameters.
    //---------------------

    bool                                         _useBuiltin;
    float                                        _gamma;
    float                                        _exposure;
    float                                        _scale;
    CGparameter                                  _cgGamma;
    CGparameter                                  _cgExposure;
    

  private:

    //------------------
    // Not implemented.
    //------------------

    Display ();
};

} // namespace ExrPlay

#endif // _EXRPLAY_DISPLAY_H_
