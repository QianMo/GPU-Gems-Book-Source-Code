#ifndef _EXRPLAY_DISPLAY_LUT_H_
#define _EXRPLAY_DISPLAY_LUT_H_

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

#include <half.h>
#include <Display.h>

namespace ExrPlay
{

//------------------------------------------------------------------
//
//    This display window applies an exposure, a LUT, and a gamma
//    ramp to a sequence of OpenEXR images and plays them back as
//    fast as it can.
//
//------------------------------------------------------------------

class DisplayLut : public Display
{
  public:

    //-----------------------------------------------------------------
    // Constructor.
    //
    // frames is the list of OpenEXR frames to be played back.  dim is
    // the dimensions of a frame (all frames must have the same 
    // dimensions).  lutName is the name of a file containing a LUT.
    //
    // The window name is optional.
    //-----------------------------------------------------------------

    DisplayLut (const std::vector<Imf::Rgba *> & frames, 
		const Imath::V2i & dim,
		float rate,
		const std::string & lutName,
		bool useBuiltin = false,
		bool usePdr = false,
		const std::string & name = "Display");


    virtual ~DisplayLut ();

    virtual void init ();
    virtual void display ();
    virtual void keyboard (unsigned char key, int x, int y);


  protected:

    //-------------------------------------------------------------
    // The filmlook luts are loaded from a file and are mapped to
    // GL textures, one for each color channel (don't apply 
    // filmlook to alpha).
    //-------------------------------------------------------------

    const std::string _lutName;
    std::vector<half> _lutR, _lutG, _lutB;
    GLuint            _lutTex[3];

    
    //------------------------------------------------------------------
    // The identity luts output a value equal to their input value.
    // By choosing between the filmlook luts and the identity luts
    // at display time, we allow the user to toggle filmlook on the
    // fly.
    //------------------------------------------------------------------

    std::vector<half> _idLutR, _idLutG, _idLutB;
    GLuint            _idLutTex[3];

    //---------------------
    // Display parameters.
    //---------------------

    bool              _applyLut;


  private:

    //------------------
    // Not implemented.
    //------------------

    DisplayLut ();

    void loadFilmlookLuts ();
    void loadIdLuts ();
};

} // namespace ExrPlay

#endif // _EXRPLAY_DISPLAY_LUT_H_
