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

#include <DisplayLut.h>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <unistd.h>

#include <Iex.h>
#include <ImfIO.h>
#include <ImathMath.h>
#include <FrameRateCounter.h>

using namespace std;

namespace ExrPlay
{

//
// This display mode doesn't have a built-in shader.
//

DisplayLut::DisplayLut (const vector<Imf::Rgba *> & frames,
			const Imath::V2i & dim,
			float rate,
			const string & lutName,
			bool useBuiltin,
			bool usePdr,
			const string & name)
    : Display (frames, dim, rate, false, usePdr, name),
      _lutName (lutName),
      _applyLut (true)
{
    _cgProgramName = "DisplayLut.cg";
}


DisplayLut::~DisplayLut ()
{
}


namespace
{

void
checkErrorEof (ifstream & is)
{
    if (!Imf::checkError (is))
    {
	THROW (Iex::IoExc, "Incomplete lut file");
    }
}


void
createLut (GLuint tex, const vector<half> & lut)
{
    if (lut.size () != 256 * 256)
    {
	std::cerr << "Lut must have exactly " << 256 * 256 << "entries."
		  << std::endl;
	exit (1);
    }

    GLenum target = GL_TEXTURE_RECTANGLE_NV;

    glBindTexture (target, tex);

    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D (target, 0, GL_FLOAT_R16_NV, 256, 256, 0,
		  GL_LUMINANCE, GL_HALF_FLOAT_NV, &lut[0]);
}


void
fillLutFromFile (ifstream & is, vector<half> & lut, int & line,
		 const string & expectedName)
{
    //
    // Load a channel from the specified file into the lut.
    //
    // File format:
    //
    // The file is divided into 3 sections, one for each color channel
    // R, G, and B (this happens to be the order in which the sections
    // occur in the file, as well).
    //
    // Each section begins with the name of the channel ("red",
    // "green", and "blue" are the required names).
    //
    // The next line gives the number of entries in the lut for
    // the corresponding channel; call this number n.  What follows
    // are n lines, each containing a single FP value.  The FP value
    // on the first such line is the lookup value for half value 0.0.
    // The next line is the lookup value for half value
    // 0.00000005960464478 (the next-largest half value after 0.0),
    // the next line is the lookup value for half value
    // 0.00000011920928955 (the next-largest half value), etc.
    //
    // We don't typically deal with negative color values, so
    // our luts always begin at 0.0 and are monotonically increasing.
    //
    // There are exactly 31744 non-negative half values.  n must
    // be at least 1 and cannot be larger than 31744.  If a particular 
    // section does not contain a lookup value for every non-negative 
    // half value, the remaining lookup values are filled in from the 
    // last line in the section.  (A more robust scheme could
    // extrapolate the lookup table curve out to 1.0,
    // for example.)
    //

    static const half hmax  = HALF_MAX;
    static const half hzero = 0;
    string chanName;

    is >> chanName;
    checkErrorEof (is);
    if (chanName != expectedName)
    {
	THROW (Iex::IoExc, "Invalid channel name, expected \"" << expectedName
	       << "\", got \"" << chanName << "\"");
    }
    line++;

    int len;

    is >> len;
    checkErrorEof (is);
    line++;

    if (len < 1 || len > hmax.bits () + 1)
    {
	THROW (Iex::IoExc, "Invalid length " << len << " in lut file");
    }

    for (int i = hzero.bits (); i != hzero.bits () + len; ++i)
    {
	is >> lut[i];
	checkErrorEof (is);
	line++;
    }

    //
    // Fill any remaining lut entries.
    //

    for (int i = hzero.bits () + len; i != hmax.bits () + 1; ++i)
	lut[i] = lut[len - 1];
}


void
fillIdLut (vector<half> & lut)
{
    for (int i = 0; i != (1 << 16); ++i)
    {
	lut[i].setBits (i);
    }
}

} // namespace


void
DisplayLut::loadFilmlookLuts ()
{
    ifstream is (_lutName.c_str ());

    if (!is)
    {
	std::cerr << "Can't open " << _lutName << std::endl;
	exit (1);
    }

    //
    // We're storing half values in our luts, so we have to use
    // TEXTURE_RECTANGLE_NV textures, which are indexed 
    // [0,0]-[w,h].
    //
    // Small (less than 4k or so) luts could be encoded as a lut
    // of height 1, but larger luts will not fit in a single
    // dimension due to texture size limits.
    //
    // A 256x256 texture wastes some space, as our file-based
    // luts never contain negative values and may not even cover
    // all positive half values; but they're easy to index and
    // cover all possibilities.
    // 

    _lutR.resize (256 * 256, 0.0);
    _lutG.resize (256 * 256, 0.0);
    _lutB.resize (256 * 256, 0.0);

    //
    // Fill the lut from the file.
    //
    // Note that there is no lut applied to the alpha channel.
    //

    int line = 1;
    try
    {
	fillLutFromFile (is, _lutR, line, "red");
	fillLutFromFile (is, _lutG, line, "green");
	fillLutFromFile (is, _lutB, line, "blue");
    }
    catch (exception & e)
    {
        std::cerr << e.what () << " (line " << line << ")" << std::endl;
	exit (1);
    }
}


void
DisplayLut::loadIdLuts ()
{
    _idLutR.resize (256 * 256);
    _idLutG.resize (256 * 256);
    _idLutB.resize (256 * 256);

    fillIdLut (_idLutR);
    fillIdLut (_idLutG);
    fillIdLut (_idLutB);
}


void
DisplayLut::init ()
{
    Display::init ();

    loadFilmlookLuts ();

    glGenTextures (3, _lutTex);

    createLut (_lutTex[0], _lutR);
    createLut (_lutTex[1], _lutG);
    createLut (_lutTex[2], _lutB);

    loadIdLuts ();

    glGenTextures (3, _idLutTex);

    createLut (_idLutTex[0], _idLutR);
    createLut (_idLutTex[1], _idLutG);
    createLut (_idLutTex[2], _idLutB);
}


void
DisplayLut::keyboard (unsigned char key, int x, int y)
{
    //
    // Allow user to enable/disable LUT.
    //

    switch (key)
    {
      case 'l':
	  _applyLut = !_applyLut;
	  std::cout << "lut is " << (_applyLut ? "enabled" : "disabled")
		    << std::endl;
	  break;
      default:
	  return Display::keyboard (key, x, y);
    }

    glutPostRedisplay ();
}


void
DisplayLut::display ()
{
    //
    // Bind the display lut textures.
    //

    GLenum target = GL_TEXTURE_RECTANGLE_NV;
    GLuint * lut = _applyLut ? _lutTex : _idLutTex;

    glActiveTextureARB (GL_TEXTURE1_ARB);
    glBindTexture (target, lut[0]);
	
    glActiveTextureARB (GL_TEXTURE2_ARB);
    glBindTexture (target, lut[1]);
	
    glActiveTextureARB (GL_TEXTURE3_ARB);
    glBindTexture (target, lut[2]);
	
    checkGlErrors ("DisplayLut::display");
    
    Display::display ();
}

} // namespace ExrPlay
