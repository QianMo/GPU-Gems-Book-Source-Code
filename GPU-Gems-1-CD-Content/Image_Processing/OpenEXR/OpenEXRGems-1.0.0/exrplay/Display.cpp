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

#include <Display.h>

#include <iostream>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <GL/glx.h>

#include <Iex.h>
#include <ImathMath.h>
#include <FrameRateCounter.h>

using namespace std;

namespace ExrPlay
{

namespace
{

int64_t
now ()
{
    timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

} // namespace


Display::Display (const std::vector<Imf::Rgba *> & frames,
		  const Imath::V2i & dim,
		  float rate,
		  bool useBuiltin,
		  bool usePdr,
		  const std::string & name)
    : Glut::Window (dim, name),
      _useBuiltin (useBuiltin),
      _imageTexId (0),
      _rateLimit (true),
      _period ((int64_t) (1.0 / rate * 1000000)),
      _nextFrame (true),
      _pause (false),
      _usePdr (usePdr),
      _pdrMem (0),
      _frames (frames),
      _gamma (2.2),
      _exposure (0),
      _scale (1.0),
      _cgProgramName ("Display.cg")
{
}


Display::~Display ()
{
    if (_pdrMem)
    {
	glFlushPixelDataRangeNV (GL_WRITE_PIXEL_DATA_RANGE_NV);
	glXFreeMemoryNV (_pdrMem);
    }
}


// static
void
Display::cgErrorCallback ()
{
    //
    // Apparently it's not safe (in GNU/Linux, anyway) to throw from
    // the glut main loop, so we have to resort to printing out an
    // error message and exiting, yuck.
    //

    std::cerr << "Cg error: " << cgGetErrorString (cgGetError ()) << std::endl;
    exit (1);
}


void
Display::checkGlErrors (const char * where) const
{
    GLenum error = glGetError ();
    if (error != GL_NO_ERROR)
    {
	std::cerr << "GL error in " << where << ": "
		  << gluErrorString (error) << std::endl;
	exit (1);
    }
}


const char *
Display::theProgram () const
{
    static const char * prog =
	"!!FP1.0\n"
	"DECLARE expMult;\n"
	"TEX H0, f[TEX0], TEX0, RECT;\n"
	"MUL H0.xyz, H0, expMult.x;\n"
	"POWH o[COLH].x, H0.x, 0.45454545;\n"
	"POWH o[COLH].y, H0.y, 0.45454545;\n"
	"POWH o[COLH].z, H0.z, 0.45454545;\n"
	"END";

    return prog;
}


void
Display::init ()
{
    //
    // Set up default ortho view.
    //

    glLoadIdentity ();
    glViewport (0, 0, _dim.x, _dim.y);
    glOrtho (0, _dim.x, _dim.y, 0, -1, 1);

    if (_usePdr)
    {
	size_t pdrSize = _dim.x * _dim.y * sizeof (Imf::Rgba);
	_pdrMem = (Imf::Rgba *) glXAllocateMemoryNV (pdrSize, 0.0, 1.0, 1.0);

	if (!_pdrMem)
	{
	    std::cerr << "Can't allocate AGP memory for texture." << std::endl;
	    std::cerr << "Falling back to regular memory." << std::endl;
	    _usePdr = false;
	}
	else
	{
	    glEnableClientState (GL_WRITE_PIXEL_DATA_RANGE_NV);
	    glPixelDataRangeNV (GL_WRITE_PIXEL_DATA_RANGE_NV,
				pdrSize, 
				_pdrMem);
	}
    }

    if (!cgGLIsProfileSupported (CG_PROFILE_FP30))
    {
	std::cerr << "This display method requires CG_PROFILE_FP30 support, "
		  << "but it's not available.";
	exit (1);
    }

    cgSetErrorCallback (cgErrorCallback);
    
    CGcontext cgcontext = cgCreateContext ();

    if (_useBuiltin)
    {
	glBindProgramNV (GL_FRAGMENT_PROGRAM_NV, 1);

	glLoadProgramNV (GL_FRAGMENT_PROGRAM_NV, 
			 1, 
			 strlen (theProgram ()),
			 (const GLubyte *) theProgram ());
	glEnable (GL_FRAGMENT_PROGRAM_NV);
    }
    else
    {
	//
	// Compile a Cg program from a file.
	//
    
	std::string cgpath;
	char * userpath = getenv ("EXRPLAY_CG_PATH");
	if (userpath)
	    cgpath = std::string (userpath) + "/" + _cgProgramName;
	else
	    cgpath = std::string (PKG_DATA_DIR) + "/" + _cgProgramName;
	    
	_cgprog = cgCreateProgramFromFile (cgcontext, 
					   CG_SOURCE,
					   cgpath.c_str (),
					   CG_PROFILE_FP30, 
					   0, 0);

	cgGLLoadProgram (_cgprog);
	cgGLBindProgram (_cgprog);
	cgGLEnableProfile (CG_PROFILE_FP30);
	
	//
	// Create the display parameters.
	//

	_cgGamma = cgGetNamedParameter (_cgprog, "gamma");
	_cgExposure = cgGetNamedParameter (_cgprog, "expMult");
    }

    //
    // Create image texture.
    //

    GLenum target = GL_TEXTURE_RECTANGLE_NV;
    
    glGenTextures (1, &_imageTexId);
    glBindTexture (target, _imageTexId);

    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D (target, 0, GL_FLOAT_RGBA16_NV, _dim.x, _dim.y, 0, 
		  GL_RGBA, GL_HALF_FLOAT_NV, 0);

    checkGlErrors ("Display::init");
}


void
Display::keyboard (unsigned char key, int x, int y)
{
    switch (key)
    {
      case 27:
      case 'q':
	  exit (0);
      case 'e':
	  _exposure -= 0.5;
	  std::cout << "exposure is now " << _exposure << std::endl;
	  break;
      case 'E':
	  _exposure += 0.5;
	  std::cout << "exposure is now " << _exposure << std::endl;
	  break;
      case 'g':
	  _gamma -= 0.1;
	  std::cout << "gamma is now " << _gamma << std::endl;
	  break;
      case 'G':
	  _gamma += 0.1;
	  std::cout << "gamma is now " << _gamma << std::endl;
	  break;
      case ' ':
	  _pause = !_pause;
	  break;
    }

    if (!_pause)
	_nextFrame = true;
    glutPostRedisplay ();
}


void
Display::idle ()
{
    if (!_pause)
    {
	_nextFrame = true;
	glutPostRedisplay ();
    }
}


void
Display::display ()
{
    static int64_t drawtime = now ();
    static int64_t drawlatency = now ();
    static unsigned int frame = 0;
    static uint64_t totalFrames = 0;
    static FrameRateCounter fps;

    GLenum target = GL_TEXTURE_RECTANGLE_NV;

    glActiveTextureARB (GL_TEXTURE0_ARB);
    glBindTexture (target, _imageTexId);
    
    if (_nextFrame)
    {
	if (_rateLimit)
	{
	    drawtime += _period;
	    int64_t snooze = drawtime - now () - drawlatency;
	    if (snooze > 0)
	    {
		timespec req, rem;
		req.tv_sec = snooze / 1000000;
		req.tv_nsec = (snooze * 1000) % 1000000000;
		for (; nanosleep (&req, &rem) == EINTR;)
		    req = rem;
	    }
	    else
		drawtime = now ();    // catch up
	    drawlatency = now ();
	}

	frame = totalFrames++ % _frames.size ();

	//
	// Update the texture with the next frame.
	//
	// glTexSubImage2D is faster than glTexImage2D for reloading
	// texture data.
	//
	
	Imf::Rgba * texp;
	if (_usePdr)
	{
	    memcpy (_pdrMem,
		    _frames[frame],
		    _dim.x * _dim.y * sizeof (Imf::Rgba));
	    texp = _pdrMem;
	}
	else
	    texp = _frames[frame % _frames.size ()];
	
	glTexSubImage2D (target, 0, 0, 0, _dim.x, _dim.y, GL_RGBA, 
			 GL_HALF_FLOAT_NV, texp);
    }

    //
    // Set up display parameters.  
    //
    // The gamma used by the fragment shader is the inverse of the 
    // monitor gamma.
    //

    // XXX HACK

    if (!_useBuiltin)
    {
	cgGLSetParameter1f (_cgGamma, 1/_gamma);
	
	cgGLSetParameter1f (_cgExposure, Imath::Math<float>::pow (2, _exposure) * _scale);
    }

    //
    // Draw a textured quad.
    //

    glBegin (GL_QUADS);
    glTexCoord2f (0.0,    0.0);    glVertex2f(0.0,    0.0);
    glTexCoord2f (_dim.x, 0.0);    glVertex2f(_dim.x, 0.0);
    glTexCoord2f (_dim.x, _dim.y); glVertex2f(_dim.x, _dim.y);
    glTexCoord2f (0.0,    _dim.y); glVertex2f(0.0,    _dim.y);
    glEnd ();

    glutSwapBuffers ();

    checkGlErrors ("Display::display");

    if (_nextFrame)
    {
	fps.tick (_dim.x * _dim.y * sizeof (Imf::Rgba));

	//
	// Display the frame rate once every 100 frames.
	//

	if ((totalFrames % 100) == 0)
	    std::cout << fps.fps () << " fps, "
		      << fps.bandwidth () / (1024*1024)
		      << " MB/s" << std::endl;
	_nextFrame = false;
	drawlatency = now () - drawlatency;
    }
}

} // namespace ExrPlay
