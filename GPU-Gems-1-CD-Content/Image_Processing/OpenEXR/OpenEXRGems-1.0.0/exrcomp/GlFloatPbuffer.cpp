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

#include <GlFloatPbuffer.h>

#include <Iex.h>

namespace
{

const int fbAttrList[] =
{
    GLX_RED_SIZE,               16,
    GLX_GREEN_SIZE,             16,
    GLX_BLUE_SIZE,              16,
    GLX_ALPHA_SIZE,             16,
    GLX_STENCIL_SIZE,           0,
    GLX_DEPTH_SIZE,             0,
    GLX_FLOAT_COMPONENTS_NV,    true,
    GLX_DRAWABLE_TYPE,          GLX_PBUFFER_BIT,
    0
};

const int pbAttrList[] =
{
    GLX_LARGEST_PBUFFER,        true,
    GLX_PRESERVED_CONTENTS,     true,
    0
};

} // namespace


GlFloatPbuffer::GlFloatPbuffer (const Imath::V2i & dim)
    : _dim (dim),
      _display (0),
      _context (0),
      _pbuffer (0),
      _prevDisplay (0),
      _prevContext (0),
      _prevDrawable (None)
{
    _prevDisplay  = glXGetCurrentDisplay ();
    if (!_prevDisplay)
    {
	//
	// Try to connect to the default display.
	//

	_prevDisplay = XOpenDisplay (0);
	if (!_prevDisplay)
	{
	    THROW (Iex::BaseExc, "Can't create pbuffer "
		   << "(can't connect to default display)");
	}
    }

    _prevDrawable = glXGetCurrentDrawable ();
    _prevContext  = glXGetCurrentContext ();

    _display = _prevDisplay;
    int screen = DefaultScreen (_display);

    int unused;

    GLXFBConfig * config = glXChooseFBConfigSGIX (_display, screen, fbAttrList,
						  &unused);
    if (!config)
    {
	THROW (Iex::BaseExc, "Can't create pbuffer "
	       << "(glXChooseFBConfigSGIX failed)");
    }

    _pbuffer = glXCreateGLXPbufferSGIX (_display, config[0], _dim.x, _dim.y,
					pbAttrList);
    if (!_pbuffer)
    {
	THROW (Iex::BaseExc, "Can't create pbuffer "
	       << "(glXCreateGLXPbufferSGIX failed)");
    }

    _context = glXCreateContextWithConfigSGIX (_display, config[0],
					       GLX_RGBA_TYPE,
					       glXGetCurrentContext (),
					       true);
    if (!_context)
    {
	THROW (Iex::BaseExc, "Can't create pbuffer "
	       << "(glXCreateContextWithConfigSGIX failed)");
    }
}


GlFloatPbuffer::~GlFloatPbuffer ()
{
    if (_context && _display)
	glXDestroyContext (_display, _context);
    if (_pbuffer && _display)
	glXDestroyGLXPbufferSGIX (_display, _pbuffer);

    //
    // Restore previous context, if valid.
    //
    
    if (_prevDisplay && (_prevDrawable != None) && _prevContext)
	glXMakeCurrent (_prevDisplay, _prevDrawable, _prevContext);
}


void
GlFloatPbuffer::activate ()
{
    //
    // Save current context.
    //

    _prevDisplay  = glXGetCurrentDisplay ();
    _prevDrawable = glXGetCurrentDrawable ();
    _prevContext  = glXGetCurrentContext ();

    glXMakeCurrent (_display, _pbuffer, _context);
}


void
GlFloatPbuffer::deactivate ()
{
    //
    // Restore previous context, if valid.
    //

    if (_prevDisplay && (_prevDrawable != None) && _prevContext)
	glXMakeCurrent (_prevDisplay, _prevDrawable, _prevContext);
}
