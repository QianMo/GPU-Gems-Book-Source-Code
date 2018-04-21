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

#include <Comp.h>

#include <iostream>
#include <string>
#include <GL/glu.h>
#include <Cg/cgGL.h>

#include <Iex.h>
#include <GlFloatPbuffer.h>

namespace Comp
{

namespace
{

void
cgErrorCallback (void)
{
    std::cerr << "Cg error: " << cgGetErrorString (cgGetError ()) << std::endl;
    exit (1);
}

void
checkGlErrors ()
{
    GLenum error = glGetError ();
    if (error != GL_NO_ERROR)
    {
	std::cerr << "GL error: gluErrorString (error)" << std::endl;
	exit (1);
    }
}

void
comp (const Imath::V2i & dim,
      const Imf::Rgba * imageA,
      const Imf::Rgba * imageB,
      Imf::Rgba * imageC,
      const char * cgProgramName)
{
    GlFloatPbuffer pbuffer (dim);

    pbuffer.activate ();

    //
    // Set up default ortho view.
    //

    glLoadIdentity ();
    glViewport (0, 0, dim.x, dim.y);
    glOrtho (0, dim.x, dim.y, 0, -1, 1);

    //
    // Create input textures.
    //

    GLuint inTex[2];
    glGenTextures (2, inTex);

    GLenum target = GL_TEXTURE_RECTANGLE_NV;

    glActiveTextureARB (GL_TEXTURE0_ARB);
    glBindTexture (target, inTex[0]);

    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D (target, 0, GL_FLOAT_RGBA16_NV, dim.x, dim.y, 0, 
		  GL_RGBA, GL_HALF_FLOAT_NV, imageA);

    glActiveTextureARB (GL_TEXTURE1_ARB);
    glBindTexture (target, inTex[1]);

    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glPixelStorei (GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D (target, 0, GL_FLOAT_RGBA16_NV, dim.x, dim.y, 0, 
		  GL_RGBA, GL_HALF_FLOAT_NV, imageB);

    //
    // Compile the Cg program and load it.
    //

    cgSetErrorCallback (cgErrorCallback);

    CGcontext cgcontext = cgCreateContext ();

    std::string cgpath;
    char * userpath = getenv ("EXRCOMP_CG_PATH");
    if (userpath)
	cgpath = std::string (userpath) + "/" + cgProgramName;
    else
	cgpath = std::string (PKG_DATA_DIR) + "/" + cgProgramName;

    CGprogram cgprog = cgCreateProgramFromFile (cgcontext,
						CG_SOURCE,
						cgpath.c_str (),
						CG_PROFILE_FP30,
						0, 0);
    cgGLLoadProgram (cgprog);
    cgGLBindProgram (cgprog);
    cgGLEnableProfile (CG_PROFILE_FP30);

    //
    // Render to pbuffer.
    //

    glEnable (GL_FRAGMENT_PROGRAM_NV);
    glBegin (GL_QUADS);
    glTexCoord2f (0.0,   0.0);   glVertex2f(0.0,   0.0);
    glTexCoord2f (dim.x, 0.0);   glVertex2f(dim.x, 0.0);
    glTexCoord2f (dim.x, dim.y); glVertex2f(dim.x, dim.y);
    glTexCoord2f (0.0,   dim.y); glVertex2f(0.0,   dim.y);
    glEnd ();
    glDisable (GL_FRAGMENT_PROGRAM_NV);

    //
    // Read pixels out of pbuffer.
    //

    glReadPixels (0, 0, dim.x, dim.y, GL_RGBA, GL_HALF_FLOAT_NV, imageC);

    checkGlErrors ();

    pbuffer.deactivate ();
}

}

void
over (const Imath::V2i & dim,
      const Imf::Rgba * imageA,
      const Imf::Rgba * imageB,
      Imf::Rgba * imageC)
{
    comp (dim, imageA, imageB, imageC, "CompOver.cg");
}

void
in (const Imath::V2i & dim,
    const Imf::Rgba * imageA,
    const Imf::Rgba * imageB,
    Imf::Rgba * imageC)
{
    comp (dim, imageA, imageB, imageC, "CompIn.cg");
}

void
out (const Imath::V2i & dim,
     const Imf::Rgba * imageA,
     const Imf::Rgba * imageB,
     Imf::Rgba * imageC)
{
    comp (dim, imageA, imageB, imageC, "CompOut.cg");
}

}
