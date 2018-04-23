/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
// (This is the Modified BSD License)

#include <iostream>
#include <vector>

#include "gpu.h"


#ifdef XWINDOWS
extern Display *GpuDisplay;
#endif

extern void GpuError (const char *message, ...);
extern void GpuCreationMakeCurrent();
extern void GpuCreationRelease();


void
gpu_pbuffer_force_link ()
{}



GpuPBuffer::GpuPBuffer (unsigned int w, unsigned int h, int bitdepth) 
    : GpuDrawable (w, h)
{
#ifndef DISABLE_GL    
#ifdef XWINDOWS
    open_display();

    int n;
    int screen = DefaultScreen (GpuDisplay);
    std::vector<int> fb_attribute;
    fb_attribute.push_back (GLX_DRAWABLE_TYPE);
    fb_attribute.push_back (GLX_PBUFFER_BIT);
    fb_attribute.push_back (GLX_DOUBLEBUFFER);
    fb_attribute.push_back (false);
    fb_attribute.push_back (GLX_RED_SIZE);
    fb_attribute.push_back (bitdepth);
    fb_attribute.push_back (GLX_GREEN_SIZE);
    fb_attribute.push_back (bitdepth);
    fb_attribute.push_back (GLX_BLUE_SIZE);
    fb_attribute.push_back (bitdepth);
    fb_attribute.push_back (GLX_ALPHA_SIZE);
    fb_attribute.push_back (bitdepth);
    fb_attribute.push_back (GLX_DEPTH_SIZE);
    fb_attribute.push_back (24);
    if (bitdepth > 8) {
        fb_attribute.push_back (GLX_FLOAT_COMPONENTS_NV);
        fb_attribute.push_back (true);
    }
#ifdef RTT    
    if (rtt) {
        fb_attribute.push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV);
        fb_attribute.push_back(true);
    }
#endif    
    fb_attribute.push_back (None);
    fbconfig = glXChooseFBConfig(GpuDisplay, screen, &fb_attribute[0], &n);
    if (n == 0 || fbconfig == NULL) {
        GpuError ("suitable floating point pbuffer format");
        // FIXME How guarantee caller switches to cpu-only?
        return;
    }
    
#if 0
    for (int i = 0; i < n; i++) {
        std::cout << "FB configuration #" << i << ":\n";
        int rsize;
        glXGetFBConfigAttrib(GpuDisplay, fbconfig[i], GLX_RED_SIZE, &rsize);
        std::cout << "    red size = " << rsize << "\n";
        int asize;
        glXGetFBConfigAttrib(GpuDisplay, fbconfig[i],GLX_ALPHA_SIZE,&asize);
        std::cout << "    alpha size = " << asize << "\n";
        int dsize;
        glXGetFBConfigAttrib(GpuDisplay, fbconfig[i],GLX_DEPTH_SIZE,&dsize);
        std::cout << "    depth size = " << dsize << "\n";
        int maxw;
        glXGetFBConfigAttrib(GpuDisplay, fbconfig[i], GLX_MAX_PBUFFER_WIDTH,
            &maxw);
        std::cout << "    max width = " << maxw << "\n";
        int maxh;
        glXGetFBConfigAttrib(GpuDisplay, fbconfig[i],
            GLX_MAX_PBUFFER_HEIGHT, &maxh);
        std::cout << "    max height = " << maxh << "\n";
    }
    std::cout << "End\n";
#endif

    
    int pb_attribute[] = { GLX_PBUFFER_WIDTH, w,
                           GLX_PBUFFER_HEIGHT, h,
                           GLX_PRESERVED_CONTENTS, True,
                           None };
    pbuffer = glXCreatePbuffer (GpuDisplay, fbconfig[0], pb_attribute);

    // Statically save the first context and share all others
    extern GLXContext GpuCreationContext;
    glx_context = glXCreateNewContext (GpuDisplay,
        *fbconfig, GLX_RGBA_TYPE, GpuCreationContext, True);
    if (GpuCreationContext == NULL) {
        GpuCreationContext = glx_context;
        if (GpuOGL::set_root_context != NULL)
            GpuOGL::set_root_context (GpuCreationContext);
    }
#endif

#ifdef WINNT
    GpuCreationMakeCurrent ();
    HDC hdc = wglGetCurrentDC();
    
    std::vector<int> pf_attribs;
    pf_attribs.push_back(WGL_DRAW_TO_PBUFFER_ARB);
    pf_attribs.push_back(true);
    pf_attribs.push_back(WGL_RED_BITS_ARB);
    pf_attribs.push_back(bitdepth);
    pf_attribs.push_back(WGL_GREEN_BITS_ARB);
    pf_attribs.push_back(bitdepth);
    pf_attribs.push_back(WGL_BLUE_BITS_ARB);
    pf_attribs.push_back(bitdepth);
    pf_attribs.push_back(WGL_ALPHA_BITS_ARB);
    pf_attribs.push_back(bitdepth);
    pf_attribs.push_back(WGL_DEPTH_BITS_ARB);
    pf_attribs.push_back(24);
    if (bitdepth > 8) {
        pf_attribs.push_back(WGL_FLOAT_COMPONENTS_NV);
        pf_attribs.push_back(true);
    }
#ifdef RTT    
    if (rtt) {
        pf_attribs.push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV);
        pf_attribs.push_back(true);
    }
#endif    
    pf_attribs.push_back(0);
    
    int pformat;
    unsigned int n;
    wglChoosePixelFormatARB (hdc, &pf_attribs[0], NULL, 1, &pformat, &n);
    if (n == 0) {
        std::cerr << "Cannot find a suitable floating point pbuffer format\n";
        abort();
    }
    
    // create a pbuffer and an associated GL rendering context
    std::vector<int> pb_attribs;
#ifdef RTT    
    if (rtt) {
        // render to texture
        pb_attribs.push_back(WGL_TEXTURE_TARGET_ARB);
        pb_attribs.push_back(WGL_TEXTURE_RECTANGLE_NV);
        pb_attribs.push_back(WGL_TEXTURE_FORMAT_ARB);
        pb_attribs.push_back(WGL_TEXTURE_FLOAT_RGBA_NV);
    }
#endif    
    pb_attribs.push_back(0);

    pbuffer = wglCreatePbufferARB (hdc, pformat, w, h, &pb_attribs[0]);
    if (!pbuffer) {
        DWORD err = GetLastError();
        char *errstr;
        switch (err) {
        case ERROR_INVALID_PIXEL_FORMAT: errstr = "invalid pixel format";break;
        case ERROR_NO_SYSTEM_RESOURCES: errstr = "no system resources";break;
        case ERROR_INVALID_DATA: errstr = "invalid data";break;
        default: errstr = "unknown problem"; break;
        }
        fprintf (stderr, "Error: cannot create pbuffer : %s ", errstr);
        abort ();
    }
    this->hdc = wglGetPbufferDCARB (pbuffer);
    if (!this->hdc) {
        fprintf (stderr, "Error: Cannot get pbuffer dc");
        abort();
    }
    hglrc = wglCreateContext (this->hdc);
    extern GpuContext GpuCreationContext;
    DASSERT (GpuCreationContext != NULL);
    if (!wglShareLists (GpuCreationContext, hglrc)) {
        fprintf (stderr, "Cannot share OpenGL contexts\n");
        fflush (stderr);
    }

#ifdef RTT    
    if (rtt) {
      // create texture for render-to-texture
      target = GL_TEXTURE_RECTANGLE_NV;
      glGenTextures(1, &tex);
      glBindTexture(target, tex);
      glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }
#endif
    
    GpuCreationRelease();
#endif
#endif    
}



GpuPBuffer::~GpuPBuffer()
{
#ifndef DISABLE_GL    
#ifdef XWINDOWS
    glXDestroyPbuffer (GpuDisplay, pbuffer);
#endif
#ifdef WINNT
    wglDeleteContext (hglrc);
    wglReleasePbufferDCARB (pbuffer, hdc);
    wglDestroyPbufferARB (pbuffer);
#endif
#endif    
}
