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
/////////////////////////////////////////////////////////////////////////////

#ifndef ACCUMULATE_H
#define ACCUMULATE_H

#include "gpu.h"

class GpuAccumulate {
public:
    GpuAccumulate () : window (NULL), pbuffer (NULL), canvas (NULL),
        accumtex ("accumtex"), fp ("accumfp", NULL), vp ("accumvp", NULL) {}
    ~GpuAccumulate () { delete canvas; delete window; delete pbuffer; }

    // call before accumulating tiles to accumulate tiles into a single
    // texture that is only one strip high, which assumes tiles will
    // be processed in row-major order and read back afterwards to 
    // be saved in an image.  Uses a pbuffer for rendering.
    void begin_strip (int resx, int tileh, int fy, int bitdepth);
    
    // call before accumulating to accumulate tiles into an onscreen window
    void begin_image (int resx, int resy, int bitdepth, bool usewindow);

    // call after accumulating either a strip or the entire image
    GpuTexture &end ();
    
    // accumulate this tile into the internal texture buffer
    void tile (GpuTexture &tile, int xorigin, int yorigin, 
               int resx, int resy, int bitdepth);

private:
    GpuWindow *window;
    GpuPBuffer *pbuffer;
    GpuCanvas *canvas;
    GpuTexture accumtex;
    bool accumulating_strips;
    bool usewindow;
    bool clear_accumtex;
    GpuDrawmode drawmode;
    GpuFragmentProgram fp;
    GpuVertexProgram vp;

    // allocate buffers and create fragment and vertex programs
    void initialize (int w, int h, int bitdepth, bool usewindow);

    // simple event handler for when we render to a live window
    friend int accumulate_event_handler (GpuEvent &event);
};

    
    


#endif // ACCUMULATE_H
