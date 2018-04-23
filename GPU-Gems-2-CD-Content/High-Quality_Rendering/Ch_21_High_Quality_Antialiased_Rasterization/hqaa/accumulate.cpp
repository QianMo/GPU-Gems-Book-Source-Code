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

#include "accumulate.h"


// A simple event handler to repaint the window and allow ESCAPE to quit

int
accumulate_event_handler (GpuEvent &event)
{
    GpuAccumulate *accumulate = (GpuAccumulate *)event.user_data;
    switch (event.type) {
    case GpuEvent::Redraw: {
        GpuPrimitive rect (0, accumulate->window->w(),
                           0, accumulate->window->h());
        accumulate->drawmode.texture (0, accumulate->accumtex);
        accumulate->drawmode.texture_clear (1);
        rect.render (accumulate->drawmode, *accumulate->canvas);
        glFlush ();  // drawing to the front buffer, must flush here?
        return 0;
    }
    case GpuEvent::KeyDown:
        if (event.key == GpuKeyEscape)
            accumulate->window->quit();
        return 0;
        break;

    default:
        /*EMPTY*/
        break;
    }

    return 1;
}



// Common function for allocating a window or pbuffer of the specified
// dimensions, setting up fragment and vertex programs, and global
// drawmode attributes.

void
GpuAccumulate::initialize (int w, int h, int bitdepth, bool usewindow)
{
    if (usewindow && window != NULL && window->w() == w && window->h() == h)
        return;
    if (!usewindow && pbuffer != NULL && pbuffer->w() == w && pbuffer->h() ==h)
        return;
    
    static const char *accum_fp10 = 
        "!!FP1.0\n"
        "SUB R0, f[WPOS], p[0];\n"
        "TEX R0, R0, TEX1, RECT;\n"
        "TEX R1, f[WPOS].xyyy, TEX0, RECT;\n"
        "ADDR o[COLR], R0, R1;\n"
        "END\n";
    fp.load (accum_fp10);
    drawmode.fragment_program (&fp);
    
    const char *vertex_vp20 = 
        "!!VP2.0\n"
        "MOV o[TEX0], v[8];\n"
        "MOV R1, v[0];\n"
        "DP4 R0.x, c[0], R1;\n"
        "DP4 R0.y, c[1], R1;\n"
        "DP4 R0.z, c[2], R1;\n"
        "DP4 R0.w, c[3], R1;\n"
        "MOV o[HPOS], R0;\n"
        "END\n";
    vp.load (vertex_vp20);
    vp.parameter (0, GL_MODELVIEW_PROJECTION_NV);
    drawmode.vertex_program (&vp);

    drawmode.view (w, h);
    drawmode.zbuffer (GpuDrawmode::ZBUF_OFF);
    drawmode.drawbuffer (GpuDrawmode::BUF_FRONT);

    // allocate window or pbuffer
    delete pbuffer;
    delete window;
    
    if (usewindow) {
        window = new GpuWindow (100, 100, w, h, false /*doublebuffered*/);
        canvas = new GpuCanvas (*window);
        window->set_event_handler (accumulate_event_handler, this);
        canvas->clear ();   // clear window once at beginning
        accumtex.load (*canvas, 0, 0, w, h, bitdepth);
        window->show (true);
    } else {
        pbuffer = new GpuPBuffer (w, h, bitdepth);
        canvas = new GpuCanvas (*pbuffer);
    }
    this->usewindow = usewindow;
}



// Setup rendering for a new strip of tiles, updating the accumulation
// texture by moving the top overlap section of the previous strip to
// the bottom of the current accumulation texture.
void
GpuAccumulate::begin_strip (int resx, int tileh, int fy, int bitdepth)
{
    initialize (resx, tileh, bitdepth, false);

    // Copy the top of the last strip to the bottom of the new accumulation
    // texture and clear the rest out.  We're a bit tricky here and reuse
    // the standard fragment program with the tile and accumulation texture
    // reversed so that we can use the f[WPOS]-p[0] computation normally
    // used for the tile to move the accumtex region. 
    canvas->clear ();
    GpuPrimitive rect (0, resx, 0, 2*fy);
    drawmode.texture_clear (0);
    drawmode.texture (1, accumtex);
    fp.parameter (0, Vector4 (0, (float)-(tileh-2*(fy+1)), 0, 0));
    drawmode.fragment_program (&fp);
    rect.render (drawmode, *canvas);
    accumtex.load (*canvas, 0, 0, resx, tileh, bitdepth);
    accumulating_strips = true;
}



void
GpuAccumulate::begin_image (int resx, int resy, int bitdepth, bool usewindow)
{
    initialize (resx, resy, bitdepth, usewindow);
    accumulating_strips = false;
}



GpuTexture &
GpuAccumulate::end ()
{
    if (usewindow) 
        GpuWindowMainloop();
    return accumtex;
}



void
GpuAccumulate::tile (GpuTexture &tile, int xorigin, int yorigin, 
                     int resx, int resy, int bitdepth)
{
    // bind the textures and render an accumulation rectangle
    drawmode.texture (0, accumtex);
    drawmode.texture (1, tile);
    int y = accumulating_strips ? 0 : yorigin;
    fp.parameter (0, Vector4 ((float)xorigin, (float)y, 0, 0));
    drawmode.fragment_program (&fp);
    GpuPrimitive rect (xorigin, xorigin+tile.w(), y, y+tile.h());
    rect.render (drawmode, *canvas);

    // copy the newly accumulated region into the accum texture
    int w = tile.w();
    int h = tile.h();
    if (xorigin < 0) {
        w += xorigin;
        xorigin = 0;
    }
    if (yorigin < 0) {
        h += yorigin;
        yorigin = 0;
    }
    if (xorigin + w >= resx)
        w -= xorigin + w - resx;
    if (yorigin + h >= resy)
        h -= yorigin + h - resy;
    if (accumulating_strips)
        yorigin = 0;
    accumtex.load (*canvas, xorigin, yorigin, w, h, bitdepth,
                   0 /*depth compare*/, false /*pad*/, xorigin, yorigin);

    // if we are displaying in a window, make sure to update
    if (usewindow) {
        window->repaint();
        GpuWindowMainloop (false);
    }
}
