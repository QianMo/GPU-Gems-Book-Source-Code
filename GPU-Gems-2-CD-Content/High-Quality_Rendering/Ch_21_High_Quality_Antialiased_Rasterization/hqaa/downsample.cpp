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

#include "downsample.h"


GpuDownsample::GpuDownsample () :
    last_ssx (-1), last_ssy (-1), xpass (NULL), ypass (NULL),
    pbuffer (NULL), canvas (NULL), tiletex ("downsampletex")
{}



GpuDownsample::~GpuDownsample ()
{
    delete xpass;
    delete ypass;
    delete pbuffer;
    delete canvas;
}



// Sample the filter function at the given subpixel offset, t,
// assuming a pixel sample rate of px,py.  Isxpass is true if the
// t offset is in X and false if it is in Y.
static float
sample_filter (Filter2D& filter, float t, int px, int py, bool isxpass)
{
    float fx = 0;                               // initialize to zero in X & Y
    float fy = 0;
    if (isxpass)                                // sampling in X
        fx = t / px;
    else                                        // sampling in Y
        fy = t / py;

    return filter (fx, fy);                     // sample the filter function
}



// Compute the sum of all total filter samples over the entire valid
// radius of the filter along either the x or y axis, assuming a
// pixel sample rate of px,py.  Isxpass is true if we are summing in X
// and false if we are summing in Y.  The result value is used to
// normalize the filter for each sample point.
//
// If the subpixel sample rate is even, we use offset = 0.5 to move
// the t values to the center of the subpixel.
static float
sum_filter (Filter2D& filter, int r, int px, int py, float offset, bool isxpass)
{
    float total = 0;                            // total filter value
    for (int x = -r; x <= r; x++)
        total += sample_filter (filter, x + offset, px, py, isxpass);
    return total;
}



// Generate a fragment program which sums a set of sample points along
// either X or Y.  We generate separate fragment programs for X & Y.
//
// For each sample point, we compute the filter weight at that point
// and a texture coordinate which is used to lookup the source texel.
//
// The fragment program uses f[WPOS] as the origin and computes
// offsets which are used to look up texels in the source image.  The
// source image is magnified by the px or py subpixel sample rate.
//
// The source image is also assumed to be padded with a one pixel
// black boundary since texture lookups are clamped to the edge value
// (we use point sampled, rectangular floating point textures on the
// GPU).
//
// It is important to note that special care must be taken to account
// for even and odd subpixel sample rates.  Points in the destination
// image are transformed into the source image.  The center of a pixel
// in the destination image is transformed onto a pixel boundary in
// the source image when the subpixel sample rate is even and onto a
// pixel center when the subpixel sample rate is odd.  The 0.5 offset
// for even rates must be applied both to the texture coordinates and
// the filter sample locations.
// 

static const char *
generate_xory_pass_fp10 (Filter2D& filter, int px, int py, bool isxpass)
{
    // compute a conservative number of sample points
    float filtersize = isxpass ? px * filter.width() : py * filter.height();
    int r = (int)ceil (filtersize  / 2.0);

    // allocate a string large enough to hold the entire program
    int nlines = 7 * r + 10;    // 3 lines per sample, 2*(r+1) samples
    char *code = (char *) malloc (nlines * 80 * sizeof (char));
    ASSERT (code != NULL);
    code[0] = '\0';

    // Start program (temporaries are always initialized to zero)
    strcat (code, "!!FP1.0\n");

    // apply a 0.5 offset to even subpixel rates to move the texture
    // coordinate value to the center of the subpixel
    float even_offset = ((isxpass ? px : py) % 2 == 0) ? 0.5f : 0;

    // compute filter normalization factor
    float total_weight = sum_filter (filter, r, px, py, even_offset, isxpass);

    // compute the filter pad in the destination image space.
    // this is the pad that was added to the destination image
    // as computed in Hider::allocate_images()
    filtersize = isxpass ? filter.width() : filter.height();
    int filterpad = (int) ceil (filtersize / 2 - 0.5);

    // loop over all potential sample points.  Note that this may
    // include one or possibly two values outside the valid support
    // region for the filter.  This is safe since we only output code
    // for nonzero filter weights.
    for (int x = -r; x <= r; x++) {
        // sample the filter function at 'x' offset from the pixel center
        float weight = sample_filter (filter, x + even_offset, px, py, isxpass);
        weight /= total_weight;                         // normalize filter
        if (weight == 0)                                // skip zero samples
            continue;

        // compute texel coordinate offset from WPOS:
        //   o WPOS is always at the center of a pixel (ie. [0.5, 0.5])
        //   - source images are padded with a one pixel black border
        //   - account for the filter pad in the destination image
        char buf[1024];                                 // line buffer
        if (isxpass)
            sprintf (buf, "MADR R0.xy, f[WPOS].xyyy, {%d, 1}.xyyy, "
                "{%f, 1}.xyyy;\n", px, x + 1 - filterpad * px + even_offset);
        else
            sprintf (buf, "MADR R0.xy, f[WPOS].xyyy, {1, %d}.xyyy, "
                "{1, %f}.xyyy;\n", py, x + 1 - filterpad * py + even_offset);
        strcat (code, buf);

        // retrieve the texture color
        strcat (code, "TEX R3, R0.xyyy, TEX0, RECT;\n");
            
        // accumulate color from source texture using the filte weight
        sprintf (buf, "MADR R2, R3, {%f}.xxxx, R2;\n", weight);
        strcat (code, buf);
    }

    strcat (code, "MOVR o[COLR], R2;\n");
    strcat (code, "END\n");

    return code;
}



// Two Pass Separable Filtered Downsample
//
// Downsize the supersampled source image into a destination image
// that includes the filter region using a procedurally generated
// fragment shader.
//
// This program relies on correctly sized source and destination
// images allocated in allocate_images() and allocate_sample_images().
// The destination image must include the proper filter padding and
// hence destination images will overlap on screen.  Source images do
// not have any filter padding, and are scaled by the subpixel sample
// rate (px, py).

GpuTexture &
GpuDownsample::tile (GpuTexture &fbtex, int tx, int ty, int ssx, int ssy,
                     int dstw, int dsth, Filter2D& filter, int bitdepth)
{
    // create a new pbuffer and canvas if we need to based on the required
    // resolution of the temporary passes
    if (pbuffer == NULL ||
        (int)pbuffer->w() < dstw || (int)pbuffer->h() < dsth*ssy) {
        delete canvas;
        delete pbuffer;
        pbuffer = new GpuPBuffer (dstw, dsth*ssy, bitdepth);
        canvas = new GpuCanvas (*pbuffer);
    }

    // only create a new program if we need to.  program depends on ssx & ssy.
    // FIXME program depends on filter width & height as well (but not res'n).
    if (xpass == NULL || ssx != last_ssx || ssy != last_ssy) {
        last_ssx = ssx;
        last_ssy = ssy;
        delete ypass;
        delete xpass;
        
        // generate fragment programs for each pass
        const char *code = generate_xory_pass_fp10 (filter, ssx, ssy, true);
        xpass = new GpuFragmentProgram ("xpass", code);
        free ((char *)code);
        code = generate_xory_pass_fp10 (filter, ssx, ssy, false);
        ypass = new GpuFragmentProgram ("ypass", code);
        free ((char *)code);
    }

    // compute the resolution of the source texture
    int srcw = tx*ssx;
    int srch = ty*ssy;

    // dimensions of destination image in each pass
    int pw = dstw;
    int ph = srch;

    GpuDrawmode drawmode;
    drawmode.texture (0, fbtex);                        // set src in tex0
    drawmode.fragment_program (xpass);                  // set fragment prog
    drawmode.view (pw, ph);                             // set 2D pixel view

    GpuPrimitive rect0 (0, pw, 0, ph);                  // fullscreen rect
    rect0.render (drawmode, *canvas);                   // compute first pass

    // As a result of eliminating the intermediary pass image, the gpu
    // library doesn't know the real dimensions of the framebuffer so
    // we must pass the real viewport in the optional parameters. This
    // problem only shows up if the source width is smaller than the
    // destination width (eg. subpixel rate of 1,1)
    fbtex.load (*canvas, 0, 0, pw, ph, bitdepth, 0, true);

    ph = dsth;                                          // second pass height

    drawmode.view (pw, ph);                             // dest 2D view
    drawmode.fragment_program (ypass);                  // set fragment prog

    GpuPrimitive rect1 (0, pw, 0, ph);                  // fullscreen rect
    rect1.render (drawmode, *canvas);                   // compute second pass

    // store the final downsampled and filtered padded tile in texture
    tiletex.load (*canvas, 0, 0, dstw, dsth, bitdepth);

    return tiletex;
}
