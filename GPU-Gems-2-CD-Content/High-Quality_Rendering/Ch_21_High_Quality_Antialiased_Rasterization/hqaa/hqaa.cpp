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

#include <iostream>

#include "ppm.h"
#include "argparse.h"
#include "gridfile.h"
#include "downsample.h"
#include "accumulate.h"



static std::vector <char *> cmdlinefiles;       // input file list
static std::vector <GpuPrimitive *> prims;      // primitive list
static int total_quads = 0;



static void
usage (void)
{
    std::cerr << "hqaa [options][scenefile]\n";
    std::cerr << "Which: Renders a scene using high-quality, ";
    std::cerr << "gpu-accelerated scan conversion\n";
    std::cerr << "Where:\n";
    std::cerr << "       scenefile          Input scene file\n";
    std::cerr << "       -help              Display usage message\n";
    std::cerr << "       -camera x y z      Camera location\n";
    std::cerr << "       -lookat x y z      Camera direction\n";
    std::cerr << "       -fov angle         Camera field of view\n";
    std::cerr << "       -nearfar near far  Camera clipping planes\n";
    std::cerr << "       -resolution x y    Image resolution in pixels\n";
    std::cerr << "       -bucketsize x y    Bucket size in pixels\n";
    std::cerr << "       -supersample x y   Super-sample resolution\n";
    std::cerr << "       -filter name x y   Filter name and radii in pixels\n";
    std::cerr << "       -bitdepth b        32, 16, or 8 bits per channel\n";
    std::cerr << "       -o filename        Output image filename\n";
}



// Construct a GpuPrimitive from the Grid structure input from scene file.
// Side effect is to update the global bbox and total quad count.

static GpuPrimitive *
create_primitive (Grid &grid)
{
    int ps, pt;
    Vector3 *P;
    if (!grid.data ("P", ps, pt, &P)) {
        std::cerr << "Cannot find position in grid\n";
        exit (-1);
    }
    int cs, ct;
    Vector3 *C;
    if (!grid.data ("C", cs, ct, &C)) {
        std::cerr << "Cannot find color in grid\n";
        exit (-1);
    }

    if (ps != cs || pt != ct) {
        std::cerr << "Only supports grids where number of position "
            "elements match number of color elements\n";
        exit (-1);
    }

    GpuPrimitive *p = new GpuPrimitive(GpuPrimitive::QUADMESH, ps, pt, P);
    p->texcoord (0, C);

    total_quads += (ps - 1) * (pt - 1);
    
    return p;
}



// Compute a standard projection matrix from a field of view, hither and
// yon plans and screen resolution.  This does not include tile offsets.

Matrix4
compute_cam2ras (float fov, float hither, float yon, int resx, int resy)
{
    // Set up a projection matrix from camera to screen space (incl. aspect)
    float aspect = float (resx) / float (resy);
    Matrix4 cam2scr = Matrix4::PerspectiveMatrix (fov, aspect, hither, yon);

    // Set up a 2D matrix from screen space to NDC (halve and translate)
    Matrix4 scr2ndc =
        Matrix4 (0.5,  0,  0, 0,
                  0, -0.5, 0, 0,
                  0,   0,  1, 0,
                 0.5, 0.5, 0, 1);

    // Set up a 2D matrix from NDC to raster space (OpenGL projection matrix)
    Matrix4 ndc2ras = 
	Matrix4 ((float)resx,  0,           0, 0,
                 0,            (float)resy, 0, 0,
                 0,            0,           1, 0,
                 0,            0,           0, 1);


    Matrix4 cam2ras = cam2scr * scr2ndc * ndc2ras;
    return cam2ras;
}



// Compute an offset matrix for the tile at origin (x,h) with width "tx"
// and height "ty" where the fullscreen resolution is resx by resy.

Matrix4
compute_cam2tile (Matrix4 &cam2ras, int resx, int resy, bool flipy,
                  int x, int y, int tx, int ty)
{
    // Construct a camera to ndc transform that displays only
    // this hider's viewport.
    float flipper = flipy ? -1.0f : 1.0f;
    float offset = flipy ? 1.0f : 0.0f;

    Matrix4 ras2ndc = Matrix4 (
        1.0f/resx,    0,        0, 0,
        0,        flipper/resy, 0, 0,
        0,            0,        1, 0,
        0,          offset,     0, 1);

    // find the lower left corner of this tile in ndc coordinates
    float xndc = (float)x / resx;
    float yndc = (float)y / resy;

    // find the inverse width and height of this tile in ndc coordinates
    float inv_wndc = (float)resx / tx;
    float inv_hndc = (float)resy / ty;

    // construct a matrix to zoom this tile's ndc region to [-1,1]x[-1,1]
    Matrix4 T0 = Matrix4::TranslationMatrix (Vector3 (-xndc, -yndc, 0));
    Matrix4 S  = Matrix4::ScaleMatrix (2*inv_wndc, 2*inv_hndc,1);
    Matrix4 T1 = Matrix4::TranslationMatrix (Vector3 (-1, -1, 0));
    Matrix4 ndc2tile = T0 * S * T1;

    // construct the camera to tile matrix using cam2ras and the above mats
    Matrix4 cam2tile = cam2ras * ras2ndc * ndc2tile;
    return cam2tile;
}



// Clear the screen (tile) and render all of the primitives.  This could
// be optimized to only render primitives that touch the current tile.

static void
render (GpuDrawmode &drawmode, GpuCanvas &canvas)
{
    canvas.clear ();
    for (size_t i = 0; i < prims.size(); i++)
        prims[i]->render (drawmode, canvas);
}



static int
parse_files (int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
        cmdlinefiles.push_back (strdup (argv[i]));
    return 0;
}



int
main (int argc, char *argv[])
{
    bool help_flag = false;
    int resx = 720;          // full pixel resolution
    int resy = 486;
    int ssx = 4;             // super-sample resolution
    int ssy = 4;
    int tx = 32;             // bucket dimensions
    int ty = 32;
    char *filtername = NULL; // filter type
    float filterx = 2.0f;    // filter diameter
    float filtery = 2.0f;
    int bitdepth = 32;       // floating point channel size (8, 16, or 32 bits)
    int nchannels = 4;       // RGBA for all buffers
    char *outfile = NULL;    // eg. "strip.ppm" to save to file in strips
    Vector3 camera (0,0,0);  // camera location
    Vector3 lookat (0,0,0);  // camera direction is lookat-camera
    float fov = 60;          // perspective camera matrix field of view
    float hither = 0.1f;     // near clipping plane
    float yon = 10000;       // far clipping plane
    char *gridfile = NULL;   // input grid file name

    ArgParse ap (argc, argv);
    if (ap.parse (
            "%*", parse_files,
            "-help", &help_flag,
            "-camera %f %f %f", &camera[0], &camera[1], &camera[2],
            "-lookat %f %f %f", &lookat[0], &lookat[1], &lookat[2],
            "-fov %f", &fov,
            "-nearfar %f %f", &hither, &yon,
            "-resolution %d %d", &resx, &resy,
            "-bucketsize %d %d", &tx, &ty,
            "-supersample %d %d", &ssx, &ssy,
            "-filter %S %f %f", &filtername, &filterx, &filtery,
            "-bitdepth %d", &bitdepth,
            "-o %S", &outfile,
            NULL) < 0) {
        std::cerr << ap.error_message() << std::endl;
        usage ();
        return EXIT_FAILURE;
    }

    if (help_flag) {
        usage ();
        return EXIT_SUCCESS;
    }
    
    // check some conditions we know we can't handle
    if (bitdepth != 32 && bitdepth != 16 && bitdepth != 8) {
        std::cerr << "Invalid bitdepth " << bitdepth << " use 32, 16, or 8.\n";
        return EXIT_FAILURE;
    }
    if (nchannels != 4) {
        std::cerr << "Invalid number of channels " << nchannels << " use 4\n";
        return EXIT_FAILURE;
    }
    if (ssx < 1 || ssy < 1) {
        std::cerr << "Invalid super sample resolution " << ssx << " x ";
        std::cerr << ssy << "\n";
        return EXIT_FAILURE;
    }
    int max_tex_dim = GpuOGL::max_texture_dimension ();
    if (tx * ssx > max_tex_dim || ty * ssy > max_tex_dim) {
        std::cerr << "Supersample tile resolution of " << tx * ssx << " x ";
        std::cerr << ty * ssy << " larger than max " << max_tex_dim << "\n";
        return EXIT_FAILURE;
    }
    if (tx * ssx * ty * ssy * (bitdepth/8) > max_tex_dim * max_tex_dim * 4) {
        std::cerr << "Supersample tile resolution of " << tx * ssx << " x ";
        std::cerr << ty * ssy << " at bit depth " << bitdepth;
        std::cerr << " will not fit into GPU memory\n";
        return EXIT_FAILURE;
    }
    if (resx * ssx > max_tex_dim) {
        std::cerr << "Supersample image width " << resx*ssx << " exceeds ";
        std::cerr << "maximum accumulation buffer size of " << max_tex_dim;
        std::cerr << ", add support for tiled accumulation\n";
        return EXIT_FAILURE;
    }
    if (cmdlinefiles.size() > 1) {
        std::cerr << "Specified " << (int)cmdlinefiles.size();
        std::cerr << "input scene files, but only one is allowed\n";
        return EXIT_FAILURE;
    } else if (cmdlinefiles.empty ()) {
        std::cerr << "No scene files\n";
        return EXIT_FAILURE;
    }
    gridfile = cmdlinefiles[0];
    
    // create primitives parsed from gridfile, if any
    std::cout << "Reading gridfile \"" << gridfile << "\"...\n" << std::flush;
    GridFile gf;
    if (!gf.read (gridfile)) {
        std::cerr << gf.errorstr() << "\n";
        exit (-1);
    }
    for (size_t i = 0; i < gf.gridlist.size(); i++) {
        GpuPrimitive *p = create_primitive (*gf.gridlist[i]);
        prims.push_back (p);
    }
    std::cout << "Read in " << total_quads << " quads\n" << std::flush;

    // Create a drawing mode to hold the GPU state vector
    GpuDrawmode drawmode;

    // Create a fragment program that copies TEX0 to the output color
    // Note: floating point framebuffers *require* a fragment program
    static const char *tex0col_fp10 = 
        "!!FP1.0\n"
        "MOVR o[COLR], f[TEX0];\n"      // get color value from vertex
        "END\n";
    GpuFragmentProgram fp ("tex0col", tex0col_fp10);
    drawmode.fragment_program (&fp);

    // Create a vertex program that copys TEX0 and computes the output
    // position using the modelview projection stored in param 0
    // Note: the default vertex program will also work, but this is
    // slightly more efficient since it only copies TEX0
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
    GpuVertexProgram vp ("vertex", vertex_vp20);
    vp.parameter (0, GL_MODELVIEW_PROJECTION_NV);
    drawmode.vertex_program (&vp);

    drawmode.zbuffer (GpuDrawmode::ZBUF_LESS);
    
    Matrix4 cam2ras = compute_cam2ras (fov, hither, yon, resx, resy);

    // Create the filter we'll need
    if (filtername == NULL)
        filtername = "gaussian";
    Filter2D *filter = Filter2D::MakeFilter (filtername, filterx, filtery);
    if (filter == NULL) {
        std::cerr << "Cannot construct filter \"" << filtername;
        std::cerr << "\" of size " << filterx << " x " << filtery << "\n";
        return EXIT_FAILURE;
    }

    // Allocate a nchannel, fp32, RGBA tile readback buffer
    GpuDownsample downsample;
    int fxup = int (ceil (filter->width() / 2.0 + 0.5));  // roundup
    int fyup = int (ceil (filter->height() / 2.0 + 0.5));
    int tilew = tx + 2 * fxup;
    int tileh = ty + 2 * fyup;

    // Create a drawing surface to render a single super sampled tile
    GpuPBuffer pbuffer (tilew*ssx, tileh*ssy, bitdepth);
    GpuCanvas canvas (pbuffer);

    // Pad the final resolution by the filter radius to generate the
    // pixels we need to properly filter the edges of the final image
    int fx = int (ceil (filter->width()  / 2.0 - 0.5));
    int fy = int (ceil (filter->height() / 2.0 - 0.5));

    // Open an output file so we can write scanlines as we go
    Ppm ppm;
    GpuAccumulate accumulate;
    bool usewindow = outfile == NULL;
    if (outfile) {
        if (!ppm.open (outfile, nchannels, resx, resy)) {
            std::cerr << "Cannot open output ppm file\n";
            return EXIT_FAILURE;
        }
    } else {
        accumulate.begin_image (resx, resy, bitdepth, usewindow);
    }
        
    // Render all tiles
    for (int y = -fy; y < resy+fy; y += ty) {
        
        if (outfile)
            accumulate.begin_strip (resx, tileh, fy, bitdepth);
        
        for (int x = -fx; x < resx+fx; x += tx) {

            // Compute the matrix to render this tile fullscreen
            Matrix4 cam2tile = compute_cam2tile (cam2ras, resx, resy,
                                                 usewindow, x, y, tx, ty);
            drawmode.view (cam2tile, tx*ssx, ty*ssy);
            
            // Draw the entire scene
            render (drawmode, canvas);

            // Create a texture from the current framebuffer
            // Note: passing "true" as the last parameter of these
            // texture construtors forces the image to be padded with
            // a one pixel black border.
            GpuTexture fbtex ("downsamplefb");
            fbtex.load (canvas, 0,0, tx*ssx, ty*ssy, bitdepth, 0, true);

            // Downsample the rendered texture and store in a new texture
            GpuTexture &tile = downsample.tile (fbtex, tx, ty, ssx, ssy,
                                                tilew, tileh,*filter,bitdepth);
            
            // Accumulate the resulting texture into the final image
            accumulate.tile (tile, x - fxup, y - fyup, resx, resy, bitdepth);
        }

        // If we are outputting an image, output the current strip of scanlines
        if (outfile) {
            // write out a block of finished scanlines (not including overlap)
            GpuTexture &strip = accumulate.end ();  // retrieve strip texture
            static float *stripbuf = NULL;          // tile strip readback buf
            if (stripbuf == NULL)
                stripbuf = (float *)malloc(resx*tileh*nchannels*sizeof(float));
            strip.readback (stripbuf, resx, tileh, GL_FLOAT_RGBA_NV, GL_FLOAT);
            int ymin = std::max (0, y-fy-1);        // don't output unfinished
            int ymax = std::min (y+ty-fy-1, resy);  //    scanlines
            float *scanline = stripbuf;
            for (int yy = ymin; yy < ymax; ++yy, scanline += resx*nchannels) {
                if (!ppm.write_scanline (yy, scanline)) {
                    std::cerr << "Cannot write ppm scanline " << yy << "\n";
                    return EXIT_FAILURE;
                }
            }
        }
    }

    // Close the output file or keep the window open
    if (outfile && !ppm.close ()) {
        std::cerr << "Cannot close ppm file\n";
        return EXIT_FAILURE;
    } else {
        accumulate.end();
    }
}
