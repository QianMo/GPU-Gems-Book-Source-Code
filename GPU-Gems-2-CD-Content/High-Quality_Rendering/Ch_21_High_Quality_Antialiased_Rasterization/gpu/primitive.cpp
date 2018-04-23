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

#include <malloc.h>

#include "dassert.h"
#include "gpu.h"
#include "hash.h"

std::vector<Vector4> GpuPrimitive::constant_color;
std::vector<int> GpuPrimitive::free_constant_idx;



GpuPrimitive::GpuPrimitive (int n, Vector3 *polygon) 
    : type (POLYGON), nu (n), nv (0), buftypebits (0)
{
    P = (Vector3 *)malloc (n * sizeof (Vector3));
    memcpy (P, &polygon[0], n * sizeof (Vector3));
}



GpuPrimitive::GpuPrimitive (GpuPrimitiveType type, unsigned short nu,
                            unsigned short nv, Vector3 *P)
    : type (type), nu (nu), nv (nv), buftypebits (0), P (P)
{
    // assume other attributes are unset
    texcoord_clear();
    for (int i = 0; i < max_attribs; i++) texbuf[i] = NULL;
}



GpuPrimitive::GpuPrimitive (int xmin, int xmax, int ymin, int ymax) 
    : type (RECTANGLE), nu (0), nv (0), buftypebits (0)
{
    DASSERT (xmax > xmin && ymax > ymin);
    int *rect = (int *)malloc (4 * sizeof (int));
    rect[0] = xmin;
    rect[1] = ymin;
    rect[2] = xmax;
    rect[3] = ymax;
    P = *(Vector3 **)&rect;
}



GpuPrimitive::GpuPrimitive (Bbox3 &bbox) 
    : type (BBOX), nu(0), nv(0), buftypebits (0)
{
    Bbox3 *copy = (Bbox3 *)malloc (sizeof (Bbox3));
    memcpy (copy, &bbox, sizeof (Bbox3));
    P = *(Vector3 **) &copy;
}



GpuPrimitive::~GpuPrimitive ()
{
    if (type == POLYGON || type == BBOX || type == RECTANGLE)
        free (P);
}



void
GpuPrimitive::texcoord (int index, int nchannels, float *coords)
{
    DASSERT (index >= 0 && index < max_attribs);
    DASSERT (type != RECTANGLE && type != POLYGON && type != BBOX);
    
    if (coords == NULL)                 // don't enable unless we have data
        nchannels = 0;

    DASSERT (nchannels < 5);
    
    // if this buffer was using a constant, clear the constant
    if (texcoord_status (index) == TEXCOORD_CONST) {
        release_constant_id ((int)texbuf[index]);
        texbuf[index] = 0;
    }

    // Set the state of this buffer so we know the number of float channels
    texcoord_state (index, nchannels);

    GpuCreationMakeCurrent();
    DASSERT (gpu_test_opengl_error());
    
    if (coords == NULL)
        glMultiTexCoord3fARB (GL_TEXTURE0_ARB + index, 0, 0, 0);
    else
        texbuf[index] = coords;
    
    DASSERT (gpu_test_opengl_error());
    GpuCreationRelease();
}



void
GpuPrimitive::texcoord (int index, Vector3 *coords)
{
    texcoord (index, 3, (float *)coords);
}



void
GpuPrimitive::texcoord (int index, Vector4 *coords)
{
    texcoord (index, 4, (float *)coords);
}



GLuint
GpuPrimitive::get_constant_id (float x, float y, float z, float w)
{
    GLuint i;

    if (!free_constant_idx.empty()) {
        i = free_constant_idx[free_constant_idx.size() - 1];
        free_constant_idx.pop_back();
        constant_color[i] = Vector4 (x, y, z, w);
    } else {
        if (constant_color.empty()) {
            constant_color.push_back (Vector4 (0, 0, 0, 0));
            constant_color.push_back (Vector4 (1, 1, 1, 1));
        }
        if (x == 0 && y == 0 && z == 0 && w == 0) {
            return 0;
        } else if (x == 1 && y == 1 && z == 1 && w == 1) {
            return 1;
        }
        
        constant_color.push_back (Vector4 (x, y, z, w));
        i = (GLuint)constant_color.size() - 1;
    }

    return i;
}



void
GpuPrimitive::release_constant_id (GLuint i)
{
    if (i > 1)                  // first two indices are reserved
        free_constant_idx.push_back (i);
}



void
GpuPrimitive::texcoord (int index, float x, float y, float z, float w)
{
    DASSERT (index >= 0 && index < max_attribs);
    DASSERT (type != RECTANGLE && type != POLYGON && type != BBOX);

    // delete the old buffer object and store the constant index
    if (texcoord_status (index) == TEXCOORD_CONST)
        release_constant_id ((int)texbuf[index]);
    texbuf[index] = (float *)get_constant_id (x, y, z, w);
    texcoord_state (index, TEXCOORD_CONST);
}



bool
GpuPrimitive::validate (const GpuDrawmode &drawmode) const
{
    GpuFragmentProgram *fp = drawmode.fragment_program();
    GpuVertexProgram *vp = drawmode.vertex_program();
    
    char errbuf[4096];
    errbuf[0] = '\0';
    
    // Check to make sure that the fragment program inputs are all
    // set as vertex program outputs
    if (fp != NULL) {
        for (int i = 0; i < 16; ++i) {
            char fpbuf[128], vpbuf[128];
            sprintf (fpbuf, "f[TEX%d]", i);
            sprintf (vpbuf, "o[TEX%d]", i);
            if (vp != NULL && fp->find (fpbuf) && !vp->find (vpbuf)) {
                sprintf (errbuf, "Fragment program \"%s\" uses texcoord %d "
                         "which is not set in vertex program \"%s\"\n",
                         fp->title(), i, vp->title());
                goto error;
            }
            if (vp != NULL && !fp->find (fpbuf) && vp->find (vpbuf) &&
                // FIXME: Special exemption for TEX0 and motion blur,
                //        as noted in gpuhider.cpp
                (i != 0 || texcoord_status (1) == TEXCOORD_ON)) {
                sprintf (errbuf, "Vertex program \"%s\" passes texcoord %d "
                         "which is not used in fragment program \"%s\"\n",
                         vp->title(), i, fp->title());
                goto error;
            }
#if 0
            // Specially-tough checking for fixed function vps
            if (vp == NULL && !fp->find (fpbuf)) {
                sprintf (errbuf, "Fixed function vertex program passes "
                         "texcoord %d which is not used in fragment "
                         "program \"%s\"\n", i, fp->title());
                goto error;
            }
#endif                
        }
    }

    return true;

 error:

    fprintf (stderr, "Gpu ERROR: %s", errbuf);
    fflush (stderr);
    
    return false;
    
}



void
GpuPrimitive::render (const GpuDrawmode &drawmode, GpuCanvas &canvas,
    unsigned int mask_texcoords, bool mask_color, bool mask_depth)
{
    canvas.update (drawmode);

#ifdef DEBUG    
    validate (drawmode);
#endif    

    if (mask_color)
        glColorMask (GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    if (mask_depth)
        glDepthMask (GL_FALSE);

    switch (type) {
    case RECTANGLE: rect (drawmode, canvas); break;
    case POLYGON:   polygon (drawmode, canvas); break;
    case BBOX:      bbox (drawmode, canvas); break;

    case QUADMESH:
                    quadmesh (drawmode, canvas, mask_texcoords);
        break;
    }

    if (mask_color)
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    if (mask_depth)
        glDepthMask (GL_TRUE);
}



void
GpuPrimitive::quadmesh_vertex (int u, int v, unsigned int mask_texcoords)
{
    int idx = v * nu + u;
    for (int i = 0; i < max_attribs; i++) {
        if (mask_texcoords & (1 << i)) {
            glMultiTexCoord4fARB (GL_TEXTURE0_ARB + i, 0, 0, 0, 0);
        } else {
            int j;
            switch (texcoord_status (i)) {
            case TEXCOORD_OFF:
                // Texture coordinate is unused, set to black to be safe
                glMultiTexCoord4fARB (GL_TEXTURE0_ARB + i, 0, 0, 0, 0);
                break;

            case TEXCOORD_ON:
                // Texture coordinate array id stored in texbuf[]
                DASSERT (texbuf[i] != NULL);
                switch (texcoord_nchannels (i)) {
                case 0:
                    DASSERT (0);
                    break;
                case 1:
                    glMultiTexCoord1fARB (GL_TEXTURE0_ARB + i,
                                          texbuf[i][idx]);
                    break;
                case 2:
                    glMultiTexCoord2fvARB (GL_TEXTURE0_ARB + i,
                                          (GLfloat *)&texbuf[i][idx*2]);
                    break;
                case 3:
                    glMultiTexCoord3fvARB (GL_TEXTURE0_ARB + i,
                                          (GLfloat *)&texbuf[i][idx*3]);
                    break;
                case 4:
                    glMultiTexCoord4fvARB (GL_TEXTURE0_ARB + i,
                                          (GLfloat *)&texbuf[i][idx*4]);
                    break;
                }
                break;
                
            case TEXCOORD_CONST:
                // Constant index stored in texbuf[i]
                j = (int)texbuf[i];
                glMultiTexCoord4fvARB (GL_TEXTURE0_ARB + i,
                                       (GLfloat *)&constant_color[j][0]);
                break;
            }
        }
    }
    glVertex3fv ((GLfloat *)&P[idx][0]);
}


    
void
GpuPrimitive::quadmesh (const GpuDrawmode &drawmode, GpuCanvas &canvas,
                        unsigned int mask_texcoords)
{
    DASSERT (gpu_test_opengl_error());
    for (int v = 0; v < nv-1; ++v) {
        for (int u = 0; u < nu-1; ++u) {
            glBegin (GL_POLYGON);
            quadmesh_vertex (u, v,     mask_texcoords);
            quadmesh_vertex (u+1, v,   mask_texcoords);
            quadmesh_vertex (u+1, v+1, mask_texcoords);
            quadmesh_vertex (u, v+1,   mask_texcoords);
            glEnd ();
        }
    }
    DASSERT (gpu_test_opengl_error());
}



void
GpuPrimitive::rect (const GpuDrawmode &drawmode, GpuCanvas &canvas)
{
    // Note: values stashed in strange places in constructor
    int *rect = *(int **)&P;
    glRecti (rect[0], rect[1], rect[2], rect[3]);
}



void
GpuPrimitive::bbox (const GpuDrawmode &drawmode, GpuCanvas &canvas)
{
    Bbox3 &bbox = *(*(Bbox3 **)&P);
    
    // create a unit cube display list
    Vector3 v0 (bbox[XMIN], bbox[YMIN], bbox[ZMIN]);
    Vector3 v1 (bbox[XMAX], bbox[YMIN], bbox[ZMIN]);
    Vector3 v2 (bbox[XMAX], bbox[YMAX], bbox[ZMIN]);
    Vector3 v3 (bbox[XMIN], bbox[YMAX], bbox[ZMIN]);
    Vector3 v4 (bbox[XMIN], bbox[YMIN], bbox[ZMAX]);
    Vector3 v5 (bbox[XMAX], bbox[YMIN], bbox[ZMAX]);
    Vector3 v6 (bbox[XMAX], bbox[YMAX], bbox[ZMAX]);
    Vector3 v7 (bbox[XMIN], bbox[YMAX], bbox[ZMAX]);

    DASSERT (gpu_test_opengl_error());
    
    // Note: use a counter-clockwise vertex orientation for OpenGL,
    //       and that we look down the negative z axis, so zmin is further!
    glBegin (GL_QUADS);
    glVertex3fv (v0.data());
    glVertex3fv (v3.data());
    glVertex3fv (v2.data());
    glVertex3fv (v1.data());
        
    glVertex3fv (v1.data());
    glVertex3fv (v2.data());
    glVertex3fv (v6.data());
    glVertex3fv (v5.data());

#if 0
    // we never need to draw the backmost face of the cube
    glVertex3fv (v7.data());
    glVertex3fv (v4.data());
    glVertex3fv (v5.data());
    glVertex3fv (v6.data());
#endif        
    glVertex3fv (v0.data());
    glVertex3fv (v4.data());
    glVertex3fv (v7.data());
    glVertex3fv (v3.data());
        
    glVertex3fv (v3.data());
    glVertex3fv (v7.data());
    glVertex3fv (v6.data());
    glVertex3fv (v2.data());

    glVertex3fv (v1.data());
    glVertex3fv (v5.data());
    glVertex3fv (v4.data());
    glVertex3fv (v0.data());
    glEnd();

    DASSERT (gpu_test_opengl_error());
}



void
GpuPrimitive::polygon (const GpuDrawmode &drawmode, GpuCanvas &canvas)
{
    DASSERT (gpu_test_opengl_error());
    glBegin (GL_POLYGON);
    for (unsigned int i = 0; i < nu; i++)
        glVertex3fv (P[i].data());
    glEnd();
    DASSERT (gpu_test_opengl_error());
}

