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


#include "gpu.h"
#include "dassert.h"



#ifdef DEBUG

#include <list>

typedef std::list<const GpuTexture *> GpuTextureList;
GpuTextureList texture_list;            // FIXME thread-protect

bool
valid_texture (const GpuTexture *tex)
{
    GpuTextureList::iterator i;
    for (i = texture_list.begin(); i != texture_list.end(); i++)
        if (*i == tex)
            return true;
    return false;
}



bool
remove_texture (const GpuTexture *tex)
{
    GpuTextureList::iterator i;
    for (i = texture_list.begin(); i != texture_list.end(); i++) {
        if (*i == tex) {
            i = texture_list.erase (i);
            return true;
        }
    }
    return false;
}



void
add_texture (const GpuTexture *tex)
{
    if (!valid_texture (tex))
        texture_list.push_back (tex);
}


#else
bool valid_texture (const GpuTexture *tex)  { return true; }
bool remove_texture (const GpuTexture *tex) { return true; }
void add_texture (const GpuTexture *tex) {}
#endif


static unsigned long long
update_textureid ()
{
    static unsigned long long globaltexid = 0;
    unsigned long long id = ++globaltexid;
    return id;
}


GpuTexture::GpuTexture (const char *name) 
    : name (name), id (0), textureid (update_textureid()), width (0), height(0)
{}



static bool
create_texture (int w, int h, int channel_bytes, GLenum format, GLenum type,
                const void *data, GLuint &id, int xmin, int xmax,
                int ymin, int ymax, int old_w, int old_h, int step,
                int txoff, int tyoff)
{
    GpuCreationMakeCurrent();
    DASSERT (gpu_test_opengl_error());

    if (xmax - xmin < 0)
        xmax = xmin + w - 1;
    if (ymax - ymin < 0)
        ymax = ymin + h - 1;

#ifdef DEBUG    
    DASSERT (xmax - xmin < w);
    DASSERT (ymax - ymin < h);
    if (old_w == 0) {
        DASSERT (old_h == 0);
    } else {
        DASSERT (txoff + xmax - xmin <= old_w);  // insure new region inside
        DASSERT (tyoff + ymax - ymin <= old_h);  //    existing texture
    }
#endif

    // always grab the data starting at (xmin, ymin)
    data = ((const char *)data) + (xmin + ymin * w) * step;
    
    // create a new texture if this is the first time it is loaded
    if (id == 0) {
        DASSERT (old_w == 0);
        glGenTextures (1, &id);
    }

    glBindTexture (GL_TEXTURE_RECTANGLE_NV, id);

    int align = (((w * step) % 4) == 0) ? 4 : 1;  // account for stride != 4
    glPixelStorei (GL_UNPACK_ALIGNMENT, align);
    glPixelStorei (GL_UNPACK_ROW_LENGTH, w);
    glPixelStorei (GL_UNPACK_IMAGE_HEIGHT, h);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER,
        GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER,
        GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S,
        GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T,
        GL_CLAMP_TO_EDGE);

    GLenum glformat = format;
    if (channel_bytes == 4) {
        if (glformat == GL_RGBA)
            glformat = GL_FLOAT_RGBA32_NV;
        else if (glformat == GL_RGB)
            glformat = GL_FLOAT_RGB32_NV;
        else if (glformat == GL_ALPHA || glformat == GL_RED)
            glformat = GL_FLOAT_R32_NV;
    }

    if (old_w != 0) {
        glTexSubImage2D (GL_TEXTURE_RECTANGLE_NV, 0, txoff, tyoff,
                         xmax-xmin+1, ymax-ymin+1, format, type, data);
    } else {
        glTexImage2D (GL_TEXTURE_RECTANGLE_NV, 0, glformat, xmax-xmin+1,
                      ymax-ymin+1, 0, format, type, data);
    }

    DASSERT (gpu_test_opengl_error());
    
    GpuCreationRelease();

    return old_w == 0;
}



void
GpuTexture::load (const Vector3 *color, int w, int h, int xmin, int xmax,
                  int ymin, int ymax, int txoff, int tyoff)
{
    if (create_texture (w, h, 4, GL_RGB, GL_FLOAT, color, id, xmin, xmax,
                        ymin, ymax, width, height, 3*sizeof(float),
                        txoff, tyoff)) {
        width = w;
        height = h;
        textureid = update_textureid ();
        add_texture (this);
    }
}



void
GpuTexture::load (const float *data, int w, int h, int xmin, int xmax,
                  int ymin, int ymax, int txoff, int tyoff)
{
    if (create_texture (w, h, 4, GL_RED, GL_FLOAT, data, id, xmin, xmax,
                        ymin, ymax, width, height, sizeof(float),
                        txoff, tyoff)) {
        width = w;
        height = h;
        textureid = update_textureid ();
        add_texture (this);
    }
}



void
GpuTexture::load (GpuCanvas &canvas, int x, int y, int w, int h,
                  int bitdepth, int depth_compare_op, bool pad_border,
                  int txoff, int tyoff)
{
    GpuOGL::copy_tex_image_stat++;

    if (txoff == -1 && tyoff == -1) {
        width = w;
        height = h;
    }

    GLenum format;
    if (depth_compare_op != 0) {
        format = GL_DEPTH_COMPONENT;
    } else {
        switch (bitdepth) {
        case 8:  format = GL_RGBA;            break;
        case 16: format = GL_FLOAT_RGBA16_NV; break;
        case 32: format = GL_FLOAT_RGBA_NV;   break;
        default: format = GL_FLOAT_RGBA_NV;   break;
        }
    }

#ifndef DISABLE_GL    
    canvas.make_current();
    canvas.update_clear();  // make canvas up-to-date in case nothing was drawn
    DASSERT (gpu_test_opengl_error());
    
    // generate a new texture id
    if (id == 0)
        glGenTextures (1, &id);
    glBindTexture (GL_TEXTURE_RECTANGLE_NV, id);
    
    DASSERT (gpu_test_opengl_error());
    if (pad_border) {
        GLenum glformat = format;
        if (format != GL_DEPTH_COMPONENT)
            glformat = GL_RGBA;
        // FIXME/BC: Should we be passing black pixels instead of NULL?
        glTexImage2D (GL_TEXTURE_RECTANGLE_NV, 0, format, w+2, h+2, 0,
            glformat, GL_FLOAT, NULL);

    }
    
    DASSERT (gpu_test_opengl_error());
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER,
        GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER,
        GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S,
        GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T,
        GL_CLAMP_TO_EDGE);
    // generates a gl error because texture rectangles can't have more
    // than one mipmap level on nv30 -- to be supported on nv40
//    glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAX_LEVEL, 0);

    DASSERT (gpu_test_opengl_error());
    // Setup image data values to place non-power-of-2 textures
    // within a valid texture
    glPixelStorei (GL_UNPACK_ROW_LENGTH, w);
//    glPixelStorei (GL_UNPACK_IMAGE_HEIGHT, h);

    DASSERT (gpu_test_opengl_error());
    if (format == GL_DEPTH_COMPONENT && depth_compare_op != 0) {
        // setup the texture comparison functions for shadow mapping
        glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_COMPARE_MODE_ARB,
            GL_COMPARE_R_TO_TEXTURE_ARB);
        glTexParameteri (GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_COMPARE_FUNC_ARB,
            depth_compare_op);
    }
    
    DASSERT (gpu_test_opengl_error());
    if (pad_border) {
        glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 1, 1, x, y, w, h);

        // fill in the border (911: should use render-to-texture clear!)
        // this way, the contents of the framebuffer are cleared (which is
        // always okay right now, but might be a bit dangerous).

        // Note that doing the clear and copying from the framebuffer to
        // texture is 4x the speed of using glTexSubImage2D!
        
        glClearColor (0,0,0,0);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glCopyTexSubImage2D (GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, 1, h+2);
        glCopyTexSubImage2D (GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, w+2, 1);
        glCopyTexSubImage2D (GL_TEXTURE_RECTANGLE_NV, 0, w+1, 0, 0, 0, 1, h+2);
        glCopyTexSubImage2D (GL_TEXTURE_RECTANGLE_NV, 0, 0, h+1, 0, 0, w+2, 1);
    } else if (txoff != -1 && tyoff != -1) {
        glCopyTexSubImage2D (GL_TEXTURE_RECTANGLE_NV, 0,txoff,tyoff, x,y, w,h);
    } else {
        glCopyTexImage2D (GL_TEXTURE_RECTANGLE_NV, 0, format, x, y, w, h,0);
    }
    
    DASSERT (gpu_test_opengl_error());
    add_texture (this);
    textureid = update_textureid ();
#endif    
}


void
GpuTexture::unload ()
{
    GpuCreationMakeCurrent();
    DASSERT (gpu_test_opengl_error());
    if (id != 0)
        glDeleteTextures (1, &id);
    GpuCreationRelease();
    id = 0;
    width = 0;
    height = 0;
    textureid = update_textureid();
}



GpuTexture::~GpuTexture ()
{
    remove_texture (this);
    unload ();
    name = "deleted";                           // catch dangling pointers
}



void
GpuTexture::bind (int texunit) const
{
    DASSERT (gpu_test_opengl_error());
    DASSERT (texunit < max_texunit);
    DASSERT (id == 0 || valid_texture (this));  // allow unloaded texture binds
    glActiveTextureARB (GL_TEXTURE0_ARB + texunit);
    glBindTexture (GL_TEXTURE_RECTANGLE_NV, id);
    DASSERT (gpu_test_opengl_error());
}



void
GpuTexture::readback (void *buf, int w, int h, GLenum glformat,
                      GLenum gltype) const
{
    int texunit = max_texunit + 1;              // bind to unused texture unit

    GpuCreationMakeCurrent();
    DASSERT (gpu_test_opengl_error());
    glActiveTextureARB (GL_TEXTURE0_ARB + texunit);
    glBindTexture (GL_TEXTURE_RECTANGLE_NV, id);

    GLint width, height;
    glGetTexLevelParameteriv (GL_TEXTURE_RECTANGLE_NV, 0,
        GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv (GL_TEXTURE_RECTANGLE_NV, 0,
        GL_TEXTURE_HEIGHT, &height);
    DASSERT (width == w && height == h);
    
    GLint internal_format;
    glGetTexLevelParameteriv (GL_TEXTURE_RECTANGLE_NV, 0,
        GL_TEXTURE_INTERNAL_FORMAT, (GLint *)&internal_format);
    DASSERT (glformat == internal_format);
    DASSERT (gpu_test_opengl_error());
    
    int nchannels;
    bool alpha = false, depth = false;
    switch (glformat) {
    case GL_RGBA:
        alpha = true;
        nchannels = 4;
        break;

    case GL_FLOAT_RGBA_NV:
        alpha = true;
        nchannels = 4;
        glformat = GL_RGBA;
        break;
        
    case GL_FLOAT_R_NV:
        nchannels = 1;
        break;

    case GL_ALPHA:
    case GL_LUMINANCE:
    case GL_RED:
        alpha = true;
        nchannels = 1;
        break;

    case GL_RGB:
        nchannels = 3;
        break;

    case GL_DEPTH_COMPONENT:
        nchannels = 1;
        depth = true;

    default:
        fprintf (stderr, "format = %x  %d\n", glformat, glformat);
        abort();
        break;
    }

    // readback the data 
    glGetTexImage (GL_TEXTURE_RECTANGLE_NV, 0, glformat, gltype, buf);
    glBindTexture (GL_TEXTURE_RECTANGLE_NV, 0);

    DASSERT (gpu_test_opengl_error());
    GpuCreationRelease();
}
