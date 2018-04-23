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
#include "fmath.h"
#include "dassert.h"

// Include gluErrorString() after other GL headers for extensions.
#include <GL/glu.h>



bool
gpu_test_opengl_error ()
{
    GLuint errogl = glGetError ();
    if (errogl != GL_NO_ERROR) {
        const char *errstr = (const char *)gluErrorString (errogl);
        std::cerr << "OpenGL Error: " << errstr << "\n";
#ifndef WINNT
	// FIXME_WINNT: Don't die on gl errors just yet
        return false;
#endif
    }

    return true;
}


static unsigned long long
update_id ()
{
    static unsigned long long globalid = 0;
    unsigned long long id = ++globalid;
    return id;
}



static unsigned long long
update_texunitid ()
{
    static unsigned long long texunitid = 0;
    return ++texunitid;
}



GpuOcclusionQuery::GpuOcclusionQuery (const char *name)
    : name (name), id (0)
{}



bool
GpuOcclusionQuery::finished () const
{
    DASSERT (gpu_test_opengl_error());
    DASSERT (glIsOcclusionQueryNV (id));
    GLuint done;
    glGetOcclusionQueryuivNV (id, GL_PIXEL_COUNT_AVAILABLE_NV, &done);
    DASSERT (gpu_test_opengl_error());
    return (bool)done;
}



GLuint
GpuOcclusionQuery::count ()
{
    DASSERT (gpu_test_opengl_error());
    DASSERT (glIsOcclusionQueryNV (id));
    GLuint count;
    glGetOcclusionQueryuivNV (id, GL_PIXEL_COUNT_NV, &count);
    DASSERT (gpu_test_opengl_error());
    return count;
}



GpuOcclusionQuery::~GpuOcclusionQuery ()
{
    DASSERT (gpu_test_opengl_error());
    if (id != 0) {
        DASSERT (glIsOcclusionQueryNV (id));
        glDeleteOcclusionQueriesNV (1, &id);
    }
    DASSERT (gpu_test_opengl_error());
}



GpuDrawmode::GpuDrawmode () 
    : drawmodeid (0), viewid (0), texunitid (0),
      width (0), height (0), cullmode (CULL_OFF), zbufmode (ZBUF_OFF),
      drawbuffermode (BUF_NONE), logicopmode (LOGIC_OFF),
      fp (NULL), vp (NULL), errstr (NULL), errogl (GL_NO_ERROR)
{
    c2s = Matrix4::Ident();
    w2c = Matrix4::Ident();
    drawmodeid = update_id();
    viewid = drawmodeid;
    for (int i = 0; i < GpuTexture::max_texunit; i++) {
        texunit[i] = NULL;
    }
}



GpuDrawmode::~GpuDrawmode ()
{
}


// Compare this drawmode to the OpenGL state.  Return NULL if it maches,
// or else a string describing the first difference found.
const char *
GpuDrawmode::opengl_diff_error () const
{
    GLboolean b;
    GLint i;
    static char buf[2048];

    DASSERT (gpu_test_opengl_error());
    
    // Check cull mode
    glGetBooleanv (GL_CULL_FACE, &b);
    if (b) {
        if (cullmode == CULL_OFF)
            return "cull is on";
        glGetIntegerv (GL_FRONT_FACE, &i);
        if (i == GL_CW) {
            if (cullmode != CULL_CW)
                return "cull is cw";
        } else if (i == GL_CCW) {
            if (cullmode != CULL_CCW)
                return "cull is ccw";
        } else {
            sprintf (buf, "cull direction %d unknown", i);
            return buf;
        }
    } else if (cullmode != CULL_OFF)
        return "cull is off";

    // Check zbuffer
    glGetBooleanv (GL_DEPTH_TEST, &b);
    if (b) {
        if (zbufmode == ZBUF_OFF)
            return "zbuf is on";
        glGetIntegerv (GL_DEPTH_FUNC, &i);
        if (i == GL_LESS) {
            if (zbufmode != ZBUF_LESS)
                return "zbuf is less";
        } else if (i == GL_ALWAYS) {
            if (zbufmode != ZBUF_ALWAYS)
                return "zbuf is always";
        } else if (i == GL_EQUAL) {
            if (zbufmode != ZBUF_EQUAL)
                return "zbuf is equal";
        } else if (i == GL_LEQUAL) {
            if (zbufmode != ZBUF_LESS_OR_EQUAL)
                return "zbuf is less or equal";
        } else if (i == GL_GREATER) {
            if (zbufmode != ZBUF_GREATER)
                return "zbuf is greater";
        } else if (i == GL_GEQUAL) {
            if (zbufmode != ZBUF_GREATER_OR_EQUAL)
                return "zbuf is greater or equal";
        } else {
            sprintf (buf, "zbuf mode %d unknown", i);
            return buf;
        }
    } else if (zbufmode != ZBUF_OFF)
        return "zbuf is off";
    
    // Check drawmode
    glGetIntegerv (GL_DRAW_BUFFER, &i);
    if (i == GL_FRONT) {
        if (drawbuffermode != BUF_FRONT && drawbuffermode != BUF_NONE)
            return "draw buffer is front";
    } else if (i == GL_BACK) {
        if (drawbuffermode != BUF_BACK && drawbuffermode != BUF_NONE)
            return "draw buffer is back";
    }
    
    // logic operation
    glGetBooleanv (GL_COLOR_LOGIC_OP, &b);
    if (b) {
        if (logicopmode == LOGIC_OFF)
            return "color logic operation enabled";
        glGetIntegerv (GL_LOGIC_OP_MODE, &i);
        if (i != GL_XOR) {
            sprintf (buf, "color logic operation mode %d unknown", i);
            return buf;
        }
    } else {
        if (logicopmode == LOGIC_XOR)
            return "color logic operation is disabled";
    }

#if 0
    for (int j = 0; j < max_texunit; j++) {
//    glEnable (GL_TEXTURE_2D);
    DASSERT (gpu_test_opengl_error());
    
        glActiveTextureARB (GL_TEXTURE0_ARB + j);
//    glEnable (GL_TEXTURE_2D);
    DASSERT (gpu_test_opengl_error());
    
        if (texunit[j] != NULL) {
            glGetBooleanv (GL_TEXTURE_RECTANGLE_NV, &b);
            if (!b)
                return "2D textures not enabled";
        }
        
        glGetIntegerv (GL_TEXTURE_BINDING_RECTANGLE_NV, &i);
        if (texunit[j] == NULL) {
            if (i != 0) {
                sprintf (buf, "texture %d should be off, but is using "
                    "texture %d", j, i);
                return buf;
            }
        } else {
#ifdef DEBUG            
            extern bool valid_texture (const GpuTexture *tex);
            DASSERT (valid_texture (texunit[j]));
#endif                
            if (!texunit[j]->validate(i)) {
                sprintf (buf, "texture %d using texture %d", j, i);
                return buf;
            }
        }
    }
    
#endif
    // Note: it would be studly to check every glGet value... incl. extensions

    DASSERT (gpu_test_opengl_error());
    
    return NULL;
}



bool
GpuDrawmode::validate () const
{
    // Compare this drawmode to the OpenGL state and report the error
    errstr = opengl_diff_error ();
    if (errstr != NULL)
        return false;
        
    // Check the OpenGL error status and use GLU to get an error string
    // Safer to do this opengl check after making the context current above?
    errogl = glGetError ();
    if (errogl != GL_NO_ERROR) {
        errstr = (const char *)gluErrorString (errogl);
        return false;
    }

    return true;
}



ostream&
operator<< (ostream& out, const GpuDrawmode& a)
{
    static char *indent = "  ";
    out << "drawmode id: " << a.drawmodeid << "\n";
    out << indent << "c2s: " << a.c2s << "\n";
    out << indent << "w2c: " << a.w2c << "\n";
    out << indent << "cull: ";
    switch (a.cullmode) {
    case GpuDrawmode::CULL_OFF: out << "off"; break;
    case GpuDrawmode::CULL_CW: out << "cw"; break;
    case GpuDrawmode::CULL_CCW: out << "ccw"; break;
    }
    out << "\n";
    out << indent << "zbuffer: ";
    switch (a.zbufmode) {
    case GpuDrawmode::ZBUF_OFF: out << "off"; break;
    case GpuDrawmode::ZBUF_ALWAYS: out << "always"; break;
    case GpuDrawmode::ZBUF_LESS: out << "less"; break;
    case GpuDrawmode::ZBUF_EQUAL: out << "equal"; break;
    case GpuDrawmode::ZBUF_LESS_OR_EQUAL: out << "less or equal"; break;
    case GpuDrawmode::ZBUF_GREATER: out << "greater"; break;
    case GpuDrawmode::ZBUF_GREATER_OR_EQUAL: out << "greater or equal"; break;
    }
    out << "\n";
    out << indent << "draw buffer: ";
    switch (a.drawbuffermode) {
    case GpuDrawmode::BUF_FRONT: out << "front"; break;
    case GpuDrawmode::BUF_BACK: out << "back"; break;
    case GpuDrawmode::BUF_NONE: out << "unitialized"; break;
    }
    out << "\n";
    out << indent << "frag prog: ";
    if (a.fp == NULL)
        out << "none";
    else
        out << *a.fp;
    out << "\n";
    out << indent << "vert prog: ";
    if (a.vp == NULL)
        out << "none";
    else
        out << *a.vp;
    out << "\n";
    for (int i = 0; i < GpuTexture::max_texunit; i++) {
        out << indent << "texture[" << i << "]: ";
        if (a.texunit[i] != NULL)
            out << *a.texunit[i];
        else
            out << "none";
        out << "\n";
    }

    return out << "\n";
}



void
GpuDrawmode::reset ()
{
    GpuDrawmode tmp;            // create a new mode with default settings
    *this = tmp;                // copy assignment
}



void
GpuDrawmode::view (const Matrix4 &c2s, int width, int height)
{
    DASSERT (width > 0);
    DASSERT (height > 0);
    
    bool changed = false;
    
    if (this->c2s != c2s) {
        this->c2s = c2s;
        changed = true;
    }

    if (width != this->width || height != this->height) {
        this->width = width;
        this->height = height;
        changed = true;
    }

    if (changed) {
        drawmodeid = update_id();
        viewid = drawmodeid;
    }
}



void
GpuDrawmode::view (int width, int height)
{
    DASSERT (width > 0);
    DASSERT (height > 0);

    Matrix4 c2s = Matrix4::OrthoMatrix (0, float(width), 0, float(height), -1, 1);
    if (width != this->width || height != this->height ||
        c2s != this->c2s) {
        this->width = width;
        this->height = height;
        this->c2s = c2s;
        drawmodeid = update_id();
        viewid = drawmodeid;
        GpuOGL::pixel_view_stat++;
    }
}



void
GpuDrawmode::model (const Matrix4 &w2c)
{
    if (this->w2c != w2c) {
        this->w2c = w2c;
        drawmodeid = update_id();
        viewid = drawmodeid;
    }
}



const GpuTexture *
GpuDrawmode::texture (int index, const GpuTexture &texture)
{
    DASSERT (index >= 0 && index < GpuTexture::max_texunit);
    const GpuTexture *oldtexture = texunit[index];
    drawmodeid = update_id();
    texunitid = update_texunitid();
    this->texunit[index] = &texture;
    return oldtexture;
}



void
GpuDrawmode::texture_clear (int index)
{
    bool update = false;
    
    if (index == -1) {
        for (int i = 0; i < GpuTexture::max_texunit; i++) {
            if (texunit[i] != NULL)
                update = true;
            texunit[i] = NULL;
        }
    } else {
        update = texunit[index] != NULL;
        texunit[index] = NULL;
    }
    if (update)
        texunitid = update_texunitid();
}



GpuFragmentProgram *
GpuDrawmode::fragment_program (GpuFragmentProgram *program)
{
    GpuFragmentProgram *oldfp = fp;
    if (fp != program) {
        drawmodeid = update_id();
        fp = program;
    }
    return oldfp;
}



GpuVertexProgram *
GpuDrawmode::vertex_program (GpuVertexProgram *program)
{
    GpuVertexProgram *oldvp = vp;
    if (vp != program) {
        drawmodeid = update_id();
        vp = program;
    }
    return oldvp;
}



GpuDrawmode::GpuCull
GpuDrawmode::cull (GpuCull state)
{
    GpuCull oldstate = cullmode;
    if (state != cullmode) {
        drawmodeid = update_id();
        cullmode = state;
    }
    return oldstate;
}



GpuDrawmode::GpuZbuffer
GpuDrawmode::zbuffer (GpuZbuffer state)
{
    GpuZbuffer oldstate = zbufmode;
    if (state != zbufmode) {
        drawmodeid = update_id();
        zbufmode = state;
    }
    return oldstate;
}

GpuDrawmode::GpuDrawBuffer
GpuDrawmode::drawbuffer (GpuDrawBuffer state)
{
    GpuDrawBuffer olddrawbuffermode = drawbuffermode;
    if (state != drawbuffermode) {
        drawmodeid = update_id();
        drawbuffermode = state;
    }
    return olddrawbuffermode;
}



GpuDrawmode::GpuLogicOp
GpuDrawmode::logicop (GpuLogicOp state)
{
    GpuLogicOp oldstate = logicopmode;
    if (logicopmode != state) {
        drawmodeid = update_id();
        logicopmode = state;
    }
    return oldstate;
}



// Update the canvas' drawmode and set all non-matching OpenGL state
void
GpuCanvas::update (const GpuDrawmode &drawmode)
{
    DASSERT (drawmode.width > 0);
    DASSERT (drawmode.height > 0);

    // start by making this context current
    make_current ();

    DASSERT (gpu_test_opengl_error());
    GpuOGL::drawmode_update_stat++;
    
    // quick check to see if entire state already matches
    if (curdrawmode == drawmode) {
#ifdef DEBUG
        if (!drawmode.validate())
            fprintf (stderr, "Gpu ERROR: %s\n", drawmode.error_string());
#endif    
        DASSERT (gpu_test_opengl_error());
        GpuOGL::drawmode_update_match_stat++;
        return;
    }

    // check special view state to avoid using memcmp to check matrices
    if (curdrawmode.viewid != drawmode.viewid) {
        curdrawmode.viewid = drawmode.viewid;
        curdrawmode.c2s = drawmode.c2s;
        curdrawmode.w2c = drawmode.w2c;
        curdrawmode.width = drawmode.width;
        curdrawmode.height = drawmode.height;
        glViewport (0,0, drawmode.width, drawmode.height);
        glMatrixMode (GL_PROJECTION);
        glLoadMatrixf ((const GLfloat *)drawmode.c2s.data());
        glMatrixMode (GL_MODELVIEW);
        glLoadMatrixf ((const GLfloat *)drawmode.w2c.data());
        GpuOGL::drawmode_update_view_stat++;
    }

    if (curdrawmode.cullmode != drawmode.cullmode) {
        curdrawmode.cullmode = drawmode.cullmode;
        switch (drawmode.cullmode) {
        case GpuDrawmode::CULL_OFF:
            glDisable (GL_CULL_FACE);
            break;
        case GpuDrawmode::CULL_CW:
            glEnable (GL_CULL_FACE);
            glFrontFace (GL_CW);
            break;
        case GpuDrawmode::CULL_CCW:
            glEnable (GL_CULL_FACE);
            glFrontFace (GL_CCW);
            break;
        }
        GpuOGL::drawmode_update_cull_stat++;
    }

    if (curdrawmode.zbufmode != drawmode.zbufmode) {
        curdrawmode.zbufmode = drawmode.zbufmode;
        switch (drawmode.zbufmode) {
        case GpuDrawmode::ZBUF_OFF:
            glDisable (GL_DEPTH_TEST);
            break;
        case GpuDrawmode::ZBUF_LESS:
            glEnable (GL_DEPTH_TEST);
            glDepthFunc (GL_LESS);
            break;
        case GpuDrawmode::ZBUF_ALWAYS:
            glEnable (GL_DEPTH_TEST);
            glDepthFunc (GL_ALWAYS);
            break;
        case GpuDrawmode::ZBUF_EQUAL:
            glEnable (GL_DEPTH_TEST);
            glDepthFunc (GL_EQUAL);
            break;
        case GpuDrawmode::ZBUF_LESS_OR_EQUAL:
            glEnable (GL_DEPTH_TEST);
            glDepthFunc (GL_LEQUAL);
            break;
        case GpuDrawmode::ZBUF_GREATER:
            glEnable (GL_DEPTH_TEST);
            glDepthFunc (GL_GREATER);
            break;
        case GpuDrawmode::ZBUF_GREATER_OR_EQUAL:
            glEnable (GL_DEPTH_TEST);
            glDepthFunc (GL_GEQUAL);
            break;
        }
        GpuOGL::drawmode_update_zbuf_stat++;
    }

    if (curdrawmode.drawbuffermode != drawmode.drawbuffermode) {
        curdrawmode.drawbuffermode = drawmode.drawbuffermode;
        switch (drawmode.drawbuffermode) {
        case GpuDrawmode::BUF_FRONT:
            glDrawBuffer (GL_FRONT);
            break;
        case GpuDrawmode::BUF_BACK:
            glDrawBuffer (GL_BACK);
            break;
        case GpuDrawmode::BUF_NONE:
            /*EMPTY*/
            break;
        }
        GpuOGL::drawmode_update_drawbuf_stat++;
    }

    if (curdrawmode.logicopmode != drawmode.logicopmode) {
        curdrawmode.logicopmode = drawmode.logicopmode;
        if (drawmode.logicopmode == GpuDrawmode::LOGIC_OFF) {
            glDisable (GL_COLOR_LOGIC_OP);
            glLogicOp (GL_COPY);
        } else {
            DASSERT (drawmode.logicopmode == GpuDrawmode::LOGIC_XOR);
            glEnable (GL_COLOR_LOGIC_OP);
            glLogicOp (GL_XOR);
        }
        GpuOGL::drawmode_update_logicop_stat++;
    }

    if (curdrawmode.fp != drawmode.fp) {
        curdrawmode.fp = drawmode.fp;
        if (drawmode.fp == NULL) 
            glDisable (GL_FRAGMENT_PROGRAM_NV);
        else
            drawmode.fp->bind ();
        GpuOGL::drawmode_update_fp_stat++;
    }

    // update program parameters if needed
    if (curdrawmode.fp != NULL)
        curdrawmode.fp->update();
    
    if (curdrawmode.vp != drawmode.vp) {
        curdrawmode.vp = drawmode.vp;
        if (drawmode.vp == NULL) 
            glDisable (GL_VERTEX_PROGRAM_NV);
        else
            drawmode.vp->bind ();
        GpuOGL::drawmode_update_vp_stat++;
    }

    // update program parameters if needed
    if (curdrawmode.vp != NULL)
        curdrawmode.vp->update();

    if (curdrawmode.texunitid != drawmode.texunitid) {
        curdrawmode.texunitid = drawmode.texunitid;
        for (int i = 0; i < GpuTexture::max_texunit; i++) {
            // can't checked cached pointers since they don't work if the
            // texture was destroyed and recreated in the same memory block,
            // so instead use the texture operator==
            if (curdrawmode.texunit[i] == NULL && drawmode.texunit[i] == NULL)
                continue;
        
            curdrawmode.texunit[i] = drawmode.texunit[i];
        
            if (drawmode.texunit[i] != NULL) {
                drawmode.texunit[i]->bind (i);
            } else {
                glActiveTextureARB (GL_TEXTURE0_ARB + i);
                glBindTexture (GL_TEXTURE_RECTANGLE_NV, 0);
            }
        }
        GpuOGL::drawmode_update_texture_stat++;
    }
    
    // clear the screen if bits are set
    update_clear();
    
#ifdef DEBUG
    if (!drawmode.validate())
        fprintf (stderr, "Gpu ERROR: %s\n", drawmode.error_string());
#endif    
    DASSERT (gpu_test_opengl_error());
}


