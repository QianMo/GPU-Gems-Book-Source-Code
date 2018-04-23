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



bool
GpuProgram::collides_tracked_matrix (int index)
{
#ifdef DEBUG
    int min = std::max (0, index - 3);
    for (int i = min; i < index; i++)
        if (param[i].ismatrix())
            return true;
#endif
    return false;
}



void
GpuProgram::parameter (int index, Vector4 v)
{
    DASSERT (index < max_param);
    DASSERT (!collides_tracked_matrix (index));
    if (param[index].set (v))
        parameters_updated = false;
}



void
GpuProgram::parameter (int index, GLenum matrix, GLenum transform)
{
    DASSERT (index >= 0 && index < max_param);
    DASSERT (type == GL_VERTEX_PROGRAM_NV);
    DASSERT (!collides_tracked_matrix (index));
    if (param[index].set (matrix, transform))
        parameters_updated = false;
    
//    for (int j = index+1; j < index+4; j++)             // clear next three
//        param[i][j].set (GL_NONE, GL_IDENTITY_NV);
//    fprintf (stderr, "Tracked %p i=%d index=%d  matrix=%ud  transform=%ud\n",
//        this, i, index, matrix, transform);
}



void
GpuProgram::parameter (int index, const GpuProgramParameter &param)
{
    DASSERT (index >= 0 && index < max_param);

    // FIXME: This shouldn't be necessary, but somehow the test below
    // is failing, and we need to make sure that the parameters are
    // actually updated.
    parameters_updated = false;
    
    if (this->param[index] != param) {
        this->param[index] = param;
        parameters_updated = false;
    }
}



GpuProgram::GpuProgram (GLenum type, const char *name, const char *code) 
    : type (type), name (name), code (NULL), id (0),
      downloaded (false), parameters_updated (false)
    
{
    load (code);
}



GpuProgram::~GpuProgram ()
{
    unload ();          // frees the code string
    name = "deleted";
}



void
GpuProgram::download ()
{
    if (downloaded)
        return;
    
    // clear the flag
    downloaded = true;
    
    if (code == NULL) {
        unload ();
        parameters_updated = false;
        return;
    }

    GpuCreationMakeCurrent();
    DASSERT (gpu_test_opengl_error());

    // generate a new id if first time or just deleted
    if (id == 0)
        glGenProgramsNV (1, &id);

    // download the code to the gpu
    glLoadProgramNV (type, id, (GLsizei)strlen (code), (const GLubyte *)code);

    parameters_updated = false;

#ifdef DEBUG
    GLint pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_NV, &pos);

    if (pos >= 0) {
        if (pos == (GLint) strlen (code))
            fprintf (stderr, "Semantic error in program:\n%s\n", code);
        else
            fprintf (stderr, "Syntax error at:\n%s\n", code + pos);
    }
#endif
    DASSERT (gpu_test_opengl_error());
    GpuCreationRelease();
}



void
GpuProgram::load (const char *code)
{
    // set the flag so we know to update later
    downloaded = false;
    
    if (this->code) {
        free (this->code);
        this->code = NULL;
    }
    
    if (code == NULL)  // leave program id valid so we unload later
        return;

    // copy the code for downloading later, when we update the canvas
    this->code = strdup (code);
}



void
GpuProgram::unload ()
{
    if (id) {
        GpuCreationMakeCurrent();
        DASSERT (gpu_test_opengl_error());
        glDeleteProgramsNV (1, &id);
        DASSERT (gpu_test_opengl_error());
        GpuCreationRelease();
        id = 0;
    }
}



bool
GpuProgram::match (const GpuProgram &p) const
{
    // compare program code
    if (strcmp (p.code, code) != 0) {
        fprintf (stderr, "Programs \"%s\" and \"%s\" don't match:\n%s\n\n%s\n",
            name, p.name, code, p.code);
        return false;
    }

    // compare program parameters
    for (int i = 0; i < max_param; ++i) {
        if (param[i] != p.param[i]) {
            fprintf (stderr, "Programs \"%s\" and \"%s\" have different "
                "parameter %d values\n", name, p.name, i);
            return false;
        }
    }

//    fprintf (stderr, "Programs \"%s\" and \"%s\" match:\n%s\n\n%s\n",
//        name, p.name, code, p.code);

    return true;
}
        

bool
GpuProgramParameter::set (Vector4 &v)
{
    if (matrix != GL_NONE || transform != GL_IDENTITY_NV || val != v) {
        matrix = GL_NONE;
        transform = GL_IDENTITY_NV;
        val = v;
        return true;
    }
    return false;
}



bool
GpuProgramParameter::set (GLenum m, GLenum t)
{
    if (matrix != m || transform != t || val != UNINITIALIZED_FLOAT) {
        matrix = m;
        transform = t;
        val = UNINITIALIZED_FLOAT;
        return true;
    }
    return false;
}



void
GpuProgramParameter::update (GLenum type, int index)
{
    if (matrix != GL_NONE) {
        DASSERT (type == GL_VERTEX_PROGRAM_NV);
        glTrackMatrixNV (type, index, matrix, transform);
    } else if (!uninitialized(val[0])) {
        if (type == GL_VERTEX_PROGRAM_NV) 
            glProgramParameter4fvNV (type, index, val.data());
        else if (type == GL_FRAGMENT_PROGRAM_NV) 
            glProgramLocalParameter4fvARB (type, index, val.data());
    }
}



void
GpuProgram::bind ()
{
    // update the program if needed
    download ();
    
    DASSERT (id != 0);
    DASSERT (gpu_test_opengl_error());

    // enable and bind the program to the proper program type 
    glEnable (type);
    glBindProgramNV (type, id);

    DASSERT (gpu_test_opengl_error());
}



// update the program parameters if they are out-of-date
void
GpuProgram::update ()
{
    if (parameters_updated)
        return;
    
    parameters_updated = true;
    GpuOGL::drawmode_update_prog_param_stat++;
    
    DASSERT (gpu_test_opengl_error());
    for (int i = 0; i < max_param; i++) {
        param[i].update (type, i);
        DASSERT (gpu_test_opengl_error());
    }
}



bool
GpuProgram::find (const char *str) const
{
    return strstr (code, str) != NULL;
}
