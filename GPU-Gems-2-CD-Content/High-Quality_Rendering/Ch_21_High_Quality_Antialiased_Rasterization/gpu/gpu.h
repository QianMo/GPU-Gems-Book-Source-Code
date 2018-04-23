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

#ifndef GPU_H
#define GPU_H

/*
  Overview
  --------
  The primary Gpu drawing classes are the GpuDrawmode state vector,
  GpuPrimitive for actual drawing commands, GpuTexture for creating
  and loading textures, GpuProgram for creating and loading vertex
  and fragment programs, and GpuOcclusionQuery for creating and using
  occlusion queries.  The GpuCanvas, GpuPBuffer, GpuWindow and other
  classes are used to create drawing surfaces.

  The GpuDrawmode class contains all OpenGL state information.  Each
  GpuCanvas contains a GpuDrawmode state vector which describes the
  current OpenGL state for that GLX context.  You can set all the
  standard things in the GpuDrawmode such as culling state, zbuffer
  function, current fragment program, etc.

  Drawing is done by creating a GpuPrimitive, setting any texture
  coordinate vertex arrays, and then rendering the primitive into a
  canvas using a drawmode. Note that rectangles and bboxes do not use
  their texture coordinate vertex arrays.

  Before doing any actual drawing, the canvas's state is
  updated to match the passed in drawmode right before drawing.  The
  important point here is that the OpenGL state is not set until just
  before drawing and not when the various GpuDrawmode state setting
  functions are called.

  A number of features are provided to help debug OpenGL problems:
    - In debug mode, gl errors are checked in every call
    - You can print the entire drawmode state vector
    - You can validate the drawmode state vector against the real OpenGL state
    - All OpenGL state is set in GpuCanvas::update()
    - All textures and programs are named for easy identification
    - A convenience function (gpu_test_opengl_error) that is good for asserts
    - Textures are validated in debug mode


  The GPU library requires that the application insure that the same
  GpuCanvas is not used in two separate threads simultaneously.  If
  this happens, the program will exit when GL reports an error while
  trying to make the canvas current.

  The GPU library always shares the objects between all contexts.
  This implies that it doesn't matter which context is active when
  creating or destroying objects, however, the GL requires that *some*
  canvas is current for these operations.  The library uses an
  internal 1x1 canvas to create objects such as textures, pbuffers,
  and programs, but only if no other context is active.

  The application should never call GpuCanvas::make_current if it uses
  the GPU library properly.  If the application does its own gl
  rendering outside the GPU library, it may need to call make_current
  and release.

  The application should only call GpuCanvas::release if it wants to
  free up a GpuCanvas.

  Occlusion queries require that the same canvas is current throughout
  all rendering operations.  It is up to the user to pass the same
  context for all primitive rendering commands while a query is
  active, however the library will check to make sure that this is the
  case in DEBUG mode.

    
  Examples
  --------
  
  Simplest way to draw a 2D rectangle:
  
    GpuPBuffer pbuffer (256, 256);
    GpuCanvas canvas (pbuffer);
    GpuDrawmode drawmode ();
    GpuPrimitive rect (xmin, xmax, ymin, ymax);
    rect.render (drawmode, canvas);

    
  Create a 2x2 texture and bind it to texture unit 0:

    GpuTexture texture ("my texture");
    Vector3 color[4] = {{1,0,0}, {0,1,0}, {0,0,1}, {1,1,1}};
    texture.load (&color, 2, 2);
    drawmode.texture (0, &texture);

    
  Make the drawmode 3D drawing mode and create and draw a quadmesh:

    Matrix4 c2s = Matrix4::PerspectiveMatrix (45, 1, 0.01, 10000);
    drawmode.view (&c2s, 256, 256);
    Vector3 P[4] = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}};
    GpuQuadmesh quadmesh (2, 2, P)
    Vector3 texcoord[4] = {{1,0,0}, {0,1,0}, {0,0,1}, {1,1,1}};
    quadmesh.texcoord (0, texcoord);
    quadmesh.render (drawmode, canvas);


  Creating and using a fragment program with a constant parameter:
  
    const char *fp10 = "!!FP1.0\nMOVR o[COLR], p[0];\nEND\n";
    GpuFragmentProgram fp ("red", fp10);
    fp.parameter (GL_FRAGMENT_PROGRAM_NV, 0, Vector4(1,0,0,0));
    drawmode.fragment_program (&fp);
    
    
  An example of how to do an occlusion query with multiple draw statements:

    GpuOcclusionQuery oq ("depth peel");
    canvas.begin_occlusion_query (oq);
    quadmesh.render (drawmode, canvas);
    ...
    quadmesh.render (drawmode, canvas);
    canvas.end_occlusion_query (oq);
    ... < do something to hide latency > ...
    printf ("occlusion query had %d visible fragments\n", oq.count());
    
  
  An example of how copy-from-fb-to-texture works:
  
    GpuTexture fromfb ("rendered texture");
    fromfb.load (canvas, 0, 0, 256, 256);
    drawmode.texture (0, &fromfb);

*/

// Define internally used preprocessor macros from platform variables
#ifdef LINUX
#define XWINDOWS
#endif

#ifdef MACOSX
#define XWINDOWS
#endif

// Include platform specific headers
#ifdef XWINDOWS
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glx.h>
#define GpuContext GLXContext
#endif

#ifdef WINNT
#include <unistd.h>
#include <gl/gl.h>
#include <gl/glext.h>
#include <gl/wglext.h>
#include <glh/glh_extensions.h>
#define GpuContext HGLRC
#pragma warning (disable : 4786)
#endif

#ifdef CARBON
#include <HIToolbox/MacWindows.h>
#include <HIToolbox/Controls.h>
#include <HIToolbox/ControlDefinitions.h>
#include <CarbonCore/ToolUtils.h>
#define GpuContext AGLContext
#endif

#include <iostream>
using std::ostream;

#include "hash.h"
#include "fmath.h"
#include "vecmat.h"
#include "color.h"
#include "dassert.h"
#include "peakcounter.h"

using namespace Gelato;


class GpuCanvas;


// NOTE: All name strings are assumed to be allocated by the caller and
//       live for the lifetime of the named object to avoid extra allocation

// Create and allocate a texture on the GPU using either passed in
// data or directly from the current framebuffer using a copy or
// by locking the framebuffer with render-to-texture commands.
//
// The texture must be bound to the drawing mode before use.

class GpuTexture {
 public:
    // Construct an invalid texture, must be loaded first
    GpuTexture (const char *name);

    // Delete the texture and any associated Gpu memory
    ~GpuTexture ();

    // Construct a texture from 3-channel floating point data as a Vector3
    void load (const Vector3 *color, int w, int h,
               int xmin=0, int xmax=-1, int ymin=0, int ymax=-1,
               int txoff=0, int tyoff=0);
    
    // Construct a texture from 3-channel floating point data as a Color3
    void load (const Color3 *color, int w, int h,
               int xmin=0, int xmax=-1, int ymin=0, int ymax=-1,
               int txoff=0, int tyoff=0) {
        load ((const Vector3 *)color, w, h, xmin,xmax, ymin,ymax, txoff,tyoff);
    }
    
    // Construct a texture from 1-channel float point data
    void load (const float *data, int w, int h,
               int xmin=0, int xmax=-1, int ymin=0, int ymax=-1,
               int txoff=0, int tyoff=0);

    // Construct texture from the existing framebuffer by copy.
    // If depth_compare_op != 0, bitdepth is ignored and forced to 24bit depth
    void load (GpuCanvas &canvas, int x, int y, int w, int h,
               int bitdepth, int depth_compare_op=0, bool pad_border=false,
               int txoff=-1, int tyoff=-1);

    // free any associated gpu memory
    void unload ();
    
    // bind this texture to the given GL texture unit
    // (only GpuCanvas::update() should use this)
    void bind (int texunit) const;

    // readback the image data from the GPU
    void readback (void *buf, int w, int h, GLenum glformat, 
                   GLenum gltype) const;

    int w() const { return width; }
    int h() const { return height; }
            
    bool validate (GLuint id) const { return id == this->id; }
    
    friend bool operator== (const GpuTexture &a, const GpuTexture &b) {
        return a.textureid == b.textureid;
    }

    friend bool operator!= (const GpuTexture &a, const GpuTexture &b) {
        return a.textureid != b.textureid;
    }

    friend ostream& operator<< (ostream& out, const GpuTexture& a) {
        return out << a.name;
    }

    static const int max_texunit = 9;
    
 private:
    const char *name;
    GLuint id;                          // OpenGL id
    unsigned long long textureid;       // unique id for tracking changes
    int width, height;
};



// For internal use by GpuProgram, only.
// Program parameters are used to set program constants and tracked matrices
class GpuProgramParameter {
 public:
    // default constructor initialize list 
    GpuProgramParameter ()
        : matrix (GL_NONE), transform (GL_IDENTITY_NV),
        val (UNINITIALIZED_FLOAT) { }

    // set a constant local parameter at sp, returns true if changed
    bool set (Vector4 &v);

    // track a matrix into local parameters, returns true if changed
    bool set (GLenum matrix, GLenum transform);

    // set the parameter (or skip) in the vertex program state
    void update (GLenum type, int index);

    friend bool operator== (const GpuProgramParameter &a,
        const GpuProgramParameter &b) {
        return a.matrix == b.matrix && a.transform == b.transform &&
            a.val == b.val;
    }
    
    friend bool operator!= (const GpuProgramParameter &a,
        const GpuProgramParameter &b) {
        return a.matrix != b.matrix || a.transform != b.transform ||
            a.val != b.val;
    }
    
    bool ismatrix () const { return matrix != GL_NONE; }
            
 private:
    GLenum matrix;                      // GL_NONE == no tracking
    GLenum transform;
    Vector4 val;                        // UNINITIALIZED_FLOAT for invalid
};



// Create and load a fragment program on the GPU.  The fragment program
// must be bound to the current drawing mode before use.

class GpuProgram {
 public:
    GpuProgram (GLenum type, const char *name, const char *code=NULL);
    virtual ~GpuProgram ();

    void load (const char *code);
    
    // Set a constant program parameter
    void parameter (int index, Vector4 v);

    // Track a matrix into 4 consecutive parameters (error checked)
    // Note: only works with vertex programs
    //
    // matrix should be one of:
    //    GL_NONE, GL_MODELVIEW, GL_PROJECTION, GL_TEXTURE,
    //    GL_TEXTUREi_ARB, GL_MODELVIEW_PROJECTION_NV, or GL_MATRIXi_NV
    //
    // transform should be one of:
    //    GL_IDENTITY_NV, GL_INVERSE_NV,
    //    GL_TRANSPOSE_NV, GL_INVERSE_TRANSPOSE_NV
    void parameter (int index, GLenum matrix, GLenum transform=GL_IDENTITY_NV);

    // Copy assignment
    void parameter (int index, const GpuProgramParameter &param);
    
    friend ostream& operator<< (ostream& out, const GpuProgram &a) {
        return out << a.name;
    }

    // compare program code and parameters, return true if they match
    bool match (const GpuProgram &p) const;

    // returns true if the string exists in the program code (for debugging)
    bool find (const char *str) const;
    
    // returns the name of the program (mostly for debugging)
    const char *title () { return name; }

    // Canvases should be the only thing that calls download and bind
    friend class GpuCanvas;
    
 private:
    GLenum type;                                        // vertex or fragment
    const char *name;
    char *code;
    GLuint id;
    bool downloaded;
    
    static const int max_param = 16;
    GpuProgramParameter param[max_param];
    bool parameters_updated;        // whether current param values sent to GL
    bool collides_tracked_matrix (int index);
    void unload ();
    void download ();
    void update ();                     // update parameters if needed
    void bind ();                       // activate program
};


    
class GpuFragmentProgram : public GpuProgram {
 public:
    GpuFragmentProgram (const char *name, const char *code) :
        GpuProgram (GL_FRAGMENT_PROGRAM_NV, name, code) { }
};



class GpuVertexProgram : public GpuProgram {
 public:
    GpuVertexProgram (const char *name, const char *code) :
        GpuProgram (GL_VERTEX_PROGRAM_NV, name, code) { }
};



class GpuOcclusionQuery {
 public:
    GpuOcclusionQuery (const char *name);
    ~GpuOcclusionQuery ();

    GLuint count ();                    // forces flush first time called
    bool finished () const;             // returns true if done rendering
    
    bool validate (GLuint id) const { return id == id; }
    friend ostream& operator<< (ostream& out, const GpuOcclusionQuery& a) {
        return out << a.name;
    }

    friend class GpuCanvas;
    
 private:
    const char *name;
    GLuint id;
};



// The current drawing mode state vector.  You must create and set
// values in a drawmode before you can draw any primitives.  Each
// GpuCanvas has a current drawmode state vector which is updated
// when primitives are drawn.
//
// The default values are:
//
//   - 2D view that covers the entire canvas
//   - no fragment or vertex programs  (note fragment prog required for fp32!)
//   - culling is off
//   - zbuffering is off
//   - logic op is off

class GpuDrawmode {
 public:
    GpuDrawmode ();
    ~GpuDrawmode ();

    // reset the entire drawmode to the default state
    void reset ();

    // Setup viewing matrices, default is a 2D mode that covers entire canvas
    void view (const Matrix4 &c2s, int width, int height);

    // Setup a 2D orthographic view from (0,0) to (w,h)
    void view (int w, int h);

    // Set the modelview matrix, defaults to identity
    void model (const Matrix4 &w2c);
    
    // Return the current dimensions of this drawmode (could be removed?)
    int w () const { return width; }
    int h () const { return height; }
            
    // Bind the texture to the specified texture unit
    const GpuTexture *texture (int index, const GpuTexture &texture);

    // Return the currently bound texture or NULL if nothing is bound
    const GpuTexture *texture (int index) { return texunit[index]; }

    // Clear a texture unit, or all units by default
    void texture_clear (int index=-1);
    
    // Bind the program and return previously bound value, NULL unbinds
    GpuFragmentProgram *fragment_program (GpuFragmentProgram *program);
    GpuVertexProgram *vertex_program (GpuVertexProgram *program);

    // Return the currently set program
    GpuFragmentProgram *fragment_program () const { return fp; } 
    GpuVertexProgram *vertex_program () const { return vp; }
    
    // Set the culling mode, default is OFF, return previous value
    enum GpuCull { CULL_OFF, CULL_CW, CULL_CCW };
    GpuCull cull (GpuCull state);

    // Set the zbuffer mode, default is OFF, return previous value
    enum GpuZbuffer { ZBUF_OFF, ZBUF_ALWAYS, ZBUF_LESS, ZBUF_EQUAL, 
                      ZBUF_LESS_OR_EQUAL, ZBUF_GREATER, ZBUF_GREATER_OR_EQUAL};
    GpuZbuffer zbuffer (GpuZbuffer state);

    // Set the draw buffer
    enum GpuDrawBuffer { BUF_FRONT, BUF_BACK, BUF_NONE };
    GpuDrawBuffer drawbuffer (GpuDrawBuffer drawbuffermode);

    // Switch logic op, return previous value
    enum GpuLogicOp { LOGIC_OFF, LOGIC_XOR };
    GpuLogicOp logicop (GpuLogicOp state);

    // Compare this state against values returned by OpenGL state queries
    // and check the OpenGL error status.  Check error string for details
    bool validate () const;
    const char *error_string (void) const { return errstr; }
    
    // Compare two drawmodes (fast comparison based on unique id)
    friend bool operator== (const GpuDrawmode &a, const GpuDrawmode &b) {
        return a.drawmodeid == b.drawmodeid;
    }
    
    friend bool operator!= (const GpuDrawmode &a, const GpuDrawmode &b) {
        return a.drawmodeid != b.drawmodeid;
    }
    
    // Print current state for debugging
    friend ostream& operator<< (ostream& out, const GpuDrawmode& a);

    friend class GpuCanvas;                             // allow direct access
    
 private:
    unsigned long long drawmodeid;                      // unique identifier
    unsigned long long viewid;                          // same for view only
    unsigned long long texunitid;                       // same for textures
    Matrix4 c2s;                                        // current view
    Matrix4 w2c;                                        // current modelview
    int width, height;                                  // viewport (0,0,w,h)
    GpuCull cullmode;
    GpuZbuffer zbufmode;
    GpuDrawBuffer drawbuffermode;
    GpuLogicOp logicopmode;
    GpuFragmentProgram *fp;                             // null == none
    GpuVertexProgram *vp;                               // null == none

    const GpuTexture *texunit[GpuTexture::max_texunit]; // null == none

    mutable const char *errstr;                         // OpenGL or ours
    mutable GLenum errogl;                              // OpenGL error

    const char *opengl_diff_error () const;
};



// Base class for rendering geometry.  Based on the "type" parameter,
// this class can render 3D quadmeshes, a 2D rectangle or a 3D bounding box
//
// Supported types are not implemented with derivation because that would
// force you to derive multiple new classes if you wanted to override the
// setup function for all primitive types.
//
// Primitives are created with only position, and individual texture
// coordinates are set using the texcoord functions.  Subclassed primitives
// can override the setup() and draw() functions which are called from
// within the render() function.
//
// Note that the render function takes both a Canvas (drawing surface)
// and a Drawmode (state vector).  The state vector is passed by reference,
// so render methods should copy it before modifying.

class GpuPrimitive {
 public:

    // Supported primitive drawing types 
    enum GpuPrimitiveType { RECTANGLE, POLYGON, BBOX, QUADMESH };

    // Constructs a 2D rectangle (No texcoords, use program_parameters instead)
    GpuPrimitive (int xmin, int xmax, int ymin, int ymax);

    // Constructs a polygon (No texcoords, use program_parameters instead)
    GpuPrimitive (int n, Vector3 *polygon);

    // Constructs a bbox (No texcoords, use program_parameters instead)
    GpuPrimitive (Bbox3 &bbox);
    
    // General constructor for quadmesh
    GpuPrimitive (GpuPrimitiveType type, unsigned short nu,
        unsigned short nv, Vector3 *P0);

    // Destroys any allocated objects
    virtual ~GpuPrimitive ();

    // Return the type of this primitive
    GpuPrimitiveType primtype () const { return type; }
    unsigned short ucount () const { return nu; }
    unsigned short vcount () const { return nv; }

    // Render the primitive into Canvas using Drawmode.
    // The mask_texcoord bit flags disable the texture coordinates during
    // rendering.  0x01=texcoord#1, 0x02=texcoord#2, 0x03=#1 & #2, etc..
    // If mask_color or mask_depth are true, the appropriate buffer is disabled
    void render (const GpuDrawmode &drawmode, GpuCanvas &canvas,
        unsigned int mask_texcoord=0, bool mask_color=0, bool mask_depth=0);

    // Enums to help make texcoord masking clearer.  For example:
    //     TEXCOORD0 | TEXCOORD3
    // will mask off the first and third texcoord arrays.
    enum { TEXCOORD0=1, TEXCOORD1=2, TEXCOORD2=4, TEXCOORD3=8, TEXCOORD4=16 };
            
    // Set the texture coordinate pointers, P1 is passed in texture #1
    void texcoord (int index, Vector3 *coords);
    void texcoord (int index, Vector4 *coords);
    void texcoord (int index, int nchannels, float *coords);

    // Set the constant value for a texture coordinate, eliminating 
    // any previous value.  Note that constant values can also be passed
    // as program constants, but this requires a separate fragment program
    void texcoord (int index, float x, float y, float z, float w);

    // return the state of the texture unit
    enum TexcoordStatus { TEXCOORD_OFF=0, TEXCOORD_ON=1, TEXCOORD_CONST=5 };
    TexcoordStatus texcoord_status (int index) const {
        switch (texcoord_state (index)) {
        case 0: return TEXCOORD_OFF;
        case 1:
        case 2:
        case 3:
        case 4: return TEXCOORD_ON;
        case 5: return TEXCOORD_CONST;
        default:
            DASSERT (0);
            return TEXCOORD_OFF;
        }
    }

 protected:
    GpuPrimitiveType type;                      // selects draw method
    unsigned short nu, nv;                      // vertex array dimensions

    unsigned long buftypebits;                  // bit array for each attrib
    Vector3 *P;
    static const int max_attribs = 5;
    float *texbuf[max_attribs];
    
    // Returns the value for this texunit from the buftypebits array.
    // Each texture unit needs 3 bits for 8 possible values:
    //   0 == off, 1-4 == number of float channels, 5 == constant
    int texcoord_state (int texunit) const {
        DASSERT (max_attribs * 3 <= 28);  // insure buftypebits is big enough
        DASSERT (texunit >= 0 && texunit < max_attribs);
        return (int)((buftypebits >> (texunit * 3)) & 7);
    }

    // Set the value for this texunit in the buftypebits array.
    void texcoord_state (int texunit, int state) {
        DASSERT (state >= 0 && state <= TEXCOORD_CONST);
        DASSERT (texunit >= 0 && texunit < max_attribs);
        unsigned long v = state << (texunit * 3);
        buftypebits = (buftypebits & ~(7 << (texunit*3)) ) | v;
    }

    // Convenience routines for accessing the bitarraybits in different ways:

    // Clear all of the buffers 
    void texcoord_clear () { buftypebits = 0; }

    // Return the number of floating point channels for this unit
    int texcoord_nchannels (int texunit) {
        int s = texcoord_state (texunit);
        DASSERT (s < max_attribs);
        return s;
    }

    // Returns the GLenum for the DrawElements type for this prim
    GLenum gltype () {
        switch (type) {
        case RECTANGLE:
        case POLYGON:
        case BBOX:
            DASSERT (0);
        case QUADMESH:
            return GL_QUAD_STRIP;
            break;
        default:
            DASSERT (0);
            return 0;
        }
    }

    // static texcoord constants
    static std::vector<Vector4> constant_color;         // texcoord constants
    static std::vector<int> free_constant_idx;          // free list
    GLuint get_constant_id (float x, float y, float z, float w);
    void release_constant_id (GLuint i);

    // Drawing functions called based on primitive type
    void rect (const GpuDrawmode &drawmode, GpuCanvas &canvas);
    void polygon (const GpuDrawmode &drawmode, GpuCanvas &canvas);
    void bbox (const GpuDrawmode &drawmode, GpuCanvas &canvas);
    void quadmesh (const GpuDrawmode &drawmode, GpuCanvas &canvas,
                   unsigned int mask_texcoords);
    void quadmesh_vertex (int nu, int nv, unsigned int mask_texcoords);
    
    // Test to see if vertex program outputs match fragment program inputs
    bool validate (const GpuDrawmode &drawmode) const;
};


// provides basic window, mouse and keyboard events for onscreen
// windows.  It is designed so that switching between on and offscreen
// rendering is easy, while still providing reasonable, simple UI
// events for onscreen windows.
//
// The GpuCanvas class is the primary drawing surface.  The GpuCanvas
// constructor requires a GpuDrawable, an abstract base class that can
// be created using one of the derived classes: GpuWindow for onscreen
// rendering and events, GpuPBuffer for offscreen floating point
// rendering, or GpuTexture for render-to-texture.
//
// Note, all work is done on the local machine, regardless of $DISPLAY etc.

// Called if a needed graphics feature is missing (GL extension, etc).
// Forces switch to cpu-only mode.
void GpuMissingFeature (const char *feature);

// Add a file descriptor and callback which will be monitored in select()
//
// WARNING, HACK AHEAD:
//
// Windows doesn't fully support the read and write select stuff.
// It works for socket file descriptors, and we have overloaded these
// functions to work with a special broadcast message that can be 
// sent to any application when it adds a read select on fd=-1.
// In this special case, which we use to emulate fifos in iv,
// we ignore the user_data value entirely and pass back the WPARAM
// of the message to allow applications to send basic message information
// to each other.
void GpuAddReadSelect (int fd, void (*callback)(void *user_data),
    void *user_data);
void GpuAddWriteSelect (int fd, void (*callback)(void *user_data),
    void *user_data);
void GpuRemoveSelect (int fd);


// Some useful ascii keyboard values in a nice device-independent enum
enum { GpuKeyPageUp=85, GpuKeyPageDown=86, GpuKeyEscape=27,
       GpuKeyEnter=13, GpuKeyBackspace=8, GpuKeyDelete=-1,
       GpuKeyLeftArrow=81, GpuKeyUpArrow=82,
       GpuKeyRightArrow=83, GpuKeyDownArrow=84,
       GpuKeyHome=80 };



class GpuDrawable {
 public:

    GpuDrawable (int w, int h);
    
    unsigned int w() const { return width; }
    unsigned int h() const { return height; }

    friend class GpuCanvas;

    virtual ~GpuDrawable();
    
 protected:
    int width, height;

    GpuCanvas *canvas;               // backpointer

    void global_enables();
    bool enables_initialized;

    // These functions define the OS-dependent portions of the drawable
    // and are called via the GpuCanvas class instead of directly.
    virtual void swap_buffers();
    virtual void make_current();                // call before drawing 
    virtual void release();                     // call when done drawing
    
    // these are members that the canvas class needs for both
    // onscreen and offscreen drawables

    void open_display();

#ifdef XWINDOWS
    GLXContext glx_context;
    virtual GLXDrawable glx_drawable() = 0;
#endif
    
#ifdef WINNT
    HWND hwnd;
    HGLRC hglrc;
    HDC hdc;
#endif

};

    

//
// Windows are for onscreen rendering and basic event management
//
class GpuWindow;     


// Structure returned to application event handler callback
struct GpuEvent {
    // Constructor used internally by GpuWindow to send events
    GpuEvent(GpuWindow& window, GpuCanvas& canvas, void *user_data);

    GpuCanvas& canvas;
    GpuWindow& window;
    
    enum Type {NoEvent, Redraw, KeyDown, KeyUp, MouseDown, MouseUp,
               MouseDrag, Resize, CloseWindow};

    Type type;
    char key;           // valid if KeyPress
    int x, y;           // valid if Mouse*
    int w, h;           // valid if Resize
    
    enum Button {LeftMouse=0, MiddleMouse=1, RightMouse=2, WheelUpMouse=3,
                 WheelDownMouse=4, NoMouse};
    Button button;      // valid if Mouse*

    bool alt;           // state of the modifier keys for all events
    bool shift;
    bool control;

    void *user_data;
};


// User-provided event handler for redraw, key and mouse events
// return 0 if event is handled, -1 on error, and 1 if not handled
typedef int (*GpuEventHandler)(GpuEvent &event);


class GpuWindow : public GpuDrawable {
 public:
    GpuWindow (int x, int y, unsigned int w, unsigned int h, 
               bool doublebuffered = true);

    ~GpuWindow ();
    
    // blocking function starts all event handling (including repaints)
    // function exits when quit() is called within some event handler.
    // manages events for multiple windows.
    friend void GpuWindowMainloop (bool block=true);

#ifdef XWINDOWS
    friend int gpu_window_handle_event (XEvent& event);
#endif

    // call this to break out of the mainloop
    void quit() { break_out_of_mainloop = true; };

    // Functions which change the window state:
    void show (bool status);                    // show/hide window
    void configure (int x, int y, int w, int h);// move or resize window
    void repaint();                             // draw now
    void post_repaint();                        // post draw event to queue
    void title(char *title, char *icontitle=0); // set window & icon title
    void resizeable (bool status);              // allow user resizes
    void raise();                               // raise to top of stack
    void timeout (long sec, long usec,          // timeout function callback
        void (*callback)(void *user_data), void *user_data);
    void blackout (bool status);

    enum CursorType { ArrowCursor, WatchCursor, HandCursor,
                      ZoomInCursor, ZoomOutCursor, WipeCursor, CrossCursor,
                      LastCursor };

    CursorType cursor (CursorType cursor);      // returns previous
    
    int x() { return xorigin; }
    int y() { return yorigin; }
    
    // insert your own event handler.  default handler will capture the
    // current image in any existing canvas, and create an image viewer
    void set_event_handler (GpuEventHandler event_handler, void *user_data);

#ifdef XWINDOWS
    friend GpuWindow* gpu_find_by_xwindow (Window window);
#endif

#ifdef WINNT
    // Allow the main window processing function to access window vars
    friend LRESULT CALLBACK win32_wndproc (HWND hwnd, UINT message, 
                                           WPARAM wparam, LPARAM lparam);

    // WARNING, HACK AHEAD:  (used for iv server mode)
    //
    // Send a special global message to all applications which will
    // have their read select callback invoked with WPARAM if the
    // file descriptor they monitor is -1.
    static bool broadcast_message (WPARAM wparam=0, LPARAM lparam=0);

    // returns true if the window event is a special Gpu broadcast message
    static bool is_broadcast_message (MSG msg);
#endif
    
 private:
    int xorigin, yorigin;
    GpuEventHandler event_handler;   // application callback
    void *user_data;                    // callback data
    bool break_out_of_mainloop;
    char *window_title;
    bool blackout_on;
    int blackout_x;
    int blackout_y;
    int blackout_w;
    int blackout_h;
    bool mapped;
    CursorType curcursor;

#ifdef XWINDOWS
    
    // Override the virtual function in the base class
    GLXDrawable glx_drawable() { return window; }

    Window window;
    Atom quit_atom;
#endif

#ifdef WINNT
    friend LRESULT CALLBACK gpu_win32_wndproc(HWND hwnd, UINT message, 
        WPARAM wparam, LPARAM lparam);
#endif
};



//
// Offscreen rendering surface
//
class GpuPBuffer : public GpuDrawable {
 public:
    GpuPBuffer (unsigned int w, unsigned int h, int bitdepth=32);
    ~GpuPBuffer();

 private:

#ifdef XWINDOWS
    GLXDrawable glx_drawable() { return pbuffer; }
    GLXFBConfig *fbconfig;
    GLXPbuffer pbuffer;
#endif

#ifdef WINNT
    HPBUFFERARB pbuffer;
#endif    

    bool mapped_gpu;
    bool error_mapping_gpu;
    unsigned char *context_origin;
};


    
//
// OpenGL drawing setup and event management
//
class GpuCanvas {
 public:
    // You may draw to the canvas immediately after calling this constructor.
    // Onscreen windows are guaranteed to be visible.
    GpuCanvas (GpuDrawable& drawable);
    ~GpuCanvas();
    
    // No effect on single buffered windows
    void swap_buffers() { drawable.swap_buffers(); }

    // returns 0=ok, -1=warning, -2=error
    int verify_extensions (const char *required_extensions);

    // renders text at the specified 2D location
    void text (float x, float y, float z, char *text);

    // Access functions
    unsigned int w() const { return drawable.w(); }
    unsigned int h() const { return drawable.h(); }

    // Clear the image before the next draw operation
    void clear (float r=0, float g=0, float b=0, float a=0, bool z=true);
    void clearz ();        // only clear zbuffer, rgba is left untouched
    
    // Start and stop an occlusion query, only one active query at a time
    void begin_occlusion_query (GpuOcclusionQuery &oq);
    void end_occlusion_query (GpuOcclusionQuery &oq);

    // Set this canvas to match the passed in drawmode
    void update (const GpuDrawmode &drawmode);
    void update_clear ();
    
    // Read float data directly back from the GPU.  glformat should be one
    // of GL_RED, GL_ALPHA, GL_RGB, GL_RGBA, etc, and must be plausibly
    // matched to channels.
    void readback (float *data, int w, int h, int channels,
                   GLenum glformat, bool frontbuffer=true);

    // Applications should NEVER call these functions if they use
    // the GPU library properly!  They are used internally.
    void make_current() { drawable.make_current(); }
    void release() { drawable.release(); }

 protected:
    GpuDrawable& drawable;
    GpuOcclusionQuery *oquery;                          // null == none
    GLuint text_base;
    bool text_initialized;
    GpuDrawmode curdrawmode;
    bool clearrgba, cleardepth;                         // clear on next update
    Vector4 clearcolor;
    GLuint initialize_text (char *font_name);
};


namespace GpuOGL {

    // Check the driver version and opengl extensions, -1=warning, -2=error
    int verify_display ();
    
    // ring the bell at a volume between 0 - 100
    void bell (int volume);

    // Returns NULL if no errors have occurred
    const char *error_string();
    
    // return the dimensions of the local screen
    void screen_size (int &width, int &height);

    // return the GL's best guess at a max texture size
    int max_texture_dimension ();
    
    extern long long read_pixels_stat;
    extern unsigned long long read_pixels_bytes_stat;
    extern long long draw_pixels_stat;
    extern long long tex_image_stat;
    extern unsigned long long tex_image_bytes_stat;
    extern long long copy_tex_image_stat;
    extern long long swap_buffers_stat;
    extern long long make_current_stat;
    extern long long release_stat;
    extern long long pixel_view_stat;
    extern long long clear_stat;
    extern PeakCounter canvas_stat;

    extern long long drawmode_update_stat;
    extern long long drawmode_update_match_stat;
    extern long long drawmode_update_view_stat;
    extern long long drawmode_update_texture_stat;
    extern long long drawmode_update_fp_stat;
    extern long long drawmode_update_vp_stat;
    extern long long drawmode_update_prog_param_stat;
    extern long long drawmode_update_cull_stat;
    extern long long drawmode_update_zbuf_stat;
    extern long long drawmode_update_drawbuf_stat;
    extern long long drawmode_update_logicop_stat;

    extern void (*set_root_context)(GpuContext ctx);
};

extern bool gpu_test_opengl_error ();


// Routines to make a canvas/context active (some canvas, any canvas,
// must be active for any OpenGL call or other Gpu operation).  If
// there is already an active canvas, it just uses the current one
// (these are automatically no-ops).  But if no canvas were active, it
// makes a special "creation" canvas active.
extern void GpuCreationMakeCurrent ();
extern void GpuCreationRelease ();


#endif // GPU_H
