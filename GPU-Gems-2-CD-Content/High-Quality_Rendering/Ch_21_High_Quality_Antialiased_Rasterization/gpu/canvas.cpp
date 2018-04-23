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
#include <cassert>
#include <stdarg.h>
#include <malloc.h>
#include <vector>

#include "gpu.h"
#include "dassert.h"

// Include gluErrorString() after other GL headers for extensions.
#include <GL/glu.h>

long long GpuOGL::make_current_stat = 0;
long long GpuOGL::release_stat = 0;
long long GpuOGL::swap_buffers_stat = 0;
long long GpuOGL::pixel_view_stat = 0;
long long GpuOGL::clear_stat = 0;
PeakCounter GpuOGL::canvas_stat;
long long GpuOGL:: read_pixels_stat = 0;
unsigned long long GpuOGL:: read_pixels_bytes_stat = 0;
long long GpuOGL:: draw_pixels_stat = 0;
long long GpuOGL:: tex_image_stat = 0;
unsigned long long GpuOGL:: tex_image_bytes_stat = 0;
long long GpuOGL:: copy_tex_image_stat = 0;
long long GpuOGL::drawmode_update_stat = 0;
long long GpuOGL::drawmode_update_match_stat = 0;
long long GpuOGL::drawmode_update_view_stat = 0;
long long GpuOGL::drawmode_update_texture_stat = 0;
long long GpuOGL::drawmode_update_fp_stat = 0;
long long GpuOGL::drawmode_update_vp_stat = 0;
long long GpuOGL::drawmode_update_prog_param_stat = 0;
long long GpuOGL::drawmode_update_cull_stat = 0;
long long GpuOGL::drawmode_update_zbuf_stat = 0;
long long GpuOGL::drawmode_update_drawbuf_stat = 0;
long long GpuOGL::drawmode_update_logicop_stat = 0;

#ifdef WINNT
static const char *GpuOGL_required_extensions = 
"GL_VERSION_1_5 "
"WGL_ARB_extensions_string "
"WGL_ARB_pbuffer "
"WGL_ARB_pixel_format "
"WGL_ARB_render_texture "
"WGL_NV_float_buffer "
"GL_ARB_depth_texture "
"GL_ARB_multitexture "
"GL_ARB_occlusion_query "
"GL_ARB_vertex_program "
"GL_NV_depth_clamp "
"GL_NV_float_buffer "
"GL_NV_occlusion_query "
"GL_NV_texture_rectangle "
"GL_NV_vertex_program "
"GL_NV_fragment_program ";
#else
static const char *GpuOGL_required_extensions = 
"GL_ARB_depth_texture "
"GL_ARB_multitexture "
"GL_ARB_occlusion_query "
"GL_ARB_point_parameters "
"GL_ARB_point_sprite "
"GL_ARB_vertex_buffer_object "
"GL_ARB_vertex_program "
"GL_EXT_point_parameters "
"GL_NV_depth_clamp "
"GL_NV_float_buffer "
"GL_NV_occlusion_query "
"GL_NV_packed_depth_stencil "
"GL_NV_point_sprite "
"GL_NV_primitive_restart "
"GL_NV_texture_rectangle "
"GL_NV_vertex_program "
"GL_NV_fragment_program ";
#endif


// A global window list, declared here to insure it is constructed
// before the first window is created.  If we get away from the static
// constructor for the GpuCreationDrawable below, we can move this
// back into window.cc and make it static.  This list is mostly used
// to know when to drop out of the main window event loop.  If we
// switch over completely to FLTK and get rid of the Gpu event
// processing, we can remove this list.
std::vector<GpuWindow *> GpuWindowList; 

// Initialize the per-thread list to NULL using a static constructor.
// It seems wise to have this before any static window creation.
GpuDrawable *GpuCurrentDrawable = NULL;

// We need a single special window to be created before all other
// windows in the system to use to validate the display, and,
// under WIN32, to setup all the wgl and OpenGL extension function
// pointers.  This must be a window, and not a pbuffer because
// under WIN32, you need a wgl function to create a pbuffer,
// and you must have a valid OpenGL context to get wgl func ptrs.
//
// We make the window static, since it is not necessary in any
// function, but the canvas is not since we need to use it when
// constructing pbuffers.
//
// Note that we share this same window for all pbuffers and display
// validation functions.  This means that we always create a single
// 1x1 window before all others, but we don't create these small
// windows needlessly whenever we create a pbuffer.
//
// WARNING: Do not create any other global static windows, pbuffers
//          or canvases or this important creation order will break.
//
// WARNING: Under WIN32, the window "class" is given the function
//          pointer to the event processing loop, so it is unwise
//          to create local GpuWindow classes since the OS may
//          write through the invalid pointer after it is deleted
//          from the stack.  Only destroy windows at program exit
//          or if the window manager destroys the window.

// FIXME: Strange performance issues arise if we don't create static surfaces!
GpuContext GpuCreationContext = NULL;
static GpuCanvas *GpuCreationCanvas = NULL;

#define STATIC_SURFACES
#ifdef STATIC_SURFACES
#  ifdef WINNT
static GpuDrawable *GpuCreationDrawable = new GpuWindow (0, 0, 1, 1);
#  else
static GpuDrawable *GpuCreationDrawable = new GpuPBuffer (1, 1);
#  endif
#else
static GpuDrawable *GpuCreationDrawable = NULL;
#endif




static inline void
GpuInitializeCreationContext ()
{
    if (GpuCreationCanvas)
        return;
#ifndef STATIC_SURFACES    
#ifdef WINNT
    GpuCreationDrawable = new GpuWindow (0, 0, 1, 1);
    extern bool remove_from_windowlist (GpuWindow *window);
    remove_from_windowlist ((GpuWindow *)GpuCreationDrawable);
#else
    GpuCreationDrawable = new GpuPBuffer (1, 1);
#endif
#endif    
    GpuCreationCanvas = new GpuCanvas (*GpuCreationDrawable);
}


void
GpuCreationMakeCurrent ()
{
    if (GpuCurrentDrawable)
        return;
    GpuInitializeCreationContext();
    GpuCreationCanvas->make_current ();
}


void
GpuCreationRelease ()
{
    if (GpuCurrentDrawable != GpuCreationDrawable)
        return;
    GpuCreationCanvas->release ();
    GpuCurrentDrawable = NULL;
}

    


#ifdef XWINDOWS

static char *HelveticaFontName =
	"-adobe-helvetica-medium-r-normal--14-140-75-75-p-*-iso8859-1";

// Global Display opened in open_display()
Display *GpuDisplay = NULL;

// We always open a local display, regardless of $DISPLAY etc.
const char *GpuDisplayName = ":0";

#endif

#ifdef WINNT
static char *HelveticaFontName = "NOTUSED";
#endif

static char *GpuOGL_ErrorString = NULL;


void
GpuError  (const char *message, ...)
{
    char buf0[1024], buf1[1024];
    va_list ap;
    va_start (ap, message);
    vsprintf (buf0, message, ap);
    va_end (ap);
#ifdef WINNT
    sprintf (buf1, "Gpu Error: %s", buf0);
#else
    sprintf (buf1, "Display \"%s\": %s.",XDisplayName(GpuDisplayName),buf0);
#endif
    if (GpuOGL_ErrorString != NULL)
        free (GpuOGL_ErrorString);
    GpuOGL_ErrorString = strdup (buf1);
}


const char *
GpuOGL::error_string()
{
    const char *str = GpuOGL_ErrorString;
    return str;
}

// OpenGL considers it an error to call glEnable inside a begin/end
// block.  This includes occlusion query begin/end. 

void
GpuDrawable::global_enables()
{
#ifndef DISABLE_GL
    glEnable (GL_TEXTURE_2D);
    glEnable (GL_TEXTURE_RECTANGLE_NV);
    glDisable (GL_LIGHTING);
    glColor3f (1, 1, 1);
#endif
}
    

// Grab the hardware rendering context and block others from drawing
void
GpuDrawable::make_current()
{
    if (GpuCurrentDrawable == this)
        return;
    GpuCurrentDrawable = this;

    GpuOGL::make_current_stat++;
#ifndef DISABLE_GL    
#ifdef XWINDOWS
    if (!glXMakeContextCurrent (GpuDisplay, glx_drawable(), glx_drawable(),
                                glx_context)) {
        std::cerr << "Cannot make OpenGL context " << glx_drawable() <<
            " current.\n";
        exit (EXIT_FAILURE);
    }
#endif

#ifdef WINNT
    if (!wglMakeCurrent (hdc, hglrc)) {
        std::cerr << "Cannot make OpenGL context current.\n";
        DASSERT (0);
        exit (EXIT_FAILURE);
    }
#endif

    // We must wait until a canvas is current before setting all enables
    if (!enables_initialized) {
        enables_initialized = true;
        global_enables();

#ifdef WINNT
        // only initialize these once, not for each drawable
        static bool initialized = false;
        if (!initialized) 
            glh_init_extensions (GpuOGL_required_extensions);
#endif    
    }
#endif    
}



// Release the hardware rendering context so others may draw
void
GpuDrawable::release()
{
    GpuOGL::release_stat++;
#ifndef DISABLE_GL    
#ifdef WINNT
    DASSERT (GpuCurrentDrawable == this);
    if (!wglMakeCurrent (NULL, NULL)) {
        LPVOID lpMsgBuf;
        if (!FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | 
                           FORMAT_MESSAGE_FROM_SYSTEM | 
                           FORMAT_MESSAGE_IGNORE_INSERTS,
                           NULL,
                           GetLastError(),
                           MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                           (LPTSTR) &lpMsgBuf,
                           0,
                           NULL )) {
            fprintf (stderr, "ERROR wglMakeCurrent: %s\n", lpMsgBuf);
            fflush (stderr);
        }
    }
    ReleaseDC (hwnd, hdc);
#endif
#ifdef XWINDOWS
    if (!glXMakeContextCurrent (GpuDisplay, None, None, NULL)) {
        std::cerr << "Cannot make OpenGL context current.\n";
        exit (EXIT_FAILURE);
    }
#endif
#endif
    GpuCurrentDrawable = NULL;
}



void
GpuDrawable::swap_buffers()
{
    GpuOGL::swap_buffers_stat++;
#ifndef DISABLE_GL    
#ifdef XWINDOWS    
    glXSwapBuffers (GpuDisplay, glx_drawable());
#endif
    
#ifdef WINNT
    SwapBuffers (hdc);
#endif
#endif    
}



// Constructor initializes global variables and creates a
// platform specific OpenGL Window
GpuCanvas::GpuCanvas (GpuDrawable& drawable) 
    : drawable(drawable), oquery (NULL),
      text_initialized(false),
      clearrgba (true), cleardepth (true), clearcolor (0.0)
{
    GpuOGL::canvas_stat++;
    drawable.canvas = this;
}



GpuCanvas::~GpuCanvas()
{
    if (GpuCurrentDrawable == &drawable)
        drawable.release();
    GpuOGL::canvas_stat--;
}



GLuint
GpuCanvas::initialize_text (char *font_name)
{
#ifndef DISABLE_GL    

    GLuint base;
    
#ifdef XWINDOWS    
    Font id;
    XFontStruct *font_info;
    int first, last, firstbitmap, i;
    int firstrow, lastrow;
    int maxchars;

    font_info = XLoadQueryFont (GpuDisplay, font_name);
    if (font_info == NULL) {
        return 0;
    }
    id = font_info->fid;

    // First and Last char in a row of chars
    first = (int)font_info->min_char_or_byte2;
    last = (int)font_info->max_char_or_byte2;

    // First and Last row of chars, important for multibyte charset's
    firstrow = (int)font_info->min_byte1;
    lastrow = (int)font_info->max_byte1;

    // How many chars in the charset
    maxchars = 256 * lastrow + last;
    
    base = glGenLists (maxchars+1);
    if (base == 0) {
        return 0;
    }

    // Get offset to first char in the charset
    firstbitmap = 256 * firstrow + first;

    // for each row of chars, call glXUseXFont to build the bitmaps.
    for (i=firstrow; i<=lastrow; i++) {
        glXUseXFont (id, firstbitmap, last-first+1, base+firstbitmap);
        firstbitmap += 256;
    }

#endif

#ifdef WINNT
    SelectObject (drawable.hdc, GetStockObject (SYSTEM_FONT));
    
    base = glGenLists (256);
    if (base == 0) {
        return 0;
    }

    wglUseFontBitmaps (drawable.hdc, 0, 256, base);
#endif
    
    return base;
#endif    
}



void
GpuCanvas::text (float x, float y, float z, char *text)
{
#ifndef DISABLE_GL    
#if defined(XWINDOWS) | defined(WINNT)
    if (!text_initialized) {
        text_initialized = true;
        text_base = initialize_text (HelveticaFontName);
    }

    if (text_base == 0) return;
    glRasterPos3f (x, y, z);
    glPixelStorei (GL_UNPACK_ALIGNMENT, 4);
    glListBase (text_base);
    glCallLists ((GLsizei)strlen (text), GL_UNSIGNED_BYTE, text);
#endif
#endif    
}




GpuDrawable::GpuDrawable (int w, int h) 
    : width(w), height(h), canvas(NULL), enables_initialized(false)
{}



GpuDrawable::~GpuDrawable ()
{
#ifndef DISABLE_GL    
#ifdef WINNT
    wglDeleteContext (hglrc);
#else
    glXDestroyContext (GpuDisplay, glx_context);
#endif
#endif    
}



void
GpuDrawable::open_display()
{
#ifndef DISABLE_GL    
#ifdef XWINDOWS
    if (GpuDisplay == NULL) {
        // We ignore $DISPLAY, and always open a local connection.
        // (If the caller wants to override this, they're stuck, because
        // this code is called by the static drawable's constructor,
        // before main() is run.)
        GpuDisplay = XOpenDisplay (":0");

        if (GpuDisplay == NULL) {
            fprintf (stderr, "ERROR: Can't open display \"%s\".  Exiting.\n",
                     XDisplayName (GpuDisplayName));
            // FIXME Run in cpu mode instead
            exit (EXIT_FAILURE);
        }
    }
    
    // Neither GpuWindow nor GpuPBuffer can work without GLX
    int error_base;
    int event_base;
    if (!glXQueryExtension (GpuDisplay, &error_base, &event_base)) {
        fprintf (stderr, "Missing GLX extension");
    }
#endif
#endif
}



// If the named environment variable is set to anything other than
// "0", including the empty string, then it is consider to be enabled
static bool
nv_env_var_enabled (const char *name)
{
    char *envstr = getenv (name);
    if (envstr == NULL)
        return false;
    if (envstr[0] != '0')
        return true;
    if (strlen (envstr) > 1)
        return true;
    return false;
}

// Make sure we have extensions we need.
// By the way, the reason this is in the GpuCanvas class
// is to force the caller to construct a GpuCanvas first;
// GL functions like glGetString might not work until a context
// is attached to a drawable, and constructing a GpuCanvas does that.
int
GpuCanvas::verify_extensions (const char *required_extensions)
{
#ifndef DISABLE_GL    
    make_current();

    const char *extensions = NULL;
#ifdef WINNT
    const char *wgl_extensions = NULL;
#endif
    
    if (required_extensions != NULL) {

        // get the list of supported extensions
        extensions = (const char *) glGetString (GL_EXTENSIONS);

#ifdef WINNT
        // get the list of supported WGL extensions
        if (wglGetExtensionsStringARB) {
            wgl_extensions =
                (const char *) wglGetExtensionsStringARB (wglGetCurrentDC());
        }
#endif // WINNT

        // duplicate the input list to avoid changing argument
        char *dup_required_extensions = strdup (required_extensions);

        // use strtok to check each of the required extensions
        char *p;
        for (char *tok = strtok_r (dup_required_extensions, " ", &p);
             tok != NULL; tok = strtok_r (NULL, " ", &p)) {
            // GL_VERSION_xxx is not found in either the GL or the WGL
            // extension strings - but is used by glh_init_extensions
            // to check the version number - so skip it to avoid generating 
            // a bogus error message
            if (strstr(tok, "GL_VERSION_"))
                continue;

            // FIXME should test for ' ' or '\0' after extension; currently
            // we match substrings of extensions.
            if ((strstr (extensions, tok) == NULL)
#ifdef WINNT
                && (strstr (wgl_extensions, tok) == NULL)
#endif // WINNT
                ) {
#ifdef DEBUG            
                GpuError ("Missing OpenGL extension \"%s\"", tok);
#else
                GpuError ("Missing OpenGL extension");
#endif
                free (dup_required_extensions);
                goto error;
            }
        }

        free (dup_required_extensions);
    }
    
#endif

    release();
    return 0;

 error:
    release();
    return -2;
}



int
GpuOGL::verify_display ()
{
    GpuInitializeCreationContext();
    
    // Make a GLX connection so we can query it.
    // man glXIntro says GL functions don't work until a context
    // is attached to a drawable, so, go ahead and make a pbuffer
    // and attach a canvas to it.
    //
    // This already verifies that we can open the display, etc.
    // FIXME Switch to cpu-only if this fails.
    int errcode =
        GpuCreationCanvas->verify_extensions (GpuOGL_required_extensions);

    if (errcode != 0)
        return errcode;
    
    return 0;
}



void
GpuCanvas::readback (float *data, int w, int h, int channels, 
                     GLenum glformat, bool frontbuffer)
{
    make_current();

    GpuOGL::read_pixels_stat++;
    GpuOGL::read_pixels_bytes_stat += w*h*sizeof(float)*channels;
#ifndef DISABLE_GL    
    // allocate new image if existing one is not big enough
    GLenum readbuf = frontbuffer ? GL_FRONT : GL_BACK;
    glReadBuffer(readbuf);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei (GL_UNPACK_ROW_LENGTH, w);
    // Not allowed under Windows, check Linux!
//    glPixelStorei (GL_UNPACK_IMAGE_HEIGHT, h);
    glReadPixels(0, 0, w, h, glformat, GL_FLOAT, (GLvoid*)data);
#endif    
}


#if 0
void
GpuCanvas::draw_image(void *data, int w, int h, int stride, int row_length,
                      GLenum gl_format, GLenum gl_type,
                      double x, double y, double xscale, double yscale,
                      int px, int py)
{
    make_current();
    
    GpuOGL::draw_pixels_stat++;
#ifndef DISABLE_GL    
    // Set up GL to accept non-aligned images if necessary
    if ((stride % 4) == 0) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    } else if ((stride % 2) == 0) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    } else {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    }

    glPixelZoom(xscale, yscale);
    
    // Set the viewport so that we always draw from 0,0 or
    // a more positive value within the viewport.  We'll set
    // the glPixelStore values to make sure we only draw the
    // portion of the bitmap that is onscreen.
    //
    // Setting RasterPos to a position outside the viewport will cause
    // the raster pos to be invalid and the entire DrawPixels call to
    // be ignored.  The approved GL fix is to set a valid pixel position
    // and then call glBitmap to offset the raster pos (see the glRasterPos
    // and glBitmap manual pages) passing a NULL bitmap.
    glRasterPos2f(0, 0);
    glBitmap(0, 0, 0, 0, x, y, NULL);

    // Allow unpacking of data within the image viewport
    glPixelStorei(GL_UNPACK_ROW_LENGTH, row_length);

    // offset the data to px, py
    data = ((unsigned char *)data) + (px + py * row_length) * stride;

    // draw the visible portion of the image
    glDrawPixels(w, h, gl_format, gl_type, data);
#endif    
}
#endif


void
GpuCanvas::clear (float r, float g, float b, float a, bool z)
{
    clearcolor[0] = r;
    clearcolor[1] = g;
    clearcolor[2] = b;
    clearcolor[3] = a;
    clearrgba = true;
    cleardepth = z;
}



void
GpuCanvas::clearz ()
{
    cleardepth = true;
}



void
GpuCanvas::update_clear ()
{
    if (!clearrgba && !cleardepth)
        return;

    GLbitfield clearmask = 0;
    if (clearrgba) {
        glClearColor (clearcolor[0],clearcolor[1],clearcolor[2],clearcolor[3]);
        clearmask |= GL_COLOR_BUFFER_BIT;
    }
    if (cleardepth)
        clearmask |= GL_DEPTH_BUFFER_BIT;
    glClear (clearmask);
    clearrgba = false;
    cleardepth = false;
    GpuOGL::clear_stat++;
}



void
GpuCanvas::begin_occlusion_query (GpuOcclusionQuery &oq)
{
    DASSERT (oquery == NULL);        // insure no existing query active
    oquery = &oq;
    make_current();                  // activate this context
    if (oq.id == 0) {
        DASSERT (gpu_test_opengl_error());
        glGenOcclusionQueriesNV (1, &oq.id);
        DASSERT (gpu_test_opengl_error());
    }
    DASSERT (oq.id != 0);           // insure new query is valid
    DASSERT (gpu_test_opengl_error());
    glBeginOcclusionQueryNV (oq.id);
    DASSERT (gpu_test_opengl_error());
}



void
GpuCanvas::end_occlusion_query (GpuOcclusionQuery &oq)
{
    DASSERT (oquery == &oq);        // insure existing active query
    DASSERT (oquery->id != 0);       // insure existing query is valid
    DASSERT (glIsOcclusionQueryNV (oquery->id));
    DASSERT (gpu_test_opengl_error());
    glEndOcclusionQueryNV ();
    DASSERT (gpu_test_opengl_error());
    oquery = NULL;
}



int
GpuOGL::max_texture_dimension ()
{
    GLint w;
    GpuCreationMakeCurrent ();
    glGetIntegerv (GL_MAX_TEXTURE_SIZE, &w);
    GpuCreationRelease ();
    return w;
}
