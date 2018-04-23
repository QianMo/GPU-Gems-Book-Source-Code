/*********************************************************************NVMH3****

  Copyright NVIDIA Corporation 2004
  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
  *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
  OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
  NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
  CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
  LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
  INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.

  ****************************************************************************/

#include "pug.h"
#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <GL/wglext.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>

#define PUG_MAX_STREAMS 9      // max number of streams -- WPOS + TEXCOORD0-7

static CGcontext cgContext;    // Cg context
static bool createdCgContext;  // did we create the Cg context?
static bool createdGLContext;  // did we create the window and OpenGL context?
static HWND hwnd;              // handle to (invisible) window
static HDC  hDC;               // window device context
static HGLRC hwinRC;           // window render context

static char* incPath;
static bool initialized = false;

static PUGProgram* passthruProg[4];

#define GET_GLERROR(ret)                                          \
    {                                                             \
        GLenum err = glGetError();                                \
        if (err != GL_NO_ERROR) {                                 \
            fprintf(stderr, "[%s line %d] GL Error: %s\n",        \
                    __FILE__, __LINE__, gluErrorString(err));     \
            fflush(stderr);                                       \
            exit(1); \
            return (ret);                                         \
         }                                                        \
    }

static HGLRC sharedhpbufferRC;
static int sharedViewport[2] = {0, 0};

static PFNWGLCREATEPBUFFERARBPROC      wglCreatePbufferARB;
static PFNWGLQUERYPBUFFERARBPROC       wglQueryPbufferARB;
static PFNWGLGETPBUFFERDCARBPROC       wglGetPbufferDCARB;
static PFNWGLCHOOSEPIXELFORMATARBPROC  wglChoosePixelFormatARB;
static PFNWGLBINDTEXIMAGEARBPROC       wglBindTexImageARB;
static PFNWGLRELEASETEXIMAGEARBPROC    wglReleaseTexImageARB;
static PFNWGLDESTROYPBUFFERARBPROC     wglDestroyPbufferARB;
static PFNWGLRELEASEPBUFFERDCARBPROC   wglReleasePbufferDCARB;
static PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglGetPixelFormatAttribivARB;

static PFNGLDRAWBUFFERSATIPROC         glDrawBuffersATI;
static PFNGLMULTITEXCOORD2FPROC        glMultiTexCoord2f;

static PFNGLACTIVETEXTUREARBPROC       glActiveTextureARB;
static PFNGLBINDBUFFERPROC             glBindBufferARB;
static PFNGLDELETEBUFFERSPROC          glDeleteBuffersARB;
static PFNGLGENBUFFERSPROC             glGenBuffersARB;
static PFNGLBUFFERDATAPROC             glBufferDataARB;
static PFNGLMAPBUFFERPROC              glMapBufferARB;
static PFNGLUNMAPBUFFERPROC            glUnmapBufferARB;

static PUGBuffer *curBuffer;

static bool pugActivateBuffer(PUGBuffer *buf, PUGTarget target);

static bool
createWindow()
{
    WNDCLASS wc;
    HINSTANCE hinst;

    hinst = GetModuleHandle(NULL);
  
    if (GetClassInfo(hinst, "pug", &wc) == 0) {
        ZeroMemory(&wc, sizeof(wc));
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = (WNDPROC)DefWindowProc;
        wc.hInstance = hinst;
        wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.lpszClassName = "pug";
        if (RegisterClass(&wc) == 0)
            return false;
    }
  
    hwnd = CreateWindow("pug", "pug",
                        WS_CLIPSIBLINGS|WS_CLIPCHILDREN|WS_POPUP,
                        0, 0, 0, 0,
                        NULL, NULL, hinst, NULL );
    return (hwnd != 0);
}


static bool
createGLContext()
{
    PIXELFORMATDESCRIPTOR pfd;

    if (hwnd == 0)
        return false;

    hDC = GetDC(hwnd);
    if (hDC == 0)
        return false;
 
    // Set pixel format
    ZeroMemory(&pfd, sizeof(pfd));
    pfd.nSize    = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags  = PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.iLayerType = PFD_MAIN_PLANE;
    
    int iformat = ChoosePixelFormat(hDC, &pfd);
    if (iformat == 0) 
        return false;

    if (SetPixelFormat(hDC, iformat, &pfd) == 0) 
        return false;
 
    // Create gl context
    hwinRC = wglCreateContext(hDC);
    if (hwinRC == 0)
        return false;
 
    if (wglMakeCurrent(hDC, hwinRC) == 0)
        return false;

    return true;
}


static bool
initGL(void)
{
#define WGL_INIT_FUNC(B,A) A = (B) wglGetProcAddress(#A); if(!A) err = true;
    bool err = false;

    WGL_INIT_FUNC(PFNWGLCREATEPBUFFERARBPROC,     wglCreatePbufferARB);
    WGL_INIT_FUNC(PFNWGLQUERYPBUFFERARBPROC,      wglQueryPbufferARB);
    WGL_INIT_FUNC(PFNWGLGETPBUFFERDCARBPROC,      wglGetPbufferDCARB);
    WGL_INIT_FUNC(PFNWGLCHOOSEPIXELFORMATARBPROC, wglChoosePixelFormatARB);
    WGL_INIT_FUNC(PFNWGLBINDTEXIMAGEARBPROC,      wglBindTexImageARB);
    WGL_INIT_FUNC(PFNWGLRELEASETEXIMAGEARBPROC,   wglReleaseTexImageARB);
    WGL_INIT_FUNC(PFNWGLDESTROYPBUFFERARBPROC,    wglDestroyPbufferARB);
    WGL_INIT_FUNC(PFNWGLRELEASEPBUFFERDCARBPROC,  wglReleasePbufferDCARB);
    WGL_INIT_FUNC(PFNWGLGETPIXELFORMATATTRIBIVARBPROC, wglGetPixelFormatAttribivARB);

    WGL_INIT_FUNC(PFNGLMULTITEXCOORD2FPROC,       glMultiTexCoord2f);
#if 0
    // For eventual support of MRT & async readback.
    WGL_INIT_FUNC(PFNGLDRAWBUFFERSATIPROC,        glDrawBuffersATI);
    WGL_INIT_FUNC(PFNGLBINDBUFFERPROC,            glBindBufferARB);
    WGL_INIT_FUNC(PFNGLDELETEBUFFERSPROC,         glDeleteBuffersARB);
    WGL_INIT_FUNC(PFNGLGENBUFFERSPROC,            glGenBuffersARB);
    WGL_INIT_FUNC(PFNGLBUFFERDATAPROC,            glBufferDataARB);
    WGL_INIT_FUNC(PFNGLMAPBUFFERPROC,             glMapBufferARB);
    WGL_INIT_FUNC(PFNGLUNMAPBUFFERPROC,           glUnmapBufferARB);
#endif
    WGL_INIT_FUNC(PFNGLACTIVETEXTUREARBPROC,      glActiveTextureARB);

    return !err;
}


static void
handleCgError(void) 
{
    CGerror err = cgGetError();
    fprintf(stderr, "Cg error: %s\n", cgGetErrorString(err));
    if (err == 1)
        fprintf(stderr, "last listing:\n%s\n", cgGetLastListing(cgContext));
    fflush(stderr);
}

static CGcontext
initCg()
{
    cgSetErrorCallback(handleCgError);
    return cgCreateContext();
}


////////////////////////////////////////////////////////////////////


bool pugInit(const char* includePath, CGcontext cgcontext, 
             bool createOGLContext)
{
    if (createOGLContext)
    {   
        createdGLContext = true;
        if (!(createWindow() && createGLContext()))
            return false;
    }
    else
    {
        hDC    = wglGetCurrentDC();
        hwinRC = wglGetCurrentContext();
        createdGLContext = false;
    }
    
    if (!initGL())
        return false;

    if (cgcontext == 0) {
        cgContext = initCg();
        createdCgContext = true;
    } else {
        cgContext = cgcontext;
        createdCgContext = false;
    }

    if (cgContext == 0)
        return false;

    initialized = true;
   
    if (includePath != NULL) {
        incPath = new char[strlen(includePath) + 2];
        strcpy(incPath, includePath);
        // append a slash if necessary.
        if (incPath[strlen(incPath)-1] != '/')
            strcat(incPath, "/");
    } else {
        strcpy(incPath, "./");
    }
    
    // create path to pugReduce.cg
    char *path = new char[strlen(incPath) + strlen("pugreduce.cg") + 1];
    strcpy(path, incPath);
    strcat(path, "pugreduce.cg");
    
    const char* fpargs[2];
    fpargs[0] = new char[strlen("-DCOMPONENTS=1")+1];
    fpargs[1] = 0;

    // create the passthru programs for reductions
    sprintf(const_cast<char*>(fpargs[0]), "-DCOMPONENTS=1");
    passthruProg[0] = pugLoadProgram(path, "passthru", fpargs);
    sprintf(const_cast<char*>(fpargs[0]), "-DCOMPONENTS=2");
    passthruProg[1] = pugLoadProgram(path, "passthru", fpargs);
    sprintf(const_cast<char*>(fpargs[0]), "-DCOMPONENTS=3");
    passthruProg[2] = pugLoadProgram(path, "passthru", fpargs);
    sprintf(const_cast<char*>(fpargs[0]), "-DCOMPONENTS=4");
    passthruProg[3] = pugLoadProgram(path, "passthru", fpargs);
    if (passthruProg[0] == NULL ||
        passthruProg[1] == NULL ||
        passthruProg[2] == NULL ||
        passthruProg[3] == NULL) 
    {
        fprintf(stderr, "Unable to load passthru program\n");
        initialized = false;
    }

    delete [] fpargs[0];
    delete [] path;

    return initialized;
}

bool pugCleanup()
{
    if (createdCgContext)
        cgDestroyContext(cgContext);

    if (createdGLContext && hwinRC) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(hwinRC);
    }
    // destroy window
    if (hwnd)
        DestroyWindow(hwnd);
    sharedhpbufferRC = 0;

    delete [] incPath;

    return true;
}

struct PUGParam {
    const char *name;
    int texunit;
    int val;
};

class PUGProgram {
public:
    PUGProgram(void) { prog = 0; numDomains = numTexParams = 0; progType = PUG_PROGRAM_DEFAULT; }
    PUGProgram(CGprogram p);
    ~PUGProgram();

    int numTexParams;
    PUGParam texParams[PUG_MAX_STREAMS];

    bool setTextureParameter(const char *name, int val);
    bool setTextureParameters(void);
    void scanTexParams(void);

    int numDomains;
    PUGRect domain[PUG_MAX_STREAMS];

    enum ProgramType {
        PUG_PROGRAM_DEFAULT,
        PUG_PROGRAM_REDUCTION,
        PUG_PROGRAM_NUM_TYPES
    };

    ProgramType progType;
    CGprogram prog;
    CGprogram reduceProgs[2]; // 2D and 1D so we can reduce non-square buffers
};

PUGProgram::PUGProgram(CGprogram p)
{
    prog = p;
    numDomains = 0;
    scanTexParams();
}

void PUGProgram::scanTexParams(void)
{
    CGparameter param;

    numTexParams = 0;

    param = cgGetFirstLeafParameter(prog, CG_PROGRAM);
    while (param) {
        CGresource res = cgGetParameterBaseResource(param);
        if (res == CG_TEXUNIT0) {
            texParams[numTexParams].name = cgGetParameterName(param); 
            texParams[numTexParams].texunit = 
                cgGetParameterResource(param) - CG_TEXUNIT0;
            numTexParams++;
        }
        param = cgGetNextLeafParameter(param);
    }
}

bool PUGProgram::setTextureParameter(const char *name, int val)
{
    int i;
    for (i = 0; i < numTexParams; i++) {
        if (strcmp(name, texParams[i].name) == 0) {
            texParams[i].val = val;
            return true;
        }
    }
    return false;
}

bool PUGProgram::setTextureParameters(void)
{
    int i;
    for (i = 0; i < numTexParams; i++) {
        glActiveTextureARB(texParams[i].texunit + GL_TEXTURE0_ARB); 
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, texParams[i].val);
    }
    glActiveTextureARB(GL_TEXTURE0_ARB);
    return true;
}

PUGBuffer::PUGBuffer(int w, int h, int nc, HDC dc, HPBUFFERARB pbuf, HGLRC rc,
                     GLuint th, GLuint ib, bool db)
{
    width = w;
    height = h;
    nComponents = nc;
    hpbuffer = pbuf;
    hpbufferDC = dc;
    hpbufferRC = rc;
    texHandle = th;
    imageBuffer = ib;
    bound[0] = bound[1] = false;
    doublebuffered = db;
    currentDrawBuffer = PUG_FRONT;
}

PUGRect pugTransposeRect(PUGRect &rect)
{
    return PUGRect(rect.y00, rect.y10, rect.y01, rect.y11,
                   rect.x00, rect.x10, rect.x01, rect.x11);
}

bool
pugBindDomain(PUGProgram *prog, const char *paramName, const PUGRect &domain)
{
    if (prog == 0)
        return false;

    // Get param handle
    CGparameter param = cgGetNamedParameter(prog->prog, paramName);
    if (param == 0)
        return false;

    // Get param resource; if not a texcoord, fail.
    CGresource rsrc = cgGetParameterResource(param);
    int num = rsrc - CG_TEX0;
    if (num < 0 || num >= PUG_MAX_STREAMS)
        return false;
    if (num >= prog->numDomains)
        prog->numDomains = num+1;
    prog->domain[num] = domain;
    return true;
}

// Helper function used by pugLoadProgram and pugLoadReductionProgram
CGprogram _loadProgram(const char* filename, const char*entrypoint, 
                       CGprofile profile, const char**args)
{
    // create compiler arguments
    int numArgs = 0;
    const char** fpargs;
    if (args != NULL)
    {
        for (int i = 0; args[i] != 0; ++i) numArgs++; // count args
    }
    fpargs = new const char*[numArgs + 2];
    for (int i = 0; i < numArgs; ++i) fpargs[i] = args[i];

    // include path argument so programs can find pug.cg    
    fpargs[numArgs] = new char[3 + strlen(incPath)];
    sprintf(const_cast<char*>(fpargs[numArgs]), "-I%s", incPath);
    fpargs[numArgs+1] = 0;

    return cgCreateProgramFromFile(cgContext, CG_SOURCE, filename, 
                                   profile, entrypoint, fpargs);
}

PUGProgram *
pugLoadProgram(const char *filename, const char *entrypoint, const char **args)
{
    CGprofile fragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    cgGLSetOptimalOptions(fragmentProfile);
    cgSetAutoCompile(cgContext, CG_COMPILE_MANUAL);

    CGprogram prog = _loadProgram(filename, entrypoint, fragmentProfile, args);

    if (prog != 0) {
        cgCompileProgram(prog);
        cgGLLoadProgram(prog);
        GET_GLERROR(0);
        cgGLEnableProfile(fragmentProfile);
        GET_GLERROR(0);
    }

    GET_GLERROR(0);

    return new PUGProgram(prog);
}

PUGProgram *
pugLoadReductionProgram(const char *filename, const char *reduceOp, 
                        int samples /* = 2 */, int components /* = 1 */, 
                        const char **args /* = NULL */)
{
    if (!initialized)
        return NULL;

    assert(samples == 2 || samples == 4); // for now...
    assert(components > 0 && components < 5);

    CGprofile fragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    cgGLSetOptimalOptions(fragmentProfile);
    cgSetAutoCompile(cgContext, CG_COMPILE_MANUAL);

    const char* fpargs[2];
    fpargs[0] = new char[2 + strlen("-DCOMPONENTS=1")];
    sprintf(const_cast<char*>(fpargs[0]), "-DCOMPONENTS=%d", components);
    fpargs[1] = 0;

    CGprogram prog1D = _loadProgram(filename, "reduce_2samples", 
                                    fragmentProfile, fpargs);
   
    if (prog1D != 0) {
        CGparameter reduceImpl = 
            cgCreateParameter(cgContext, cgGetNamedUserType(prog1D, reduceOp));
        cgConnectParameter(reduceImpl, cgGetNamedParameter(prog1D, "reducer"));
        cgCompileProgram(prog1D);
        cgGLLoadProgram(prog1D);
        GET_GLERROR(0);
        cgGLEnableProfile(fragmentProfile);
        GET_GLERROR(0);
        //printf("%s\n", cgGetProgramString(prog, CG_COMPILED_PROGRAM));
    }

    GET_GLERROR(0);

    PUGProgram *program = new PUGProgram(prog1D);

    program->progType = PUGProgram::PUG_PROGRAM_REDUCTION;
    program->reduceProgs[0] = prog1D;

    CGprogram prog2D = _loadProgram(filename, "reduce_4samples", 
                                    fragmentProfile, fpargs);

    if (prog2D != 0) {
        CGparameter reduceImpl = 
            cgCreateParameter(cgContext, cgGetNamedUserType(prog2D, reduceOp));
        cgConnectParameter(reduceImpl, cgGetNamedParameter(prog2D, "reducer"));
        cgCompileProgram(prog2D);
        cgGLLoadProgram(prog2D);
        GET_GLERROR(0);
        cgGLEnableProfile(fragmentProfile);
        GET_GLERROR(0);
        //printf("%s\n", cgGetProgramString(prog, CG_COMPILED_PROGRAM));
    }
    program->reduceProgs[1] = prog2D;

    delete [] fpargs[0];

    return program;
}

bool
pugBindTexture(PUGBuffer *buf, PUGTarget target)
{
    if (!initialized || buf == NULL) return false;

    if (!((buf->hpbuffer == NULL) || buf->bound[target]))
    {
        GLenum targetBuffer = WGL_FRONT_LEFT_ARB;
        if (target == PUG_BACK)
            targetBuffer = WGL_BACK_LEFT_ARB;

        glBindTexture(GL_TEXTURE_RECTANGLE_NV, buf->texHandle);
        if (!wglBindTexImageARB(buf->hpbuffer, targetBuffer))
            return false;
    }
    buf->bound[target] = true;
    return true;
}

bool
pugBindStream(PUGProgram *prog, const char *baseParam, PUGBuffer *buf, 
              PUGTarget target)
{
    if (!initialized || prog == NULL || baseParam == NULL || buf == NULL)
        return false;

    char fullParam[1024];
    sprintf(fullParam, "%s.sampler", baseParam);
    if (!prog->setTextureParameter(fullParam, buf->texHandle))
        return false;
    GET_GLERROR(false);

    return pugBindTexture(buf, target);
}



bool
pugBindFloat(PUGProgram *prog, const char *pName,
                 float v0, float v1, float v2, float v3)
{
    if (!initialized || prog == NULL || pName == NULL)
        return false;

    CGparameter param = cgGetNamedParameter(prog->prog, pName);
    if (!param) return false;
    cgSetParameter4f(param, v0, v1, v2, v3);
    return true;
}


bool
pugRunProgram(PUGProgram *prog, PUGBuffer *outputBuffer)
{
    if (!initialized || outputBuffer == 0) return false;
    PUGRect range(0, outputBuffer->width, 0, outputBuffer->height);
    return pugRunProgram(prog, outputBuffer, range, 
                         outputBuffer->currentDrawBuffer);
}

bool
pugRunProgram(PUGProgram *prog, PUGBuffer *outputBuffer, const PUGRect range)
{
    return pugRunProgram(prog, outputBuffer, range,
                         outputBuffer->currentDrawBuffer);
}

bool
pugRunProgram(PUGProgram *prog, PUGBuffer *outputBuffer, const PUGRect range,
              PUGTarget target)
{
    if (!initialized || !prog || !outputBuffer) return false;

    pugActivateBuffer(outputBuffer, target);

    GET_GLERROR(false);
    if (outputBuffer->doublebuffered)   
    {
        switch(target)
        {
        case PUG_FRONT:
            outputBuffer->currentDrawBuffer = PUG_FRONT;
            glDrawBuffer(GL_FRONT);
            GET_GLERROR(false);
            break;
        case PUG_BACK:
            outputBuffer->currentDrawBuffer = PUG_BACK;
            glDrawBuffer(GL_BACK);
            GET_GLERROR(false);
            break;
        default:
            break;
        }
        GET_GLERROR(false);
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, outputBuffer->width, 0, outputBuffer->height, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    GET_GLERROR(false);

    cgGLBindProgram(prog->prog);
    GET_GLERROR(false);
    prog->setTextureParameters();
    GET_GLERROR(false);

    glBegin(GL_QUADS);
    int i;
    for (i = 0; i < prog->numDomains; i++) {
        glMultiTexCoord2f(GL_TEXTURE0_ARB+i,
                (float)prog->domain[i].x00, (float)prog->domain[i].y00);
    }
    glVertex2i(range.x00, range.y00);
    for (i = 0; i < prog->numDomains; i++) {
        glMultiTexCoord2f(GL_TEXTURE0_ARB+i,
                (float)prog->domain[i].x10, (float)prog->domain[i].y10);
    }
    glVertex2i(range.x10, range.y10);
    for (i = 0; i < prog->numDomains; i++) {
        glMultiTexCoord2f(GL_TEXTURE0_ARB+i,
                (float)prog->domain[i].x11, (float)prog->domain[i].y11);
    }
    glVertex2i(range.x11, range.y11);
    for (i = 0; i < prog->numDomains; i++) {
        glMultiTexCoord2f(GL_TEXTURE0_ARB+i,
                (float)prog->domain[i].x01, (float)prog->domain[i].y01);
    }
    glVertex2i(range.x01, range.y01);
    glEnd();

    GET_GLERROR(false);
    return true;
}

bool
pugActivateBuffer(PUGBuffer *buf, PUGTarget target)
{
    if (!initialized || buf == 0)
        return false;

    if (buf->bound[target])
        pugReleaseBuffer(buf, target);

    if (curBuffer == buf)
    {
        GET_GLERROR(NULL);
        return true;
    }

    if (!wglMakeCurrent(buf->hpbufferDC, buf->hpbufferRC))
        return false;

    curBuffer = buf;
    GET_GLERROR(NULL);
    return true;
}



PUGBuffer *
pugAllocateBuffer(int width, int height, PUGBufferMode mode, 
                  int components    /* = 1 */,
                  bool doublebuffer /* = false */)
{
  
    HPBUFFERARB hpbuffer = NULL;
    HDC hpbufferDC = NULL;
    HGLRC hpbufferRC = NULL;
    GLuint imageBuffer = 0;

    if (!initialized)
        return false;

    if (mode == PUG_WRITE || mode == PUG_READWRITE) 
    {
        assert(components == 1 || components == 4);

        int gbaBits = 0;
        GLenum bindEnum = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV;
        int pbTexFormat = WGL_TEXTURE_FLOAT_R_NV;
        if (components == 4) 
        {
            gbaBits = 32;
            bindEnum = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV;
            pbTexFormat = WGL_TEXTURE_FLOAT_RGBA_NV;
        }

        static const float fAttrib[] = {0, 0};
        static int pixelformat;

        int attribs[] = {
            WGL_DRAW_TO_PBUFFER_ARB, true,
            WGL_SUPPORT_OPENGL_ARB,  true,
            WGL_DOUBLE_BUFFER_ARB,   doublebuffer,
            WGL_PIXEL_TYPE_ARB,      WGL_TYPE_RGBA_ARB,
            WGL_RED_BITS_ARB,        32,
            WGL_GREEN_BITS_ARB,      gbaBits,
            WGL_BLUE_BITS_ARB,       gbaBits,
            WGL_ALPHA_BITS_ARB,      gbaBits,
            WGL_FLOAT_COMPONENTS_NV, true,
            bindEnum,                true,
            WGL_DEPTH_BITS_ARB,      0,
            WGL_STENCIL_BITS_ARB,    0, 
    //CO            WGL_AUX_BUFFERS_ARB,     4,
            0,                       0
        };


        unsigned int count;
        int status = wglChoosePixelFormatARB(hDC, attribs,
                    NULL, 1, &pixelformat, &count);
        if (count == 0 || status == 0)
            return NULL;

        int pbAttrib[] = {
            WGL_PBUFFER_LARGEST_ARB, true,
            WGL_TEXTURE_TARGET_ARB,  WGL_TEXTURE_RECTANGLE_NV,
            WGL_TEXTURE_FORMAT_ARB,  pbTexFormat,
            0, 0
        };
        hpbuffer = wglCreatePbufferARB(hDC, pixelformat, width, height, pbAttrib);
        if (hpbuffer == 0)
            return NULL;
    
        hpbufferDC = wglGetPbufferDCARB(hpbuffer);
        if (hpbufferDC == 0)
            return NULL;
    
        if (sharedhpbufferRC == 0)
            sharedhpbufferRC = wglCreateContext(hpbufferDC);
        hpbufferRC = sharedhpbufferRC; // wglGetCurrentContext();
        if (hpbufferRC == 0)
            return NULL;

        GLint texFormat = WGL_NO_TEXTURE_ARB;
        wglQueryPbufferARB(hpbuffer, WGL_TEXTURE_FORMAT_ARB, &texFormat);
        assert(texFormat != WGL_NO_TEXTURE_ARB);

        if (!wglMakeCurrent(hpbufferDC, hpbufferRC))
            return NULL;

        // share with the window..
        wglShareLists(hwinRC, hpbufferRC);

        // verify double buffered
        if (doublebuffer)
        {
            int attrib = WGL_DOUBLE_BUFFER_ARB;
            int value = -1;
            if (!wglGetPixelFormatAttribivARB(hpbufferDC, pixelformat, 0, 1, &attrib, &value)) {
                printf("Could not get pixel format?\n");
                exit(1);
            }

            bool bDoubleBuffered = (value?true:false);

            if (!bDoubleBuffered) {
                printf("Warning: double buffered pixel format requested and not received!\n");
                exit(4);
            }
        }

        // switch back
        //if (!wglMakeCurrent(hDC, hwinRC))
        //    return NULL;
    }

    // possibly grow viewport to fit
    if (mode != PUG_READ && 
        (width > sharedViewport[0] || height > sharedViewport[1]))
    {
        sharedViewport[0] = max(width, sharedViewport[0]);
        sharedViewport[1] = max(height, sharedViewport[1]);
        glViewport(0, 0, sharedViewport[0], sharedViewport[1]);
    }

    GLuint texHandle;
    glGenTextures(1, &texHandle);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, texHandle);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    GET_GLERROR(NULL);

    PUGBuffer *ret = new PUGBuffer(width, height, components, 
                        hpbufferDC, hpbuffer, hpbufferRC, texHandle, imageBuffer, doublebuffer);
    curBuffer = ret;
    return ret;
}

void
pugDeleteBuffer(PUGBuffer* buf)
{
    if (curBuffer == buf)
    {
        curBuffer = NULL;
    }

    pugReleaseBuffer(buf, PUG_FRONT);
    pugReleaseBuffer(buf, PUG_BACK);
    glDeleteTextures(1, &(buf->texHandle));

    buf->hpbufferRC = 0;
    wglReleasePbufferDCARB(buf->hpbuffer, buf->hpbufferDC);
    wglDestroyPbufferARB(buf->hpbuffer);
    
    delete buf;
    buf = NULL;
}

bool
pugInitBuffer(PUGBuffer *buf, const float *data, PUGTarget target)
{
    if (!initialized || !buf || !data) return false;
    assert(1 == buf->nComponents || 4 == buf->nComponents);
    GLenum internalFormat = GL_FLOAT_R_NV;
    GLenum format = GL_RED;
    switch (buf->nComponents)
    {
    case 1:
        internalFormat = GL_FLOAT_R_NV;
        format = GL_RED;
        break;
    case 4:
        internalFormat = GL_FLOAT_RGBA_NV;
        format = GL_RGBA;
        break;
    default:
        break;
    }
    
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    
    bool ret = true;

    if (NULL != buf->hpbuffer) // it is a pbuffer
    {
        ret = pugBindTexture(buf, target);
    }
    else
    {
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, buf->texHandle);
    }

    if (ret)
    {
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, internalFormat, 
                     buf->width, buf->height, 0, format, GL_FLOAT, data);
        GET_GLERROR(false);
    }
    return ret;
}
    
bool
pugReadMemory(float *out, PUGBuffer *buf)
{
    PUGRect rect(0, buf->width, 0, buf->height);
    return pugReadMemory(out, buf, rect);
}

bool
pugReadMemory(float *out, PUGBuffer *buf, PUGRect rect)
{
    // XXX need to make buffer current if not already...
    if (!initialized) return false;

    assert(buf->nComponents == 1 || buf->nComponents == 4);
    // TODO: handle multiple buffers

    if (curBuffer != buf)
        pugActivateBuffer(buf, buf->currentDrawBuffer);

    if (buf->doublebuffered)
    {
        switch(buf->currentDrawBuffer)
        {
        case PUG_FRONT:
            glReadBuffer(GL_FRONT);
            break;
        case PUG_BACK:
            glReadBuffer(GL_BACK);
            break;
        default:
            break;
        }
    }
    
    glReadPixels(rect.x00, rect.y00, rect.x11, rect.y11,
                 buf->nComponents == 1 ? GL_RED : GL_RGBA, 
                 GL_FLOAT, out);

    GET_GLERROR(false);
    return true;
}


bool
pugReleaseBuffer(PUGBuffer *buf, PUGTarget target)
{
    if (!initialized || buf == NULL || buf->hpbuffer == NULL) return false;

    GLenum targetBuffer = WGL_FRONT_LEFT_ARB;
    if (target == PUG_BACK)
        targetBuffer = WGL_BACK_LEFT_ARB;

    if (!wglReleaseTexImageARB(buf->hpbuffer, targetBuffer))
        return false;
        
    buf->bound[target] = false;
    return true;
}

bool
pugMakeWindowCurrent()
{
    if (!hDC || !hwinRC)
        return false;

    if (TRUE == wglMakeCurrent(hDC, hwinRC))
    {
        curBuffer = NULL;
        return true;
    }

    return false;
}


void
pugWaitForThreads()
{
    glFinish();
}

inline bool 
_isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
_floorPow2(int n)
{
    // method 1
    //float nf = (float)n;
    //return 1 << (((*(int*)&nf) >> 23) - 127); 

    // method 2
    return 1 << (int)_logb((float)n);

    // method 3
    /*int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);*/
}

/*inline int getGreatestDivisor(int n, int start)
{
    assert(isPowerOfTwo(start));

    int gd = start;

    do 
    {
        if (!(n % start)) return gd;
    	gd >>= 1;
    } while(gd > 1);

    return gd;
}*/

inline void 
_reduceStep1D_rows(PUGProgram* prog, PUGBuffer* dstbuf, PUGTarget target, 
                   int cols, int rows)
{
    PUGRect range = PUGRect(0, cols, 0, rows);
    PUGRect domain = PUGRect(cols, 2*cols, 0, rows);
    if (!pugBindDomain(prog, "offset2", domain)) {
        fprintf(stderr, "Couldn't bind offset2 domain.\n");
        exit(1);
    }
    if (!pugRunProgram(prog, dstbuf, range, target)) {
        fprintf(stderr, "Couldn't run sum prog.\n");
        exit(1);
    }
}

inline void 
_reduceStep1D_cols(PUGProgram* prog, PUGBuffer* dstbuf, PUGTarget target, 
                   int cols, int rows)
{
    PUGRect range = PUGRect(0, cols, 0, rows);
    PUGRect domain = PUGRect(0, cols, rows, 2*rows);
    if (!pugBindDomain(prog, "offset2", domain)) {
        fprintf(stderr, "Couldn't bind offset2 domain.\n");
        exit(1);
    }
    if (!pugRunProgram(prog, dstbuf, range, target)) {
        fprintf(stderr, "Couldn't run sum prog.\n");
        exit(1);
    }
}


inline void 
_reduceStep2D(PUGProgram* prog, PUGBuffer* dstbuf, PUGTarget target,
              int cols, int rows)
{

    PUGRect range = PUGRect(0, cols, 0, rows);
    PUGRect domain = PUGRect(cols, 2*cols, 0, rows);

    if (!pugBindDomain(prog, "offset2", domain)) {
        fprintf(stderr, "Couldn't bind offset2 domain.\n");
        exit(1);
    }
    domain = PUGRect(cols, 2*cols, rows, 2*rows);
    if (!pugBindDomain(prog, "offset3", domain)) {
        fprintf(stderr, "Couldn't bind offset3 domain.\n");
        exit(1);
    }
    domain = PUGRect(0, cols, rows, 2*rows);
    if (!pugBindDomain(prog, "offset4", domain)) {
        fprintf(stderr, "Couldn't bind offset4 domain.\n");
        exit(1);
    }
    if (!pugRunProgram(prog, dstbuf, range, target)) {
        fprintf(stderr, "Couldn't run sum prog.\n");
        exit(1);
    }
}

PUGBuffer* 
pugReduce2D(PUGProgram *prog, PUGBuffer *inbuf, PUGBuffer *dblbuf, 
            int nRows, int nCols)
{
    assert(dblbuf->doublebuffered);
    assert(prog->progType == PUGProgram::PUG_PROGRAM_REDUCTION);

    if (nRows <= 1 && nCols <= 1) return inbuf;

    PUGBuffer *srcbuf = NULL, *dstbuf = NULL;

    if (_isPowerOfTwo(nRows) && _isPowerOfTwo(nCols))
    {
        prog->prog = prog->reduceProgs[1]; // switch to 2D reduction program

        int pass = 0;
        PUGRect domain, range;

        srcbuf = inbuf;
        dstbuf = dblbuf;

        PUGTarget target = PUG_BACK;
        PUGTarget source = PUG_FRONT;
        if (inbuf->doublebuffered)
        {
            source = inbuf->currentDrawBuffer;
        }

        for (int m = nCols / 2, n = nRows / 2; m > 0 || n > 0; m /= 2, n /= 2)
        {
            target = (source == PUG_BACK) ? PUG_FRONT : PUG_BACK;
            if (!pugBindStream(prog, "src", srcbuf, source)) {
                fprintf(stderr, "Couldn't bind src buffer.\n");
                exit(1);
            }

            if (m > 0 && n > 0)
            {
                _reduceStep2D(prog, dstbuf, target, m, n);
            }
            else {
                prog->prog = prog->reduceProgs[0]; // switch to 1D reduction program

                if (m > 0)
                    _reduceStep1D_rows(prog, dstbuf, target, m, 1);
                else
                    _reduceStep1D_cols(prog, dstbuf, target, 1, n);
            }

            srcbuf = dblbuf;
            source = target;
        }
    }
    else
    {
        prog->prog = prog->reduceProgs[0]; // switch to 1D reduction program

        // reduce rows, then reduce last column
        dstbuf = pugReduce1D(prog, inbuf, dblbuf, PUG_DIMENSION_X, nRows, nCols);
        dstbuf = pugReduce1D(prog, dstbuf, dblbuf, PUG_DIMENSION_Y, nRows, 1);
    }

    return dstbuf;
}

// Perform reduction on rows of vector in inbuf, using front/back of dblbuf for ping-ponging,
// thereby preserving inbuf itself.
// 
// Returns final result as a pointer to dblbuf, which holds sum in element 0 of each row.
// Note that dblbuf MUST BE DOUBLE BUFFERED
PUGBuffer*
pugReduce1D(PUGProgram *prog, PUGBuffer *inbuf, PUGBuffer *dblbuf, 
            PUGDimension dim, int nRows, int nCols)
{
    assert(dblbuf->doublebuffered);
    assert(prog->progType == PUGProgram::PUG_PROGRAM_REDUCTION);
    assert(dim == PUG_DIMENSION_X || dim == PUG_DIMENSION_Y);

    int nReduceDim = (dim == PUG_DIMENSION_X) ? nCols : nRows;
    if (nReduceDim <= 1) return inbuf;
    prog->prog = prog->reduceProgs[0]; // switch to 1D reduction program

    int pass = 0;
    PUGRect domain, range;
    PUGBuffer *srcbuf, *dstbuf;
    
    srcbuf = inbuf;
    dstbuf = dblbuf;
    PUGTarget target = PUG_BACK;
    PUGTarget source = PUG_FRONT;
    if (inbuf->doublebuffered)
    {
        source = inbuf->currentDrawBuffer;
    }

    if (_isPowerOfTwo(nReduceDim))
    {

        for (int n = nReduceDim / 2; n > 0; n = n / 2)
        {
            target = (source == PUG_BACK) ? PUG_FRONT : PUG_BACK;
            if (!pugBindStream(prog, "src", srcbuf, source)) {
                fprintf(stderr, "Couldn't bind src buffer.\n");
                exit(1);
            }

            if (dim == PUG_DIMENSION_X)
                _reduceStep1D_rows(prog, dstbuf, target, n, nRows);
            else
                _reduceStep1D_cols(prog, dstbuf, target, nCols, n);

            srcbuf = dblbuf;
            source = target;
        }
    }
    else
    {
        PUGProgram* passthru = passthruProg[inbuf->nComponents - 1];
        int n = nReduceDim;
        int p2 = 0, np2 = 0;
        int p2l = 0, p2m = 0, p2r = 0;

        while(n > 1)
        {
            if (_isPowerOfTwo(n))
            {
                p2 = n;
                np2 = 0;
            }
            else
            {
                p2 = _floorPow2(n);
                np2 = n - p2;
            }

            p2l = np2;
            p2r = p2l + p2;
            p2m = p2l + p2 / 2;

            target = (source == PUG_BACK) ? PUG_FRONT : PUG_BACK;

            if (pass < 2 && np2 > 0)
            {
                if (!pugBindStream(passthru, "src", srcbuf, source)) {
                    fprintf(stderr, "Couldn't bind src buffer.\n");
                    exit(1);
                }
                range = (dim == PUG_DIMENSION_X) ?
                    PUGRect(0, p2l, 0, nRows) : PUGRect(0, nCols, 0, p2l);
                if (!pugRunProgram(passthru, dstbuf, range, target)) {
                    fprintf(stderr, "Couldn't run passthru prog.\n");
                    exit(1);
                }
            }

            if (!pugBindStream(prog, "src", srcbuf, source)) 
            {
                fprintf(stderr, "Couldn't bind src buffer.\n");
                exit(1);
            }

            if (dim == PUG_DIMENSION_X)
            {
                domain = PUGRect(p2m, p2r, 0, nRows);
                range = PUGRect(p2l, p2m, 0, nRows);
            }
            else
            {
                domain = PUGRect(0, nCols, p2m, p2r);
                range = PUGRect(0, nCols, p2l, p2m);
            }
            if (!pugBindDomain(prog, "offset2", domain)) 
            {
                fprintf(stderr, "Couldn't bind offset2 domain.\n");
                exit(1);
            }
            if (!pugRunProgram(prog, dstbuf, range, target)) 
            {
                fprintf(stderr, "Couldn't run sum prog.\n");
                exit(1);
            }

            pass++;
            n = p2m;
            srcbuf = dblbuf;
            source = target;
        }
    }
    return dstbuf;
}
