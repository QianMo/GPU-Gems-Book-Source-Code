#ifndef OGLFUNC_HPP
#define OGLFUNC_HPP

// MCH: We are going to override Linux GLEXT prototypes to get around linux ABI issues
#ifdef WIN32
#include <windows.h>
#define GL_GLEXT_PROTOTYPES 1
#endif

#include <GL/gl.h>

#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

#ifndef GL_VERSION_1_1
#error GL ERROR: The gl.h version on this computer is very old.
#endif

#ifndef GL_VERSION_1_2
typedef void (APIENTRYP PFNGLMULTITEXCOORD2FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (APIENTRYP PFNGLMULTITEXCOORD4FVARBPROC) (GLenum target, const GLfloat *v);
#define RUNTIME_BONUS_GL_FNS_1 \
   XXX(PFNGLMULTITEXCOORD2FVARBPROC,   glMultiTexCoord2fvARB) \
   XXX(PFNGLMULTITEXCOORD4FVARBPROC,   glMultiTexCoord4fvARB)
#else
#define RUNTIME_BONUS_GL_FNS_1
#endif

#ifndef GL_ARB_multitexture
typedef void (APIENTRYP PFNGLACTIVETEXTUREARBPROC) (GLenum texture);
#define GL_TEXTURE0_ARB                   0x84C0
#define GL_MAX_TEXTURE_UNITS              0x84E2
#define RUNTIME_BONUS_GL_FNS_2 \
   XXX(PFNGLACTIVETEXTUREARBPROC,      glActiveTextureARB)
#else
#define RUNTIME_BONUS_GL_FNS_2
#endif

#ifndef GL_TEXTURE0
#define GL_TEXTURE0 GL_TEXTURE0_ARB
#endif
#ifndef GL_NV_fragment_program
typedef void (APIENTRY * PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC) (GLuint id, GLsizei len, const GLubyte *name, const GLfloat* x);
#define RUNTIME_BONUS_GL_FNS_NV \
   XXX(PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC,        glProgramNamedParameter4fvNV)   
#elif !GL_GLEXT_PROTOTYPES   
typedef void (APIENTRY * PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC) (GLuint id, GLsizei len, const GLubyte *name, const GLfloat* x);
#define RUNTIME_BONUS_GL_FNS_NV \
   XXX(PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC,        glProgramNamedParameter4fvNV)            
#else
#define RUNTIME_BONUS_GL_FNS_NV
#endif

#ifndef GL_ARB_vertex_program
typedef void (APIENTRYP PFNGLGENPROGRAMSARBPROC) (GLsizei n, GLuint *programs);
typedef void (APIENTRYP PFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (APIENTRYP PFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const GLvoid *string);
typedef void (APIENTRYP PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) (GLenum target, GLuint index, const GLfloat *params);
#define GL_VERTEX_PROGRAM_ARB             0x8620
#define GL_PROGRAM_ERROR_POSITION_ARB     0x864B
#define GL_PROGRAM_ERROR_STRING_ARB       0x8874
#define GL_PROGRAM_FORMAT_ASCII_ARB       0x8875
#define RUNTIME_BONUS_GL_FNS_3 \
   XXX(PFNGLGENPROGRAMSARBPROC,        glGenProgramsARB)               \
   XXX(PFNGLBINDPROGRAMARBPROC,        glBindProgramARB)               \
   XXX(PFNGLPROGRAMSTRINGARBPROC,      glProgramStringARB)             \
   XXX(PFNGLPROGRAMLOCALPARAMETER4FVARBPROC, glProgramLocalParameter4fvARB)
#elif !GL_GLEXT_PROTOTYPES
typedef void (APIENTRYP PFNGLGENPROGRAMSARBPROC) (GLsizei n, GLuint *programs);
typedef void (APIENTRYP PFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (APIENTRYP PFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const GLvoid *string);
typedef void (APIENTRYP PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) (GLenum target, GLuint index, const GLfloat *params);
#define RUNTIME_BONUS_GL_FNS_3 \
   XXX(PFNGLGENPROGRAMSARBPROC,        glGenProgramsARB)               \
   XXX(PFNGLBINDPROGRAMARBPROC,        glBindProgramARB)               \
   XXX(PFNGLPROGRAMSTRINGARBPROC,      glProgramStringARB)             \
   XXX(PFNGLPROGRAMLOCALPARAMETER4FVARBPROC, glProgramLocalParameter4fvARB)
#else
#define RUNTIME_BONUS_GL_FNS_3
#endif

#ifndef GL_ARB_fragment_program
#define GL_FRAGMENT_PROGRAM_ARB           0x8804
#define GL_MAX_TEXTURE_COORDS_ARB                  0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS_ARB             0x8872
#endif

#ifndef GL_NV_texture_rectangle
#define GL_TEXTURE_RECTANGLE_NV           0x84F5
#endif

#ifndef GL_ARB_texture_rectangle
#define GL_TEXTURE_RECTANGLE_ARB          0x84F5
#endif

#ifndef GL_ATI_draw_buffers
#define GL_MAX_DRAW_BUFFERS_ATI              0x8824

#define GL_DRAW_BUFFER0_ATI                  0x8825
#define GL_DRAW_BUFFER1_ATI                  0x8826
#define GL_DRAW_BUFFER2_ATI                  0x8827
#define GL_DRAW_BUFFER3_ATI                  0x8828
typedef void (APIENTRYP PFNGLDRAWBUFFERSATIPROC) (GLsizei n, const GLenum *bufs);
#endif
#define RUNTIME_BONUS_GL_FNS_ATI \
   XXX(PFNGLDRAWBUFFERSATIPROC,        glDrawBuffersATI)

#define RUNTIME_BONUS_GL_FNS \
   RUNTIME_BONUS_GL_FNS_1 RUNTIME_BONUS_GL_FNS_2 RUNTIME_BONUS_GL_FNS_3

/***** WGL API *****/
#ifdef WIN32

#ifndef WGL_ARB_pbuffer
DECLARE_HANDLE(HPBUFFERARB);
typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int *piAttribList);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer, HDC hDC);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
#endif

#ifndef WGL_ARB_pixel_format
typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
#endif

#ifndef WGL_ARB_render_texture
typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
#endif



#define RUNTIME_BONUS_WGL_FNS \
   XXX(PFNWGLCREATEPBUFFERARBPROC,     wglCreatePbufferARB)     \
   XXX(PFNWGLGETPBUFFERDCARBPROC,      wglGetPbufferDCARB)      \
   XXX(PFNWGLRELEASEPBUFFERDCARBPROC,  wglReleasePbufferDCARB)  \
   XXX(PFNWGLDESTROYPBUFFERARBPROC,    wglDestroyPbufferARB)    \
   XXX(PFNWGLCHOOSEPIXELFORMATARBPROC, wglChoosePixelFormatARB) \
   XXX(PFNWGLBINDTEXIMAGEARBPROC,      wglBindTexImageARB)      \
   XXX(PFNWGLRELEASETEXIMAGEARBPROC,   wglReleaseTexImageARB)
#endif // WIN32
  

/* Declare undefined functions */
#define XXX(type, fn) \
   extern type fn;

#ifdef WIN32
RUNTIME_BONUS_WGL_FNS
#endif

RUNTIME_BONUS_GL_FNS
RUNTIME_BONUS_GL_FNS_ATI
RUNTIME_BONUS_GL_FNS_NV

#undef XXX

namespace brook {
  void initglfunc(void);
}

#endif

