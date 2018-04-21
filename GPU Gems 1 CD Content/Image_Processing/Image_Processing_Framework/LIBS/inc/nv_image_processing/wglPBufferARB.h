#ifndef WGL_P_BUFFER_ARB_H
#define WGL_P_BUFFER_ARB_H


//
// Includes
//

#include <windows.h>
#include <GL/gl.h>
#include <GL/wglext.h>


extern PFNWGLCREATEPBUFFERARBPROC      wglCreatePbufferARB;
extern PFNWGLGETPBUFFERDCARBPROC       wglGetPbufferDCARB;
extern PFNWGLRELEASEPBUFFERDCARBPROC   wglReleasePbufferDCARB;
extern PFNWGLDESTROYPBUFFERARBPROC     wglDestroyPbufferARB;
extern PFNWGLQUERYPBUFFERARBPROC       wglQueryPbufferARB;


void initPBufferARB(HDC hDC);

#endif // WGL_P_BUFFER_ARB_H