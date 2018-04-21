#ifndef WGL_PIXEL_FORMAT_ARB_H
#define WGL_PIXEL_FORMAT_ARB_H


//
// Includes
//

#include <GL/wglext.h>
#include <GL/glext.h>


extern PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglGetPixelFormatAttribivARB;
extern PFNWGLGETPIXELFORMATATTRIBFVARBPROC wglGetPixelFormatAttribfvARB;
extern PFNWGLCHOOSEPIXELFORMATARBPROC      wglChoosePixelFormatARB;


void initPixelFormatARB(HDC hDC);

#endif // WGL_PIXEL_FORMAT_ARB_H