//
// Includes
//

#include <windows.h>
#include <GL/gl.h>

#include "wglPixelFormatARB.h"
#include "wglExtensionsStringARB.h"


PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglGetPixelFormatAttribivARB;
PFNWGLGETPIXELFORMATATTRIBFVARBPROC wglGetPixelFormatAttribfvARB;
PFNWGLCHOOSEPIXELFORMATARBPROC      wglChoosePixelFormatARB;


void initPixelFormatARB(HDC hDC)
{
    ASSERT_EXTENSION_SUPPORT(WGL_ARB_pixel_format);

    INIT_FUNCT_PTR( wglGetPixelFormatAttribivARB,   PFNWGLGETPIXELFORMATATTRIBIVARBPROC );
    INIT_FUNCT_PTR( wglGetPixelFormatAttribfvARB,   PFNWGLGETPIXELFORMATATTRIBFVARBPROC );
    INIT_FUNCT_PTR( wglChoosePixelFormatARB,        PFNWGLCHOOSEPIXELFORMATARBPROC      );
}
