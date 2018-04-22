//
// Includes
//

#include <windows.h>
#include <GL/gl.h>
#include "wglRenderTextureARB.h"

#include "wglExtensionsStringARB.h"


PFNWGLBINDTEXIMAGEARBPROC        wglBindTexImageARB;
PFNWGLRELEASETEXIMAGEARBPROC     wglReleaseTexImageARB;
PFNWGLSETPBUFFERATTRIBARBPROC    wglSetPbufferAttribARB;


void initRenderTextureARB(HDC hDC)
{
    ASSERT_EXTENSION_SUPPORT(WGL_ARB_render_texture);

    INIT_FUNCT_PTR(wglBindTexImageARB,      PFNWGLBINDTEXIMAGEARBPROC);
    INIT_FUNCT_PTR(wglReleaseTexImageARB,   PFNWGLRELEASETEXIMAGEARBPROC);
    INIT_FUNCT_PTR(wglSetPbufferAttribARB,  PFNWGLSETPBUFFERATTRIBARBPROC);
}
