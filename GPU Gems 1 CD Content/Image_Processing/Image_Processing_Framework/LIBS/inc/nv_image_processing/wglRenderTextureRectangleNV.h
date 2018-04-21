#ifndef WGL_RENDER_TEXTURE_RECTANGLE_NV
#define WGL_RENDER_TEXTURE_RECTANGLE_NV


//
// Includes
//

#include <GL/wglext.h>



extern PFNWGLBINDTEXIMAGEARBPROC        wglBindTexImageARB;
extern PFNWGLRELEASETEXIMAGEARBPROC     wglReleaseTexImageARB;
extern PFNWGLSETPBUFFERATTRIBARBPROC    wglSetPbufferAttribARB;


void initRenderTextureARB(HDC hDC);


#endif // WGL_RENDER_TEXTURE_RECTANGLE_NV
