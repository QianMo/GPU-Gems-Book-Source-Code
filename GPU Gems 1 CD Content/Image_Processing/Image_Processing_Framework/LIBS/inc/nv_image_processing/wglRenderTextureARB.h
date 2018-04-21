#ifndef WGL_RENDER_TEXTURE_ARB_H
#define WGL_RENDER_TEXTURE_ARB_H


//
// Includes
//

#include <GL/wglext.h>



extern PFNWGLBINDTEXIMAGEARBPROC        wglBindTexImageARB;
extern PFNWGLRELEASETEXIMAGEARBPROC     wglReleaseTexImageARB;
extern PFNWGLSETPBUFFERATTRIBARBPROC    wglSetPbufferAttribARB;


void initRenderTextureARB(HDC hDC);



#endif // WGL_RENDER_TEXTURE_ARB_H