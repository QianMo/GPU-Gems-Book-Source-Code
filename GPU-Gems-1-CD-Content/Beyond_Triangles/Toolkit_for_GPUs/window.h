#ifndef WINDOW_H
#define WINDOW_H

#include "GL/glew.h"
#include "GL/wglew.h"

class Window{
public:
  Window(){
    ispbuffer = false;
    pixels = 0;
  };

  ~Window(){
    Release();
  };

  void Release() {
    if( ispbuffer ){
      //if( wglGetCurrentContext() == rc )
      //  wglMakeCurrent(0,0);
      //wglDeleteContext(rc);
      wglReleasePbufferDCARB(pixels, dc);
      wglDestroyPbufferARB(pixels);
      ispbuffer = false;
    }
  }

  void MakeCurrent(){

    HDC curdc = wglGetCurrentDC();

    HGLRC currc = wglGetCurrentContext();

    if( curdc != dc || currc != rc){

      //fprintf(stderr, "\tswitching to %s\n", name);
      wglMakeCurrent(dc, rc);

    }

    //else

      //fprintf(stderr, "\tredundant switch to %s\n", name);
  };

  void SetContext(){
    dc = wglGetCurrentDC();
    rc = wglGetCurrentContext();

    strcpy(name, "colorbuffer");
  };

  int CreateContext( int w, int h, const char* n ){
    ispbuffer = true;
    int in_attribs[] = {
      WGL_RED_BITS_ARB,               32,
      WGL_GREEN_BITS_ARB,             32,
      WGL_BLUE_BITS_ARB,              32,
      WGL_ALPHA_BITS_ARB,             32,
      WGL_COLOR_BITS_ARB,            128,  
      WGL_DEPTH_BITS_ARB,             24,
      WGL_STENCIL_BITS_ARB,            8,
      WGL_FLOAT_COMPONENTS_NV,        GL_TRUE,
      WGL_DRAW_TO_PBUFFER_ARB,        GL_TRUE,
      //WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV, GL_TRUE,
      0,0
    };
    
    HDC old_dc = wglGetCurrentDC();
    int format;
	  unsigned int n_formats;
    HPBUFFERARB pbuffer;
    HDC new_dc;
    HGLRC new_rc;

    //int pbuffer_attribs[] = { WGL_TEXTURE_FORMAT_ARB,         WGL_TEXTURE_FLOAT_RGBA_NV,
    //                          WGL_TEXTURE_TARGET_ARB,         WGL_TEXTURE_RECTANGLE_NV,
    //                          0,0
    //                        };

    int pbuffer_attribs[] = { 0 };

    if (!wglChoosePixelFormatARB(old_dc, in_attribs, NULL, 1, &format, &n_formats))
    {
        fprintf(stderr, "Failed to choose a suitable pixel format.\n");
        return -1;
    }

    if (!(pbuffer = wglCreatePbufferARB(old_dc, format, w, h, pbuffer_attribs)))
    {
        fprintf(stderr, "Pbuffer creation failed\n");
        return -1;
    }

    if (!(new_dc = wglGetPbufferDCARB(pbuffer)))
    {
        fprintf(stderr, "Couldn't get pbuffer device context\n");
        return -1;
    }

    if (!(new_rc = wglCreateContext(new_dc)))
    {
        fprintf(stderr, "Couldn't create pbuffer rendering context\n");
        return -1;
    }

    if (!wglShareLists(wglGetCurrentContext(), new_rc))
    {
        fprintf(stderr, "Couldn't share display list (wglShareLists)\n");
        return -1;
    }

    dc = new_dc;
    rc = new_rc;
    pixels = pbuffer;
    strcpy(name, n);
    return 1;
  };

  HPBUFFERARB GetPBuffer() { return pixels; }
private:
  HDC dc;
  HGLRC rc;
  HPBUFFERARB pixels;
  bool ispbuffer;

  char name[1024];

};

#endif /*WINDOW_H*/