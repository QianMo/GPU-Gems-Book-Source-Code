// oglwindow.hpp
#ifndef OGLWINDOW_HPP
#define OGLWINDOW_HPP

#ifdef WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#include <GL/glx.h>
#endif

#include "oglfunc.hpp"
#ifndef WIN32
typedef void *HGLRC; 
#endif
namespace brook {
  
  class OGLWindow {
    
  public:
    OGLWindow();
    ~OGLWindow();
    
    void initPbuffer( const int   (*viAttribList)[4][64],
                      const float (*vfAttribList)[4][16],
                      const int   (*vpiAttribList)[4][16]);
    
    bool bindPbuffer(unsigned int width,
                     unsigned int height,
                     unsigned int numOutputs, 
                     unsigned int numComponents);

    void makeCurrent();
    void shareLists(HGLRC inContext );
    
  private:
    
#ifdef WIN32
    HGLRC hglrc;
    HGLRC hglrc_window;
    HPBUFFERARB hpbuffer;
    HWND hwnd;
    HDC hwindowdc;
    HDC hpbufferdc;

    int pixelformat[4][4];
    int piAttribList[4][16];

#else
     /* See note in oglwindow.cpp for why things are static */ 
     static Display   *pDisplay;
     static int iScreen;
     static Window     glxWindow;
     static Colormap cmap;
     static XVisualInfo *visual;
     static GLXFBConfig *glxConfig[4];
     static GLXPbuffer  glxPbuffer;
     static GLXContext  glxContext;
     static int piAttribList[4][16];

     static bool static_window_created;
     static bool static_pbuffers_initialized;
#endif
    
    unsigned int currentPbufferComponents;
    unsigned int currentPbufferOutputs;
    unsigned int currentPbufferWidth;
    unsigned int currentPbufferHeight;
  };

}


#endif

