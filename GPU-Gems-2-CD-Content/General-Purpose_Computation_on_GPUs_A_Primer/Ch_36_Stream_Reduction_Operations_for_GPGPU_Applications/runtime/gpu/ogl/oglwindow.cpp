
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../gpucontext.hpp"

#include "oglfunc.hpp"
#include "oglwindow.hpp"

using namespace brook;

static const char window_name[] = "Brook GL Render Window";


static void 
appendVendorAttribs( int   iAttribList[4][4][64],
                     float fAttribList[4][16],
                     int   piAttribList[4][16],
                     const int   (*viAttribList)[4][64],
                     const float (*vfAttribList)[4][16],
                     const int   (*vpiAttribList)[4][16]) {

  int i,j,k,m;

  for (m=0; m<4; m++) {
    for (i=0; i<4; i++) {
      for (j=0; j<63; j+=2) {
        GPUAssert(j<64, "Error: no room for base attribs");
        if (iAttribList[m][i][j]   == 0 && 
            iAttribList[m][i][j+1] == 0)
          break;
      }
      
      if (viAttribList) {
        for (k=0; k<63; k+=2) { 
          GPUAssert(j<64, "Error: no room for vendor attribs");
          
          iAttribList[m][i][j++]  = (*viAttribList)[i][k];
          iAttribList[m][i][j++]  = (*viAttribList)[i][k+1];
          
          if ((*viAttribList)[i][k]   == 0 && 
              (*viAttribList)[i][k+1] == 0)
            break;
        }
      }
    }
  }

  for (i=0; i<4; i++) {
    for (j=0; j<16; j+=2) {
      GPUAssert(j<16, "Error: no room for base attribs");
      if (fAttribList[i][j]   == 0.0f && 
          fAttribList[i][j+1] == 0.0f)
        break;
    }
    

    if (vfAttribList) {
      for (k=0; k<16; k+=2) { 
        GPUAssert(j<16, "Error: no room for vendor attribs");
        
        fAttribList[i][j++]  = (*vfAttribList)[i][k];
        fAttribList[i][j++]  = (*vfAttribList)[i][k+1];
        
        if ((*vfAttribList)[i][k]   == 0.0f && 
            (*vfAttribList)[i][k+1] == 0.0f)
          break;
      }
    }


    for (j=0; j<16; j+=2) {
      GPUAssert(j<16, "Error: no room for base attribs");
      if (piAttribList[i][j]   == 0 && 
          piAttribList[i][j+1] == 0)
        break;
    }

    if (vpiAttribList) {
      for (k=0; k<16; k+=2) { 
        GPUAssert(j<16, "Error: no room for vendor attribs");
        
        piAttribList[i][j++]  = (*vpiAttribList)[i][k];
        piAttribList[i][j++]  = (*vpiAttribList)[i][k+1];
        
        if ((*vpiAttribList)[i][k]   == 0 && 
            (*vpiAttribList)[i][k+1] == 0)
          break;
      }
    }
  }
}


#ifdef WIN32

#ifndef WGL_ARB_pixel_format
#define WGL_ACCELERATION_ARB           0x2003
#define WGL_SUPPORT_OPENGL_ARB         0x2010
#define WGL_DOUBLE_BUFFER_ARB          0x2011
#define WGL_COLOR_BITS_ARB             0x2014
#define WGL_RED_BITS_ARB               0x2015
#define WGL_GREEN_BITS_ARB             0x2017
#define WGL_BLUE_BITS_ARB              0x2019
#define WGL_DEPTH_BITS_ARB             0x2022
#define WGL_ALPHA_BITS_ARB             0x201B
#define WGL_STENCIL_BITS_ARB           0x2023
#define WGL_AUX_BUFFERS_ARB            0x2024
#define WGL_FULL_ACCELERATION_ARB      0x2027
#endif

#ifndef WGL_ARB_pbuffer
#define WGL_DRAW_TO_PBUFFER_ARB        0x202D
#endif

#define CRGBAX(c,r,g,b,a,x) \
  WGL_DRAW_TO_PBUFFER_ARB, GL_TRUE, \
  WGL_ACCELERATION_ARB,    WGL_FULL_ACCELERATION_ARB, \
  WGL_DEPTH_BITS_ARB,      0,\
  WGL_STENCIL_BITS_ARB,    0,\
  WGL_DOUBLE_BUFFER_ARB,   GL_FALSE, \
  WGL_SUPPORT_OPENGL_ARB,  GL_TRUE, \
  WGL_AUX_BUFFERS_ARB,     x, \
  WGL_COLOR_BITS_ARB,      c, \
  WGL_RED_BITS_ARB,        r, \
  WGL_GREEN_BITS_ARB,      g, \
  WGL_BLUE_BITS_ARB,       b, \
  WGL_ALPHA_BITS_ARB,      a

#define BASEATTRIB(x) \
{CRGBAX(32,32,0,0,0,x), 0, 0}, \
{CRGBAX(64,32,32,0,0,x), 0, 0}, \
{CRGBAX(96,32,32,32,0,x), 0, 0}, \
{CRGBAX(128,32,32,32,32,x), 0, 0}


static const int 
baseiAttribList[4][4][64] = { {BASEATTRIB(0)},
                              {BASEATTRIB(1)},
                              {BASEATTRIB(2)},
                              {BASEATTRIB(3)} };

static const float
basefAttribList[4][16] = { {0.0f,0.0f}, {0.0f,0.0f}, {0.0f,0.0f}, {0.0f,0.0f}};
                           
static int
basepiAttribList[4][16] = { {0, 0}, {0, 0}, {0, 0}, {0, 0} };


static HWND
create_window (void) {
  HINSTANCE hinstance;
  WNDCLASS wc;
  DWORD window_style;
  HWND hwnd;

  /*
   * These parameters are useless since the window is never shown nor
   * rendered into.
   */
  const int window_width = 10;
  const int window_height = 10;
  const int window_x = 0;
  const int window_y = 0;

  hinstance = GetModuleHandle( NULL );

  // Create the window class
  if (!GetClassInfo(hinstance, window_name, &wc)) {
      wc.style = CS_OWNDC;
      wc.lpfnWndProc = (WNDPROC) DefWindowProc;
      wc.cbClsExtra = 0;
      wc.cbWndExtra = 0;
      wc.hInstance = hinstance;
      wc.hIcon = LoadIcon( NULL, IDI_APPLICATION );
      wc.hCursor = LoadCursor( NULL, IDC_ARROW );
      wc.hbrBackground = NULL;
      wc.lpszMenuName = NULL;
      wc.lpszClassName = window_name;

      ATOM result = RegisterClass( &wc );
      assert (result);
    }

  window_style = ( WS_CLIPSIBLINGS | WS_CLIPCHILDREN );
  window_style |= WS_POPUP;

  // Create the window
  hwnd = CreateWindow( window_name, window_name,
		       window_style,
		       window_x,
		       window_y,
		       window_width,
		       window_height,
		       NULL, NULL, hinstance, NULL );

  if (!hwnd)
    GPUError ("Failed to create window");

  return hwnd;
}


/*
 * bSetupPixelFormat --
 *
 * This function creates a default GL context which is never really used by
 * Brook but is needed to construct a pbuffer
 */

static BOOL
bSetupPixelFormat(HDC hdc)
{
  int status;
  PIXELFORMATDESCRIPTOR *ppfd;
  PIXELFORMATDESCRIPTOR pfd = {
    sizeof(PIXELFORMATDESCRIPTOR),  //  size of this pfd
    1,                              // version number
    PFD_DRAW_TO_WINDOW |            // support window
    PFD_SUPPORT_OPENGL |            // support OpenGL
    PFD_DOUBLEBUFFER,               // double buffered
    PFD_TYPE_RGBA,                  // RGBA type
    0,                              // 24-bit color depth
    0, 0, 0, 0, 0, 0,               // color bits ignored
    0,                              // no alpha buffer
    0,                              // shift bit ignored
    0,                              // no accumulation buffer
    0, 0, 0, 0,                     // accum bits ignored
    0,                              // set depth buffer
    0,                              // set stencil buffer
    0,                              // no auxiliary buffer
    PFD_MAIN_PLANE,                 // main layer
    0,                              // reserved
    0, 0, 0                         // layer masks ignored
  };
  int pixelformat;
  
  ppfd = &pfd;

  pixelformat = ChoosePixelFormat( hdc, ppfd );
  
  GPUAssert(pixelformat, "ChoosePixelFormat failed");
  
  status = SetPixelFormat(hdc, pixelformat, ppfd);
  
  GPUAssert(status, "SetPixelFormat failed");
  
  return TRUE;
}


OGLWindow::OGLWindow() {
  BOOL status;

  /* Create a window */
  hwnd = create_window();
  hwindowdc = GetDC(hwnd);

  /* Initialize the initial window GL_context */
  status = bSetupPixelFormat(hwindowdc);
  GPUAssert(status,
            "Unable to set window pixel format");

  hglrc_window = wglCreateContext(hwindowdc);
  GPUAssert(hglrc_window,
            "Unable to create window GL context");

  status = wglMakeCurrent(hwindowdc, hglrc_window);
  GPUAssert(status,
            "Unable to make current the window GL context");


  initglfunc();

  hglrc = NULL;
  hpbuffer = NULL;
  hpbufferdc = NULL;
}

OGLWindow::~OGLWindow() {
  wglMakeCurrent(hwindowdc, NULL);
  wglDeleteContext(hglrc_window);

  if (hglrc)
    wglDeleteContext(hglrc);
  if (hpbuffer)
    wglDestroyPbufferARB(hpbuffer);

  DeleteDC(hwindowdc);
  DestroyWindow(hwnd);
}


void 
OGLWindow::initPbuffer( const int   (*viAttribList)[4][64],
                        const float (*vfAttribList)[4][16],
                        const int   (*vpiAttribList)[4][16]) {

  int   (*iAttribList)[4][64]  = (int   (*)[4][64])
    malloc (sizeof(baseiAttribList));
  float (*fAttribList)[16]     = (float (*)[16]) 
	malloc (sizeof(basefAttribList));
 
  unsigned int numFormats;
  BOOL status;

  memcpy(iAttribList,  baseiAttribList,  sizeof(baseiAttribList));
  memcpy(fAttribList,  basefAttribList,  sizeof(basefAttribList));
  memcpy(piAttribList, basepiAttribList, sizeof(basepiAttribList));

  /* Initialize gl functions */
  initglfunc();

  /* Append vendor specific attribs */
  appendVendorAttribs( iAttribList, fAttribList, piAttribList,
                       viAttribList, vfAttribList, vpiAttribList);

  for (int ncomp = 0; ncomp < 4; ncomp++) {
    /* Fetch the pixel formats for pbuffers */
    for (int i=0; i<4; i++) {
      status = wglChoosePixelFormatARB(hwindowdc, iAttribList[ncomp][i],
                                       fAttribList[i], 1,
                                       &(pixelformat[ncomp][i]),
                                       &numFormats);
    
    GPUAssert(numFormats > 0 && status,
              "ChoosePixelFormat failed to find a pbuffer format");
    }
  }

  currentPbufferComponents = 4;
  currentPbufferWidth = 16;
  currentPbufferHeight = 16;
  currentPbufferOutputs = 1;

  hpbuffer = wglCreatePbufferARB(hwindowdc,
                                 pixelformat
                                 [currentPbufferOutputs-1]
                                 [currentPbufferComponents-1],
                                 currentPbufferWidth,
                                 currentPbufferHeight,
                                 piAttribList[currentPbufferComponents-1]);

  if (!hpbuffer)
    GPUError("Failed to create float pbuffer");

  hpbufferdc = wglGetPbufferDCARB (hpbuffer);

  if (!hpbufferdc)
    GPUError("Failed to get pbuffer DC");

  hglrc = wglCreateContext( hpbufferdc );

  if (!hglrc)
    GPUError("Failed to create GL context");

  if (!wglMakeCurrent( hpbufferdc, hglrc ))
    GPUError("Failed to bind GL context");

}


bool
OGLWindow::bindPbuffer(unsigned int width,
                       unsigned int height,
                       unsigned int numOutputs,
                       unsigned int numComponents) {

  unsigned int i;
  bool switched_contexts = false;

  /* If the pbuffer of the right size is already active,
  ** return immediately
  */
  if (currentPbufferComponents == numComponents &&
      currentPbufferOutputs >= numOutputs &&
      currentPbufferWidth >= width &&
      currentPbufferHeight >= height)
    return false;

  GPUAssert(hpbufferdc, "hpbufferdc = NULL");
  GPUAssert(numComponents > 0 &&
            numComponents <= 4,
            "Cannot hand pbuffers other than 1-4 components");
  GPUAssert(width <= 2048 || height <= 2048, 
            "Cannot handle pbuffers greater than 2048");
  GPUAssert(numOutputs < 5,
            "Cannot handle more than 4 outputs");
  GPUAssert(numOutputs > 0,
            "Creating Pbuffer with zero outputs?");

  /* Tear down the old pbuffer */
  if (!wglMakeCurrent (hpbufferdc, NULL))
    GPUError("MakeCurrent Failed");

  if (!wglReleasePbufferDCARB (hpbuffer, hpbufferdc))
    GPUError("ReleasePbufferDC Failed");

  if (!wglDestroyPbufferARB (hpbuffer))
    GPUError("DestroyPbufferARB Failed");

  GPUAssert(hwindowdc, "hwindowdc = NULL");
  GPUAssert(pixelformat[numOutputs-1][numComponents-1], 
            "Invalid pixelformat");
  GPUAssert(piAttribList[numComponents-1], "Invalid piAttribList");
  GPUAssert(2048, "Bogus 2048");

  /* Find the largest power of two which contains the size */
  if (width > currentPbufferWidth) {
    for (i=16; i<width; i*=2) 
      /* do nothing */;
    GPUAssert(i<=2048,
              "Cannot create pbuffers larger than 2048");
    currentPbufferWidth = i;
  }
  if (height > currentPbufferHeight) {
    for (i=16; i<height; i*=2) 
      /* do nothing */;
    GPUAssert(i<=2048,
              "Cannot create pbuffers larger than 2048");
    currentPbufferHeight = i;
  }

  currentPbufferComponents = numComponents;
  
  /* Create a fresh pbuffer */
  hpbuffer = wglCreatePbufferARB(hwindowdc,
                                 pixelformat
                                 [currentPbufferOutputs < numOutputs ? 
                                  (numOutputs-1) : 
                                  (currentPbufferOutputs-1)]
                                 [numComponents-1],
                                 currentPbufferWidth,
                                 currentPbufferHeight,
                                 piAttribList[numComponents-1]);

  if (!hpbuffer)
    GPUError("Failed to create pbuffer, possibly out of memory");
  
  hpbufferdc = wglGetPbufferDCARB (hpbuffer);
  
  if (!hpbufferdc)
    GPUError("Failed to get pbuffer dc");
  
  if (currentPbufferOutputs < numOutputs) {
    // Since there are a different number of buffers
    // we have to create a new context
    HGLRC new_hglrc;
    new_hglrc = wglCreateContext( hpbufferdc );
    GPUAssert(new_hglrc, "Invalid glrc");
    if (!wglShareLists(hglrc, new_hglrc)) 
      GPUError("wglShareLists failed");
    hglrc = new_hglrc;
    currentPbufferOutputs = numOutputs;
    switched_contexts = true;
  }

  GPUAssert(hglrc, "Invalid glrc");

  if (!wglMakeCurrent( hpbufferdc, hglrc ))
    GPUError("Failed to make current GL context");

  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  
  return switched_contexts;
}

void OGLWindow::makeCurrent()
{
  wglMakeCurrent( hpbufferdc, hglrc );
}

void OGLWindow::shareLists( HGLRC inContext )
{
  wglShareLists( hglrc, inContext );
}

#else

/* Linux version */

#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>


/*
** Static OGLWindow members:
**
** For some reason, NVIDIA drivers don't seem to 
** like me creating a window, then destroying it, 
** then creating a new window and pbuffer.
** The driver hangs inside of gl calls.
** However if I just create a window and pbuffer
** it is fine.  So we switched to using a static 
** window and pbuffer.  Lame but it works.
*/

Display     *OGLWindow::pDisplay;
int          OGLWindow::iScreen;
Window       OGLWindow::glxWindow;
Colormap     OGLWindow::cmap;
XVisualInfo *OGLWindow::visual;
GLXFBConfig *OGLWindow::glxConfig[4];
GLXPbuffer   OGLWindow::glxPbuffer;
GLXContext   OGLWindow::glxContext;

int          OGLWindow::piAttribList[4][16];

bool         OGLWindow::static_window_created = false;
bool         OGLWindow::static_pbuffers_initialized = false;

//This is non square because ATI can't do 2048^2 for some odd reason
#define PBUFFER_WIDTH 2048
#define PBUFFER_HEIGHT 1024

#define CRGBA(c, r,g,b,a) \
        GLX_RED_SIZE,               r, \
        GLX_GREEN_SIZE,             g, \
        GLX_BLUE_SIZE,              b, \
        GLX_ALPHA_SIZE,             a, \
        GLX_STENCIL_SIZE,           0, \
        GLX_DEPTH_SIZE,             0, \
        GLX_DRAWABLE_TYPE,          GLX_PBUFFER_BIT, \
        GLX_DOUBLEBUFFER,           0

#define BASEIATTRIB { \
{CRGBA(32,32,0,0,0), 0, 0}, \
{CRGBA(64,32,32,0,0), 0, 0}, \
{CRGBA(96,32,32,32,0), 0, 0}, \
{CRGBA(128,32,32,32,32), 0, 0} }

static const int 
baseiAttribList[4][4][64] = { BASEIATTRIB,
							  BASEIATTRIB,
							  BASEIATTRIB,
							  BASEIATTRIB};

static const float
basefAttribList[4][16] = { {0.0f,0.0f}, 
                           {0.0f,0.0f}, 
                           {0.0f,0.0f},
                           {0.0f,0.0f}};

#define PBATTRIB \
    GLX_PRESERVED_CONTENTS, GL_TRUE, \
    GLX_PBUFFER_WIDTH, PBUFFER_WIDTH, \
    GLX_PBUFFER_HEIGHT, PBUFFER_HEIGHT, \
    GLX_LARGEST_PBUFFER, 0

static int
basepiAttribList[4][16] = { {PBATTRIB, 0, 0}, \
							{PBATTRIB, 0, 0}, \
							{PBATTRIB, 0, 0}, \
							{PBATTRIB, 0, 0}};

OGLWindow::OGLWindow() {
  int attrib[] = { GLX_RGBA, None };
  XSetWindowAttributes swa;

  /* Bail if the window is already created */
  if (static_window_created)
    return;
  
  pDisplay = XOpenDisplay(NULL);
  if (pDisplay == NULL) {
    fprintf (stderr, "Could not connect to X Server.\n");
    exit(1);
  }
    
  iScreen  = DefaultScreen(pDisplay);

  visual = glXChooseVisual(pDisplay, iScreen, attrib);
  if (visual == NULL) {
    fprintf (stderr, "Could not create window visual.\n");
    exit(1);
  }

  glxContext = glXCreateContext(pDisplay, 
                                visual, 
                                0, GL_TRUE);  
  if (glxContext == NULL) {
    fprintf (stderr, "Could not create GL Context.\n");
    exit(1);
  }

  cmap = XCreateColormap (pDisplay, 
                          RootWindow(pDisplay, iScreen),
                          visual->visual, AllocNone);
  swa.border_pixel = 0;
  swa.colormap = cmap;

  glxWindow = XCreateWindow(pDisplay,
                            RootWindow(pDisplay, iScreen),
                            0, 0, 1, 1, 0, visual->depth, InputOutput,
                            visual->visual, CWBorderPixel | CWColormap,
                            &swa);
  if (!glXMakeCurrent(pDisplay, glxWindow, glxContext)) {
    fprintf (stderr, "OGLWindow: Could not make current.\n");
    exit(1);
  }

  glFinish();

  initglfunc();
  
  static_window_created = true;
}


void 
OGLWindow::initPbuffer( const int   (*viAttribList)[4][64],
                        const float (*vfAttribList)[4][16],
                        const int   (*vpiAttribList)[4][16]) {

  int iConfigCount;   
  int i;

  /* Bail if the pbuffers are already initialized */
  if (static_pbuffers_initialized)
    return;
  
  int   (*iAttribList)[4][64]  = (int   (*)[4][64])
    malloc (sizeof(baseiAttribList));
  float fAttribList[4][16]  = {{0.0f, 0.0f}, {0.0f, 0.0f}, 
                               {0.0f, 0.0f}, {0.0f, 0.0f}};

  memcpy(iAttribList,  baseiAttribList,  sizeof(baseiAttribList));
  memcpy(piAttribList, basepiAttribList, sizeof(basepiAttribList));

  /* Append vendor specific attribs */
  appendVendorAttribs( iAttribList, fAttribList, piAttribList,
                       viAttribList, vfAttribList, vpiAttribList);

  if (pDisplay == NULL) {
    fprintf (stderr, "Could not connect to X Server.\n");
    exit(1);
  }

  // Destroy Window 
  glXMakeCurrent(pDisplay, None, NULL);
  glXDestroyContext(pDisplay, glxContext);
  XDestroyWindow(pDisplay, glxWindow);
  XFreeColormap(pDisplay, cmap);
  XFree(visual);

  for (i=0; i<4; i++) {

    glxConfig[i] = glXChooseFBConfig(pDisplay, 
                                     iScreen, 
                                     iAttribList[0][i], 
                                     &iConfigCount);
    
    if (iConfigCount == 0) {
      fprintf (stderr, "OGL Window: No floating point pbuffer "
               "format found for float%d.\n", i+1);
      exit(1);
    }

    if (!glxConfig[i] || !glxConfig[i][0]) {
      fprintf(stderr, "OGL Window:  glXChooseFBConfig() failed\n");
      exit(1);
    }   

  }
  
  glxPbuffer = glXCreatePbuffer(pDisplay, 
                                glxConfig[3][0], 
                                piAttribList[3]);
  
  if (!glxPbuffer) {
    fprintf(stderr, "OGL Window: glXCreatePbuffer() failed\n");
    exit(1);
  }
  
  // Default to the 4 component
  glxContext = glXCreateNewContext(pDisplay, 
                                   glxConfig[3][0], 
                                   GLX_RGBA_TYPE, 
                                   0, GL_TRUE);
  if (!glxContext) {
    fprintf(stderr, "OGL Window: glXCreateContextWithConfig() failed\n");
    exit (1);
  }
     
  if (!glXMakeCurrent(pDisplay, glxPbuffer, glxContext)) {
    fprintf (stderr, "initPbuffer: glXMakeCurrent Failed\n");
    exit(1);
  }
  
  glFinish();

  currentPbufferComponents = 4;

  free (iAttribList);

  static_pbuffers_initialized = true;
}



bool
OGLWindow::bindPbuffer(unsigned int width,
                       unsigned int height,
                       unsigned int numOutputs,
                       unsigned int numComponents) {

  /* Sadly, Linux doesn't seem to like rebinding a
  ** context to a different format pbuffer.  So 
  ** we just run everything in a float4 pbuffer.
  */

  if (width > PBUFFER_WIDTH ||
      height > PBUFFER_HEIGHT) {
    fprintf (stderr, "Pbuffer not big enough\n");
    fprintf (stderr, "User requested %d x %d\n",
             width, height);
    exit(1);
  }

  if (numOutputs > 1) {
    fprintf (stderr, "Don't support multiple output on Linux yet\n");
    exit(1);
  }

  return true;

#if 0
  static const int pbAttribList[] =
    {
      GLX_PRESERVED_CONTENTS, GL_TRUE,
      GLX_PBUFFER_WIDTH, 2048,
      GLX_PBUFFER_HEIGHT, 2048,
      0
    };

  if (currentPbufferComponents == ncomponents) 
    return;

  assert (ncomponents > 0 &&
          ncomponents <= 4);

  glxPbuffer = glXCreatePbuffer(pDisplay, 
				glxConfig[ncomponents-1][0], 
				piAttribList[ncomponents-1]);
  
  if (glxPbuffer == 0) {
    fprintf (stderr, "Error: Could not create float%d pbuffer.\n",
             ncomponents);
  }

  if (!glXMakeCurrent(pDisplay, glxPbuffer, glxContext)) {
    fprintf (stderr, "bindPbuffer: glXMakeCurrent Failed\n");
    exit(1);
  }

  glFinish();

  currentPbufferComponents = ncomponents;
#endif

}

void OGLWindow::makeCurrent()
{
  glXMakeCurrent(pDisplay, glxPbuffer, glxContext);
}

void shareLists( HGLRC inContext )
{
  GPUAssert( false, "Haven't implemented share lists under glX..." );
}

OGLWindow::~OGLWindow() 
{
  // Switched everything to static
#if 0
  glXDestroyContext(pDisplay, glxContext);
  if (glxPbuffer)
    glXDestroyPbuffer(pDisplay, glxPbuffer);
  XDestroyWindow(pDisplay, glxWindow);
  XFreeColormap(pDisplay, cmap);
  XFree(visual);
  XCloseDisplay(pDisplay);
#endif
}

void OGLWindow::shareLists( HGLRC inContext )
{
//FIXME
}

#endif

