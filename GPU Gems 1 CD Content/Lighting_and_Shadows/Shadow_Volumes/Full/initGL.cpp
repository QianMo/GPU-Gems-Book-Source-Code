/**
  @file initGL.cpp

  @maintainer Morgan McGuire, (matrix@graphics3d.com)
  @cite Portions written by Kevin Egan, (ktegan@cs.brown.edu)

  @created 2002-01-01
  @edited  2002-07-02
 */

#include <G3DAll.h>
#include <SDL.h>
#include <SDL_syswm.h>


void initGL(int winWidth, int winHeight) {
    // Under Windows, reset the last error so that our debug box
    // gives the correct results
    #if _WIN32
        SetLastError(0);
    #endif

	const int minimumDepthBits = 16;
	const int desiredDepthBits = 24;

	const int minimumStencilBits = 8;
	const int desiredStencilBits = 8;

    const bool fullscreen = false;

	if (SDL_Init(SDL_INIT_NOPARACHUTE | SDL_INIT_VIDEO |
                SDL_INIT_JOYSTICK | SDL_INIT_AUDIO) < 0) {
        std::string msg = format("Unable to initialize SDL: %s\n",
                SDL_GetError());

        error("Critical Error", msg, true);
		exit(1);
	}

	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, desiredDepthBits);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, GL_TRUE);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, desiredStencilBits);
    // This is available in the patched version of the SDL that
    // allows wgl multi-sampling extensions.  For more infom on
    // the (soon to be released) patch email Kevin (ktegan@cs.brown.edu)
	//SDL_GL_SetAttribute(SDL_GL_MULTISAMPLE_SIZE, 4);

	// Create a width x height OpenGL screen 
    int flags =  SDL_HWSURFACE | SDL_OPENGL | (fullscreen ? SDL_FULLSCREEN : 0);
	if (SDL_SetVideoMode(winWidth, winHeight, 0, flags) == NULL) {
		SDL_Quit();
        error("Critical Error", "Can't create a window.", true);
		exit(2);
	}

	SDL_SysWMinfo info;
	SDL_VERSION(&info.version);
	SDL_GetWMInfo(&info);

	// Set the title bar
	SDL_WM_SetCaption("Shadow Demo", NULL);

	glViewport(0, 0, winWidth, winHeight);

    #define LOAD_GL_EXTENSION(name) \
        *((void**)&name) = wglGetProcAddress(#name); \
        debugAssertM(name != NULL, "Unable to load required extension!")

    // Load the OpenGL Extensions we want
    LOAD_GL_EXTENSION(glMultiTexCoord2fvARB);
    LOAD_GL_EXTENSION(glMultiTexCoord2fARB);
    LOAD_GL_EXTENSION(glActiveTextureARB);
    LOAD_GL_EXTENSION(wglSwapIntervalEXT);

#ifdef _NVIDIA_EXTENSIONS_
    LOAD_GL_EXTENSION(glGetOcclusionQueryuivNV);
    LOAD_GL_EXTENSION(glEndOcclusionQueryNV);
    LOAD_GL_EXTENSION(glBeginOcclusionQueryNV);
    LOAD_GL_EXTENSION(glGenOcclusionQueryNV);
#endif

    #undef LOAD_GL_EXTENSION

    if (wglSwapIntervalEXT != NULL) {
        // turn on asyncronous video refresh
        wglSwapIntervalEXT(0);
    }
}

#ifdef WIN32
extern HWND SDL_Window;

HDC getWindowHDC() {
    // Get Windows HDC
    SDL_SysWMinfo info;

    SDL_VERSION(&info.version);

    int result = SDL_GetWMInfo(&info);

    if (result != 1) {
        debugAssertM(false, SDL_GetError());
    }

    HDC hdc = GetDC(info.window);

    if (hdc == 0) {
        debugAssert(hdc != 0);
    }

    return hdc;
}
#endif
