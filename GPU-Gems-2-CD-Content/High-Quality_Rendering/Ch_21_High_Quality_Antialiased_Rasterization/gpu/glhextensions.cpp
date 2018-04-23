// This bit of strangeness creates a function which we use to
// initialize function pointers for all of the NVidia OpenGL extension
// functions that normally require wglGetProcAddress
#ifdef WINNT
#include <windows.h>
#define GLH_EXT_SINGLE_FILE
#include <glh/glh_extensions.h>
#endif
