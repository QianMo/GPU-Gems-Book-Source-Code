
#ifdef WIN32
#include <windows.h>
#else
#define GLX_GLXEXT_LEGACY
#include <GL/glx.h>
#endif

#include <GL/gl.h>

#include "oglfunc.hpp"
#include "../gpucontext.hpp"

#define XXX(type, fn) \
   type fn;

#ifdef WIN32
RUNTIME_BONUS_WGL_FNS
#endif

RUNTIME_BONUS_GL_FNS
RUNTIME_BONUS_GL_FNS_ATI
RUNTIME_BONUS_GL_FNS_NV

#undef XXX


static bool hasextension(const char *ext) {
  const char *extensions = (const char *) glGetString(GL_EXTENSIONS);
  return (strstr(extensions, ext) != NULL);
}


static void checkextension (const char *ext) {
   if (!hasextension(ext)) {
      const char *card = (const char *) glGetString(GL_RENDERER);
      fprintf (stderr, "Extension %s not found for graphics card: \n %s\n",
               ext, card);
      exit(1);
  }
}
               

void brook::initglfunc(void) {

   checkextension("GL_ARB_fragment_program");
   checkextension("GL_ARB_multitexture");
   
   if (!hasextension("GL_NV_texture_rectangle") &&
       !hasextension("GL_ARB_texture_rectangle") &&
       !hasextension("GL_EXT_texture_rectangle")) {
      const char *card = (const char *) glGetString(GL_RENDERER);
      fprintf (stderr, "Texture Rectangle extension not found for "
               "graphics card: \n %s\n", card);
      exit(1);
   }

#ifdef WIN32
#define  XXX(type, fn) fn = (type) wglGetProcAddress(#fn); \
                       GPUAssert(fn, "Failed to load" #fn);
#else
#define  XXX(type, fn) fn = (type) glXGetProcAddressARB((const GLubyte *) #fn); \
                       GPUAssert(fn, "Failed to load" #fn);
#endif

  RUNTIME_BONUS_GL_FNS;

#ifdef WIN32
  RUNTIME_BONUS_WGL_FNS;
#endif

#undef XXX   
#ifdef WIN32
#define  XXX(type, fn) fn = (type) wglGetProcAddress(#fn);
#else
#define  XXX(type, fn) fn = (type) glXGetProcAddressARB((const GLubyte *) #fn);
#endif

  RUNTIME_BONUS_GL_FNS_ATI;
  RUNTIME_BONUS_GL_FNS_NV;

}


