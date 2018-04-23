// --------------------------------------------------------
// Author: Sylvain.Lefebvre@imag.fr
// --------------------------------------------------------
#ifndef __GLUX_PLUGIN__
#define __GLUX_PLUGIN__
// --------------------------------------------------------
#include "gluxLoader.h"
// --------------------------------------------------------
#ifndef GLUX_NO_OUTPUT  
# include <iostream>
#endif
// --------------------------------------------------------
#define GL_GLEXT_LEGACY
#define GLX_GLXEXT_LEGACY
#define WGL_WGLEXT_LEGACY
// --------------------------------------------------------
#ifdef WIN32
# include <windows.h>
# define GLUX_LOAD_PROC(s) ::wglGetProcAddress(s)
#else
# define APIENTRY
# define GLUX_LOAD_PROC(s) ::glXGetProcAddressARB((GLubyte *)s)
# define GLX_GLXEXT_PROTOTYPES
# define GLX_GLXEXT_LEGACY
# include <GL/glx.h>
#endif
# define APIENTRYP APIENTRY *
#include <GL/gl.h>
// --------------------------------------------------------
namespace glux
{
  
  class gluxPlugin
  {
  protected:
    
    char       *m_szIdString;
    bool        m_bAvailable;
    bool        m_bRequired;
    bool        m_bInitDone;
    bool        m_bDisabled;
    bool        m_bDevel;
    gluxPlugin *m_Linked;

  public:
    // automatic plugin registration
    gluxPlugin(bool required);
    
    const char  *getIdString() const;
    bool         isAvailable();
    bool         isRequired();
    void         linkTo(gluxPlugin *);
    bool         init(int flags=0);
    bool         isDevel() const {return (m_bDevel);}
    bool         isDisabled() const;
    void         setDisabled(bool b);
    virtual bool load();
  };

}
// --------------------------------------------------------
#define GLUX_NEW_PLUGIN(idstr)      \
namespace glux \
{ \
    class gluxPlugin_##idstr : public gluxPlugin \
    { \
      public: \
      gluxPlugin_##idstr(bool b) : gluxPlugin(b) \
	{ \
	  m_szIdString=#idstr; \
          registerPlugin(this); \
        } \
      bool load(); \
    }; \
}
// --------------------------------------------------------
#define GLUX_EMPTY_PLUGIN(idstr)      \
namespace glux \
{ \
    class gluxPlugin_##idstr : public gluxPlugin \
    { \
      public: \
      gluxPlugin_##idstr(bool b) : gluxPlugin(b) \
	{ \
	  m_szIdString=#idstr; \
          registerPlugin(this); \
        } \
    }; \
}
// --------------------------------------------------------
#define GLUX_LOAD(idstr)    static glux::gluxPlugin_##idstr glux_plugin_##idstr(false);
#define GLUX_REQUIRE(idstr) static glux::gluxPlugin_##idstr glux_plugin_##idstr(true);
// --------------------------------------------------------
#define GLUX_PLUGIN_LOAD(idstr) bool glux::gluxPlugin_##idstr::load()
#define GLUX_CHECK_EXTENSION_STRING(ext) (strstr((const char *)glGetString(GL_EXTENSIONS),ext) == NULL)
// --------------------------------------------------------
#endif
// --------------------------------------------------------
