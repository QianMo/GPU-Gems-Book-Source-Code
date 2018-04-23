// --------------------------------------------------------
// Author: Sylvain.Lefebvre@imag.fr
// --------------------------------------------------------
#ifndef __GLUX_LOADER__
#define __GLUX_LOADER__
// --------------------------------------------------------
#ifndef GLUX_NO_OUTPUT  
# include <iostream>
# include <sstream>
#endif
// --------------------------------------------------------
#include <map>
#include <string>
// --------------------------------------------------------
#define GLUX_NOT_LOADED   -1
#define GLUX_NOT_AVAILABLE 0
#define GLUX_NOT_DEVL      0
#define GLUX_AVAILABLE     1
#define GLUX_DEVL          2
#define GLUX_IS_AVAILABLE(pl) glux_plugin_##pl.isAvailable()
#define GLUX_IS_DEVL(pl)      glux_plugin_##pl.isDevel()
#define GLUX_IS_LOADED GLUX_IS_AVAILABLE
// --------------------------------------------------------
namespace glux
{
  class gluxPlugin;

  void init(int flags,const char *profile);
  void shutdown();
  void registerPlugin(gluxPlugin *pl);
}
// --------------------------------------------------------
void gluxInit(int flags=0,const char *profile=NULL);
void gluxShutdown();
int  gluxIsAvailable(const char *);
int  gluxIsDevl(const char *);
// --------------------------------------------------------
#endif
// --------------------------------------------------------
