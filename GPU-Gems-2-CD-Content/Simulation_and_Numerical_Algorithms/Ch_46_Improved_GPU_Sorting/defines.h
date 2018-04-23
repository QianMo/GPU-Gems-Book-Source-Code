/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/
#ifndef GBDEFINES_HH
#define GBDEFINES_HH

#ifdef WIN32 // WIN32

#include "_windows.h"

// C4786: identifier was truncated to '255' characters in the debug information
// C4251: foo 'needs to have dll-interface to be used by clients of class' bar
// C4275: non dll-interface class 'foo' used as base for dll-interface class 'bar'
// C4231: nonstandard extension used : 'extern' before template explicit instantiation
// C4291: new with no matching operator delete 
#  pragma warning( 4 : 4786 4251 4275 4231 4291)
#  pragma warning( disable : 4786 4251 4275 4231 4291) 

#endif

#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <limits>

#ifdef IRIX
#include <assert.h>
#else
#include <cassert>
#endif

#ifdef DEBUG
#define infomsg(s) std::cout<<typeid(*this).name()<<": \033[32mINFO:\033[0m "<<s<<std::endl
#define staticinfomsg(n,s) std::cout<<n<<": \033[32mINFO:\033[0m "<<s<<std::endl
#else
#define infomsg(s) std::cout<<s<<std::endl
#define staticinfomsg(n,s) std::cout<<s<<std::endl
#endif

#ifdef DEBUG
#define debugmsg(s) std::cerr<<typeid(*this).name()<<": \033[36mDEBUG:\033[0m "<<s<<std::endl
#define staticdebugmsg(n,s) std::cerr<<n<<": \033[36mDEBUG:\033[0m "<<s<<std::endl
#else
#define debugmsg(s)
#define staticdebugmsg(n,s)
#endif

#ifdef DEBUG
#define warningmsg(s) std::cerr<<typeid(*this).name()<<": \033[33mWARNING:\033[0m "<<s<<std::endl
#define staticwarningmsg(n,s) std::cerr<<n<<": \033[33mWARNING:\033[0m "<<s<<std::endl
#else
#define warningmsg(s) std::cerr<<"\033[33mWARNING:\033[0m "<<s<<std::endl
#define staticwarningmsg(n,s) std::cerr<<"\033[33mWARNING:\033[0m "<<s<<std::endl
#endif

#ifdef DEBUG
#define errormsg(s) std::cerr<<typeid(*this).name()<<": \033[31mERROR:\033[0m "<<s<<std::endl
#define staticerrormsg(n,s) std::cerr<<n<<": \033[31mERROR:\033[0m "<<s<<std::endl
#else
#define errormsg(s) std::cerr<<"\033[31mERROR:\033[0m "<<s<<std::endl
#define staticerrormsg(n,s) std::cerr<<"\033[31mERROR:\033[0m "<<s<<std::endl
#endif

// check for OpenGL-Errors
#define checkGLError(msg) {const GLenum e = glGetError(); if ((e != 0) && (e != GL_NO_ERROR)) {staticerrormsg(\
   "OpenGL-ERROR <0x" << std::setfill('0') << std::hex << std::setw(4) << e << std::dec << ">",\
   msg <<" in " __FILE__ "(" << __LINE__ << "): " << gluErrorString(e));}}

#endif // GBDEFINES_HH
/*----------------------------------------------------------------------
|
| $Log$
|
+---------------------------------------------------------------------*/
