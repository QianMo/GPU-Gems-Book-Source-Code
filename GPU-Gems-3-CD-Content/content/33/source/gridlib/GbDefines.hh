/*----------------------------------------------------------------------
|
| $Id: GbDefines.hh,v 1.20 2004/11/08 11:01:27 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/
#ifndef GBDEFINES_HH
#define GBDEFINES_HH

// This is a version modified for GPUGems from the original gridlib file

#ifdef WIN32 // WIN32

// C4786: identifier was truncated to '255' characters in the debug information
// C4251: foo 'needs to have dll-interface to be used by clients of class' bar
// C4275: non dll-interface class 'foo' used as base for dll-interface class 'bar'
// C4231: nonstandard extension used : 'extern' before template explicit instantiation
// C4291: new with no matching operator delete 
#  pragma warning( 4 : 4786 4251 4275 4231 4291)
#  pragma warning( disable : 4786 4251 4275 4231 4291 4305 ) 

#    ifdef _LIB
#      define EXPIMP_TEMPLATE
#    else
#      define EXPIMP_TEMPLATE extern
#    endif
#    define GRIDLIB_API 
#    define INSTANTIATE( T_ ) 
#    define INSTANTIATEF( F_ ) 

#    ifdef _LIB
#      define EXPIMP_TEMPLATE_IO
#    else
#      define EXPIMP_TEMPLATE_IO extern
#    endif
#    define GRIDLIB_IO_API
#    define INSTANTIATE_IO( T_ )
#    define INSTANTIATEF_IO( F_ )

#    ifdef _LIB
#      define EXPIMP_TEMPLATE_VIS
#    else
#      define EXPIMP_TEMPLATE_VIS extern
#    endif
#    define GRIDLIB_VIS_API
#    define INSTANTIATE_VIS( T_ ) 
#    define INSTANTIATEF_VIS( F_ )


#else // WIN32 -> NOT WIN32

#  define INSTANTIATE( T_ ) 
#  define INSTANTIATEF( F_ )
#  define INSTANTIATE_IO( T_ ) 
#  define INSTANTIATEF_IO( F_ ) 
#  define INSTANTIATE_VIS( T_ ) 
#  define INSTANTIATEF_VIS( F_ ) 
#  define GRIDLIB_API
#  define GRIDLIB_IO_API
#  define GRIDLIB_VIS_API
#  define EXPIMP_TEMPLATE
#  define EXPIMP_TEMPLATE_IO
#  define EXPIMP_TEMPLATE_VIS


#endif // NOT WIN32



#ifndef INLINE
#ifdef OUTLINE
#define INLINE
#else
#define INLINE inline
#endif
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

// the GNU compiler wants template friends to be fully specialized

#ifdef __GNUG__
#define GB_TEXPORT <>
#else
#define GB_TEXPORT
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

// the following is here to steer memory pooling
#define MEMPOOLINC 64*1024
//1048576
//5242880
//2048
// this defines the paging size
#define MEMPOOLBLOCKS 4096

// some macros for better code readability used in rendering subsystem
#define GR_DECLARE_DEFAULT_STATE(classname) \
    public: \
        static classname* getDefault () { return kDefault_; } \
    protected: \
        static classname* kDefault_; \
        friend class _##classname##InitTermDS

#define GR_IMPLEMENT_DEFAULT_STATE(classname) \
    classname* classname::kDefault_ = NULL; \
    class _##classname##InitTermDS { \
    public: \
        _##classname##InitTermDS () { \
            classname::kDefault_ = new classname; \
            GrRenderState::Type eType = classname::kDefault_->getType(); \
            classname::defaultStates_[eType] = classname::kDefault_; \
        } \
        ~_##classname##InitTermDS () { \
            delete classname::kDefault_; \
        } \
    }; \
    static _##classname##InitTermDS _force##classname##InitTermDS

// control endianess
#ifdef LINUX
#define bigEndianMachine false
#define littleEndianMachine true
#else
#define bigEndianMachine true
#define littleEndianMachine false
#endif
#define swabi2(i2) (((i2) >> 8) + (((i2) & 255) << 8))
#define swabi4(i4) (((i4) >> 24) + (((i4) >> 8) & 65280) + (((i4) & 65280) << 8) + (((i4) & 255) << 24))

/* Zum einfacheren Abfragen von OpenGL-Errors */
#define CHECK_GL_ERROR(msg) {const GLenum e = glGetError(); if ((e != 0) && (e != GL_NO_ERROR)) {std::cout\
   << "OpenGL-ERROR <0x" << std::setfill('0') << std::hex << std::setw(4) << e << std::dec\
   << "> in " __FILE__ "(" << __LINE__ << ")!" << std::endl << gluErrorString(e) << msg << std::endl;}}

#endif // GBDEFINES_HH
