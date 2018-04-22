/**
 @file platform.h

 #defines for platform specific issues.

 @maintainer Morgan McGuire, matrix@graphics3d.com

 @created 2003-06-09
 @edited  2004-01-06
 */

#ifndef G3D_PLATFORM_H
#define G3D_PLATFORM_H

/**
 The version number of G3D in the form: MmmBB -> 
 version M.mm [beta BB]
 */
#define G3D_VER 60012


#ifdef _MSC_VER 
    #define G3D_WIN32 
#elif __linux__ 
    #define G3D_LINUX
#elif __APPLE__ 
    #define G3D_OSX
#else
    #error Unknown platform 
#endif

#if defined(G3D_WIN32)
    #define SSE
#endif


// Verify that the supported compilers are being used and that this is a known
// processor.

#ifdef G3D_LINUX
    #ifndef __GNUC__
        #error G3d only supports the gcc compiler on Linux.
    #endif

    #ifndef __i386__
        #error G3D only supports x86 machines on Linux.
    #endif

    #ifndef __cdecl
        #define __cdecl __attribute__((cdecl))
    #endif

    #ifndef __stdcall
        #define __stdcall __attribute__((stdcall))
    #endif
#endif

#ifdef G3D_OSX
    #ifndef __GNUC__
        #error G3D only supports the gcc compiler on OS X.
    #endif

    #ifndef __POWERPC__
        #error G3D only supports PowerPC processors on OS X.
    #endif

    #ifndef __cdecl
        #define __cdecl __attribute__((cdecl))
    #endif
#endif


#ifdef G3D_WIN32
    // Old versions of MSVC (6.0 and previous) don't
    // support C99 for loop scoping rules.  This fixes them.
    #if (_MSC_VER <= 1200)
        // This trick will generate a warning; disable the warning
        #pragma warning (disable : 4127)
        #define for if (false) {} else for
    #endif


    // On MSVC, we need to link against the multithreaded DLL version of
    // the C++ runtime because that is what SDL and ZLIB are compiled
    // against.  This is not the default for MSVC, so we set the following
    // defines to force correct linking.  
    //
    // For documentation on compiler options, see:
    //  http://msdn.microsoft.com/library/default.asp?url=/library/en-us/vccore/html/_core_.2f.md.2c_2f.ml.2c_2f.mt.2c_2f.ld.asp
    //  http://msdn.microsoft.com/library/default.asp?url=/library/en-us/vccore98/HTML/_core_Compiler_Reference.asp
    //

    // DLL runtime
    #ifndef _DLL
	    #define _DLL
    #endif

    // Multithreaded runtime
    #ifndef _MT
	    #define _MT 1
    #endif

    // Ensure that we aren't forced into the static lib
    #ifdef _STATIC_CPPLIB
	    #undef _STATIC_CPPLIB
    #endif

    #ifdef _DEBUG
        #pragma comment(linker, "/NODEFAULTLIB:LIBCD.LIB")
        #pragma comment(linker, "/DEFAULTLIB:MSVCRTD.LIB")
    #else
        #pragma comment(linker, "/NODEFAULTLIB:LIBC.LIB")
        #pragma comment(linker, "/DEFAULTLIB:MSVCRT.LIB")
    #endif

    // Now set up external linking
    #define ZLIB_DLL

    #pragma comment(lib, "zdll.lib")
    #pragma comment(lib, "ws2_32.lib")
    #pragma comment(lib, "winmm.lib")
    #pragma comment(lib, "imagehlp.lib")
    #pragma comment(lib, "version.lib")

    #ifdef _DEBUG
        // zlib and SDL were linked against the release MSVCRT; force
        // the debug version.
        #pragma comment(linker, "/NODEFAULTLIB:MSVCRT.LIB")

        // Don't link against G3D when building G3D itself.
        #ifndef G3D_BUILDING_LIBRARY_DLL
           #pragma comment(lib, "G3D-debug.lib")
        #endif
    #else
        // Don't link against G3D when building G3D itself.
        #ifndef G3D_BUILDING_LIBRARY_DLL
            #pragma comment(lib, "G3D.lib")
        #endif
    #endif

#endif

// Header guard
#endif
