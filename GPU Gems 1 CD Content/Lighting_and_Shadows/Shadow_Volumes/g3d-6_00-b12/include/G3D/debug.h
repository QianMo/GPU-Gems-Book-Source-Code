/**
 @file debug.h

 @maintainer Morgan McGuire, matrix@graphics3d.com

 @created 2001-08-26
 @edited  2004-01-01

  Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
*/

#ifndef G3D_DEBUG_H
#define G3D_DEBUG_H

#include "G3D/platform.h"
#ifdef G3D_WIN32
    #include <crtdbg.h>
#endif

#include "G3D/debugPrintf.h"
#include "G3D/debugAssert.h"

namespace G3D {


/**
 Useful for debugging purposes.  Note: On windows, 
 this will helpfully return "false" for a stack pointer.
 */
inline bool isValidHeapPointer(const void* x) {
    #ifdef G3D_WIN32
        return (_CrtIsValidHeapPointer(x) != 0) && (x != (void*)0xcccccccc) && (x != (void*)0xdeadbeef) && (x != (void*)0xfeeefeee);
    #else
        return x != NULL;
    #endif
}

/**
 Returns true if the pointer is likely to be
 a valid pointer (instead of an arbitrary number). 
 Useful for debugging purposes.
 */
inline bool isValidPointer(const void* x) {
    #ifdef G3D_WIN32
        return (_CrtIsValidPointer(x, 0, true) != 0) && (x != (void*)0xcccccccc) && (x != (void*)0xdeadbeef) && (x != (void*)0xfeeefeee);
    #else
        return x != NULL;
    #endif
}

}

#endif
