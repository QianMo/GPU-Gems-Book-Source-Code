/**
  @file debugAssert.h
 
  debugAssert(expression);
  debugAssertM(expression, message);
 
  @cite
     John Robbins, Microsoft Systems Journal Bugslayer Column, Feb 1999.
     <A HREF="http://msdn.microsoft.com/library/periodic/period99/feb99_BUGSLAYE_BUGSLAYE.htm">
     http://msdn.microsoft.com/library/periodic/period99/feb99_BUGSLAYE_BUGSLAYE.htm</A>
 
  @cite 
     Douglas Cox, An assert() Replacement, Code of The Day, flipcode, Sept 19, 2000
     <A HREF="http://www.flipcode.com/cgi-bin/msg.cgi?showThread=COTD-AssertReplace&forum=cotd&id=-1">
     http://www.flipcode.com/cgi-bin/msg.cgi?showThread=COTD-AssertReplace&forum=cotd&id=-1</A>
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2001-08-26
  @edited  2003-08-05

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
 */

#ifndef G3D_DEBUGASSERT_H
#define G3D_DEBUGASSERT_H

#include <string>


/**
 * @def debugAssert(exp)
 * Breaks if the expression is false. If G3D_DEBUG_NOGUI is defined, prompts at
 * the console, otherwise pops up a dialog.  The user may then break (debug), 
 * ignore, ignore always, or halt the program.
 *
 * The assertion is also posted to the clipboard under Win32.
 */

/**
 * @def debugAssertM(exp, msg)
 * Breaks if the expression is false and displays a message. If G3D_DEBUG_NOGUI 
 * is defined, prompts at the console, otherwise pops up a dialog.  The user may
 * then break (debug), ignore, ignore always, or halt the program.
 *
 * The assertion is also posted to the clipboard under Win32.
 */

/**
 * @def __debugPromptShowDialog__
 * @internal
 */

#ifdef _DEBUG

    #ifndef G3D_OSX
        #ifdef _MSC_VER
            #define debugBreak() _asm { int 3 }
        #else
            #define debugBreak() __asm__ __volatile__ ( "int $3" )
        #endif
    #else
        #define debugBreak() #error "No debug break on OS X"
    #endif


    #define debugAssert(exp) debugAssertM(exp, "Debug assertion failure")

    #ifdef G3D_DEBUG_NOGUI
        #define __debugPromptShowDialog__ false
    #else
        #define __debugPromptShowDialog__ true
    #endif

    #define debugAssertM(exp, message) { \
        static bool __debugAssertIgnoreAlways__ = false; \
        if (!__debugAssertIgnoreAlways__ && !(exp)) { \
            if (G3D::_internal::_handleDebugAssert_(#exp, message, __FILE__, __LINE__, __debugAssertIgnoreAlways__, __debugPromptShowDialog__)) { \
                 debugBreak(); \
            } \
        } \
    }

    #define alwaysAssertM debugAssertM

#else  // Release
    #ifdef G3D_DEBUG_NOGUI
        #define __debugPromptShowDialog__ false
    #else
        #define __debugPromptShowDialog__ true
    #endif

    // In the release build, just define away assertions.
    #define debugAssert(exp)
    #define debugAssertM(exp, message)
    #define debugBreak()

    // But keep the always assertions
    #define alwaysAssertM(exp, message) { \
        if (!(exp)) { \
            G3D::_internal::_handleErrorCheck_(#exp, message, __FILE__, __LINE__, __debugPromptShowDialog__); \
            exit(-1); \
        } \
    }

#endif  // if debug



/**
 * @def debugBreak()
 *
 * Break at the current location (i.e. don't push a procedure stack frame
 * before breaking).
 */

namespace G3D {  namespace _internal {

/**
 Pops up an assertion dialog or prints an assertion

 ignoreAlways      - return result of pressing the ignore button.
 useGuiPrompt      - if true, shows a dialog
 */
bool _handleDebugAssert_(
    const char* expression,
    const std::string& message,
    const char* filename,
    int         lineNumber,
    bool&       ignoreAlways,
    bool        useGuiPrompt);

void _handleErrorCheck_(
    const char* expression,
    const std::string& message,
    const char* filename,
    int         lineNumber,
    bool        useGuiPrompt);

}; }; // namespace

#endif

