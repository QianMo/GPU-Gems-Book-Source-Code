/**
  @file g3derror.h
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2000-08-26
  @created 2002-08-04

  Copyright 2000-2003, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_ERROR_H
#define G3D_ERROR_H

#include <string>
#include "G3D/prompt.h"
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
    // disable: "C++ exception handler used"
    #pragma warning (disable : 4530)
#endif // _MSC_VER

/**
  @def error(level, message, showPrompt)
  error(const char *level, const char *message, bool showPrompt);
  error(const char *level, const string &message, bool showPrompt);
  error(const string &level, const string &message, bool showPrompt);
 
  Logs an error to stderr with the time, date, file, line, and message.
  Displays an error message.  If prompt is true, with an interactive prompt.
  If errorLevel is "Critical Error" the option to continue is not shown.
 
   @return Returns QUIT_ERROR if the user wants to quit, IGNORE_ERROR
      if the user wants to continue.
 */
#define error(level, message, showPrompt) \
    G3D::_internal::_utility_error(\
        level,\
        message,\
        showPrompt,\
        __FILE__,\
        __LINE__)

namespace G3D {

/**
 Return values from error().
 */
enum ErrorConstant {IGNORE_ERROR, QUIT_ERROR};


namespace _internal {
ErrorConstant _utility_error(
    const char* level,
    const char* message,
    bool        showPrompt,
    const char* filename,
    int line);

ErrorConstant _utility_error(const char *level, const std::string &message, bool showPrompt, const char *filename, int line);

ErrorConstant _utility_error(const std::string &level, const std::string &message, bool showPrompt,const char *filename, int line);


} /* namespace */

}; // namespace

#endif
