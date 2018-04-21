/**
  @file Log.h

  @maintainer Morgan McGuire, morgan@graphics3d.com
  @cite Backtrace by Aaron Orenstein
  @created 2001-08-04
  @edited  2003-08-04
 */

#ifndef G3D_LOG_H
#define G3D_LOG_H

#include <stdio.h>
#include <string>
#include "G3D/platform.h"

#ifndef G3D_WIN32
    #include <stdarg.h>
#endif

namespace G3D {

/**
 System log for debugging purposes.  The first log opened
 is the "common log" and can be accessed with the static
 method common().  If you access common() and a common log
 does not yet exist, one is created for you.
 */
class Log {
private:

    /**
     Log messages go here.
     */
    FILE*                   logFile;

    std::string             filename;

    static Log*             commonLog;

    int                     stripFromStackBottom;

    /**
     Prints the time & stack trace.
     */
    void printHeader();

public:

    /**
     @param stripFromStackBottom Number of call stacks to strip from the
     bottom of the stack when printing a trace.  Useful for hiding
     routines like "main" and "WinMain".  If the specified file cannot
     be opened for some reason, tries to open "c:/tmp/log.txt" or
     "c:/temp/log.txt" instead.
     */
    Log(const std::string& filename = "log.txt",
        int stripFromStackBottom    = 0);

    virtual ~Log();

    /**
     Returns the handle to the file log.
     */
    FILE* getFile() const;

    /**
     Marks the beginning of a logfile section.
     */
    void section(const std::string& s);

    /**
     Given arguments like printf, writes characters to the debug text overlay.
     */
    void __cdecl printf(const char* fmt ...);

    static Log* common();

    static std::string getCommonLogFilename();

    void print(const std::string& s);

    void println(const std::string& s);
};

}

#endif
