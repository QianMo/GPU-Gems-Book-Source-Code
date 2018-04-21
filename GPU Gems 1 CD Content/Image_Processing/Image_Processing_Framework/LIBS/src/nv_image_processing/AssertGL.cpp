
#include "AssertGL.h"

#include <iostream>

#ifdef _WIN32
    #include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>


void gl_assert(const char * zFile, unsigned int nLine)
{
    GLenum nErrorCode = glGetError();

    if (nErrorCode != GL_NO_ERROR)
    {
        const GLubyte * zErrorString = gluErrorString(nErrorCode);
        std::cerr << "Assertion failed (" <<zFile << ":"
                  << nLine << ": " << zErrorString << std::endl;

        exit(-1);
    }
}
