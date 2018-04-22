// 
// Includes
//

#include "AssertCG.h"

#include <iostream>

#ifdef _WIN32
    #include <windows.h>
#endif

#include <gl/gl.h>
#include <gl/glu.h>
#include <Cg/cg.h>


void cg_assert(const char * zFile, unsigned int nLine)
{
    CGerror nError;
    const char * zErrorString;

    if ((nError = cgGetError()) != CG_NO_ERROR)
    {
        zErrorString = cgGetErrorString(nError);
        std::cerr << "Assertion failed (" << zFile << ":"
                  << nLine << ": " << zErrorString << std::endl;

        exit(-2);
    }
}
