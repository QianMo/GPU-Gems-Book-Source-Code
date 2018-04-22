#ifndef WGL_EXTENSIONS_STRING_ARB_H
#define WGL_EXTENSIONS_STRING_ARB_H


//
// Includes
//

#include <windows.h>
#include <GL/gl.h>
#include <GL/wglext.h>

#include <iostream>
#include <string>


//
// Macros
//

#define INIT_FUNCT_PTR(F,T)                                     \
    F = (T) wglGetProcAddress(#F);                              \
    if (0 == F)                                                 \
        std::cerr << "Error: Pointer to " << #F                 \
                  << " pointer was not found." << std::endl;    \

#define ASSERT_EXTENSION_SUPPORT(E)                             \
{                                                               \
    std::string sExtensions = (char *)                          \
        wglGetExtensionsStringARB(wglGetCurrentDC());           \
                                                                \
    if (-1 == sExtensions.find(std::string(#E)))                \
    {                                                           \
        std::cerr << #E << " extension not supported."          \
                  << std::endl;                                 \
                                                                \
        exit(1);                                                \
    }                                                           \
}


extern PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB;

void initExtensionsStringARB(HDC hDC);

#endif // WGL_EXTENSIONS_STRING_ARB_H