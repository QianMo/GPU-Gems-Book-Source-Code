//
// Includes
//

#include "wglExtensionsStringARB.h"

#include <iostream>
#include <string>


PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB;


void initExtensionsStringARB(HDC hDC)
{
    INIT_FUNCT_PTR(wglGetExtensionsStringARB, PFNWGLGETEXTENSIONSSTRINGARBPROC);
}
