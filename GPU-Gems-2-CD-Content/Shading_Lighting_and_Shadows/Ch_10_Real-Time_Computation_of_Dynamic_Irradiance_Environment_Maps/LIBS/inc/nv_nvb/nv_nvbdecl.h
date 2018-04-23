/*********************************************************************NVMH1****
File:
nv_nvbdecl.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_nvbdecl_h_
#define _nv_nvbdecl_h_

#ifdef NV_NVB_DLL

    #ifdef NV_NVB_EXPORTS
        #define DECLSPEC_NV_NVB __declspec(dllexport)
    #else
        #define DECLSPEC_NV_NVB __declspec(dllimport)
    #endif

#else
    #define DECLSPEC_NV_NVB
#endif

#endif  // _nv_nvbdecl_h_
