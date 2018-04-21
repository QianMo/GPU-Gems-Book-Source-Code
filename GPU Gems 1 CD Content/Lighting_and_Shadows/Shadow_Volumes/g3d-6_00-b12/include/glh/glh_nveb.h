/*
    glh - is a platform-indepenedent C++ OpenGL helper library
      (by Cass Everitt)

    Copyright (c) 2000 NVIDIA Corporation
    All rights reserved.

    Redistribution and use in source and binary forms, with or
        without modification, are permitted provided that the following
        conditions are met:

     * Redistributions of source code must retain the above
           copyright notice, this list of conditions and the following
           disclaimer.

     * Redistributions in binary form must reproduce the above
           copyright notice, this list of conditions and the following
           disclaimer in the documentation and/or other materials
           provided with the distribution.

     * The names of contributors to this software may not be used
           to endorse or promote products derived from this software
           without specific prior written permission.

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
           ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
           LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
           FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
           REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
           INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
           BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
           LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
           CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
           LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
           ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
           POSSIBILITY OF SUCH DAMAGE.

    GLH by:        Cass Everitt - cass@r3.nu
    This file by:  Todd Kulick - todd@kulick.com
*/

// Special nVIDIA Effects Browser support for GLUT applications
// Copyright (c) NVIDIA 2000

#ifndef _GLH_NVEB_H_
#define _GLH_NVEB_H_

#ifdef GLH_NVEB_USING_NVPARSE
#define NVEB_USING_NVPARSE 1
#endif

#ifdef GLH_NVEB_USING_CGGL
#define NVEB_USING_CGGL 1
#endif

#ifdef _WIN32
#if defined(GLH_EXT_SINGLE_FILE) || defined(GLH_EXTENSIONS_SINGLE_FILE) 
#include "NVEBGlutAPI.c"
#endif
#endif

#include "NVEBGlutAPI.h"

#endif
