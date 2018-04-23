// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Bryan Chan, Michael D. McCool
// 
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
// 
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
// 
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
// 
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
#ifndef SHUTIL_KERNELSURFMAP_HPP 
#define SHUTIL_KERNELSURFMAP_HPP 

#include "ShProgram.hpp"

/** \file ShKernelSurfMap.hpp
 * 
 */

namespace ShUtil {

using namespace SH;

class ShKernelSurfMap {
  public:
    /** Bump program
     * Takes a gradient direction and applies 
     * IN(0) ShAttrib2f gradient  - gradient
     * IN(1) ShNormal3f normalt    - normalized normal vector (tangent space) 
     *
     * OUT(0) ShNormal3f normalt   - perturbed normal (tangent space)
     */
    static ShProgram bump();

    /** VCS Bump program
     * Takes a gradient direction and applies 
     * IN(0) ShAttrib2f gradient  - gradient
     * IN(1) ShNormal3f normal    - normalized normal vector (VCS)
     * IN(2) ShVector3f tangent   - normalized tangent vector (VCS)
     * IN(3) ShVector3f tangent2  - normalized secondary tangent (VCS)
     *
     * OUT(0) ShNormal3f normal   - perturbed normal (VCS)
     */
    static ShProgram vcsBump();
};

}

#endif
