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
#ifndef SHUTIL_KERNELLIGHT_HPP 
#define SHUTIL_KERNELLIGHT_HPP 

#include "ShMatrix.hpp"
#include "ShTexture.hpp"
#include "ShProgram.hpp"

/** \file ShKernelLight.hpp
 * A set of light shaders
 * Light shaders can use any of the vertex shader outputs from ShKerneLib::vsh 
 * and must output one irrad representing the irradiance at a surface of type T (probably ShColor3f).
 */

namespace ShUtil {

using namespace SH;

class ShKernelLight {
  public:
    /** Omnidirectional light program
     * IN(0) T lightColor - color;
     *
     * OUT(0) T irrad - irradiance
     */
    template<typename T>
    static ShProgram pointLight();

    /** Spotlight program 
     * linear falloff from (lightVec | lightDir) == -1 to -cos(fallofAngle)
     *
     * Takes a gradient direction and applies 
     * IN(0) T lightColor - color;
     * IN(1) ShAttrib1f falloff  - angle in radians where spotlight intensity begins to go to 0 
     * IN(0) ShAttrib1f lightAngle - angle in radians where spotlight intensity = 0
     * IN(2) ShVector3f lightDir - light direction (VCS) 
     *
     * The following usually comes from shVsh
     * IN(3) ShPoint3f lightVec - light vector at surface point (VCS) 
     *
     * OUT(0) T irrad - irradiance
     */
    template<typename T>
    static ShProgram spotLight();

    /** 2D-Textured light program  
     *
     * Takes as input
     * IN(0) ShAttrib1f scaling - scaling on the texture (tiles texture)
     * IN(1) ShAttrib1f lightAngle - angle in radians for fov of light 
     * IN(2) ShVector3f lightDir - direction light faces (VCS)
     * IN(3) ShVector3f lightUp - up direction of light, must be orthogonal to lightDir (VCS)
     *
     * The following typically come from shVsh 
     * IN(3) ShVector3f lightVec - light vector at surface point (VCS) 
     *
     * OUT(0) T irrad - irradiance
     */
    template<typename T>
    static ShProgram texLight2D(const ShBaseTexture2D<T> &tex);
};

}

#include "ShKernelLightImpl.hpp"

#endif
