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
#ifndef SHUTIL_KERNELSURFACE_HPP 
#define SHUTIL_KERNELSURFACE_HPP 

#include <string>
#include "ShMatrix.hpp"
#include "ShTexture.hpp"
#include "ShProgram.hpp"

/** \file ShKernelSurface.hpp
 * These are the surface shader kernels.
 * They must take as input an irrad of type T, which is the irradiance from a single light source
 * and should use only the globals "halfVec", "lightVec", "lightPos", "halfVect", or "lightVect". 
 *
 * Output must be a single variable of type T named result (Most likely a ShColor3f or 4f)
 */

namespace ShUtil {

using namespace SH;

class ShKernelSurface {
  private:
    // returns the string prefix concatenated with index
    static std::string makeName(std::string prefix, int index); 

  public:
    /** Diffuse fragment program 
     * IN(0) T kd                 - diffuse coefficient (kd) could be ShColor?f
     * IN(1) T irrad              - irradiance from the light source
     *
     * IN(1) ShNormal3f normal    - normal (xCS)
     * IN(2) ShVector3f lightVec  - light vector (xCS)
     * IN(3) ShPosition4f posh    - positino (HDCS)
     *
     * OUT(0) T result            - output result 
     *
     * xCS is usually either VCS or tangent space 
     */
    template<typename T>
    static ShProgram diffuse();

    /** Specular fragment program 
     * IN(0) T ks                 - specular coefficient (ks) could be ShColor?f
     * IN(1) ShAttrib1f specExp   - specular exponent 
     * IN(1) T irrad              - irradiance from the light source
     *
     * IN(2) ShNormal3f normal    - normal (xCS)
     * IN(3) ShVector3f halfVec   - half vector (xCS)
     * IN(4) ShVector3f lightVec  - light vector (xCS)
     * IN(5) ShPosition4f posh    - positino (HDCS)
     *
     * OUT(0) T result            - output result 
     *
     * xCS is usually either VCS or tangent space 
     */
    template<typename T>
    static ShProgram specular();

    /** Phong fragment program 
     * IN(0) T kd                 - diffuse coefficient (kd) could be ShColor?f
     * IN(1) T ks                 - specular coefficient (ks) could be ShColor?f
     * IN(2) ShAttrib1f specExp   - specular exponent
     * IN(1) T irrad              - irradiance from the light source
     *
     * IN(3) ShNormal3f normal    - normal (xCS)
     * IN(4) ShVector3f halfVec   - half vector (xCS)
     * IN(5) ShVector3f lightVec  - light vector (xCS)
     * IN(6) ShPosition4f posh    - position (HDCS)
     *
     * OUT(0) T result            - output colour 
     *
     * xCS is usually either VCS or tangent space 
     */
    template<typename T>
    static ShProgram phong();

    /** Gooch illustrative shading 
     * IN(0) T kd                 - diffuse coefficient (kd) could be ShColor?f
     * IN(1) T cool               - cool multiplier (when light | normal == -1) 
     * IN(2) T warm               - warm multiplier (when light | normal == 1)
     *
     * IN(1) T irrad              - irradiance from the light source
     * IN(3) ShNormal3f normal    - normal (xCS)
     * IN(5) ShVector3f lightVec  - light vector (xCS)
     * IN(6) ShPosition4f posh    - position (HDCS)
     *
     * OUT(0) T result            - output colour 
     */
    template<typename T>
    static ShProgram gooch();

    /** null surface
     * IN(0) T irrad
     * IN(1) ShPosition4f posh
     *
     * OUT(0) T result = irrad
     */
    template<typename T>
    static ShProgram null();
};

}

#include "ShKernelSurfaceImpl.hpp"

#endif
