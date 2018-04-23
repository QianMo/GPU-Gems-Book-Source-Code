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
#ifndef SHUTIL_KERNELLIGHTIMPL_HPP 
#define SHUTIL_KERNELLIGHTIMPL_HPP 

#include <sstream>
#include "ShSyntax.hpp"
#include "ShPosition.hpp"
#include "ShManipulator.hpp"
#include "ShAlgebra.hpp"
#include "ShProgram.hpp"
#include "ShNibbles.hpp"
#include "ShKernelLight.hpp"

/** \file ShKernelLightImpl.hpp
 * This is an implementation of useful kernels and nibbles (simple kernels).
 */

namespace ShUtil {

using namespace SH;

template <typename T>
ShProgram ShKernelLight::pointLight() {
  ShProgram kernel =  SH_BEGIN_PROGRAM() {
    typename T::InputType SH_DECL(lightColor);
    typename T::OutputType SH_DECL(irrad) = lightColor;
  } SH_END;
  return kernel;
}

template<typename T>
ShProgram ShKernelLight::spotLight() {
  ShProgram kernel =  SH_BEGIN_PROGRAM() {
    typename T::InputType SH_DECL(lightColor);
    ShInputAttrib1f SH_DECL(falloff);
    ShInputAttrib1f SH_DECL(lightAngle);
    ShInputVector3f SH_DECL(lightDir);
    ShInputVector3f SH_DECL(lightVec);

    typename T::OutputType SH_DECL(irrad); 

    lightDir = normalize(lightDir);
    lightVec = normalize(lightVec);
    ShAttrib1f t = -lightDir | lightVec;
    ShAttrib1f cosf = cos(falloff);
    ShAttrib1f cosang = cos(lightAngle);

    irrad = lightColor;
    irrad *= t > cosang; // if outside light angle, always 0 
    irrad *= (t < cosf) * (t - cosang) / (cosf - cosang) + (t >= cosf); // linear blend between start of falloff and 0 
  } SH_END;
  return kernel;
}

template<typename T>
ShProgram ShKernelLight::texLight2D(const ShBaseTexture2D<T> &tex) {
  ShProgram kernel =  SH_BEGIN_PROGRAM() {
    ShInputAttrib1f SH_DECL(scaling);
    ShInputAttrib1f SH_DECL(lightAngle);
    ShInputVector3f SH_DECL(lightDir);
    ShInputVector3f SH_DECL(lightUp);
    ShInputVector3f SH_DECL(lightVec);

    typename T::OutputType SH_DECL(irrad); 

    lightDir = normalize(lightDir);
    lightUp = normalize(lightUp);
    lightVec = normalize(lightVec);
    ShVector3f lightHoriz = cross(lightDir, lightUp);

    ShAttrib2f texcoord;
    texcoord(0) = frac(((-lightVec | lightHoriz) + ShConstAttrib1f(0.5f)) * scaling);
    texcoord(1) = frac(((-lightVec | lightUp) + ShConstAttrib1f(0.5f)) * scaling);

    irrad = tex(texcoord); 
  } SH_END;
  return kernel;
}

}

#endif
