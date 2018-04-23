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
#ifndef SHUTIL_KERNELSURFACEIMPL_HPP 
#define SHUTIL_KERNELSURFACEIMPL_HPP 

#include <sstream>
#include "ShSyntax.hpp"
#include "ShPosition.hpp"
#include "ShManipulator.hpp"
#include "ShAlgebra.hpp"
#include "ShProgram.hpp"
#include "ShNibbles.hpp"
#include "ShKernelSurface.hpp"
#include "ShFunc.hpp"

/** \file ShKernelSurfaceImpl.hpp
 * This is an implementation of useful surface kernels 
 */

namespace ShUtil {

using namespace SH;

template<typename T>
ShProgram ShKernelSurface::diffuse() {
  ShProgram kernel = SH_BEGIN_FRAGMENT_PROGRAM {
    typename T::InputType SH_DECL(kd);
    typename T::InputType SH_DECL(irrad);
    ShInputNormal3f SH_DECL(normal);
    ShInputVector3f SH_DECL(lightVec);
    ShInputPosition4f SH_DECL(posh);

    irrad *= pos(dot(normalize(normal), normalize(lightVec)));
    typename T::OutputType SH_DECL(result);
    result = irrad * kd; 
  } SH_END;
  return kernel;
}

template<typename T>
ShProgram ShKernelSurface::specular() {
  ShProgram kernel = SH_BEGIN_FRAGMENT_PROGRAM {
    typename T::InputType SH_DECL(ks);
    ShInputAttrib1f SH_DECL(specExp);
    typename T::InputType SH_DECL(irrad);

    ShInputNormal3f SH_DECL(normal);
    ShInputVector3f SH_DECL(halfVec);
    ShInputVector3f SH_DECL(lightVec);
    ShInputPosition4f SH_DECL(posh);

    normal = normalize(normal);
    halfVec = normalize(halfVec);
    lightVec = normalize(lightVec);
    irrad *= pos(normal | lightVec);

    typename T::OutputType SH_DECL(result);
    result = irrad * ks * pow(pos(normal | halfVec),specExp); 
  } SH_END;
  return kernel;
}


template<typename T>
ShProgram ShKernelSurface::phong() {
  ShProgram kernel = SH_BEGIN_PROGRAM("gpu:fragment") {
    typename T::InputType SH_DECL(kd);
    typename T::InputType SH_DECL(ks);
    ShInputAttrib1f SH_DECL(specExp);
    typename T::InputType SH_DECL(irrad);

    ShInputNormal3f SH_DECL(normal);
    ShInputVector3f SH_DECL(halfVec);
    ShInputVector3f SH_DECL(lightVec);
    ShInputPosition4f SH_DECL(posh);

    typename T::OutputType SH_DECL(result);

    normal = normalize(normal);
    halfVec = normalize(halfVec);
    lightVec = normalize(lightVec);
    irrad *= pos(normal | lightVec);
    result = irrad * (kd + ks * pow(pos(normal | halfVec), specExp)); 
  } SH_END;
  return kernel;
}

template<typename T>
ShProgram ShKernelSurface::gooch() {
  ShProgram kernel = SH_BEGIN_PROGRAM("gpu:fragment") {
    typename T::InputType SH_DECL(kd);
    typename T::InputType SH_DECL(cool);
    typename T::InputType SH_DECL(warm);
    typename T::InputType SH_DECL(irrad);

    ShInputNormal3f SH_DECL(normal);
    ShInputVector3f SH_DECL(lightVec);
    ShInputPosition4f SH_DECL(posh);

    typename T::OutputType SH_DECL(result);

    normal = normalize(normal);
    lightVec = normalize(lightVec);
    result = lerp(mad((normal | lightVec), ShConstAttrib1f(0.5f), ShConstAttrib1f(0.5f)) * irrad, warm, cool) * kd;
  } SH_END;
  return kernel;
}

template<typename T>
ShProgram ShKernelSurface::null() {
  ShProgram kernel = SH_BEGIN_PROGRAM("gpu:fragment") {
    typename T::InputType SH_DECL(irrad);
    ShInputPosition4f SH_DECL(posh);

    typename T::OutputType SH_DECL(result) = irrad;
  } SH_END;
  return kernel;
}

}

#endif
