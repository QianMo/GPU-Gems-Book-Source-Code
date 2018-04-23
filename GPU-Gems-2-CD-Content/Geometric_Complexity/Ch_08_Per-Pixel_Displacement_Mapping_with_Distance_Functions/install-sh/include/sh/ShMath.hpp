// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
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
#ifndef SHMATH_HPP
#define SHMATH_HPP

/** @file ShMath.hpp
 * Some C99 math functions that windows/apple don't have
 */

#include <cmath>

namespace {
#ifdef WIN32
inline float log2f(float a) { return logf(a) / logf(2.0f); }
inline float exp2f(float a) { return powf(2.0f, a); }
inline float exp10f(float a) { return powf(10.0f, a); }
#endif
#ifdef __APPLE__
inline float exp10f(float a) { return powf(10.0f, a); }
#endif
}

#endif
