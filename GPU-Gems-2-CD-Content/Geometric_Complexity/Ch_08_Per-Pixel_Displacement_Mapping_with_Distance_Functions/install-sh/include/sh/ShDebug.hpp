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
#ifndef SHDEBUG_HPP
#define SHDEBUG_HPP

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef SH_DEBUG
#include <iostream>
#include <cstdlib>

#define SH_DEBUG_PRINT(x) { std::cerr << __FILE__ << ":" << __LINE__ << ": " << x << std::endl; }
#define SH_DEBUG_WARN(x) { SH_DEBUG_PRINT("Warning: " << x) }
#define SH_DEBUG_ERROR(x) { SH_DEBUG_PRINT("Error: " << x) }
#define SH_DEBUG_ASSERT(cond) { if (!(cond)) { SH_DEBUG_ERROR("Assertion failed: " << # cond); abort(); } }

#else

#define SH_DEBUG_PRINT(x)
#define SH_DEBUG_WARN(x)
#define SH_DEBUG_ERROR(x)
#define SH_DEBUG_ASSERT(cond)

#endif // SH_DEBUG

#endif
