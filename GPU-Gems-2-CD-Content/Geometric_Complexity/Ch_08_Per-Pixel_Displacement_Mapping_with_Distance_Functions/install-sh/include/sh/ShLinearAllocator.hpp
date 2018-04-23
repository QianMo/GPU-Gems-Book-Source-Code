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
#ifndef SHLINEARALLOCATOR_HPP
#define SHLINEARALLOCATOR_HPP

#include <map>
#include "ShDllExport.hpp"
#include "ShVariableNode.hpp"
#include "ShBackend.hpp"

namespace SH {

struct
SH_DLLEXPORT ShLifeTime {
  ShLifeTime()
  {
  }
  
  ShLifeTime(const ShVariableNodePtr& var, int first)
    : var(var), first(first), last(first)
  {
  }
  
  SH::ShVariableNodePtr var;
  int first, last;

  void mark(int index)
  {
    if (first > index) first = index;
    if (last < index) last = index;
  }
  
  bool operator<(const ShLifeTime& other) const
  {
    return first < other.first;
  }
};

/** A simple, basic-block based linear register allocator.
 */
class
SH_DLLEXPORT ShLinearAllocator {
public:
  ShLinearAllocator(ShBackendCodePtr backendCode);
  
  // Mark that a variable is alive at a given index.
  void mark(const ShVariableNodePtr& var, int index);

  // Dump the life times to stderr
  void debugDump();
  
  // Calls back the backend with register allocation/deallocation requests.
  void allocate();

private:
  ShBackendCodePtr m_backendCode;
  typedef std::map<ShVariableNodePtr, ShLifeTime> LifetimeMap;
  LifetimeMap m_lifetimes;
};

}

#endif
