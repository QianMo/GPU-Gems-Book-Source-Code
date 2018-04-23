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
#ifndef SHCONTEXT_HPP
#define SHCONTEXT_HPP

#include <string>
#include <map>
#include "ShDllExport.hpp"
#include "ShProgram.hpp"

namespace SH {

class
SH_DLLEXPORT ShContext {
public:
  static ShContext* current();

  /// 0 means no optimizations. The default level is 2.
  int optimization() const;
  void optimization(int level);

  /// Whether exceptions are being thrown instead of error messages
  /// printed to stdout. The default is to throw exceptions.
  bool throw_errors() const;
  void throw_errors(bool on);

  /// Disable a particular optimization. All optimizations are
  /// enabled by default. Disabling an optimization takes place in
  /// addition to whatever effects the optimization level has.
  void disable_optimization(const std::string& name);
  /// Enable a particular optimization (rather, stop disabling it)
  void enable_optimization(const std::string& name);
  /// Check whether an optimization is disabled
  bool optimization_disabled(const std::string& name) const;
  
  typedef std::map<std::string, ShProgram> BoundProgramMap;

  BoundProgramMap::iterator begin_bound();
  BoundProgramMap::iterator end_bound();

  /// \internal
  void set_binding(const std::string& unit, ShProgram program);

  /// The program currently being constructed. May be null.
  ShProgramNodePtr parsing();

  /// Start constructing the given program
  void enter(const ShProgramNodePtr& program);

  /// Finish constructing the current program
  void exit();
  
private:
  ShContext();

  int m_optimization;
  bool m_throw_errors;
  
  BoundProgramMap m_bound;
  std::stack<ShProgramNodePtr> m_parsing;

  std::set<std::string> m_disabled_optimizations;
  
  static ShContext* m_instance;

  // NOT IMPLEMENTED
  ShContext(const ShContext& other);
  ShContext& operator=(const ShContext& other);
};

typedef ShContext::BoundProgramMap::iterator ShBoundIterator;

/// Get beginning of bound program map for current context
SH_DLLEXPORT
ShBoundIterator shBeginBound();

/// Get end of bound program map for current context
SH_DLLEXPORT
ShBoundIterator shEndBound();

}

#endif
