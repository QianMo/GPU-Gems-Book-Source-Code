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
#ifndef SHBACKEND_HPP
#define SHBACKEND_HPP

#include <vector>
#include <iosfwd>
#include <string>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShProgram.hpp"
#include "ShVariableNode.hpp"

namespace SH  {

class ShStream;

class
SH_DLLEXPORT ShBackendCode : public ShRefCountable {
public:
  virtual ~ShBackendCode();

  /// Used by a register allocater to signal that a register should be
  /// allocated to var. Return true iff the allocation succeeded.
  virtual bool allocateRegister(const ShVariableNodePtr& var) = 0;

  /// Used by the register allocator to signal that the register used
  /// by var can be used by other registers in future
  /// allocateRegister() calls.
  virtual void freeRegister(const ShVariableNodePtr& var) = 0;
  
  /// Upload this shader code to the GPU.
  virtual void upload() = 0;

  /// Bind this shader code after it has been uploaded.
  virtual void bind() = 0;

  /// Update the value of a uniform parameter after it has changed.
  virtual void updateUniform(const ShVariableNodePtr& uniform) = 0;

  virtual std::ostream& print(std::ostream& out) = 0;

  /// Prints input and output specification in target-specific format
  // (Useful for how to format long tuple input on targets 
  // that only support limited tuple lengths) 
  virtual std::ostream& describe_interface(std::ostream& out) = 0;
};

typedef ShPointer<ShBackendCode> ShBackendCodePtr;
typedef ShPointer<const ShBackendCode> ShBackendCodeCPtr;

class ShTransformer;
class
SH_DLLEXPORT ShBackend : public ShRefCountable {
public:
  virtual ~ShBackend();
  virtual std::string name() const = 0;

  /// Generate the backend code for a particular shader. Ensure that
  /// ShEnvironment::shader is the same as shader before calling this,
  /// since extra variables may be declared inside this function!
  virtual ShBackendCodePtr generateCode(const std::string& target,
                                        const ShProgramNodeCPtr& shader) = 0;

  // execute a stream program, if supported
  virtual void execute(const ShProgramNodeCPtr& program, ShStream& dest) = 0;
  
  typedef std::vector< ShPointer<ShBackend> > ShBackendList;

  static ShBackendList::iterator begin();
  static ShBackendList::iterator end();

  static ShPointer<ShBackend> lookup(const std::string& name);

protected:
  ShBackend();
  
private:
  static void init();

  static ShBackendList* m_backends;
  static bool m_doneInit;
};

typedef ShPointer<ShBackend> ShBackendPtr;
typedef ShPointer<const ShBackend> ShBackendCPtr;

}

#endif
