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
#ifndef SHSTATEMENT_HPP
#define SHSTATEMENT_HPP

#include <iosfwd>
#include <set>
#include <list>
#include "ShOperation.hpp"
#include "ShDllExport.hpp"
#include "ShVariable.hpp"

namespace SH {

/** Dummy class representing additional information that can be stored
 *  in statements.
 */
class 
SH_DLLEXPORT
ShStatementInfo {
public:
  virtual ~ShStatementInfo();

  virtual ShStatementInfo* clone() const = 0;
  
protected:
  ShStatementInfo();

};

/** A single statement.
 * Represent a statement of the form 
 * <pre>dest := src[0] op src[1]</pre>
 * or, for unary operators: 
 * <pre>dest := op src[0]</pre>
 * or, for op == SH_OP_ASN:
 * <pre>dest := src[0]</pre>
 */
class
SH_DLLEXPORT ShStatement {
public:
  ShStatement(ShVariable dest, ShOperation op);
  ShStatement(ShVariable dest, ShOperation op, ShVariable src);
  ShStatement(ShVariable dest, ShVariable src0, ShOperation op, ShVariable src1);
  ShStatement(ShVariable dest, ShOperation op, ShVariable src0, ShVariable src1, ShVariable src2);
  ShStatement(const ShStatement& other);
  
  ~ShStatement();

  ShStatement& operator=(const ShStatement& other);
  
  
  ShVariable dest;
  ShVariable src[3];
  
  ShOperation op;

  // Used by the optimizer and anything else that needs to store extra
  // information in statements.
  // Anything in here will be deleted when this statement is deleted.
  std::list<ShStatementInfo*> info;

  // Return the first entry in info whose type matches T, or 0 if no
  // such entry exists.
  template<typename T>
  T* get_info();

  // Delete and remove all info entries matching the given type.
  template<typename T>
  void destroy_info();

  // Add the given statement information to the end of the info list.
  void add_info(ShStatementInfo* new_info);

  // Remove the given statement information from the list.
  // Does not delete it, so be careful!
  void remove_info(ShStatementInfo* old_info);
  
  bool marked;

  friend SH_DLLEXPORT std::ostream& operator<<(std::ostream& out, const SH::ShStatement& stmt);
};


template<typename T>
T* ShStatement::get_info()
{
  for (std::list<ShStatementInfo*>::iterator I = info.begin(); I != info.end(); ++I) {
    T* item = dynamic_cast<T*>(*I);
    if (item) {
      return item;
    }
  }
  return 0;
}

template<typename T>
void ShStatement::destroy_info()
{
  for (std::list<ShStatementInfo*>::iterator I = info.begin(); I != info.end();) {
    T* item = dynamic_cast<T*>(*I);
    if (item) {
      I = info.erase(I);
      delete item;
    } else {
      ++I;
    }
  }
}

} // namespace SH

#endif
