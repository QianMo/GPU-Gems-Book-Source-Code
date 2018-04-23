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
#ifndef SHVARIABLE_HPP
#define SHVARIABLE_HPP

#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShSwizzle.hpp"
#include "ShUtility.hpp"
#include "ShMetaForwarder.hpp"
#include "ShVariant.hpp"
#include "ShVariableNode.hpp"

namespace SH {

class ShProgram;

/** A reference and interface to a generic n-tuple variable.
 * Note: subclasses should not keep any additional data. All data
 * relevant to the node should be stored in m_node. This is due to
 * instances of subclasses of ShVariable being sliced when they get
 * placed in ShStatements.
*/
class 
SH_DLLEXPORT ShVariable : public ShMetaForwarder {
public:
  ShVariable();
  ShVariable(const ShVariableNodePtr& node);
  ShVariable(const ShVariableNodePtr& node, const ShSwizzle& swizzle, bool neg);

  ~ShVariable() {}

  ShVariable& operator=(const ShProgram& prg);
  
  bool null() const; ///< true iff node is a null pointer.
  
  bool uniform() const; ///< Is this a uniform (non-shader specific) variable?
  bool hasValues() const; ///< Does this variable have constant
                          ///(host-local) values?
  
  int size() const; ///< Get the number of elements in this variable,
                    /// after swizzling.
                    
  ShValueType valueType() const; ///< Returns index of the data type held in this node, or 0 if no node. 

  /**@name Metadata
   * This data is useful for various things, including asset
   * management.
   */
  //@{

  
  /// Set a range of values for this variable
  // TODO check if this works when swizzle contains one index more than once
  void rangeVariant(const ShVariant* low, const ShVariant* high);

  /// Obtain a lower bounds on this variable (tuple of same size as this)
  // @{
  ShVariantPtr lowBoundVariant() const;

  /// Obtain an upper bounds on this variable (tuple of same size as this)
  ShVariantPtr highBoundVariant() const;

  //@}
  
  /// Obtain the swizzling (if any) applied to this variable.
  const ShSwizzle& swizzle() const;

  /// Obtain the actual node this variable refers to.
  const ShVariableNodePtr& node() const;

  /// Return true if this variable is negated
  bool neg() const;

  bool& neg();

  ///
  
  /// Returns a copy of the variant (with swizzling & proper negation)
  ShVariantPtr getVariant() const;
  ShVariantPtr getVariant(int index) const;

  /** Sets result to this' variant if possible.
   * Otherwise, if swizzling or negation are required, then 
   * makes a copy into result.
   * @returns whether a copy was allocated
   *
   * (This function should only be used internally. the ref count
   * on result will be 1 if it's allocated as a copy.  You may
   * assign this to a refcounted pointer, and then manually release a ref.
   * @todo type figure out a cleaner way) 
   *
   * Since this allows you to possibly change the variant values without
   * triggering a uniform update, if loadVariant returns false, you must
   * call updateVariant() afterwards if you change any values in result. 
   * @{
   */
  bool loadVariant(ShVariant *&result) const;
  void updateVariant();
  // @}

  
  /** Sets the elements of this variant from other accounting for 
   * this' writemask and negations
   * @{
   */
  void setVariant(const ShVariant* other, bool neg, const ShSwizzle &writemask);
  void setVariant(ShVariantCPtr other, bool neg, const ShSwizzle &writemask);
  // @}
 
  /** Sets the indicated element of this' variant from other 
   * @{
   */
  void setVariant(const ShVariant* other, int index);
  void setVariant(ShVariantCPtr other, int index);
  // @}

  /** Sets this' variant from the contents of other
   * @{
   */
  void setVariant(const ShVariant* other);
  void setVariant(ShVariantCPtr other);
  // @}


  ShVariable operator()() const; ///< Identity swizzle
  ShVariable operator()(int) const;
  ShVariable operator()(int, int) const;
  ShVariable operator()(int, int, int) const;
  ShVariable operator()(int, int, int, int) const;
  ShVariable operator()(int size, int indices[]) const;
  
  ShVariable operator-() const;

  bool operator==(const ShVariable& other) const;
  bool operator!=(const ShVariable& other) const { return !((*this) == other); }

  /// @internal used by ShMatrix to set up its rows when swizzled.
  /// Sets this variable's node, swizzle, and negation bit to be
  /// identical to the given variable.
  void clone(const ShVariable& other);
  
protected:
  
  ShVariableNodePtr m_node; ///< The actual variable node we refer to.
  ShSwizzle m_swizzle; ///< Swizzling applied to this variable.
  bool m_neg; ///< True iff this variable is negated

  friend SH_DLLEXPORT std::ostream& operator<<(std::ostream& out, const ShVariable& shVariableToPrint);
};

}

#endif
