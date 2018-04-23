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
#ifndef SHTOKENIZER_HPP
#define SHTOKENIZER_HPP

#include <stack>
#include <queue>
#include <vector>
#include <iosfwd>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShVariable.hpp"
#include "ShBlock.hpp"
#include "ShException.hpp"
#include "ShToken.hpp"

namespace SH {

/** \brief A token argument, e.g. to SH_IF or SH_FOR.
 *
 * Each argument consists of a result variable, containing the value
 * of the argument, and a block list describing how that value is to
 * be computed */
struct 
SH_DLLEXPORT ShTokenArgument {
  ShTokenArgument(const ShVariable& result, const ShBlockListPtr& blockList)
    : result(result), blockList(blockList)
  {
  }
  
  ShVariable result; ///< Contains the value of the argument
  ShBlockListPtr blockList; ///< Specifies how result is computed
};

/** An exception indicating a tokenizer error.
 * Call message() on this exception to determine a detailed report of
 * the error.
 */
class 
SH_DLLEXPORT ShTokenizerException : public ShException {
public:
  ShTokenizerException(const std::string& error);
};
  
/** A tokenizer.
 * This is used during construction of the program, i.e. within a
 * BeginShader/EndShader block, to process control statements in a
 * tokenized fashion.
 *
 * Once the shader has been parsed there is no more need for the
 * tokenizer.
 *
 */
class 
SH_DLLEXPORT ShTokenizer {
public:
  ShTokenizer();

  /** @name Pushing and processing arguments
   * These should always be called together, like so:
   * pushArgQueue() && pushArg() && processArg(foo) && pushArg() &&
   * processArg(bar)
   */
  ///@{
  /// Call this to alllocate an argument queue. Always returns true.
  bool pushArgQueue();
  /// Indicate that an argument is coming. Always returns true.
  bool pushArg();
  /// Tokenize an argument, then add it to the argument queue. Always returns true.
  bool processArg(const ShVariable& result);
  ///@}

  /** @name Retrieving arguments */
  ///@{
  /// Retrieve and remove the oldest parsed argument at the current level
  ShTokenArgument getArgument();
  /// Pop the argument context (call after you've retrieved all your arguments)
  void popArgQueue();
  ///@}

  /// Get the currently active list
  ShBlockListPtr blockList();

private:
  std::stack<ShBlockListPtr> m_listStack;
  std::stack< std::queue<ShTokenArgument> > m_argQueueStack;
};

}

#endif
