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
#ifndef SHTOKEN_HPP
#define SHTOKEN_HPP

#include <vector>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShBlock.hpp"

namespace SH {

struct ShTokenArgument;
  
/** Possible types a token can have.
 * If you add to this list or change it, be sure to change tokenNames in
 * ShToken.cpp.
 */
enum ShTokenType {
  SH_TOKEN_IF,
  SH_TOKEN_ELSE,
  SH_TOKEN_ENDIF,
  SH_TOKEN_WHILE,
  SH_TOKEN_ENDWHILE,
  SH_TOKEN_DO,
  SH_TOKEN_UNTIL,
  SH_TOKEN_FOR,
  SH_TOKEN_ENDFOR,
  SH_TOKEN_BREAK,
  SH_TOKEN_CONTINUE,
};

/** A token in the (unparsed) parse tree.
 * This represents a token such as SH_IF. The token can optionally
 * have some arguments, see ShTokenArgument. Later these tokens
 * will be parsed into real control structures by the parser.
 */
class 
SH_DLLEXPORT ShToken : public ShBlock {
public:
  ShToken(ShTokenType type);
  ~ShToken();

  /// Return the type of this token.
  ShTokenType type();
  
  void print(std::ostream& out, int indent) const;

  /// Any arguments bound to the token. May be empty.
  std::vector<ShTokenArgument> arguments;
  
private:
  ShTokenType m_type;
};

typedef ShPointer<ShToken> ShTokenPtr;
typedef ShPointer<const ShToken> ShTokenCPtr;


}

#endif
