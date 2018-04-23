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
#ifndef SHEXCEPTION_HPP
#define SHEXCEPTION_HPP

#include <string>
#include "ShDllExport.hpp"

namespace SH {

/** A generic exception.
 * You should derive from this class to make more specific exceptions.
 */
class
SH_DLLEXPORT ShException {
public:
  /// Construct a general exception with the given message.
  ShException(const std::string& message);
  virtual ~ShException(); // Make this class virtual in orer to get
                          // RTTI info in.
  
  /// Return an informative message describing the exception.
  const std::string& message() const;
  
protected:
  std::string m_message;
};

/** An exception representing a parse error.
 */
class
SH_DLLEXPORT ShParseException : public ShException {
public:
  ShParseException(const std::string& message);
};

/** An exception representing a scoping violation
 */
class
SH_DLLEXPORT ShScopeException : public ShException {
public:
  ShScopeException(const std::string& message);
};

/** An exception relating to an ShImage operation.
 */
class
SH_DLLEXPORT ShImageException : public ShException {
public:
  ShImageException(const std::string& message);
};

/** An exception relating to an ShAlgebra operation.
 */
class
SH_DLLEXPORT ShAlgebraException : public ShException {
public:
  ShAlgebraException(const std::string& message);
};

/** An exception relating to an ShOptimizer operation.
 */
class
SH_DLLEXPORT ShOptimizerException : public ShException {
public:
  ShOptimizerException(const std::string& message);
};

/** An exception relating to an ShTransformer operation.
 */
class
SH_DLLEXPORT ShTransformerException : public ShException {
public:
  ShTransformerException(const std::string& message);
};

/** An exception relating to an ShBackend operation.
 */
class
SH_DLLEXPORT ShBackendException : public ShException {
public:
  ShBackendException(const std::string& message);
};

}

#endif
