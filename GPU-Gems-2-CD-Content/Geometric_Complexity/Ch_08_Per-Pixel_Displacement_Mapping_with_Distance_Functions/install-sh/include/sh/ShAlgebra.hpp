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
#ifndef SHALGEBRA_HPP
#define SHALGEBRA_HPP

#include <string>
#include "ShDllExport.hpp"
#include "ShProgram.hpp"
#include "ShSyntax.hpp"
#include "ShAttrib.hpp"

namespace SH {

/** Replace inputs of b with outputs of a.
 * Functional composition.
 * Let A = a.outputs.size(), B = b.inputs.size().
 * If A > B, extra outputs are kept at end
 * If A < B, extra inputs are kept at end
 */
SH_DLLEXPORT
ShProgram connect(ShProgram a, ShProgram b);

/** Combine a and b.
 * Use all inputs from a and b and all outputs from a and b,
 * concatenated in order,
 * and perform all operations from both programs.
 */
SH_DLLEXPORT
ShProgram combine(ShProgram a, ShProgram b);

/** Combine a and b.
 * Use all inputs from a and b and all outputs from a and b,
 * combined by name,
 * and perform all operations from both programs.
 *
 * This combine detects pairs of inputs with matching names and types.
 * If this occurs, the later input is discarded and replaced with
 * a copy of the earlier one.   Unnamed inputs are all considered to
 * be unique.
 *
 * For instance, if a has inputs x, y, k, x, z and b has inputs w, y, x, v
 * then the result has inputs x, y, k, z, w, v
 */
SH_DLLEXPORT
ShProgram namedCombine(ShProgram a, ShProgram b);

/** Replace inputs of b with outputs of a.
 * Functional composition.
 * The outputs of a and inputs of b must all be named.
 *
 * For each output of a in positional order, this connects the output with an 
 * input of b of the same name/type that is not already connected with
 * another output of a.
 * Extra inputs remain at the end.  Extra outputs remain iff keepExtra = true 
 */
SH_DLLEXPORT
ShProgram namedConnect(ShProgram a, ShProgram b, bool keepExtra = false );

/** Renames all inputs named oldName to newName.
 */
SH_DLLEXPORT
ShProgram renameInput(ShProgram a, const std::string& oldName, const std::string& newName);

/** Renames all outputs named oldName to newName.
 */
SH_DLLEXPORT
ShProgram renameOutput(ShProgram a, const std::string& oldName, const std::string& newName);

/** Swizzles named outputs of a to match named inputs of b.
 * This only works on programs with inputs/outputs that all have unique names. 
 * Also, the inputs of b must be a subset of the outputs of a.
 */
SH_DLLEXPORT
ShProgram namedAlign(ShProgram a, ShProgram b);

/** Replaces parameter with attribute.
 * Replaces a uniform parameter by appending a
 * varying input attribute to the end of the list of inputs.
 */
SH_DLLEXPORT
ShProgram replaceUniform(ShProgram a, const ShVariable &var); 

/** Equivalent to combine(a,b).
 */
SH_DLLEXPORT
ShProgram operator&(ShProgram a, ShProgram b);

/** Equivalent to connect(b,a).
 */
SH_DLLEXPORT
ShProgram operator<<(ShProgram a, ShProgram b);

/** Equivalent to replaceUniform(p,var).
 */
SH_DLLEXPORT
ShProgram operator>>(ShProgram p, const ShVariable &var); 

/** Application operator.
 * The operator used for combine can also be used to apply a program
 * to a variable.   The implementation supports currying with delayed
 * read, which is equivalent to replacing an input with a parameter.
 */
template<int N, typename T>
ShProgram operator<<(ShProgram a, const ShGeneric<N, T>& v) {
  ShProgram vNibble = SH_BEGIN_PROGRAM() {
    ShAttrib<N, SH_OUTPUT, T> out;
    out.node()->specialType(v.node()->specialType());
    out = v;
  } SH_END_PROGRAM;
  return connect(vNibble, a); 
}

}

#endif
