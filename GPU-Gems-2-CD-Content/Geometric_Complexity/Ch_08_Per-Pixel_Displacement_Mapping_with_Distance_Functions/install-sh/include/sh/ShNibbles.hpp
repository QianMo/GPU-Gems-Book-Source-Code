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
#ifndef SHNIBBLES_HPP
#define SHNIBBLES_HPP

/** \file ShNibbles.hpp
 * A set of nibbles that are functions that return ShPrograms for
 * basic operations.
 */

#include "ShBaseTexture.hpp"
#include "ShProgram.hpp"
#include "ShMatrix.hpp"

namespace SH {

/** \brief Creates a keep nibble.
 * The keep nibble passes through one input of type T named with
 * the given name.
 */
template<typename T>
ShProgram keep(const std::string & name = ""); 

/** \brief Creates a dup nibble.
 * The dup nibble makes an extra copy of on input of type T named with
 * the given name.
 */
template<typename T>
ShProgram dup(const std::string & name = "var"); 

/** \brief Creates a lose nibble.
 * The lose nibble discards through one input of type T named with
 * the given name.
 */
template<typename T>
ShProgram lose(const std::string & name = "");

/** \brief 2D texture lookup nibble 
 * Inputs: IN(0) ShTexCoord1f texcoord
 * Outputs: OUT(0) name = tex(tc) */
template<typename T> 
ShProgram access(const ShBaseTexture1D<T> &tex, const std::string & tcname = "texcoord", const std::string & name = "result");

/** \brief 2D texture lookup nibble 
 * Inputs: IN(0) ShTexCoord2f texcoord
 * Outputs: OUT(0) name = tex(tc) */
template<typename T> 
ShProgram access(const ShBaseTexture2D<T> &tex, const std::string & tcname = "texcoord", const std::string & name = "result");

/** \brief Rect texture lookup nibble 
 * Inputs: IN(0) ShTexCoord2f texcoord
 * Outputs: OUT(0) name = tex(tc) */
template<typename T> 
ShProgram access(const ShBaseTextureRect<T> &tex, const std::string & tcname = "texcoord", const std::string & name = "result");

/** \brief 3D texture lookup nibble 
 * Inputs: IN(0) ShTexCoord3f texcoord
 * Outputs: OUT(0) name = tex(tc) */
template<typename T> 
ShProgram access(const ShBaseTexture3D<T> &tex, const std::string & tcname = "texcoord", const std::string & name = "result");

/** \brief Cube texture lookup nibble 
 * Inputs: IN(0) ShTexCoord3f texcoord
 * Outputs: OUT(0) name = tex(tc) */
template<typename T> 
ShProgram access(const ShBaseTextureCube<T> &tex, const std::string & tcname = "texcoord", const std::string & name = "result");

/** \brief transformation nibble 
 * Creates a nibble that transforms a variable of type T by 
 * a matrix of type ShMatrix<Rows, Cols, Binding, V>
 * Inputs: IN(0) attrib
 * Outputs: OUT(0) T name = m | attrib
 */
template<typename T, int Rows, int Cols, ShBindingType Binding, typename T2>
ShProgram transform(const ShMatrix<Rows, Cols, Binding, T2> &m, const std::string & name = "result");

/** \brief Casting nibble 
 * Casts from tuple type T to type T2
 * If T::typesize < T2::typesize, pads with 0 components at end.
 * If T::typesize > T2::typesize, truncates components at end. 
 * Inputs: IN(0) T in
 * Outputs: OUT(0) T2 name 
 *
 */
template<typename T, typename T2>
ShProgram cast(const std::string & name = "result");

/** \brief Fill-Casting nibble 
 * Casts from tuple type T to type T2
 * If T::typesize < T2::typesize, pads with repeated last component at end. 
 * If T::typesize > T2::typesize, truncates components at end. 
 * Inputs: IN(0) T in
 * Outputs: OUT(0) T2 name 
 *
 */
template<typename T, typename T2>
ShProgram fillcast(const std::string & name = "result");

/**@{ \brief Nibbles for unary operators
 * Inputs: IN(0) T name 
 * Outputs: OUT(0) T name 
 */
template<typename T> ShProgram abs(const std::string & name = "result");
template<typename T> ShProgram acos(const std::string & name = "result");
template<typename T> ShProgram asin(const std::string & name = "result");
template<typename T> ShProgram cos(const std::string & name = "result");
template<typename T> ShProgram frac(const std::string & name = "result");
template<typename T> ShProgram sin(const std::string & name = "result");
template<typename T> ShProgram sqrt(const std::string & name = "result");
template<typename T> ShProgram normalize(const std::string & name = "result");
template<typename T> ShProgram pos(const std::string & name = "result");
//@}

/**@{ \brief Nibbles for binary operators
 * Inputs: IN(0) T1 a
 *         IN(1) T2 b (by default T2 = T1)
 * Outputs: OUT(0) name  (type is T1 or T2, whichever has more components)
 */
#define SHNIBBLE_BINARY_OP_DECL(opname) \
  template<typename T1, typename T2> ShProgram opname(const std::string & output_name = "result",\
      const std::string & input_name0 = "x", const std::string & input_name1 = "y"); \
  template<typename T1> ShProgram opname(const std::string & output_name = "result",\
      const std::string & input_name0 = "x", const std::string & input_name1 = "y"); 
SHNIBBLE_BINARY_OP_DECL(add);
SHNIBBLE_BINARY_OP_DECL(sub);
SHNIBBLE_BINARY_OP_DECL(mul);
SHNIBBLE_BINARY_OP_DECL(div);
SHNIBBLE_BINARY_OP_DECL(pow);
SHNIBBLE_BINARY_OP_DECL(slt);
SHNIBBLE_BINARY_OP_DECL(sle);
SHNIBBLE_BINARY_OP_DECL(sgt);
SHNIBBLE_BINARY_OP_DECL(sge);
SHNIBBLE_BINARY_OP_DECL(seq);
SHNIBBLE_BINARY_OP_DECL(sne);
SHNIBBLE_BINARY_OP_DECL(max);
SHNIBBLE_BINARY_OP_DECL(min);
SHNIBBLE_BINARY_OP_DECL(mod);

/** @{ \brief Nibble for dot product operator
 * Inputs: IN(0) T a
 *         IN(1) T b
 * Outputs: OUT(0) ShAttrib1f name  
 */
template<typename T> ShProgram dot(const std::string & name = "result");
//@}

/** @{ \brief Nibble for lerp operator
 * Inputs: IN(0) T1 a
 *         IN(1) T1 b
 *         IN(2) T2 alpha (T2 = T1 by default)
 * Outputs: OUT(0) T1 name = a * alpha + b * (1 - alpha)  
 */
template<typename T1, typename T2>
ShProgram lerp(const std::string & name = "result");

template<typename T1>
ShProgram lerp(const std::string & name = "result");
//@}

}


#include "ShNibblesImpl.hpp"

#endif
