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
#ifndef SHNIBBLESIMPL_HPP
#define SHNIBBLESIMPL_HPP

#include "ShNibbles.hpp"
#include "ShTexCoord.hpp"

namespace SH {

template<typename T>
ShProgram keep(const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InOutType SH_NAMEDECL(attr, name); 
  } SH_END_PROGRAM;
  nibble.name("keep");
  return nibble;
}

template<typename T>
ShProgram dup(const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InputType SH_NAMEDECL(attr, name); 
    typename T::OutputType SH_NAMEDECL(attr1, name + "_1") = attr; 
    typename T::OutputType SH_NAMEDECL(attr2, name + "_2") = attr; 
  } SH_END_PROGRAM;
  nibble.name("dup");
  return nibble;
}

template<typename T>
ShProgram lose(const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InputType SH_NAMEDECL(attr, name);
  } SH_END_PROGRAM;
  nibble.name("lose");
  return nibble;
};

template<typename T>
ShProgram access(const ShBaseTexture1D<T> &tex, const std::string &tcname, const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    ShInputTexCoord1f SH_NAMEDECL(tc, tcname);
    typename T::OutputType SH_NAMEDECL(result, name) = tex(tc);
  } SH_END;
  nibble.name("access");
  return nibble;
}

template<typename T>
ShProgram access(const ShBaseTexture2D<T> &tex, const std::string & tcname, const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    ShInputTexCoord2f SH_NAMEDECL(tc, tcname);
    typename T::OutputType SH_NAMEDECL(result, name) = tex(tc);
  } SH_END;
  nibble.name("access");
  return nibble;
}

template<typename T>
ShProgram access(const ShBaseTextureRect<T> &tex, const std::string & tcname, const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    ShInputTexCoord2f SH_NAMEDECL(tc, tcname);
    typename T::OutputType SH_NAMEDECL(result, name) = tex(tc);
  } SH_END;
  nibble.name("access");
  return nibble;
}

template<typename T>
ShProgram access(const ShBaseTexture3D<T> &tex, const std::string & tcname, const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    ShInputTexCoord3f SH_NAMEDECL(tc, tcname);
    typename T::OutputType SH_NAMEDECL(result, name) = tex(tc);
  } SH_END;
  nibble.name("access");
  return nibble;
}

template<typename T>
ShProgram access(const ShBaseTextureCube<T> &tex, const std::string & tcname, const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    ShInputTexCoord3f SH_NAMEDECL(tc, tcname);
    typename T::OutputType SH_NAMEDECL(result, name) = tex(tc);
  } SH_END;
  nibble.name("access");
  return nibble;
}

template<typename T, int Rows, int Cols, ShBindingType Binding, typename T2>
ShProgram transform(const ShMatrix<Rows, Cols, Binding, T2> &m, const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InOutType SH_NAMEDECL(attrib, name) = m | attrib;
  } SH_END;
  nibble.name("transform");
  return nibble;
}

template<typename T, typename T2>
ShProgram cast(const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InputType SH_NAMEDECL(in, name);
    typename T2::OutputType SH_NAMEDECL(out, name) = cast<T2::typesize>( in );
  } SH_END;
  nibble.name("cast");
  return nibble;
}

template<typename T, typename T2>
ShProgram fillcast(const std::string & name) {
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InputType SH_NAMEDECL(in, name);
    typename T2::OutputType SH_NAMEDECL(out, name) = fillcast<T2::typesize>( in );
  } SH_END;
  nibble.name("fillcast");
  return nibble;
}

#define SHNIBBLE_UNARY_OP(opfunc, opcode) \
template<typename T>\
ShProgram opfunc(const std::string & name) {\
  ShProgram nibble = SH_BEGIN_PROGRAM() {\
    typename T::InOutType SH_NAMEDECL(x, name) = opcode;\
  } SH_END;\
  nibble.name(# opfunc); \
  return nibble; \
}

SHNIBBLE_UNARY_OP(abs, abs(x));
SHNIBBLE_UNARY_OP(acos, acos(x));
SHNIBBLE_UNARY_OP(asin, asin(x));
SHNIBBLE_UNARY_OP(cos, cos(x));
SHNIBBLE_UNARY_OP(frac, frac(x));
SHNIBBLE_UNARY_OP(sin, sin(x));
SHNIBBLE_UNARY_OP(sqrt, sqrt(x));
SHNIBBLE_UNARY_OP(normalize, normalize(x));
SHNIBBLE_UNARY_OP(pos, pos(x));

#define SHNIBBLE_BINARY_OP(opfunc, opcode) \
template<typename T1, typename T2> \
ShProgram opfunc(const std::string & output_name, \
    const std::string & input_name0, const std::string & input_name1) { \
  ShProgram nibble = SH_BEGIN_PROGRAM() { \
    typename T1::InputType SH_NAMEDECL(a, input_name0); \
    typename T2::InputType SH_NAMEDECL(b, input_name1); \
    typename SelectType<(T1::typesize > T2::typesize), typename T1::OutputType, typename T2::OutputType>::type\
      SH_NAMEDECL(result, output_name) = opcode; \
  } SH_END; \
  nibble.name(# opfunc); \
  return nibble; \
} \
\
template<typename T1> \
ShProgram opfunc(const std::string & output_name,\
    const std::string & input_name0, const std::string & input_name1 ) { \
  return opfunc<T1, T1>(output_name, input_name0, input_name1); \
}

SHNIBBLE_BINARY_OP(add, a + b)
SHNIBBLE_BINARY_OP(sub, a - b)
SHNIBBLE_BINARY_OP(mul, a * b)
SHNIBBLE_BINARY_OP(div, a / b) 
SHNIBBLE_BINARY_OP(pow, pow(a, b))
SHNIBBLE_BINARY_OP(slt, a < b) 
SHNIBBLE_BINARY_OP(sle, a <= b) 
SHNIBBLE_BINARY_OP(sgt, a > b) 
SHNIBBLE_BINARY_OP(sge, a >= b) 
SHNIBBLE_BINARY_OP(seq, a == b) 
SHNIBBLE_BINARY_OP(sne, a != b) 
SHNIBBLE_BINARY_OP(mod, mod(a, b))
SHNIBBLE_BINARY_OP(min, min(a, b))
SHNIBBLE_BINARY_OP(max, max(a, b))

template<typename T> 
ShProgram dot(const std::string & name) { 
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T::InputType SH_DECL(a); 
    typename T::InputType SH_DECL(b); 
    ShOutputAttrib1f SH_NAMEDECL(result, name) = dot(a, b); 
  } SH_END; 
  nibble.name("dot");
  return nibble;
}

template<typename T1, typename T2>
ShProgram lerp(const std::string & name) { 
  ShProgram nibble = SH_BEGIN_PROGRAM() {
    typename T1::InputType SH_DECL(a); 
    typename T1::InputType SH_DECL(b); 
    typename T2::InputType SH_DECL(alpha); 
    typename T1::OutputType SH_NAMEDECL(result, name) = lerp(alpha, a, b); 
  } SH_END; 
  nibble.name("lerp");
  return nibble;
}

template<typename T1>
ShProgram lerp(const std::string & name) { 
  return lerp<T1, T1>(name);
}


}

#endif
