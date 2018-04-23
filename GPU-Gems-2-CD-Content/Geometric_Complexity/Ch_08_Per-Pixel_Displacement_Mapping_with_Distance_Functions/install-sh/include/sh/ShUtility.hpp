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
#ifndef SHUTILITY_HPP
#define SHUTILITY_HPP

/** @file Utility.hh
 * Various utility functions, mostly for internal use.
 */

#include <iosfwd>
#include <map>
#include "ShDllExport.hpp"

namespace SH {

/// Print "indent" spaces to out.
SH_DLLEXPORT
std::ostream& shPrintIndent(std::ostream& out, int indent);

/** Check a boolean condition at compile time.
 * This follows Alexandrescu's excellent book "Modern C++ Design"
 */
template<bool B> struct ShCompileTimeChecker
{
  ShCompileTimeChecker(...);
};
template<> struct ShCompileTimeChecker<false> {
};

#define SH_STATIC_CHECK(expr, msg) \
{ \
  class SH_ERROR_##msg {} y; \
  (void)sizeof(ShCompileTimeChecker<(expr)>(y));\
}

struct 
SH_DLLEXPORT ShIgnore {
  template<typename T>
  T& operator&(T& other) { return other; }
};

/// SelectType::type == B ? T1 : T2 
template<bool B, typename T1, typename T2>
struct SelectType;

template<typename T1, typename T2>
struct SelectType<true, T1, T2> {
  typedef T1 type;
};

template<typename T1, typename T2>
struct SelectType<false, T1, T2> {
  typedef T2 type;
};

/// MatchType::matches = (T1 == T2)
template<typename T1, typename T2>
struct MatchType {
  static const bool matches = false; 
};

template<typename T>
struct MatchType<T, T> {
  static const bool matches = true; 
};

template<typename T, typename T1, typename T2>
struct MatchEitherType {
    static const bool matches = MatchType<T1, T>::matches ||
                                    MatchType<T2, T>::matches;
};

/** Returns true if T matches a given templated type.
 * For example, MatchTemplateType<int, ShInterval>::matches == false
 * but MatchTemplateType<ShInterval<int>, ShInterval>::matches == true
 *
 * You can subclass this like this:
 * template<typename T> struct MatchMyType: public MatchTemplateType<T, MyType> {}; 
 * to match your own complex types with less typing (hah hah, stupid pun).
 *
 * The basic class here is standards compliant and works in VC .NET 2003,
 * but not sure what will happen in more complex template vodoo.
 */
template<typename T, template<typename A> class B>
struct MatchTemplateType {
    static const bool matches = false; 
};

template<typename T, template<typename A> class B>
struct MatchTemplateType<B<T>, B> {
    static const bool matches = true; 
};

/** Takes a templated type and returns its template parameter. */
template<typename T, template<typename A> class B>
struct TemplateParameterType; 

template<typename T, template<typename A> class B>
struct TemplateParameterType<B<T>, B> {
    typedef T type;
};

}

#endif
