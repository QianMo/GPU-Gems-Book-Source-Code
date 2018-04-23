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
#ifndef SHSWIZZLE_HPP
#define SHSWIZZLE_HPP

#include <iosfwd>
//#include <vector>
#include "ShDllExport.hpp"
#include "ShException.hpp"

namespace SH {

/** Represents swizzling of a variable.
 * Swizzling takes at least one element from an n-tuple and in
 * essence makes a new n-tuple with those elements in it. To actually
 * perform a swizzle using Sh, you should use the operator() on the
 * variable you are swizzling. This class is internal to Sh.
 *
 * Swizzles can be combined ("swizzle algebra") using the
 * operator*(). This class currently only supports host-time constant
 * swizzles, ie. you cannot use shader variables to specify swizzling
 * order.
 *
 * This class is also used for write masks, at least at the
 * intermediate level.
 *
 * Note that, at the moment, when combining swizzles indices are
 * checked to be sane, but original indices are not checked for
 * sanity, since currently swizzles don't know anything about (in
 * particular the size of) the tuple which they are swizzling.
 */
class
SH_DLLEXPORT
ShSwizzle {
public:

  // Null swizzle
  ShSwizzle();

  /// Identity swizzle: does nothing at all.
  ShSwizzle(int srcSize);
  /// Use one element from the original tuple.
  ShSwizzle(int srcSize, int i0);
  /// Use two elements from the original tuple.
  ShSwizzle(int srcSize, int i0, int i1);
  /// Use three elements from the original tuple.
  ShSwizzle(int srcSize, int i0, int i1, int i2);
  /// Use four elements from the original tuple.
  ShSwizzle(int srcSize, int i0, int i1, int i2, int i3);
  /// Use an arbitrary number of elements from the original tuple.
  ShSwizzle(int srcSize, int size, int* indices);

  ShSwizzle(const ShSwizzle& other);
  ~ShSwizzle();

  ShSwizzle& operator=(const ShSwizzle& other);
  
  /// Combine a swizzle with this one, as if it occured after this
  /// swizzle occured.
  ShSwizzle& operator*=(const ShSwizzle& other);

  /// Combine two swizzles with left-to-right precedence.
  ShSwizzle operator*(const ShSwizzle& other) const;

  /// Determine how many elements this swizzle results in.
  int size() const { return m_size; }

  /// Obtain the index of the \a i'th element. 0 <= i < size().
  /// This is int so that printing out the result won't give something
  /// weird
  int operator[](int i) const;

  /// Determine whether this is an identity swizzle.
  bool identity() const;

  /// Determine whether two swizzles are identical
  bool operator==(const ShSwizzle& other) const;
  
private:
  // copies the other swizzle's elements 
  void copy(const ShSwizzle &other, bool islocal);

  // throws an exception if index < 0 or index >= m_srcSize
  void checkSrcSize(int index); 

  // allocates the m_indices array to current m_size
  // returns true 
  bool alloc(); 

  // deallocates the m_indices array 
  void dealloc();

  // returns whether we're using local 
  bool local() const;

  // returns the identity swiz value on this machine
  int idswiz() const;

  // Declare these two first so alignment problems don't make the ShSwizzle struct larger
  int m_srcSize;
  int m_size;

  // when srcSize <= 255 and size <= 4, use local.
  // local is always initialized to 0x03020101, so identity comparison is
  // just an integer comparison using intval
  union {
    unsigned char local[4];
    int intval;
    int* ptr;
  } m_index;

  friend SH_DLLEXPORT std::ostream& operator<<(std::ostream& out, const ShSwizzle& swizzle);
};

/// Thrown when an invalid swizzle is specified (e.g. an index in the
/// swizzle is out of range).
class
SH_DLLEXPORT ShSwizzleException : public ShException 
{
public:
  ShSwizzleException(const ShSwizzle& s, int idx, int size);
};
  
}

#include "ShSwizzleImpl.hpp"
  
#endif
