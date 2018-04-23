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
#ifndef SHREFCOUNT_HPP
#define SHREFCOUNT_HPP

#include <utility>
#include "ShDllExport.hpp"

// #define SH_REFCOUNT_DEBUGGING

#ifdef SH_REFCOUNT_DEBUGGING
#include <iostream>
#include <iomanip>
#include "ShDebug.hpp"

#define SH_RCDEBUG_GREEN std::cerr << "[32m"
#define SH_RCDEBUG_RED std::cerr << "[31m"
#define SH_RCDEBUG_BLUE std::cerr << "[34m"
#define SH_RCDEBUG_NORMAL std::cerr << "[0m"

#endif

namespace SH {

/** A class which can be reference-counted.
 * These are classes you can wrap in an ShRefCount. Inherit from this
 * if you want to reference-count your class.
 */
class 
SH_DLLEXPORT
ShRefCountable 
{
public:
  ShRefCountable()
    : m_refCount(0)
  {
  }

  ShRefCountable(const ShRefCountable&)
    : m_refCount(0)
    // any new RefCountable should have a zero refcount, even if it's
    // made as a copy
  {
  }

  ShRefCountable& operator=(const ShRefCountable&)
  {
    // we don't actually change refCount here
    // this is indeed the intended behaviour
    return *this;
  }

#ifdef SH_REFCOUNT_DEBUGGING
  // Just to make this polymorphic, so typeid() works as expected
  virtual ~ShRefCountable() {}   
#endif
  
  int acquireRef() const
  {
#ifdef SH_REFCOUNT_DEBUGGING
    SH_RCDEBUG_GREEN;
    std::cerr << "   [+] " << std::setw(10) << this << " <" << typeid(*this).name() << ">"
              << ": " << m_refCount << "->" << (m_refCount + 1) << std::endl;
    SH_RCDEBUG_NORMAL;
#endif
    return ++m_refCount;
  }
  
  int releaseRef() const
  {
#ifdef SH_REFCOUNT_DEBUGGING
    SH_RCDEBUG_RED;
    std::cerr << "   [-] " << std::setw(10) << this << " <" << typeid(*this).name() << ">"
              << ": " << m_refCount << "->" << (m_refCount - 1) << std::endl;
    SH_RCDEBUG_NORMAL;
#endif
    return --m_refCount;
  }

  int refCount() const
  {
    return m_refCount;
  }

private:
  mutable int m_refCount;
};

/** A reference-counting smart pointer. 
 */
template<typename T>
class ShPointer 
{
public:
  ShPointer();
  ShPointer(T* object);
  ShPointer(const ShPointer& other);
  template<typename S>
  ShPointer(const ShPointer<S>& other);
  
  ~ShPointer();

  ShPointer& operator=(T* other);
  ShPointer& operator=(const ShPointer& other);
  template<typename S>
  ShPointer& operator=(const ShPointer<S>& other);

  /// Two references are equal if they point to the same object.
  bool operator==(const ShPointer& other) const;

  /// Two references are equal if they point to the same object.
  bool operator!=(const ShPointer& other) const;

  /// Actually compares the pointers.
  bool operator<(const ShPointer& other) const;

  T& operator*() const;
  T* operator->() const;

  /// Return true iff this is not a reference to a null pointer
  /// @todo Maybe use boost's unspecified-bool-type trick
  operator bool() const;

  /// Obtain the total amount of references to the referenced object.
  int refCount() const;

  /// Obtain a pointer to the object we reference count
  T* object() const;

  void swap(ShPointer& other);

private:
  void releaseRef();
  
  T* m_object;
};

template<typename T, typename S>
ShPointer<T> shref_static_cast(const ShPointer<S>& other);

template<typename T, typename S>
ShPointer<T> shref_dynamic_cast(const ShPointer<S>& other);

template<typename T, typename S>
ShPointer<T> shref_const_cast(const ShPointer<S>& other);

} // namespace SH

#include "ShRefCountImpl.hpp"

#endif
