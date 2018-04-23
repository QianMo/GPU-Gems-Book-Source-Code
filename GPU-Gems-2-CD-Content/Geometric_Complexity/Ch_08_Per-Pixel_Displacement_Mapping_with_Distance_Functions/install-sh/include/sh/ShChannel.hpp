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
#ifndef SHCHANNEL_HPP
#define SHCHANNEL_HPP

#include "ShChannelNode.hpp"
#include "ShProgram.hpp"
#include "ShGeneric.hpp"

namespace SH {

/** The client interface to a single-channel typed data stream.
 * Really this hides an ShChannelNode which is the true representation
 * of the channel.   The template argument is the element type stored.
 * @todo copy constructor, operator=(), etc.
 */
template<typename T>
class ShChannel : public ShMetaForwarder {
public:
  /// Construct a channel without any associated memory.
  ShChannel();
  /// Construct a channel with \a count elements in \a memory
  ShChannel(const ShMemoryPtr& memory, int count);

  /// Set this channel to use \a count elements in \a memory
  void memory(const ShMemoryPtr& memory, int count);

  /// Return the number of elements in this channel
  int count() const;
  /// Return this channel's memory
  ShPointer<const ShMemory> memory() const;
  /// Return this channel's memory
  ShMemoryPtr memory();

  /// Fetch the current element from this stream.
  /// This is only useful in stream programs
  T operator()() const;

  /// Indexed lookup from the stream
  template<typename T2>
  T operator[](const ShGeneric<1, T2>& index) const;

  /// Return the node internally wrapped by this channel object
  ShChannelNodePtr node();
  /// Return the node internally wrapped by this channel object
  const ShChannelNodePtr node() const;

  /// Execute fully bound single-output stream program and place result in channel.
  ShChannel& operator=(const ShProgram& program);
  
private:
  /// The node this object is a thin wrapper for
  ShChannelNodePtr m_node;
};

/** Apply a programs to a single channel.
 * Bind a channel as an input to a program.   The implementation
 * supports currying, and returns a program with one less input.
 */
template<typename T>
ShProgram connect(const ShChannel<T>& channel, const ShProgram& program);

/** Equivalent to connect(p,c). 
 * Bind a channel as an input to a program.   The implementation
 * supports currying, and returns a program with one less input.
 */
template<typename T>
ShProgram operator<<(const ShProgram& program, const ShChannel<T>& channel);

}

#include "ShChannelImpl.hpp"

#endif
