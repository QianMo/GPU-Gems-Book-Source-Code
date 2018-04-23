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
#ifndef SHSTREAM_HPP
#define SHSTREAM_HPP

#include <list>
#include "ShDllExport.hpp"
#include "ShChannel.hpp"
#include "ShChannelNode.hpp"

namespace SH {

/** Dynamic list of channels.
 * The stream keep track (by reference) of an ordered lists of 
 * data channels, to be used as inputs and outputs of stream operations.
 * @see ShChannel
 */
class
SH_DLLEXPORT ShStream {
public:
  ShStream(const ShChannelNodePtr& node);
  
  template<typename T>
  ShStream(const ShChannel<T>& channel);

  typedef std::list<ShChannelNodePtr> NodeList;

  NodeList::const_iterator begin() const;
  NodeList::const_iterator end() const;
  NodeList::iterator begin();
  NodeList::iterator end();
  int size() const;

  template<typename T>
  void append(const ShChannel<T>& channel);
  template<typename T>
  void prepend(const ShChannel<T>& channel);

  void append(const ShChannelNodePtr& node);
  void prepend(const ShChannelNodePtr& node);
  // Execute fully bound stream program and place results in stream.
  ShStream& operator=(const ShProgram& program);
  
private:
  std::list<ShChannelNodePtr> m_nodes;
};

/** Combine two streams.
 * This concatenates the list of channels in the component streams.
 */
template<typename T1, typename T2>
ShStream combine(const ShChannel<T1>& left, const ShChannel<T2>& right);

/** Combine a stream and a channel.
 * This concatenates the given channel to the end of the list of
 * channels in the stream.
 */
template<typename T2>
ShStream combine(const ShStream& left, const ShChannel<T2>& right);

/** Combine a channel and a stream.
 * This concatenates the given channel to the start of the list of
 * channels in the stream.
 */
template<typename T1>
ShStream combine(const ShChannel<T1>& left, const ShStream& right);

SH_DLLEXPORT
ShStream combine(const ShStream& left, const ShStream& right);

/** An operator alias for combine between channels.
 */
template<typename T1, typename T2>
ShStream operator&(const ShChannel<T1>& left, const ShChannel<T2>& right);

/** An operator alias for combine between a stream and a channel.
 */
template<typename T2>
ShStream operator&(const ShStream& left, const ShChannel<T2>& right);

/** An operator alias for combine between a channel and a stream.
 */
template<typename T1>
ShStream operator&(const ShChannel<T1>& left, const ShStream& right);

/** An operator alias for combine between two streams.
 */
SH_DLLEXPORT
ShStream operator&(const ShStream& left, const ShStream& right);

/** Apply a program to a stream. 
 * This function connects streams onto the output of programs
 * TODO: is this right?  why is the stream argument first?
 */
SH_DLLEXPORT
ShProgram connect(const ShStream& stream, const ShProgram& program);

/** An operator alias for connect(p,s).
 */
SH_DLLEXPORT
ShProgram operator<<(const ShProgram& program, const ShStream& stream);


}

#include "ShStreamImpl.hpp"

#endif
