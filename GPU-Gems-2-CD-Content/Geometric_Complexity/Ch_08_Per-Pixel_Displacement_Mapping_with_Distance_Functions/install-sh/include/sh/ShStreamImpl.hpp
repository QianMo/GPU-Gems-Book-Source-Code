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
#ifndef SHSTREAMLISTIMPL_HPP
#define SHSTREAMLISTIMPL_HPP

#include "ShStream.hpp"
#include "ShChannel.hpp"

namespace SH {

template<typename T>
ShStream::ShStream(const ShChannel<T>& channel)
{
  m_nodes.push_back(channel.node());
}

template<typename T>
void ShStream::append(const ShChannel<T>& channel)
{
  m_nodes.push_back(channel.node());
}

template<typename T>
void ShStream::prepend(const ShChannel<T>& channel)
{
  m_nodes.push_front(channel.node());
}

// Ways to form a combined stream
template<typename T1, typename T2>
ShStream combine(const ShChannel<T1>& left, const ShChannel<T2>& right)
{
  ShStream stream(left);
  stream.append(right);
  return stream;
}

template<typename T2>
ShStream combine(const ShStream& left, const ShChannel<T2>& right)
{
  ShStream stream = left;
  stream.append(right);
  return stream;
}

template<typename T1>
ShStream combine(const ShChannel<T1>& left, const ShStream& right)
{
  ShStream stream = right;
  stream.prepend(left);
  return stream;
}

// Aliases for combine
template<typename T1, typename T2>
ShStream operator&(const ShChannel<T1>& left, const ShChannel<T2>& right)
{
  return combine(left, right);
}

template<typename T2>
ShStream operator&(const ShStream& left, const ShChannel<T2>& right)
{
  return combine(left, right);
}

template<typename T1>
ShStream operator&(const ShChannel<T1>& left, const ShStream& right)
{
  return combine(left, right);
}

}
#endif
