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
#ifndef SH_METAIMPL_HPP
#define SH_METAIMPL_HPP

#include "ShMeta.hpp"

namespace SH {

inline ShMeta::~ShMeta()
{
}

inline std::string ShMeta::name() const
{
  return meta("n"); 
}

inline void ShMeta::name(const std::string& n)
{
  meta("n", n);
}

inline bool ShMeta::has_name() const
{
  return !meta("n").empty(); 
}

inline bool ShMeta::internal() const
{
  return !meta("i").empty(); 
}

inline void ShMeta::internal(bool i)
{
  meta("i", i ? "1" : "");
}

inline std::string ShMeta::title() const
{
  return meta("t");
}

inline void ShMeta::title(const std::string& t)
{
  meta("t", t);
}

inline std::string ShMeta::description() const
{
  return meta("d");
}

inline void ShMeta::description(const std::string& d)
{
  meta("d", d);
}

inline std::string ShMeta::meta(const std::string& key) const
{
  if(!m_meta) return std::string(); 

  MetaMap::const_iterator I = m_meta->find(key);
  if (I == m_meta->end()) return std::string();
  return I->second;
}

inline void ShMeta::meta(const std::string& key, const std::string& value)
{
  if(!m_meta) m_meta = new MetaMap();
  (*m_meta)[key] = value;
}

}

#endif

