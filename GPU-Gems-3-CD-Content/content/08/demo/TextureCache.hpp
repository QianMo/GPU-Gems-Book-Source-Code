#pragma once

#include <set>
#include <algorithm>
#include "RenderableTexture2D.hpp"

//--------------------------------------------------------------------------------------
class TextureCache
{
private:
  typedef std::set<RenderableTexture2D*> TextureSet;
  TextureSet m_Textures;

  // Not implemented
  TextureCache(const TextureCache &);

public:
  TextureCache() {}

  ~TextureCache()
  {
    ReleaseAll();
  }

  void ReleaseAll()
  {
    for (TextureSet::iterator i = m_Textures.begin(); i != m_Textures.end(); ++i) {
      SAFE_DELETE(*i);
    }
    m_Textures.clear();
  }

  // NOTE: Silently does nothing if called with a null pointer
  void Add(const TextureSet::value_type &t)
  {
    if (t) {
      m_Textures.insert(t);
    }
  }

  TextureSet::value_type Get()
  {
    if (!m_Textures.empty()) {
      TextureSet::value_type T = *m_Textures.begin();
      m_Textures.erase(m_Textures.begin());
      return T;
    } else {
      throw std::runtime_error("No texture available in cache!");
    }
  }

  bool Empty() const
  {
    return m_Textures.empty();
  }

  // Explicitly remove a given texture from the cache
  void Remove(const TextureSet::value_type &t)
  {
    m_Textures.erase(t);
  }

  // Adopt (add to us, remove from them) all of the textures from another cache
  void AdoptAll(TextureCache &c)
  {
    while (!c.Empty()) {
      Add(c.Get());
    }
  }
};