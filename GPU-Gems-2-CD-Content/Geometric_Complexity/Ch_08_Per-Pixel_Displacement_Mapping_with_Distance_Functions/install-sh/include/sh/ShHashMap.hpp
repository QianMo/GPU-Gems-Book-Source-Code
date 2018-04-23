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
#ifndef SHHASHMAP_HPP
#define SHHASHMAP_HPP

#ifdef WIN32
#include <hash_map>
#else
#include <ext/hash_map>
#endif

#include <cstddef>
#include <iosfwd>


namespace SH {

/** @file ShHashMap.hpp
 * A wrapper around hash_map that behaves properly under both GNU libstdc++ 
 * and Microsoft's VS .NET libraries. The interface is the common subset
 * of functionality available in both implementations.
 * (This means that some of the "unusual" methods in the VC++ implementation
 * that really shouldn't be methods like lower_bound, upper_bound, etc.
 * are not available)
 *
 * The Less functor is only used in VS .NET, and the Equal functor is only
 * used under libstdc++.
 */
#ifdef WIN32
#define SH_STD_HASH(T) stdext::hash_compare<T>
/** Implementation of hash_compare interface */
template<class Key, class Hash, class Less>
class ShHashCompare {
  public:
    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;
    size_t operator( )(const Key& key) const { return m_hash(key); }
    bool operator( )(const Key& key1, const Key& key2) { return m_less(key1, key2); }

  private:
    Hash m_hash;
    Less m_less;
};

#else
#define SH_STD_HASH(T) ShHashFunc<T> 
template<typename T>
struct ShHashFunc {
  size_t operator()(const T& data) const
  { return size_t(data); }
};
#endif

template<class Key, class Data, class Hash=SH_STD_HASH(Key),
  class Less=std::less<Key>, class Equal=std::equal_to<Key> >
class ShHashMap {
#ifdef WIN32
    typedef stdext::hash_map<Key, Data, ShHashCompare<Key, Hash, Less> > map_type;
#else
    typedef __gnu_cxx::hash_map<Key, Data, Hash, Equal> map_type;
#endif
  public:

    typedef Key key_type;
    typedef Data data_type;
    typedef std::pair<const key_type, data_type> value_type;
    typedef int size_type;

    typedef typename map_type::iterator iterator;
    typedef typename map_type::const_iterator const_iterator;

    /** Default constructor/copy constructor should work */

    /** Wrapped member functions */
   
    /** Iterators */
    iterator begin() 
    { return m_map.begin(); }

    const_iterator begin() const 
    { return m_map.begin(); }

    iterator end() 
    { return m_map.end(); }

    const_iterator end() const 
    { return m_map.end(); }

    
    /** Map size */
    size_type size() const 
    { return m_map.size(); }

    size_type max_size() const 
    { return m_map.max_size(); }

    bool empty() const 
    { return m_map.empty(); }

    /** Insert/Delete elements */
    void clear()
    { m_map.clear(); }

    std::pair<iterator, bool> insert(const value_type &value)
    { return m_map.insert(value); }

    iterator insert(iterator hint, const value_type& value)
    { return m_map.insert(hint, value); }

    template<class InputIterator>
    void insert(InputIterator first, InputIterator last)
    { m_map.insert(first, last); }

    void erase(iterator pos)
    { m_map.erase(pos); }

    void erase(iterator first, iterator last)
    { m_map.erase(first, last); }

    size_type erase(const value_type& value)
    { return m_map.erase(value); }

    /** Search */
    iterator find(const key_type &key) 
    { return m_map.find(key); }

    const_iterator find(const key_type &key) const
    { return m_map.find(key); }

    size_type count(const key_type &key) const
    { return m_map.count(key); }

    std::pair<iterator, iterator> equal_range(const key_type &key)
    { return m_map.equal_range(key); }

    std::pair<const_iterator, const_iterator> equal_range(const key_type &key) const
    { return m_map.equal_range(key); }

    /** Copy/assign */
    void swap(ShHashMap &other) 
    { m_map.swap(other.m_map); }

    data_type& operator[](const key_type &key) 
    { return m_map[key]; }

  private:
    map_type m_map;
};

/** Some useful stuff */
template<class Key1, class Key2, class Hash1=SH_STD_HASH(Key1), class Hash2=SH_STD_HASH(Key2)>
struct ShPairHash {
  typedef std::pair<Key1, Key2> key_type;
  Hash1 m_hash1;
  Hash2 m_hash2;
  size_t operator()(const key_type &key) const 
  { return m_hash1(key.first) | ~m_hash2(key.second); } 
};

template<class Key1, class Key2, class Data, class Hash1=SH_STD_HASH(Key1), class Hash2=SH_STD_HASH(Key2)>
class ShPairHashMap: public ShHashMap<std::pair<Key1, Key2>, Data, ShPairHash<Key1, Key2, Hash1, Hash2> > {
  typedef std::pair<Key1, Key2> key_type; 
  public:
    Data& operator()(const Key1& key1, const Key2& key2) 
    {
      return operator[](key_type(key1, key2));
    }
};

template<class T1, class T2, class T3>
struct ShTriple
{
  T1 v1;
  T2 v2;
  T3 v3;

  ShTriple(const T1 &v1, const T2 &v2, const T3 &v3)
    : v1(v1), v2(v2), v3(v3) {}

  bool operator==(const ShTriple &b) const
  { return v1 == b.v1 && v2 == b.v2 && v3 == b.v3; } 

  bool operator<(const ShTriple &b) const
  { 
    return v1 < b.v1 || 
           (v1 == b.v1 && 
           (v2 < b.v2 ||
           (v2 == b.v2 && 
           (v3 < b.v3))));  
  } 
};

template<class Key1, class Key2, class Key3, 
  class Hash1=SH_STD_HASH(Key1), class Hash2=SH_STD_HASH(Key2), 
  class Hash3=SH_STD_HASH(Key3)> 
struct ShTripleHash {
  typedef ShTriple<Key1, Key2, Key3> key_type;
  Hash1 hash1;
  Hash2 hash2;
  Hash3 hash3;
  size_t operator()(const key_type &key) const
  { return hash1(key.v1) + hash2(key.v2) + hash3(key.v3); } 
};

template<class Key1, class Key2, class Key3, class Data, 
  class Hash1=SH_STD_HASH(Key1), class Hash2=SH_STD_HASH(Key2), 
  class Hash3=SH_STD_HASH(Key3)> 
class ShTripleHashMap: public ShHashMap<ShTriple<Key1, Key2, Key3>, Data,
    ShTripleHash<Key1, Key2, Key3, Hash1, Hash2, Hash3> > {
  public:
    typedef ShTriple<Key1, Key2, Key3> key_type;
    Data& operator()(const Key1 &key1, const Key2 &key2, const Key3 &key3)
    {
      return operator[](key_type(key1, key2, key3));
    }
};

template<class T1, class T2, class T3, class T4>
struct ShPairPair
{
  T1 v1;
  T2 v2;
  T3 v3;
  T4 v4;

  ShPairPair(const T1 &v1, const T2 &v2, const T3 &v3, const T4 &v4)
    : v1(v1), v2(v2), v3(v3), v4(v4) {}

  bool operator==(const ShPairPair &b) const
  { return v1 == b.v1 && v2 == b.v2 && v3 == b.v3 && v4 == b.v4; }

  bool operator<(const ShPairPair &b) const
  { 
    return v1 < b.v1 || 
           (v1 == b.v1 && 
           (v2 < b.v2 ||
           (v2 == b.v2 && 
           (v3 < b.v3 ||
           (v3 == b.v3 && 
           (v4 < b.v4))))));  
  } 
};

template<class Key1, class Key2, class Key3, class Key4, 
  class Hash1=SH_STD_HASH(Key1), class Hash2=SH_STD_HASH(Key2), 
  class Hash3=SH_STD_HASH(Key3), class Hash4=SH_STD_HASH(Key4)>
struct ShPairPairHash {
  typedef ShPairPair<Key1, Key2, Key3, Key4> key_type;
  Hash1 hash1;
  Hash2 hash2;
  Hash3 hash3;
  Hash4 hash4;
  size_t operator()(const key_type &key) const
  { return hash1(key.v1) + hash2(key.v2) + hash3(key.v3) + hash4(key.v4); } 
};

template<class Key1, class Key2, class Key3, class Key4, class Data, 
  class Hash1=SH_STD_HASH(Key1), class Hash2=SH_STD_HASH(Key2), 
  class Hash3=SH_STD_HASH(Key3), class Hash4=SH_STD_HASH(Key4)>
class ShPairPairHashMap: public ShHashMap<ShPairPair<Key1, Key2, Key3, Key4>, Data,
    ShPairPairHash<Key1, Key2, Key3, Key4, Hash1, Hash2, Hash3, Hash4> > {
  public:
    typedef ShPairPair<Key1, Key2, Key3, Key4> key_type;
    Data& operator()(const Key1 &key1, const Key2 &key2, const Key3 &key3, const Key4 &key4) 
    {
      return operator[](key_type(key1, key2, key3, key4));
    }
};

#undef SH_STD_HASH

}

#endif
