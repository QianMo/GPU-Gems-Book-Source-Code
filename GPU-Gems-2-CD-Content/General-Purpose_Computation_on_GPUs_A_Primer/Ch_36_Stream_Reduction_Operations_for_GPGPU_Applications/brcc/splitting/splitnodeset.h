// splitnodeset.h
#ifndef __SPLITNODESET_H__
#define __SPLITNODESET_H__

#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif

#include <vector>

class SplitNode;

class SplitNodeSet
{
public:
  SplitNodeSet(); // default construct to empty
  SplitNodeSet( size_t inCount ); // default construct to empty of a certain size...

  void swap( SplitNodeSet& inSet ) {
    values.swap( inSet.values );
  }

  void clear() {
    values.clear();
  }

  size_t capacity() const {
    return values.size();
  }

  bool intersects( const SplitNodeSet& inSet ) const;

  void operator=( const SplitNodeSet& inSet ) {
    values = inSet.values;
  }

  void operator|=( const SplitNodeSet& inSet );
  void operator|=( SplitNode* inNode );
  void operator&=( const SplitNodeSet& inSet );
  void operator/=( const SplitNodeSet& inSet );

  void setUnion( const SplitNodeSet& inA, const SplitNodeSet& inB );
  void setIntersect( const SplitNodeSet& inA, const SplitNodeSet& inB );
  void setDifference( const SplitNodeSet& inA, const SplitNodeSet& inB );

  bool contains( SplitNode* inNode ) const;
  bool contains( size_t inIndex ) const;

  void insert( SplitNode* inNode );
  void insert( size_t inIndex );

  void remove( SplitNode* inNode );
  void remove( size_t inIndex );

  class iterator
  {
  public:
    iterator( const SplitNodeSet& inSet, size_t inIndex );

    iterator operator++(); // preinc?
    iterator operator++(int); // postinc?

    size_t operator*();

    bool operator==( const iterator& other ) {
      return index == other.index;
    }

    bool operator!=( const iterator& other ) {
      return index != other.index;
    }

  private:
    void advance();
    void validate();

    size_t index;
    const SplitNodeSet& set;
  };

  iterator begin() const;
  iterator end() const;

private:
  typedef std::vector<bool> BitVector;

  BitVector values;
};

#endif
