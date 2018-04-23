 // splitnodeset.cpp
#include "splitnodeset.h"

#include "splitnode.h"

#include <assert.h>

static size_t min( size_t a, size_t b ) {
  return a < b ? a : b;
}

static size_t max( size_t a, size_t b ) {
  return a > b ? a : b;
}

SplitNodeSet::SplitNodeSet()
{
}

SplitNodeSet::SplitNodeSet( size_t inCount )
{
  values.resize( inCount, false );
}

bool SplitNodeSet::intersects( const SplitNodeSet& inSet ) const
{
  size_t range = min( capacity(), inSet.capacity() );
  for( size_t i = 0; i < range; i++ )
  {
    if( values[i] && inSet.values[i] )
      return true;
  }
  return false;
}

void SplitNodeSet::operator|=( const SplitNodeSet& inSet )
{
  size_t range = max( capacity(), inSet.capacity() );
  values.resize( range, false );

  for( size_t i = 0; i < inSet.capacity(); i++ )
    values[i] = values[i] || inSet.values[i];
}

void SplitNodeSet::operator|=( SplitNode* inNode ) {
  insert( inNode );
}

void SplitNodeSet::operator&=( const SplitNodeSet& inSet )
{
  size_t range = max( capacity(), inSet.capacity() );
  values.resize( range, false );

  for( size_t i = 0; i < inSet.capacity(); i++ )
    values[i] = values[i] && inSet.values[i];
}

void SplitNodeSet::operator/=( const SplitNodeSet& inSet )
{
  size_t range = max( capacity(), inSet.capacity() );
  values.resize( range, false );

  for( size_t i = 0; i < inSet.capacity(); i++ )
    values[i] = values[i] && !inSet.values[i];
}

void SplitNodeSet::setUnion( const SplitNodeSet& inA, const SplitNodeSet& inB )
{
  *this = inA;
  *this |= inB;
}

void SplitNodeSet::setIntersect( const SplitNodeSet& inA, const SplitNodeSet& inB )
{
  *this = inA;
  *this &= inB;
}

void SplitNodeSet::setDifference( const SplitNodeSet& inA, const SplitNodeSet& inB )
{
  *this = inA;
  *this /= inB;
}

bool SplitNodeSet::contains( SplitNode* inNode ) const {
  return contains( inNode->getDagOrderIndex() );
}

bool SplitNodeSet::contains( size_t inIndex ) const
{
  if( inIndex >= capacity() ) return false;
  return values[ inIndex ];
}

void SplitNodeSet::insert( SplitNode* inNode ) {
  insert( inNode->getDagOrderIndex() );
}

void SplitNodeSet::insert( size_t inIndex )
{
  values.resize( max( inIndex+1, capacity() ), false );
  values[ inIndex ] = true;
}

void SplitNodeSet::remove( SplitNode* inNode ) {
  remove( inNode->getDagOrderIndex() );
}

void SplitNodeSet::remove( size_t inIndex )
{
  values.resize( max( inIndex+1, capacity() ), false );
  values[ inIndex ] = false;
}

SplitNodeSet::iterator::iterator( const SplitNodeSet& inSet, size_t inIndex )
   : index(inIndex), set(inSet)
{
  validate();
}

SplitNodeSet::iterator SplitNodeSet::iterator::operator++() {
  advance();
  return *this;
}
SplitNodeSet::iterator SplitNodeSet::iterator::operator++(int) {
  iterator result = *this;
  advance();
  return result;
}

size_t SplitNodeSet::iterator::operator*() {
  return index;
}

void SplitNodeSet::iterator::advance()
{
  index++;
  validate();
}

void SplitNodeSet::iterator::validate()
{
  size_t capacity = set.capacity();
  while( index != capacity )
  {
    if( set.contains( index ) )
      return;
    index++;
  }
}

SplitNodeSet::iterator SplitNodeSet::begin() const {
  return iterator( *this, 0 );
}

SplitNodeSet::iterator SplitNodeSet::end() const {
  return iterator( *this, capacity() );
}
