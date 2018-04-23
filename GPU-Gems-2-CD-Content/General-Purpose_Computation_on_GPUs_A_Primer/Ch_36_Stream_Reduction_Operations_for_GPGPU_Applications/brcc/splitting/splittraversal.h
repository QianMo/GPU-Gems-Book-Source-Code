// splittraversal.h
#ifndef __SPLITTRAVERSAL_H__
#define __SPLITTRAVERSAL_H__
#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif

#include <iostream>
#include <vector>
#include <set>

class SplitNode;

class SplitNodeTraversal
{
public:
  void operator()( SplitNode* inNode ) {
    traverse( inNode );
  }

  void operator()( const std::vector<SplitNode*>& inNodes ) {
    for( std::vector<SplitNode*>::const_iterator i = inNodes.begin(); i != inNodes.end(); ++i )
      traverse( *i );
  }

  void operator()( const std::set<SplitNode*>& inNodes ) {
    for( std::set<SplitNode*>::const_iterator i = inNodes.begin(); i != inNodes.end(); ++i )
      traverse( *i );
  }

  virtual void traverse( SplitNode* inNode ) = 0;

  void traverseGraphChildren( SplitNode* inNode );
};

class SplitArgumentCounter
{
public:
  SplitArgumentCounter()
    : samplerCount(0), constantCount(0), texcoordCount(0), outputCount(0) {}

  int samplerCount;
  int constantCount;
  int texcoordCount;
  int outputCount;
};

class SplitArgumentTraversal :
  public SplitNodeTraversal
{
public:
  SplitArgumentTraversal( std::ostream& inStream, SplitNode* inOutputPosition )
    : stream(inStream), outputPosition(inOutputPosition), hasOutput(false) {}

  void traverse( SplitNode* inNode );

private:
  SplitArgumentCounter argumentCounter;
  std::ostream& stream;
  SplitNode* outputPosition;
  bool hasOutput;
};

class SplitStatementTraversal :
  public SplitNodeTraversal
{
public:
  SplitStatementTraversal( std::ostream& inStream, SplitNode* inOutputPosition )
    : stream(inStream), outputPosition(inOutputPosition) {}
  void traverse( SplitNode* inNode );

private:
  std::ostream& stream;
  SplitNode* outputPosition;
};

class SplitAnnotationTraversal :
  public SplitNodeTraversal
{
public:
  SplitAnnotationTraversal( std::ostream& inStream, SplitNode* inOutputPosition )
    : stream(inStream), outputPosition(inOutputPosition) {}
  void traverse( SplitNode* inNode );

private:
  std::ostream& stream;
  SplitNode* outputPosition;
};

#endif
