// splitdominator.cpp

#include "splitting.h"

#include <fstream>

void SplitTree::dominatorDFS( SplitNode* inNode, SplitNode* inParent, size_t& ioID )
{
  if( inParent )
    inNode->_graphParents.push_back( inParent );

  if( inNode->_spanningNodeID > 0 )
    return;

  inNode->_spanningParent = inParent;

  inNode->_spanningNodeID = ioID++;
  inNode->_spanningSemidominatorID = inNode->_spanningNodeID;
  _rdsNodeList.push_back( inNode );

  size_t childCount = inNode->getGraphChildCount();
  for( size_t i = 0; i < childCount; i++ )
  {
    SplitNode* child = inNode->getIndexedGraphChild(i);
    dominatorDFS( child, inNode, ioID );
  }

  inNode->_dagOrderIndex = _dagOrderNodeList.size();
  _dagOrderNodeList.push_back( inNode );
}

void SplitTree::buildDominatorTree()
{
//  std::cerr << "buildDominatorTree" << std::endl;

//  std::cerr << "step 1" << std::endl;

  // build the immediate dominator info...
  // step 1 - dfs
  size_t id = 1;
  dominatorDFS( _pseudoRoot, NULL, id );

//  std::cerr << "steps 2,3" << std::endl;
  // step 2 and 3, build semidominators
  {for( size_t i = _rdsNodeList.size(); i > 0; i-- )
  {
    SplitNode* w = _rdsNodeList[i-1];

//    std::cerr << "operating on ";
//    w->dump( std::cerr );
//    std::cerr << std::endl;

//    std::cerr << "step 2" << std::endl;
    // step 2
    for( std::vector<SplitNode*>::iterator j = w->_graphParents.begin(); j != w->_graphParents.end(); ++j )
    {
      SplitNode* v = *j;
      SplitNode* u = v->eval();

//      std::cerr << "u is ";
//      u->dump( std::cerr );
//      std::cerr << std::endl;

      if( u->_spanningSemidominatorID < w->_spanningSemidominatorID )
        w->_spanningSemidominatorID = u->_spanningSemidominatorID;
    }
    _rdsNodeList[ w->_spanningSemidominatorID-1 ]->_spanningBucket.insert( w );

    SplitNode* parent = w->_spanningParent;
    if( parent )
    {
      parent->link( w );

//      std::cerr << "step 3" << std::endl;
      // step 3
      for( NodeSet::iterator k = parent->_spanningBucket.begin(); k != parent->_spanningBucket.end(); ++k )
      {
        SplitNode* v = *k;
        SplitNode* u = v->eval();
        v->_pdtDominator = (u->_spanningSemidominatorID < v->_spanningSemidominatorID) ? u : parent;
      }
      parent->_spanningBucket.clear();
    }
  }}

//  std::cerr << "step 4" << std::endl;
  // step 4
  {for( size_t i = 0; i < _rdsNodeList.size(); i++ )
  {
    SplitNode* w = _rdsNodeList[i];
    if( !w->_spanningParent )
      w->_pdtDominator = 0;
    else if( w->_pdtDominator != _rdsNodeList[w->_spanningSemidominatorID-1])
      w->_pdtDominator = w->_pdtDominator->_pdtDominator;
  }}

//  std::cerr << "step PDT" << std::endl;
  // we have dominator info... 
  // now we need to prune it to the Partial Dominator Tree,
  // and the list of MR nodes...
  {for( size_t i = 0; i < _dagOrderNodeList.size(); i++ )
  {
    SplitNode* n = _dagOrderNodeList[i];

    // we add it to the partial tree if it is multiply-referenced
    // or has some multiply-referenced descendants (which would
    // have already been added).
    // TIM: we ignore nodes that are taken to be "trivial"
    // (that is, those that should never be saved)
    if( n->_graphParents.size() > 1 /*&& !n->isTrivial()*/ )
    {
      n->_isPDTNode = true;
      _multiplyReferencedNodes.push_back( n );
      if( n->_pdtDominator )
        n->_pdtDominator->_pdtChildren.push_back( n );
    }
    else if( n->_pdtChildren.size() != 0 )
    {
      n->_isPDTNode = true;
      if( n->_pdtDominator )
        n->_pdtDominator->_pdtChildren.push_back( n );
    }
  }}

//  dumpDominatorTree();
}

void SplitTree::dumpDominatorTree()
{
  std::ofstream dumpFile("dominator_dump.txt");
  dumpDominatorTree( dumpFile, _pseudoRoot );
}

void SplitTree::dumpDominatorTree( std::ostream& inStream, SplitNode* inNode, int inLevel )
{
  for( int l = 0; l < inLevel; l++ )
    inStream << "   ";
  
  inNode->dump( inStream );
  inStream << std::endl;

  size_t childCount = inNode->_pdtChildren.size();
  for( size_t c = 0; c < childCount; c++ )
  {
    SplitNode* child = inNode->_pdtChildren[c];
    assert( child->_pdtDominator == inNode );
    dumpDominatorTree( inStream, child, inLevel+1 );
  }
}
