// splittree.cpp
#include "splittree.h"

#include "splitnode.h"
#include "splitbuilder.h"
#include "splitcompiler.h"
#include "splitsubset.h"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <algorithm>

// uncomment this to switch to slow-as-hell sort of exhaustive search
//#define SPLIT_SEARCH_EXHAUSTIVE

// uncomment this to turn on the greedy merging that improves RDS
#define SPLIT_SEARCH_MERGE

// uncomment this line for RDS + an after-the-fact merge
//#define SPLIT_SEARCH_MERGE_AFTER

// uncomment this line to allow RDS*merge to remove useless nodes
// during the incremental search, not just at the end
#define SPLIT_REMOVE_USELESS_NODES

static unsigned long timeSplitCounter;
static unsigned long timePrintingCounter;
static unsigned long timeCompilingCounter;

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>

#pragma comment(lib,"winmm")

static unsigned long getTime()
{
  DWORD result = timeGetTime();
  return (unsigned long)result;
}
#else

static unsigned long getTime()
{
  return 0;
}

#endif

SplitTree::SplitTree( FunctionDef* inFunctionDef, 
                      const SplitCompiler& inCompiler )
   : _pseudoRoot(NULL), _resultValue(NULL), _compiler(inCompiler) 
{
  _functionName = inFunctionDef->FunctionName()->name;
  std::cout << "$$$$$ creating a split tree for " << _functionName << std::endl;

  build( inFunctionDef );
}

SplitTree::SplitTree( FunctionDef* inFunctionDef, 
                      const SplitCompiler& inCompiler, 
                      const std::vector<SplitNode*>& inArguments )
   : _pseudoRoot(NULL), _resultValue(NULL), _compiler(inCompiler)
{
  build( inFunctionDef, inArguments );
}

SplitTree::~SplitTree()
{
}

//static std::ofstream dumpFile;

void SplitTree::printTechnique( const SplitTechniqueDesc& inTechniqueDesc, std::ostream& inStream )
{
  // TIM: sanity check:
  for( size_t i = 0; i < _dagOrderNodeList.size(); i++ )
    assert( _dagOrderNodeList[i]->getDagOrderIndex() == i );

//  dumpFile.open( "dump.txt" );

//  preRdsMagic();

  timeSplitCounter = 0;
  timePrintingCounter = 0;
  timeCompilingCounter = 0;

  unsigned long splitStart = getTime();

  {for( NodeList::iterator i = _outputList.begin(); i != _outputList.end(); ++i ) {
    (*i)->_isFinalOutput = true;
  }}

#ifdef SPLIT_SEARCH_EXHAUSTIVE
  exhaustiveSearch();
#else
  rdsSearch();
#endif

  unsigned long splitStop = getTime();
  timeSplitCounter += splitStop - splitStart;

  // TIM: we need to split somewhere
  for( size_t i = 0; i < _outputList.size(); i++ )
  {
    assert( _outputList[i]->_splitHere );
  }

  // assign "registers" to all split nodes
  {for( NodeList::iterator i = _dagOrderNodeList.begin(); i != _dagOrderNodeList.end(); ++i ) {
    (*i)->setTemporaryID( 0 );
  }}
  int outputID = 0;
  {for( NodeList::iterator i = _outputList.begin(); i != _outputList.end(); ++i ) {
    (*i)->setTemporaryID( _outputArgumentIndices[outputID++] );
  }}
  int temporaryID = 1;
  {for( NodeList::iterator i = _dagOrderNodeList.begin(); i != _dagOrderNodeList.end(); ++i )
  {
    if( (*i)->isMarkedAsSplit() && ((*i)->getTemporaryID() == 0) )
      (*i)->setTemporaryID( -(temporaryID++) );
  }}
  int temporaryCount = temporaryID-1;

  // now we go through the passes and print them?
  inStream << "\t.technique( gpu_technique_desc()" << std::endl;
  if( temporaryCount )
  {
    inStream << "\t\t.temporaries(" << temporaryCount << ")" << std::endl;
  }

  unmark( SplitNode::kMarkBit_Printed );

  std::cout << "time split " << timeSplitCounter << std::endl;
  std::cout << "time compile " << timeCompilingCounter << std::endl;
  std::cout << "time print " << timePrintingCounter << std::endl;
  std::ofstream cdump("configuration.dump");
  dumpPassConfiguration(cdump  );

  for( PassSet::iterator p = _passes.begin(); p != _passes.end(); ++p )
    rdsPrintPass( *p, inStream );

  inStream << "\t)";

//  dumpFile.close();
}

void SplitTree::dumpPassConfiguration( std::ostream& inStream )
{
  inStream << "Split configuration generated for " << _functionName << std::endl;
  
#if defined(SPLIT_SEARCH_EXHAUSTIVE)
  inStream << "Exhaustive search" << std::endl;
#elif defined(SPLIT_SEARCH_MERGE)
  inStream << "RDS with integrated merge" << std::endl;
#else
  inStream << "Standard RDS" << std::endl;
#endif

  size_t nodeCount = _dagOrderNodeList.size();
  inStream << "totalNodes = " << nodeCount << std::endl;

  size_t mrNodeCount = 0;
  for( NodeList::iterator n = _dagOrderNodeList.begin(); n != _dagOrderNodeList.end(); ++n )
  {
    SplitNode* node = *n;
    if( node->getGraphParentCount() > 1 )
      mrNodeCount++;
  }

  inStream << "multiplyReferencedNodes = " << mrNodeCount << std::endl;

  inStream << "passCount = " << _passes.size() << std::endl;

  int totalCost = 0;

  for( PassSet::iterator p = _passes.begin(); p != _passes.end(); ++p )
  {
    inStream << "<pass>" << std::endl;
/*
    for( SplitNodeSet::iterator n = (*p)->allOutputs.begin(); n != (*p)->allOutputs.end(); ++n )
    {
      SplitNode* node = _dagOrderNodeList[ *n ];
      inStream << "<split id=" << (*n) << ">" << std::endl;
      for( SplitNodeSet::iterator i = node->_parentSplits.begin(); i != node->_parentSplits.end(); ++i )
      {
        inStream << "<parent id=" << (*i) << ">" << std::endl;
      }
      inStream << "</split>" << std::endl;
    }
*/

    SplitShaderHeuristics heuristics = (*p)->heuristics;

    inStream << "cost = " << heuristics.cost << std::endl;

    totalCost += heuristics.cost;

    inStream << "textureInstCount = " << heuristics.textureInstructionCount << std::endl;
    inStream << "arithInstCount = " << heuristics.arithmeticInstructionCount << std::endl;
    inStream << "samplerCount = " << heuristics.samplerRegisterCount << std::endl;
    inStream << "interpCount = " << heuristics.interpolantRegisterCount << std::endl;
    inStream << "constCount = " << heuristics.constantRegisterCount << std::endl;
    inStream << "tempCount = " << heuristics.temporaryRegisterCount << std::endl;
    inStream << "outputCount = " << heuristics.outputRegisterCount << std::endl;

    inStream << "</pass>" << std::endl;
  }

  inStream << "totalCost = " << totalCost << std::endl;
}

void SplitTree::exhaustiveSearch()
{
  // first label the outputs
  for( NodeList::iterator i = _outputList.begin(); i != _outputList.end(); ++i )
  {
    (*i)->_splitHere = true;
  }

  // now collect all the unlabeled, nontrivial nodes:
  NodeList nodesToConsider;
  for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
  {
    if( (*j)->isMarkedAsSplit() ) continue;
    if( !(*j)->canBeSaved() ) continue;
    if( *j == _pseudoRoot ) continue;
    nodesToConsider.push_back( *j );
  }

  size_t nodeCount = nodesToConsider.size();

  int bestScore = INT_MAX;

  for( size_t subsetSize = 0; subsetSize < nodeCount; subsetSize++ )
  {
    std::cout << "considering subsets of size " << subsetSize << " out of " << nodeCount << std::endl;

    int bestScoreForSubsetSize = INT_MAX;
    exhaustiveSubsetSearch( subsetSize, nodesToConsider, bestScoreForSubsetSize );

    std::cout << "best split has score: " << bestScoreForSubsetSize << std::endl;

    if( bestScoreForSubsetSize != INT_MAX )
    {
      if( (bestScore != INT_MAX) && (bestScoreForSubsetSize > bestScore) )
      {
        // there probably isn't a better partition, lets use this :)
        break;
      }

      if( bestScoreForSubsetSize < bestScore )
        bestScore = bestScoreForSubsetSize;
    }
  }

  std::cout << "best overall score found before giving up: " << bestScore << std::endl;
}

void SplitTree::exhaustiveSubsetSearch( size_t inSubsetSize, const NodeList& inNodes, int& outBestScore )
{
  size_t subsetSize = inSubsetSize;
  size_t nodeCount = inNodes.size();
  SplitSubsetGenerator2 generator( subsetSize, nodeCount );

  outBestScore = INT_MAX;

  while( generator.hasMore() )
  {
    generator.getNext();
    size_t i;
    for( i = 0; i < subsetSize; i++ )
      inNodes[ generator.getIndexedValue(i) ]->_splitHere = true;

    int score;
    if( exhaustiveSplitIsValid( score ) )
    {
      // TIM: find the optimal merge...

      rdsMergePasses();
      score = getPartitionCost();

      if( score < outBestScore )
        outBestScore = score;
    }

    for( i = 0; i < subsetSize; i++ )
      inNodes[ generator.getIndexedValue(i) ]->_splitHere = false;
  }
}

bool SplitTree::exhaustiveSplitIsValid( int& outScore )
{
  int totalCost = 0;
  for( NodeList::iterator i = _dagOrderNodeList.begin(); i != _dagOrderNodeList.end(); ++i )
  {
    SplitNode* node = *i;
    if( !node->isMarkedAsSplit() ) continue;

    SplitShaderHeuristics heuristics;
    if( !rdsCompile( node, heuristics ) )
      return false;

    node->setHeuristics( heuristics );

    totalCost += heuristics.cost;
  }

//  std::cout << "exhaustive search found valid split with cost: " << totalCost << std::endl;

  outScore = totalCost;

  return true;
}

struct SplitValidMergeInfo
{
  SplitNode* a;
  SplitNode* b;
  int score;
};

static bool ValidMergeOrder( const SplitValidMergeInfo& a, const SplitValidMergeInfo& b )
{
  return (a.score > b.score);
}

void SplitTree::rdsMergePasses( bool inLastTime )
{
  for( PassSet::iterator k = _passes.begin(); k != _passes.end(); ++k )
    delete *k;
  _passes.clear();

  // first collect the set of passes...
  for( NodeList::iterator i = _dagOrderNodeList.begin(); i != _dagOrderNodeList.end(); ++i )
  {
    SplitNode* node = *i;

    // TIM: it is bad to do no initialization... :(
    node->_parentSplits.clear();

    if( node->isMarkedAsSplit() )
      _passes.insert( rdsCreatePass( node ) );
  }

  for( PassSet::iterator j = _passes.begin(); j != _passes.end(); ++j )
  {
    accumulateChildSplits( (*j)->singletonNode );
    accumulateParentSplits( (*j)->singletonNode );
  }

  for( PassSet::iterator j = _passes.begin(); j != _passes.end(); ++j )
  {
    rdsAccumulatePassAncestors( *j );
    rdsAccumulatePassDescendents( *j );
  }

#if defined(SPLIT_REMOVE_USELESS_NODES)
  bool allowRemoval = true;
#else
  bool allowRemoval = inLastTime;
#endif

#if defined(SPLIT_SEARCH_MERGE) || defined(SPLIT_SEARCH_MERGE_AFTER)

#if defined(SPLIT_SEARCH_MERGE_AFTER)
  if( inLastTime ) {
#endif

    std::vector<SplitValidMergeInfo> validMerges;

    {for( PassSet::iterator i = _passes.begin(); i != _passes.end(); )
    {
      SplitPassInfo* a = *i++;

      for( PassSet::iterator j = i; j != _passes.end(); ++j )
      {
        SplitPassInfo* b = *j;

        int score = rdsCanMergePasses( a, b, allowRemoval );
        if( score < 0 ) continue;

        std::cout << "*";

        SplitValidMergeInfo info;

        // TIM: HACK: assume only a single output
        info.a = a->singletonNode;
        info.b = b->singletonNode;
        info.score = score;

        validMerges.push_back( info );
      }
    }}

    // sort them so that we do better-scoring merges first:
    std::sort( validMerges.begin(), validMerges.end(), ValidMergeOrder );

    // now process them in order and see what happens
    {for( std::vector<SplitValidMergeInfo>::iterator i = validMerges.begin(); i != validMerges.end(); ++i )
    {
      SplitValidMergeInfo& merge = *i;

      SplitPassInfo* passA = merge.a->_assignedPass;
      if( passA == NULL) continue;

      SplitPassInfo* passB = merge.b->_assignedPass;
      if( passB == NULL ) continue;

      // they were already merged...
      if( passA == passB ) continue;

      int score = rdsCanMergePasses( passA, passB, allowRemoval );
      if( score < 0 ) continue;

      rdsMergePasses( passA, passB, allowRemoval );
    }}

#if defined(SPLIT_SEARCH_MERGE_AFTER)
  }
#endif

#endif
}

SplitPassInfo* SplitTree::rdsCreatePass( SplitNode* inNode )
{
//  dumpFile << "CREATE MERGE PASS " << inNode->getTemporaryID() << std::endl << "% ";
//  inNode->dump( dumpFile );
//  dumpFile << std::endl;

  SplitPassInfo* result = new SplitPassInfo();

  result->singletonNode = inNode;
  result->allOutputs.insert( inNode );
  result->usefulOutputs.insert( inNode );

  SplitShaderHeuristics heuristics = inNode->getHeuristics();

  /* TIM: we assume nodes already have the proper heuristic installed
  SplitShaderHeuristics heuristics;
  rdsCompile( inNode, heuristics );
  assert( heuristics.valid );
  */

  result->cost = heuristics.cost;
  result->heuristics = heuristics;

//  dumpFile << "CREATED " << (void*)result << " with cost " << result->cost << std::endl;

  inNode->_assignedPass = result;

  return result;
}

void SplitTree::accumulateChildSplits( SplitNode* inSplit )
{
  unmark( SplitNode::kMarkBit_Descendent );
  inSplit->_childSplits.clear();

  size_t childCount = inSplit->getGraphChildCount();
  for( size_t i = 0; i < childCount; i++ )
  accumulateChildSplitsRec( inSplit, inSplit->getIndexedGraphChild(i) );
}

void SplitTree::accumulateChildSplitsRec( SplitNode* inSplit, SplitNode* inDescenent )
{
  if( inDescenent->isMarked( SplitNode::kMarkBit_Descendent ) ) return;
  inDescenent->mark( SplitNode::kMarkBit_Descendent );

  if( inDescenent->isMarkedAsSplit() )
  {
    inSplit->_childSplits.insert( inDescenent );
  }
  else
  {
    size_t childCount = inDescenent->getGraphChildCount();
    for( size_t i = 0; i < childCount; i++ )
      accumulateChildSplitsRec( inSplit, inDescenent->getIndexedGraphChild(i) );
  }
}

void SplitTree::accumulateParentSplits( SplitNode* inSplit )
{
  unmark( SplitNode::kMarkBit_Ancestor );
  inSplit->_parentSplits.clear();

  size_t parentCount = inSplit->getGraphParentCount();
  for( size_t i = 0; i < parentCount; i++ )
    accumulateParentSplitsRec( inSplit, inSplit->getIndexedGraphParent(i) );
}

void SplitTree::accumulateParentSplitsRec( SplitNode* inSplit, SplitNode* inAncestor )
{
  if( inAncestor->isMarked( SplitNode::kMarkBit_Ancestor ) ) return;
  inAncestor->mark( SplitNode::kMarkBit_Ancestor );

  if( inAncestor->isMarkedAsSplit() )
  {
    inSplit->_parentSplits.insert( inAncestor );
  }
  else
  {
    size_t parentCount = inAncestor->getGraphParentCount();
    for( size_t i = 0; i < parentCount; i++ )
      accumulateParentSplitsRec( inSplit, inAncestor->getIndexedGraphParent(i) );
  }
}

void SplitTree::rdsAccumulatePassAncestors( SplitPassInfo* ioPass )
{
  if( ioPass->ancestorVisited ) return;
  ioPass->ancestorVisited = true;

  for( SplitNodeSet::iterator i = ioPass->allOutputs.begin(); i != ioPass->allOutputs.end(); ++i )
  {
    SplitNode* node = _dagOrderNodeList[ *i ];

    for( SplitNodeSet::iterator j = node->_parentSplits.begin(); j != node->_parentSplits.end(); ++j )
    {
      SplitNode* parentSplit = _dagOrderNodeList[ *j ];
      SplitPassInfo* parentPass = parentSplit->_assignedPass;

      rdsAccumulatePassAncestors( parentPass );

      ioPass->ancestors |= parentPass->ancestors;
      ioPass->ancestors |= parentSplit;
    }
  }
}

void SplitTree::rdsAccumulatePassDescendents( SplitPassInfo* ioPass )
{
  if( ioPass->descendentVisited ) return;
  ioPass->descendentVisited = true;

  for( SplitNodeSet::iterator i = ioPass->allOutputs.begin(); i != ioPass->allOutputs.end(); ++i )
  {
    SplitNode* node = _dagOrderNodeList[ *i ];

    for( SplitNodeSet::iterator j = node->_childSplits.begin(); j != node->_childSplits.end(); ++j )
    {
      SplitNode* childSplit = _dagOrderNodeList[ *j ];
      SplitPassInfo* childPass = childSplit->_assignedPass;

      rdsAccumulatePassDescendents( childPass );

      ioPass->descendents |= childPass->descendents;
      ioPass->descendents |= childSplit;
    }
  }
}

int SplitTree::rdsCanMergePasses( SplitPassInfo* inA, SplitPassInfo* inB, bool inAllowRemoval )
{
  // check basic validity

  // if there is an ancestor of one pass
  // that is also a descendent of the other
  // then we can't merge
  if( inA->descendents.intersects( inB->ancestors ) )
    return -1;

  if( inB->descendents.intersects( inA->ancestors ) )
    return -1;

  SplitNodeSet allOutputs;
  allOutputs.setUnion( inA->allOutputs, inB->allOutputs );

  SplitNodeSet usefulOutputs = allOutputs;

  SplitNodeSet uselessNodes;
  for( SplitNodeSet::iterator n = allOutputs.begin(); n != allOutputs.end(); ++n )
  {
    SplitNode* node = _dagOrderNodeList[ *n ];

    // can't merge actual outputs out of existence... that would be bad
    if( node->_isFinalOutput ) continue;

    bool okay = false;
    for( SplitNodeSet::iterator p = node->_parentSplits.begin(); p != node->_parentSplits.end(); ++p )
    {
      if( !allOutputs.contains( *p ) )
        okay = true; // parent wasn't there, it's worth saving...
    }
    if( !okay )
    {
      if( !inAllowRemoval )
        return -1;

      usefulOutputs.remove( node );
      uselessNodes.insert( node );
    }
  }

  // now we need to generate a shader for all of these outputs...

  // HACK: we have to "un-split" the useless nodes
  // so that we can compile the result as if they were un-split...
  for( SplitNodeSet::iterator u = uselessNodes.begin(); u != uselessNodes.end(); ++u )
  {
    SplitNode* node = _dagOrderNodeList[ *u ];
    node->_splitHere = false;
  }

  SplitShaderHeuristics heuristics;
  bool valid = rdsCompile( usefulOutputs, heuristics );

  // HACK: re-split the useless nodes, or else!
  for( SplitNodeSet::iterator u = uselessNodes.begin(); u != uselessNodes.end(); ++u )
  {
    SplitNode* node = _dagOrderNodeList[ *u ];
    node->_splitHere = true;
  }

  if( !valid )
    return -1;

  int score = (inA->cost + inB->cost) - heuristics.cost;
  return score;
}

void SplitTree::rdsMergePasses( SplitPassInfo* inA, SplitPassInfo* inB, bool inAllowRemoval )
{
  // merge two passes assuming validity check has passed...

  std::cout << "!";

  SplitNodeSet allOutputs;
  allOutputs.setUnion( inA->allOutputs, inB->allOutputs );

  SplitNodeSet usefulOutputs = allOutputs;

  SplitNodeSet uselessNodes;
  for( SplitNodeSet::iterator n = allOutputs.begin(); n != allOutputs.end(); ++n )
  {
    SplitNode* node = _dagOrderNodeList[ *n ];

    // can't merge actual outputs out of existence... that would be bad
    if( node->_isFinalOutput ) continue;

    bool okay = false;
    for( SplitNodeSet::iterator p = node->_parentSplits.begin(); p != node->_parentSplits.end(); ++p )
    {
      if( !allOutputs.contains( *p ) )
        okay = true; // parent wasn't there, it's worth saving...
    }
    if( !okay )
    {
      assert( inAllowRemoval );

      std::cout << "^";
//      std::cout << "making useless: " << (*n) << std::endl;

      usefulOutputs.remove( node );
      uselessNodes.insert( node );
  
      node->_splitHere = false;
      node->_assignedPass = NULL;
    }
  }

  SplitShaderHeuristics heuristics;
  bool valid = rdsCompile( usefulOutputs, heuristics );
  assert( valid );

  int score = (inA->cost + inB->cost) - heuristics.cost;
  assert( score >= 0 );

  SplitPassInfo* result = new SplitPassInfo();
  result->allOutputs.swap( allOutputs );
  result->usefulOutputs.swap( usefulOutputs );

  result->cost = heuristics.cost;
  result->heuristics = heuristics;

  // calculate ancestors of the merged set...
  result->ancestors.setUnion( inA->ancestors, inB->ancestors );
  result->ancestors /= inB->allOutputs;
  result->ancestors /= inA->allOutputs;

  // calculate descendents of the merged set
  result->descendents.setUnion( inA->descendents, inB->descendents );
  result->descendents /= inB->allOutputs;
  result->descendents /= inA->allOutputs;

  _passes.erase( inA );
  _passes.erase( inB );

  delete inA;
  delete inB;

  _passes.insert( result );

  // TIM: finalize the merge by making the outputs of
  // the chosen merged pass know which pass outputs them
  for( SplitNodeSet::iterator i = result->usefulOutputs.begin(); i != result->usefulOutputs.end(); ++i )
  {
    SplitNode* node = _dagOrderNodeList[ *i ];
    node->_assignedPass = result;
  }
}

void SplitTree::rdsPrintPass( SplitPassInfo* inPass, std::ostream& inStream )
{
  if( inPass == NULL )
    return;

  if( inPass->printVisited ) return;
  inPass->printVisited = true;

  for( SplitNodeSet::iterator j = inPass->descendents.begin(); j != inPass->descendents.end(); ++j )
    rdsPrintPass( _dagOrderNodeList[*j]->_assignedPass, inStream );

//  dumpFile << "PRINT PASS " << (void*)inPass << std::endl;
//  for( NodeSet::iterator i = inPass->outputs.begin(); i != inPass->outputs.end(); ++i )
//  {
//    SplitNode* node = *i;
//    dumpFile << "% ";
//    node->dump( dumpFile );
//    dumpFile << std::endl;
//  }

  NodeList outputs;
  for( SplitNodeSet::iterator i = inPass->usefulOutputs.begin(); i != inPass->usefulOutputs.end(); ++i )
    outputs.push_back( _dagOrderNodeList[ *i ] );

  SplitShaderHeuristics unused;
  _compiler.compile( *this, outputs, inStream, unused, true );
}

void SplitTree::unmark( int inMarkBit ) const
{
  SplitNode::MarkBit markBit = (SplitNode::MarkBit)(inMarkBit);
  for( NodeList::const_iterator i = _dagOrderNodeList.begin(); i != _dagOrderNodeList.end(); ++i )
    (*i)->unmark( markBit );
  _outputPositionInterpolant->unmark( markBit );
}

void SplitTree::rdsSearch()
{

  // TIM: this is a pretty good early
  // exit, but it assumes a single output...
  /*if( _outputList.size() == 1 )
  {
    if( rdsCompile( _outputList[0] ) )
      return;
  }*/

  {for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
  {
    (*j)->_wasConsidered = false;
    (*j)->_wasSaved = false;
    (*j)->_wasConsideredSave = false;
    (*j)->_wasSavedSave = false;
    (*j)->_wasConsideredRecompute = false;
    (*j)->_wasSavedRecompute = false;
  }}

  int bestCost;

  size_t index = 0;
  size_t count = _multiplyReferencedNodes.size();
  for( NodeList::iterator i = _multiplyReferencedNodes.begin(); i != _multiplyReferencedNodes.end(); ++i )
  {
    SplitNode* node = *i;

    std::cout << "search step is considering node " << (void*)node <<
      " - number " << index++ << " of " << count << std::endl;

    if( !node->canBeSaved() )
    {
      std::cout << "trivial, skipping" << std::endl;
      continue;
    }

    bool trySave = true;
    bool tryRecompute = true;
    int saveCost = 0;
    int recomputeCost = 0;

    if( i != _multiplyReferencedNodes.begin() )
    {
      if( !node->_wasConsideredSave && !node->_wasConsideredRecompute )
      {
        // this node has no real impact, it never got looked at
//        std::cout << "skipping !!!" << std::endl;
        continue; 
      }

      if( node->_wasConsideredSave
        && node->_wasConsideredRecompute
        && (node->_wasSavedSave == node->_wasSavedRecompute) )
      {
        if( node->_wasSavedSave )
        {
//          std::cout << "skipping save" << std::endl;
          trySave = false;
          saveCost = bestCost;
        }
        else
        {
//          std::cout << "skipping recompute" << std::endl;
          tryRecompute = false;
          recomputeCost = bestCost;
        }
      }
    }
/*
    {for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
    {
      (*j)->_wasConsidered = false;
      (*j)->_wasSaved = false;
    }}*/

    if( trySave )
    {
      {for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
      {
        (*j)->_wasConsidered = false;
        (*j)->_wasSaved = false;
      }}

      node->_rdsFixedMarked = true;
      node->_rdsFixedUnmarked = false;
      saveCost = rdsCompileConfiguration();

      {for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
      {
        (*j)->_wasConsideredSave = (*j)->_wasConsidered;
        (*j)->_wasSavedSave = (*j)->_wasSaved;
      }}
    }
    std::cout << "####### save cost is " << saveCost << std::endl;

    if( tryRecompute )
    {
      {for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
      {
        (*j)->_wasConsidered = false;
        (*j)->_wasSaved = false;
      }}

      node->_rdsFixedMarked = false;
      node->_rdsFixedUnmarked = true;
      recomputeCost = rdsCompileConfiguration();

      {for( NodeList::iterator j = _dagOrderNodeList.begin(); j != _dagOrderNodeList.end(); ++j )
      {
        (*j)->_wasConsideredRecompute = (*j)->_wasConsidered;
        (*j)->_wasSavedRecompute = (*j)->_wasSaved;
      }}
    }
    std::cout << "####### recompute cost is " << recomputeCost << std::endl;

    if( saveCost < recomputeCost )
    {
      bestCost = saveCost;
      node->_rdsFixedUnmarked = false;
      node->_rdsFixedMarked = true;

      std::cout << "####### @final decision is to save " << (void*)node << std::endl << "  ";
//      node->dump( std::cout );
//      std::cout << std::endl;
    }
    else
    {
      bestCost = recomputeCost;
      node->_rdsFixedUnmarked = true;
      node->_rdsFixedMarked = false;

      std::cout << "####### final decision is to recompute " << (void*)node << std::endl << "  ";
//      node->dump( dumpFile );
//      dumpFile << std::endl;
    }
  }

  // use the resulting configuration
  // for one final compile pass
  // int finalCost = rdsCompileConfiguration( true );
  // dumpFile << "final cost is " << finalCost << std::endl;
}

int SplitTree::rdsCompileConfiguration( bool inLastTime )
{
  rdsSubdivide();
  rdsMergePasses( inLastTime );
/*
  dumpPassConfiguration( std::cout );
  
  std::ofstream flub( "flub.txt" );
  for( PassSet::iterator p = _passes.begin(); p != _passes.end(); ++p )
    rdsPrintPass( *p, flub );
*/
  return getPartitionCost();
}

void SplitTree::rdsMerge( SplitNode* n, SplitShaderHeuristics& outHeuristics )
{
  assert( n );

//  dumpFile << "MERGE " << (void*)n << std::endl << "   ";
//  n->dump( dumpFile );
//  dumpFile << std::endl;

  // unvisit nodes
  unmark( SplitNode::kMarkBit_Merged );

  size_t childCount = n->getGraphChildCount();
  for( size_t i = 0; i < childCount; i++ )
  {
    SplitNode* child = n->getIndexedGraphChild(i);
    rdsMergeRec( child );
  }

  rdsTryMerge( n, outHeuristics );
}

void SplitTree::rdsTryMerge( SplitNode* n, SplitShaderHeuristics& outHeuristics )
{
  assert( n );

//  dumpFile << "TRY MERGE " << (void*)n << std::endl;
//  n->dump( dumpFile );
//  dumpFile << std::endl;

  // first try to merge with all children
  if( rdsCompile( n, outHeuristics ) )
    return;

//  dumpFile << "whole thing didn't work, trying to split" << std::endl;

  // count the number of unsaved kids
  size_t childCount = n->getGraphChildCount();
  NodeList unsavedChildren;
  for( size_t i = 0; i < childCount; i++ )
  {
    SplitNode* child = n->getIndexedGraphChild(i);
    if( !child->isMarkedAsSplit() )
      unsavedChildren.push_back( child );
  }
  size_t unsavedChildCount = unsavedChildren.size();

  assert( unsavedChildCount > 0 );

  size_t subsetSize = unsavedChildCount;
  while( subsetSize-- > 0 )
  {
    // try to do merges with the given subset size
//    dumpFile << "trying merges of " << subsetSize << " of the " << unsavedChildCount << " children" << std::endl;
    if( rdsMergeSome( n, unsavedChildren, subsetSize, outHeuristics ) )
      return;
  }

  assert( false );
}

bool SplitTree::rdsMergeSome( SplitNode* n, const NodeList& inUnsavedChildren, size_t inSubsetSize, SplitShaderHeuristics& outHeuristics )
{
  const NodeList& unsavedChildren = inUnsavedChildren;
  size_t unsavedChildCount = unsavedChildren.size();
  size_t subsetSize = inSubsetSize;

  std::vector< size_t > validSubsets;
  std::vector< SplitShaderHeuristics > validHeuristics;

  SplitSubsetGenerator generator( subsetSize, unsavedChildCount );
  while( generator.hasMore() )
  {
    size_t subsetBitfield = generator.getNext();

//    dumpFile << "subset chosen was " << subsetBitfield << std::endl;

    for( size_t i = 0; i < unsavedChildCount; i++ )
      unsavedChildren[i]->_splitHere = (subsetBitfield & (1 << i)) == 0;

    SplitShaderHeuristics subsetHeuristics;
    if( rdsCompile( n, subsetHeuristics ) )
    {
//      dumpFile << "subset " << subsetBitfield << " was valid (cost = " << subsetHeuristics.cost << ")" << std::endl;

      validSubsets.push_back( subsetBitfield );
      validHeuristics.push_back( subsetHeuristics );
    }
  }

  size_t validSubsetCount = validSubsets.size();
  
  if( validSubsetCount == 0 )
    return false;

  size_t bestSubset = validSubsets[0];
  SplitShaderHeuristics bestHeuristics = validHeuristics[0];
  size_t i; 
  for( i = 1; i < validSubsetCount; i++ )
  {
    size_t otherSubset = validSubsets[i];
    SplitShaderHeuristics otherHeuristics = validHeuristics[i];

    if( otherHeuristics.cost < bestHeuristics.cost )
    {
      bestSubset = otherSubset;
      bestHeuristics = otherHeuristics;
    }
  }

//  dumpFile << "subset " << bestSubset << " was chosen (cost = " << bestHeuristics.cost << ")" << std::endl;

  // set the state of the child nodes to reflect
  // the chosen subset
  for( i = 0; i < unsavedChildCount; i++ )
  {
    if( (bestSubset & (1 << i)) == 0 )
    {
      unsavedChildren[i]->_splitHere = true;

      SplitShaderHeuristics heuristics;
      rdsCompile( unsavedChildren[i], heuristics );
      unsavedChildren[i]->setHeuristics( heuristics );

//      dumpFile << "merge step decided to @save " << (void*)(unsavedChildren[i]) << std::endl << "  ";
//      unsavedChildren[i]->dump( dumpFile );
//      dumpFile << std::endl;
    }
    else
      unsavedChildren[i]->_splitHere = false;
  }

  outHeuristics = bestHeuristics;
  return true;
}

void SplitTree::rdsMergeRec( SplitNode* n )
{
  assert( n );

//  dumpFile << "MERGE REC " << (void*)n << "   " << std::endl;
//  n->dump( dumpFile );
//  dumpFile << std::endl;

  if( n->isMarkedAsSplit() )
  {
//    dumpFile << "ignored as it was a split" << std::endl;
    return;
  }

  if( n->isPDTNode() )
  {
//    dumpFile << "ignored as it was a PDT node" << std::endl;
    return;
  }

  if( n->isMarked( SplitNode::kMarkBit_Merged ) )
  {
//    dumpFile << "ignored as it was an already-considered node" << std::endl;
    return;
  }

  n->mark( SplitNode::kMarkBit_Merged );

  size_t childCount = n->getGraphChildCount();
  
  // leaf nodes had better always work...
  if( childCount == 0 )
  {
    if( !rdsCompile( n ) )
      assert( false );
  }

  // first merge the subtrees
  for( size_t i = 0; i < childCount; i++ )
  {
    SplitNode* child = n->getIndexedGraphChild(i);
    rdsMergeRec( child );
  }

  SplitShaderHeuristics unused;
  rdsTryMerge( n, unused );

//  dumpFile << "MERGE REC DONE " << (void*)n << std::endl;
}

void SplitTree::rdsSubdivide()
{
  for( NodeList::iterator i = _dagOrderNodeList.begin(); i != _dagOrderNodeList.end(); ++i )
    (*i)->rdsUnmark();
  _outputPositionInterpolant->rdsUnmark();

  SplitShaderHeuristics unused;
  rdsSubdivide( _pseudoRoot, unused );
}

void SplitTree::rdsSubdivide( SplitNode* t, SplitShaderHeuristics& outHeuristics )
{
  assert( t );
  assert( t->isPDTNode() );

//  dumpFile << "SUBDIVIDE " << (void*)t << "   " << std::endl;
//  t->dump( dumpFile );
//  dumpFile << std::endl;

  // if it fits in a single pass, just do that...
  if( rdsCompile( t, outHeuristics ) )
  {
//    dumpFile << "it worked in a single pass!!" << std::endl;
    return;
  }

  size_t childCount = t->getPDTChildCount();
  for( size_t i = 0; i < childCount; i++ )
  {
    SplitNode* child = t->getIndexedPDTChild(i);

    // subdivide the child
    SplitShaderHeuristics childHeuristics;
    rdsSubdivide( child, childHeuristics );

    // try again to compile the current node
    if( rdsCompile( t, outHeuristics ) )
    {
//      dumpFile << "early exist after child subdivide" << std::endl;
      return;
    }

    // make save/recompute decision
    // for the child node
    rdsDecideSave( child, childHeuristics );
  }

  // last chance - compile again and hope it works
  if( rdsCompile( t, outHeuristics ) )
  {
//    dumpFile << "early exist after all children subdivided" << std::endl;
    return;
  }

//  dumpFile << "have to apply merging to this subtree" << std::endl;
  // otherwise we need to apply merging
  rdsMerge( t, outHeuristics );
}

void SplitTree::rdsDecideSave( SplitNode* n, const SplitShaderHeuristics& inHeuristics )
{
  if( !n->isMultiplyReferenced() )
    return;

//  dumpFile << "subdivide is deciding whether to save " << (void*)n << std::endl;
//  n->dump( dumpFile );
//  dumpFile << std::endl;

  n->setHeuristics( inHeuristics );

  if( n->_rdsFixedMarked )
  {
//    dumpFile << "fixed as saved" << std::endl;
    n->_splitHere = true;
    n->setHeuristics( inHeuristics );
  }
  else if( n->_rdsFixedUnmarked )
  {
//    dumpFile << "fixed as unsaved" << std::endl;
    n->_splitHere = false;
  }
  else
  {
    n->_splitHere = !( inHeuristics.recompute );

    n->_wasConsidered = true;
    n->_wasSaved = n->_splitHere;

    if( n->_splitHere )
      n->setHeuristics( inHeuristics );

//    dumpFile << "heuristic decided to " << (n->_splitHere ? "@save" : "recompute" ) << std::endl;
  }
}

bool SplitTree::rdsCompile( SplitNode* inNode )
{
  SplitShaderHeuristics heuristics;
  rdsCompile( inNode, heuristics );
  return heuristics.valid;
}

bool SplitTree::rdsCompile( SplitNode* inNode, SplitShaderHeuristics& outHeuristics )
{
  if( inNode == _pseudoRoot )
  {
    // TIM: the pseudo root will only
    // compile succesfully if all
    // of its children are splits
    bool valid = true;
    size_t childCount = _pseudoRoot->getGraphChildCount();
    for( size_t i = 0; i < childCount; i++ )
    {
      if( !_pseudoRoot->getIndexedGraphChild(i)->isMarkedAsSplit() )
        valid = false;
    }

    outHeuristics.cost = 0;
    outHeuristics.recompute = true;
    outHeuristics.valid = valid;
    return valid;
  }

  // TIM: we really need to handle this better:
  if( !inNode->canBeSaved() )
  {
    outHeuristics.cost = 0;
    outHeuristics.recompute = true;
    outHeuristics.valid = true;
    return true;
  }

  SplitNodeSet outputSet;
  outputSet.insert( inNode );

  return rdsCompile( outputSet, outHeuristics );
}

bool SplitTree::rdsCompile( const SplitNodeSet& inNodes, SplitShaderHeuristics& outHeuristics )
{
  unsigned long startCompile = getTime();

  NodeList nodes;
  for( SplitNodeSet::iterator i = inNodes.begin(); i != inNodes.end(); ++i )
    nodes.push_back( _dagOrderNodeList[ *i ] );

  std::ostringstream nullStream;
  _compiler.compile( *this, nodes, nullStream, outHeuristics );

  unsigned long stopCompile = getTime();
  timeCompilingCounter += stopCompile - startCompile;

  return outHeuristics.valid;
}

int SplitTree::getPartitionCost()
{
  int totalCost = 0;
  for( PassSet::iterator i = _passes.begin(); i != _passes.end(); ++i )
    totalCost += (*i)->cost;

  return totalCost;
}

void SplitTree::printShaderFunction( const NodeList& inOutputs, std::ostream& inStream ) const
{
  unsigned long startPrint = getTime();

  SplitArgumentTraversal printArguments(inStream,_outputPositionInterpolant);
  SplitStatementTraversal printStatements(inStream,_outputPositionInterpolant);

  for( size_t i = 0; i < _dagOrderNodeList.size(); i++ )
    _dagOrderNodeList[i]->unmarkAsOutput();
  _outputPositionInterpolant->unmarkAsOutput();

  for( NodeList::const_iterator j = inOutputs.begin(); j != inOutputs.end(); ++j )
    (*j)->markAsOutput();

  // create the wrapper for the function
  inStream << "void main(" << std::endl;

  unmark( SplitNode::kMarkBit_SubPrinted );
  printArguments( inOutputs );

  inStream << " ) {" << std::endl;

  unmark( SplitNode::kMarkBit_SubPrinted );
  printStatements( inOutputs );

  inStream << "}" << std::endl;

  unsigned long stopPrint = getTime();
  timePrintingCounter += stopPrint - startPrint;
}

void SplitTree::printArgumentAnnotations( const NodeList& inOutputs, std::ostream& inStream ) const
{
  SplitAnnotationTraversal printAnnotations(inStream,_outputPositionInterpolant);

  unmark( SplitNode::kMarkBit_SubPrinted );
  printAnnotations( inOutputs );
}

static FunctionType* getFunctionType( FunctionDef* inFunctionDef )
{
  Decl* functionDecl = inFunctionDef->decl;
  assert( functionDecl->form->type == TT_Function );
  return ((FunctionType*)functionDecl->form);
}

void SplitTree::build( FunctionDef* inFunctionDef, const std::vector<SplitNode*>& inArguments )
{
  // TIM: hack to make temporaries have a shared position stream
  _outputPositionInterpolant = new InputInterpolantSplitNode( -1, 0, kSplitBasicType_Float2 );

  FunctionType* functionType = getFunctionType( inFunctionDef );

  SplitTreeBuilder builder( *this );

//  std::cerr << "function args: " << functionType->nArgs << " args passed: " << inArguments.size();

  assert( functionType->nArgs == inArguments.size() );

  unsigned int i;
  for( i = 0; i < functionType->nArgs; i++ )
  {
    Decl* argumentDecl = functionType->args[i];
    builder.addArgument( argumentDecl, i, inArguments[i] );
  }

  Statement* statement = inFunctionDef->head;
  while( statement )
  {
    statement->buildSplitTree( builder );

    statement = statement->next;
  }

  _resultValue = builder.getResultValue();

  // we were called with arguments
  // thus we don't deal with creating
  // output nodes, or with building
  // the dominator tree...
}

void SplitTree::build( FunctionDef* inFunctionDef )
{
  // TIM: hack to make temporaries have a shared position stream
  _outputPositionInterpolant = new InputInterpolantSplitNode( -1, 0, kSplitBasicType_Float2 );

  FunctionType* functionType = getFunctionType( inFunctionDef );
  Statement* headStatement = inFunctionDef->head;

  SplitTreeBuilder builder( *this );
  unsigned int i;
  for( i = 0; i < functionType->nArgs; i++ )
  {
    Decl* argumentDecl = functionType->args[i];
    builder.addArgument( argumentDecl, i );
  }

  Statement* statement = headStatement;
  while( statement )
  {
    statement->buildSplitTree( builder );

    statement = statement->next;
  }

  _pseudoRoot = new SplitRootNode();

  _resultValue = builder.getResultValue();

  for( i = 0; i < functionType->nArgs; i++ )
  {
    Decl* argumentDecl = functionType->args[i];
    Type* argumentType = argumentDecl->form;
    std::string name = argumentDecl->name->name;

    if( (argumentType->getQualifiers() & TQ_Out) != 0 )
    {
      SplitNode* outputValue = builder.findVariable( name )->getValueNode();

      _pseudoRoot->addChild( outputValue );

      _outputList.push_back( outputValue );
      _outputArgumentIndices.push_back( (i+1) );
    }
  }

  if( _resultValue )
  {
    _pseudoRoot->addChild( _resultValue );
    _outputList.push_back( _resultValue );
    _outputArgumentIndices.push_back( 0 );
  }

  buildDominatorTree();
//  std::cerr << "done" << std::endl;
}
