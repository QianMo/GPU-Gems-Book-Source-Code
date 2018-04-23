// splittree.h
#ifndef __SPLITTREE_H__
#define __SPLITTREE_H__
#ifdef _WIN32
#pragma warning(disable:4786)
//debug symbol warning
#endif

#include "splitconfiguration.h"
#include "splitcompiler.h"
#include "splitnodeset.h"

#include <string>
#include <map>
#include <set>
#include <vector>

class FunctionDef;
class SplitNode;
class SplitTechniqueDesc;
class SplitCompiler;

class SplitPassInfo
{
public:
  SplitPassInfo()
  {
    cost = 0;
    ancestorVisited = false;
    descendentVisited = false;
    printVisited = false;
    singletonNode = 0;
  }

//  typedef std::set< SplitNode* > NodeSet;

  SplitNodeSet allOutputs;
  SplitNodeSet usefulOutputs;

  SplitNodeSet ancestors;
  SplitNodeSet descendents;

  int cost;

  bool ancestorVisited;
  bool descendentVisited;
  bool printVisited;

  SplitShaderHeuristics heuristics;

  SplitNode* singletonNode;
};

class SplitTree
{
public:
  friend class SplitTreeBuilder;
  typedef std::map< std::string, SplitNode* > NodeMap;
  typedef std::vector< SplitNode* > NodeList;
  typedef std::set< SplitNode* > NodeSet;

  SplitTree( FunctionDef* inFunctionDef, const SplitCompiler& inCompiler );
  SplitTree( FunctionDef* inFunctionDef, const SplitCompiler& inCompiler, const std::vector<SplitNode*>& inArguments );
  virtual ~SplitTree();

  void printTechnique( const SplitTechniqueDesc& inTechniqueDesc, std::ostream& inStream );

  void printShaderFunction( std::ostream& inStream );

  // print a shader function def for the given subset of outputs
  void printShaderFunction( const NodeList& inOutputs, std::ostream& inStream ) const;
  void printArgumentAnnotations( const NodeList& inOutputs, std::ostream& inStream ) const;

  const SplitCompiler& getComplier() {
    return _compiler;
  }

  SplitNode* getResultValue() {
    return _resultValue;
  }

  SplitNode* getOutputPositionInterpolant() {
    return _outputPositionInterpolant;
  }

  const SplitConfiguration& getConfiguration() const {
    return _compilerConfiguration;
  }


private:

  void build( FunctionDef* inFunctionDef );
  void build( FunctionDef* inFunctionDef, const std::vector<SplitNode*>& inArguments );

  void buildDominatorTree();
  void dominatorDFS( SplitNode* inNode, SplitNode* inParent, size_t& ioID );
  void dumpDominatorTree();
  void dumpDominatorTree( std::ostream& inStream, SplitNode* inNode, int inLevel = 0 );

  void exhaustiveSearch();
  void exhaustiveSubsetSearch( size_t inSubsetSize, const NodeList& inNodes, int& outScore );
  bool exhaustiveSplitIsValid( int& outScore );

  void rdsSearch();
  int rdsCompileConfiguration( bool inLastTime = false );
  void rdsSubdivide();
  void rdsSubdivide( SplitNode* t, SplitShaderHeuristics& outHeuristics );
  void rdsDecideSave( SplitNode* n, const SplitShaderHeuristics& inHeuristics );

  void rdsMerge( SplitNode* n, SplitShaderHeuristics& outHeuristics );
  void rdsTryMerge( SplitNode* n, SplitShaderHeuristics& outHeuristics );
  bool rdsMergeSome( SplitNode* n, const NodeList& inUnsavedChildren, size_t inSubsetSize, SplitShaderHeuristics& outHeuristics );
  void rdsMergeRec( SplitNode* n );


  typedef std::set< SplitPassInfo* > PassSet;

  PassSet _passes;
  void rdsMergePasses( bool inLastTime = false );
  SplitPassInfo* rdsCreatePass( SplitNode* inNode );
  int rdsCanMergePasses( SplitPassInfo* inA, SplitPassInfo* inB, bool inAllowRemoval );
  void rdsMergePasses( SplitPassInfo* inA, SplitPassInfo* inB, bool inAllowRemoval );
  void rdsPrintPass( SplitPassInfo* inPass, std::ostream& inStream );

  void dumpPassConfiguration( std::ostream& inStream );

  void accumulateChildSplits( SplitNode* inSplit );
  void accumulateChildSplitsRec( SplitNode* inSplit, SplitNode* inDescenent );
  void accumulateParentSplits( SplitNode* inSplit );
  void accumulateParentSplitsRec( SplitNode* inSplit, SplitNode* inAncestor );

  void rdsAccumulatePassAncestors( SplitPassInfo* ioPass );
//  void rdsAccumulatePassAncestorsRec( SplitNode* inNode, SplitPassInfo* ioPass );
  void rdsAccumulatePassDescendents( SplitPassInfo* ioPass );
//  void rdsAccumulatePassDescendentsRec( SplitNode* inNode, SplitPassInfo* ioPass );

  void unmark( int inMarkBit ) const;

  NodeList _rdsNodeList;

  bool rdsCompile( SplitNode* inNode );
  bool rdsCompile( SplitNode* inNode, SplitShaderHeuristics& outHeuristics );
  bool rdsCompile( const SplitNodeSet& inNodes, SplitShaderHeuristics& outHeuristics );

  int getPartitionCost();


  SplitNode* _pseudoRoot;

  NodeList _outputList;
  std::vector<int> _outputArgumentIndices;
  NodeList _multiplyReferencedNodes;

  NodeList _dagOrderNodeList;

  SplitNode* _outputPositionInterpolant;
  SplitNode* _resultValue;

  const SplitCompiler& _compiler;
  SplitConfiguration _compilerConfiguration;

  std::string _functionName;
};

#endif
