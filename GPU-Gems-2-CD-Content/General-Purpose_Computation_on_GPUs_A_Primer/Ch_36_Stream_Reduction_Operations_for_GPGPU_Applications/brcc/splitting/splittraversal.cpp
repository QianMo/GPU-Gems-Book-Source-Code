 // splittraversal.cpp
#include "splittraversal.h"

#include "splitnode.h"

void SplitNodeTraversal::traverseGraphChildren( SplitNode* inNode )
{
  size_t childCount = inNode->getGraphChildCount();
  for( size_t i = 0; i < childCount; i++ )
    traverse( inNode->getIndexedGraphChild(i) );
}

void SplitArgumentTraversal::traverse( SplitNode* inNode )
{
  if( inNode->isMarked( SplitNode::kMarkBit_SubPrinted ) ) return;
  inNode->mark( SplitNode::kMarkBit_SubPrinted );

  bool traverseChildren = true;

  if( inNode->isMarkedAsOutput() )
  {
    traverseGraphChildren( inNode );

    if( hasOutput )
      stream << ",\n\t\t";
    else
      stream << "\t\t";

    // ignore other goo, and assume it needs an output
    stream << "out float4 ";
    inNode->printTemporaryName( stream );
    stream << " : COLOR" << argumentCounter.outputCount++;
    hasOutput = true;

    traverseChildren = false;
  }
  else if( inNode->isMarkedAsSplit() )
  {
    traverse( outputPosition );

    if( hasOutput )
      stream << ",\n\t\t";
    else
      stream << "\t\t";

    // ignore other goo, and assume it is a saved value...
    stream << "uniform _stype ";
    inNode->printTemporaryName( stream );
    stream << "_saved : register(s" << argumentCounter.samplerCount++ << ")";
    hasOutput = true;

    traverseChildren = false;
  }

  if( traverseChildren )
    traverseGraphChildren( inNode );

  if( !inNode->needsArgument() ) return;

  if( hasOutput )
    stream << ",\n\t\t";
  else
    stream << "\t\t";

  inNode->printArgumentInfo( stream, argumentCounter );
  hasOutput = true;
}

void SplitStatementTraversal::traverse( SplitNode* inNode )
{
  if( inNode->isMarked( SplitNode::kMarkBit_SubPrinted ) ) return;
  inNode->mark( SplitNode::kMarkBit_SubPrinted );


  if( !inNode->needsTemporaryExpression() ) return;

  if( inNode->isMarkedAsOutput() )
  {
    traverseGraphChildren( inNode );

    inNode->printTemporaryName( stream );
    stream << " = float4(";
    inNode->printTemporaryExpression( stream );
    switch( inNode->inferredType )
    {
    case kSplitBasicType_Float:
      stream << ",0,0,0";
      break;
    case kSplitBasicType_Float2:
      stream << ",0,0";
      break;
    case kSplitBasicType_Float3:
      stream << ",0";
      break;
    case kSplitBasicType_Float4:
      break;
    default:
       // Ian: What if Unknown type???
       break;
    }
    stream << ");" << std::endl;
    return;
  }

  if( inNode->isMarkedAsSplit() )
  {
    traverse( outputPosition );

    stream << inNode->inferredType;
    stream << " ";
    inNode->printTemporaryName( stream );
    stream << " = ";

    // TIM: fake it for now
    stream << "__sample2(";
    inNode->printTemporaryName( stream );
    stream << "_saved, ";
    outputPosition->printTemporaryName( stream );
    stream << ");" << std::endl;
    return;
  }

  traverseGraphChildren( inNode );

  stream << "\t";
  if( inNode->needsTemporaryVariable() )
  {
    if( inNode->inferredType == kSplitBasicType_Unknown )
    {
      std::cerr << "unknown intermediate type for: ";
      inNode->dump( std::cerr );
      std::cerr << std::endl;
    }

    // if it needs us to declare the var, do it...
    stream << inNode->inferredType;
    stream << " ";
  }
  inNode->printTemporaryName( stream );
  stream << " = ";
  inNode->printTemporaryExpression( stream );
  stream << ";";
//  stream << " // ";
//  inNode->dump( stream );
//  inNode->printExpression( stream );
  stream << std::endl;
}

void SplitAnnotationTraversal::traverse( SplitNode* inNode )
{
  if( inNode->isMarked( SplitNode::kMarkBit_SubPrinted ) ) return;
  inNode->mark( SplitNode::kMarkBit_SubPrinted );


  if( inNode->isMarkedAsOutput() )
  {
    traverseGraphChildren( inNode );

    stream << "\t\t\t";
    stream << ".output(" << inNode->getTemporaryID() << ",0)";
    stream << "\n";
    return;
  }

  if( inNode->isMarkedAsSplit() )
  {
    traverse( outputPosition );

    stream << "\t\t\t";
    stream << ".sampler(" << inNode->getTemporaryID() << ",0)";
    stream << "\n";
    // mark temporary sampler...
    return;
  }

  traverseGraphChildren( inNode );

  if( !inNode->needsAnnotation() ) return;

  stream << "\t\t\t";
  inNode->printAnnotationInfo( stream );
  stream << "\n";
}
