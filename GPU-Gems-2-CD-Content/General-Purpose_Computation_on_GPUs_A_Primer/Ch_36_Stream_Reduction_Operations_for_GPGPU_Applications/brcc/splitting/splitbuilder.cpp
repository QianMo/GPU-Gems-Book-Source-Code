// splitbuilder.cpp
#include "splitbuilder.h"

#include "splitnode.h"
#include "splittree.h"
#include "splitcompiler.h"
#include "../brtkernel.h"

#include <assert.h>

static SplitBasicType getInferredType( BaseType* inType )
{
  switch(inType->typemask)
  {
  case BT_Float:
    return kSplitBasicType_Float;
    break;
  case BT_Float2:
    return kSplitBasicType_Float2;
    break;
  case BT_Float3:
    return kSplitBasicType_Float3;
    break;
  case BT_Float4:
    return kSplitBasicType_Float4;
    break;
  default:
    return kSplitBasicType_Unknown;
    break;
  }
}

SplitTreeBuilder::SplitTreeBuilder( SplitTree& ioTree )
  : tree(ioTree), compiler(ioTree.getComplier()), _resultValue(NULL)
{
}

SplitNode* SplitTreeBuilder::getOutputInterpolant() {
  return tree.getOutputPositionInterpolant();
}

SplitNode* SplitTreeBuilder::addArgument( Decl* inDeclaration, int inArgumentIndex )
{
  std::string name = inDeclaration->name->name;
  Type* type = inDeclaration->form;
  TypeQual quals = type->getQualifiers();
  
  SplitBasicType inferredType = getInferredType( type->getBase() );

  SplitNode* result = NULL;
  if( (quals & TQ_Reduce) != 0 ) // reduction arg
  {
    result = new ReduceArgumentSplitNode( name, inferredType, inArgumentIndex );
  }
  else if( (quals & TQ_Iter) != 0 ) // iterator arg
  {
    result = new IteratorArgumentSplitNode( name, inferredType, inArgumentIndex );
  }
  else if( (quals & TQ_Out) != 0 ) // output arg
  {
    result = new OutputArgumentSplitNode( name, inferredType, inArgumentIndex, *this );
  }
  else if( type->isStream() ) // non-reduce stream
  {
    result = new StreamArgumentSplitNode( name, inferredType, inArgumentIndex, *this );
  }
  else if( type->isArray() ) // gather stream
  {
    result = new GatherArgumentSplitNode( name, inferredType, inArgumentIndex, *this );
  }
  else // non-stream constant
  {
    result = new ConstantArgumentSplitNode( name, inferredType, inArgumentIndex );
  }

  nodeMap[name] = result;
  return result;
}

SplitNode* SplitTreeBuilder::addArgument( Decl* inDeclaration, int inArgumentIndex, SplitNode* inValue )
{
  std::string name = inDeclaration->name->name;
  Type* type = inDeclaration->form;
  TypeQual quals = type->getQualifiers();

  SplitBasicType inferredType = getInferredType( type->getBase() );

  SplitNode* result = inValue;
  if( (quals & TQ_Reduce) != 0 ) // reduction arg
  {
  }
  if( (quals & TQ_Iter) != 0 ) // iterator arg
  {
    // possibly need to convert type...
    if( inferredType != inValue->inferredType )
      result = addConstructor( inferredType, inValue );
  }
  if( (quals & TQ_Out) != 0 ) // output arg
  {
  }
  else if( type->isStream() ) // non-reduce stream
  {
    // possibly need to convert type...
    if( inferredType != inValue->inferredType )
      result = addConstructor( inferredType, inValue );
  }
  else if( type->isArray() ) // gather stream
  {
  }
  else // non-stream constant
  {
    // possibly need to convert type...
    if( inferredType != inValue->inferredType )
      result = addConstructor( inferredType, inValue );
  }

  nodeMap[name] = result;
  return result;
}

void SplitTreeBuilder::addVariable( const std::string& inName, Type* inForm )
{
  // TIM: TODO: actually create something to represent the binding... :(
  nodeMap[inName] = new LocalVariableSplitNode( inName, getInferredType( inForm->getBase() ) );
}


SplitNode* SplitTreeBuilder::addConstant( Constant* inConstant )
{
  SplitNode* result = new BrtConstantSplitNode( inConstant );
  return result;
}

SplitNode* SplitTreeBuilder::addConstant( int inValue )
{
  SplitNode* result = new BrtConstantSplitNode( inValue );
  return result;
}

SplitNode* SplitTreeBuilder::addMember( SplitNode* inValue, const std::string& inName )
{
  assert( inValue );
  SplitNode* value = inValue->getValueNode();

  SplitNode* result = new BrtMemberSplitNode( value, inName );
  return result;
}

SplitNode* SplitTreeBuilder::addUnaryOp( const std::string& inOperation, SplitNode* inOperand )
{
  assert( inOperand );
  SplitNode* operand = inOperand->getValueNode();

  return new UnaryOpSplitNode( inOperation, operand );
}

SplitNode* SplitTreeBuilder::addBinaryOp( const std::string& inOperation, SplitNode* inLeft, SplitNode* inRight )
{
  assert( inLeft );
  assert( inRight );
  SplitNode* result = new BrtBinaryOpSplitNode( inOperation, inLeft->getValueNode(), inRight->getValueNode() );
  return result;
}

SplitNode* SplitTreeBuilder::addGather( SplitNode* inStream, const std::vector<SplitNode*> inIndices )
{
  assert( inStream );
  for( size_t i = 0; i < inIndices.size(); i++ )
    assert( inIndices[i] );

  GatherArgumentSplitNode* stream = inStream->isGatherArgument();
  assert(stream);
  InputSamplerSplitNode* sampler = stream->getSampler();

  // TIM: for now
  assert( inIndices.size() == 1 );
  SplitNode* index = inIndices[0]->getValueNode();

  SplitNode* textureCoordinate = index;

  if( compiler.mustScaleAndBiasGatherIndices() )
  {
    SplitNode* scaled = addBinaryOp( "*", index, stream->getScale() );
    SplitNode* biased = addBinaryOp( "+", scaled, stream->getBias() );
    textureCoordinate = biased;
  }

  SplitNode* result = new TextureFetchSplitNode( sampler, textureCoordinate );
  return result;
}

SplitNode* SplitTreeBuilder::addCast( BaseType* inType, SplitNode* inValue )
{
  SplitBasicType inferredType = getInferredType( inType );
  SplitNode* result = new CastSplitNode( inferredType, inValue->getValueNode() );
  return result;
}

SplitNode* SplitTreeBuilder::addConstructor( BaseType* inType, const std::vector<SplitNode*>& inArguments )
{
  for( size_t a = 0; a < inArguments.size(); a++ )
    assert( inArguments[a] );

  std::vector<SplitNode*> argumentValues;
  for( std::vector<SplitNode*>::const_iterator i = inArguments.begin(); i != inArguments.end(); ++i )
    argumentValues.push_back( (*i)->getValueNode() );

  SplitBasicType inferredType = getInferredType( inType );
  SplitNode* result = new ConstructorSplitNode( inferredType, argumentValues );
  return result;
}

SplitNode* SplitTreeBuilder::addConstructor( SplitBasicType inType, SplitNode* inX, SplitNode* inY, SplitNode* inZ, SplitNode* inW )
{
  assert( inX );

  std::vector<SplitNode*> argumentValues;
  if( inX )
    argumentValues.push_back( inX->getValueNode() );
  if( inY )
    argumentValues.push_back( inY->getValueNode() );
  if( inZ )
    argumentValues.push_back( inZ->getValueNode() );
  if( inW )
    argumentValues.push_back( inW->getValueNode() );

  SplitNode* result = new ConstructorSplitNode( inType, argumentValues );
  return result;
}

SplitNode* SplitTreeBuilder::addIndexof( const std::string& inName )
{
  SplitNode* variable = findVariable( inName );
  assert( variable );

//  variable->dump( std::cerr );

  IndexofableSplitNode* stream = variable->isIndexofable();
  assert( stream );
  return stream->getIndexofValue();
}

SplitNode* SplitTreeBuilder::addConditional( SplitNode* inCondition, SplitNode* inConsequent, SplitNode* inAlternate )
{
  assert( inCondition );
  assert( inConsequent );
  assert( inAlternate );
  SplitNode* result = new ConditionalSplitNode( inCondition->getValueNode(), inConsequent->getValueNode(), inAlternate->getValueNode() );
  return result;
}

SplitNode* SplitTreeBuilder::findVariable( const std::string& inName )
{
  NodeMap::iterator i = nodeMap.find( inName );
  if( i != nodeMap.end() )
    return (*i).second;
  else
  {
    std::cerr << "Undefined variable found in split-tree build process : " << inName << std::endl;
    return NULL;
  }
}

SplitNode* SplitTreeBuilder::assign( const std::string& inName, SplitNode* inValue )
{
  assert( inValue );

//  std::cerr << "assign to " << inName << std::endl;

  NodeMap::iterator i = nodeMap.find(inName);
  if( i == nodeMap.end() )
  {
    std::cerr << "failed to find " << inName << " to assign to!!!" << std::endl;
    return NULL;
  }

  SplitNode* variable = (*i).second;

  variable->assign( inValue->getValueNode()->getValueNode() );

  return variable;
}

SplitNode* SplitTreeBuilder::addFunctionCall( Expression* inFunction, const std::vector<SplitNode*>& inArguments )
{
  // we have to inline the function, applied to those arguments...

  // first find the function by name...
  assert(inFunction->etype == ET_Variable);
  Variable* variable = (Variable*)inFunction;
  std::string functionName = variable->name->name;
  SymEntry* entry = variable->name->entry;

  if( entry != NULL && entry->IsFctDecl() )
  {
//    std::cerr << functionName << " {" << std::endl;

    FunctionDef* function = entry->u2FunctionDef;

    BRTGPUKernelCode* kernelCode = new BRTPS20KernelCode( *function );

//    std::cout << "foo " << (void*)kernelCode << std::endl;

    function = kernelCode->fDef;

    // whoopee :)
    SplitTree subfunctionTree( function, compiler, inArguments );

    // TIM: is that enough?
//    std::cerr << "} " << functionName << std::endl;

//    std::cout << "bar " << (void*)kernelCode << std::endl;
    delete kernelCode;
    SplitNode* resultValue = subfunctionTree.getResultValue();
    return resultValue;
  }
  else
  {
    // let's assume that it's a built-in function
    std::string name = variable->name->name;
    SplitNode* result = new FunctionCallSplitNode( name, inArguments );
    return result;
  }
}


