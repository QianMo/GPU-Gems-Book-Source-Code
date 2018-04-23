// splitcompilerhlsl.cpp
#include "splitcompilerhlsl.h"

#include "splitting.h"
#include "../fxc.h"

#include <string>
#include <sstream>

void SplitCompilerHLSL::printHeaderCode( std::ostream& inStream ) const
{
  inStream
//    << "#ifdef USERECT\n"
//    << "#define _stype   samplerRECT\n"
//    << "#define _sfetch  texRECT\n"
//    << "#define __sample1(s,i) texRECT((s),float2(i,0))\n"
//    << "#define __sample2(s,i) texRECT((s),(i))\n"
//    << "#define _computeindexof(a,b) float4(a, 0, 0)\n"
//    << "#else\n"
    << "#define _stype   sampler\n"
    << "#define _sfetch  tex2D\n"
    << "#define __sample1(s,i) tex1D((s),(i))\n"
    << "#define __sample2(s,i) tex2D((s),(i))\n";
//    << "#define _computeindexof(a,b) (b)\n"
//    << "#endif\n\n";

}

static inline int max( int a, int b ) {
  return a > b ? a : b;
}

void SplitCompilerHLSL::compileShader(
  const std::string& inHighLevelCode, std::ostream& outLowLevelCode, const SplitConfiguration& inConfiguration, SplitShaderHeuristics& outHeuristics ) const
{
  ShaderResourceUsage usage;

  bool shouldValidate = inConfiguration.validateShaders;
//  bool shouldValidate = false;

  char* assemblerBuffer = compile_fxc( "unknown", inHighLevelCode.c_str(), CODEGEN_PS20, &usage, shouldValidate );
/*
  std::cout << "size is " << inHighLevelCode.length() << std::endl;

  char* assemblerBuffer = NULL;
  if( inHighLevelCode.length() < 65536 )
  {
    usage.arithmeticInstructionCount = 12;
    usage.textureInstructionCount = 2;
    usage.constantRegisterCount = 2;
    usage.interpolantRegisterCount = 1;
    usage.outputRegisterCount = 1;
    usage.samplerRegisterCount = 2;
    usage.temporaryRegisterCount = 3;
    char* assemblerBuffer = strdup("fibble");
  }*/

  if( assemblerBuffer == NULL )
  {
    outHeuristics.valid = false;
    outHeuristics.cost = 0;
    outHeuristics.recompute = true;
    return;
  }

  int textureInstructionCount = usage.textureInstructionCount;
  int arithmeticInstructionCount = usage.arithmeticInstructionCount;
  int samplerCount = usage.samplerRegisterCount;
  int interpolantCount = usage.interpolantRegisterCount;
  int constantCount = usage.constantRegisterCount;
  int temporaryCount = usage.temporaryRegisterCount;
  int outputCount = usage.outputRegisterCount;

  outHeuristics.arithmeticInstructionCount = arithmeticInstructionCount;
  outHeuristics.textureInstructionCount = textureInstructionCount;
  outHeuristics.samplerRegisterCount = samplerCount;
  outHeuristics.interpolantRegisterCount = interpolantCount;
  outHeuristics.constantRegisterCount = constantCount;
  outHeuristics.temporaryRegisterCount = temporaryCount;
  outHeuristics.outputRegisterCount = outputCount;
/*
  if( !shouldValidate )
  {
    if( (textureInstructionCount > inConfiguration.maximumTextureInstructionCount)
      || (arithmeticInstructionCount > inConfiguration.maximumArithmeticInstructionCount)
      || (samplerCount > inConfiguration.maximumSamplerCount)
      || (interpolantCount > inConfiguration.maximumSamplerCount)
      || (constantCount > inConfiguration.maximumConstantCount)
      || (temporaryCount > inConfiguration.maximumTemporaryCount)
      || (outputCount > inConfiguration.maximumOutputCount) )
    {
      std::cout << "compile failed because of configuration limits..." << std::endl;
      outHeuristics.valid = false;
      outHeuristics.cost = 0;
      outHeuristics.recompute = true;
      return;
    }
  }*/

  // int passCost = inConfiguration.passCost;
  // int textureInstructionCost = inConfiguration.textureInstructionCost;
  // int arithmeticInstructionCost = inConfiguration.arithmeticInstructionCost;
  // int samplerCost = inConfiguration.samplerCost;
  // int interpolantCost = inConfiguration.interpolantCost;
  // int constantCost = inConfiguration.constantCost;
  // int temporaryCost = inConfiguration.temporaryCost;
  // int outputCost = inConfiguration.outputCost;

  // TIM: shader model is interesting

  // TIM: we write 3 times as slowly on the ATI
  // when we do more than one output...

// nonlinear cost model
  
  int totalOutputCost = outputCount == 1 ? 6 : 3 * 6 * outputCount;

  // count all issues, (not just arith)
  int totalInstructionIssueCost = arithmeticInstructionCount + textureInstructionCount;

  int totalTextureInstructionCost = 8*textureInstructionCount;

  int bestCaseInstructionCost = max( totalInstructionIssueCost, totalTextureInstructionCost );
  int worstCastInstructionCost = totalInstructionIssueCost + totalTextureInstructionCost;

  // TIM: one extra instruction inserted by ATI?
  // seems to be confirmed by data...
  int averagedInstructionCost = (bestCaseInstructionCost + worstCastInstructionCost) / 2;

  // shader time = max( instruction exec, output write ) + pass overhead
  int shaderCost = max( averagedInstructionCost, totalOutputCost) + 9 + 1*outputCount;


// linear cost model
//  int shaderCost = arithmeticInstructionCount + 8*textureInstructionCount + 6*outputCount + 10;

/*
  int shaderCost = passCost
    + textureInstructionCost*textureInstructionCount
    + arithmeticInstructionCost*arithmeticInstructionCount
    + samplerCost*samplerCount
    + interpolantCost*interpolantCount
    + constantCost*constantCount
    + temporaryCost*temporaryCount
    + outputCost*outputCount;*/

  bool shouldRecompute = true;
//  if( textureInstructionCount*2 > inConfiguration.maximumTextureInstructionCount )
//    shouldRecompute = false;
//  if( arithmeticInstructionCount*2 > inConfiguration.maximumArithmeticInstructionCount )
//    shouldRecompute = false;


  if( textureInstructionCount*4 > inConfiguration.maximumTextureInstructionCount*3 )
    shouldRecompute = false;
  if( arithmeticInstructionCount*4 > inConfiguration.maximumArithmeticInstructionCount*3 )
    shouldRecompute = false;

//  if( samplerCount > 8 )
//    shouldRecompute = false;
//  if( interpolantCount > 4 )
//    shouldRecompute = false;
//  if( constantCount > 4 )
//    shouldRecompute = false;
//  if( temporaryCount > 8 )
//    shouldRecompute = false;

  outHeuristics.valid = true;
  outHeuristics.cost = shaderCost;
  outHeuristics.recompute = shouldRecompute;

  outLowLevelCode << assemblerBuffer;
  free( assemblerBuffer );
}
