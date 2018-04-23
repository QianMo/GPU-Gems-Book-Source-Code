// splitconfiguration.h
#ifndef __SPLITCONFIGURATION_H__
#define __SPLITCONFIGURATION_H__

#ifdef _WIN32
#pragma warning(disable:4786)
//debug symbol warning
#endif

class SplitConfiguration
{
public:
  SplitConfiguration();

  void load( const char* inFileName );
  void save( const char* inFileName );

  bool validateShaders;

  int maximumArithmeticInstructionCount;
  int maximumTextureInstructionCount;
  int maximumSamplerCount;
  int maximumInterpolantCount;
  int maximumConstantCount;
  int maximumTemporaryCount;
  int maximumOutputCount;

  int passCost;
  int textureInstructionCost;
  int arithmeticInstructionCost;
  int samplerCost;
  int interpolantCost;
  int constantCost;
  int temporaryCost;
  int outputCost;
};

#endif
