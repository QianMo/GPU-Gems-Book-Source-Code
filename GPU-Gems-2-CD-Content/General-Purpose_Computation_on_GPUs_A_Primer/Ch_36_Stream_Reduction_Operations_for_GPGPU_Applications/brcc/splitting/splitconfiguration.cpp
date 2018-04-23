// splitconfiguration.cpp
#include "splitconfiguration.h"

#include <fstream>

SplitConfiguration::SplitConfiguration()
{
  validateShaders = true;
  
  maximumArithmeticInstructionCount = 64;
  maximumTextureInstructionCount = 16;
  maximumSamplerCount = 16;
  maximumInterpolantCount = 8;
  maximumConstantCount = 32;
  maximumTemporaryCount = 12;

  passCost = 24;
  textureInstructionCost = 8;
  arithmeticInstructionCost = 1;
  samplerCost = 0;
  interpolantCost = 0;
  constantCost = 0;
  temporaryCost = 0;
  outputCost = 5;
}

void SplitConfiguration::load( const char* inFileName )
{
  std::ifstream stream( inFileName );

  stream >> validateShaders;

  stream >> maximumArithmeticInstructionCount;
  stream >> maximumTextureInstructionCount;
  stream >> maximumSamplerCount;
  stream >> maximumInterpolantCount;
  stream >> maximumConstantCount;
  stream >> maximumTemporaryCount;

  stream >> passCost;
  stream >> textureInstructionCost;
  stream >> arithmeticInstructionCost;
  stream >> samplerCost;
  stream >> interpolantCost;
  stream >> constantCost;
  stream >> temporaryCost;
  stream >> outputCost;
}

void SplitConfiguration::save( const char* inFileName )
{
  std::ofstream stream( inFileName );

  stream << validateShaders << std::endl;

  stream << maximumArithmeticInstructionCount << std::endl;
  stream << maximumTextureInstructionCount << std::endl;
  stream << maximumSamplerCount << std::endl;
  stream << maximumInterpolantCount << std::endl;
  stream << maximumConstantCount << std::endl;
  stream << maximumTemporaryCount << std::endl;

  stream << passCost << std::endl;
  stream << textureInstructionCost << std::endl;
  stream << arithmeticInstructionCost << std::endl;
  stream << samplerCost << std::endl;
  stream << interpolantCost << std::endl;
  stream << constantCost << std::endl;
  stream << temporaryCost << std::endl;
  stream << outputCost << std::endl;
}
