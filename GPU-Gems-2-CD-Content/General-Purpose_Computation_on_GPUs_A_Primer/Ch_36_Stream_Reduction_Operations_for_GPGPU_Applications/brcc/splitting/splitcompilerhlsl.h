// splitcompilerhlsl.h
#ifndef __SPLITCOMPILERHLSL_H__
#define __SPLITCOMPILERHLSL_H__

#ifdef _WIN32
#pragma warning(disable:4786)
//debug symbol warning
#endif

#include "splitcompiler.h"
#include <string>
class SplitCompilerHLSL
  : public SplitCompiler
{
protected:
  virtual void printHeaderCode( std::ostream& inStream ) const;
  virtual void compileShader( const std::string& inHighLevelCode, std::ostream& outLowLevelCode, const SplitConfiguration& inConfiguration, SplitShaderHeuristics& outHeuristics ) const;

  // TIM: complete hack, even for me
  virtual bool mustScaleAndBiasGatherIndices() const { return true; }
};

#endif
