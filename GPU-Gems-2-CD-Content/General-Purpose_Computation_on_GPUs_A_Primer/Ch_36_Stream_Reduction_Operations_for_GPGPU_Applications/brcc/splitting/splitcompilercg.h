// splitcompilercg.h
#ifndef __SPLITCOMPILERCG_H__
#define __SPLITCOMPILERCG_H__

#ifdef _WIN32
#pragma warning(disable:4786)
//debug symbol warning
#endif

#include "splitcompiler.h"
#include "../codegen.h"
#include <string>

class SplitCompilerCg
  : public SplitCompiler
{
public:
  SplitCompilerCg( CodeGenTarget inTarget )
    : _target(inTarget) {}

protected:
  virtual void printHeaderCode( std::ostream& inStream ) const;
  virtual void compileShader( const std::string& inHighLevelCode, std::ostream& outLowLevelCode, const SplitConfiguration& inConfiguration, SplitShaderHeuristics& outHeuristics ) const;

private:
  CodeGenTarget _target;
};

#endif
