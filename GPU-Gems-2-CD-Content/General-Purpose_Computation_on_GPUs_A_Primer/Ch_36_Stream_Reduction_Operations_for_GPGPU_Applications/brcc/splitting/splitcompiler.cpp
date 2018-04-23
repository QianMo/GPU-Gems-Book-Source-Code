// splitcompiler.cpp
#include "splitcompiler.h"

#include "splitting.h"

#include <string>
#include <sstream>

void SplitCompiler::compile(
  const SplitTree& inTree, const NodeList& inOutputs, std::ostream& inStream, SplitShaderHeuristics& outHeuristics, bool forReal ) const
{
  std::ostringstream bodyStream;

  printHeaderCode( bodyStream );
  inTree.printShaderFunction( inOutputs, bodyStream );
  printFooterCode( bodyStream );

//  std::cerr << "\n###about to compile: \n" << bodyStream.str() << "\n###" << std::endl;

  std::ostringstream assemblerStream;
  compileShader( bodyStream.str(), assemblerStream, inTree.getConfiguration(), outHeuristics );
  std::string assemblerCode = assemblerStream.str();

  if( assemblerCode == "" && forReal )
    assert(false);

  // now we spit out the annotated version of things:
  inStream << "\t\t.pass( gpu_pass_desc(" << std::endl;
  printStringConstant( assemblerCode, "\t\t\t", inStream );
  inStream << " )" << std::endl;

  inTree.printArgumentAnnotations( inOutputs, inStream );

  inStream << "\t\t)" << std::endl;
}

void SplitCompiler::printStringConstant( const std::string& inString, const std::string& inPrefix, std::ostream& inStream ) const
{
  const char* s = inString.c_str();

  inStream << inPrefix << "\"";

  char c;
  while( (c = *s++) != '\0' )
  {
    if( c == '\n' )
      inStream << "\\n\"" << std::endl << inPrefix << "\"";
    else
      inStream << c;
  }

  inStream << "\"";
}
