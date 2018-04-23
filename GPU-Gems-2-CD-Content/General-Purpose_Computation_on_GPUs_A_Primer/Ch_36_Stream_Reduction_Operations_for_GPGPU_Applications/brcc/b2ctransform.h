/*
 * b2ctransform.h --
 *
 *      Interface to brtvector.hpp-- transforms kernels into C++ code that can
 *      execute them on a CPU.
 */
#ifndef __BROOK2CPP_H__
#define __BROOK2CPP_H__
#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif
#include <map>
#include <string>
#include <set>
class functionProperties:public std::set<unsigned int>{public:
   std::set<std::string>callees;
   bool contains (unsigned int i) {
      return find(i)!=end();
   }
   bool notcontains(unsigned int i) {return !contains(i);}
   bool calls (std::string s) {return callees.find(s)!=callees.end();}
   bool notcalls(std::string s) {return !calls(s);}
};
extern std::map<std::string,functionProperties> FunctionProp;

class FunctionDef;
extern bool reduceNeeded(const FunctionDef * fDef);

extern void
Brook2Cpp_IdentifyIndexOf(class TransUnit * tu);

extern void
Brook2Cpp_ConvertKernel(FunctionDef *kDef);
extern void 
BrookReduce_ConvertKernel (FunctionDef * kdef);
extern void
BrookCombine_ConvertKernel (FunctionDef * kdef);
extern FunctionDef * changeFunctionCallForIndexOf(FunctionDef * fd);
#endif




