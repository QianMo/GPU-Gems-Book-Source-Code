#ifdef _WIN32
#pragma warning(disable:4786)
#include <ios>
#else
#include <iostream>
#endif
#include <sstream>
#include <iomanip>

#include "ctool.h"
void printGatherIntrinsics(std::ostream&out) {
   for (unsigned int iterindex = 0;iterindex<2;++iterindex) {
      for (unsigned int i=1;i<=4;++i) {
         for (unsigned int j=1;j<=2;++j) {
            out << "kernel void streamGatherOp"<<i<<j;
            if (iterindex)
               out << "I";
            out << " (out float";
            if (i>1)
               out << i;
            out << " t <>, ";
            if (iterindex)
               out << "iter ";
            out << "float";
            if (j>1)
               out << j;
            out << " index<>, float";
            if (i>1)
               out << i;
            out << " array";
            for (unsigned int k=0;k<j;++k) {
               out<<"[]";
            }
            out << ") {"<<std::endl;
            indent(out,1);
            out << "t = array[index];"<<std::endl;
            out << "}"<<std::endl;
         }
      }
   }
}
std::string printGatherIntrinsics () {
   std::ostringstream out;
   printGatherIntrinsics(out);
   return out.str();
}
void ComputeGatherIntrinsics(std::string &o,std::string path, std::string file) {
   if (o.find("STREAM_GATHER_FETCH")!=std::string::npos) {
      o=printGatherIntrinsics()+"\n#line 1\n"+o;//re_add #line
   }
}
