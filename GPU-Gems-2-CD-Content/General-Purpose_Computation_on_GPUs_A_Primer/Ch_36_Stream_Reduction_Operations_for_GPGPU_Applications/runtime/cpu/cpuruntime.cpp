#include "cpu.hpp"

namespace brook{

  const char * CPU_RUNTIME_STRING="cpu";
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  CPURuntime::CPURuntime() {}

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  Kernel * CPURuntime::CreateKernel(const void* blah[]) {
    return new CPUKernel(blah);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  Stream * CPURuntime::CreateStream(unsigned int fieldCount, 
                                    const StreamType fieldTypes[],
                                    unsigned int dims, 
                                    const unsigned int extents[]) {
    return new CPUStream(fieldCount, fieldTypes, dims, extents);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  Iter * CPURuntime::CreateIter(StreamType type,
                                unsigned int dims, 
                                const unsigned int extents[], 
                                const float ranges[]) {
    return new CPUIter(type, dims, extents, ranges);
  } 
}

