#ifndef SHPOOL_HPP
#define SHPOOL_HPP

#define SH_USE_MEMORY_POOL

#ifdef SH_USE_MEMORY_POOL

#include <cstddef>
#include "ShDllExport.hpp"

namespace SH {

class 
SH_DLLEXPORT
ShPool {
public:
  ShPool(std::size_t element_size, std::size_t block_size);

  void* alloc();
  void free(void*);

private:
  std::size_t m_element_size;
  std::size_t m_block_size;

  void* m_next;
};

}

#endif // SH_USE_MEMORY_POOL

#endif
