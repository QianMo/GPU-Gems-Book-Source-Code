#if !defined(__GNUC__) || __GNUC__ != 2
#include <ios>
#endif
#include <iostream>
#include <sstream>
#include <iomanip>
#include <assert.h>

#include "oglcheckgl.hpp"
#include "oglfunc.hpp"

static const unsigned int nglErrors = 6;

static char glError_txt[][32] = {
    "GL_INVALID_ENUM",
    "GL_INVALID_VALUE",
    "GL_INVALID_OPERATION",
    "GL_STACK_OVERFLOW",
    "GL_STACK_UNDERFLOW",
    "GL_OUT_OF_MEMORY",
};


void brook::__check_gl(int line, char *file) {
  GLenum r = glGetError();
  
  if (r != GL_NO_ERROR) {
    if (r - GL_INVALID_ENUM >= nglErrors)
      std::cerr << "GL: Unknown GL error on line "
                << line << " of " << "%s\n";
    else
      std::cerr << "GL: glGetError returned "
                << glError_txt[r - GL_INVALID_ENUM]
                << " on line "<< line << " of " << file << "\n";
    
    assert (r==GL_NO_ERROR);
  }
  return;
}
