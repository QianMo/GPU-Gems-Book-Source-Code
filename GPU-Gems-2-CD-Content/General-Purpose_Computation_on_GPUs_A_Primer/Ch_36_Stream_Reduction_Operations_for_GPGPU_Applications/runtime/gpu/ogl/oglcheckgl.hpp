
#ifndef OGLCHECKGL_HPP
#define OGLCHECKGL_HPP

namespace brook {

#define CHECK_GL() __check_gl(__LINE__, __FILE__)

  void __check_gl(int line, char *file);

}


#endif

