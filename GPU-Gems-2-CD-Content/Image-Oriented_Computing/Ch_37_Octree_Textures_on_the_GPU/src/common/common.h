#ifndef __COMMON__
#define __COMMON__

#include <iostream>
#include "CCoreException.h"

#ifndef min 
#define min(a,b) ((a)<(b)?(a):(b))
#endif

#ifndef max
#define max(a,b) ((a)>(b)?(a):(b))
#endif

#define CHECK_GLERROR(m) if (glGetError()) std::cerr << std::endl << "ERROR: OpenGL - " << m << "\n\tline " << __LINE__ << " file " << __FILE__ << std::endl << std::endl;

#endif
