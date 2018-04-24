// ************************************************
// common.hpp
// authors: Lee Howes and David B. Thomas
//
// Simple funcationality common to various files.
// ************************************************


#ifndef common_hpp
#define common_hpp

#include <stdlib.h>

std::vector < char >StrToVec(const char *str)
{
	return std::vector < char >(str, str + strlen(str) + 1);
}

#endif // #ifndef common_hpp
