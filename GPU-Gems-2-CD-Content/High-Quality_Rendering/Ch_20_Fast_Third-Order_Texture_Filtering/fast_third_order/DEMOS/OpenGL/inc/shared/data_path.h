#ifndef DATA_PATH_H
#define DATA_PATH_H

#ifdef _WIN32
#  pragma warning(disable:4786)   // symbol size limitation ... STL
#endif

#include <stdio.h>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

class data_path
{
public:
    std::string              file_path;
    std::string              path_name;
    std::vector<std::string> path;

    std::string get_path(std::string filename);
    std::string get_file(std::string filename);
	
    // data files, for read only
    FILE *  fopen(std::string filename, const char * mode = "rb");
    
#ifdef WIN32
    int fstat(std::string filename, struct _stat * stat);
#else
    int fstat(std::string filename, struct stat * stat);
#endif
};

#endif
