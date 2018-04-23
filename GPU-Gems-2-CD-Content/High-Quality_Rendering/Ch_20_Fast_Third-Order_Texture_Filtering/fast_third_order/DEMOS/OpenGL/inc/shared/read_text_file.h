#ifndef READ_TEXT_FILE_H__CONVENIENCE
#define READ_TEXT_FILE_H__CONVENIENCE

#include <shared/data_path.h>

data_path get_png_path();
void set_text_path(const data_path & newpath);

#ifdef WIN32
void set_module_handle(unsigned long hM);
void set_module_restypename(const char * tname);
#endif
char * read_text_file(const char * filename, const char * type, unsigned long hModule=NULL);
char * read_text_file(const char * filename);

#endif
