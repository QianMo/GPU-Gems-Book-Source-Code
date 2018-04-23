//---------------------------------------------------------------------------
#ifndef __OSADAPTER__
#define __OSADAPTER__
//---------------------------------------------------------------------------
#ifdef WIN32
// WINDOWS
#  define OS_FILE_SEPARATOR '\\'
#  define drand48() (((double)(rand() % 255))/255.0)
#else
// LINUX
#  define stricmp           strcasecmp
#  define OS_FILE_SEPARATOR '/'
#endif
//---------------------------------------------------------------------------
class OSAdapter
{
public:
  static void        convertName(char *);
  static const char *convertName(const char *);
};
//---------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
