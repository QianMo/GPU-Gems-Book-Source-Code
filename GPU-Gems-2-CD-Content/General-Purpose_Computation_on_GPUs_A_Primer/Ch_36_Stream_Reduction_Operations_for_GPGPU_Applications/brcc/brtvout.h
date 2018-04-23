#ifdef _WIN32
#pragma warning(disable:4786)
#endif
#include <set>
#include <map>
#include <string>
#define INF_SENTINEL
typedef  std::map<std::string,std::set<unsigned int> > VoutFunctionType;
extern VoutFunctionType voutFunctions;
void transform_vout(class TransUnit*);

