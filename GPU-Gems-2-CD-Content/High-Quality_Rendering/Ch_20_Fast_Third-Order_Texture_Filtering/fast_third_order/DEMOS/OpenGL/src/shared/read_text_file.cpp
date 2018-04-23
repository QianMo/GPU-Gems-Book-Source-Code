#include <shared/read_text_file.h>
#include <shared/data_path.h>
#include <stdio.h>
#ifdef WIN32
#include <windows.h> // Module loading
#endif

using namespace std;

namespace
{
	data_path path;
#ifdef WIN32
	/// Structure containing resource-oriented data
	struct resource_access
	{
		resource_access() : hModule(NULL), res_type_id(0) {}
		HMODULE hModule;
		std::string res_type_name;
		unsigned long res_type_id;
	};
	resource_access resource;
#endif
}

#ifdef WIN32
void set_module_handle(unsigned long hM) 
{ 
	resource.hModule = (HMODULE)hM; 
}
void set_module_restypename(const char * restypename) 
{
	if(HIWORD(restypename))
	{
		resource.res_type_name = restypename;
		resource.res_type_id = 0;
	}
	else
	{
		//resource.res_type_name.clear(); should exist (cf STL Doc)
		resource.res_type_id = (unsigned long)restypename;
	}
}
#endif
data_path get_text_path() { return path; }
void      set_text_path(const data_path & newpath) { path = newpath; }
/**
	
 **/
char * read_text_file(const char * filename)
{
	if(path.path.size() < 1)
	{
		path.path.push_back(".");
		path.path.push_back("../../../MEDIA/programs");
		path.path.push_back("../../../../MEDIA/programs");
		path.path.push_back("../../../../../../../MEDIA/programs");
	}

    if (!filename) return 0;
#ifdef WIN32
    struct _stat f_stat;
#else
    struct stat f_stat;
#endif
    if (path.fstat(filename, &f_stat))
	{
		long size = f_stat.st_size;

		char * buf = new char[size+1];

		FILE *fp = 0;
		if (!(fp = path.fopen(filename, "r")))
		{
			fprintf(stderr,"Cannot open \"%s\" for read!\n", filename);
			return 0;
		}

		int bytes;
		bytes = fread(buf, 1, size, fp);

		buf[bytes] = 0;

		fclose(fp);
		return buf;
	}
	fprintf(stderr,"Cannot open \"%s\" for stat read!\n", filename);
	return 0;
}
#ifdef WIN32
/**
	Add the resource reading 
 **/
char * read_text_file(const char * filename, const char * type, unsigned long hModule)
{
	if(hModule) set_module_handle(hModule);
	if(type)	set_module_restypename(type);
	char *buf = read_text_file(filename);
	if(buf)
		return buf;
	else
	{
		BOOL bRes;
		fprintf(stderr,"Trying resource...\n");
		HRSRC hr = FindResource(resource.hModule, filename, 
			resource.res_type_id ? (LPCSTR)resource.res_type_id : resource.res_type_name.c_str());
		if(hr)
		{
			HGLOBAL hg = LoadResource(resource.hModule, hr);
			if(hg)
			{
				DWORD sz = SizeofResource(resource.hModule, hr);
				LPCSTR tmpstr = (LPCSTR)LockResource(hg);
				char * buf = new char[sz+1];
				strncpy(buf, tmpstr, sz);
				buf[sz] = 0;
				bRes = FreeResource(hg);
				return buf;
			}
		}
	}
	fprintf(stderr,"Cannot find \"%s\" into the resource...\n", filename);
	return 0;
}
#else
char * read_text_file(const char * filename, const char * type, unsigned long hModule)
{
    return read_text_file(filename);
}
#endif
