#include <shared/data_path.h>
#ifdef WIN32
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <fcntl.h>

using namespace std;

string data_path::get_path(std::string filename)
{
    FILE* fp;
    bool found = false;
	for(unsigned int i=0; i < path.size(); i++)
	{
		path_name = path[i] + "/" + filename;
		fp = ::fopen(path_name.c_str(), "r");

		if(fp != 0)
        {     
            fclose(fp);
            found = true;
            break;
        }
	}

    if (found == false)
    {
        path_name = filename;
        fp = ::fopen(path_name.c_str(),"r"); // added by Ashu, for fully qualified path case.
	    if (fp != 0)
        {
            fclose(fp);
            found = true;
        }
    }

    if (found == false)
    {
        for(unsigned int i=0; i < path.size(); i++)
        {
            unsigned int pos; 
            string fname = filename;
            while ((pos = fname.find_first_of("\\/")) != string::npos)
            {
                pos++; 
                
                fname = fname.substr(pos, fname.length()-pos);
                path_name = path[i] + "/" + fname;
                fp = ::fopen(path_name.c_str(), "r");

                if(fp != 0)
                {     
                    fclose(fp);
                    found = true;
                    break;
                }
            }
            
            if (found) break;
        }
    }
    
    if (found == false)
        return "";

    int loc = path_name.rfind('\\');
    if (loc == -1)
    {
        loc = path_name.rfind('/');
    }

    if (loc != -1)
    	file_path = path_name.substr(0, loc);
    else
        file_path = ".";
    return file_path;
}

string data_path::get_file(std::string filename)
{
    FILE* fp;
    
    for(unsigned int i=0; i < path.size(); i++)
    {
        path_name = path[i] + "/" + filename;
        fp = ::fopen(path_name.c_str(), "r");

        if(fp != 0)
        {     
            fclose(fp);
            return path_name;
        }
    }
    
    path_name = filename;
    fp = ::fopen(path_name.c_str(),"r"); // added by Ashu, for fully qualified path case.
    if (fp != 0)
    {
        fclose(fp);
        return path_name;
    }

	for(i=0; i < path.size(); i++)
	{
		unsigned int pos; 
		string fname = filename;
		while ((pos = fname.find_first_of("\\/")) != string::npos)
		{
			pos++; 
			
			fname = fname.substr(pos, fname.length()-pos);
			path_name = path[i] + "/" + fname;
			fp = ::fopen(path_name.c_str(), "r");

			if(fp != 0)
			{     
				fclose(fp);
				return path_name;
			}
		}
	}

    printf("file not found: %s\n" ,filename.c_str());

    return "";
}

// data files, for read only
FILE * data_path::fopen(std::string filename, const char * mode)
{

	for(unsigned int i=0; i < path.size(); i++)
	{
		std::string s = path[i] + "/" + filename;
		FILE * fp = ::fopen(s.c_str(), mode);

		if(fp != 0)
			return fp;
		else if (!strcmp(path[i].c_str(),""))
		{
			FILE* fp = ::fopen(filename.c_str(),mode); // added by Ashu, for fully qualified path case.
			if (fp != 0)
				return fp;
		}
	}
	// no luck... return null
	return 0;
}

//  fill the file stats structure 
//  useful to get the file size and stuff
int data_path::fstat(std::string filename, 
#ifdef WIN32
		     struct _stat 
#else
		     struct stat
#endif
		     * stat)
{
	for(unsigned int i=0; i < path.size(); i++)
	{
		std::string s = path[i] + "/" + filename;
#ifdef WIN32
		int fh = ::_open(s.c_str(), _O_RDONLY);
#else
		int fh = ::open(s.c_str(), O_RDONLY);
#endif
		if(fh >= 0)
        {
#ifdef WIN32
            int result = ::_fstat( fh, stat );
#else
	    int result = ::fstat (fh,stat);
#endif
            if( result != 0 )
            {
                fprintf( stderr, "An fstat error occurred.\n" );
                return 0;
            }
#ifdef WIN32
            ::_close( fh );
#else
	    ::close (fh);
#endif
			return 1;
    	}
    }
    // no luck...
    return 0;
}
