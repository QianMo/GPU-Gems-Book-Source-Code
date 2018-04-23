
/* 
GetFilePath.h

A convenient way to search the NVSDK for media files
*/


#ifndef H_GETFILEPATH_H
#define H_GETFILEPATH_H

#include <windows.h>
#include <TCHAR.H>
#include <string>
#include "shared/NV_Common.h"
#include "shared/NV_Error.h"
#include "shared/nvFileSearch.h"

typedef std::basic_string<TCHAR> tstring; 

namespace GetFilePath
{
	static tstring strStartPath;

	typedef tstring (*GetFilePathFunction)(const tstring &, bool bVerbose );
	
	// This variable allows us to recursively search for media
	class NVMediaSearch : public NVFileSearch
	{
		public:
		tstring m_strPath;
		virtual bool FileFoundCallback(const WIN32_FIND_DATA& FindData, const tstring& strDir)
		{
			UNREFERENCED_PARAMETER(FindData);
			m_strPath = strDir;
			
			return false;
		}
	};


	static tstring GetModulePath() { return strStartPath; }
	static void        SetModulePath(const tstring &strPath)
	{
		tstring::size_type Pos = strPath.find_last_of(_T("\\"), strPath.size());
		if (Pos != strPath.npos)
			strStartPath = strPath.substr(0, Pos);
		else
			strStartPath = _T(".");
	}

	static void		SetDefaultModulePath()
	{
		DWORD ret;
		TCHAR buf[MAX_PATH];
		ret = GetModuleFileName( NULL, buf, MAX_PATH );		// get name for current module
		if( ret == 0 )
		{
			FMsg(TEXT("SetDefaultModulePath() GetModuleFileName() failed!\n"));
			assert( false );
		}

		SetModulePath( buf );
	}


	
	static tstring GetFullPath(const tstring &strRelativePath)
	{
		TCHAR buf[MAX_PATH];
		TCHAR *pAdr;
		GetFullPathName(strRelativePath.data(), MAX_PATH, buf, &pAdr);
		return buf;
	}

	//a helper class to save and restore the currentDirectory
	class DirectorySaver
	{
	private:
		TCHAR savedDirectory[MAX_PATH];
	public:
		DirectorySaver( )
		{
			// Save current directory
			GetCurrentDirectory(MAX_PATH, savedDirectory);
		}
		~DirectorySaver( )
		{
	  		// return to previous directory
	        SetCurrentDirectory(this->savedDirectory);
		}
	};

    // Recursively searchs the given path until it finds the file. Returns "" if 
    // file can't be found
    static tstring FindMediaFile(const tstring &strFilename, const tstring &mediaPath, bool bVerbose = false )
    {
        WIN32_FIND_DATA findInfo;
        HANDLE hFind;
        tstring result;
    
        
		//save and auto restore the current working directory
		DirectorySaver whenIGoOutOfScopeTheCurrentWorkingDirectoryWillBeRestored;
		
        if (!SetCurrentDirectory(mediaPath.data()))
		{
			// DWORD tmp2 = GetLastError();
			if( bVerbose )
			{	
				FMsg(TEXT("FindMediaFile Couldn't SetCurrentDirectory to [%s].  Returning empty string\n"), mediaPath.c_str() );
			}
            return _T("");
		}

        // check if file is in current directory
        FILE *fp;
        fp = _tfopen(strFilename.data(), _T("r"));
        if (fp)
        {
            fclose(fp);
            return mediaPath + _T("\\") + strFilename;
        }
		else
		{	
			if( bVerbose )
			{
				// report where the file is NOT
				FMsg(TEXT("FindMediaFile: File [%s] is not in %s\n"), strFilename.c_str(), mediaPath.c_str() );
			}
		}
    
        // if file not in current directory, search for all directories
        // and search inside them until the file is found
        hFind = FindFirstFile( _T( "*.*" ) , &findInfo);
        if (hFind == INVALID_HANDLE_VALUE)
            return _T("");
        
        result = _T("");
        do
        {
            if (findInfo.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            {
                // Make sure we don't try to delete the '.' or '..' directories
                if ((_tcscmp(findInfo.cFileName, _T("..")) == 0) || (_tcscmp(findInfo.cFileName, _T(".")) == 0))
                    continue;
                
				// Add directory to the media path
				// Keep same original file name
                result = FindMediaFile(strFilename, mediaPath + _T("\\") + findInfo.cFileName, bVerbose );
                if (result != _T(""))
                    break;                    
            }
        } while (FindNextFile(hFind, &findInfo));
    
        FindClose(hFind);
        
        return result;
    }

	//////////////////////////////////////////////////////////////////////////////
	// Gets the full path to the requested file by searching in the MEDIA directory.
    // (note: for a pointer to a 1-parameter version of this function, see below.)
	static tstring GetFilePath(const tstring& strFileName, bool bVerbose )
	{
        // Check if strFilename already points to a file
        FILE *fp;
        fp = _tfopen(strFileName.data(), _T("r"));
        if (fp)
        {
            fclose(fp);
            return strFileName;
        }

		// You should call SetModulePath before using GetFilePath()
		// If not, it will set the module path for you

		tstring strMediaPath = GetModulePath();
		if( strMediaPath.empty() == true )
		{
			SetDefaultModulePath();
			strMediaPath = GetModulePath();
		}

		// Find the last occurrence of '\' or '/'
		// This has the effect of backing up 4 directories, implying a direct dependence on
		//  the location of the calling .exe in order to start looking for media in the right
		//  place.  This is bad, so a more general search for the first subdirectory \MEDIA 
		//  should be put in place
		// TODO : see above
    	strMediaPath = strMediaPath.substr(0, strMediaPath.find_last_of(_T("\\/")));
	    strMediaPath = strMediaPath.substr(0, strMediaPath.find_last_of(_T("\\/")));
	    strMediaPath = strMediaPath.substr(0, strMediaPath.find_last_of(_T("\\/")));
	    strMediaPath = strMediaPath.substr(0, strMediaPath.find_last_of(_T("\\/")));
        strMediaPath += _T("\\MEDIA");
			
		tstring result;
        result = FindMediaFile(strFileName, strMediaPath, bVerbose);
           
        if (result != _T(""))
            return result;

        //////////////////// for local shaders /////////////////////////
		strMediaPath = GetModulePath();
        strMediaPath += _T("\\Shaders");

        result = FindMediaFile(strFileName, strMediaPath, bVerbose);
           
        if (result != _T(""))
            return result;

        //////////////////// for local ../shaders /////////////////////////
		strMediaPath = GetModulePath();
    	strMediaPath = strMediaPath.substr(0, strMediaPath.find_last_of(_T("\\/")));
        strMediaPath += _T("\\Shaders");

        result = FindMediaFile(strFileName, strMediaPath, bVerbose);
           
        if (result != _T(""))
            return result;

		// If prog gets to here, the find has failed.
		// Return the input file name so other apps can report the failure
		//  to find the file.
		if( bVerbose )
			FMsg(TEXT("GetFilePath() Couldn't find : %s\n"), strFileName.c_str() );

		return strFileName;
	};

    // Use these wrapper functions if you need to pass a pointer [callback] 
    // to a 1-parameter version of the GetFilePath() function:
    static tstring GetFilePath(const tstring& strFileName) {
        return GetFilePath(strFileName, false);
    }
	static tstring GetFilePathVerbose(const tstring& strFileName) {
        return GetFilePath(strFileName, true);
    }

};	 // namespace GetFilePath


#endif
