/*********************************************************************NVMH4****
Path:  SDK\LIBS\inc\shared
File:  nvfilesearch.h

Copyright NVIDIA Corporation 2002
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.



Comments:
    
	  File searching class, with wildcard support.  Inherit from it and delare
	  the FileFoundCallback virtual function.  This will get called each time
	  a matching file is found, with the file details and directory.

	Call:
	
	  Search->FindFile("*.*", "c:\", true);
	  to find every file on a user's hard disk.

  cmaughan@nvidia.com
      
        
******************************************************************************/
#ifndef H__NVFILESEARCH_H
#define H__NVFILESEARCH_H

//@@#include <shared/NVDX8Macros.h>

#include <string>

typedef std::basic_string<TCHAR> tstring; 

class NVFileSearch
{
public:
	NVFileSearch::NVFileSearch()
	{
		ZeroMemory(&m_FindData, sizeof(WIN32_FIND_DATA));
	}

	NVFileSearch::~NVFileSearch()
	{

	}

	virtual bool FileFoundCallback(const WIN32_FIND_DATA& FindData, const tstring strDirectory) = 0;

	virtual void FindFile(const tstring strFileString, const tstring strDirectoryStart, bool bRecurse)
	{
		tstring strDirectory;
		strDirectory.resize(MAX_PATH);
		DWORD dwNewSize = GetCurrentDirectory(MAX_PATH, &strDirectory[0]);
		strDirectory.resize(dwNewSize);
		GetCurrentDirectory(dwNewSize, &strDirectory[0]);

		SetCurrentDirectory(strDirectoryStart.c_str());

		WalkDirectory(strFileString, bRecurse);

		SetCurrentDirectory(strDirectory.c_str());
	}

protected:
	virtual void WalkDirectory(const tstring strFileString, bool bRecurse)
	{
		HANDLE hFind;

		m_bRecurse = bRecurse;

		tstring strDirectory;
		strDirectory.resize(MAX_PATH);
		DWORD dwNewSize = GetCurrentDirectory(MAX_PATH, &strDirectory[0]);
		strDirectory.resize(dwNewSize);
		GetCurrentDirectory(dwNewSize, &strDirectory[0]);

		hFind = FindFirstFile(strFileString.c_str(), &m_FindData);
		
		if (hFind == INVALID_HANDLE_VALUE)
			m_bOK = false;
		else
			m_bOK = true;

		while (m_bOK)
		{
			if (!(m_FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				// Grab the current directory, so that we can reset it after the callback.
				// Otherwise, if the callback changes the current directory in any way,
				// our search will abort prematurely.
				// Alternately, we could require the callback to not have any current
				// directory changes, but this is an easy fix, and performance is not
				// likely an issue.
				tstring strCurrentDirectory;
				strCurrentDirectory.resize(MAX_PATH);
				dwNewSize = GetCurrentDirectory(MAX_PATH, &strCurrentDirectory[0]);
				strCurrentDirectory.resize(dwNewSize);
				GetCurrentDirectory(dwNewSize, &strCurrentDirectory[0]);

				FileFoundCallback(m_FindData, strDirectory);

				// Reset the current directory to what it was before the callback.
				SetCurrentDirectory(strCurrentDirectory.c_str());
			}

			m_bOK = FindNextFile(hFind, &m_FindData);
		}

		if (hFind != INVALID_HANDLE_VALUE)
			FindClose(hFind);

		if (m_bRecurse)
		{
			hFind = FindFirstChildDir();

			if (hFind == INVALID_HANDLE_VALUE)
				m_bOK = false;
			else 
				m_bOK = true;

			while (m_bOK)
			{
				if (SetCurrentDirectory(m_FindData.cFileName))
				{
					WalkDirectory(strFileString, true);
				
					SetCurrentDirectory((TCHAR*)(".."));				
				}
				m_bOK = FindNextChildDir(hFind);
			}

			if (hFind != INVALID_HANDLE_VALUE)
				FindClose(hFind);
		}
	}

	virtual BOOL IsChildDir()
	{
		return ((m_FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && (m_FindData.cFileName[0] != '.'));
	}

	virtual BOOL FindNextChildDir(HANDLE hFindFile)
	{
		BOOL bFound = FALSE;
		do
		{
			bFound = FindNextFile(hFindFile, &m_FindData);
		} while (bFound && !IsChildDir());

		return bFound;
	}

	virtual HANDLE FindFirstChildDir()
	{
		BOOL bFound;
		HANDLE hFindFile = FindFirstFile((TCHAR*)("*.*"), &m_FindData);

		if (hFindFile != INVALID_HANDLE_VALUE)
		{
			bFound = IsChildDir();

			if (!bFound)
			{
				bFound = FindNextChildDir(hFindFile);
			}

			if (!bFound)
			{
				FindClose(hFindFile);
				hFindFile = INVALID_HANDLE_VALUE;
			}
		}

		return hFindFile;
	}

protected:
	BOOL m_bRecurse;
	BOOL m_bOK;
	BOOL m_bIsDir;
	WIN32_FIND_DATA m_FindData;
};

#endif
