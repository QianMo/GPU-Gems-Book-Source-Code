// -----------------------------------------------------------------------------
// 
// Contents:
//      SearchPath class
//
// Description:
//      SearchPath is a little helper class for finding files.
//          It can search for a given file name in a number of different 
//      locations that were previously specified. If the 
//
// Author:
//      Frank Jargstorff (12/11/2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include <SearchPath.h>
#include <fstream>


// -----------------------------------------------------------------------------
// SearchPath class
//
    // 
    // Construction and destruction
    //
    
        // Default constructor
        //
nvSearchPath::nvSearchPath()
{
 
}


    //
    // Public methods
    //

        // addPath
        //
        // Description:
        //      Add a path where files should be searched for.
        //
        // Parameters:
        //      sPath - a new path.
        //
        // Returns:
        //      None
        //
        // Note:
        //      If the path does not end in a valid separator (e.g. '\'
        //      or '/') one is automatically appended.
        //
        void
nvSearchPath::addPath(std::string sPath)
{
    char nLastCharacter = *(sPath.end() - 1);
    if (nLastCharacter != '/' && nLastCharacter != '\\')
        sPath += "/";
    _aPaths.push_back(sPath);
}

        // findPath
        //
        // Description:
        //      Tries to locate a file in any of the registered locations.
        //
        // Parameters:
        //      sFilename - the file to look for.
        //
        // Returns:
        //      The path to the file if it was found or an empty string ""
        //      if the file is not in any of the registered locations.
        //
        // Note:
        //      Any returned path ends with a '/' character so it can be
        //      directly appended with the filename to yield a valid
        //      path.
        //
        std::string
nvSearchPath::findPath(std::string sFilename)
        const
{
    std::ifstream oFile;
    bool bFound = false;
    tiConstPath iPath;
    for (iPath = _aPaths.begin(); 
         iPath != _aPaths.end();
         ++iPath)
    {
        oFile.open(((*iPath) + sFilename).c_str(), std::ios_base::in);
        if (oFile.is_open())
            bFound = true;
        oFile.close();
        if (bFound)
            break;
    }
    
    if (bFound) 
        return *iPath;
 
    return "";
}
