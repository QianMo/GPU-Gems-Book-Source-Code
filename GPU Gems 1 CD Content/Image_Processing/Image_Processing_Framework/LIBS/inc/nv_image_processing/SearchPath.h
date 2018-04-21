#ifndef SEARCH_PATH_H
#define SEARCH_PATH_H
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

#include <string>
#include <vector>


// -----------------------------------------------------------------------------
// SearchPath class
//
class nvSearchPath
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    nvSearchPath();

            // Destructor
            //
            virtual
   ~nvSearchPath()    
            { ; };


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
            void
    addPath(std::string sPath);
    
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
            std::string
    findPath(std::string sFilename)
            const;
    

private:
    //
    // Private data
    //

    std::vector<std::string> _aPaths;
    
    //
    // Private data types
    //
    
    typedef std::vector<std::string>::const_iterator tiConstPath;
};

#endif // SEARCH_PATH_H