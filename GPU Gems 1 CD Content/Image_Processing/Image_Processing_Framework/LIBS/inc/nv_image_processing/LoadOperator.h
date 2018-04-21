#ifndef IMAGE_LOAD_H
#define IMAGE_LAOD_H
// -----------------------------------------------------------------------------
// 
// Contents:
//      LoadOperator class
//
// Description:
//      LoadOperator loads DDS images from files.
//          The operator has some infrastructure to support additional image
//      formats in the future.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "SourceOperator.h"

#include <iostream>


// -----------------------------------------------------------------------------
// LoadOperator class
//
class LoadOperator: public SourceOperator
{
public:
    //
    // Public types
    //

    enum teFileType
    {
        DDS_FILE
    };


    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    LoadOperator();

            // Destructor
            //
            virtual
   ~LoadOperator() 
            {
                ; // empty
            }


    //
    // Public methods
    //

            // setFileName
            //
            // Description:
            //      Set the name of the image to be loaded.
            //
            // Parameters:
            //      sFileName - Name (and path) of the image to load.
            //
            // Returns:
            //      None
            //
            void
    setFileName(std::string sFileName);

            // setFileType
            //
            // Description:
            //      Set the image file type of the image to be loaded.
            //
            // Parameters:
            //      eFileType - One of the file types listed in the 
            //          file-type enum.
            //
            // Returns:
            //      None
            //
            // Note:
            //      The class's default filetype is DDS_FILE.
            //
            void
    setFileType(teFileType eFileType);

            // fileType
            //
            // Description:
            //      Returns the current file type.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      The current filetype.
            //
            teFileType
    fileType()
            const;

            // dirty
            //
            // Description:
            //      Has the state of the operator or any operators
            //      that this operator draws data from changed?
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      A boolean telling whether the operator is dirty.
            //
            virtual
            bool
    dirty();

            // image
            //
            // Description:
            //      Gets the operator's output image.
            //          This method will usually result in a complete
            //      reevaluation of the pipeline!
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      A new image.
            //
            virtual
            Image
    image();


            
private:
    //
    // Private methods
    //

            // loadImage
            //
            // Description:
            //      This is a helper method that loads the actual image.
            //
            // Parameters:
            //      None - the method gets its information from the class state.
            //
            // Returns:
            //      true  - on success,
            //      false - otherwise. 
            // 
            bool
    loadImage(Image * pImage);


private:
    // 
    // Private data
    //

    bool _bDirty;

    std::string _sFileName;
    teFileType  _eFileType;
    
    Image       _oImage;
};

#endif // IMAGE_LOAD_H