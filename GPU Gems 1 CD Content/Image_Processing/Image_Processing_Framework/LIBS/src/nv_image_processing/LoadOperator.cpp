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
// Includes
//

#include "LoadOperator.h"
#include <nv_dds/nv_dds.h>

#include <GL/glext.h>

#include "AssertGL.h"


//
// Namespaces
//

using namespace nv_dds;


// -----------------------------------------------------------------------------
// LoadOperator implementation
//


// 
// Construction and destruction
//

        // Default constructor
        //
LoadOperator::LoadOperator(): _bDirty(false)
                            , _sFileName("")
                            , _eFileType(LoadOperator::DDS_FILE)
{
}


//
// Public methods
//

        // setFilename
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
LoadOperator::setFileName(std::string sFileName)
{
    _sFileName = sFileName;
    _bDirty = true;
}

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
LoadOperator::setFileType(LoadOperator::teFileType eFileType)
{
    _eFileType = eFileType;
    _bDirty = true;
}

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
        LoadOperator::teFileType
LoadOperator::fileType()
        const
{
    return _eFileType;
}

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
        bool
LoadOperator::dirty()
{
    return _bDirty;
}

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
        Image
LoadOperator::image()
{
    if (_bDirty)
    {
        if (!loadImage(&_oImage))
        {
            std::cerr << "Couldn't load image." << std::endl;
            assert(false);
        }
        
        _bDirty = false;
    }

    return _oImage;
}

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
LoadOperator::loadImage(Image * pImage)
{
    switch (_eFileType)
    {
        case DDS_FILE:
        {
            CDDSImage oPicture;

            if (!oPicture.load(_sFileName, false))
                return false;

            int nWidth  = oPicture.get_width();
            int nHeight = oPicture.get_height();

            pImage->setSize(nWidth, nHeight);
            GL_ASSERT_NO_ERROR;
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, pImage->textureID());
            GL_ASSERT_NO_ERROR;
            bool bSuccess = oPicture.upload_textureRectangle();
            GL_ASSERT_NO_ERROR;
            
            return bSuccess;
        }
        break;
    }

    return true;
}
