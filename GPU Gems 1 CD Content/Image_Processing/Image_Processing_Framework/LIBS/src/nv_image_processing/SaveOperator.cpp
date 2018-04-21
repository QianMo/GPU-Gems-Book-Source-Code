// -----------------------------------------------------------------------------
//
// Contents:
//      class SaveOperator
//
// Description:
//      The SaveOperator saves DDS images to disk. 
//
// Author:
//      Frank Jargstorff (08/27/03)
//
// Note:
//      The operator is not yet implemented.
//
// -----------------------------------------------------------------------------


//
// Includes
//

#include "SaveOperator.h"

#include "SourceOperator.h"
#include "AssertGL.h"
#include <nv_dds/nv_dds.h>

#include <assert.h>
#include <iostream>


//
// Namespaces
//

using namespace nv_dds;


// -----------------------------------------------------------------------------
// SaveOperator class
//


    //
    // Construction and destruction
    //

        // Default constructor
        //
        // Description:
        //
SaveOperator::SaveOperator()
{
}


    //
    // Public methods
    //

        // save
        //
        // Description:
        //      Get the image from the pipeline and save it to disk.
        //
        // Parameters:
        //      sFileName - a string containing the filename.
        //
        // Returns:
        //      None        
        //
        void    
SaveOperator::save(std::string sFileName)
{
    Image oInputImage = _pSourceOperator->image();;

    unsigned int nWidth  = static_cast<unsigned int>(oInputImage.width() );
    unsigned int nHeight = static_cast<unsigned int>(oInputImage.height());

    unsigned char * pPixels = new unsigned char[nWidth*nHeight*4];

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, oInputImage.textureID());
    glGetTexImage(GL_TEXTURE_RECTANGLE_NV, 0, GL_BGRA, GL_UNSIGNED_BYTE, pPixels);

    CTexture  oTexture(nWidth, nHeight, 1, nWidth*nHeight*4, pPixels);
    CDDSImage oImage;
    
    oImage.create_textureFlat(GL_BGRA, 4, oTexture);
                                // Save without flipping (second parameter false)
    oImage.save(sFileName, false);
}

        // update
        //
        // Description:
        //      Makes the view get a new image from the pipeline.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None        
        //
        void    
SaveOperator::update()
{
    if (_pSourceOperator)
    {   
        _oImage = _pSourceOperator->image();
        GL_ASSERT_NO_ERROR;
    }
}
