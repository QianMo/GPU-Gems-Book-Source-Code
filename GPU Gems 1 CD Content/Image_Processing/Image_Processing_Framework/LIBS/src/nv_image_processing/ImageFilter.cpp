// -----------------------------------------------------------------------------
// 
// Contents:
//      ImageFilter class
//
// Description:
//      ImageFilter is an abstract base class for image filters.
//          An image filter is an operator consuming image data and producing
//      transformed image data. It acts as a sink and source at the same time.
//          By deriving from SourceOperator and SinkOperator the class inherits
//      the image-handling interface and update mechanism.
//          The default image() method implementation uses a Cg fragment
//      program to do the actual image processing. In order to derive a new 
//      image filter all that needs to be done is to write a Cg program 
//      implementing the desired image processing algorithm, load the program 
//      in the classes constructor and overload the setCgParameters() method
//      to provide the Cg program with it's (uniform) input data.
//          If the Cg program uses parameters that a user of the class should
//      be able to specify then the writer of a filter operator also needs to 
//      implement the necessary setter- and getter methods for these
//      parameters.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "ImageFilter.h"

#include "ShaderManager.h"

#include "AssertCG.h"
#include "AssertGL.h"

#include <GL/glu.h>
#include <GL/glext.h>

#include <assert.h>


// -----------------------------------------------------------------------------
// ImageFilter implementation
//

        // Default constructor
        //
ImageFilter::ImageFilter(): _oCgFragmentProgram(0)
                          , _hoInputImage(0)
{
    GL_ASSERT_NO_ERROR;
}

        // Destructor
        //
ImageFilter::~ImageFilter()
{
    ;
}

        // setSourceOperator
        //
        // Description:
        //      Registers a source operator.
        //          The image operator is registered and 
        //      any call to the view's display method will render
        //      the operator's output image.
        //          To unregister an image use this function to set
        //      the image pointer to NULL.
        //      
        // Paramters:
        //      pImage - pointer to an image class object.
        //
        // Returns:
        //      None
        //
        void
ImageFilter::setSourceOperator(SourceOperator * pSourceOperator)
{
    SinkOperator::setSourceOperator(pSourceOperator);
    _bDirty = true;
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
ImageFilter::dirty()
{
    if (_bDirty) 
        return true;

    if (_pSourceOperator)
    {
        _bDirty = _pSourceOperator->dirty();
    }

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
ImageFilter::image()
{
    Image oOutputImage;

    if (_pSourceOperator != 0)
    {
        Image oInputImage  = _pSourceOperator->image();
        oOutputImage.setSize(oInputImage.width(), oInputImage.height());

        oOutputImage.renderBegin();
        {
                                // Set OpenGL state
            glViewport(0, 0, (GLsizei) oOutputImage.width(), (GLsizei) oOutputImage.height());
  
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_LIGHTING);       
            glDisable(GL_CULL_FACE);     
    
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluOrtho2D(0, oOutputImage.width(), 0, oOutputImage.height());
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

                                // Set Cg state
            cgGLEnableProfile(ShaderManager::gVertexIdentityProfile);
            cgGLBindProgram(ShaderManager::gVertexIdentityShader);
            CG_ASSERT_NO_ERROR;

            cgGLEnableProfile(cgFragmentProfile());
            cgGLBindProgram(_oCgFragmentProgram);
            CG_ASSERT_NO_ERROR;

                                // Set the model view matrix for the vertex program.
            cgGLSetStateMatrixParameter(ShaderManager::gVertexIdentityModelView, 
                                        CG_GL_MODELVIEW_PROJECTION_MATRIX,
                                        CG_GL_MATRIX_IDENTITY);

                                // If the following assertion fails it might
                                // be that the filter derived from ImageFilter
                                // did not correctly bind the _hoInputImage
                                // handle to the fragment program's input texture.
                                // This should usually be done in the derived
                                // class's constructor.
            assert(_hoInputImage != 0);
            cgGLSetTextureParameter(_hoInputImage, oInputImage.textureID());
            cgGLEnableTextureParameter(_hoInputImage);
            CG_ASSERT_NO_ERROR;

            setCgParameters();

            const float nWidth  = static_cast<float>(oInputImage.width());
            const float nHeight = static_cast<float>(oInputImage.height());

            glBegin(GL_QUADS);
                glTexCoord2f(0.0f,   0.0f);     glVertex3f(  0.0f, 0.0f,    0.0f);
                glTexCoord2f(nWidth, 0.0f);     glVertex3f(nWidth, 0.0f,    0.0f);
                glTexCoord2f(nWidth, nHeight);  glVertex3f(nWidth, nHeight, 0.0f);
                glTexCoord2f(0.0f,   nHeight);  glVertex3f(  0.0f, nHeight, 0.0f);
            glEnd();
            CG_ASSERT_NO_ERROR;

            cgGLDisableTextureParameter(_hoInputImage);
            cgGLDisableProfile(ShaderManager::gVertexIdentityProfile);
            cgGLDisableProfile(cgFragmentProfile());
            CG_ASSERT_NO_ERROR;
        }
        oOutputImage.renderEnd();

        _bDirty = false;
    }

    return oOutputImage;
}


