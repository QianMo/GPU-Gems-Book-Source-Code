// -----------------------------------------------------------------------------
// 
// Contents:
//      ScotopicFilter class
//
// Description:
//      This fiter is the complete implemetation of an algorithm described in 
//      "A Spatial Post-Processing Algorithm for Images of Night Scenes" by 
//      William B. Thompson et al.
//          In contrast to simple filters that are build on the ImageFilter base
//      class in a straight forward fashion this filter actually contains
//      a little sub-filter graph. Looking at the constructor and the image()
//      function shows how this is done.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "ScotopicFilter.h"

#include <iostream>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <nv_image_processing/ShaderManager.h>
#include <nv_image_processing/AssertGL.h>
#include <nv_image_processing/AssertCG.h>

#include <math.h>


// -----------------------------------------------------------------------------
// ScotopicFilter implementation
//

    //
    // Private static data initialization
    //

CGprogram   ScotopicFilter::_goScotopicShader = 0;

CGparameter ScotopicFilter::_gInputImage      = 0;
CGparameter ScotopicFilter::_gBlurryImage     = 0;
CGparameter ScotopicFilter::_gBlurrierImage   = 0;
CGparameter ScotopicFilter::_gGamma           = 0;


// 
// Construction and destruction
//

        // Default constructor
        //
ScotopicFilter::ScotopicFilter(): _nGamma(1.25f)
{
    _oBlurFilter.setSourceOperator(&_oNightFilter);
    _oBlurrierFilter.setSourceOperator(&_oNightFilter);

    if (_goScotopicShader == 0)
    {
                                // Set up the fragment program
        _goScotopicShader = cgCreateProgramFromFile(ShaderManager::gCgContext, 
                CG_SOURCE, 
                "../../../../MEDIA/programs/cg_scotopic/Scotopic.cg",
                cgFragmentProfile(), 
                0, 
                0
             );
        CG_ASSERT_NO_ERROR;
        cgGLLoadProgram(_goScotopicShader);
        CG_ASSERT_NO_ERROR;
    }

                                // Grab the necessary parameters
    if (_gInputImage == 0)
        _gInputImage = cgGetNamedParameter(_goScotopicShader, "oImage");
    if (_gBlurryImage == 0)
        _gBlurryImage = cgGetNamedParameter( _goScotopicShader, "oBlurryImage");
    if (_gBlurrierImage == 0)
        _gBlurrierImage = cgGetNamedParameter( _goScotopicShader, "oBlurrierImage");
    if (_gGamma == 0)
        _gGamma = cgGetNamedParameter( _goScotopicShader, "nGamma");  
    CG_ASSERT_NO_ERROR;

                                // Set the fragment program.
    _oCgFragmentProgram = _goScotopicShader;
    _hoInputImage = _gInputImage;
}



//
// Public methods
//


        // sigma
        //
        // Description:
        //      Get the filter's standard deviation sigma.
        //
        // Parameters:
        //      None
        //
        // Return:
        //      The current standard deviation.
        //
        float
ScotopicFilter::sigma()
        const
{
    return _oBlurFilter.sigma();
}

        // setSigma
        //
        // Description:
        //      Set a new standard deviation sigma.
        //          This will recalculate the filter
        //      kernel and possibly even change the kernel's
        //      size. The maxium size is a 7x7 kernel. This
        //      means that sigmas greater 2 are not aproximated
        //      very well anymore.
        //
        // Parameters:
        //      nSigma - the new sigma value.
        //
        // Returns:
        //      None
        //
        void
ScotopicFilter::setSigma(float nSigma)
{
    _oBlurFilter.setSigma(nSigma);
    _oBlurrierFilter.setSigma(1.6f * nSigma);
}


        // gamma
        //
        // Description:
        //      Get the filter's sharpening factor gamma.
        //
        // Parameters:
        //      None
        //
        // Return:
        //      The current standard deviation.
        //
        float
ScotopicFilter::gamma()
        const
{
    return _nGamma;
}

        // setGamma
        //
        // Description:
        //      Set a new sharpening factor gamma.
        //          This will render the filter as dirty and
        //      cause a recomposition of the image at the next
        //      update.
        //
        // Parameters:
        //      nGamma - the new gamma value.
        //
        // Returns:
        //      None
        //
        void
ScotopicFilter::setGamma(float nGamma)
{
	if (nGamma < 0.0f) 
		_nGamma = 0.0f;
	else
		_nGamma = nGamma;

    _bDirty = true;
}

        // brightness
        //
        float
ScotopicFilter::brightness()
        const
{
    return _oNightFilter.brightness();
}

        // setBrighness
        //
        void
ScotopicFilter::setBrightness(float nBrightness)
{
    _oNightFilter.setBrightness(nBrightness);
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
ScotopicFilter::dirty()
{
    if (_bDirty) 
        return true;
    
    return _oBlurFilter.dirty();
}

        // setSourceOperator
        //
        // Description:
        //      Register the input image operator.
        //      The filter acts on the image it retrives
        //      from the input operator.
        //      
        // Paramters:
        //      pInputOperator - pointer to an image class object.
        //          To unregister an input operator set 
        //          pInputOperator to 0.
        //
        // Returns:
        //      None
        //
        void
ScotopicFilter::setSourceOperator(SourceOperator * pSourceOperator)
{
    ImageFilter::setSourceOperator(pSourceOperator);
    _oNightFilter.setSourceOperator(_pSourceOperator);
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
ScotopicFilter::image()
{
    Image oOutputImage;

    if (_pSourceOperator != 0)
    {
        Image oBlurryImage   = _oBlurFilter.image();
        Image oBlurrierImage = _oBlurrierFilter.image();

        oOutputImage.setSize(oBlurryImage.width(), oBlurryImage.height());

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

            cgGLEnableProfile(cgFragmentProfile());
            cgGLBindProgram(_oCgFragmentProgram);

            cgGLSetTextureParameter(_gBlurryImage, oBlurryImage.textureID());
            cgGLEnableTextureParameter(_gBlurryImage);
            cgGLSetTextureParameter(_gBlurrierImage, oBlurrierImage.textureID());
            cgGLEnableTextureParameter(_gBlurrierImage);

                                // Set the model view matrix for the vertex program.
            cgGLSetStateMatrixParameter(ShaderManager::gVertexIdentityModelView, 
                                        CG_GL_MODELVIEW_PROJECTION_MATRIX,
                                        CG_GL_MATRIX_IDENTITY);

            setCgParameters();

            const float nWidth  = static_cast<float>(oBlurryImage.width());
            const float nHeight = static_cast<float>(oBlurryImage.height());

            glBegin(GL_QUADS);
                glTexCoord2f(0.0f,   0.0f);     glVertex3f(  0.0f, 0.0f,    0.0f);
                glTexCoord2f(nWidth, 0.0f);     glVertex3f(nWidth, 0.0f,    0.0f);
                glTexCoord2f(nWidth, nHeight);  glVertex3f(nWidth, nHeight, 0.0f);
                glTexCoord2f(0.0f,   nHeight);  glVertex3f(  0.0f, nHeight, 0.0f);
            glEnd();

            cgGLDisableTextureParameter(_gBlurryImage);
            cgGLDisableTextureParameter(_gBlurrierImage);

            cgGLDisableProfile(CG_PROFILE_VP20);
            cgGLDisableProfile(cgFragmentProfile());
        }
        oOutputImage.renderEnd();

        _bDirty = false;
    }

    return oOutputImage;
}


//
// Protected methods
//

        // setCgParameters
        //
        // Description:
        //      This method is used in the image() method to 
        //      set provide the Cg programs with the correct
        //      parameters
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        void
ScotopicFilter::setCgParameters()
{
    cgGLSetParameter1f(_gGamma, _nGamma);
}

        // cgFragmentProfile
        //
        // Description:
        //      Get the fragment profile required for this filter's
        //      fragment program.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      CGprofile
        //
        CGprofile
ScotopicFilter::cgFragmentProfile()
        const
{
    return CG_PROFILE_FP30;
}
