// -----------------------------------------------------------------------------
// 
// Contents:
//      GaussFilter1D class
//
// Description:
//      A simple one-dimensional Gaussian blurr filter.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "GaussFilter1D.h"

#include <iostream>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "ShaderManager.h"

#include <math.h>
#include <limits>
#include <assert.h>

#include "AssertGL.h"
#include "AssertCG.h"


// -----------------------------------------------------------------------------
// GaussFilter1D implementation
//


    //
    // Private static data
    //

CGprogram GaussFilter1D::_gaFragmentProgram[2] = {0, 0};


    // 
    // Construction and destruction
    //

        // Default constructor
        //
GaussFilter1D::GaussFilter1D(): _nSigma(1.0f)
                              , _eOrientation(HORIZONTAL_FILTER)
                              , _hFilterTexture(0)
                              , _hKernel(0)
{
    if (_gaFragmentProgram[0] == 0)
    {
                                    // Create argument array and terminate with a null pointer
        const char * aArguments[3];
        aArguments[2] = 0x00000000;

                                    // Two strings to write the respective parameters to.
        char zOrientation[128];
        char zMaxRadius[128];
        
        sprintf(zOrientation, "-DE_VERTICAL");
        aArguments[0] = zOrientation;
        sprintf(zMaxRadius, "-DN_MAX_RADIUS=%i", N_MAX_RADIUS);
        aArguments[1] = zMaxRadius;
                                    // Set up the fragment program
        _gaFragmentProgram[VERTICAL_FILTER] = cgCreateProgramFromFile(ShaderManager::gCgContext, CG_SOURCE, 
                                                                CG_SOURCE_PATH "Gauss1D.cg",
                                                                cgFragmentProfile(), 0, &aArguments[0]);
        CG_ASSERT_NO_ERROR;
        cgGLLoadProgram(_gaFragmentProgram[VERTICAL_FILTER]);
        CG_ASSERT_NO_ERROR;
        _oCgFragmentProgram = _gaFragmentProgram[VERTICAL_FILTER];
        
        sprintf(zOrientation, "-DE_HORIZONTAL");
        aArguments[0] = zOrientation;
                                    // Set up the fragment program
        _gaFragmentProgram[HORIZONTAL_FILTER] = cgCreateProgramFromFile(ShaderManager::gCgContext, CG_SOURCE, 
                                                                CG_SOURCE_PATH "Gauss1D.cg",
                                                                cgFragmentProfile(), 0, &aArguments[0]);
        CG_ASSERT_NO_ERROR;
        cgGLLoadProgram(_gaFragmentProgram[HORIZONTAL_FILTER]);
        CG_ASSERT_NO_ERROR;
    }

    glGenTextures(1, &_hFilterTexture);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hFilterTexture);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GL_ASSERT_NO_ERROR;

                                // Create the texture. For the time
                                // being the texture is filled with garbage.
                                // The call to setSigma below fixes this.
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA16_NV,
                 N_MAX_RADIUS + 1, 1, 0, 
                 GL_RGBA, GL_FLOAT, &_aFilterKernel[0]);
    GL_ASSERT_NO_ERROR;
    
    setOrientation(_eOrientation);
    setSigma(_nSigma);
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
GaussFilter1D::sigma()
        const
{
    return _nSigma;
}

        // setSigma
        //
        // Description:
        //      Set a new standard deviation sigma.
        //          This will recalculate the filter
        //      kernel.
        //
        // Parameters:
        //      nSigma - the new sigma value.
        //
        // Returns:
        //      None
        //
        void
GaussFilter1D::setSigma(float nSigma)
{
    assert(nSigma >= 0.0f);
    _nSigma = nSigma;
    recalculateKernel(_nSigma);

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hFilterTexture);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 
                    N_MAX_RADIUS + 1, 1, GL_RGBA, GL_FLOAT, &_aFilterKernel[0]);
    GL_ASSERT_NO_ERROR;
    
    _bDirty = true;
}

        // orientation
        //
        // Description:
        //      Determines if the filter is applied in horizontally or vertically.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      Enum value indicating the orientation.
        //
        GaussFilter1D::teOrientation
GaussFilter1D::orientation()
        const
{
    return _eOrientation;
}
        
        // setOrientation
        //
        // Description:
        //      Sets if the filter is applied horizontally or vertically.
        //
        // Parameters:
        //      eOrientation - the orientation.
        //
        // Returns:
        //      None
        //
        void
GaussFilter1D::setOrientation(GaussFilter1D::teOrientation eOrientation)
{
    _eOrientation = eOrientation;
    _oCgFragmentProgram = _gaFragmentProgram[_eOrientation];
    
                                // query and set the parameter handles for
                                // the new program.
    _hoInputImage       = cgGetNamedParameter(_oCgFragmentProgram, "oImage");
    CG_ASSERT_NO_ERROR;
    _hKernel            = cgGetNamedParameter(_oCgFragmentProgram, "oKernel");
    
    _bDirty = true;
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
GaussFilter1D::setCgParameters()
{
    cgGLSetTextureParameter(_hKernel, _hFilterTexture);
    cgGLEnableTextureParameter(_hKernel);
    CG_ASSERT_NO_ERROR;
}


            // recalculateKernel
            //
            void
GaussFilter1D::recalculateKernel(float nSigma)
{
    int iX;

    if (nSigma <= std::numeric_limits<float>::epsilon())
    {
                                // In the fragment shader we tap the center
                                // point twice for simplicity. Thus we use
                                // half the weight one would expect.
        _aFilterKernel[0][0] = 0.5f;
        _aFilterKernel[0][1] = 0.5f;
        _aFilterKernel[0][2] = 0.5f;
        _aFilterKernel[0][3] = 0.5f;
        
        for (iX = 1; iX < N_MAX_RADIUS + 1; iX++)
        {
            _aFilterKernel[iX][0] = 0.0f;
            _aFilterKernel[iX][1] = 0.0f;
            _aFilterKernel[iX][2] = 0.0f;
            _aFilterKernel[iX][3] = 0.0f;
         }
    }
    else
    {
        float nDivByTwoSigmaSquare = 1.0/(2*nSigma*nSigma);
        float nScale = 0.0f;
 
        for (iX = 0; iX < N_MAX_RADIUS + 1; iX++)
        {
            float nValue = expf(-(iX*iX)*nDivByTwoSigmaSquare);

            _aFilterKernel[iX][0] = nValue;
            _aFilterKernel[iX][1] = nValue;
            _aFilterKernel[iX][2] = nValue;
            _aFilterKernel[iX][3] = nValue;
            nScale               += nValue;
        }

        nScale = 0.5f/nScale;

        for (iX = 0; iX < N_MAX_RADIUS + 1; iX++)
        {
            _aFilterKernel[iX][0] *= nScale;
            _aFilterKernel[iX][1] *= nScale;
            _aFilterKernel[iX][2] *= nScale;
            _aFilterKernel[iX][3] *= nScale;
        }
    }
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
GaussFilter1D::cgFragmentProfile()
        const
{
    return CG_PROFILE_FP30;
}
