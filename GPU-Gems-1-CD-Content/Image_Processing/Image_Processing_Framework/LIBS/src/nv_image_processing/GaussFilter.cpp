// -----------------------------------------------------------------------------
// 
// Contents:
//      GaussFilter class
//
// Description:
//      A simple Gaussian blurr filter.
//          The implementation uses a texture to pass the filter kernel 
//      to the shader. In order to allow for different filter sizes the
//      operator creates several shader programs one for each filter size.
//      Based on the standard deviation Sigma the operator recalculates the 
//      filter kernel and switches to a fragment program with the correct
//      filter kernel size.
//          This implementation of a Gaussian filter is "naive" since it does
//      not make any use of the special properties of the Gaussian. Gaussian
//      kernels in 2D are "separable" with would allow for a two-pass
//      implementation using a one-dimensional filter kernel for each pass
//      which would result in a huge performance increase and would allow for 
//      much bigger filter sizes without running out of fragment shader 
//      instructions.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "GaussFilter.h"

#include <iostream>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "ShaderManager.h"

#include <math.h>
#include <limits>

#include "AssertGL.h"
#include "AssertCG.h"


// -----------------------------------------------------------------------------
// GaussFilter implementation
//


    //
    // Private static data
    //

CGprogram GaussFilter::_gaFragmentProgram[N_MAX_RADIUS+1] = {0};

    // 
    // Construction and destruction
    //

        // Default constructor
        //
GaussFilter::GaussFilter(): _nSigma(1.0f)
                          , _hKernelTexture(0)
{
    if (_gaFragmentProgram[0] == 0)
    {
                                    // Create argument array and terminate with a null pointer
        const char * aArguments[4];
        aArguments[3] = 0x00000000;

                                    // Two strings to write the respective parameters to.
        char zRadius[128];
        char zMaxRadius[128];
        char zElements[128];
        
        for (int nRadius = 0; nRadius <= N_MAX_RADIUS; ++nRadius)
        {                        
            sprintf(zRadius,    "-DN_RADIUS=%i", nRadius                           );
            sprintf(zMaxRadius, "-DN_MAX_RADIUS=%i", N_MAX_RADIUS                  );
            sprintf(zElements,  "-DN_ELEMENTS=%i", (2*nRadius + 1)*(2*nRadius + 1) );

            aArguments[0] = zRadius;
            aArguments[1] = zMaxRadius;
            aArguments[2] = zElements;

                                        // Set up the fragment program
            _gaFragmentProgram[nRadius] = cgCreateProgramFromFile(ShaderManager::gCgContext, CG_SOURCE, 
                                                                 CG_SOURCE_PATH "Gauss.cg",
                                                                 cgFragmentProfile(), 0, &aArguments[0]);
            CG_ASSERT_NO_ERROR;
            cgGLLoadProgram(_gaFragmentProgram[nRadius]);
            CG_ASSERT_NO_ERROR;
        }
    }

    glGenTextures(1, &_hKernelTexture);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hKernelTexture);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GL_ASSERT_NO_ERROR;

    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA16_NV,
                 (2*N_MAX_RADIUS + 1), (2*N_MAX_RADIUS + 1), 0, 
                 GL_RGBA, GL_FLOAT, &_aKernel[0][0]);
    GL_ASSERT_NO_ERROR;

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
GaussFilter::sigma()
        const
{
    return _nSigma;
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
GaussFilter::setSigma(float nSigma)
{
    if (nSigma < 0.0f)
        _nSigma = 0.0f;
    else
        _nSigma = nSigma;

    int nKernelRadius = optimalKernelRadius(_nSigma);
                                   // if the radius is greater than the maximum 
                                   // reset it to the maximum.
    if (nKernelRadius > N_MAX_RADIUS)
        nKernelRadius = N_MAX_RADIUS;

    recalculateKernel(_nSigma, nKernelRadius);

                                // Kernel radius changed
    if (nKernelRadius != _nKernelRadius)    
    {
                                // set new radius,
        _nKernelRadius      = nKernelRadius;
                                // set correct vertex program,
        _oCgFragmentProgram = _gaFragmentProgram[nKernelRadius];
                                // query and set the parameter handles for
                                // the new program.
        _hoInputImage       = cgGetNamedParameter(_oCgFragmentProgram, "oImage");
        CG_ASSERT_NO_ERROR;
        _hKernel            = cgGetNamedParameter(_oCgFragmentProgram, "oKernel");
        CG_ASSERT_NO_ERROR;
    }

    _bDirty = true;

    std::cerr << "Set sigma to " << _nSigma << ". New kernel radius: " << _nKernelRadius << std::endl;
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
GaussFilter::setCgParameters()
{
    cgGLSetTextureParameter(_hKernel, _hKernelTexture);
    cgGLEnableTextureParameter(_hKernel);
    CG_ASSERT_NO_ERROR;
}


            // recalculateKernel
            //
            void
GaussFilter::recalculateKernel(float nSigma, int nKernelRadius)
{
    float nScale = 0.0f;
    int iX, iY;
    int nSquareX, nSquareY;
    int nKernelDiameter = 2*nKernelRadius + 1;
    float nDivByTwoSigmaSquare = 1.0/(2*nSigma*nSigma);

    if (nSigma <= std::numeric_limits<float>::epsilon() && nKernelRadius == 0)
    {
        _aKernel[0][0] = 1.0f;
        _aKernel[0][1] = 1.0f;
        _aKernel[0][2] = 1.0f;
        _aKernel[0][3] = 1.0f;
    }
    else
    {
        for (iX=0; iX<nKernelDiameter; iX++)
            for (iY=0; iY<nKernelDiameter; iY++)
            {
                nSquareX = (nKernelRadius-iX)*(nKernelRadius-iX);
                nSquareY = (nKernelRadius-iY)*(nKernelRadius-iY);

                    _aKernel[iX*nKernelDiameter + iY][0] = expf(-(nSquareX + nSquareY)*nDivByTwoSigmaSquare);
                    _aKernel[iX*nKernelDiameter + iY][1] = expf(-(nSquareX + nSquareY)*nDivByTwoSigmaSquare);
                    _aKernel[iX*nKernelDiameter + iY][2] = expf(-(nSquareX + nSquareY)*nDivByTwoSigmaSquare);
                    _aKernel[iX*nKernelDiameter + iY][3] = expf(-(nSquareX + nSquareY)*nDivByTwoSigmaSquare);

                nScale += _aKernel[iX*nKernelDiameter + iY][0];
            }

        for (iX=0; iX<nKernelDiameter; iX++)
            for (iY=0; iY<nKernelDiameter; iY++)
            {
                _aKernel[iX*nKernelDiameter + iY][0] /= nScale;
                _aKernel[iX*nKernelDiameter + iY][1] /= nScale;
                _aKernel[iX*nKernelDiameter + iY][2] /= nScale;
                _aKernel[iX*nKernelDiameter + iY][3] /= nScale;
            }
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hKernelTexture);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, N_MAX_RADIUS - nKernelRadius, N_MAX_RADIUS - nKernelRadius, 
                    nKernelDiameter, nKernelDiameter, GL_RGBA, GL_FLOAT, &_aKernel[0][0]);
}

        // optimalKernelRadius
        //
        // Description:
        //      Calculates a filter kernel radius so that the
        //      error at the borders < 10%.
        //
        // Parameters:
        //      nSigma - the standard deviation of the Gaussian
        //          bell the filter aproximates.
        //
        // Returns:
        //      An int with the radius.
        // 
        // Note:
        //      Method assumes that nSigma >= 0. Otherwise results
        //      are undefined.
        //
        int
GaussFilter::optimalKernelRadius(float nSigma)
        const
{
                                // if sigma is zero return zero
    if (nSigma <= std::numeric_limits<float>::epsilon())
        return 0;

                                // else determin ideal radius
    int nRadius = ceil(2.146 * nSigma);
 
    return nRadius;
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
GaussFilter::cgFragmentProfile()
        const
{
    return CG_PROFILE_FP30;
}
