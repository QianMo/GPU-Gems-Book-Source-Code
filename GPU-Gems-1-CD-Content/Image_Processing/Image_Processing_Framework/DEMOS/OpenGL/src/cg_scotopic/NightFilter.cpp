// -----------------------------------------------------------------------------
// 
// Contents:
//      NightFilter class
//
// Description:
//      This image filter basically implements what is also called "Hollywood
//      Night". This filter is part of a more advance "night-scene filter" 
//      described in "A Spatial Post-Processing Algorithm for Images of Night
//      Scenes" by William B. Thompson et al. It basically simulates how human
//      perception changes in weakly lit scenes; mainly the fact that blue
//      objects appear brighter compared to yellowish objects with decreasing
//      light intensity.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "NightFilter.h"

#include <iostream>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <nv_image_processing/ShaderManager.h>
#include <nv_image_processing/AssertGL.h>
#include <nv_image_processing/AssertCG.h>


// -----------------------------------------------------------------------------
// NightFilter class
//

    //
    // Private static data initialization
    //

CGprogram   NightFilter::_goNightShader     = 0;

CGparameter NightFilter::_gInputImage       = 0;
CGparameter NightFilter::_gColorTransform   = 0;
CGparameter NightFilter::_gBlue             = 0;
CGparameter NightFilter::_gIntensity        = 0;


// 
// Construction and destruction
//

        // Default constructor
        //
NightFilter::NightFilter()
{
                                // Initialize data items.
    _mColorTransform[0][0] = 0.5149f;    _mColorTransform[0][1] = 0.3244f;    _mColorTransform[0][2] = 0.1607f;     _mColorTransform[0][3] = 0.0f; 
    _mColorTransform[1][0] = 0.2654f;    _mColorTransform[1][1] = 0.6704f;    _mColorTransform[1][2] = 0.0642f;     _mColorTransform[1][3] = 0.0f; 
    _mColorTransform[2][0] = 0.0248f;    _mColorTransform[2][1] = 0.1248f;    _mColorTransform[2][2] = 0.8504f;     _mColorTransform[2][3] = 0.0f; 
    _mColorTransform[3][0] = 0.0000f;    _mColorTransform[3][1] = 0.0000f;    _mColorTransform[3][2] = 0.0000f;     _mColorTransform[3][3] = 0.0f; 

    _vBlue[0] = .62f;   _vBlue[1] = 0.6f;   _vBlue[2] = 1.0f;
    
    _nBrightness = 0.13f;  

    if (_goNightShader == 0)
    {
                                // Set up the fragment program
        _goNightShader = cgCreateProgramFromFile(ShaderManager::gCgContext, CG_SOURCE, 
                               "../../../../MEDIA/programs/cg_scotopic/Night.cg", 
                               cgFragmentProfile(), 0, 0
                             );
        CG_ASSERT_NO_ERROR;

        cgGLLoadProgram(_goNightShader);
        glGetError();
        CG_ASSERT_NO_ERROR;
    }
                                // Grab the necessary parameters
    if (_gInputImage == 0)
        _gInputImage = cgGetNamedParameter(_goNightShader, "oImage");
    if (_gColorTransform == 0)
        _gColorTransform = cgGetNamedParameter(_goNightShader, "mColorTransform");
    if (_gBlue == 0)
        _gBlue = cgGetNamedParameter(_goNightShader, "vBlue");
    if (_gIntensity == 0)
        _gIntensity = cgGetNamedParameter(_goNightShader, "nIntensity");

                                // Set fragment program.
    _oCgFragmentProgram = _goNightShader;
                                // Set the input image parameter
    _hoInputImage = _gInputImage;
    GL_ASSERT_NO_ERROR;
}



//
// Public methods
//

        // brightness
        //
        float
NightFilter::brightness()
        const
{
    return _nBrightness;
}

        // setBrightness
        //
        void
NightFilter::setBrightness(float nBrightness)
{
    if (nBrightness < 0.0f)
        _nBrightness = 0.0f;
    else
        _nBrightness = nBrightness;
    
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
NightFilter::setCgParameters()
{
    cgGLSetMatrixParameterfr(_gColorTransform, &(_mColorTransform[0][0]));
    cgGLSetParameter3fv(_gBlue, &(_vBlue[0]));
    cgGLSetParameter1f(_gIntensity, _nBrightness);
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
NightFilter::cgFragmentProfile()
        const
{
    return CG_PROFILE_FP30;
}
