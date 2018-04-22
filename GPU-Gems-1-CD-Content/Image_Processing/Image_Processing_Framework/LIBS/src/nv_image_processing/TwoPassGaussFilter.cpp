// -----------------------------------------------------------------------------
// 
// Contents:
//      TwoPassGaussFilter class
//
// Description:
//      A simple Gaussian blurr filter.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "TwoPassGaussFilter.h"

#include <iostream>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "AssertGL.h"
#include "AssertCG.h"


// -----------------------------------------------------------------------------
// TwoPassGaussFilter implementation
//


    // 
    // Construction and destruction
    //

        // Default constructor
        //
TwoPassGaussFilter::TwoPassGaussFilter()
{
    _oHorizontalFilter.setOrientation(GaussFilter1D::HORIZONTAL_FILTER);
    _oVerticalFilter.setOrientation(GaussFilter1D::VERTICAL_FILTER);
    
    _oHorizontalFilter.setSourceOperator(&_oVerticalFilter);
    
    setSigma(1.0f);
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
TwoPassGaussFilter::sigma()
        const
{
    return _oVerticalFilter.sigma();
}

        // setSigma
        //
        // Description:
        //      Set a new standard deviation sigma.
        //
        // Parameters:
        //      nSigma - the new sigma value.
        //
        // Returns:
        //      None
        //
        void
TwoPassGaussFilter::setSigma(float nSigma)
{
    _oVerticalFilter.setSigma(nSigma);
    _oHorizontalFilter.setSigma(nSigma);
    
    _bDirty = true;
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
TwoPassGaussFilter::setSourceOperator(SourceOperator * pSourceOperator)
{
    ImageFilter::setSourceOperator(pSourceOperator);
    _oVerticalFilter.setSourceOperator(_pSourceOperator);
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
TwoPassGaussFilter::image()
{
    _bDirty = false;
    
    return _oHorizontalFilter.image();
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
TwoPassGaussFilter::setCgParameters()
{
    CG_ASSERT_NO_ERROR;
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
TwoPassGaussFilter::cgFragmentProfile()
        const
{
    return CG_PROFILE_FP30;
}
