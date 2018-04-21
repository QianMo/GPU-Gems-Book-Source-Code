#ifndef GAUSS_FILTER_1D_H
#define GAUSS_FILTER_1D_H
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

#include "ImageFilter.h"

#include "wglPBufferARB.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>


//
// Constants
//

#define N_MAX_RADIUS 12


// -----------------------------------------------------------------------------
// GaussFilter1D class
//
class GaussFilter1D: public ImageFilter
{
public:
    //
    // Public datatypes
    //
    
    enum teOrientation
    {
        VERTICAL_FILTER,
        HORIZONTAL_FILTER
    };
    
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    GaussFilter1D();

            // Destructor
            //
            virtual
   ~GaussFilter1D()
            {
                ; // empty
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
    sigma()
            const;

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
    setSigma(float nSigma);
    
            // orientation
            //
            // Description:
            //      Determines if the filter is applied in horizontally or vertically.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      Enum value indication the orientation.
            //
            teOrientation
    orientation()
            const;
            
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
    setOrientation(teOrientation eOrientation);


protected:
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
            virtual
            void
    setCgParameters();

            // recalculateKernel
            //
            void
    recalculateKernel(float nSigma);

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
            virtual
            CGprofile
    cgFragmentProfile()
            const;

private:
    // 
    // Private data
    //

    float         _nSigma;         // Standard deviation
    teOrientation _eOrientation;
                                // The filter kernel
    GLuint        _hFilterTexture;
    float         _aFilterKernel[N_MAX_RADIUS + 1][4];  

    CGparameter   _hKernel;
    
    
    //
    // Private static data
    // 

                                // An array with the 2 fragment programs;
                                // one for vertical, one for horizontal filtering
                                //
    static CGprogram _gaFragmentProgram[2];
};

#endif // GAUSS_FILTER_1D_H