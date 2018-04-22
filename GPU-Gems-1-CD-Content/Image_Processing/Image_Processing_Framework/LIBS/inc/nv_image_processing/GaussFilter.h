#ifndef GAUSS_FILTER_H
#define GAUSS_FILTER_H
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

#include "ImageFilter.h"

#include "wglPBufferARB.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>


//
// Constants
//

#define N_MAX_RADIUS 7


// -----------------------------------------------------------------------------
// GaussFilter class
//
class GaussFilter: public ImageFilter
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    GaussFilter();

            // Destructor
            //
            virtual
   ~GaussFilter()
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
    recalculateKernel(float nSigma, int nKernelRadius);


            // optimalKernelRadius
            //
            // Description:
            //      Calculates a filter kernel radius so that the
            //      error at the borders < 10%.
            //
            // Parameters:
            //      nSigme - the standard deviation of the Gaussian
            //          bell the filter aproximates.
            //
            // Returns:
            //      An int with the radius.
            // 
            int
    optimalKernelRadius(float nSigma)
            const;

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

    float       _nSigma;         // Standard deviation
    int         _nKernelRadius;
                                // The filter kernel
    GLuint      _hKernelTexture;
    float       _aKernel[(2*N_MAX_RADIUS + 1)*(2*N_MAX_RADIUS + 1)][4];  

    CGparameter _hKernel;


    //
    // Private static data
    // 

                                // An array with the 8 fragment programs for
                                // filter kernels from size 1x1 to 15x15
                                // (r=0,...,7).
    static CGprogram _gaFragmentProgram[N_MAX_RADIUS+1];
};

#endif // GAUSS_FILTER_H