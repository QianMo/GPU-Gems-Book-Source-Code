#ifndef TWO_PASS_GAUSS_FILTER_H
#define TWO_PASS_GAUSS_FILTER_H
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

#include "GaussFilter1D.h"


// -----------------------------------------------------------------------------
// TwoPassGaussFilter class
//
class TwoPassGaussFilter: public ImageFilter
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    TwoPassGaussFilter();

            // Destructor
            //
            virtual
   ~TwoPassGaussFilter()
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
    setSourceOperator(SourceOperator * pSourceOperator);

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
            virtual
            Image
    image();


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
    
    GaussFilter1D _oVerticalFilter;
    GaussFilter1D _oHorizontalFilter;
};

#endif // TWO_PASS_GAUSS_FILTER_H