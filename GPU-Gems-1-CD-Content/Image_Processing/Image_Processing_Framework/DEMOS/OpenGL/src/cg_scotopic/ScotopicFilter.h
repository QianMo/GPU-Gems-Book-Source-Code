#ifndef SCOTOPIC_FILTER_H
#define SCOTOPIC_FILTER_H
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

#include <nv_image_processing/ImageFilter.h>
#include <nv_image_processing/GaussFilter.h>

#include "NightFilter.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>



// -----------------------------------------------------------------------------
// ScotopicFilter class
//
class ScotopicFilter: public ImageFilter
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    ScotopicFilter();

            // Destructor
            //
            virtual
   ~ScotopicFilter()
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
    gamma()
            const;

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
    setGamma(float nGamma);

            // brightness
            //
            float
    brightness()
            const;

            // setBrighness
            //
            void
    setBrightness(float nBrightness);

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
            virtual
            void
    setSourceOperator(SourceOperator * pInputOperator);

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
            virtual
            bool
    dirty();

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

            // renderOutput
            //
            void
    renderOutput(Image * pOutputImage, const Image & rInputImage);

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

    float _nGamma;              // Sharpening factor.

    NightFilter _oNightFilter;
    GaussFilter _oBlurFilter;
    GaussFilter _oBlurrierFilter;


    //
    // Private static data
    //

    static CGprogram _goScotopicShader;

    static CGparameter _gInputImage;
    static CGparameter _gBlurryImage;
    static CGparameter _gBlurrierImage;
    static CGparameter _gGamma;
};

#endif // SCOTOPIC_FILTER_H