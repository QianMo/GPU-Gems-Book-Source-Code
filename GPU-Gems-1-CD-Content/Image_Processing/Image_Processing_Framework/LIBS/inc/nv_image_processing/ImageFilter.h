#ifndef IMAGE_FILTER_H
#define IMAGE_FILTER_H
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

#include "SinkOperator.h"
#include "SourceOperator.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>


//
// Constant defines
//

#define CG_SOURCE_PATH "../../../../MEDIA/programs/nv_image_processing/"



// -----------------------------------------------------------------------------
// ImageFilter class
//
class ImageFilter: public SinkOperator, public SourceOperator
{
public:
    //
    // Construction and destruction
    //

            // Default constructor
            //
    ImageFilter();

            // Destructor
            //
            virtual
   ~ImageFilter();

    // 
    // Public methods
    //

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
            virtual
            void
    setSourceOperator(SourceOperator * pSourceOperator);

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

            // setCgParameters
            //
            // Description:
            //      This method is used in the image() method to 
            //      provide the Cg programs with the correct
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
    setCgParameters() = 0;

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
            const = 0;


protected:
    //
    // Protected data
    //

                                // Handle to the fragment program used by the 
                                // image method.
    CGprogram   _oCgFragmentProgram;

                                // Handle to the fragment program's input image
                                // paramter. Every fragment program used by a 
                                // image filter must have a texture parameter
                                // to which the input image is being bound. 
                                // Usually the constructor of a derived image
                                // filter will bind this handle to the program's
                                // input texture parameter.
    CGparameter _hoInputImage;

    
    //
    // Friends
    //

    friend class ScotopicFilter;
    friend class ImageView;
};


#endif // IMAGE_FILTER_H