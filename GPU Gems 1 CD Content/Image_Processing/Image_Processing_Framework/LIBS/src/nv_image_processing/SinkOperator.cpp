// -----------------------------------------------------------------------------
// 
// Contents:
//      SinkOperator class
//
// Description:
//      The SinkOperator has a Source operator that it draws image data from.
//          Examples of sinks in the image processing pipeline are operators
//      that save images to files, or display them on the screen. Also all 
//      kinds of image filters are sinks since they consume data from a source.
//
// Author:
//      Frank Jargstorff (9/10/2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "SinkOperator.h"



// -----------------------------------------------------------------------------
// SinkOperator class
//
    // 
    // Construction and destruction
    //
    
        // Default constructor
        //
SinkOperator::SinkOperator(): _pSourceOperator(0)
                            , _bDirty(true)
{
};

        // Destructor
        //
SinkOperator::~SinkOperator() 
{
    ; // empty
}


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
        void
SinkOperator::setSourceOperator(SourceOperator * pSourceOperator)
{
    _pSourceOperator = pSourceOperator;
    _bDirty = true;
}

