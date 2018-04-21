#ifndef SINK_OPERATOR_H
#define SINK_OPERATOR_H
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


//
// Forward declarations
//

class SourceOperator;


// -----------------------------------------------------------------------------
// SinkOperator class
//
class SinkOperator
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    SinkOperator();

            // Destructor
            //
            virtual
   ~SinkOperator();


    //
    // Public methods
    //

            // setSourceOperator
            //
            // Description:
            //      Registers a source operator.
            //          To unregister a source operator use this function 
            //      to set the source operator to 0 (NULL).
            //      
            // Paramters:
            //      pSourceOperator - pointer to the source operator.
            //
            // Returns:
            //      None
            //
            virtual
            void
    setSourceOperator(SourceOperator * pSourceOperator);


protected:
    //
    // Protected data
    //

    SourceOperator * _pSourceOperator;

    bool _bDirty;
};

#endif // SINK_OPERATOR_H