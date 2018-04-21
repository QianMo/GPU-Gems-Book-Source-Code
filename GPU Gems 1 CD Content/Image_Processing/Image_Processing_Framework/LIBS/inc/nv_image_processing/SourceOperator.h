#ifndef SOURCE_OPERATOR_H
#define SOURCE_OPERATOR_H
// -----------------------------------------------------------------------------
// 
// Contents:
//      SourceOperator class
//
// Description:
//      SourceOperator is for operators that can create an output image. 
//          The operator also has a method dirty() that tells if the operator's
//      last output is still valid.
//          For chains of operators dirty() should stick to the following 
//      policy:
//          - If any of the parameters of a source operator have changed that
//            would lead to different output the operator must return dirty.
//          - If the operator uses input images from other operators and any
//            one of these operators is dirty() the operator must return dirty.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include "Image.h"


// -----------------------------------------------------------------------------
// SourceOperator class
//
class SourceOperator
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    SourceOperator()
            {
            };

            // Destructor
            //
            virtual
   ~SourceOperator() 
            {
                ; // empty
            }


    //
    // Public methods
    //

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
    dirty() = 0;

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
    image() = 0;

};

#endif // SOURCE_OPERATOR_H