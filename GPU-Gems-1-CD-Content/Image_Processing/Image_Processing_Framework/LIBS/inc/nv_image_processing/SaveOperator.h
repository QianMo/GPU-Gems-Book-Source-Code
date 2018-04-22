#ifndef SAVE_OPERATOR_H
#define SAVE_OPERATOR_H
// -----------------------------------------------------------------------------
//
// Contents:
//      class SaveOperator
//
// Description:
//      The SaveOperator saves DDS images to disk. 
//
// Author:
//      Frank Jargstorff (08/27/03)
//
// Note:
//      The operator is not yet implemented.
//
// -----------------------------------------------------------------------------


//
// Includes
//

#include "Image.h"
#include "SinkOperator.h"

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include <string>


//
// Forward declarations
//

class SourceOperator;


// -----------------------------------------------------------------------------
// SaveOperator class
//
class SaveOperator: public SinkOperator
{
public:
    //
    // Construction and destruction
    //

            // Default constructor
            //
            // Description:
            //
    SaveOperator();


    //
    // Public methods
    //

            // save
            //
            // Description:
            //      Get the image from the pipeline and save it to disk.
            //
            // Parameters:
            //      sFileName - a string containing the filename.
            //
            // Returns:
            //      None        
            //
            void    
    save(std::string sFileName);

            // update
            //
            // Description:
            //      Makes the view get a new image from the pipeline.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None        
            //
            void    
    update();


private:
    // 
    // Private data
    //

    Image            _oImage;
};


#endif // SAVE_OPERATOR_H