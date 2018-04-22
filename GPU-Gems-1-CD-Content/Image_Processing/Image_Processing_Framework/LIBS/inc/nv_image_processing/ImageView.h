#ifndef IMAGE_VIEW_H
#define IMAGE_VIEW_H
// -----------------------------------------------------------------------------
//
// Contents:
//      class ImageView
//
// Description:
//      ImageWiew draws output from ImageOperators into OpenGL viewports. 
//          ImageView provides methods to register and unrester ImageOperators,
//      position and scale the output image, and to adapt to varying viewport 
//      sizes.
//          ImageView is part of a model-view-controller pattern. The model
//      being ImageOperator. Since ImageOperators are composits the actual
//      model is rather a complex hierarchy of interconnected ImageOperators
//      and ImageFilters than a single object.
//
// Author:
//      Frank Jargstorff (01/21/2003)
//
// -----------------------------------------------------------------------------


//
// Includes
//

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "SinkOperator.h"
#include "Image.h"



// -----------------------------------------------------------------------------
// ImageView class
//
class ImageView: public SinkOperator
{
public:
    //
    // Public static data
    //

            // aDefaultBackgroundColor
            //
            // Description:
            //      This constant contains the default background color.
            //      On contruction every ImageView instance will clear
            //      the frame buffer using this default color.
            //      The color is a dark gray (0.3 Luminance).
            //
            static 
            const 
    float aDefaultBackgroundColor[3];


public:
    //
    // Construction and destruction
    //

            // Default constructor
            //
            // Description:
            //
    ImageView();


    //
    // Public methods
    //

            // recenter
            //
            // Description:
            //      Recenters the image in the viewport for the next
            //      display cycle.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    center();

            // imagePositionX
            //
            // Description:
            //      Image posion on the drawing plane.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      The x-coordinate of the image's lower left corner
            //      within the view's window (with window's origin in the
            //      lower left corner, also).
            //
            int
    imagePositionX()
            const;

            // imagePositionY
            //
            // Description:
            //      Image posion on the drawing plane.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      The y-coordinate of the image's lower left corner
            //      within the view's window (with window's origin in the
            //      lower left corner, also).
            //
            int
    imagePositionY()
            const;

            // setImagePosition
            //
            // Description:
            //      Places the image at a new position.
            //
            // Parameters:
            //      nX - The new x-coordinate.
            //      nY - The new y-coordinate.
            //
            // Returns:
            //      None
            //
            void
    setImagePosition(int nX, int nY);


            // zoomFactor
            //
            // Description:
            //      Get the zoom factor.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      The zoom factor.
            //
            float
    zoomFactor()
            const;

            // setZoomFactor
            //
            // Description:
            //      Set the zoom factor.
            //
            // Parameters:
            //      nZoomFactor - the new zoom factor.
            //
            // Returns:
            //      None
            //
            void
    setZoomFactor(float nZoomFactor);

            // changeZoomFactor
            //
            // Description:
            //      Change zoom factor by a given percentage.
            //
            // Parametes:
            //      nPercent - the increase or decrease in percent
            //          of the original image size.
            //
            // Returns:
            //      None
            //
            void
    changeZoomFactor(float nPercent);

            // reshape
            //
            // Description:
            //      Reshape the view to reflect new vieport dimensions.
            //
            // Parameters:
            //      nViewportWidth  - the new width of the viewport.
            //      nViewportHeight - the new height of the viewport.
            //
            // Returns:
            //      None
            //
            void
    reshape(int nViewportWidth, int nViewportHeight);

            // display
            //
            // Description:
            //      Renders the registered image to the screen.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            void 
    display();

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

    int   _nViewportWidth;
    int   _nViewportHeight;

    float _nZoomFactor;

    int   _nImagePositionX;
    int   _nImagePositionY;

    bool  _bRecenter;

    float _nBrightness;

    float _aBackgroundColor[3];

    Image       _oImage;
                                // Cg related stuff
    CGprogram   _oCgFragmentProgram;
    CGprofile   _oCgFragmentProfile;

    CGparameter _hImage;
};


#endif // IMAGE_VIEW_H