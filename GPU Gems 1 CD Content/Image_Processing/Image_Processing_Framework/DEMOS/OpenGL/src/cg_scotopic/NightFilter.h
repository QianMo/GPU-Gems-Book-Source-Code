#ifndef NIGHT_FILTER_H
#define NIGHT_FILTER_H
// -----------------------------------------------------------------------------
// 
// Contents:
//      NightFilter class
//
// Description:
//      This image filter basically implements what is also called "Hollywood
//      Night". This filter is part of a more advance "night-scene filter" 
//      described in "A Spatial Post-Processing Algorithm for Images of Night
//      Scenes" by William B. Thompson et al. It basically simulates how human
//      perception changes in weakly lit scenes; mainly the fact that blue
//      objects appear brighter compared to yellowish objects with decreasing
//      light intensity.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------


//
// Include
//

#include <nv_image_processing/ImageFilter.h>

#include <Cg/cg.h>
#include <Cg/cgGL.h>



// -----------------------------------------------------------------------------
// ImageLoad class
//
class NightFilter: public ImageFilter
{
public:
    // 
    // Construction and destruction
    //
    
            // Default constructor
            //
    NightFilter();

            // Destructor
            //
            virtual
   ~NightFilter()
            {
                ; // empty
            }


    //
    // Public methods
    //
    
            // brightness
            //
            float
    brightness()
            const;

            // setBrightness
            //
            void
    setBrightness(float nBrightness);
    

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

    float _mColorTransform[4][4];
    float _vBlue[3];
    float _nBrightness;


    //
    // Private static data
    //

                                // Handle to the fragment program
    static CGprogram    _goNightShader;

                                // Handles to the fragment program's parameters.
    static CGparameter  _gInputImage;
    static CGparameter  _gColorTransform;
    static CGparameter  _gBlue;
    static CGparameter  _gIntensity;
};

#endif // NIGHT_FILTER_H