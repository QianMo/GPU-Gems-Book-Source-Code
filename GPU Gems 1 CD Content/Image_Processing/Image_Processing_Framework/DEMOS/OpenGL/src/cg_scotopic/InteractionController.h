#ifndef INTERACTION_CONTROLLER_H
#define INTERACTION_CONTROLLER_H
// -----------------------------------------------------------------------------
// 
// Contents:
//      InteractionController class
//
// Description:
//      The interaction controller provides a simple interface to setup
//      and reconfigure the demo application's image processing pipeline.
//      It also provides a set of methods that allow to easily and in a uniform
//      way forward GUI commands for tweaking parameters to the pipeline.
//          The InteractionController is part of a model-view-controller 
//      pattern. Other classes involved in this collaboration are 
//          - the ImageView
//          - hierarchies of ImageOperators
//      ImageView is the only implementation of a view in this MVC pattern.
//      The hierarchies of ImageOperators are the model.
//
// Author:
//      Frank Jargstorff (2003)
//
// -----------------------------------------------------------------------------



//
// Forward declarations
//

class LoadOperator;
class SaveOperator;
class GaussFilter;
class NightFilter;
class ScotopicFilter;
class GaussFilter1D;
class TwoPassGaussFilter;

class ImageView;

class ParamBase;
class ParamListGL;


// -----------------------------------------------------------------------------
// InteractionController class
//
class InteractionController
{
public:
    //
    // Public typdefs
    //

    enum tePipelineMode
    {
        DISPLAY_MODE,
        GAUSS_FILTER_MODE,
        NIGHT_FILTER_MODE,
        SCOTOPIC_FILTER_MODE,
        GAUSS_1D_FILTER_MODE,
        TWO_PASS_GAUSS_FILTER_MODE
    };


    // 
    // Construction and destruction
    //
    
            // Constructor
            //
    InteractionController(LoadOperator          & rLoadOperator,
                          GaussFilter           & rGaussFilter,
                          NightFilter           & rNightFilter,
                          ScotopicFilter        & rScotopicFilter,
                          GaussFilter1D         & rGaussFilter1D,
                          TwoPassGaussFilter    & rTwoPassGaussFilter,
                          SaveOperator          & rSaveOperator,
                          ImageView             & rImageView);

            // Destructor
            //
            virtual
   ~InteractionController();


    //
    // Public methods
    //

            // mouse
            //
            // Description:
            //      Handle mouse events
            //
            // Parameters:
            //      nX - x-coordinate where event happened.
            //      nY - y-coordinate where event happened.
            //
            // Returns:
            //      None
            //
            void
    mouse(int nX, int nY);
    
            // move
            //
            // Description:
            //      Handle mouse movements
            //
            // Parameters:
            //      nX - mouse position
            //      nY - mouse position
            //
            // Returns:
            //      None
            //
            void
    move(int nX, int nY);
            
            // special
            //
            // Description:
            //      Handle special keys
            //
            // Parameters:
            //      nKey - key code.
            //      nX - cursor position x.
            //      nY - cursor position y.
            //
            // Returns:
            //      None
            //
            void
    special(int key, int x, int y);
    
            // setPipelineMode
            //
            void
    setPipelineMode(tePipelineMode ePipelineMode);

            // save
            //
            // Description:
            //      Trigger a save operation on the save operator.
            //          All saves are done to the file Scotopic.dds
            //      that gets created in the current working director.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    save();

			// renderSliders
			//
			// Description:
			//		Renders the sliders controlling parameters.
			//
			// Parameters:
			//		None
			//
			// Returns:
			//		None
			//
			void
	renderSliders()
			const;


private:
    //
    // Private methods
    //

            // Default constructor (not implemented).
    InteractionController();

            // Copy constructor (not implemented).
    InteractionController(const InteractionController &);

            // Assignment operator (not implemented).
    operator= (const InteractionController &);
    
            // updateGaussParameters
            //
            // Description:
            //      Set the filter's parameters with slider values.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    updateGaussParameters();
    
            // updateNightParameters
            //
            // Description:
            //      Set the filter's parameters with slider values.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    updateNightParameters();
    
            // updateScotopicParameters
            //
            // Description:
            //      Set the filter's parameters with slider values.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    updateScotopicParameters();

            // updateGauss1dParameters
            //
            // Description:
            //      Set the filter's parameters with slider values.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    updateGauss1dParameters();
    
            // updateTwoPassGaussParameters
            //
            // Description:
            //      Set the filter's parameters with slider values.
            //
            // Parameters:
            //      None
            //
            // Returns:
            //      None
            //
            void
    updateTwoPassGaussParameters();
    

    //
    // Private data
    //

    tePipelineMode    _ePipelineMode;

    LoadOperator        * _pLoadOperator;

    GaussFilter         * _pGaussFilter;
    NightFilter         * _pNightFilter;
    ScotopicFilter      * _pScotopicFilter;
    GaussFilter1D       * _pGaussFilter1D;
    TwoPassGaussFilter  * _pTwoPassGaussFilter;

    ParamBase           * _pScotopicSigmaParameter;
    ParamBase           * _pScotopicGammaParameter;
    ParamBase           * _pScotopicBrightnessParameter;
	ParamListGL		    * _pScotopicSliders;
	
	ParamBase           * _pNightBrightnessParameter;
	ParamListGL         * _pNightSliders;
	
	ParamBase           * _pGaussSigmaParameter;
	ParamListGL         * _pGaussSliders;
	
	ParamBase           * _pGauss1dSigmaParameter;
	ParamListGL         * _pGauss1dSliders;
	
	ParamBase           * _pTwoPassGaussSigmaParameter;
	ParamListGL         * _pTwoPassGaussSliders;

    SaveOperator        * _pSaveOperator;
    ImageView           * _pImageView;

};

#endif // INTERACTION_CONTROLLER_H