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
// Include
//

#include <nv_image_processing/ImageView.h>
#include <nv_image_processing/LoadOperator.h>
#include <nv_image_processing/SaveOperator.h>
#include <nv_image_processing/GaussFilter.h>
#include <nv_image_processing/GaussFilter1D.h>
#include <nv_image_processing/TwoPassGaussFilter.h>

#include <paramgl/paramgl.h>

#include "InteractionController.h"
#include "Nightfilter.h"
#include "ScotopicFilter.h"

#include <iostream>


//
// Namespaces
//

using namespace std;


// 
// Constants
//

const float gnDeltaSigma = 0.1f;
const float gnDeltaGamma = 0.1f;
const float gnDeltaBrightness = 0.05f;


// -----------------------------------------------------------------------------
// InteractionController implementation
//


// 
// Construction and destruction
//

            // Constructor
            //
InteractionController::InteractionController(LoadOperator       & rLoadOperator,
                                             GaussFilter        & rGaussFilter,
                                             NightFilter        & rNightFilter,
                                             ScotopicFilter     & rScotopicFilter,
                                             GaussFilter1D      & rGaussFilter1D,
                                             TwoPassGaussFilter & rTwoPassGaussFilter,
                                             SaveOperator       & rSaveOperator,
                                             ImageView          & rImageView)
                : _ePipelineMode(InteractionController::DISPLAY_MODE)
                , _pLoadOperator(       &rLoadOperator      )
                , _pGaussFilter(        &rGaussFilter       )
                , _pNightFilter(        &rNightFilter       )
                , _pScotopicFilter(     &rScotopicFilter    )
                , _pGaussFilter1D(  &   rGaussFilter1D      )
                , _pTwoPassGaussFilter( &rTwoPassGaussFilter)
                , _pScotopicSigmaParameter(0)
                , _pScotopicGammaParameter(0)
                , _pScotopicBrightnessParameter(0)
				, _pScotopicSliders(0)
				, _pGauss1dSigmaParameter(0)
				, _pGauss1dSliders(0)
				, _pTwoPassGaussSigmaParameter(0)
				, _pTwoPassGaussSliders(0)
                , _pSaveOperator(   &rSaveOperator   )
                , _pImageView(      &rImageView      )
{
	_pGaussSliders                  = new ParamListGL("Gauss Filter Parameters");
	_pGaussSigmaParameter           = new Param<float>("Sigma", 0.25f, 0.0f, 10.0f, 0.1f);
    _pGaussSliders->AddParam(_pGaussSigmaParameter);
    
	_pNightSliders                  = new ParamListGL("Night Filter Parameters");
	_pNightBrightnessParameter      = new Param<float>("Brightness", 1.0f, 0.0f, 2.0f, 0.05f);
	_pNightSliders->AddParam(_pNightBrightnessParameter);

	_pScotopicSliders               = new ParamListGL("Scotopic Filter Parameters");
    _pScotopicSigmaParameter        = new Param<float>("Sigma",      0.25f, 0.0f, 10.0f, 0.1f);
    _pScotopicGammaParameter        = new Param<float>("Gamma",      1.0f,  0.0f, 10.0f, 0.1f);
    _pScotopicBrightnessParameter   = new Param<float>("Brightness", 1.0f,  0.0f,  2.0f, 0.05f);
 	_pScotopicSliders->AddParam(_pScotopicSigmaParameter);
	_pScotopicSliders->AddParam(_pScotopicGammaParameter);
	_pScotopicSliders->AddParam(_pScotopicBrightnessParameter);
	
	_pGauss1dSliders                = new ParamListGL("Gauss 1D Filter Parameters");
	_pGauss1dSigmaParameter         = new Param<float>("Sigma", 1.0f, 0.0f, 10.0f, 0.1f);
	_pGauss1dSliders->AddParam(_pGauss1dSigmaParameter);
	
	_pTwoPassGaussSliders           = new ParamListGL("2-pass Gauss Filter Parameters");
	_pTwoPassGaussSigmaParameter    = new Param<float>("Sigma", 1.0f, 0.0f, 10.0f, 0.1f);
	_pTwoPassGaussSliders->AddParam(_pTwoPassGaussSigmaParameter);
	


                                // Since the default pipeline has no
                                // operators we hook up the load to
                                // the view.
    _pImageView->setSourceOperator(_pLoadOperator);
    _pSaveOperator->setSourceOperator(_pLoadOperator);
    _pImageView->update();
    _pImageView->center();

                                // Now lets set up the other pipelines
                                // but not connect them to the view.
    _pGaussFilter->setSourceOperator(_pLoadOperator);
    _pNightFilter->setSourceOperator(_pLoadOperator);
    _pScotopicFilter->setSourceOperator(_pLoadOperator);
    _pGaussFilter1D->setSourceOperator(_pLoadOperator);
    _pTwoPassGaussFilter->setSourceOperator(_pLoadOperator);
}
    
    


        // Destructor
        //
InteractionController::~InteractionController() 
{
    ; // empty
}


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
InteractionController::mouse(int nX, int nY)
{
    switch (_ePipelineMode)
    {
        case GAUSS_FILTER_MODE:
            _pGaussSliders->Mouse(nX, nY);
            updateGaussParameters();
        break;
        case NIGHT_FILTER_MODE:
            _pNightSliders->Mouse(nX, nY);
            updateNightParameters();
        break;
        case SCOTOPIC_FILTER_MODE:
            _pScotopicSliders->Mouse(nX, nY);
            updateScotopicParameters();
        break;
        case GAUSS_1D_FILTER_MODE:
            _pGauss1dSliders->Mouse(nX, nY);
            updateGauss1dParameters();
        break;
        case TWO_PASS_GAUSS_FILTER_MODE:
            _pTwoPassGaussSliders->Mouse(nX, nY);
            updateTwoPassGaussParameters();
        break;

    }
}

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
InteractionController::special(int nKey, int nX, int nY)
{
    switch (_ePipelineMode)
    {
        case GAUSS_FILTER_MODE:
            _pGaussSliders->Special(nKey, nX, nY);
            updateGaussParameters();
        break;
        case NIGHT_FILTER_MODE:
            _pNightSliders->Special(nKey, nX, nY);
            updateNightParameters();
        break;
        case SCOTOPIC_FILTER_MODE:
            _pScotopicSliders->Special(nKey, nX, nY);
                updateScotopicParameters();
        break;
        case GAUSS_1D_FILTER_MODE:
            _pGauss1dSliders->Special(nKey, nX, nY);
            updateGauss1dParameters();
        break;
        case TWO_PASS_GAUSS_FILTER_MODE:
            _pTwoPassGaussSliders->Special(nKey, nX, nY);
            updateTwoPassGaussParameters();
        break;
    }
}

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
InteractionController::move(int nX, int nY)
{
    switch (_ePipelineMode)
    {
        case GAUSS_FILTER_MODE:
            _pGaussSliders->Motion(nX, nY);
            updateGaussParameters();
        break;
        case NIGHT_FILTER_MODE:
            _pNightSliders->Motion(nX, nY);
            updateNightParameters();
        break;
        case SCOTOPIC_FILTER_MODE:
            _pScotopicSliders->Motion(nX, nY);
            updateScotopicParameters();
        break;
        case GAUSS_1D_FILTER_MODE:
            _pGauss1dSliders->Motion(nX, nY);
            updateGauss1dParameters();
        break;
        case TWO_PASS_GAUSS_FILTER_MODE:
            _pTwoPassGaussSliders->Motion(nX, nY);
            updateTwoPassGaussParameters();
        break;
    }
}

        // setPiplineMode
        //
        void
InteractionController::setPipelineMode(InteractionController::tePipelineMode ePipelineMode)
{
    _ePipelineMode = ePipelineMode;

    switch (_ePipelineMode)
    {
        case DISPLAY_MODE:
            _pImageView->setSourceOperator(_pLoadOperator);
            _pSaveOperator->setSourceOperator(_pLoadOperator);
        break;

        case GAUSS_FILTER_MODE:
            _pImageView->setSourceOperator(_pGaussFilter);
            _pSaveOperator->setSourceOperator(_pGaussFilter);
        break;

        case NIGHT_FILTER_MODE:
            _pImageView->setSourceOperator(_pNightFilter);
            _pSaveOperator->setSourceOperator(_pNightFilter);
        break;

        case SCOTOPIC_FILTER_MODE:
            _pImageView->setSourceOperator(_pScotopicFilter);
            _pSaveOperator->setSourceOperator(_pScotopicFilter);
        break;
        
        case GAUSS_1D_FILTER_MODE:
            _pImageView->setSourceOperator(_pGaussFilter1D);
            _pSaveOperator->setSourceOperator(_pGaussFilter1D);
        break;
        case TWO_PASS_GAUSS_FILTER_MODE:
            _pImageView->setSourceOperator(_pTwoPassGaussFilter);
            _pSaveOperator->setSourceOperator(_pTwoPassGaussFilter);
        break;
    }
}

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
InteractionController::save()
{
    _pSaveOperator->save("Scotopic.dds");
}

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
InteractionController::renderSliders()
		const
{
    switch (_ePipelineMode)
    {
        case GAUSS_FILTER_MODE:
            _pGaussSliders->Render(0,0);
        break;
        case NIGHT_FILTER_MODE:
            _pNightSliders->Render(0,0);
        break;
        case SCOTOPIC_FILTER_MODE:
            _pScotopicSliders->Render(0,0);
        break;
        case GAUSS_1D_FILTER_MODE:
            _pGauss1dSliders->Render(0,0);
        break;
        case TWO_PASS_GAUSS_FILTER_MODE:
            _pTwoPassGaussSliders->Render(0,0);
        break;
    }
}

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
InteractionController::updateGaussParameters()
{
    _pGaussFilter->setSigma(_pGaussSigmaParameter->GetFloatValue());
}

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
InteractionController::updateNightParameters()
{
    _pNightFilter->setBrightness(_pNightBrightnessParameter->GetFloatValue());
}

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
InteractionController::updateScotopicParameters()
{
    _pScotopicFilter->setSigma(_pScotopicSigmaParameter->GetFloatValue());
    _pScotopicFilter->setGamma(_pScotopicGammaParameter->GetFloatValue());
    _pScotopicFilter->setBrightness(_pScotopicBrightnessParameter->GetFloatValue());
}

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
InteractionController::updateGauss1dParameters()
{
    _pGaussFilter1D->setSigma(_pGauss1dSigmaParameter->GetFloatValue());
}

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
InteractionController::updateTwoPassGaussParameters()
{
    _pTwoPassGaussFilter->setSigma(_pTwoPassGaussSigmaParameter->GetFloatValue());
}


