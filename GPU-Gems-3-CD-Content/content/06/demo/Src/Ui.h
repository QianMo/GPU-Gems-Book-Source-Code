#ifndef __UI_H
#define __UI_H

#include "Cfg.h"
#include "DXUTgui.h"
#include "DXUTcamera.h"

struct Parameters;
class Ui
{
public:
	enum {
		IdcToggleFullScreen,
		IdcHierarchyDepth,

		IdcInstancingType,
		IdcWindType,
		IdcTreeType,

		IdcTrunkAmplitude,

		IdcInertiaPropagation, IdcInertiaDelay,
		IdcBranchFrequencyModifier,

		IdcSeparator0,
		IdcRuleName0, IdcRuleName1, IdcRuleName2,
		IdcAngleShift0, IdcAngleShift1, IdcAngleShift2,
		IdcAmplitude0, IdcAmplitude1, IdcAmplitude2,
		IdcFrequency0, IdcFrequency1, IdcFrequency2,

		IdcAngleShift0_,
		IdcAmplitude0_, IdcAmplitude1_,

		MaxIdcDynamicCount,
		IdcStaticOffset = MaxIdcDynamicCount,
		MaxIdcCount = MaxIdcDynamicCount + MaxIdcDynamicCount
	};

public:
	Ui() : mAlreadyInitialized(false) {}
	void init(Parameters &dstParams, PCALLBACKDXUTGUIEVENT pCallback, void* pUserContext = 0);
	void update(float sceneRadius);

	enum nDirection { Push, Pull };
	void refresh(nDirection dir = Pull);

	Parameters const& pull();
	void push(Parameters const& params);

public:
	bool MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

protected:
	CDXUTDirectionWidget		mWindControl;			// wind direction widget
	CDXUTDialogResourceManager	mDialogResourceManager;	// manager for shared resources of dialogs
	CDXUTDialog					mHUD;					// manages the system UI
	CDXUTDialog					mSampleUI;				// dialog for sample specific controls
	CDXUTDialog					mSimulationUI;			// dialog for simulation specific controls

	Parameters*					mParams;				// sample parameters

	bool						mAlreadyInitialized;
};

#endif