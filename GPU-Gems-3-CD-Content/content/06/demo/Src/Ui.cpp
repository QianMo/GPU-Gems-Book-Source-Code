#include "DXUT.h"
#include "Parameters.h"
#include "Ui.h"
#include "UiUtils.h"

void Ui::init(Parameters& params, PCALLBACKDXUTGUIEVENT pCallback, void* pUserContext)
{
	if (mAlreadyInitialized)
		return;

	mParams = &params;

	mWindControl.SetLightDirection(D3DXVECTOR3(1.0f, 0.0f, 0.0f));
	mWindControl.SetButtonMask(MOUSE_LEFT_BUTTON);

    mHUD.Init(&mDialogResourceManager);
    mSampleUI.Init(&mDialogResourceManager);
	mSimulationUI.Init(&mDialogResourceManager);

	// HUD controls
    mHUD.SetCallback(pCallback, pUserContext); int iY = 10;
    mHUD.AddButton(IdcToggleFullScreen, L"Toggle full screen", 35, iY, 125, 22);

	// SystemUI controls
    mSampleUI.SetCallback(pCallback, pUserContext); iY = 10;
    WCHAR sz[100]; sz[0] = 0;
    mSampleUI.AddStatic(IdcHierarchyDepth + IdcStaticOffset, sz, 15, iY += 24, 125, 22);
	mSampleUI.AddSlider(IdcHierarchyDepth, 30, iY += 24, 100, 22, 1, Parameters::MaxHierarchyDepth, mParams->hierarchyDepth);

    mSampleUI.AddComboBox(IdcTreeType, 0, iY += 24, 180, 24, L'T');
    mSampleUI.GetComboBox(IdcTreeType)->AddItem(L"Weak Fir (T)ree", (void*)0);
    mSampleUI.GetComboBox(IdcTreeType)->AddItem(L"Fir (T)ree", (void*)1);
    mSampleUI.GetComboBox(IdcTreeType)->AddItem(L"Birch (T)ree", (void*)2);
    mSampleUI.GetComboBox(IdcTreeType)->AddItem(L"Small Birch (T)ree", (void*)3);

    mSampleUI.AddComboBox(IdcWindType, 0, iY += 24, 180, 24, L'W');
    mSampleUI.GetComboBox(IdcWindType)->AddItem(L"Turbulent (W)ind A", (void*)0);
    mSampleUI.GetComboBox(IdcWindType)->AddItem(L"Turbulent (W)ind B", (void*)1);
    mSampleUI.GetComboBox(IdcWindType)->AddItem(L"Smooth (W)ind with Noise", (void*)2);
    mSampleUI.GetComboBox(IdcWindType)->AddItem(L"Monotonic Sine (W)ind", (void*)3);

	mSampleUI.AddComboBox(IdcInstancingType, 0, iY += 24, 180, 24, L'I');
	mSampleUI.GetComboBox(IdcInstancingType)->AddItem(L"(I)nstances: 1", (void*)0);
    mSampleUI.GetComboBox(IdcInstancingType)->AddItem(L"(I)nstances: 10", (void*)1);
    mSampleUI.GetComboBox(IdcInstancingType)->AddItem(L"(I)nstances: 30", (void*)2);
    mSampleUI.GetComboBox(IdcInstancingType)->AddItem(L"(I)nstances: 100", (void*)3);
    mSampleUI.GetComboBox(IdcInstancingType)->AddItem(L"(I)nstances: 256", (void*)4);
    mSampleUI.GetComboBox(IdcInstancingType)->AddItem(L"(I)nstances: 1000", (void*)5);


	// mSimulationUI controls
	mSimulationUI.SetCallback(pCallback, pUserContext); iY = 10;

    int tY = iY;
	mSimulationUI.AddStatic(IdcTrunkAmplitude + IdcStaticOffset, sz, -60, iY += 24, 125, 22);
    mSimulationUI.AddSlider(IdcTrunkAmplitude, -45, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(0.2f));
	iY = tY;
    mSimulationUI.AddStatic(IdcBranchFrequencyModifier + IdcStaticOffset, sz, 50, iY += 24, 125, 22);
    mSimulationUI.AddSlider(IdcBranchFrequencyModifier, 64, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(5.0f));

	tY = iY;
    mSimulationUI.AddStatic(IdcInertiaPropagation + IdcStaticOffset, sz, 50, iY += 24, 125, 22);
    mSimulationUI.AddSlider(IdcInertiaPropagation, 64, iY += 16, 100, 22,
		toSlider(0.5f), toSlider(2.0f));
	iY = tY;
    mSimulationUI.AddStatic(IdcInertiaDelay + IdcStaticOffset, sz, -60, iY += 24, 125, 22);
    mSimulationUI.AddSlider(IdcInertiaDelay, -45, iY += 16, 100, 22,
		toSlider(-1.0f), toSlider(1.0f));

	iY += 16;
    StringCchPrintf(sz, 100, L"---------------------------------------------------------------"); 
    mSimulationUI.AddStatic(IdcSeparator0, sz, -60, iY, 240, 22);
	iY += 16;

    StringCchPrintf(sz, 100, L"Rule [Front]"); 
    mSimulationUI.AddStatic(IdcRuleName0, sz, -86, iY += 24, 125, 22);

	mSimulationUI.AddStatic(IdcFrequency0 + IdcStaticOffset, sz, 35 + 16, iY += 0, 125, 22);
    mSimulationUI.AddSlider(IdcFrequency0, 50 + 16, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(5.0f));
	tY = iY;
	mSimulationUI.AddStatic(IdcAngleShift0 + IdcStaticOffset, sz, 35 + 16, iY += 20, 125, 22);
    mSimulationUI.AddSlider(IdcAngleShift0, 50 + 16, iY += 16, 100, 22,
		toSlider(-1.0f), toSlider(1.0f));
	mSimulationUI.AddStatic(IdcAmplitude0 + IdcStaticOffset, sz, 35 + 16, iY += 20, 125, 22);
	mSimulationUI.AddSlider(IdcAmplitude0, 50 + 16, iY += 16, 100, 22, 
		toSlider(0.0f), toSlider(0.4f));

	iY = tY;
	mSimulationUI.AddStatic(IdcAngleShift0_ + IdcStaticOffset, sz, 35 + 16 -110, iY += 20, 125, 22);
    mSimulationUI.AddSlider(IdcAngleShift0_, 50 + 16 -110, iY += 16, 100, 22,
		toSlider(-1.0f), toSlider(1.0f));
	mSimulationUI.AddStatic(IdcAmplitude0_ + IdcStaticOffset, sz, 35 + 16 -110, iY += 20, 125, 22);
	mSimulationUI.AddSlider(IdcAmplitude0_, 50 + 16 -110, iY += 16, 100, 22, 
		toSlider(0.0f), toSlider(0.4f));


	iY += 10;
    StringCchPrintf(sz, 100, L"Rule [Back]"); 
	mSimulationUI.AddStatic(IdcRuleName1, sz, 24 -110, iY += 24, 125, 22);

	mSimulationUI.AddStatic(IdcFrequency1 + IdcStaticOffset, sz, 35 + 16, iY += 0, 125, 22);
    mSimulationUI.AddSlider(IdcFrequency1, 50 + 16, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(5.0f));
	tY = iY;
	mSimulationUI.AddStatic(IdcAmplitude1 + IdcStaticOffset, sz, 35 + 16, iY += 20, 125, 22);
    mSimulationUI.AddSlider(IdcAmplitude1, 50 + 16, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(0.4f));

	iY = tY;
	mSimulationUI.AddStatic(IdcAmplitude1_ + IdcStaticOffset, sz, 35 + 16 -110, iY += 20, 125, 22);
    mSimulationUI.AddSlider(IdcAmplitude1_, 50 + 16 -110, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(0.4f));

	iY += 10;
    StringCchPrintf(sz, 100, L"Rule [Side]"); 
    mSimulationUI.AddStatic(IdcRuleName2, sz, 24 -110, iY += 24, 125, 22);

	mSimulationUI.AddStatic(IdcFrequency2 + IdcStaticOffset, sz, 35 + 16, iY += 0, 125, 22);
    mSimulationUI.AddSlider(IdcFrequency2, 50 + 16, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(5.0f));
	mSimulationUI.AddStatic(IdcAngleShift2 + IdcStaticOffset, sz, 35 + 16, iY += 20, 125, 22);
    mSimulationUI.AddSlider(IdcAngleShift2, 50 + 16, iY += 16, 100, 22,
		toSlider(-0.5f), toSlider(1.0f));
	mSimulationUI.AddStatic(IdcAmplitude2 + IdcStaticOffset, sz, 35 + 16, iY += 20, 125, 22);
    mSimulationUI.AddSlider(IdcAmplitude2, 50 + 16, iY += 16, 100, 22,
		toSlider(0.0f), toSlider(0.4f));

	refresh(Push);

	mAlreadyInitialized = true;
}

void Ui::refresh(nDirection dir)
{
	assert(mParams);
	bool read = (dir == Pull);

	updateSlider(mSampleUI.GetSlider(IdcHierarchyDepth), mSampleUI.GetStatic(IdcHierarchyDepth + IdcStaticOffset), 
		L"SLOD(hierarchy depth): %d", mParams->hierarchyDepth, read);

	updateSlider(mSimulationUI.GetSlider(IdcTrunkAmplitude), mSimulationUI.GetStatic(IdcTrunkAmplitude + IdcStaticOffset), 
		L"Trunk amplitude: %0.2f", mParams->simulation.trunkAmplitude, read);

	updateSlider(mSimulationUI.GetSlider(IdcInertiaPropagation), mSimulationUI.GetStatic(IdcInertiaPropagation + IdcStaticOffset), 
		L"Inertia propagation: %0.2f", mParams->simulation.inertiaPropagation, read);
	updateSlider(mSimulationUI.GetSlider(IdcInertiaDelay), mSimulationUI.GetStatic(IdcInertiaDelay + IdcStaticOffset), 
		L"Inertia delay: %0.2f", mParams->simulation.inertiaDelay, read);
	updateSlider(mSimulationUI.GetSlider(IdcBranchFrequencyModifier), mSimulationUI.GetStatic(IdcBranchFrequencyModifier + IdcStaticOffset), 
		L"Branch freq: %0.2f", mParams->simulation.branchFrequencyModifier, read);

	LPCWSTR angleShiftTemplate = L"Angle shift: %0.2f";
	updateSlider(mSimulationUI.GetSlider(IdcAngleShift0), mSimulationUI.GetStatic(IdcAngleShift0 + IdcStaticOffset), 
		angleShiftTemplate, mParams->simulation.angleShift[SimulationParameters::RuleFront], read);
	updateSlider(mSimulationUI.GetSlider(IdcAngleShift2), mSimulationUI.GetStatic(IdcAngleShift2 + IdcStaticOffset), 
		angleShiftTemplate, mParams->simulation.angleShift[SimulationParameters::RuleSide], read);

	LPCWSTR amplitudeTemplate = L"Amplitude: %0.2f";
	updateSlider(mSimulationUI.GetSlider(IdcAmplitude0), mSimulationUI.GetStatic(IdcAmplitude0 + IdcStaticOffset), 
		amplitudeTemplate, mParams->simulation.amplitude[SimulationParameters::RuleFront], read);
	updateSlider(mSimulationUI.GetSlider(IdcAmplitude1), mSimulationUI.GetStatic(IdcAmplitude1 + IdcStaticOffset), 
		amplitudeTemplate, mParams->simulation.amplitude[SimulationParameters::RuleBack], read);
	updateSlider(mSimulationUI.GetSlider(IdcAmplitude2), mSimulationUI.GetStatic(IdcAmplitude2 + IdcStaticOffset), 
		amplitudeTemplate, mParams->simulation.amplitude[SimulationParameters::RuleSide], read);

	LPCWSTR frequencyTemplate = L"Frequency: %0.2f";
 	updateSlider(mSimulationUI.GetSlider(IdcFrequency0), mSimulationUI.GetStatic(IdcFrequency0 + IdcStaticOffset), 
		frequencyTemplate, mParams->simulation.frequency[SimulationParameters::RuleFront], read);
 	updateSlider(mSimulationUI.GetSlider(IdcFrequency1), mSimulationUI.GetStatic(IdcFrequency1 + IdcStaticOffset), 
		frequencyTemplate, mParams->simulation.frequency[SimulationParameters::RuleBack], read);
 	updateSlider(mSimulationUI.GetSlider(IdcFrequency2), mSimulationUI.GetStatic(IdcFrequency2 + IdcStaticOffset), 
		frequencyTemplate, mParams->simulation.frequency[SimulationParameters::RuleSide], read);

	updateSlider(mSimulationUI.GetSlider(IdcAngleShift0_), mSimulationUI.GetStatic(IdcAngleShift0_ + IdcStaticOffset), 
		angleShiftTemplate, mParams->simulation.angleShift_[SimulationParameters::RuleFront], read);
	updateSlider(mSimulationUI.GetSlider(IdcAmplitude0_), mSimulationUI.GetStatic(IdcAmplitude0_ + IdcStaticOffset), 
		amplitudeTemplate, mParams->simulation.amplitude_[SimulationParameters::RuleFront], read);
	updateSlider(mSimulationUI.GetSlider(IdcAmplitude1_), mSimulationUI.GetStatic(IdcAmplitude1_ + IdcStaticOffset), 
		amplitudeTemplate, mParams->simulation.amplitude_[SimulationParameters::RuleBack], read);

	if(read)
		return;

	updateComboBox(mSampleUI.GetComboBox(IdcWindType), mParams->windType);
}

Parameters const& Ui::pull()
{
	mParams->windDir = -mWindControl.GetLightDirection();
	return *mParams;
}

void Ui::push(Parameters const& params)
{
	assert(mParams);
	*mParams = params;
	refresh(Push);
}

void Ui::update(float objectRadius)
{
   mWindControl.SetRadius(objectRadius * 0.5f);
}

bool Ui::MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	bool noFurtherProcessing = false;
    noFurtherProcessing = mDialogResourceManager.MsgProc(hWnd, uMsg, wParam, lParam);
    if(noFurtherProcessing)
        return true;
    noFurtherProcessing = mHUD.MsgProc(hWnd, uMsg, wParam, lParam);
    if(noFurtherProcessing)
		return true;
    noFurtherProcessing = mSampleUI.MsgProc(hWnd, uMsg, wParam, lParam);
    if(noFurtherProcessing)
		return true;
    noFurtherProcessing = mSimulationUI.MsgProc(hWnd, uMsg, wParam, lParam);
    if(noFurtherProcessing)
		return true;

	mWindControl.HandleMessages(hWnd, uMsg, wParam, lParam);
	return false;
}