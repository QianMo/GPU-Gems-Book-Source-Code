#include "DXUT.h"
#include "SampleCore.h"

#include "Ui.h"
#include "Wind.h"
#include "TreeData.h"

namespace {
	D3DXVECTOR2 animateWindPower(size_t windType, float t)
	{
		float windPower = 0.0f;
		switch(windType)
		{
		case 0:
			windPower = windWithPseudoTurbulence(t);
			break;
		case 1:
			windPower = windWithPseudoTurbulence2(t);
			break;
		case 2:	
			windPower = windSmoothWithSlightNoise(t);
			break;
		case 3:
			windPower = windPeriodicWithNoise(t);
			break;
		}

		return D3DXVECTOR2(
			0.5f - windPower,
			windPeriodicWithNoise(t) * 0.1f);
	}

	D3DXQUATERNION animateWindRotation(size_t windType, D3DXVECTOR2 const& direction, float power, float t)
	{
		return calcWindRotation(
			direction,
			animateWindPower(windType, t) * power);
	}
}


SampleCore::SampleCore(IDirect3DDevice& d3dDevice)
:	mDevice(&d3dDevice), mSceneRadius(10.0f)
{
	// setup default parameters
	mParameters.instancingType = Parameters::InstancingCPU;
	mParameters.treeCount = 1;
	mTreeInstances.resize(mParameters.treeCount);

	mParameters.hierarchyDepth = 3;
	mParameters.windType = 0;
	mParameters.treeType = 0;

	mParameters.simulation.inertiaPropagation = 0.75f;
	mParameters.simulation.inertiaDelay = 0.2f;

	mParameters.simulation.branchFrequencyModifier = 1.0f;
	for(size_t q = 0; q < SimulationParameters::MaxRuleCount; ++q)
	{
		mParameters.simulation.angleShift[q] = 0.0f;
		mParameters.simulation.amplitude[q] = 0.2f;
		mParameters.simulation.frequency[q] = 1.0f;
	}
	for(size_t q = 0; q < SimulationParameters::MaxRuleCount; ++q)
	{
		mParameters.simulation.angleShift_[q] = 0.0f;
		mParameters.simulation.amplitude_[q] = 0.2f;
	}

	// front
	{
		mParameters.simulation.angleShift[SimulationParameters::RuleFront] = 0.2f;
		mParameters.simulation.amplitude[SimulationParameters::RuleFront] = 0.075f;
		mParameters.simulation.frequency[SimulationParameters::RuleFront] = 2.0f;

		mParameters.simulation.angleShift_[SimulationParameters::RuleFront] = 0.25f;
		mParameters.simulation.amplitude_[SimulationParameters::RuleFront] = 0.05f;
	}

	// side
	{
		mParameters.simulation.angleShift[SimulationParameters::RuleSide] = 0.2f;
		mParameters.simulation.amplitude[SimulationParameters::RuleSide] = 0.25f;
	}

	setTreeType(mParameters.treeType);
	updateScene();
}

void SampleCore::updateScene()
{
	// setup UI controls
    D3DXVECTOR3 vCenter = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	calcBoundingSphere(mRenderableTree, vCenter, mSceneRadius);

    D3DXMatrixTranslation(&mWorldMatrix, -vCenter.x, -vCenter.y, -vCenter.z);
    D3DXMATRIX m;
    D3DXMatrixRotationY(&m, D3DX_PI);
    mWorldMatrix *= m;
    D3DXMatrixRotationX(&m, D3DX_PI / 2.0f);
    mWorldMatrix *= m;
}

void SampleCore::update(float t)
{
	const float animationSpeed = 0.05f;

	D3DXMATRIX invWorld;
	D3DXMatrixInverse(&invWorld, 0, &mWorldMatrix);

	D3DXVec3TransformNormal(&mWindDir, &mParameters.windDir, &invWorld);
	mWindDir.z = 0;
	D3DXVec3Normalize(&mWindDir, &mWindDir);

	// update tree parameters
 	assert(mTreeInstances.size() >= mParameters.treeCount);
	for(size_t q = 0; q < mParameters.treeCount; ++q)
	{
		const float phase = sinf(static_cast<float>(q));
		const float delayedWindPower = animateWindPower(
			mParameters.windType,
			(t + phase - mParameters.simulation.inertiaDelay) * animationSpeed).x;

		mTreeInstances[q].windRotation = animateWindRotation(
			mParameters.windType,
			D3DXVECTOR2(mWindDir.x, mWindDir.y), mParameters.simulation.trunkAmplitude, (t + phase) * animationSpeed);
		mTreeInstances[q].pseudoInertiaFactor = -sinf(
			mParameters.simulation.trunkAmplitude * 10.0f * mParameters.simulation.inertiaPropagation * delayedWindPower);
		mTreeInstances[q].treePhase = phase;
	}

	// update branch parameters
	for(int q = 0; q < SimulationParameters::MaxRuleCount; ++q)
		mPackedBranchParameters.frequency[q] = 
			mParameters.simulation.frequency[q] * mParameters.simulation.branchFrequencyModifier;

	// NOTE: branch parameters are packed to ease use from the shaders
	mPackedBranchParameters.angleShift[0] = D3DXVECTOR4(
		mParameters.simulation.angleShift_[0],
		mParameters.simulation.angleShift[0],
		mParameters.simulation.angleShift[1], 0);
	mPackedBranchParameters.amplitude[0] = D3DXVECTOR4(
		mParameters.simulation.amplitude_[0],
		mParameters.simulation.amplitude[0],
		mParameters.simulation.amplitude[1], 0);

	mPackedBranchParameters.angleShift[1] = D3DXVECTOR4(
		-mParameters.simulation.angleShift_[1],
		-mParameters.simulation.angleShift[1],
		-mParameters.simulation.angleShift[0], 0);
	mPackedBranchParameters.amplitude[1] = D3DXVECTOR4(
		mParameters.simulation.amplitude_[1],
		mParameters.simulation.amplitude[1],
		mParameters.simulation.amplitude[0], 0);

	mPackedBranchParameters.angleShift[2] = D3DXVECTOR4(
		mParameters.simulation.angleShift[2],
		mParameters.simulation.angleShift[2],
		mParameters.simulation.angleShift[2], 0);
	mPackedBranchParameters.amplitude[2] = D3DXVECTOR4(
		mParameters.simulation.amplitude[2],
		mParameters.simulation.amplitude[2],
		mParameters.simulation.amplitude[2], 0);
}

void SampleCore::setTreeType(size_t treeType)
{
	assert(treeType < 4);
	BranchDesc const* treeDesc[] = {
		WEAK_FIR_TREE_DESC,
		STIFF_FIR_TREE_DESC,
		BIRCH_TREE_DESC,
		SMALL_BIRCH_TREE_DESC
	};
	SimulationParameters treeParams[] = {
		WEAK_FIR_TREE_PARAMS,
		STIFF_FIR_TREE_PARAMS,
		BIRCH_TREE_PARAMS,
		SMALL_BIRCH_TREE_PARAMS
	};

	// generate geometry suitable for rendering
	assert(mDevice);
	generateRenderableTree(*mDevice, 3, treeDesc[treeType], mRenderableTree);
	mParameters.simulation = treeParams[treeType];

	// NOTE: prefer sinusoidal wind as a default setting for 'weak' trees
	if(treeType == 0)
		mParameters.windType = 3;
	else
		mParameters.windType = 0;
}

void SampleCore::onUiEvent(Ui& ui, UINT nEvent, int nControlID, CDXUTControl* pControl)
{    
    switch(nControlID)
    {
	case Ui::IdcToggleFullScreen: DXUTToggleFullScreen(); break;

		case Ui::IdcInstancingType:
			switch(reinterpret_cast<size_t>(static_cast<CDXUTComboBox*>(pControl)->GetSelectedData()))
			{
			default:
			case 0:
				mParameters.treeCount = 1;
				break;
			case 1:
				mParameters.treeCount = 10;
				break;
			case 2:
				mParameters.treeCount = 30;
				break;
			case 3:
				mParameters.treeCount = 100;
				break;
			case 4:
				mParameters.treeCount = 256;
				break;
			case 5:
				mParameters.treeCount = 1000;
				break;
			}
			mTreeInstances.resize(mParameters.treeCount);
			break;
		case Ui::IdcWindType:
			mParameters.windType = reinterpret_cast<size_t>(static_cast<CDXUTComboBox*>(pControl)->GetSelectedData());
			break;
		case Ui::IdcTreeType:
			setTreeType(reinterpret_cast<size_t>(static_cast<CDXUTComboBox*>(pControl)->GetSelectedData()));
			updateScene();
			ui.push(mParameters);
			break;

		case Ui::IdcHierarchyDepth:
		case Ui::IdcTrunkAmplitude:
		case Ui::IdcInertiaPropagation:
		case Ui::IdcInertiaDelay:

		case Ui::IdcBranchFrequencyModifier:
		case Ui::IdcAngleShift0:
		case Ui::IdcAngleShift2:
		case Ui::IdcAmplitude0:
		case Ui::IdcAmplitude1:
		case Ui::IdcAmplitude2:
		case Ui::IdcFrequency0:
		case Ui::IdcFrequency1:
		case Ui::IdcFrequency2:

		case Ui::IdcAmplitude0_:
		case Ui::IdcAmplitude1_:
		case Ui::IdcAngleShift0_:

			ui.refresh();
		break;
    }  
}
