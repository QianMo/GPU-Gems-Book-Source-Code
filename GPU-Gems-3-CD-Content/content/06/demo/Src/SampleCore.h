#ifndef __SAMPLE_CORE_H
#define __SAMPLE_CORE_H

#include "Cfg.h"
#include <vector>
#include "Platform.h"
#include "Parameters.h"
#include "RenderableTree.h"

class Ui;
class CDXUTControl;
///////////////////////////////////////////////////////////////////////////////
// Core logic of the "GPU Procedural Wind Animation" sample for GPU Gems 3 book
//
class SampleCore
{
public:
	struct TreeInstance
	{
		D3DXVECTOR3 worldPos;
		D3DXQUATERNION windRotation;
		float treePhase;
		float pseudoInertiaFactor;
	};
	typedef std::vector<TreeInstance> TreeInstancesT;

	struct PackedBranchParameters
	{
		D3DXVECTOR4 angleShift[SimulationParameters::MaxRuleCount];	
		D3DXVECTOR4 amplitude[SimulationParameters::MaxRuleCount];
		float frequency[SimulationParameters::MaxRuleCount];
	};

public:
	SampleCore(IDirect3DDevice& d3dDevice);
	void update(float t);
	void onUiEvent(Ui& ui, UINT nEvent, int nControlID, CDXUTControl* pControl);

public:
	Parameters& getParameters() { return mParameters; }
	Parameters const& getParameters() const { return mParameters; }

	D3DXVECTOR3 const& getWindDir() const { return mWindDir; }
	D3DXMATRIX const& getWorldMatrix() const { return mWorldMatrix; }
	float getSceneRadius() const { return mSceneRadius; }

	RenderableTree const& getRenderableTree() const { return mRenderableTree; }
	TreeInstancesT const& getTreeInstances() const { return mTreeInstances; }

	PackedBranchParameters const& getPackedBranchParameters() const { return mPackedBranchParameters; }

protected:
	void updateScene();
	void setTreeType(size_t treeType = 0);

private:
	// disable copy ctor and assignment operator
	SampleCore(SampleCore const& c);
	SampleCore& operator= (SampleCore const& rhs);

private:
	IDirect3DDevice*		mDevice;

	Parameters				mParameters;
	PackedBranchParameters	mPackedBranchParameters;

	D3DXVECTOR3				mWindDir;
	D3DXMATRIX				mWorldMatrix;
	float					mSceneRadius;

	RenderableTree			mRenderableTree;
	TreeInstancesT			mTreeInstances;
};

#endif