#ifndef __PARAMETERS_H
#define __PARAMETERS_H

///////////////////////////////////////////////////////////////////////////////
// Parameters defining wind simulation
//
struct SimulationParameters
{
	enum { RuleFront, RuleBack, RuleSide, MaxRuleCount };

	float trunkAmplitude;
	float branchFrequencyModifier;

	float inertiaDelay;
	float inertiaPropagation;

	float angleShift[MaxRuleCount];	
	float amplitude[MaxRuleCount];
	float frequency[MaxRuleCount];

	float angleShift_[MaxRuleCount];	
	float amplitude_[MaxRuleCount];
};

///////////////////////////////////////////////////////////////////////////////
// Parameters defining behavior of the sample
//
struct Parameters
{
	enum { MaxHierarchyDepth = 3 };
	enum nInstancingType { InstancingGPU, InstancingCPU };

	D3DXVECTOR3	windDir;

	size_t treeCount;
	nInstancingType instancingType;

	int hierarchyDepth;
	size_t windType;
	size_t treeType;

	SimulationParameters simulation;
};

#endif