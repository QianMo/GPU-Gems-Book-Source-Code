//--------------------------------------------------------------------------------------
// File: Dx9Tree.fx
// Desc: The effect file for the "GPU Procedural Wind Animation" DirectX 9 Sample
//
// This sample illustrates procedural wind animation technique for trees as
// described in the chapter "GPU Generated, Procedural Wind Animation for Trees"
// of the book GPU Gems III.
//
// Author: Renaldas Zioma (rej@scene.lt)
//
//
//--------------------------------------------------------------------------------------
#include "quaternion.hlsl"
#include "utils.hlsl"

#define USE_SM_2 1
#if USE_SM_2
#	define VS_DEFAULT vs_2_0
#	define PS_DEFAULT ps_2_0
#	define BRANCH_ATTR_D [flatten]
#	define LOOP_ATTR_D [loop]
#else
#	define VS_DEFAULT vs_3_0
#	define PS_DEFAULT ps_3_0
#	define BRANCH_ATTR_D [branch]
#	define LOOP_ATTR_D[unroll(1)]
#endif

#define VS_INSTANCING vs_3_0
#define PS_INSTANCING ps_3_0
#define BRANCH_ATTR_I [branch]
#define LOOP_ATTR_I [loop]

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
float	fTime;								// App's time in seconds
float4x4
		mWorld;								// World matrix for object
float4x4
		mWindRotation;						// World matrix for object
float4x4
		mWorldViewProjection;				// World * View * Projection matrix
float3	vLightDir;							// Light direction
float4	vWorldPos;

float3	vWindDir;							// Wind direction
float3	vWindTangent;						// Wind tangent
float	fPseudoInertiaFactor = 1.0f;		// Inertia emulation

static const int MAX_BRANCHES = 128;
float4	vOriginsAndPhases[MAX_BRANCHES];	// Branch origins and animation phase shifts

static const int RULE_FRONT = 0;
static const int RULE_BACK = 1;
static const int RULE_SIDE = 2;
static const int MAX_RULES = 3;
float4	vAngleShifts[MAX_RULES];			// Input parameters for simulation
float4	vAmplitudes[MAX_RULES];
float	fFrequencies[MAX_RULES];


//--------------------------------------------------------------------------------------
// Vertex shader structures
//--------------------------------------------------------------------------------------
struct VsInput
{
	float3 Pos				: POSITION;
	float3 Normal			: NORMAL;
	float4 bendWeights		: BLENDWEIGHT;
	float4 branchIndices	: BLENDINDICES;
};

struct VsInstancingInput
{
	float3 Pos				: POSITION;
	float3 Normal			: NORMAL;
	float4 bendWeights		: BLENDWEIGHT;
	float4 branchIndices	: BLENDINDICES;
	
	float3 worldPos			: TEXCOORD0;
	float4 windRotation		: TEXCOORD1;
	float  treePhase		: TEXCOORD2;
	float  pseudoInertiaFactor
							: TEXCOORD3;
};

struct VsOutput
{
	float4 Pos				: POSITION;
	float4 Diffuse			: COLOR0;
};

//--------------------------------------------------------------------------------------
float3 getBranchDir(float3 objectPos)
{
	objectPos.z = 0;
	return -normalize(objectPos);
}

VsVertex bendBranch( VsVertex i,
	float3 boneOrigin, float bonePhase, float boneWeight,
	float treePhase, float pseudoInertiaFactor)
{
	VsVertex output;
	
	float3 branchPos = i.pos - boneOrigin;
	
	// determine branch orientation relative to the wind
	float dota = dot(getBranchDir(i.pos), vWindDir);
	float dotb = dot(getBranchDir(i.pos), vWindTangent);

	// calculate parameters for simualation rules
	float t = dota * 0.5f + 0.5f;
	float3 amplitudes = lerp(vAmplitudes[RULE_BACK], vAmplitudes[RULE_FRONT], t);
	float3 angleShifts = lerp(vAngleShifts[RULE_BACK], vAngleShifts[RULE_FRONT], t);
	
	float amplitude0 = lerp3(amplitudes.x, amplitudes.y, amplitudes.z, pseudoInertiaFactor);
	float angleShift0 = lerp3(angleShifts.x, angleShifts.y, angleShifts.z, pseudoInertiaFactor);

	float frequency0 = (dota > 0)? fFrequencies[RULE_FRONT]: fFrequencies[RULE_BACK];
	
	float amplitude1 = vAmplitudes[RULE_SIDE].y;
	float angleShift1 = vAngleShifts[RULE_SIDE].y * dotb;
	float frequency1 = fFrequencies[RULE_SIDE];

	// cacluate quaternion representing bending of the branch due to wind load
	// along direction of the wind
	float4 q0 = quatAxisAngle(vWindTangent,   angleShift0 + amplitude0 * sin((bonePhase + treePhase + fTime)*frequency0));
	
	// cacluate quaternion representing bending of the branch perpendicular to main trunk
	float4 q1 = quatAxisAngle(getTrunkAxis(), angleShift1 + amplitude1 * sin((bonePhase + treePhase + fTime)*frequency1));
	
	// combine bending
	float4 q = lerp(q1, q0, abs(dota));
	
	// transform branch vertices
	float3x3 windRotationMatrix = quatToMatrix(q);
	output.pos = lerp(branchPos, mul(branchPos, windRotationMatrix), boneWeight) + boneOrigin;
	output.normal = mul(i.normal, windRotationMatrix);

	return output;
}

//--------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------
VsOutput TreeVS(VsInput i, uniform int hierarchyDepth)
{
	VsOutput output;
	
	int4 branchIndexVector = i.branchIndices;
	int branchIndexArray[4] = (int[4])branchIndexVector;
	
	float4 bendWeightVector = i.bendWeights;
	float bendWeightArray[4] = (float[4])bendWeightVector;
	
	VsVertex vertex;
	vertex.pos = i.Pos;
	vertex.normal = i.Normal; 
	
	LOOP_ATTR_D
	for(int q = hierarchyDepth; q > 0; --q)
	{
		int branchIndex = branchIndexArray[q];
		BRANCH_ATTR_D
		if(branchIndex > 0)
		{
			vertex = bendBranch(vertex, vOriginsAndPhases[branchIndex].xyz, vOriginsAndPhases[branchIndex].w,
				bendWeightArray[q], 0, fPseudoInertiaFactor);
		}
	}
	
	vertex.pos = lerp(
		vertex.pos,
		mul(vertex.pos, (float3x3)mWindRotation), bendWeightArray[0]);
	vertex.pos += vWorldPos.xyz; 

	output.Pos = mul(float4(vertex.pos, 1), mWorldViewProjection);
	output.Diffuse = saturate(dot(vLightDir, vertex.normal)) + 0.33f;

	return output;
}

VsOutput TreeInstancingVS(VsInstancingInput i, uniform int hierarchyDepth)
{
	VsOutput output;
	
	int4 branchIndexVector = i.branchIndices;
	int branchIndexArray[4] = (int[4])branchIndexVector;
	
	float4 bendWeightVector = i.bendWeights;
	float bendWeightArray[4] = (float[4])bendWeightVector;

	VsVertex v;
	v.pos = i.Pos;
	v.normal = i.Normal; 

	LOOP_ATTR_I
	for(int q = hierarchyDepth; q > 0; --q)
	{
		int branchIndex = branchIndexArray[q];
		BRANCH_ATTR_I
		if(branchIndex > 0)
		{
			v = bendBranch(v, vOriginsAndPhases[branchIndex].xyz, vOriginsAndPhases[branchIndex].w,
				bendWeightArray[q], i.treePhase, i.pseudoInertiaFactor);
		}
	}

	float3x3 windRotationMatrix = quatToMatrix(i.windRotation);
	v.pos = lerp(
		v.pos,
		mul(v.pos.xyz, windRotationMatrix), bendWeightArray[0]);
	v.pos += i.worldPos;
	v.normal = mul(v.normal, (float3x3)mWorld);

	output.Pos = mul(float4(v.pos, 1), mWorldViewProjection);
	output.Diffuse = saturate(dot(-vLightDir, v.normal)) + 0.33f;

	return output;
}

//--------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------
float4 TreePS(VsOutput i) : COLOR0
{
	return i.Diffuse;
}

//--------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------
//
int iHierarchyDepth = 2;
VertexShader treeVsArray[] = { 
	compile VS_DEFAULT TreeVS(0), 
	compile VS_DEFAULT TreeVS(1),
	compile VS_DEFAULT TreeVS(2) };
VertexShader treeInstancingVsArray[] = { 
	compile VS_INSTANCING TreeInstancingVS(0), 
	compile VS_INSTANCING TreeInstancingVS(1),
	compile VS_INSTANCING TreeInstancingVS(2) };

technique Default
{
	pass P0
	{
		VertexShader = (treeVsArray[iHierarchyDepth]);
		PixelShader  = compile PS_DEFAULT TreePS();
	}
}

technique Instancing
{
	pass P0
	{
		VertexShader = (treeInstancingVsArray[iHierarchyDepth]);
		PixelShader  = compile PS_INSTANCING TreePS();
	}
}