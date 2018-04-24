//--------------------------------------------------------------------------------------
// File: Dx10Tree.fx
// Desc: The effect file for the "GPU Procedural Wind Animation" DirectX 10 Sample
//
// This sample illustrates procedural wind animation technique for trees
//     as described in  "GPU Generated, Procedural Wind Animation for Trees"
//     chapter 6 of the "GPU Gems 3" book
//
// Author: Renaldas Zioma (rej@scene.lt)
//
//--------------------------------------------------------------------------------------
#include "quaternion.hlsl"
#include "utils.hlsl"

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
float			fTime;								// App's time in seconds
float4x4		mWorld;								// World matrix for object
float4x4		mWorldViewProjection;				// World * View * Projection matrix
float3			vLightDir;							// Light direction

float3			vWindDir;							// Wind direction
float3			vWindTangent;						// Wind tangent

static const int MAX_BRANCHES = 128;
float4			vOriginsAndPhases[MAX_BRANCHES];	// Branch origins and animation phase shifts
float4			vDirections[MAX_BRANCHES];
int				iBranchCount;

static const int RULE_FRONT = 0;
static const int RULE_BACK = 1;
static const int RULE_SIDE = 2;
static const int MAX_RULES = 3;
float4			vAngleShifts[MAX_RULES];			// Input parameters for simulation
float4			vAmplitudes[MAX_RULES];
float			fFrequencies[MAX_RULES];

Buffer<float4>	bufferBranchMatrix;					// Buffer with result of branch simulation

//--------------------------------------------------------------------------------------
// State
//--------------------------------------------------------------------------------------
DepthStencilState DisableDepth
{
    DepthEnable = FALSE;
    DepthWriteMask = 0;
};

DepthStencilState EnableDepth
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
};

//--------------------------------------------------------------------------------------
// Vertex shader structures
//--------------------------------------------------------------------------------------
struct VsBranchInput
{
	uint   branchIndex		: SV_VertexID;

	float  treePhase		: PHASE;
	float  pseudoInertiaFactor
							: INERTIA;
	float3 worldPos			: WORLDPOS;
	float4 windRotation		: WINDROTATION;
};

struct VsVertexInput
{
	uint   treeIndex		: SV_InstanceID;

	float3 pos				: POSITION;
	float3 normal			: NORMAL;
	float4 bendWeights		: BLENDWEIGHT;
	uint4  branchIndices	: BLENDINDICES;
};

struct VsOutput
{
	float4 pos				: SV_POSITION;
	float4 diffuse			: COLOR0;
};

//--------------------------------------------------------------------------------------
// Simulation step
// Branch rotations are simulated and stored using stream-out functionality

float3x3 bendBranch_out(
	float3 boneOrigin, float3 branchDir, float bonePhase,
	float treePhase, float pseudoInertiaFactor)
{
	// determine branch orientation relative to the wind
	float dota = dot(branchDir, vWindDir);
	float dotb = dot(branchDir, vWindTangent);

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
	
	// convert quaternion to rotation matrix (3x3)
	float3x3 windRotationMatrix = quatToMatrix(q);
	return quatToMatrix(q);
}

//--------------------------------------------------------------------------------------
// Visualization step
// Branch rotations are read and applied to branch vertices

VsVertex bendBranch_in(VsVertex i,
	float3 branchOrigin, float branchWeight, float3x3 rotationMatrix)
{
	VsVertex output;
	
	float3 branchPos = i.pos - branchOrigin;

	output.pos = lerp(
		i.pos, 
		mul(rotationMatrix, branchPos) + branchOrigin,
		branchWeight);
	output.normal = mul(rotationMatrix, i.normal);

	return output;
}

//--------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------
struct VsStreamOut
{
	float4 row0		: TEXCOORD0;
	float4 row1		: TEXCOORD1;
	float4 row2		: TEXCOORD2;
	float4 worldPos	: TEXCOORD3;
	float4 windRot	: TEXCOORD4;
};
static const int STREAM_OUT_BRANCH_MATRIX = 0;
static const int STREAM_OUT_WORLD_POS = 3;
static const int STREAM_OUT_WIND_ROT = 4;
static const int MAX_STREAM_OUT_ELEMENTS = 5;

VsStreamOut StreamOutVS(VsBranchInput i)
{
    VsStreamOut output;
   
	uint branchIndex = i.branchIndex % iBranchCount;
	float3x3 m = bendBranch_out(
		vOriginsAndPhases[branchIndex].xyz,
		vDirections[branchIndex].xyz,
		vOriginsAndPhases[branchIndex].w,
		i.treePhase, i.pseudoInertiaFactor);

	output.row0 = float4(m._11_21_31, 0.0f);
	output.row1 = float4(m._12_22_32, 0.0f);
	output.row2 = float4(m._13_23_33, 0.0f);

	output.worldPos = float4(i.worldPos, 1.0f);
	output.windRot = i.windRotation;
	
    return output;
}

VsOutput StreamInVS(VsVertexInput i, uniform int hierarchyDepth)
{
	VsOutput output = (VsOutput)0;

	uint4 branchIndexVector = i.branchIndices;
	int branchIndexArray[4] = (int[4])branchIndexVector;
	
	float4 bendWeightVector = i.bendWeights;
	float bendWeightArray[4] = (float[4])bendWeightVector;

	VsVertex v;
	v.pos = i.pos;
	v.normal = i.normal;
		
	for(int q = hierarchyDepth; q > 0; --q)
	{
		uint branchIndex = branchIndexArray[q];
		if(branchIndex > 0)
		{
			// calc offset to branch matrix
			uint baseMatrixIndex = (i.treeIndex * iBranchCount + branchIndex) * MAX_STREAM_OUT_ELEMENTS;

			// sample branch rotation matrix from the buffer
			float4 row0 = bufferBranchMatrix.Load( baseMatrixIndex + STREAM_OUT_BRANCH_MATRIX);
			float4 row1 = bufferBranchMatrix.Load( baseMatrixIndex + STREAM_OUT_BRANCH_MATRIX + 1);
			float4 row2 = bufferBranchMatrix.Load( baseMatrixIndex + STREAM_OUT_BRANCH_MATRIX + 2);
	        float3x3 m = float3x3(row0.xyz, row1.xyz, row2.xyz);

			v = bendBranch_in(v, vOriginsAndPhases[branchIndex].xyz, bendWeightArray[q], m);
		}
	}
	
	uint baseTreeIndex = (i.treeIndex * iBranchCount) * MAX_STREAM_OUT_ELEMENTS;
	
	// sample tree instance data from the buffer
	float3 worldPos = bufferBranchMatrix.Load(baseTreeIndex + STREAM_OUT_WORLD_POS).xyz;
	float4 windRotation = bufferBranchMatrix.Load(baseTreeIndex + STREAM_OUT_WIND_ROT);
	
	// transform vertex
	float3x3 windRotationMatrix = quatToMatrix(windRotation);
	v.pos = lerp(
		v.pos,
		mul(v.pos.xyz, windRotationMatrix), bendWeightArray[0]);
	v.pos += worldPos;
	v.normal = mul(v.normal, (float3x3)mWorld);

	output.pos = mul(float4(v.pos, 1), mWorldViewProjection);
	output.diffuse = saturate(dot(-vLightDir, v.normal)) + 0.33f;

	return output;
}

//--------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------
float4 TreePS(VsOutput i) : SV_Target
{
	return i.diffuse;
}

//--------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------
//
int iHierarchyDepth = 2;
VertexShader streamInVSArray[] = {
	CompileShader(vs_4_0, StreamInVS(0)),
	CompileShader(vs_4_0, StreamInVS(1)),
	CompileShader(vs_4_0, StreamInVS(2)) };

VertexShader vsBuffer = CompileShader(vs_4_0, StreamOutVS());
GeometryShader vsBufferSO = ConstructGSWithSO(vsBuffer, 
	"TEXCOORD0.xyzw; TEXCOORD1.xyzw; TEXCOORD2.xyzw; TEXCOORD3.xyzw; TEXCOORD4.xyzw"); // VsStreamOut

//
// Simulation step
// Branch rotations are simulated and stored using stream-out functionality

technique10 StreamOutBranches
{    
    pass P0
    {      
        SetVertexShader(vsBuffer);
        SetGeometryShader(vsBufferSO);
        SetPixelShader(NULL);
        
        SetDepthStencilState(DisableDepth, 0);
    }
}

//
// Visualization step
// Branch rotations are read and applied to branch vertices

technique10 StreamInBranches
{
    pass P0
    {       
        SetVertexShader(streamInVSArray[iHierarchyDepth]);

        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, TreePS()));
        
        SetDepthStencilState(EnableDepth, 0);
    }
}
