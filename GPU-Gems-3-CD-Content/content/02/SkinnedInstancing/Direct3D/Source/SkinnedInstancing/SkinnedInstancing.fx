//----------------------------------------------------------------------------------
// File:   SkinnedInstancing.fx
// Author: Bryan Dudash
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// This is the shader code for the SkinnedInstancing sample.
//--------------------------------------------------------------------------------------

#ifndef MATRIX_PALETTE_SIZE_DEFAULT
#define MATRIX_PALETTE_SIZE_DEFAULT 50
#endif

#ifndef MAX_INSTANCE_CONSTANTS
#define MAX_INSTANCE_CONSTANTS 682
#endif

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

float4x4    g_mWorld : World;                               // World matrix for object
float4x4    g_mWorldViewProjection : WorldViewProjection;    // World * View * Projection matrix
float       g_TexOffset : TEXCOORDANIMFACTOR;

cbuffer animationvars
{
    float3x4    g_matrices[MATRIX_PALETTE_SIZE_DEFAULT];
}

struct PerInstanceData
{
    float4 world1;
    float4 world2;
    float4 world3;
    float4 color;
    uint4    animationData;
};

cbuffer cInstanceData
{
    PerInstanceData    g_Instances[MAX_INSTANCE_CONSTANTS];
}

cbuffer config
{
    uint g_InstanceMatricesWidth;
    uint g_InstanceMatricesHeight;
    float3      g_instanceColor = float3(1,1,1);
};

cbuffer cimmutable
{
    float3 g_lightPos = float3(0,50,0);
    float3 g_lightColor = (0.4,0.55,0.65);
    float  g_lightA1 = 180.f;
    float  g_lightA2 = 200.f;
    float4x4 g_Identity = {    1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
};

//--------------------------------------------------------------------------------------
// Textures & Samplers
//--------------------------------------------------------------------------------------
Texture2D g_txDiffuse           : DIFFUSE;
Texture2D g_txNormals           : NORMAL;
Texture2DArray g_txDiffuseArray : DIFFUSEARRAY;    // array of albedo for the meshes
Texture2D g_txAnimations        : ANIMATIONS;

SamplerState g_samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};
SamplerState g_samAniso
{
    Filter = ANISOTROPIC;
    AddressU = Wrap;
    AddressV = Wrap;
    MipMapLODBias = -1;
};
SamplerState g_samPoint
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

//--------------------------------------------------------------------------------------
// Vertex shader output structure
//--------------------------------------------------------------------------------------

struct A_to_VS
{
    float4 vPos         : POSITION;
    float3 vNormal      : NORMAL;
    float3 vTangent     : TANGENT;
    float2 vTexCoord0   : TEXCOORD0;
    float4 vBones       : BONES;
    float4 vWeights     : WEIGHTS;
    uint   InstanceId   : SV_InstanceID;
};

struct VS_to_PS
{
    float4 pos      : SV_Position;
    float3 color    : COLOR;
    float3 tex      : TEXTURE0;
    float3 norm     : NORMAL;
    float3 tangent    : TANGENT;
    float3 binorm    : BINORMAL;
    float3 worldPos : WORLDPOS;
};

float4x4 decodeMatrix(float3x4 encodedMatrix)
{
    return float4x4(    float4(encodedMatrix[0].xyz,0),
                        float4(encodedMatrix[1].xyz,0),
                        float4(encodedMatrix[2].xyz,0),
                        float4(encodedMatrix[0].w,encodedMatrix[1].w,encodedMatrix[2].w,1));
}

// Read a matrix(3 texture reads) from a texture containing animation data
float4x4 loadBoneMatrix(uint3 animationData,float bone)
{
    // Calculate a UV for the bone for this vertex
    float2 uv = float2(0,0);
    float4x4 rval = g_Identity;
    
    // if this texture were 1D, what would be the offset?
    uint baseIndex = animationData.x + animationData.y;
    baseIndex += (4*bone);    // 4*bone is since each bone is 4 texels to form a float4x4 
        
    // Now turn that into 2D coords
    uint baseU = baseIndex%g_InstanceMatricesWidth;
    uint baseV = baseIndex/g_InstanceMatricesWidth;
    uv.x = (float)baseU / (float)g_InstanceMatricesWidth;
    uv.y = (float)baseV / (float)g_InstanceMatricesHeight;
    
    // Note that we assume the width of the texture is an even multiple of the # of texels per bone,
    //     otherwise we'd have to recalculate the V component per lookup
    float2 uvOffset = float2(1.0/(float)g_InstanceMatricesWidth,0);

    float4 mat1 = g_txAnimations.SampleLevel( g_samPoint,float3(uv.xy,0),0);
    float4 mat2 = g_txAnimations.SampleLevel( g_samPoint,float3(uv.xy + uvOffset.xy,0),0);
    float4 mat3 = g_txAnimations.SampleLevel( g_samPoint,float3(uv.xy + 2*uvOffset.xy,0),0);
    
    // only load 3 of the 4 values, and deocde the matrix from them.
    rval = decodeMatrix(float3x4(mat1,mat2,mat3));
    
    return rval;
}

/*
    This shader is used for the instancing method.
*/
VS_to_PS CharacterAnimatedInstancedVS( A_to_VS input )
{
    VS_to_PS output;
    float3 vNormalWorldSpace;
    
    uint4 animationData = g_Instances[input.InstanceId].animationData;    
    
       // Our per instance data is stored in constants
    float4 worldMatrix1 = g_Instances[input.InstanceId].world1;
    float4 worldMatrix2 = g_Instances[input.InstanceId].world2;
    float4 worldMatrix3 = g_Instances[input.InstanceId].world3;
    float4 instanceColor = g_Instances[input.InstanceId].color;
    
    float4x4 finalMatrix;
    finalMatrix = input.vWeights.x * loadBoneMatrix(animationData,input.vBones.x);

    if(input.vWeights.y > 0) 
    {
        finalMatrix += input.vWeights.y * loadBoneMatrix(animationData,input.vBones.y);
        if(input.vWeights.z > 0)
        {
            finalMatrix += input.vWeights.z * loadBoneMatrix(animationData,input.vBones.z);
            if(input.vWeights.w > 0) finalMatrix += input.vWeights.w * loadBoneMatrix(animationData,input.vBones.w);    
        }
    }
    
    float4 vAnimatedPos     = mul(float4(input.vPos.xyz,1),finalMatrix);
    float4 vAnimatedNormal  = mul(float4(input.vNormal.xyz,0),finalMatrix);
    float4 vAnimatedTangent = mul(float4(input.vTangent.xyz,0),finalMatrix);

    // the whole model position, decode a little, we have compressed for space.
    float4x4 worldMatrix = decodeMatrix(float3x4(worldMatrix1,worldMatrix2,worldMatrix3));
    vAnimatedPos = mul(vAnimatedPos, worldMatrix);
    vAnimatedNormal = mul(float4(vAnimatedNormal.xyz,0),worldMatrix );
    vAnimatedTangent = mul(float4(vAnimatedTangent.xyz,0), worldMatrix );

    // Transform the position from object space to homogeneous projection space
    output.pos = mul(vAnimatedPos, g_mWorldViewProjection);

    // Transform the normal from object space to world space    
    output.norm = normalize(vAnimatedNormal.xyz);
    output.tangent = normalize(vAnimatedTangent.xyz);
    output.binorm = cross(output.norm,output.tangent);
    
    // Do the position too for lighting
    output.worldPos = float4(vAnimatedPos.xyz,1);
    output.tex.xy = float2(input.vTexCoord0.x,-input.vTexCoord0.y); 
    output.tex.z = 0;
    
    output.color = instanceColor.xyz;
    
    return output;
}

/*
    This shader is used for the non-instance case.
*/
VS_to_PS CharacterAnimatedVS( A_to_VS input )
{
    VS_to_PS output;
    float3 vNormalWorldSpace;
                
    float4x4 finalMatrix;
    finalMatrix = input.vWeights.x * decodeMatrix(g_matrices[input.vBones.x]);

    if(input.vWeights.y > 0) 
    {
        finalMatrix += input.vWeights.y * decodeMatrix(g_matrices[input.vBones.y]);
        if(input.vWeights.z > 0)
        {
            finalMatrix += input.vWeights.z * decodeMatrix(g_matrices[input.vBones.z]);
            if(input.vWeights.w > 0) finalMatrix += input.vWeights.w * decodeMatrix(g_matrices[input.vBones.w]);    
        }
    }
    
    float4 vAnimatedPos = mul(float4(input.vPos.xyz,1),finalMatrix);
    float4 vAnimatedNormal = mul(float4(input.vNormal.xyz,0),finalMatrix);
    float4 vAnimatedTangent = mul(float4(input.vTangent.xyz,0),finalMatrix);
        
    // Transform the position from object space to homogeneous projection space
    output.pos = mul(vAnimatedPos, g_mWorldViewProjection);

    // Transform the normal from object space to world space    
    output.norm = normalize(mul(vAnimatedNormal.xyz, (float3x3)g_mWorld)); // normal to world space
    output.tangent = normalize(mul(vAnimatedTangent.xyz, (float3x3)g_mWorld));
    output.binorm = cross(output.norm,output.tangent);
    
    // Do the position too for lighting
    output.worldPos = mul(float4(vAnimatedPos.xyz,1), g_mWorld);
    output.tex.xy = float2(input.vTexCoord0.x,-input.vTexCoord0.y); 
    output.tex.z = 0;
    
    output.color = g_instanceColor;
    
    return output;    
}

/* simple shader for our ground plane. */
VS_to_PS TerrainVS( float4 vPos : POSITION,
                         float3 vNormal : NORMAL,
                         float2 vTexCoord0 : TEXCOORD0)
{
    VS_to_PS output;
    
    // Transform the position from object space to homogeneous projection space
    output.pos = mul(float4(vPos.xyz,1), g_mWorldViewProjection);

    // Transform the normal from object space to world space    
    output.norm = normalize(mul(vNormal, (float3x3)g_mWorld)); // normal (world space)
    
    // Do the position too for lighting
    output.worldPos = mul(float4(vPos.xyz,1), g_mWorld);
    output.tex.xy = 10.f*float2(vTexCoord0.x,vTexCoord0.y); 

    return output;    
}

//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
float Kd = 0.89;
float4 ambiColor = float4(0.15,0.15,0.15,0);

float4 CharacterPS( VS_to_PS input) : SV_Target
{
    float4 result;
    float4 color = g_txDiffuse.Sample(g_samAniso,input.tex.xyz);
    
    // This interpolation allows us to use the alpha channel as a mask to not apply instance color
    // Can be used to mask out areas where you want color variation versus those areas you don't
    color = color * float4(lerp(float3(0.8,0.8,0.8),input.color.xyz,color.w),1);
  
    // simple diffuse lighting
    float3 Nn = normalize(input.norm);
    float3 lightVec = normalize(g_lightPos - input.worldPos);    
    float3 toLight = g_lightPos - input.worldPos.xyz;
    float dist = length(toLight);
    float3 Ln = toLight / dist;
    float ldn = max(0.0,dot(Ln,Nn));
    float4 diffContrib = Kd*(ldn) + ambiColor;
    float scale = g_lightA1/dist + g_lightA2 / (dist*dist);
    diffContrib = saturate(float4(g_lightColor.xyz,1)*diffContrib*scale);

    result = float4(color*diffContrib.xyz,1);    
    
    return result;
}

float4 TerrainPS( VS_to_PS input) : SV_Target
{
    float4 result;
    float4 albedo = g_txDiffuse.Sample(g_samAniso,input.tex.xy);
    float3 Nn = normalize(input.norm);
    float3 L = g_lightPos - input.worldPos;
    float dist = length(L.xz);
    float scale = saturate(g_lightA1/dist + g_lightA2 / (dist*dist));
    float3 Ln = normalize(L);
    
    float ldn = max(0.0,dot(Ln,Nn));
    float4 diffContrib = saturate(Kd*(ldn) + ambiColor);
    //result = float4(input.color,0)*albedo*diffContrib;
    result = scale * albedo*diffContrib;
    return result;
}

BlendState Opaque
{
    BlendEnable[0] = false;
};

RasterizerState Solid
{
    FillMode = Solid;
	MultisampleEnable = TRUE;
};

//--------------------------------------------------------------------------------------
// Renders scene to Render target using D3D10 Techniques
//--------------------------------------------------------------------------------------
technique10 RenderSceneWithAnimationInstanced
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, CharacterAnimatedInstancedVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, CharacterPS() ) );
        SetBlendState(Opaque,float4(0,0,0,0),0xffffffff);
        SetRasterizerState(Solid);
    }
}

technique10 RenderSceneWithAnimation
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, CharacterAnimatedVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, CharacterPS() ) );
        SetRasterizerState(Solid);
    }
}

technique10 RenderTerrain
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, TerrainVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, TerrainPS() ) );
        SetRasterizerState(Solid);
    }
}