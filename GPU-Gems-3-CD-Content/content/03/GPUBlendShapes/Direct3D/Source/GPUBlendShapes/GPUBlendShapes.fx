//----------------------------------------------------------------------------------
// File:   GPUBlendShapes.fx
// Author: Tristan Lorach
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

#if 1
typedef half		fpart;
typedef half2		fpart2;
typedef half3		fpart3;
typedef half4		fpart4;
typedef half4x3		fpart4x3;
#else
typedef float		fpart;
typedef float2		fpart2;
typedef float3		fpart3;
typedef float4		fpart4;
typedef float4x3	fpart4x3;
#endif

cbuffer cb0 : register(b0)
{
  float4x4 Proj;
  float4x4 WorldViewProj;
  float4x4 WorldViewIT;
  float4x4 ViewProj : ViewProj;
  float4x4 World : World;
  float4x4 WorldIT : WorldIT;
  float4x4 WorldView : WorldView;

  float weightBS[4] : BSWEIGHT; // this is where we'll put the weight for Blendshape pass

  float3 eyeWorld : EyeWorld; // semantic do hide the param in the UI
  float fresExp
  <
  float uimin = 0.0;
  float uimax = 5.0;
  > = 3.0;
  float ReflStrength
  <
  float uimin = 0.0;
  float uimax = 8.0;
  > = 1.4;
  float bumpStrength
  <
  float uimin = 0.0;
  float uimax = 5.0;
  > = 2.0;
  float glowStrength
  <
  float uimin = 0.0;
  float uimax = 2.0;
  > = 0.5;
  float glowExp
  <
  float uimin = 0.0;
  float uimax = 8.0;
  > = 3.0;
  };
  float fadeOutLimit = 4.8;

SamplerState sampler_dawn
{
  AddressU = Wrap;
  AddressV = Clamp;
  Filter = Min_Mag_Mip_Linear;
};
Texture2D texture_face   < string file="MediaFor_GPUBlendShapes/f_head_front_col.dds"; >;
Texture2D texture_face_gloss < string file="MediaFor_GPUBlendShapes/f_skin_spec_shift.dds"; >;
Texture2D texture_face_bump < string file="MediaFor_GPUBlendShapes/f_head_front_bmp_norm_rgb8.dds"; >;

Texture2D texture_bhead  < string file="MediaFor_GPUBlendShapes/f_head_back_col.dds"; >;
Texture2D texture_btorso < string file="MediaFor_GPUBlendShapes/f_torso_back_col.dds"; >;
Texture2D texture_ftorso < string file="MediaFor_GPUBlendShapes/f_torso_front_col.dds"; >;
Texture2D texture_ulbarm < string file="MediaFor_GPUBlendShapes/f_ularm_back_col.dds"; >;
Texture2D texture_ulfarm < string file="MediaFor_GPUBlendShapes/f_ularm_front_col.dds"; >;
Texture2D texture_urbarm < string file="MediaFor_GPUBlendShapes/f_urarm_back_col.dds"; >;
Texture2D texture_urfarm < string file="MediaFor_GPUBlendShapes/f_urarm_front_col.dds"; >;

Texture2D texture_eye_col < string file="MediaFor_GPUBlendShapes/f_eye_col.dds"; >;

TextureCube texture_env_diff < string file="CM_Forest_diffuse.dds"; >;
TextureCube texture_env_spec < string file="CM_Forest_specular.dds"; >;




//////////////////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////////////////
DepthStencilState depthEnabled
{
  DepthEnable = true;
  DepthWriteMask = All;
  DepthFunc = Less;
};
DepthStencilState depthDisabled
{
  DepthEnable = false;
  DepthWriteMask = Zero;
};
BlendState blendOFF
{
  BlendEnable[0] = false;
};
BlendState blendON
{
  BlendEnable[0] = true;
  SrcBlend = Src_Color;
  DestBlend = One;
};
BlendState blendAlpha
{
  BlendEnable[0] = true;
  SrcBlend = Src_Alpha;
  DestBlend = Inv_Src_Alpha;
};
RasterizerState cullDisabled
{
  CullMode = None;
  MultisampleEnable = TRUE;
};
RasterizerState RStateMSAA
{
  MultisampleEnable = TRUE;
};

//////////////////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////////////////
struct Head_VSIn
{
  float3 pos : position;
  float3 normal : normal;
  float3 tangent  : tangent;
  float2 tc : texcoord0;
};
struct Head_VSOut
{
  fpart4 pos : SV_Position;
  fpart3 tc : texcoord0; // z contains an additional value : y vtx in object space (for fade-out)
  fpart3 normal : normal;
  fpart3 tangent  : texcoord1;
  fpart3 binormal  : texcoord2;
  fpart3 dirtoeye  : texcoord3;
};
struct Lash_VSOut
{
  fpart4 pos : SV_Position;
  fpart4 color : color;
};

struct Dawn_PSOut
{
  fpart4 color : SV_Target0;
};


/*********************************************************************************/
/*********************************************************************************/
//
// Technique using Stream-Out buffer.
// each pass is working on 2 additional expressions.
// we use Slots in order to assemble attributes together at each pass
//
/*********************************************************************************/
/*********************************************************************************/
struct Face_VSIn
{
  float3 pos : position;
  float3 normal : normal;
  float3 tangent : tangent;
  float2 tc : texcoord0;
  float3 bsP0 : bs_position0; // from Slot #1
  float3 bsN0 : bs_normal0;   // from Slot #1
  float3 bsT0 : bs_tangent0;  // from Slot #1
  
  float3 bsP1 : bs_position1; // from Slot #2
  float3 bsN1 : bs_normal1;   // from Slot #2
  float3 bsT1 : bs_tangent1;  // from Slot #2
  
  float3 bsP2 : bs_position2; // from Slot #3
  float3 bsN2 : bs_normal2;   // from Slot #3
  float3 bsT2 : bs_tangent2;  // from Slot #3
  
  float3 bsP3 : bs_position3; // from Slot #4
  float3 bsN3 : bs_normal3;   // from Slot #4
  float3 bsT3 : bs_tangent3;  // from Slot #4
};
struct Face_VSStreamOut
{
  fpart3 pos : position;
  fpart3 normal : normal;
  fpart3 tangent  : texcoord1;
  fpart2 tc : texcoord0;
};

Face_VSStreamOut VSFace(Face_VSIn input)
{
  Face_VSStreamOut output;
  output.pos =      input.pos 
                + (weightBS[0].xxx*input.bsP0) 
                + (weightBS[1].xxx*input.bsP1)
                + (weightBS[2].xxx*input.bsP2)
                + (weightBS[3].xxx*input.bsP3);
  output.normal =   input.normal 
                + (weightBS[0].xxx*input.bsN0) 
                + (weightBS[1].xxx*input.bsN1)
                + (weightBS[2].xxx*input.bsN2)
                + (weightBS[3].xxx*input.bsN3);
  output.tangent =  input.tangent 
                + (weightBS[0].xxx*input.bsT0) 
                + (weightBS[1].xxx*input.bsT1)
                + (weightBS[2].xxx*input.bsT2)
                + (weightBS[3].xxx*input.bsT3);
  output.tc =       input.tc;

  return output;
}
GeometryShader gsStreamOutBlendedVtx = ConstructGSWithSO( CompileShader( vs_4_0, VSFace() )
  , "position.xyz; normal.xyz; texcoord1.xyz; texcoord0.xy" );
VertexShader vsFaceCompiled = CompileShader( vs_4_0, VSFace() );

//////////////////////////////////////////////////////////////////////////////////
// technique
//////////////////////////////////////////////////////////////////////////////////

technique10 nv_f_head_frontSO
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetDepthStencilState( depthDisabled, 0 );
    SetVertexShader( vsFaceCompiled );
    SetGeometryShader( gsStreamOutBlendedVtx );
    SetPixelShader(NULL);
  }
}

/*********************************************************************************/
/*********************************************************************************/
//
// Shaders for various parts of the mesh
// - for the parts without any blendshapes
// - for the final pass of the mesh with blendshape using stream-out
// - for the technique using Buffer<> template to compute blendshapes
//
/*********************************************************************************/
/*********************************************************************************/
Head_VSOut VSHead(Head_VSIn input)
{
  Head_VSOut output;
  fpart3 binorm = cross(input.normal, input.tangent);
  fpart4 wp = mul(fpart4(input.pos,1), World);
  fpart3 we = eyeWorld;

  output.pos        = mul(wp, ViewProj); 
  output.tc.xy      = input.tc;
  output.tc.z       = input.pos.y; // used to do a fade-out of the model in y...
  output.normal     = mul(input.normal, WorldIT);
  output.tangent    = mul(input.tangent, WorldIT);
  output.binormal   = mul(binorm, WorldIT);
  output.dirtoeye   = normalize(we-wp);
  return output;
}
//////////////////////////////////////////////////////////////////////////////////
// 
// 
//
//////////////////////////////////////////////////////////////////////////////////
Dawn_PSOut PSDawn(Head_VSOut input, uniform Texture2D difftex, uniform float4 col, uniform bool bUseGloss, uniform float glossValue)
{
  Dawn_PSOut output;
  fpart3 Nn = normalize(input.normal); // World normal : to hit the right texel in the cubemap
  fpart3 Tn = normalize(input.tangent);
  fpart3 Bn = normalize(input.binormal);
  fpart2 bumps;
  bumps = bumpStrength * (2.0 * texture_face_bump.Sample(sampler_dawn, input.tc.xy).rg - 1.0);
  fpart3 Nb = normalize(Nn + (bumps.x * Tn + bumps.y * Bn));

  fpart3 Vn = normalize(input.dirtoeye);
  fpart3 Rn = reflect(Vn, Nb);
  fpart3 Cdiff = texture_env_diff.Sample(sampler_dawn, Nb);

  fpart ndotv = dot(Nb, Vn);
  fpart Kglow = pow(1.0-abs(ndotv), glowExp) * glowStrength;
  fpart4 Cglow = Kglow * texture_env_diff.Sample(sampler_dawn, -Vn);
  fpart Kfres = pow(1.0-abs(ndotv), fresExp) * ReflStrength;
  fpart3 Gloss = bUseGloss ? texture_face_gloss.Sample(sampler_dawn, input.tc.xy) : glossValue;
  fpart4 Cspec = Kfres * texture_env_spec.Sample(sampler_dawn, Rn);

  fpart4 Tdiff = col * difftex.Sample(sampler_dawn, input.tc.xy);
  output.color.rgb  = Tdiff.rgb * Cdiff + (Cglow * Gloss.r) + (Cspec * Tdiff.a);
  output.color.a = saturate(10.0*(input.tc.z - fadeOutLimit));
  return output;
}

Dawn_PSOut PSDawn_NoTex(Head_VSOut input, uniform float4 col, uniform float glossValue)
{
  Dawn_PSOut output;
  fpart3 Nn = normalize(input.normal); // World normal : to hit the right texel in the cubemap

  fpart3 Vn = normalize(input.dirtoeye);
  fpart3 Rn = reflect(Vn, Nn);
  fpart3 Cdiff = texture_env_diff.Sample(sampler_dawn, Nn);

  fpart ndotv = dot(Nn, Vn);
  fpart Kglow = pow(1.0-abs(ndotv), glowExp) * glowStrength;
  fpart4 Cglow = Kglow * texture_env_diff.Sample(sampler_dawn, -Vn);
  fpart Kfres = pow(1.0-abs(ndotv), fresExp) * ReflStrength;
  fpart4 Cspec = col.a * Kfres * texture_env_spec.Sample(sampler_dawn, Rn);

  output.color.rgb  = col.rgb * Cdiff + (Cglow * glossValue) + (Cspec * col.a);
  output.color.a = saturate(10.0*(input.tc.z - fadeOutLimit));
  return output;
}

//////////////////////////////////////////////////////////////////////////////////
// 
// Techniques for static parts of the mesh
//
//////////////////////////////////////////////////////////////////////////////////
VertexShader vsHeadCompiled = CompileShader( vs_4_0, VSHead() );

technique10 nv_f_head_front
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetDepthStencilState( depthEnabled, 0 );
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_face, float4(1,1,1,1), true, 1.0) ) );
  }
}

technique10 nv_f_head_back
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetDepthStencilState( depthEnabled, 0 );
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    //SetGeometryShader( gsCompiled );
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_bhead, float4(1,1,1,1), true, 1.0) ) );
  }
}

technique10 nv_f_torso_front
{
  pass p0
  {
    SetBlendState(blendAlpha, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_ftorso, float4(1,1,1,1), false, 1.0) ) );
  }
}
technique10 nv_f_torso_back
{
  pass p0
  {
    SetBlendState(blendAlpha, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_btorso, float4(1.0,1,1.0,1), false, 1.0) ) );
  }
}
technique10 nv_f_ularm_front
{
  pass p0
  {
    SetBlendState(blendAlpha, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_ulfarm, float4(1,1,1,1), false, 1.0) ) );
  }
}
technique10 nv_f_ularm_back
{
  pass p0
  {
    SetBlendState(blendAlpha, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_ulbarm, float4(1,1,1,1), false, 1.0) ) );
  }
}
technique10 nv_f_urarm_front
{
  pass p0
  {
    SetBlendState(blendAlpha, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_urfarm, float4(1,1,1,1), false, 1.0) ) );
  }
}
technique10 nv_f_urarm_back
{
  pass p0
  {
    SetBlendState(blendAlpha, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_urbarm, float4(1,1,1,1), false, 1.0) ) );
  }
}
technique10 nv_f_tearduct1
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.777,0.444,0.383,1), 0.0) ) );
  }
}
technique10 nv_f_tearduct2
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.777,0.444,0.383,1), 0.0) ) );
  }
}
technique10 nv_f_eye_inner
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_eye_col, float4(1,1,1,1), false, 0.0) ) );
  }
}

technique10 nv_f_teeth_upper
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.9,0.9,0.9,0.0), 0.0) ) );
  }
}
technique10 nv_f_teeth_lower
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.9,0.9,0.9,0.0), 0.0) ) );
  }
}
technique10 nv_f_tongue
{
  pass p0
  {
    SetDepthStencilState( depthEnabled, 0 );
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.6,0.2,0.2,0.5), 0.0) ) );
  }
}

technique10 nv_f_lash
{
  pass p0
  {
    SetDepthStencilState( depthEnabled, 0 );
    SetVertexShader( vsHeadCompiled );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.0,0.0,0.0,0.0), 0.0) ) );
  }
}

/*********************************************************************************/
/*********************************************************************************/
//
// Technique using Buffer<> template. No need of Stream-out
//
/*********************************************************************************/
/*********************************************************************************/
cbuffer blendshapeInfos
{
    int  bsPitch;
    int  numBS = 1;
};
Buffer<uint>    bsOffsets;
Buffer<float>   bsWeights;
Buffer<float3>  bsVertices;

Head_VSOut VSFaceBufferTemplate(Head_VSIn input, uint vertexID : SV_VertexID)
{
  Head_VSOut output;
  float3 pos =      input.pos;
  float3 normal =   input.normal;
  float3 tangent =  input.tangent;
  float3 dp, dn, dt;
  for(int i=0; i<numBS; i++)
  {
      uint  offset = bsPitch * bsOffsets.Load(i);
      float weight = bsWeights.Load(i);
      dp = bsVertices.Load(offset + 3*vertexID+0);
      dn = bsVertices.Load(offset + 3*vertexID+1);
      dt = bsVertices.Load(offset + 3*vertexID+2);

      pos     += dp * weight;
      normal  += dn * weight;
      tangent += dt * weight;
  }
  fpart3 binorm = cross(input.normal, input.tangent);
  fpart4 wp = mul(fpart4(pos,1), World);
  fpart3 we = eyeWorld;

  output.pos        = mul(wp, ViewProj); 
  output.tc.xy      = input.tc;
  output.tc.z       = input.pos.y; // used to do a fade-out of the model in y...
  output.normal     = mul(input.normal, WorldIT);
  output.tangent    = mul(input.tangent, WorldIT);
  output.binormal   = mul(binorm, WorldIT);
  output.dirtoeye   = normalize(we-wp);

  return output;
}

Dawn_PSOut PSDawn_dbg(Head_VSOut input, uniform Texture2D difftex, uniform float4 col, uniform bool bUseGloss, uniform float glossValue)
{
  Dawn_PSOut output;
  output.color.rgb  = input.normal;
  output.color.a = 1.0;
  return output;
}
//////////////////////////////////////////////////////////////////////////////////
//
// Technique
//
//////////////////////////////////////////////////////////////////////////////////
technique10 nv_f_head_front_buffertemplate
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetDepthStencilState( depthEnabled, 0 );
    SetRasterizerState(RStateMSAA);
    SetVertexShader( CompileShader( vs_4_0, VSFaceBufferTemplate() ) );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn(texture_face, float4(1,1,1,1), true, 1.0) ) );
  }
}
technique10 nv_f_teeth_lower_buffertemplate
{
  pass p0
  {
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetRasterizerState(RStateMSAA);
    SetVertexShader( CompileShader( vs_4_0, VSFaceBufferTemplate() ) );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.9,0.9,0.9,0.0), 0.0) ) );
  }
}
technique10 nv_f_tongue_buffertemplate
{
  pass p0
  {
    SetDepthStencilState( depthEnabled, 0 );
    SetBlendState(blendOFF, float4(1.0, 1.0, 1.0, 1.0) ,0xffffffff);
    SetRasterizerState(RStateMSAA);
    SetVertexShader( CompileShader( vs_4_0, VSFaceBufferTemplate() ) );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.6,0.2,0.2,0.5), 0.0) ) );
  }
}

technique10 nv_f_lash_buffertemplate
{
  pass p0
  {
    SetDepthStencilState( depthEnabled, 0 );
    SetRasterizerState(RStateMSAA);
    SetVertexShader( CompileShader( vs_4_0, VSFaceBufferTemplate() ) );
    SetGeometryShader(NULL);
    SetPixelShader( CompileShader( ps_4_0, PSDawn_NoTex(float4(0.0,0.0,0.0,0.0), 0.0) ) );
  }
}

