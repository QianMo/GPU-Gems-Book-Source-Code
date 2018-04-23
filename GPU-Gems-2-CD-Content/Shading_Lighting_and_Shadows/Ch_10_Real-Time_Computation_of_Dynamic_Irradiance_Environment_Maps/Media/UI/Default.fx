//
// Default effect
// Single texture
// Copyright (c) 2000 Microsoft Corporation. All rights reserved.
//

texture Texture0 < string Name = ""; >;

float4 Ambient : Ambient 
< 
    string Object = "Geometry"; 
> = float4( 0.4f, 0.4f, 0.4f, 1.0f );
   
float4 Diffuse : Diffuse
< 
    string Object = "Geometry"; 
> = float4( 0.6f, 0.6f, 0.6f, 1.0f );    

float4 Specular : Specular
< 
    string Object = "Geometry"; 
> = float4( 1.0f, 1.0f, 1.0f, 1.0f );

float4 Emissive : Emissive
<
    string Object = "Geometry"; 
> = float4( 0.0f, 0.0f, 0.0f, 1.0f ); 

float  Power : SpecularPower
< 
    string Object = "Geometry"; 
> = 4;

float4 g_vLightColor : Diffuse
< 
    string Object = "Light0"; 
> = float4( 1.0f, 1.0f, 1.0f, 1.0f );

float4 g_vLightDirection: Direction
< 
    string Object = "Light0"; 
> = float4( 0.5f, 0.75f, 1.0f, 1.0f );   

float4 g_vCameraPosition : Position
< 
    string Object = "Camera"; 
> = float4( 0.0f, 0.0f, 0.0f, 1.0f ); 
            
float4x4 g_mWorld : World;           // World matrix
//float4x4 g_mView : View;             // View matrix
//float4x4 g_mProjection : Projection; // Projection matrix

float4x4 g_mWorldViewProjection : WorldViewProjection; // World * View * Projection matrix



//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
sampler MeshTextureSampler = 
sampler_state
{
    Texture = <Texture0>;    
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};


//--------------------------------------------------------------------------------------
// Name: VS
// Type: Vertex Shader
// Desc: Projection transform
//--------------------------------------------------------------------------------------
void VS( float4 vPosObj: POSITION,
         float3 vNormalObj: NORMAL,
         float2 vTexCoordIn: TEXCOORD0,
         out float4 vPosProj: POSITION,
         out float4 vColorOut: COLOR0,
         out float2 vTexCoordOut: TEXCOORD0
        )
{
    // Transform the position into projected space for display and world space for lighting
    vPosProj = mul( vPosObj, g_mWorldViewProjection ); 
    //PosProj = mul( vPosObj, g_mWorld );
    //vPosProj = mul( vPosProj, g_mView );
    //vPosProj = mul( vPosProj, g_mProjection );
   
    
    // Transform the normal into world space for lighting
    float3 vNormalWorld = mul( vNormalObj, (float3x3)g_mWorld );
    
    // Compute ambient and diffuse lighting
    vColorOut = g_vLightColor * Ambient;
    vColorOut += g_vLightColor * Diffuse * saturate( dot( g_vLightDirection.xyz, vNormalWorld ) );
   
    // Pass the texture coordinate
    vTexCoordOut = vTexCoordIn;
}



//--------------------------------------------------------------------------------------
// Name: PS
// Type: Pixel Shader
// Desc: Modulate the texture by the vertex color
//--------------------------------------------------------------------------------------
void PS( float4 vColorIn: COLOR0,
                float2 vTexCoord: TEXCOORD0,
                out float4 vColorOut: COLOR0 )
{  
    // Sample and modulate the texture
    //float4 vTexColor = tex2D( MeshTextureSampler, vTexCoord );
    //vColorOut = vColorIn * vTexColor;
    vColorOut = vColorIn;
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique T0
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS();    
        PixelShader = compile ps_1_1 PS();    
    }
}
