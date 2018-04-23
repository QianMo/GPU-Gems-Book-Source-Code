
//////////////////////////////////////////////////////////////////////////////
// Effect parameters /////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Global wood desctiption
// Wood Parameters

string XFile = "TexTeapot.x";
float ringscale < string UIName = "Ring Scale";
             string UIWidget = "Numeric";
             float UIStep = 5.0;
             string UIHelp = "Ring Density";     
        > = 5.0f;
float point_scale < string UIName = "Point Scale";
             string UIWidget = "Numeric";
             float UIStep = 0.05;
             string UIHelp = "Point Scale";     
        > = 0.05f, 
    turbulence < string UIName = "Turbulence";
             string UIWidget = "Numeric";
             float UIStep = 0.05;
             string UIHelp = "Turbulence";     
        > = 0.3f;

//string XFile = "bust.x";
//float ringscale = 30.0f;
//float point_scale = 0.1f, turbulence = 0.8f;


//string XFile = "sphere.x";
//float ringscale = 15.0f;
//float point_scale = 1.0f, turbulence = 1.0f;


//float3 lightwood = {0.3f, 0.12f, 0.03f};
//float3 darkwood  = {0.05f, 0.01f, 0.005f};

float3 lightwood < string UIName = "Light Wood Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The color of the wood between the rings";     
        > = {0.75f, 0.4f, 0.15f};
float3 darkwood < string UIName = "Dark Wood Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The color of the rings";     
        > = {0.05f, 0.05f, 0.05f};

float3 wood_ambient_color < string UIName = "Ambient Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "Ambient Color of the wood";     
        > = {0.3f, 0.3f, 0.3f};
float3 wood_diffuse_color < string UIName = "Diffuse Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "Diffuse Color of the wood";     
        > = {1.0f, 1.0f, 1.0f};//{0.9f, 0.6f, 0.1f};


string t0 = "HLSL Wood";

float4 BCLR = {0.4f, 0.4f, 0.4f, 1.0f};    // Background Color

// GLOBALS for lighting
// Single Directional Diffuse Light
float3 g_lhtDir
<
    string UIDirectional = "Light Direction";
> = {-0.5f, -1.0f, 1.0f};    //light Direction

float3 g_lhtCol
<
    string UIColor3 = "Light Color";
> = {0.65f, 0.65f, 0.65f}; // Light Diffuse

float3x3 g_mWld : World;    // World
float4x4 g_mTot : WorldViewProjection;    // Total

// Viewer constant
float3  g_ViewDir       = {0.0f,0.0f,-1.0f};   // viewer direction

//////////////////////////////////////////////////////////////////////////////
// Procedural Texture Vertex Shader  /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct VS_INPUT
{
    float3  Pos     : POSITION;
    float3  Normal  : NORMAL;
    float3  Tex0    : TEXCOORD0;
};

struct VS_OUTPUT
{
    float4  Pos     : POSITION;
    float4  noise3d : TEXCOORD0;               
    float4  noisePos: TEXCOORD1;           
    float3  Normal  : TEXCOORD2;       
};

VS_OUTPUT ProcTexTransformGeometry(VS_INPUT i)
{
    float4      tmp = {0.0f, 0.0f, 0.0f, 1.0f}; 
    float3      ShadowColor, color;
    float3      wNormal;
    VS_OUTPUT   o=(VS_OUTPUT)0;

    // Project
    tmp.xyz     = i.Pos;
    o.Pos       = mul( tmp, g_mTot);
    o.noisePos  = tmp * point_scale; 
    o.noise3d   = tmp * point_scale * turbulence;        
    wNormal     = mul( i.Normal, g_mWld );
    
    // normalize normals?
//    wNormal = normalize(wNormal);
    o.Normal  = wNormal;

    return o;
}

//////////////////////////////////////////////////////
texture tNSEVol
<
 string name = "noisevol.dds";
 string type = "volume";
>;

uniform sampler   noiseVol_sampler = 
sampler_state 
{
    texture = <tNSEVol>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = Point;

    AddressU = Wrap;
    AddressV = Wrap;
    AddressW = Wrap;
};



//////////////////////////////////////////////////////////////////////////////
// Wood Shader ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct PS_INPUT
{
    float4  noise3d  : TEXCOORD0;           
    float4  noisePos : TEXCOORD1;           
    float3  Normal   : TEXCOORD2;           
};

struct PS_OUTPUT
{
    float4 Col   : COLOR;
};

float3 LambertDiffuse(float3 Normal, float3 lhtDir, float3 lhtCol)
{
    float   cosang;
    
    // N.L Clamped
    cosang = max(0.0f, dot(Normal, -lhtDir.xyz));    
       
    // propogate scalar result to vector
    return (cosang * lhtCol);
}

float3 RoughSpecular(float3 Normal, float3 lhtDir, float3 lhtCol, float roughness, float specpow)
{
    float   cosang;
    float3  H = g_ViewDir -lhtDir.xyz;
    H = normalize(H);
    
    // N.H
    cosang = dot(Normal, H);    
    
    // raise to a power
    cosang = pow(cosang, specpow);    
  
    // Use Light diff color as specular color too
    return (roughness * cosang * lhtCol);
}

float3 RoughSpookySpecular(float3 Normal, float3 lhtDir, float3 lhtCol, float roughness, float specpow)
{
    float   cosang;
    float3  H = g_ViewDir -lhtDir.xyz;
    H = normalize(H);
    
    // N.H
    cosang = dot(Normal, H);    
    
    // raise to a power
    cosang = pow(cosang, specpow);    
  
    // Use Light diff color as specular color too
    return ((1.0f/roughness) * cosang * lhtCol);
}

PS_OUTPUT Wood(const PS_INPUT v, uniform float spookyness)
{   
    PS_OUTPUT o;   
    float4    colRes;
    float3    ppNormal = normalize(v.Normal);

    /* Perturb P to add irregularity */
    float3 PP = v.noisePos;
    PP += tex3D(noiseVol_sampler, v.noise3d) * 0.1f;

    /* Compute radial distance r from PP to axis of tree */
    float r = sqrt(PP.y * PP.y + PP.z * PP.z);

    
    /* Map radial distance r into ring position [0,1] */
    r *= ringscale;
    r += abs(v.noise3d.z);
    r  = frac(r);

    colRes.w = 1.0f;
        
    /* Shade using r to vary brightness of wood grain */
    colRes.rgb = (wood_ambient_color + wood_diffuse_color * LambertDiffuse(ppNormal, g_lhtDir, g_lhtCol) ) 
                * lerp(darkwood, lightwood, r);
    if (spookyness==1.0f)
       colRes.rgb = colRes.rgb + RoughSpookySpecular(ppNormal, g_lhtDir, g_lhtCol, abs(r), 16);
    else
        colRes.rgb = colRes.rgb + RoughSpecular(ppNormal, g_lhtDir, g_lhtCol, abs(r), 64);

//    colRes.rgb = v.noise3d;
    o.Col = colRes;   

    return o;
}  

   
//////////////////////////////////////////////////////////////////////////////
// Techniques                      ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

technique t0
{
    pass p0
    {        
        VertexShader = compile vs_2_0 ProcTexTransformGeometry();
        
        PixelShader = compile ps_2_0 Wood(0.0f);
    }
}

technique t1
{
    pass p0
    {        
        VertexShader = compile vs_2_0 ProcTexTransformGeometry();
        
        PixelShader = compile ps_2_0 Wood(1.0f);
    }
}

