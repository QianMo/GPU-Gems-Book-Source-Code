texture heightVals;
texture coarserNormalsTexture;

uniform float  OneOverSize;
uniform float  Size;
uniform float2 Viewport;
uniform float2 ToroidalOrigin;
uniform float2 TextureOffset;
uniform float2 ScaleFac;

uniform sampler ElevationSampler = sampler_state           
{
    Texture   = (heightVals);
    MipFilter = None;
    MinFilter = Point;
    MagFilter = Point;
    AddressU  = Wrap;
    AddressV  = Wrap;
};

uniform sampler CoarseLeveNormalSampler = sampler_state         
{
    Texture   = (coarserNormalsTexture);
    MipFilter = None;
    MinFilter = Linear;
    MagFilter = Linear;
    AddressU  = Wrap;
    AddressV  = Wrap;
};

struct OUTPUT
{
    vector position   : POSITION;
    float2 texcoords  : TEXCOORD0;
};

OUTPUT ComputeNormalsVS(float3 position : POSITION, float2 texcoords : TEXCOORD0)
{
    OUTPUT output;
    output.position   = float4(float2(position.x,-position.y) + float2(-1.0, 1.0)/Viewport,0.0,1.0); 
    output.texcoords  = texcoords*Size;
    
    return output;
}

float4 ComputeNormalsPS(OUTPUT input) : COLOR
{   
    input.texcoords  = floor(input.texcoords);
    
    // compute a local tangent vector along the X axis
    float2 texcoord0 = float2(input.texcoords.x-1, input.texcoords.y)*OneOverSize;
    float z1         = floor(tex2D(ElevationSampler, texcoord0+0.5*OneOverSize));  
    float2 texcoord1 = float2(input.texcoords.x+1, input.texcoords.y)*OneOverSize;
    float z2         = floor(tex2D(ElevationSampler, texcoord1+0.5*OneOverSize));
    float zx         = z2-z1;
    
    // compute a local tangent vector along the Y axis
    texcoord0 = float2(input.texcoords.x, input.texcoords.y-1)*OneOverSize;
    z1        = floor(tex2D(ElevationSampler, texcoord0+0.5*OneOverSize)); 
    texcoord1 = float2(input.texcoords.x, input.texcoords.y+1)*OneOverSize;
    z2        = floor(tex2D(ElevationSampler, texcoord1+0.5*OneOverSize));
    float zy = z2-z1;

    // The normal is now the cross product of the two tangent vectors
    // normal = (2*sx, 0, zx) x (0, 2*sy, zy), where sx, sy = gridspacing in x, y
    // the normal below has n_z = 1
    // ScaleFac = -0.5*ScaleFac.z/ScaleFac.x, -0.5*ScaleFac.z/ScaleFac.y
    float2 normalf = float2(zx*ScaleFac.x, zy*ScaleFac.y);
        
    // pack coordinates in [-1, +1] range to [0, 1] range
    normalf = normalf/2 + 0.5;
    
    // lookup the normals at the coarser level and pack it in normal map for current level
    float2 texcoordc = frac(input.texcoords/Size-ToroidalOrigin+1)/2.0+TextureOffset;
    float2 normalc = tex2D(CoarseLeveNormalSampler, texcoordc+0.5*OneOverSize);
    return float4(normalf.xy, normalc.xy);
}

technique ComputeNormals
{
    pass P0
    {
        VertexShader = compile vs_3_0 ComputeNormalsVS();
        PixelShader  = compile ps_2_a ComputeNormalsPS();
    }
}

