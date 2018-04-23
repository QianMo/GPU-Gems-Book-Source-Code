// texture
texture coarseLevelTexture;
texture residualTexture;
texture lookupTexture;

// Parameters 
uniform float2 TextureOffset;
uniform float  Size;
uniform float  OneOverSize;
uniform float2 Viewport;

uniform sampler CoarseLevelElevationSampler = sampler_state         // coarser level height sampler
{
    Texture   = (coarseLevelTexture);
    MipFilter = None;
    MinFilter = Point;
    MagFilter = Point;
    AddressU  = Wrap;
    AddressV  = Wrap;
};


uniform sampler ResidualSampler = sampler_state         
{
    Texture   = (residualTexture);
    MipFilter = None;
    MinFilter = Point;
    MagFilter = Point;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

uniform sampler LookupSampler = sampler_state
{
    Texture   = (lookupTexture);
    MipFilter = None;
    MinFilter = Point;
    MagFilter = Point;
    AddressU  = Wrap;
    AddressV  = Wrap;
};

struct OUTPUT
{
    vector position   : POSITION;
    float2 texcoords  : TEXCOORD0;
};

OUTPUT UpSampleVS(float3 position : POSITION, float2 texcoords : TEXCOORD0)
{
    OUTPUT output;
    output.position   = float4(float2(position.x,-position.y) + float2(-1.0, 1.0)/Viewport,0.0,1.0); 
    output.texcoords  = texcoords*Size;
    
    return output;
}

float4 UpsamplePS(float2 p_uv : TEXCOORD0) : COLOR
{
    float residual = tex2D(ResidualSampler, p_uv*OneOverSize);  
    
    p_uv = floor(p_uv);
    float2 p_uv_div2 = p_uv/2;
    float2 lookup_tij = p_uv_div2+1; 
    float4 maskType = tex2D(LookupSampler, lookup_tij);     
          
    matrix maskMatrix[4];
    maskMatrix[0] = matrix(0, 0, 0, 0,
                           0, -1.0f/16.0f, 0, 0,
                           0, 0, 0, 0,
                           1.0f/256.0f, -9.0f/256.0f, -9.0f/256.0f, 1.0f/256.0f);
                           
    maskMatrix[1] = matrix(0, 1, 0, 0,
                           0, 9.0f/16.0f, 0, 0,
                           -1.0f/16.0f, 9.0f/16.0f, 9.0f/16.0f, -1.0f/16.0f,
                           -9.0f/256.0f, 81.0f/256.0f, 81.0f/256.0f, -9.0f/256.0f);                        
                           
    maskMatrix[2] = matrix(0, 0, 0, 0,
                           0, 9.0f/16.0f, 0, 0,
                           0, 0, 0, 0,
                           -9.0f/256.0f, 81.0f/256.0f, 81.0f/256.0f, -9.0f/256.0f);
                           
    maskMatrix[3] = matrix(0, 0, 0, 0,
                           0, -1.0f/16.0f, 0, 0,
                           0, 0, 0, 0,
                           1.0f/256.0f, -9.0f/256.0f, -9.0f/256.0f, 1.0f/256.0f);

    float2 offset = float2(dot(maskType.bgra, float4(1, 1.5, 1, 1.5)), dot(maskType.bgra, float4(1, 1, 1.5, 1.5)));
    
    float z_predicted=0;
    offset = (p_uv_div2-offset+0.5)*OneOverSize+TextureOffset;
    for(int i = 0; i < 4; i++) {
        float zrowv[4];
        for (int j = 0; j < 4; j++) {
                float2 vij    = offset+float2(i,j)*OneOverSize;
                zrowv[j]      = tex2D(CoarseLevelElevationSampler, vij);
        }
        
        vector mask = mul(maskType.bgra, maskMatrix[i]);
        vector zrow = vector(zrowv[0], zrowv[1], zrowv[2], zrowv[3]);
        zrow = floor(zrow);
        z_predicted = z_predicted+dot(zrow, mask);
    }

    
    z_predicted = floor(z_predicted);
    
    // add the residual to get the actual elevation
    float zf = z_predicted + residual;  
    
    // zf should always be an integer, since it gets packed
    //  into the integer component of the floating-point texture
    zf = floor(zf);
    
    float4 uvc = floor(float4((p_uv_div2+float2(0.5f,0)), 
                              (p_uv_div2+float2(0,0.5f))))*OneOverSize+TextureOffset.xyxy; 
            
    // look up the z_predicted value in the coarser levels  
    float zc0 = floor(tex2D(CoarseLevelElevationSampler, float4(uvc.xy, 0, 1)));
    float zc1 = floor(tex2D(CoarseLevelElevationSampler, float4(uvc.zw, 0, 1)));        
    
    float zf_zd = zf + ((zc0+zc1)/2-zf+256)/512;

    return float4(zf_zd, 0, 0, 0);
}

technique Upsample
{
    pass P0
    {
        VertexShader = compile vs_3_0 UpSampleVS();
        PixelShader  = compile ps_2_a UpsamplePS();
    }
}

