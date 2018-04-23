texture CubeTexture;

sampler CubeSampler = sampler_state
{
    Texture = <CubeTexture>;
    AddressU = Clamp;
    AddressV = Clamp;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = Linear;
};

float4 ReadTextureCube_PS( float3 texcoord : TEXCOORD0 ) : COLOR
{
    return texCUBE(CubeSampler, texcoord);
}

technique SimpleCubeMapRender {
  pass p0 {
    ZEnable = FALSE;
    ZWriteEnable = FALSE;
    ColorWriteEnable = Red | Blue | Green | Alpha;
    CullMode = CCW;
    PixelShader = compile ps_2_0 ReadTextureCube_PS();
  }
}

float4 ReadTexture2D_PS( float2 texcoord : TEXCOORD0 ) : COLOR
{
    return tex2D(CubeSampler, texcoord);
}

technique Simple2DRender {
  pass p0 {
    ZEnable = FALSE;
    ZWriteEnable = FALSE;
    ColorWriteEnable = Red | Blue | Green | Alpha;
    CullMode = CCW;
    PixelShader = compile ps_2_0 ReadTexture2D_PS();
  }
}