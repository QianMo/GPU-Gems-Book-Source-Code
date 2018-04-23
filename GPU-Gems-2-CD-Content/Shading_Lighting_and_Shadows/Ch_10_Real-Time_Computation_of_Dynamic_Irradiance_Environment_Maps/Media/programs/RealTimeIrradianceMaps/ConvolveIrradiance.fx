#define NUM_RADIANCE_SAMPLES  32.f
#define NUM_RESULT_SAMPLES    32.f

#define NUM_ORDER    3.f
#define NUM_ORDER_P2 4.f


//  all the routines for performing the spherical harmonic convolution of a source
//  image.  this implementation is for dual-paraboloid maps.
//-----------------------------------------------------------------------------

texture SH_Convolve_dE_0;      // irradiance samples (dual-paraboloid map)
texture SH_Convolve_dE_1;    

texture SH_Convolve_Ylm_dW_0;  // spherical harmonic basis functions, scaled by solid angle
texture SH_Convolve_Ylm_dW_1;

texture SH_Coefficients;       // result of above convolution

texture SH_Integrate_Ylm_Al_zpos;   // evaluate the SH basis function on the Z+ cubeface
texture SH_Integrate_Ylm_Al_zneg;   // evaluate the SH basis function on the Z- cubeface
texture SH_Integrate_Ylm_Al_ypos;   // evaluate the SH basis function on the Y+ cubeface
texture SH_Integrate_Ylm_Al_yneg;   // evaluate the SH basis function on the Y- cubeface
texture SH_Integrate_Ylm_Al_xpos;   // evaluate the SH basis function on the X+ cubeface
texture SH_Integrate_Ylm_Al_xneg;   // evaluate the SH basis function on the X- cubeface

//-----------------------------------------------------------------------------

sampler Scene_0 = sampler_state
{
    Texture = <SH_Convolve_dE_0>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler Scene_1 = sampler_state
{
    Texture = <SH_Convolve_dE_1>;
    MinFilter = Linear;
    MagFilter = Linear;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler ProjectionWeights_0 = sampler_state
{
    Texture = <SH_Convolve_Ylm_dW_0>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler ProjectionWeights_1 = sampler_state
{
    Texture = <SH_Convolve_Ylm_dW_0>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler EvalWeights_xpos = sampler_state
{
    Texture = <SH_Integrate_Ylm_Al_xpos>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler EvalWeights_xneg = sampler_state
{
    Texture = <SH_Integrate_Ylm_Al_xneg>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler EvalWeights_ypos = sampler_state
{
    Texture = <SH_Integrate_Ylm_Al_ypos>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler EvalWeights_yneg = sampler_state
{
    Texture = <SH_Integrate_Ylm_Al_yneg>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler EvalWeights_zpos = sampler_state
{
    Texture = <SH_Integrate_Ylm_Al_zpos>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler EvalWeights_zneg = sampler_state
{
    Texture = <SH_Integrate_Ylm_Al_zneg>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

sampler RadianceCoefficients = sampler_state
{
    Texture = <SH_Coefficients>;
    MinFilter = Point;
    MagFilter = Point;
    MipFilter = None;
    AddressU  = Clamp;
    AddressV  = Clamp;
};

//----------------------------------------------------------------------------

struct PS_INPUT
{
  float2 position : VPOS;
};

float2 ComputeWeightSampleLocation( float2 texel, float2 pixelpos )
{
    // divide texel by NUM_ORDER_P2, then add in pixelpos/NUM_ORDER_P2
    // the shift by 1/2 texel is already added in, and dividing by NUM_ORDER_P2 converts it from
    // a 1/2-texel shift on a 32x32 image to a 1/2-texel shift on a 128x128 image
    float2 weightSample = (texel+pixelpos) / NUM_ORDER_P2;
    return weightSample;
}

float3 ConvolveSample( sampler weightMap, sampler colorMap, float2 coord, float2 pixelpos )
{
    //  color sample needs to be shifted by (0.5/64) to be aligned with center of every other
    //  texel in 64x64 paraboloid map.  then, shifted by an additional (0.5/64) to
    //  be in the center of every 2x2 block, for AA resolve.  total shift = 0.5/32, applied
    //  in main loop.  weight sample needs to select the appropriate subrect, based on pixelpos

    float3 color  = tex2D(colorMap,  coord).rgb;
    float  weight = tex2D(weightMap, ComputeWeightSampleLocation(coord, pixelpos)).r;
    return color * weight;
}

float4 ProjectToSH_PS(PS_INPUT IN) : COLOR
{
    float3 accum = 0.;

    //  this is a bit wasteful, since quite a few of the texels in a dual-paraboloid map are invalid.
    //  maybe someday I will unroll this loop and only include valid texels...  and, in the process,
    //  exploit some symmetry in the coordinates, to reduce the number of constants required...
    float2 coord = float2(0.5f/NUM_RADIANCE_SAMPLES, 0.5f/NUM_RADIANCE_SAMPLES);

    for (int j=0; j<NUM_RADIANCE_SAMPLES; j++, coord.y+=1.f/NUM_RADIANCE_SAMPLES )
    {
        coord.x = 0.5f/NUM_RADIANCE_SAMPLES;
        for (int i=0; i<NUM_RADIANCE_SAMPLES; i++, coord.x+=1.f/NUM_RADIANCE_SAMPLES)
        {
            accum += ConvolveSample( ProjectionWeights_0, Scene_0, coord, IN.position );
            accum += ConvolveSample( ProjectionWeights_1, Scene_1, coord, IN.position );
        }
    }
    return float4(accum,0.0);
}


float4 EvalSH_PS(PS_INPUT IN, uniform sampler coefficientWeight) : COLOR
{
    float2 coord_base = (IN.position+float2(0.5, 0.5)) / NUM_RESULT_SAMPLES;
    float3 accum = 0.;
    
    for ( int j=0; j<NUM_ORDER; j++ )
    {
        for ( int i=0; i<NUM_ORDER; i++ )
        {
            float2 coord_color  = float2(i,j)/NUM_ORDER_P2;
            float2 coord_weight = coord_base/NUM_ORDER_P2 + coord_color;
            coord_color += float2(0.5,0.5)/NUM_ORDER_P2;
            
            float weight = tex2D( coefficientWeight, coord_weight ).r;
            float3 color = tex2D( RadianceCoefficients, coord_color ).rgb;
            accum = color*weight;
        }
    }

    return float4(accum, 0.0);
}

//-----------------------------------------------------------------------------
//  Techniques
//

technique ProjectDualParaboloidToSH
{
    pass P0
    {
        ZEnable = false;
        CullMode = none;
        Sampler[0] = <ProjectionWeights_0>;
        Sampler[1] = <Scene_0>;
        Sampler[2] = <ProjectionWeights_1>;
        Sampler[3] = <Scene_1>;
        PixelShader  = asm //compile ps_3_0 ProjectToSH_PS();
        {
            ps_3_0
            def  c0, 0.03125, .0078125, 0.25, 0
            def  c1, 0.015625, 0.00390625, 0, .00048828125
            defi i0, 32, 0, 0, 0
            dcl vPos.xy
            dcl_2d s0
            dcl_2d s1
            dcl_2d s2
            dcl_2d s3
            
            mov r0, c0.w               // accumulator
            mov r1, c1.xxyy            // initial texel to fetch from dual-paraboloid maps (on 32x32 and 128x128 grid)
            rep i0
                mov r1.xz, c1.xxyy     // reset row texel every loop
                rep i0
                    mad r2.xy, vPos, c0.z, r1.zwzw
                    texld r4, r2, s0   // load first weight
                    texld r5, r2, s2   // load second weight
                    //mov r4, c1.w
                    //mov r5, c1.w
                    texld r2, r1, s1   // load first scene image
                    texld r3, r1, s3   // from second scene image
                    mad r0, r4.r, r2, r0
                    mad r0, r5.r, r3, r0
                    add r1.xz, r1, c1.xxyy
                endrep
                add r1.yw, r1, c1.xxyy
            endrep
            mov oC0, r0
        };
    }
}

technique EvaluateSHCubemap
{
    pass XPos
    {
        ZEnable = false;
        CullMode = none;
        Sampler[0] = <EvalWeights_xpos>;
        Sampler[1] = <RadianceCoefficients>;

        PixelShader  = asm //compile ps_3_0 EvalSH_PS(EvalWeights_xpos);
        {
          ps_3_0
          def c0, 0.0078125, 0.00390625, 0, 0
          def c1, 0, 0.25, 0.5, 0.75
          def c2, 0.125, 0.375, 0.625, 0.875
          def c3, 1.0, 0, 0, 0.11111111
          dcl vPos.xy
          dcl_2d s0
          dcl_2d s1
          mad r0.xy, vPos, c0.x, c0.y  // coord_base = (IN.position+0.5)/ (NUM_RESULT_SAMPLES*NUM_ORDER_P2)

          mov r1, c3.x     // HACKHACKHACK
          //  add r0.zw, r0.xxxy, c1.xxxx
          texld r1, c2.x, s1
          texld r2, r0, s0
          mul r3.rgb, r1, r2.r
          add r0.zw, r0.xxxy, c1.xxyx
          texld r1, c2.yxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzx
          texld r1, c2.zxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3


          add r0.zw, r0.xxxy, c1.xxxy
          texld r1, c2.xyyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyy
          texld r1, c2.yyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzy
          texld r1, c2.zyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3

          add r0.zw, r0.xxxy, c1.xxxz
          texld r1, c2.xzyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyz
          texld r1, c2.yzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzz
          texld r1, c2.zzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3
          max r3.rgb, r3, c0.w         // clamp to positive numbers
          
          mov r3.a, c0.w
          mul oC0, r3, c3.w
        };
    }
    pass XNeg
    {
        ZEnable = false;
        CullMode = none;
        Sampler[0] = <EvalWeights_xneg>;
        Sampler[1] = <RadianceCoefficients>;

        PixelShader  = asm //compile ps_3_0 EvalSH_PS(EvalWeights_xneg);
        {
          ps_3_0
          def c0, 0.0078125, 0.00390625, 0, 0
          def c1, 0, 0.25, 0.5, 0.75
          def c2, 0.125, 0.375, 0.625, 0.875
          def c3, 1.0, 0, 0, 0.1111111
          dcl vPos.xy
          dcl_2d s0
          dcl_2d s1
          mad r0.xy, vPos, c0.x, c0.y  // coord_base = (IN.position+0.5)/ (NUM_RESULT_SAMPLES*NUM_ORDER_P2)

          mov r1, c3.x    //  HACKHACKHACK
          //  add r0.zw, r0.xxxy, c1.xxxx
          texld r1, c2.x, s1
          texld r2, r0, s0
          mul r3.rgb, r1, r2.r
          add r0.zw, r0.xxxy, c1.xxyx
          texld r1, c2.yxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzx
          texld r1, c2.zxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3


          add r0.zw, r0.xxxy, c1.xxxy
          texld r1, c2.xyyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyy
          texld r1, c2.yyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzy
          texld r1, c2.zyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3

          add r0.zw, r0.xxxy, c1.xxxz
          texld r1, c2.xzyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyz
          texld r1, c2.yzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzz
          texld r1, c2.zzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3
          max r3.rgb, r3, c0.w         // clamp to positive numbers
          
          mov r3.a, c0.w
          mul oC0, r3, c3.w
        };
    }
    pass YPos
    {
        ZEnable = false;
        CullMode = none;
        Sampler[0] = <EvalWeights_ypos>;
        Sampler[1] = <RadianceCoefficients>;

        PixelShader  = asm //compile ps_3_0 EvalSH_PS(EvalWeights_ypos);
        {
          ps_3_0
          def c0, 0.0078125, 0.00390625, 0, 0
          def c1, 0, 0.25, 0.5, 0.75
          def c2, 0.125, 0.375, 0.625, 0.875
          def c3, 1.0, 0, 0, 0.11111111
          dcl vPos.xy
          dcl_2d s0
          dcl_2d s1
          mad r0.xy, vPos, c0.x, c0.y  // coord_base = (IN.position+0.5)/ (NUM_RESULT_SAMPLES*NUM_ORDER_P2)

          mov r1, c3.x    //  HACKHACKHACK
          //  add r0.zw, r0.xxxy, c1.xxxx
          texld r1, c2.x, s1
          texld r2, r0, s0
          mul r3.rgb, r1, r2.r
          add r0.zw, r0.xxxy, c1.xxyx
          texld r1, c2.yxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzx
          texld r1, c2.zxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3


          add r0.zw, r0.xxxy, c1.xxxy
          texld r1, c2.xyyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyy
          texld r1, c2.yyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzy
          texld r1, c2.zyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3

          add r0.zw, r0.xxxy, c1.xxxz
          texld r1, c2.xzyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyz
          texld r1, c2.yzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzz
          texld r1, c2.zzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3
          max r3.rgb, r3, c0.w         // clamp to positive numbers
          
          mov r3.a, c0.w
          mul oC0, r3, c3.w
        };
    }
    pass YNeg
    {
        ZEnable = false;
        CullMode = none;
        Sampler[0] = <EvalWeights_yneg>;
        Sampler[1] = <RadianceCoefficients>;

        PixelShader  = asm //compile ps_3_0 EvalSH_PS(EvalWeights_yneg);
        {
          ps_3_0
          def c0, 0.0078125, 0.00390625, 0, 0
          def c1, 0, 0.25, 0.5, 0.75
          def c2, 0.125, 0.375, 0.625, 0.875
          def c3, 1.0, 0, 0, 0.111111
          dcl vPos.xy
          dcl_2d s0
          dcl_2d s1
          mad r0.xy, vPos, c0.x, c0.y  // coord_base = (IN.position+0.5)/ (NUM_RESULT_SAMPLES*NUM_ORDER_P2)

          mov r1, c3.x    // HACKHACKHACK
          //  add r0.zw, r0.xxxy, c1.xxxx
          texld r1, c2.x, s1
          texld r2, r0, s0
          mul r3.rgb, r1, r2.r
          add r0.zw, r0.xxxy, c1.xxyx
          texld r1, c2.yxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzx
          texld r1, c2.zxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3


          add r0.zw, r0.xxxy, c1.xxxy
          texld r1, c2.xyyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyy
          texld r1, c2.yyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzy
          texld r1, c2.zyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3

          add r0.zw, r0.xxxy, c1.xxxz
          texld r1, c2.xzyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyz
          texld r1, c2.yzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzz
          texld r1, c2.zzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3
          max r3.rgb, r3, c0.w         // clamp to positive numbers
          
          mov r3.a, c0.w
          mul oC0, r3, c3.w
        };
    }
    pass ZPos
    {
        ZEnable = false;
        CullMode = none;    
        Sampler[0] = <EvalWeights_zpos>;
        Sampler[1] = <RadianceCoefficients>;

        PixelShader  = asm //compile ps_3_0 EvalSH_PS(EvalWeights_zpos);
        {
          ps_3_0
          def c0, 0.0078125, 0.00390625, 0, 0
          def c1, 0, 0.25, 0.5, 0.75
          def c2, 0.125, 0.375, 0.625, 0.875
          def c3, 1.0, 0, 0, 0.1111111
          dcl vPos.xy
          dcl_2d s0
          dcl_2d s1
          mad r0.xy, vPos, c0.x, c0.y  // coord_base = (IN.position+0.5)/ (NUM_RESULT_SAMPLES*NUM_ORDER_P2)

          mov r1, c3.x   // HACKHACKHACK
          //  add r0.zw, r0.xxxy, c1.xxxx
          texld r1, c2.x, s1
          texld r2, r0, s0
          mul r3.rgb, r1, r2.r
          add r0.zw, r0.xxxy, c1.xxyx
          texld r1, c2.yxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzx
          texld r1, c2.zxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3


          add r0.zw, r0.xxxy, c1.xxxy
          texld r1, c2.xyyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyy
          texld r1, c2.yyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzy
          texld r1, c2.zyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3

          add r0.zw, r0.xxxy, c1.xxxz
          texld r1, c2.xzyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyz
          texld r1, c2.yzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzz
          texld r1, c2.zzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3
          max r3.rgb, r3, c0.w         // clamp to positive numbers
          
          mov r3.a, c0.w
          mul oC0, r3, c3.w
        };
    }
    pass ZNeg
    {
        ZEnable = false;
        CullMode = none;    
        Sampler[0] = <EvalWeights_zneg>;
        Sampler[1] = <RadianceCoefficients>;

        PixelShader  = asm //compile ps_3_0 EvalSH_PS(EvalWeights_zneg);
        {
          ps_3_0
          def c0, 0.0078125, 0.00390625, 0, 0
          def c1, 0, 0.25, 0.5, 0.75
          def c2, 0.125, 0.375, 0.625, 0.875
          def c3, 1.0, 0, 0, 0.111111
          dcl vPos.xy
          dcl_2d s0
          dcl_2d s1
          mad r0.xy, vPos, c0.x, c0.y  // coord_base = (IN.position+0.5)/ (NUM_RESULT_SAMPLES*NUM_ORDER_P2)

          mov r1, c3.x  // HACKHACKHACK
          //  add r0.zw, r0.xxxy, c1.xxxx
          texld r1, c2.x, s1
          texld r2, r0, s0
          mul r3.rgb, r1, r2.r
          add r0.zw, r0.xxxy, c1.xxyx
          texld r1, c2.yxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzx
          texld r1, c2.zxxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3


          add r0.zw, r0.xxxy, c1.xxxy
          texld r1, c2.xyyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyy
          texld r1, c2.yyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzy
          texld r1, c2.zyxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3

          add r0.zw, r0.xxxy, c1.xxxz
          texld r1, c2.xzyy, s1
          texld r2, r0, s0
          mad r3.rgb, r1, r2.r, r3
          add r0.zw, r0.xxxy, c1.xxyz
          texld r1, c2.yzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3        
          add r0.zw, r0.xxxy, c1.xxzz
          texld r1, c2.zzxx, s1
          texld r2, r0.zwww, s0
          mad r3.rgb, r1, r2.r, r3
          max r3.rgb, r3, c0.w         // clamp to positive numbers
          
          mov r3.a, c0.w
          mul oC0, r3, c3.w
        };
    }
}


//-----------------------------------------------------------------------------

