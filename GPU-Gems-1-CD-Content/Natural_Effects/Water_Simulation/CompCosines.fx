
texture tCosineLUT;
texture tBiasNoise;

float4 cUTrans0;
float4 cUTrans1;
float4 cUTrans2;
float4 cUTrans3;
float4 cUTrans4;
float4 cUTrans5;
float4 cUTrans6;
float4 cUTrans7;
float4 cUTrans8;
float4 cUTrans9;
float4 cUTrans10;
float4 cUTrans11;
float4 cUTrans12;
float4 cUTrans13;
float4 cUTrans14;
float4 cUTrans15;

float4 cCoef0;
float4 cCoef1;
float4 cCoef2;
float4 cCoef3;
float4 cCoef4;
float4 cCoef5;
float4 cCoef6;
float4 cCoef7;
float4 cCoef8;
float4 cCoef9;
float4 cCoef10;
float4 cCoef11;
float4 cCoef12;
float4 cCoef13;
float4 cCoef14;
float4 cCoef15;

float4 cReScale;

float4 cNoiseXForm0_00;
float4 cNoiseXForm0_10;
float4 cNoiseXForm1_00;
float4 cNoiseXForm1_10;
float4 cScaleBias;

sampler CosineLUT = sampler_state
{
Texture   = <tCosineLUT>;
MinFilter = LINEAR;
MagFilter = LINEAR;
MipFilter = LINEAR;
};

sampler BiasNoise = sampler_state
{
Texture   = <tBiasNoise>;
MinFilter = LINEAR;
MagFilter = LINEAR;
MipFilter = LINEAR;
};

struct VertIn
{
	float4 Position : POSITION;
	float4 Uv    : TEXCOORD0;
};

struct VertOut
{
    float4 Position  : POSITION;
    float4 Uv0		 : TEXCOORD0; // Ripple texture coords
    float4 Uv1		 : TEXCOORD1; // Ripple texture coords
    float4 Uv2		 : TEXCOORD2; // Ripple texture coords
    float4 Uv3		 : TEXCOORD3; // Ripple texture coords
};

VertOut vs_main(VertIn In,
			uniform float4 rot0,
			uniform float4 rot1,
			uniform float4 rot2,
			uniform float4 rot3)
{
	VertOut Out;
	Out.Position = In.Position;

	float4 uv = float4(0.f, 0.f, 0.f, 1.f);

	uv.x = dot(In.Uv, rot0);
	Out.Uv0 = uv;

	uv.x = dot(In.Uv, rot1);
	Out.Uv1 = uv;

	uv.x = dot(In.Uv, rot2);
	Out.Uv2 = uv;

	uv.x = dot(In.Uv, rot3);
	Out.Uv3 = uv;

	return Out;
}

technique T0
{

    pass P0
    {
        // Vertex shader
        VertexShader = compile vs_1_1 vs_main(cUTrans0, cUTrans1, cUTrans2, cUTrans3);

        // Pixel shader
        PixelShader =
            asm
            {
				// Composite the cosines together.
				// Input map is cosine(pix) for each of
				// the 4 waves.
				//
				// The constants are set up so:
				//		Nx = -freq * amp * dirX * cos(pix);
				//		Ny = -freq * amp * dirY * cos(pix);
				//	So c[i].x = -freq[i] * amp[i] * dirX[i]
				//	etc.
				// All textures are:
				//		(r,g,b,a) = (cos(), cos(), 1, 1)
				//

				ps_1_1
			 
				tex		t0;
				tex		t1;
				tex		t2;
				tex		t3;

				mul		r0, t0_bx2, c0;
				mad		r0, t1_bx2, c1, r0;
				mad		r0, t2_bx2, c2, r0;
				mad		r0, t3_bx2, c3, r0;
				// Now scale and bias it back into range for output.
				mul		r0, r0, c4;		
				add		r0, r0, c4;
            };

        PixelShaderConstant1[0] = <cCoef0>;
        PixelShaderConstant1[1] = <cCoef1>;
        PixelShaderConstant1[2] = <cCoef2>;
        PixelShaderConstant1[3] = <cCoef3>;
        PixelShaderConstant1[4] = <cReScale>;

        Sampler[0] = (CosineLUT);
        Sampler[1] = (CosineLUT);
        Sampler[2] = (CosineLUT);
        Sampler[3] = (CosineLUT);

		// Clip/Raster state
        SrcBlend  = One;
        DestBlend = Zero;

		SpecularEnable = FALSE;
		AlphaBlendEnable = TRUE;
		FogEnable = FALSE;
		CullMode = NONE;
		ZEnable = FALSE;
    }

    pass P1
    {
        // Vertex shader
        VertexShader = compile vs_1_1 vs_main(cUTrans4, cUTrans5, cUTrans6, cUTrans7);

        // Pixel shader
        PixelShader =
            asm
            {
				// Composite the cosines together.
				// Input map is cosine(pix) for each of
				// the 4 waves.
				//
				// The constants are set up so:
				//		Nx = -freq * amp * dirX * cos(pix);
				//		Ny = -freq * amp * dirY * cos(pix);
				//	So c[i].x = -freq[i] * amp[i] * dirX[i]
				//	etc.
				// All textures are:
				//		(r,g,b,a) = (cos(), cos(), 1, 1)
				//
				// So c[0].z = 1, but all other c[i].z = 0
				// Note also the c4 used for biasing back at the end.

				ps_1_1

				tex		t0;
				tex		t1;
				tex		t2;
				tex		t3;

				mul		r0, t0_bx2, c0;
				mad		r0, t1_bx2, c1, r0;
				mad		r0, t2_bx2, c2, r0;
				mad		r0, t3_bx2, c3, r0;
				// Now bias it back into range [0..1] for output.
				mul		r0, r0, c4;		// c4 = (0.5, 0.5, 0.5, 1)
				add		r0, r0, c4;
            };

        PixelShaderConstant1[0] = <cCoef4>;
        PixelShaderConstant1[1] = <cCoef5>;
        PixelShaderConstant1[2] = <cCoef6>;
        PixelShaderConstant1[3] = <cCoef7>;
        PixelShaderConstant1[4] = <cReScale>;

        Sampler[0] = (CosineLUT);
        Sampler[1] = (CosineLUT);
        Sampler[2] = (CosineLUT);
        Sampler[3] = (CosineLUT);

		// Clip/Raster state
        SrcBlend  = One;
        DestBlend = One;

		SpecularEnable = FALSE;
		AlphaBlendEnable = TRUE;
		FogEnable = FALSE;
		CullMode = NONE;
		ZEnable = FALSE;
    }

    pass P2
    {
        // Vertex shader
        VertexShader = compile vs_1_1 vs_main(cUTrans8, cUTrans9, cUTrans10, cUTrans11);

        // Pixel shader
        PixelShader =
            asm
            {
				// Composite the cosines together.
				// Input map is cosine(pix) for each of
				// the 4 waves.
				//
				// The constants are set up so:
				//		Nx = -freq * amp * dirX * cos(pix);
				//		Ny = -freq * amp * dirY * cos(pix);
				//	So c[i].x = -freq[i] * amp[i] * dirX[i]
				//	etc.
				// All textures are:
				//		(r,g,b,a) = (cos(), cos(), 1, 1)
				//
				// So c[0].z = 1, but all other c[i].z = 0
				// Note also the c4 used for biasing back at the end.

				ps_1_1

				tex		t0;
				tex		t1;
				tex		t2;
				tex		t3;

				mul		r0, t0_bx2, c0;
				mad		r0, t1_bx2, c1, r0;
				mad		r0, t2_bx2, c2, r0;
				mad		r0, t3_bx2, c3, r0;
				// Now bias it back into range [0..1] for output.
				mul		r0, r0, c4;		// c4 = (0.5, 0.5, 0.5, 1)
				add		r0, r0, c4;
            };

        PixelShaderConstant1[0] = <cCoef8>;
        PixelShaderConstant1[1] = <cCoef9>;
        PixelShaderConstant1[2] = <cCoef10>;
        PixelShaderConstant1[3] = <cCoef11>;
        PixelShaderConstant1[4] = <cReScale>;

        Sampler[0] = (CosineLUT);
        Sampler[1] = (CosineLUT);
        Sampler[2] = (CosineLUT);
        Sampler[3] = (CosineLUT);

		// Clip/Raster state
        SrcBlend  = One;
        DestBlend = One;

		SpecularEnable = FALSE;
		AlphaBlendEnable = TRUE;
		FogEnable = FALSE;
		CullMode = NONE;
		ZEnable = FALSE;
    }
    pass P3
    {
        // Vertex shader
        VertexShader = compile vs_1_1 vs_main(cUTrans12, cUTrans13, cUTrans14, cUTrans15);

        // Pixel shader
        PixelShader =
            asm
            {
				// Composite the cosines together.
				// Input map is cosine(pix) for each of
				// the 4 waves.
				//
				// The constants are set up so:
				//		Nx = -freq * amp * dirX * cos(pix);
				//		Ny = -freq * amp * dirY * cos(pix);
				//	So c[i].x = -freq[i] * amp[i] * dirX[i]
				//	etc.
				// All textures are:
				//		(r,g,b,a) = (cos(), cos(), 1, 1)
				//
				// So c[0].z = 1, but all other c[i].z = 0
				// Note also the c4 used for biasing back at the end.

				ps_1_1

				tex		t0;
				tex		t1;
				tex		t2;
				tex		t3;

				mul		r0, t0_bx2, c0;
				mad		r0, t1_bx2, c1, r0;
				mad		r0, t2_bx2, c2, r0;
				mad		r0, t3_bx2, c3, r0;
				// Now bias it back into range [0..1] for output.
				mul		r0, r0, c4;		// c4 = (0.5, 0.5, 0.5, 1)
				add		r0, r0, c4;
            };

        PixelShaderConstant1[0] = <cCoef12>;
        PixelShaderConstant1[1] = <cCoef13>;
        PixelShaderConstant1[2] = <cCoef14>;
        PixelShaderConstant1[3] = <cCoef15>;
        PixelShaderConstant1[4] = <cReScale>;

        Sampler[0] = (CosineLUT);
        Sampler[1] = (CosineLUT);
        Sampler[2] = (CosineLUT);
        Sampler[3] = (CosineLUT);

		// Clip/Raster state
        SrcBlend  = One;
        DestBlend = One;

		SpecularEnable = FALSE;
		AlphaBlendEnable = TRUE;
		FogEnable = FALSE;
		CullMode = NONE;
		ZEnable = FALSE;
    }

	pass p4
	{
		Sampler[0] = (BiasNoise);
		Sampler[1] = (BiasNoise);

		VertexShader =
			asm
			{
			   vs_1_1
			
			   dcl_position  v0		
			   dcl_texcoord  v7		


				// Take in a screen space position,
				// transform the UVW,
				// c0 = uvXform0[0]
				// c1 = uvXform0[1]
				// c2 = uvXform1[0]
				// c3 = uvXform1[1]
				// c4 = (0,0.5,1.0,2.0)
				// c5 = (noiseScale, bias, 0, 1)

				mov oPos, v0;

				mov r0.zw, c4.xxxz; // yzw will stay constant (0,0,1);

				dp4 r0.x, v7, c0;
				dp4 r0.y, v7, c1;

				mov oT0, r0;

				dp4 r0.x, v7, c2;
				dp4 r0.y, v7, c3;

				mov oT1, r0;

				mov oD0, c5.xxzz;
				mov oD1, c5.yyzz;
			};

		VertexShaderConstant[0] = <cNoiseXForm0_00>;
		VertexShaderConstant[1] = <cNoiseXForm0_10>;
		VertexShaderConstant[2] = <cNoiseXForm1_00>;
		VertexShaderConstant[3] = <cNoiseXForm1_10>;
		VertexShaderConstant[4] = float4(0.f, 0.5f, 1.f, 2.f);
		VertexShaderConstant[5] = <cScaleBias>;

		PixelShader =
			asm
			{
				ps_1_1

				// Grab noise texture, 
				// modulate biased version by vtx color 0,
				// add to vtx color 1

				tex		t0;
				tex		t1;

				add		r0.rgb, t0_bias, t1_bias;
				+add	r0.a, t0, t1;
				mad		r0.rgb, r0, v0, v1;
			};


		// Clip/Raster state
        SrcBlend  = One;
        DestBlend = One;

		SpecularEnable = FALSE;
		AlphaBlendEnable = TRUE;
		FogEnable = FALSE;
		CullMode = NONE;
		ZEnable = FALSE;
	}
}
