
textureCUBE tEnvMap;
texture tBumpMap;


float4x4 cWorld2NDC;
float4 cWaterTint;
float4 cFrequency;
float4 cPhase;
float4 cAmplitude;
float4 cDirX;
float4 cDirY;
float4 cSpecAtten; // uvScale is w component
float4 cCameraPos; // world space
float4 cEnvAdjust;
float4 cEnvTint;
float4x4 cLocal2World;
float4 cLengths;
float4 cDepthOffset; // water level is w component
float4 cDepthScale;
float4 cFogParams;
float4 cDirXK;
float4 cDirYK;
float4 cDirXW;
float4 cDirYW;
float4 cKW;
float4 cDirXSqKW;
float4 cDirXDirYKW;
float4 cDirYSqKW;

sampler BumpSamp = sampler_state
{
Texture   = <tBumpMap>;
MinFilter = LINEAR;
MagFilter = LINEAR;
MipFilter = LINEAR;
};

sampler EnvSamp = sampler_state
{
Texture   = <tEnvMap>;
MinFilter = LINEAR;
MagFilter = LINEAR;
MipFilter = LINEAR;
};

#pragma PACK_MATRIX (ROW_MAJOR)

struct VertIn
{
	float4 Position : POSITION;
	float4 Color    : COLOR0;
};

struct VertOut
{
    float4 Position  : POSITION;
    float4 modColor  : COLOR0;
    float4 addColor  : COLOR1;
	float  Fog       : FOG;
    float4 TexCoord0 : TEXCOORD0; // Ripple texture coords
    float4 BTN_X     : TEXCOORD1; // Binormal.x, Tangent.x, Normal.x
    float4 BTN_Y     : TEXCOORD2; // Bin.y, Tan.y, Norm.y
    float4 BTN_Z     : TEXCOORD3; // Bin.z, Tan.z, Norm.z
};

void CalcSinCos(const in float4 wPos,
				const in float4 dirX,
				const in float4 dirY,
				const in float4 amplitude,
				const in float4 frequency,
				const in float4 phase,
				const in float4 lengths,
				const in float ooEdgeLength,
				const in float scale,
				out float4 sines, out float4 cosines)
{
	// Dot x and y with direction vectors
	float4 dists = dirX * wPos.xxxx;
	dists = dirY * wPos.yyyy + dists;
	
	// Scale in our frequency and add in our phase
	dists = dists * frequency;
	dists = dists + phase;

	const float kPi = 3.14159265f;
	const float kTwoPi = 2.f * kPi;
	const float kOOTwoPi = 1.f / kTwoPi;
	// Mod into range [-Pi..Pi]
	dists = dists + kPi;
	dists = dists * kOOTwoPi;
	dists = frac(dists);
	dists = dists * kTwoPi;
	dists = dists - kPi;

	float4 dists2 = dists * dists; 
	float4 dists3 = dists2 * dists;
	float4 dists4 = dists2 * dists2;
	float4 dists5 = dists3 * dists2;
	float4 dists6 = dists3 * dists3;
	float4 dists7 = dists4 * dists3;

	const float4 kSinConsts = float4(1.f, -1.f/6.f, 1.f/120.f, -1.f/5040.f);
	const float4 kCosConsts = float4(1.f, -1.f/2.f, 1.f/24.f, -1.f/720.f);
	sines = dists + dists3 * kSinConsts.yyyy + dists5 * kSinConsts.zzzz + dists7 * kSinConsts.wwww;
	cosines = kCosConsts.xxxx + dists2 * kCosConsts.yyyy + dists4 * kCosConsts.zzzz + dists6 * kCosConsts.wwww;

	float4 filteredAmp = lengths * ooEdgeLength;
	filteredAmp = max(filteredAmp, 0.f);
	filteredAmp = min(filteredAmp, 1.f);
	filteredAmp = filteredAmp * scale; 
	filteredAmp = filteredAmp * amplitude;

	sines = sines * filteredAmp;
	cosines = cosines * filteredAmp * scale;
}

float3 FinitizeEyeRay(const in float3 cam2Vtx, const in float4 envAdjust)
{
	// Compute our finitized eyeray.

	// Our "finitized" eyeray is:
	//	camPos + D * t - envCenter = D * t - (envCenter - camPos)
	// with
	//	D = (pos - camPos) / |pos - camPos| // normalized usual eyeray
	// and
	//	t = D dot F + sqrt( (D dot F)^2 - G )
	// with
	//	F = (envCenter - camPos)	=> envAdjust.xyz
	//	G = F^2 - R^2				=> nevAdjust.w
	// where R is the sphere radius.
	//
	// This all derives from the positive root of equation
	//	(camPos + (pos - camPos) * t - envCenter)^2 = R^2,
	// In other words, where on a sphere of radius R centered about envCenter
	// does the ray from the real camera position through this point hit.
	//
	// Note that F and G are both constants (one 3-point, one scalar).
	float dDotF = dot(cam2Vtx, float3(envAdjust.xyz));
	float t = dDotF + sqrt(dDotF * dDotF - envAdjust.w);
	return cam2Vtx * t - float3(envAdjust.xyz);

}

void CalcScreenPosAndFog(const in float4x4 world2NDC, const in float4 fogParams, const in float4 wPos, out float4 scrPos, out float fog)
{
	// Calc screen position and fog from screen W
	// Fog is basic linear from start distance to end distance.
	float4 sPos = mul(world2NDC, wPos);
	fog = (sPos.w + fogParams.x) * fogParams.y;
	scrPos = sPos;
}

void CalcFinalColors(const in float3 norm, 
					 const in float3 cam2Vtx, 
					 const in float opacMin, 
					 const in float opacScale,
					 const in float colorFilter,
					 const in float opacFilter,
					 const in float4 tint,
					 out float4 modColor,
					 out float4 addColor)
{
	// Calculate colors
	// Final color will be
	// rgb = Color1.rgb + Color0.rgb * envMap.rgb
	// alpha = Color0.a

	// Color 0

	// Vertex based Fresnel-esque effect.
	// Input vertex color.b limits how much we attenuate based on angle.
	// So we map 
	// (dot(norm,cam2Vtx)==0) => 1 for grazing angle
	// and (dot(norm,cam2Vtx)==1 => 1-In.Color.b for perpendicular view.
	float atten = 1.0 + dot(norm, cam2Vtx) * opacMin;

	// Filter the color based on depth
	modColor.rgb = colorFilter * atten;

	// Boost the alpha so the reflections fade out faster than the tint
	// and apply the input attenuation factors.
	modColor.a = (atten + 1.0) * 0.5 * opacFilter * opacScale * tint.a;

	// Color 1 is just a constant.
	addColor = tint;

}

void CalcEyeRayAndBumpAttenuation(const in float4 wPos, 
								  const in float4 cameraPos, 
								  const in float4 specAtten, 
								  out float3 cam2Vtx, 
								  out float pertAtten)
{
	// Get normalized vec from camera to vertex, saving original distance.
	cam2Vtx = float3(wPos.xyz) - float3(cameraPos.xyz);
	pertAtten = length(cam2Vtx);
	cam2Vtx = cam2Vtx / pertAtten;

	// Calculate our normal perturbation attenuation. This attenuation will be
	// applied to the horizontal components of the normal read from the computed
	// ripple bump map, mostly to fight aliasing. This doesn't attenuate the 
	// color computed from the normal map, it attenuates the "bumps".
	pertAtten = pertAtten + specAtten.x;
	pertAtten = pertAtten * specAtten.y;
	pertAtten = min(pertAtten, 1.f);
	pertAtten = max(pertAtten, 0.f);
	pertAtten = pertAtten * pertAtten; // Square it to account for perspective.
	pertAtten = pertAtten * specAtten.z;
}

// Depth filter channels control:
// dFilter.x => overall opacity
// dFilter.y => reflection strength
// dFilter.z => wave height
float3 CalcDepthFilter(const in float4 depthOffset, const in float4 depthScale, const in float4 wPos, const in float waterLevel)
{
	float3 dFilter = float3(depthOffset.xyz) - wPos.zzz;

	dFilter = dFilter * float3(depthScale.xyz);
	dFilter = max(dFilter, 0.f);
	dFilter = min(dFilter, 1.f);

	return dFilter;
}

// See Pos in CalcTangentBasis comments.
float4 CalcFinalPosition(in float4 wPos, 
						 const in float4 sines, 
						 const in float4 cosines, 
						 const in float depthOffset, 
						 const in float4 dirXK, 
						 const in float4 dirYK)
{
	// Height

	// Sum to a scalar
	float h = dot(sines, 1.f) + depthOffset;

	// Clamp to never go beneath input height
	wPos.z = max(wPos.z, h);


	wPos.x = wPos.x + dot(cosines, dirXK);
	wPos.y = wPos.y + dot(cosines, dirYK);

	return wPos;
}

// Normal, binormal, tangent

// Okay, here we go:
// W == sum(k w Dir.x^2 A sin())
// V == sum(k w Dir.x Dir.y A sin())
// U == sum(k w Dir.y^2 A sin())
//
// T == sum(A sin())
//
// S == sum(k Dir.x A cos())
// R == sum(k Dir.y A cos())
//
// Q == sum(k w A cos())
//
// M == sum(A cos())
//
// P == sum(w Dir.x A cos())
// N == sum(w Dir.y A cos())
//
// Then:
// Pos = (in.x + S, in.y + R, waterheight + T)
// 
// Bin = (1 - W, -V, P)
// Tan = (-V, 1 - U, N)
// Nor = (-P, -N, 1 - Q)
//
// Remember we want the transpose of Binormal, Tangent, and Normal
void CalcTangentBasis(const in float4 sines, 
					  const in float4 cosines, 
					  const in float4 dirXSqKW,
					  const in float4 dirXDirYKW,
					  const in float4 dirYSqKW,
					  const in float4 dirXW,
					  const in float4 dirYW,
					  const in float4 KW,
					  const in float pertAtten,
					  const in float3 eyeRay,
					  out float4 BTN_X,
					  out float4 BTN_Y,
					  out float4 BTN_Z,
					  out float3 norm)
{
// Note that we're swapping Y and Z and negating Z (rotation about X)
// to match the D3D convention of Y being up in cubemaps.

	BTN_X.x = 1.f + dot(sines, -dirXSqKW);
	BTN_X.y = dot(sines, -dirXDirYKW);
	BTN_X.z = dot(cosines, -dirXW);
	BTN_X.xy = BTN_X.xy * pertAtten;
	norm.x = BTN_X.z;

	BTN_Z.x = dot(sines, -dirXDirYKW);
	BTN_Z.y = 1.f + dot(sines, -dirYSqKW);
	BTN_Z.z = dot(cosines, -dirYW);
	BTN_Z.xy = BTN_Z.xy * pertAtten;
	norm.y = BTN_Z.z;

	BTN_Y.x = -dot(cosines, dirXW);
	BTN_Y.y = -dot(cosines, dirYW);
	BTN_Y.z = -(1.f + dot(sines, -KW));
	BTN_Y.xy = BTN_Y.xy * pertAtten;
	norm.z = -BTN_Y.z;


	BTN_X.w = eyeRay.x;
	BTN_Y.w = -eyeRay.z;
	BTN_Z.w = eyeRay.y;
}


VertOut vs_main(VertIn In,
				uniform float4x4 kWorld2NDC,
				uniform float4 kWaterTint,
				uniform float4 kFrequency,
				uniform float4 kPhase,
				uniform float4 kAmplitude,
				uniform float4 kDirX,
				uniform float4 kDirY,
				uniform float4 kSpecAtten, // uvScale is w component
				uniform float4 kCameraPos, // world space
				uniform float4 kEnvAdjust,
				uniform float4 kEnvTint,
				uniform float4x4 kLocal2World,
				uniform float4 kLengths,
				uniform float4 kDepthOffset, // water level is w component
				uniform float4 kDepthScale,
				uniform float4 kFogParams,
				uniform float4 kDirXK,
				uniform float4 kDirYK,
				uniform float4 kDirXW,
				uniform float4 kDirYW,
				uniform float4 kKW,
				uniform float4 kDirXSqKW,
				uniform float4 kDirXDirYKW,
				uniform float4 kDirYSqKW
				)
{
	VertOut vOut;

	// Evaluate world space base position. All subsequent calculations in world space.
	float4 wPos = mul(kLocal2World, In.Position);

	// Calculate ripple UV from position
	vOut.TexCoord0.xy = wPos.xy * kSpecAtten.ww;
	vOut.TexCoord0.z = 0.f;
	vOut.TexCoord0.w = 1.f;

	// Get our depth based filters. 
	float3 dFilter = CalcDepthFilter(kDepthOffset, kDepthScale, wPos, kDepthOffset.w);

	// Build our 4 waves

	float4 sines;
	float4 cosines;
	CalcSinCos(wPos, 
		kDirX, kDirY, 
		kAmplitude, kFrequency, kPhase, 
		kLengths, In.Color.a, dFilter.z, 
		sines, cosines);

	wPos = CalcFinalPosition(wPos, sines, cosines, kDepthOffset.w, kDirXK, kDirYK);

	// We have our final position. We'll be needing normalized vector from camera 
	// to vertex several times, so we go ahead and grab it.
	float3 cam2Vtx;
	float pertAtten;
	CalcEyeRayAndBumpAttenuation(wPos, kCameraPos, kSpecAtten, cam2Vtx, pertAtten);

	// Compute our finitized eyeray.
	float3 eyeRay = FinitizeEyeRay(cam2Vtx, kEnvAdjust);

	float3 norm;
	CalcTangentBasis(sines, cosines, 
		kDirXSqKW,
		kDirXDirYKW,
		kDirYSqKW,
		kDirXW,
		kDirYW,
		kKW,
		pertAtten,
		eyeRay,
		vOut.BTN_X,
		vOut.BTN_Y,
		vOut.BTN_Z,
		norm);

	// Calc screen position and fog
	CalcScreenPosAndFog(kWorld2NDC, kFogParams, wPos, vOut.Position, vOut.Fog.x);

	CalcFinalColors(norm, 
					cam2Vtx, 
					In.Color.b, 
					In.Color.r,
					dFilter.y,
					dFilter.x,
					kWaterTint,
					vOut.modColor,
					vOut.addColor);

	return vOut;
}


technique T0
{
	pass P0
	{
		VertexShader = compile vs_1_1 vs_main(cWorld2NDC,
											cWaterTint,
											cFrequency,
											cPhase,
											cAmplitude,
											cDirX,
											cDirY,
											cSpecAtten,
											cCameraPos,
											cEnvAdjust,
											cEnvTint,
											cLocal2World,
											cLengths,
											cDepthOffset,
											cDepthScale,
											cFogParams,
											cDirXK,
											cDirYK,
											cDirXW,
											cDirYW,
											cKW,
											cDirXSqKW,
											cDirXDirYKW,
											cDirYSqKW);

		PixelShader =
			asm
			{
				ps_1_1

				tex t0 
				texm3x3pad   t1,  t0_bx2   
				texm3x3pad   t2,  t0_bx2   
				texm3x3vspec t3,  t0_bx2  

				mad			r0.rgb, t3, v0, v1;
				+mov		r0.a, v0; 
			};

		Sampler[0] = (BumpSamp);
		Sampler[1] = (BumpSamp);
		Sampler[2] = (BumpSamp);
		Sampler[3] = (EnvSamp);

		CullMode = NONE;

        SrcBlend  = SrcAlpha;
        DestBlend = InvSrcAlpha;
	}
}