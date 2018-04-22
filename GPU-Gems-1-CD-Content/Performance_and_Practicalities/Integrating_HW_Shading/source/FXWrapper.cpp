// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes for shadows, shaders, and parameters

#include "CgFX/ICgFXEffect.h"
#include <glh/glh_extensions.h>
#include <GL/glu.h>
#include "nv_dds.h"
#include "FXWrapper.h"
#include "C4DWrapper.h"
#include "LightsMaterialsObjects.h"
#include "MatrixMath.h"
#include "Mfxmaterial.h"
#include "../../../Resource/res/description/Mmaterial.h"
#include "../../../Resource/res/description/Olight.h"
#include <string>
#include <sstream>
#include <fstream>
#include <string.h>
using namespace std;
#include <assert.h>
#include <math.h>

ShadowMaps::ShadowMaps(HDC hDC, Lights* lig, int mapSize)
: dc(hDC), lights(lig),  size(mapSize), depthBuf(0), depthHDC(0), depthHRC(0),
valid(NULL), hasActiveShadow(0), textureNumbers(0)
{
	number = lights->GetNumShadowLights();
	textureNumbers = new unsigned int[number];
	if(textureNumbers == 0)
		return;
	glGenTextures(number, textureNumbers);

	depthBuf = new HPBUFFERARB[number];
	depthHDC = new HDC[number];
	depthHRC = new HGLRC[number];
	hasActiveShadow = new bool[number];
	valid = new bool[number];
	if(depthBuf == 0 || depthHDC == 0 || depthHRC == 0 || hasActiveShadow == 0 || valid == 0)
		return;
	memset(valid, 0, number*sizeof(bool)); // so that we know which ones have been built
	memset(hasActiveShadow, 0, number*sizeof(bool)); // not valid => no shadow, also no shadow for point lights etc.

	int s;
	for(s=0; s<number; ++s)
	{
		BOOL status;
		int pixelFormat;
		unsigned int numFormats;
		int iAttributes[30];
		float fAttributes[] = {0, 0};
		iAttributes[0] = WGL_DRAW_TO_PBUFFER_ARB;
		iAttributes[1] = GL_TRUE;
		iAttributes[2] = WGL_ACCELERATION_ARB;
		iAttributes[3] = WGL_FULL_ACCELERATION_ARB;
		iAttributes[4] = WGL_COLOR_BITS_ARB;
		iAttributes[5] = 24;
		iAttributes[6] = WGL_ALPHA_BITS_ARB;
		iAttributes[7] = 8;
		iAttributes[8] = WGL_DEPTH_BITS_ARB;
		iAttributes[9] = 24;
		iAttributes[10] = WGL_BIND_TO_TEXTURE_DEPTH_NV;
		iAttributes[11] = GL_TRUE;
		iAttributes[12] = 0;

		status = wglChoosePixelFormatARB(hDC, iAttributes,
						fAttributes, 1, &pixelFormat, &numFormats);
		if (status != GL_TRUE || numFormats == 0)
		{
			C4DWrapper::Print("Shadows: No suitable OpenGL format found.");
			return;
		}

		const int pDepthBuf[] =
		{
			WGL_DEPTH_TEXTURE_FORMAT_NV, WGL_TEXTURE_DEPTH_COMPONENT_NV,
			WGL_TEXTURE_TARGET_ARB, WGL_TEXTURE_2D_ARB,
			0
		};

		depthBuf[s] = wglCreatePbufferARB(hDC, pixelFormat, size, size, pDepthBuf);
		if(depthBuf[s] == 0)
		{
			C4DWrapper::MsgBox("Shadows: Couldn't create depth buffer.");
			return;
		}

		depthHDC[s] = wglGetPbufferDCARB(depthBuf[s]);
		if(depthHDC[s] == 0)
		{
			C4DWrapper::MsgBox("Shadows: Couldn't create device context.");
			return;
		}
		depthHRC[s] = wglCreateContext(depthHDC[s]);
		if(depthHRC[s] == 0)
		{
			C4DWrapper::MsgBox("Shadows: Coudn't create render context.");
			return;
		}

		valid[s] = true;
	}
}

ShadowMaps::~ShadowMaps(void)
{
	int s;
	for(s=0; s<number; ++s)
	{
		if(valid != 0 && valid[s])
		{
			if(depthHDC[s] != 0)
			{
				wglMakeCurrent(depthHDC[s], NULL);
				if(depthHRC[s] != 0)
				{
					wglDeleteContext(depthHRC[s]);
					wglReleasePbufferDCARB(depthBuf[s], depthHDC[s]);
				}
				wglDestroyPbufferARB(depthBuf[s]);
			}
		}
	}
	delete[] hasActiveShadow;
	hasActiveShadow = 0;
	delete[] valid;
	valid = 0;
	delete[] depthHRC;
	depthHRC = 0;
	delete[] depthHDC;
	depthHDC = 0;
	delete[] depthBuf;
	depthBuf = 0;

	glDeleteTextures(number, textureNumbers);
	delete[] textureNumbers;
	textureNumbers = 0;
}

void ShadowMaps::Begin(int which)
{
	current = which;
	assert(valid[current] && hasActiveShadow[current]);

	wglMakeCurrent(depthHDC[current], depthHRC[current]);
	glEnable(GL_DEPTH_TEST);
	glViewport(0, 0, size, size);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glScaled(-1.0, 1.0, 1.0); // C4D has a left-handed coordinate system

	float angle = 0.0f;
	C4DWrapper::GetObjectFloat(lights->GetLight(lights->GetLightFromShadow(which)), LIGHT_DETAILS_OUTERANGLE, &angle);
	angle *= (float)(180.0/3.14159265);

	gluPerspective(angle, 1.0, 10.0, 1000.0); // near and far should be adapted to the scene
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	const float* pos = lights->GetPos(lights->GetLightFromShadow(which));
	const float* dir = lights->GetDir(lights->GetLightFromShadow(which));
	const float* up = lights->GetUp(lights->GetLightFromShadow(which));

	gluLookAt(pos[0], pos[1], pos[2], pos[0]+dir[0], pos[1]+dir[1], pos[2]+dir[2], up[0], up[1], up[2]);
}

void ShadowMaps::End(void)
{
	assert(valid[current] && hasActiveShadow[current]);
	glFinish();
}

int ShadowMaps::Bind(int which)
{
	if(valid == 0 || !valid[which] || !hasActiveShadow[which])
		return -1;

	glBindTexture(GL_TEXTURE_2D, textureNumbers[which]);
	glEnable(GL_TEXTURE_2D);
	wglBindTexImageARB(depthBuf[which], WGL_DEPTH_COMPONENT_NV);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_SGIX, GL_TRUE);

	return textureNumbers[which];
}

void ShadowMaps::Release(int which)
{
	if(valid == 0 || !valid[which] || !hasActiveShadow[which])
		return;

	glBindTexture(GL_TEXTURE_2D, textureNumbers[which]);	
	wglReleaseTexImageARB(depthBuf[which], WGL_DEPTH_COMPONENT_NV);
}

static void RenderSingleObjectDepth(ObjectIterator* oi)
{
	if(!oi->CastsShadow())
		return;

	float* vert = 0;
	long numVert;
	long* poly = 0;
	long numPoly;

	if(! oi->GetPolys(vert, numVert, poly, numPoly))
	{
		oi->Print("Isn't a polygon object (in computation of shadow map).", "");
		return;
	}

	float w[16];
	oi->GetWorldMatrix(w);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(w);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vert);
	glDrawElements(GL_QUADS, 4*numPoly, GL_UNSIGNED_INT, poly);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopMatrix();
}

void ShadowMaps::Render(BaseDocument* doc, Materials* mat)
{
	if(valid == 0)
		return;

	int s;
	for(s=0; s<number; ++s)
	{
		long lightType;
		C4DWrapper::GetObjectLong(lights->GetLight(lights->GetLightFromShadow(s)), LIGHT_TYPE, &lightType);
		long shadowType;
		C4DWrapper::GetObjectLong(lights->GetLight(lights->GetLightFromShadow(s)), LIGHT_SHADOWTYPE, &shadowType);

		hasActiveShadow[s] = (lightType == LIGHT_TYPE_SPOT && shadowType != LIGHT_SHADOWTYPE_NONE);

		if(valid[s] && hasActiveShadow[s])
		{
			Begin(s);
			ObjectIterator oi(doc, mat, RenderSingleObjectDepth);
			End();
		}
	}
}




FXWrapper::FXWrapper(void)
: valid(false), effect(0), effectDesc(new CgFXEFFECT_DESC),
numTechniques(0)
{}

FXWrapper::~FXWrapper(void)
{
	valid = false;

	delete effectDesc;
	effectDesc = 0;

	ReleaseEffect();
}

void FXWrapper::ReleaseEffect(void)
{
	if (effect != 0)
	{
		effect->Release();
		effect = 0;
	}
}

ICgFXEffect* FXWrapper::GetEffect(void) const
{
	return effect;
}

bool FXWrapper::IsValid(void) const
{
	return valid;
}

bool FXWrapper::PrepareAndValidateTechnique(int t)
{
	if(FAILED(effect->SetTechnique((LPCSTR)t)))
		return false;

	if(effect == 0 || FAILED(effect->Validate()))
	{
		return false;
	}
	return true;
}




FXEmulator::FXEmulator(BaseDocument* doc, BaseMaterial* mat, Lights* li,
					   int mapSize, unsigned char* const texturePic, ShadowMaps* sm)
: document(doc), baseMaterial(mat), lit(li), size(mapSize), pic(texturePic), shadows(sm)
{}

FXEmulator::~FXEmulator(void)
{}

// called once for a frame sequence
bool FXEmulator::Load(const char** errors)
{
	ReleaseEffect();

	ostringstream fxCode;
	fxCode <<
		"float4x4 wvp;\n"
		"float4x4 wit;\n"
		"float4x4 w;\n"
		"float4x4 vit;\n"
		"float4 diffCol;\n"
		"float4 lumiCol;\n"
		"texture diffuseTexture;\n"
		"float bumpHeight;\n"
		"texture normalTexture;\n"
		"float4 enviCol;\n"
		"texture enviTexture;\n"
		"texture specShapeTexture;\n"
		"float4 specCol;\n" ;
	int i;
	for(i=0; i<lit->GetNumLights(); ++i)
	{
		fxCode <<
			"float4 lightPos" << i << ";\n"
			"float4 lightCol" << i << ";\n"
			"float4 lightParams" << i << ";\n"
			"float4 lightUp" << i << ";\n"
			"float4 lightDir" << i << ";\n"
			"float4 lightSide" << i << ";\n";
	}

	int s;
	for(s=0; s<lit->GetNumShadowLights(); ++s)
	{
		fxCode <<
			"texture depthTexture" << s << ";\n";
	}

	fxCode <<
		"sampler2D diffuseSampler = sampler_state\n"
		"{	Texture = <diffuseTexture>;\n"
		"	MinFilter = Linear;\n"
		"	MagFilter = Linear;\n"
		"	MipFilter = Linear;\n"
		"};\n"
		"sampler1D specShapeSampler = sampler_state\n"
		"{	Texture = <specShapeTexture>;\n"
		"	AddressU = Clamp;\n"
		"	MinFilter = Linear;\n"
		"	MagFilter = Linear;\n"
		"	MipFilter = None;\n"
		"};\n"
		"sampler2D normalSampler = sampler_state\n"
		"{	Texture = <normalTexture>;\n"
		"	MinFilter = Linear;\n"
		"	MagFilter = Linear;\n"
		"	MipFilter = Linear;\n"
		"};\n"
		"samplerCUBE envMapSampler = sampler_state\n"
		"{	Texture = <enviTexture>;\n"
		"	MinFilter = Linear;\n"
		"	MagFilter = Linear;\n"
		"	MipFilter = Linear;\n"
		"};\n";

	for(s=0; s<lit->GetNumShadowLights(); ++s)
	{
		fxCode <<
			"sampler2D depthSampler"<<s<<" = sampler_state\n"
			"{	Texture = <depthTexture"<<s<<">;\n"
			"	MinFilter = Linear;\n"
			"	MagFilter = Linear;\n"
			"	MipFilter = None;\n"
			"	AddressU = Border;\n"
			"	AddressV = Border;\n"
			"	BorderColor = {0.0f, 0.0f, 0.0f, 1.0f};\n};\n";
	}

	fxCode <<
		"struct appdata {\n"
		"	float4 position	: POSITION;\n"
		"	float4 norm : NORMAL;\n"
		"	float4 uv: TEXCOORD0;\n"
		"	float4 tang : TEXCOORD1;\n" // or TANGENT
 		"	float4 binorm : TEXCOORD2;\n" // or BINORMAL
		"};\n"
		"struct vertexOutput {\n"
		"	float4 hPos : POSITION;\n"
		"	float4 uv : TEXCOORD0;\n"
		"	float3 norm : TEXCOORD1;\n"
		"	float3 tang : TEXCOORD2;\n"
		"	float3 binorm : TEXCOORD3;\n"
		"	float3 view : TEXCOORD4;\n"
		"	float3 wPos : TEXCOORD5;\n"
		"};\n"
		"struct pixelOutput {\n"
		"	float3 col : COLOR;\n"
		"};\n"
		"vertexOutput mainVS(appdata IN,\n"
		"	uniform float4x4 wvp,\n"
		"	uniform float4x4 wit,\n"
		"	uniform float4x4 w,\n"
		"	uniform float4x4 vit)\n"
		"{\n"
		"	vertexOutput OUT;\n"
		"	OUT.uv = IN.uv;\n"
		"	OUT.hPos = mul(wvp, IN.position);\n"
		"	OUT.norm = mul(wit, IN.norm).xyz;\n"
		"	OUT.tang = mul(w, IN.tang).xyz;\n"
		"	OUT.binorm = mul(w, IN.binorm).xyz;\n"
		"	float3 pW = mul(w, IN.position).xyz;\n"
		"	OUT.wPos = pW;\n"
		"	OUT.view = normalize(pW - vit[3].xyz);\n"
		"	return OUT;\n"
		"}\n"
		"pixelOutput mainPS(vertexOutput IN,\n"
		"	uniform sampler2D diffuseSampler,\n"
		"	uniform sampler1D specShapeSampler,\n"
		"	uniform sampler2D normalSampler,\n"
		"	uniform samplerCUBE enviSampler,\n";

	for(s=0; s<lit->GetNumShadowLights(); ++s)
	{
		fxCode << "	uniform sampler2D depthSampler" << s << ",\n";
	}

	fxCode <<
		"	uniform float4 lumiCol,\n"	
		"	uniform float4 diffCol,\n"
		"	uniform float bumpHeight,\n"
		"	uniform float4 enviCol,\n"
		"	uniform float4 specCol";

	for(i=0; i<lit->GetNumLights(); ++i)
	{
		fxCode <<
		",\n	uniform float4 lightPos" << i << ",\n"
		"	uniform float4 lightCol" << i << ",\n"
		"	uniform float4 lightParams" << i << ",\n"
		"	uniform float4 lightUp" << i << ",\n"
		"	uniform float4 lightDir" << i << ",\n"
		"	uniform float4 lightSide" << i;
	}
	
	fxCode <<
		")\n"
		"{\n"
		"	pixelOutput OUT;\n"
		"	float3 Vn = normalize(IN.view);\n"
		"	float3 Nn = normalize(IN.norm);\n"
		"	float3 tangn = normalize(IN.tang);\n"
		"	float3 binormn = normalize(IN.binorm);\n"
		"	float2 bumps = bumpHeight*(tex2D(normalSampler, IN.uv.xy).xy * 2.0 - float2(1.0, 1.0));\n"
		"	float3 Nb = normalize(bumps.x*tangn + bumps.y*binormn + Nn);\n"
		"	float3 env = texCUBE(enviSampler, reflect(Vn, Nb)).rgb;\n"
		"	float3 colorSum = lumiCol.rgb + env*enviCol.rgb;\n"
		"	float3 baseDiffCol = diffCol.rgb + tex2D(diffuseSampler, IN.uv.xy).rgb;\n";

	for(i=0; i<lit->GetNumLights(); ++i)
	{
		fxCode <<
			"	{\n" // block, to use local variables with common names
			"		float3 Ld = lightPos"<<i<<".xyz - IN.wPos;\n"
			"		float3 Ln = normalize(Ld);\n"
			"		float3 baseCol = max(0.0, dot(Ln, Nb))*baseDiffCol;\n"
			"		float spec = tex1D(specShapeSampler, dot(Vn, reflect(Ln, Nb))).r;\n"
			"		baseCol += specCol.rgb*spec;\n"
			"		float3 L1 = (Ln/dot(Ln, lightDir"<<i<<".xyz) - lightDir"<<i<<".xyz)*lightParams"<<i<<".z;\n"
			"		float shadowFactor = max(lightParams"<<i<<".x, smoothstep(1.0, lightParams"<<i<<".w, length(L1)));\n";

		if(lit->HasShadowMap(i))
		{
			fxCode <<
				"		float d = dot(Ld, lightDir"<<i<<".xyz);\n" // negative
				"		float z = 10.1010101/d + 1.01010101;\n" //+ (1.0/(1.0/10.0 - 1.0/1000.0))*1.0/d + 1.0/(1.0 - 10.0/1000.0); "
				"		float2 depthUV = float2(0.5, 0.5) + 0.5*float2(dot(L1, lightSide"<<i<<".xyz), dot(L1, lightUp"<<i<<".xyz));\n"	
				"		shadowFactor *= max(lightParams"<<i<<".y, tex2Dproj(depthSampler"<<lit->GetShadowFromLight(i)<<", float4(depthUV.x, depthUV.y, z-0.0002, 1.0)).x);\n";
		}

		fxCode <<
			"		colorSum += shadowFactor*baseCol*lightCol"<<i<<".rgb;\n"
			"	}\n";
	}

	fxCode <<
		"	OUT.col = colorSum;\n"
		"	return OUT;\n"
		"}\n"
		"technique t0\n"
		"{\n"
		"	pass p0\n" 
		"	{\n" 
		"		VertexShader = compile vs_2_x mainVS(wvp, wit, w, vit);\n"
		"		ZEnable = true;\n"
		"		ZWriteEnable = true;\n"
		"		CullMode = None;\n"
		"		PixelShader = compile ps_2_x mainPS(diffuseSampler, specShapeSampler, normalSampler, envMapSampler,\n";
	
	for(s=0; s<lit->GetNumShadowLights(); ++s)
	{
		fxCode << "			depthSampler"<<s<<",\n";
	}
		
	fxCode <<
		"			lumiCol, diffCol, bumpHeight, enviCol, specCol";

	for(i=0; i<lit->GetNumLights(); ++i)
	{
		fxCode <<
			",\n"
			"			lightPos"<<i<<", lightCol"<<i<<", lightParams"<<i<<", lightUp"<<i<<", lightDir"<<i<<", lightSide"<<i;
	}
	
	fxCode <<
		");\n"
		"	}\n"
		"}";

#ifndef NDEBUG
	ofstream of("C:\\intern.fx", ios_base::out | ios_base::trunc);
	of << fxCode.str();
	of.close();
#endif

	const char* fxString = fxCode.str().c_str();	
	
	if (FAILED(CgFXCreateEffect(fxString, 0, &effect, errors)) || errors != 0 && errors[0] != 0)
	{
#ifndef NDEBUG
		_asm{int 3}
#endif
		return false;
	}

	if (FAILED(effect->GetDesc(effectDesc)))
		return false;
	
	valid = true;
	return true;
}

bool FXEmulator::BeginRenderingObject(ObjectIterator* oi)
{
	if(!valid)
		return false;

	float a[16], b[16], c[16], w[16], v[16];

	if(! oi->GetWorldMatrix(w))
		return false;
	if(! C4DWrapper::GetViewMatrix(document, v))
		return false;
	MatrixMath::Mult(v, w, c);
	if(! C4DWrapper::GetProjMatrix(document, a))
		return false;
	MatrixMath::Mult(a, c, b);
	MatrixMath::Transpose(b, c); // CgFX wants other colum/row order
	if(FAILED(effect->SetMatrix((LPCSTR)0, c, 4, 4)))
		return false;

	MatrixMath::Invert(w, c); // CgFX wants other colum/row order
	if(FAILED(effect->SetMatrix((LPCSTR)1, c, 4, 4)))
		return false;

	MatrixMath::Transpose(w, c); // CgFX wants other colum/row order
	if(FAILED(effect->SetMatrix((LPCSTR)2, c, 4, 4)))
		return false;

	return true;
}

void FXEmulator::EndRenderingObject(void)
{
	if(!valid)
		return;

	// add more cleanup code here, if necessary
}

// BaseDocument* d only needed here because FXFromFile (sister of this class) doesn't know its doc
// same for BaseMaterial* m
void FXEmulator::BeginRendering(BaseDocument* d, BaseMaterial* m)
{
	if(!valid)
		return;

	if( !C4DWrapper::GetMaterialBool(baseMaterial, MATERIAL_USE_COLOR, &useDiff)
		|| !C4DWrapper::GetMaterialBool(baseMaterial, MATERIAL_USE_BUMP, &useBump)
		|| !C4DWrapper::GetMaterialBool(baseMaterial, MATERIAL_USE_ENVIRONMENT, &useEnvi)
		|| !C4DWrapper::GetMaterialBool(baseMaterial, MATERIAL_USE_LUMINANCE, &useLumi)
		|| !C4DWrapper::GetMaterialBool(baseMaterial, MATERIAL_USE_SPECULAR, &useSpec)
		|| !C4DWrapper::GetMaterialBool(baseMaterial, MATERIAL_USE_SPECULARCOLOR, &useSpecColor) )
		return;

	float c[16], v[16];
	if(C4DWrapper::GetViewMatrix(document, v))
	{
		MatrixMath::Invert(v, c); // CgFX wants other colum/row order, so no transpose
		effect->SetMatrix((LPCSTR)3, c, 4, 4);
	}

	float diffCol[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	if(useDiff)
	{
		float col[4];
		float val;
		if( C4DWrapper::GetMaterialVector(baseMaterial, MATERIAL_COLOR_COLOR, col)
			&& C4DWrapper::GetMaterialFloat(baseMaterial, MATERIAL_COLOR_BRIGHTNESS, &val) )
		{
				diffCol[0] = col[0]*val;
				diffCol[1] = col[1]*val;
				diffCol[2] = col[2]*val;
		}

		glGenTextures(1, &diffTextureNumber);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, diffTextureNumber);
		if(C4DWrapper::GetMaterialTexture(baseMaterial, C4DWrapper::GetChannelIdColor(), pic, size))
		{
			gluBuild2DMipmaps(GL_TEXTURE_2D, 3, size, size, GL_RGB, GL_UNSIGNED_BYTE, pic);
			effect->SetTexture((LPCSTR)6, (DWORD)diffTextureNumber);
		}
	}
	effect->SetVector((LPCSTR)4, diffCol, 4);

	float lumiCol[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	if(useLumi)
	{
		float col[4];
		float val;
		if( C4DWrapper::GetMaterialVector(baseMaterial, MATERIAL_LUMINANCE_COLOR, col)
			&& C4DWrapper::GetMaterialFloat(baseMaterial, MATERIAL_LUMINANCE_BRIGHTNESS, &val) )
		{
			lumiCol[0] = col[0]*val;
			lumiCol[1] = col[1]*val;
			lumiCol[2] = col[2]*val;
		}
	}
	effect->SetVector((LPCSTR)5, lumiCol, 4);

	float bumpHeight = 0.0;
	if(useBump)
	{
		if(C4DWrapper::GetMaterialFloat(baseMaterial, MATERIAL_BUMP_STRENGTH, &bumpHeight))
		{
			bumpHeight *= 5.0*size/256.0;
		}

		glGenTextures(1, &normTextureNumber);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, normTextureNumber);
		if(C4DWrapper::GetMaterialNormalTexture(baseMaterial, pic, size))
		{
			gluBuild2DMipmaps(GL_TEXTURE_2D, 3, size, size, GL_RGB, GL_UNSIGNED_BYTE, pic);
			effect->SetTexture((LPCSTR)8, (DWORD)normTextureNumber);
		}
	}
	effect->SetValue((LPCSTR)7, &bumpHeight, sizeof(bumpHeight));

	float enviCol[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	if(useEnvi)
	{
		float col[4];
		float val;
		if( C4DWrapper::GetMaterialVector(baseMaterial, MATERIAL_ENVIRONMENT_COLOR, col)
			&& C4DWrapper::GetMaterialFloat(baseMaterial, MATERIAL_ENVIRONMENT_BRIGHTNESS, &val) )
		{
			enviCol[0] = col[0]*val;
			enviCol[1] = col[1]*val;
			enviCol[2] = col[2]*val;
		}

		glGenTextures(1, &enviTextureNumber);
		glEnable(GL_TEXTURE_CUBE_MAP_ARB);
		glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, enviTextureNumber);
		int dir;
		for(dir=0; dir<6; ++dir)
		{
			const int which[] = {	GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB, GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB,
									GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB,
									GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB};
			if(C4DWrapper::GetMaterialEnvTexture(baseMaterial, pic, size/2, dir)) // "size/2": not _that_ large, because we have six single maps in the cube map
			{
				gluBuild2DMipmaps(which[dir], 3, size/2, size/2, GL_RGB, GL_UNSIGNED_BYTE, pic); // "size/2": see above
			}
		}
		effect->SetTexture((LPCSTR)10, (DWORD)enviTextureNumber);
	}
	effect->SetVector((LPCSTR)9, enviCol, 4);

	float specCol[] = {0.0f, 0.0f, 0.0f, 0.0f};
	float specParams[] = {0.0f, 0.0f, 0.0f, 0.0f};
	if(useSpec)
	{
		float height;
		if( C4DWrapper::GetMaterialFloat(baseMaterial, MATERIAL_SPECULAR_HEIGHT, &height) )
		{
			glGenTextures(1, &specShapeTextureNumber);
			glEnable(GL_TEXTURE_1D);
			glBindTexture(GL_TEXTURE_1D, specShapeTextureNumber);
			if(C4DWrapper::GetMaterialSpecShapeTexture(baseMaterial, pic, 1024)) // note that pic is always larger than 64*64*3 bytes
			{
				glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE, 1024, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pic);
				effect->SetTexture((LPCSTR)11, (DWORD)specShapeTextureNumber);
			}

			specCol[0] = height;
			specCol[1] = height;
			specCol[2] = height;
			
			if(useSpecColor)
			{
				float col[4];
				float val;
				if( C4DWrapper::GetMaterialVector(baseMaterial, MATERIAL_SPECULAR_COLOR, col)
					&& C4DWrapper::GetMaterialFloat(baseMaterial, MATERIAL_SPECULAR_BRIGHTNESS, &val) )
				{
					specCol[0] = col[0]*val*height;
					specCol[1] = col[1]*val*height;
					specCol[2] = col[2]*val*height;
				}
			}
		}
	}
	effect->SetVector((LPCSTR)12, specCol, 4);

	int paramNum = 13;

	int i;
	for(i=0; i<lit->GetNumLights(); ++i)
	{
		effect->SetVector((LPCSTR)paramNum++, lit->GetPos(i), 4);

		float lightCol[4] = {0.0f, 0.0f, 0.0f, 1.0f};
		float col[4];
		float val;
		if( C4DWrapper::GetObjectVector(lit->GetLight(i), LIGHT_COLOR, col)
			&& C4DWrapper::GetObjectFloat(lit->GetLight(i), LIGHT_BRIGHTNESS, &val) )
		{
			lightCol[0] = col[0]*val;
			lightCol[1] = col[1]*val;
			lightCol[2] = col[2]*val;
		}
		effect->SetVector((LPCSTR)paramNum++, lightCol, 4);

		long lightType = 0;
		C4DWrapper::GetObjectLong(lit->GetLight(i), LIGHT_TYPE, &lightType);
		float isPointLight = (lightType != LIGHT_TYPE_SPOT) ? 1.0f: 0.0f;
		long shadowType = 0;
		C4DWrapper::GetObjectLong(lit->GetLight(i), LIGHT_SHADOWTYPE, &shadowType);
		float noShadow = (lightType != LIGHT_TYPE_SPOT || shadowType == LIGHT_SHADOWTYPE_NONE) ? 1.0f: 0.0f;
		// The following parameters may be undefined for non-spot lights.
		// Pass them anyway. Won't hurt, but keeps paramNum in sync.
		float angle = 0.0f;
		C4DWrapper::GetObjectFloat(lit->GetLight(i), LIGHT_DETAILS_OUTERANGLE, &angle);
		float invTan = 1.0/(tan(0.5*angle)+0.00001);
		C4DWrapper::GetObjectFloat(lit->GetLight(i), LIGHT_DETAILS_INNERANGLE, &angle);
		float innerOuterAngRel = tan(0.5*angle)*invTan;
		float lightParams[4] = {isPointLight, noShadow, invTan, innerOuterAngRel};
		effect->SetVector((LPCSTR)paramNum++, lightParams, 4);

		effect->SetVector((LPCSTR)paramNum++, lit->GetUp(i), 4);
		effect->SetVector((LPCSTR)paramNum++, lit->GetDir(i), 4);
		effect->SetVector((LPCSTR)paramNum++, lit->GetSide(i), 4);
	}

	int s;
	for(s=0; s<lit->GetNumShadowLights(); ++s)
	{
		int depthTexNum = shadows->Bind(s);
		if(depthTexNum != -1)
			effect->SetTexture((LPCSTR)paramNum++, (DWORD)depthTexNum);
	}
}

void FXEmulator::EndRendering(void)
{
	if(!valid)
		return;
	if(useSpec)
		glDeleteTextures(1, &specShapeTextureNumber);
	if(useEnvi)
		glDeleteTextures(1, &enviTextureNumber);
	if(useBump)
		glDeleteTextures(1, &normTextureNumber);
	if(useDiff)
		glDeleteTextures(1, &diffTextureNumber);

	int s;
	for(s=0; s<lit->GetNumShadowLights(); ++s)
	{
		shadows->Release(s);
	}
}



FXFromFile::FXFromFile(void)
: paramList(0), numParameters(0), techniqueNames(0)
{}

FXFromFile::~FXFromFile(void)
{
	if(paramList != 0)
	{
		int i;
		for(i=0; i<numParameters; ++i)
		{
			delete paramList[i];
			paramList[i] = 0;
		}
		numParameters = 0;
	}
	delete[] paramList;
	paramList = 0;

	if(techniqueNames != 0)
	{
		int i;
		for(i=0; i<numTechniques; ++i)
		{
			delete[] techniqueNames[i];
			techniqueNames[i] = 0;
		}
		delete[] techniqueNames;
		techniqueNames = 0;
	}
}

bool FXFromFile::Load(char* file, const char** errors)
{
	valid = false;

	// Don't delete effectDesc; This is only deleted in the destructor.
	// Has to be a pointer, however, because we don't know its size in the .h file.

	if(paramList != 0)
	{
		int i;
		for(i=0; i<numParameters; ++i)
		{
			delete paramList[i];
			paramList[i] = 0;
		}
		delete[] paramList;
		paramList = 0;
	}
	numParameters = 0;

	if(techniqueNames != 0)
	{
		int i;
		for(i=0; i<numTechniques; ++i)
		{
			delete[] techniqueNames[i];
			techniqueNames[i] = 0;
		}
		delete[] techniqueNames;
		techniqueNames = 0;
	}
	numTechniques = 0;

	ReleaseEffect();

	if(FAILED(CgFXCreateEffectFromFileA(file, 0, &effect, errors)) || errors != 0 && errors[0] != 0)
		return false;

	if (FAILED(effect->GetDesc(effectDesc)))
		return false;

	int i;

	numTechniques = effectDesc->Techniques;
	techniqueNames = new char*[numTechniques];
	if(techniqueNames == 0)
		return false;
	memset(techniqueNames, 0, numTechniques*sizeof(char*));
	CgFXTECHNIQUE_DESC desc;
	for(i=0; i<numTechniques; ++i)
	{
		if(FAILED(effect->GetTechniqueDesc((LPCSTR)i, &desc)))
			return false;
		const char* n = desc.Name;
		if(n==0)
			n = "---";
		techniqueNames[i] = new char[strlen(n)+1];
		if(techniqueNames[i] == 0)
			return false;
		strcpy(techniqueNames[i], n);
	}
	
	numParameters = effectDesc->Parameters;
	paramList = new ParamWrapper*[numParameters];
	if(paramList == 0)
		return false;
	memset(paramList, 0, numParameters*sizeof(ParamWrapper*));
	for(i=0; i<numParameters; ++i)
	{
		paramList[i] = ParamWrapper::BuildParamWrapper(effect, i);
		if(paramList[i] == 0)
		{
			valid = false;
			return false;
		}
	}

	valid = true;
	return true;
}

// if init == false, just build the UI
// if setToDefault == false, look if current value is in range. If not, let it be initialized nonetheless.
bool FXFromFile::BuildUI_Technique(bool init, bool setToDefault, Description* description, BaseContainer* data) const
{
	if(!valid)
		return false;

	int tech;
	if(!C4DWrapper::GetTechnique(tech, data))
		return false;
	if(tech >= numTechniques)
	{
		setToDefault = true;
	}

	return C4DWrapper::BuildUI_Technique(techniqueNames, effectDesc->Techniques, setToDefault, description, data);
}

// if init == false do no checks.
// if setToDefault == false, look if current value is in range. If not, let it be initialized nonetheless.
bool FXFromFile::BuildUI_Parameters(bool init, bool setToDefault, Description* description, BaseContainer* data) const
{
	if(!valid)
		return false;

	int i;
	for(i=0; i<numParameters; ++i)
	{
		if(! paramList[i]->BuildUI(init, setToDefault, description, data))
			return false;
	}

	if(init)
	{
		// clear what may be there behind
		for(i=numParameters; i<100; ++i) // never have more than 100 parameters
		{
			C4DWrapper::RemoveData(20000+i, data);
		}
	}

	return true;
}

bool FXFromFile::BeginRenderingObject(ObjectIterator* oi)
{
	if(!valid)
		return false;

	int i;
	for(i=0; i<numParameters; ++i)
	{
		if(! paramList[i]->BeginRenderingObject(oi))
			return false;
	}
	return true;
}

void FXFromFile::EndRenderingObject(void)
{
	if(!valid)
		return;

	int i;
	for(i=0; i<numParameters; ++i)
	{
		paramList[i]->EndRenderingObject();
	}
}

bool FXFromFile::IsParamEnabled(int i)
{
	if(i<FX_TECHNIQUE)
		return true;

	if(!valid)
		return false;

	if(i<20000 || i>20100) // never have more than 100 parameters
		return true;

	if(paramList == 0) // This knid of request seems to happen when animating.
		return false;

	return paramList[i-20000]->IsEnabled();
}

void FXFromFile::BeginRendering(BaseDocument* d, BaseMaterial* m)
{
	if(!valid)
		return;

	int i;
	for(i=0; i<numParameters; ++i)
	{
		paramList[i]->BeginRendering(d, m);
	}
}

void FXFromFile::EndRendering(void)
{
	if(!valid)
		return;

	int i;
	for(i=0; i<numParameters; ++i)
	{
		paramList[i]->EndRendering();
	}
}



ParamWrapper::ParamWrapper(ICgFXEffect* effect_, int index_)
: effect(effect_), effectDesc(new CgFXEFFECT_DESC), paramDesc(new CgFXPARAMETER_DESC), index(index_), ok(false), uiName(0)
{
	if(FAILED(effect->GetDesc(effectDesc))) // this MUST NOT be deleted in advance!!! We need the storage.
		return;
	if(FAILED(effect->GetParameterDesc((LPCSTR)index, paramDesc))) // see above
		return;
	ok = true;

	const char* n = 0;
	if(!GetAnnotation("Desc", n) || n==0 || n[0]==0)
	{
		if(!GetAnnotation("Object", n) || n==0 || n[0]==0)
		{
			n = paramDesc->Name;
		}
		n = paramDesc->Name;
	}
	if(n == 0 || n[0]==0)
		n = "---";
	uiName = new char[strlen(n)+1];
	strcpy(uiName, n);
}

ParamWrapper::ParamWrapper(const ParamWrapper* pw)
: effect(pw->effect), effectDesc(0), paramDesc(0), index(pw->index), ok(pw->ok), uiName(0)
{
	uiName = new char[strlen(pw->uiName)+1];
	strcpy(uiName, pw->uiName);

	effectDesc = new CgFXEFFECT_DESC;
	paramDesc = new CgFXPARAMETER_DESC;
	memcpy(effectDesc, pw->effectDesc, sizeof(CgFXEFFECT_DESC));
	memcpy(paramDesc, pw->paramDesc, sizeof(CgFXPARAMETER_DESC));
}

ParamWrapper::~ParamWrapper(void)
{
	delete effectDesc;
	effectDesc = 0;
	delete paramDesc;
	paramDesc = 0;
	delete[] uiName;
	uiName = 0;
}

bool ParamWrapper::BeginRenderingObject(ObjectIterator* oi)
{
	return true;
}

void ParamWrapper::EndRenderingObject(void)
{
}

bool ParamWrapper::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{	
	return C4DWrapper::BuildUI_Hidden(index, description, data);
}

bool ParamWrapper::IsEnabled(void)
{
	return false;
}

ParamWrapper* ParamWrapper::BuildParamWrapper(ICgFXEffect* effect, int i)
{
	ParamWrapper p0(effect, i);
	if(! p0.Check())
		return 0;

	ParamWrapper* p = 0;
	p = new DirectionOrPositionParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p = new SuppressedParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p = new TextureParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p = new StringParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p = new MatrixParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p= new FloatParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p= new ColorParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	p= new VectorParam(&p0);
	if(p == 0)
		goto error;
	if(p->Check())
		return p;
	delete p;

	return new UnknownParam(&p0);

error:
	C4DWrapper::Print("Couldn't allocate parameter.");
	return 0;
}

bool ParamWrapper::GetAnnotation(const char* name, float &a) const
{
	unsigned int j;
	for (j=0; j<paramDesc->Annotations; ++j)
	{
		CgFXANNOTATION_DESC annotation;
		if (FAILED(effect->GetAnnotationDesc((LPCSTR)index, (LPCSTR)j, &annotation)))
			return false;

		if (annotation.Type==CgFXPT_FLOAT && stricmp(annotation.Name, name) == 0)
		{
			a = *(float*)annotation.Value;
			return true;
		}
	}
	return false;
}

bool ParamWrapper::GetAnnotation(const char* name, const char* &s) const
{
	unsigned int j;
	for (j=0; j<paramDesc->Annotations; ++j)
	{
		CgFXANNOTATION_DESC annotation;
		if (FAILED(effect->GetAnnotationDesc((LPCSTR)index, (LPCSTR)j, &annotation)))
			return false;

		if(annotation.Type==CgFXPT_STRING && stricmp(annotation.Name, name) == 0)
		{
			s = (const char*)annotation.Value;
			return true;
		}
	}
	return false;
}

void ParamWrapper::BeginRendering(BaseDocument* d, BaseMaterial* m)
{}

void ParamWrapper::EndRendering(void)
{}

bool UnknownParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	// "true" to always overwrite data that may have been in the container
	return C4DWrapper::BuildUI_String("UNKNOWN TYPE", paramDesc->Name, index, true, description, data);
}

bool UnknownParam::IsEnabled(void)
{
	return false;
}

UnknownParam::UnknownParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	ok = effectDesc!=0 && paramDesc!=0;
}

UnknownParam::~UnknownParam(void)
{}

bool FloatParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	float f = 0.12345f;

	if(init)
	{
		if(C4DWrapper::IsParamFloat(index, data))
		{
			if(!C4DWrapper::GetParamFloat(index, f, data))
				return false;
			if(f > maxVal || f < minVal)
			{
				setToDefault = true;
			}
		}
		else
		{
			setToDefault = true;
		}

		if(setToDefault)
			f = defaultValue;
	}

	return C4DWrapper::BuildUI_Float(f, minVal, maxVal, stepVal, hasSlider, uiName, index, init && setToDefault, description, data);
}

bool FloatParam::IsEnabled(void)
{
	return true;
}

FloatParam::FloatParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	ok = paramDesc->Type==CgFXPT_FLOAT && paramDesc->Dimension[0]==1 && paramDesc->Dimension[1]==0 && paramDesc->Dimension[2]==0 && paramDesc->Dimension[3]==0;
	if(!ok) return;

	if(! GetAnnotation("uimin", minVal))
		minVal = -1e9f;
	if(! GetAnnotation("uimax", maxVal))
		maxVal = 1e9f;
	if(! GetAnnotation("uistep", stepVal))
		stepVal = 0.1f;

	const char* str;
	hasSlider = GetAnnotation("gui", str) && stricmp((const char*)str, "slider")==0;
	effect->GetFloat((LPCSTR)index, &defaultValue);
}

FloatParam::~FloatParam()
{}

void FloatParam::BeginRendering(BaseDocument* d, BaseMaterial* m)
{
	float a;
	if(! C4DWrapper::GetParamFloat(m, index, a))
		return;
	
	effect->SetValue((LPCSTR)index, &a, sizeof(a)); // return value not checked?!
}

// strings are not editable, so simply always write them over existing data
bool StringParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	return C4DWrapper::BuildUI_String(str, uiName, index, true, description, data);
}

bool StringParam::IsEnabled(void)
{
	return false;
}

StringParam::StringParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	str = 0;
	ok = paramDesc->Type==CgFXPT_STRING;
	if(!ok) return;

	const char* s;
	effect->GetString((LPCSTR)index, &s);
	delete[] str;
	str = new char[strlen(s)+1];
	strcpy(str, s);
}

StringParam::~StringParam(void)
{
	delete[] str;
	str = 0;
}

bool MatrixParam::BeginRenderingObject(ObjectIterator* oi)
{
	if(!containsWorld)
		return true; // have already build it globally in BeginRendering

	float m[] = {	1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 0.0f, 1.0f	};
	float a[16], b[16], c[16], e[16];
	float *p;

	if(! oi->GetWorldMatrix(m))
			return false;

	p = m; //relict

	if(containsView)
	{
		float view[16];
		if(! C4DWrapper::GetViewMatrix(oi->GetDocument(), view))
			return false;
		MatrixMath::Mult(view, p, a);
		p = a;
	}

	if(containsProjection)
	{
		float proj[16];
		if(! C4DWrapper::GetProjMatrix(oi->GetDocument(), proj))
			return false;
		MatrixMath::Mult(proj, p, b);
		p = b;
	}

	if(isInverted)
	{
		MatrixMath::Invert(p, c);
		p = c;
	}

	if( ! isTransposed ) // CgFX assumes row-major matrices and we're manipulating OpenGL column-major matrices
	{
		MatrixMath::Transpose(p, e);
		p = e;
	}

	effect->SetMatrix((LPCSTR)index, p, 4, 4);

	return true;
}

// if init == false, look if current value is in range. If not, let it be initialized nonetheless.
// With this type, no explicit action is necessary
bool MatrixParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	return C4DWrapper::BuildUI_Hidden(index, description, data);
}

bool MatrixParam::IsEnabled(void)
{
	return false;
}

MatrixParam::MatrixParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	ok = false;

	if(stricmp(paramDesc->Semantic, "World") == 0)
	{
		containsWorld = true; containsView = false; containsProjection = false; isTransposed = false; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldI") == 0)
	{
		containsWorld = true; containsView = false; containsProjection = false; isTransposed = false; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldT") == 0)
	{
		containsWorld = true; containsView = false; containsProjection = false; isTransposed = true; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldIT") == 0)
	{
		containsWorld = true; containsView = false; containsProjection = false; isTransposed = true; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "View") == 0)
	{
		containsWorld = false; containsView = true; containsProjection = false; isTransposed = false; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "ViewI") == 0)
	{
		containsWorld = false; containsView = true; containsProjection = false; isTransposed = false; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "ViewT") == 0)
	{
		containsWorld = false; containsView = true; containsProjection = false; isTransposed = true; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "ViewIT") == 0)
	{
		containsWorld = false; containsView = true; containsProjection = false; isTransposed = true; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldView") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = false; isTransposed = false; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewI") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = false; isTransposed = false; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewT") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = false; isTransposed = true; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewIT") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = false; isTransposed = true; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "Projection") == 0)
	{
		containsWorld = false; containsView = false; containsProjection = true; isTransposed = false; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "ProjectionI") == 0)
	{
		containsWorld = false; containsView = false; containsProjection = true; isTransposed = false; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "ProjectionT") == 0)
	{
		containsWorld = false; containsView = false; containsProjection = true; isTransposed = true; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "ProjectionIT") == 0)
	{
		containsWorld = false; containsView = false; containsProjection = true; isTransposed = true; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewProjection") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = true; isTransposed = false; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewProjectionI") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = true; isTransposed = false; isInverted = true;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewProjectionT") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = true; isTransposed = true; isInverted = false;
		ok = true;
	}
	else if(stricmp(paramDesc->Semantic, "WorldViewProjectionIT") == 0)
	{
		containsWorld = true; containsView = true; containsProjection = true; isTransposed = true; isInverted = true;
		ok = true;
	}
}

MatrixParam::~MatrixParam(void)
{}

void MatrixParam::BeginRendering(BaseDocument* d, BaseMaterial* m)
{
	if(containsWorld)
		return; // cannot build it globally

	float mx[] = {	1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 0.0f, 1.0f	};
	float a[16], b[16], c[16], e[16];
	float *p = mx; // relict

	if(containsView)
	{
		float view[16];
		if(! C4DWrapper::GetViewMatrix(d, view))
			return;
		MatrixMath::Mult(view, p, a);
		p = a;
	}

	if(containsProjection)
	{
		float proj[16];
		if(! C4DWrapper::GetProjMatrix(d, proj))
			return;
		MatrixMath::Mult(proj, p, b);
		p = b;
	}

	if(isInverted)
	{
		MatrixMath::Invert(p, c);
		p = c;
	}

	if( ! isTransposed ) // CgFX assumes row-major matrices and we're manipulating OpenGL column-major matrices
	{
		MatrixMath::Transpose(p, e);
		p = e;
	}

	effect->SetMatrix((LPCSTR)index, p, 4, 4);
}

bool ColorParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	float col[] = {0.12345f, 0.12345f, 0.12345f, 0.12345f};

	if(init)
	{
		if(C4DWrapper::IsParamVector(index, data))
		{
			if(!C4DWrapper::GetParamVector(index, col, data))
				return false;
			if(col[0]<0.0f || col[0]>1.0f || col[1]<0.0f || col[1]>1.0f || col[2]<0.0f || col[2]>1.0f || col[3]<0.0f || col[3]>1.0f)
			{
				setToDefault = true;
			}
		}
		else
		{
			setToDefault = true;
		}

		if(setToDefault)
			memcpy(col, defaultValue, 4*sizeof(float));
	}

	return C4DWrapper::BuildUI_Color(col, uiName, index, init && setToDefault, description, data);
}

bool ColorParam::IsEnabled(void)
{
	return true;
}

ColorParam::ColorParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	const char* s = paramDesc->Semantic;
	ok = stricmp(s, "Ambient") == 0
		|| stricmp(s, "Diffuse") == 0
		|| stricmp(s, "Specular") == 0
		|| stricmp(s, "Emissive") == 0;
	ok = ok && paramDesc->Dimension[0] == 4;
	if(!ok) return;

	unsigned int vecSize[] = {4, 0};
	effect->GetVector((LPCSTR)index, defaultValue, vecSize);
}

ColorParam::~ColorParam(void)
{}

void ColorParam::BeginRendering(BaseDocument* d, BaseMaterial* m)
{
	float col[4];
	if(! C4DWrapper::GetParamVector(m, index, col))
		return;
	effect->SetVector((LPCSTR)index, col, 4);
}

bool TextureParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	char str[256]; // better make dynamic
	char* s = str;

	if(init)
	{
		if(C4DWrapper::IsParamFilename(index, data))
		{
			if(!C4DWrapper::GetParamFilename(index, s, data))
				return false;
		}
		else
		{
			setToDefault = true;
		}

		if(setToDefault)
			s = file;
	}

	return C4DWrapper::BuildUI_Filename(s, uiName, index, init && setToDefault, description, data);
}

bool TextureParam::IsEnabled(void)
{
	return true;
}

TextureParam::TextureParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	file = 0;

	long t = paramDesc->Type;
	ok = (t==CgFXPT_TEXTURE || t==CgFXPT_VOLUMETEXTURE || t==CgFXPT_CUBETEXTURE);
	if(!ok) return;

	textureType = t;

	char* dummy = "";
	char* s = 0;
	if(!GetAnnotation("File", s) || s==0)
	{
		if(!GetAnnotation("default", s) || s==0)
		{
			s = dummy;
		}
	}
	delete[] file;
	file = new char[strlen(s)+1];
	strcpy(file, s);
}

TextureParam::~TextureParam(void)
{
	delete[] file;
	file = 0;
}

void TextureParam::BeginRendering(BaseDocument* d, BaseMaterial* m)
{
	glGenTextures(1, &textureNumber);

	char file[256]; // better would be dynamic length
	if(! C4DWrapper::GetParamFilename(m, index, file))
		return;

	int lenFile = strlen(file); 

	if (file!=0 && lenFile<=4)
		return;

	if (strcmp(".dds", file + lenFile - 4) != 0)
	{
		C4DWrapper::Print("No .dds file: ", file);		
		return;
	}

	nv_dds::CDDSImage ddsimage;

	if (!ddsimage.load(file))
	{
		C4DWrapper::Print("Couldn't load: ", file);		
		return;
	}

	if (ddsimage.is_cubemap())
	{
		glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, textureNumber);
		ddsimage.upload_textureCubemap();
	}
	else if (ddsimage.is_volume())
	{
		glBindTexture(GL_TEXTURE_3D, textureNumber);
		ddsimage.upload_texture3D();
	}
	else if (ddsimage.get_height() == 1)
		ddsimage.upload_texture1D();
	else if (	ddsimage.get_height() == pow(2, (int)(0.5+log(ddsimage.get_height())/log(2)))
			&&	ddsimage.get_width() == pow(2, (int)(0.5+log(ddsimage.get_width())/log(2))) )
	{
		glBindTexture(GL_TEXTURE_2D, textureNumber);
		ddsimage.upload_texture2D();
	}
	else
	{
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, textureNumber);
		ddsimage.upload_textureRectangle();
	}
	effect->SetTexture((LPCSTR)index, (DWORD)textureNumber);

	return;
}

void TextureParam::EndRendering(void)
{
	glDeleteTextures(1, &textureNumber);
}

bool SuppressedParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	return C4DWrapper::BuildUI_Hidden(index, description, data);
}

bool SuppressedParam::IsEnabled(void)
{
	return false;
}

SuppressedParam::SuppressedParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	ok = paramDesc->Type==CgFXPT_FORCE_DWORD;
}

SuppressedParam::~SuppressedParam(void)
{}

bool DirectionOrPositionParam::BeginRenderingObject(ObjectIterator* oi)
{
	float v[4];
	if(! oi->GetParamDirectionOrPosition(isPosition, isWorld, index, v))
	{
		oi->Print("No object linked for position or direction parameter?", "");
		return false;
	}

	return !FAILED(effect->SetVector((LPCSTR)index, v, 4));
}

void DirectionOrPositionParam::AfterRender(void)
{
}

bool DirectionOrPositionParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	if(init)
	{
		if(! C4DWrapper::IsParamLink(index, data))
		{
			setToDefault = true;
		}
	}

	return C4DWrapper::BuildUI_Link(uiName, index, init && setToDefault, description, data);
}

bool DirectionOrPositionParam::IsEnabled(void)
{
	return true;
}

DirectionOrPositionParam::DirectionOrPositionParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	isPosition = stricmp(paramDesc->Semantic, "POSITION")==0;
	ok = (isPosition || stricmp(paramDesc->Semantic, "DIRECTION")==0) && paramDesc->Dimension[0]==4;
	if(!ok) return;

	const char* space = 0;
	isWorld = GetAnnotation("Space", space) && stricmp(space, "World")==0;
}

DirectionOrPositionParam::~DirectionOrPositionParam(void)
{}

bool VectorParam::BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data)
{
	float vec[] = {0.12345f, 0.12345f, 0.12345f, 0.12345f};

	if(init)
	{
		if(! C4DWrapper::IsParamVector(index, data))
		{
			setToDefault = true;
		}

		if(setToDefault)
			memcpy(vec, defaultValue, 4*sizeof(float));
	}

	return C4DWrapper::BuildUI_Vector(vec, uiName, index, init && setToDefault, description, data);
}

bool VectorParam::IsEnabled(void)
{
	return true;
}

VectorParam::VectorParam(const ParamWrapper* pw)
: ParamWrapper(pw)
{
	const char* s = paramDesc->Semantic;
	ok = stricmp(s, "Ambient") != 0
		&& stricmp(s, "Diffuse") != 0
		&& stricmp(s, "Specular") != 0
		&& stricmp(s, "Emissive") != 0;
	ok = ok && paramDesc->Dimension[0] == 4;
	if(!ok) return;

	unsigned int vecSize[] = {4, 0};
	effect->GetVector((LPCSTR)index, defaultValue, vecSize);
}

VectorParam::~VectorParam(void)
{}

void VectorParam::BeginRendering(BaseDocument* doc, BaseMaterial* m)
{
	float vec[4];
	if(! C4DWrapper::GetParamVector(m, index, vec))
		return;
	effect->SetVector((LPCSTR)index, vec, 4);
}