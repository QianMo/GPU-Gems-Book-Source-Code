/*
s_p_oneil@hotmail.com
Copyright (c) 2000, Sean O'Neil
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of this project nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __GameEngine_h__
#define __GameEngine_h__

#include "GLUtil.h"
#include "PBuffer.h"
#include "GameApp.h"
#include "Font.h"



class CGameEngine
{
protected:
	float m_fFPS;
	int m_nTime;
	CFont m_fFont;

	C3DObject m_3DCamera;
	CVector m_vLight;
	CVector m_vLightDirection;
	
	// Variables that can be tweaked with keypresses
	bool m_bUseHDR;
	int m_nSamples;
	GLenum m_nPolygonMode;
	float m_Kr, m_Kr4PI;
	float m_Km, m_Km4PI;
	float m_ESun;
	float m_g;
	float m_fExposure;

	float m_fInnerRadius;
	float m_fOuterRadius;
	float m_fScale;
	float m_fWavelength[3];
	float m_fWavelength4[3];
	float m_fRayleighScaleDepth;
	float m_fMieScaleDepth;
	CPixelBuffer m_pbOpticalDepth;

	CTexture m_tMoonGlow;
	CTexture m_tEarth;

	CShaderObject m_shSkyFromSpace;
	CShaderObject m_shSkyFromAtmosphere;
	CShaderObject m_shGroundFromSpace;
	CShaderObject m_shGroundFromAtmosphere;
	CShaderObject m_shSpaceFromSpace;
	CShaderObject m_shSpaceFromAtmosphere;

	CPBuffer m_pBuffer;

public:
	CGameEngine();
	~CGameEngine();
	void RenderFrame(int nMilliseconds);
	void Pause()	{}
	void Restore()	{}
	void HandleInput(float fSeconds);
	void OnChar(WPARAM c);
};

#endif // __GameEngine_h__
