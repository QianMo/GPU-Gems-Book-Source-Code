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

#ifndef __PBuffer_h__
#define __PBuffer_h__

#define MAX_PFORMATS 256
#define MAX_ATTRIBS  32

#include "GLUtil.h"

class CPBuffer
{
protected:
	int m_nWidth;
	int m_nHeight;
	int m_nFlags;

	HDC m_hDC;
	HGLRC m_hGLRC;
	HPBUFFERARB m_hBuffer;
	GLuint m_nTextureID;

	bool m_bATI;
	unsigned int m_nTarget;
	CShaderObject m_shExposure;

public:
	enum {NoFlags = 0x00, DepthBuffer = 0x01, StencilBuffer = 0x02};

	CPBuffer()							{ m_hBuffer = NULL; }
	CPBuffer(int nWidth, int nHeight, int nFlags=(DepthBuffer|StencilBuffer))
	{
		m_hBuffer = NULL;
		Init(nWidth, nHeight);
	}
	~CPBuffer()							{ Cleanup(); }

	bool Init(int nWidth, int nHeight, int nFlags=(DepthBuffer|StencilBuffer));
	void Cleanup();
	void HandleModeSwitch();

	int GetWidth()						{ return m_nWidth; }
	int GetHeight()						{ return m_nHeight; }
	int GetFlags()						{ return m_nFlags; }
	HGLRC GetHGLRC()					{ return m_hGLRC; }
	HDC GetHDC()						{ return m_hDC; }

	void MakeCurrent()
	{
		if(m_hBuffer)
			wglMakeCurrent(m_hDC, m_hGLRC);
	}
	void BindTexture(float fExposure, bool bUseExposure=true)
	{
		if(m_hBuffer && m_nTextureID)
		{
			if(bUseExposure)
				m_shExposure.Enable();
			m_shExposure.SetUniformParameter1i("s2Test", 0);
			m_shExposure.SetUniformParameter1f("fExposure", fExposure);
			glBindTexture(m_nTarget, m_nTextureID);
			wglBindTexImageARB(m_hBuffer, WGL_FRONT_LEFT_ARB);
			glEnable(m_nTarget);
		}
	}
	void ReleaseTexture()
	{
		if(m_hBuffer && m_nTextureID)
			wglReleaseTexImageARB(m_hBuffer, WGL_FRONT_LEFT_ARB);
		glDisable(m_nTarget);
		m_shExposure.Disable();
	}
};

#endif // __PBuffer_h__
