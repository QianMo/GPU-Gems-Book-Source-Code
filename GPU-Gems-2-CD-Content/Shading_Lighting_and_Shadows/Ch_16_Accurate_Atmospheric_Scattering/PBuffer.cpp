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

#include "Master.h"
#include "PBuffer.h"


bool CPBuffer::Init(int nWidth, int nHeight, int nFlags)
{
	Cleanup();
	m_nFlags = nFlags;
	m_bATI = GLUtil()->IsATI();
	m_nTarget = m_bATI ? GL_TEXTURE_2D : GL_TEXTURE_RECTANGLE_NV;	// Try 2D for nVidia if height and width are the same
	glGenTextures(1, &m_nTextureID);
	glBindTexture(m_nTarget, m_nTextureID);
	glTexParameteri(m_nTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(m_nTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(m_nTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(m_nTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Set up floating point attributes for pbuffer
	int nFormatAttributes[2*MAX_ATTRIBS];
	int nBufferAttributes[2*MAX_ATTRIBS];
	memset(nFormatAttributes, 0, sizeof(int)*2*MAX_ATTRIBS);
	memset(nBufferAttributes, 0, sizeof(int)*2*MAX_ATTRIBS);

	int i=0, j=0;
	nFormatAttributes[i++] = WGL_SUPPORT_OPENGL_ARB;
	nFormatAttributes[i++] = true;
	nFormatAttributes[i++] = WGL_DRAW_TO_PBUFFER_ARB;
	nFormatAttributes[i++] = true;

	if(m_bATI)
	{
		nFormatAttributes[i++] = WGL_PIXEL_TYPE_ARB;
		nFormatAttributes[i++] = WGL_TYPE_RGBA_FLOAT_ATI;
		nFormatAttributes[i++] = WGL_BIND_TO_TEXTURE_RGBA_ARB;
		nFormatAttributes[i++] = true;
		nBufferAttributes[j++] = WGL_TEXTURE_TARGET_ARB;
		nBufferAttributes[j++] = WGL_TEXTURE_2D_ARB;
		nBufferAttributes[j++] = WGL_TEXTURE_FORMAT_ARB;
		nBufferAttributes[j++] = WGL_TEXTURE_RGBA_ARB;
	}
	else
	{
		nFormatAttributes[i++] = WGL_PIXEL_TYPE_ARB;
		nFormatAttributes[i++] = WGL_TYPE_RGBA_ARB;
		nFormatAttributes[i++] = WGL_FLOAT_COMPONENTS_NV;
		nFormatAttributes[i++] = true;

		nFormatAttributes[i++] = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV;
		nFormatAttributes[i++] = true;
		nBufferAttributes[j++] = WGL_TEXTURE_TARGET_ARB;
		nBufferAttributes[j++] = WGL_TEXTURE_RECTANGLE_NV;	// Try 2D for nVidia if height and width are the same
		nBufferAttributes[j++] = WGL_TEXTURE_FORMAT_ARB;
		nBufferAttributes[j++] = WGL_TEXTURE_FLOAT_RGBA_NV;
	}

	nFormatAttributes[i++] = WGL_RED_BITS_ARB;
	nFormatAttributes[i++] = 16;
	nFormatAttributes[i++] = WGL_GREEN_BITS_ARB;
	nFormatAttributes[i++] = 16;
	nFormatAttributes[i++] = WGL_BLUE_BITS_ARB;
	nFormatAttributes[i++] = 16;
	nFormatAttributes[i++] = WGL_ALPHA_BITS_ARB;
	nFormatAttributes[i++] = 16;

	//if(nFlags & DepthBuffer)
	{
		nFormatAttributes[i++] = WGL_DEPTH_BITS_ARB;
		nFormatAttributes[i++] = 24;
	}
	//if(nFlags & StencilBuffer)
	{
		nFormatAttributes[i++] = WGL_STENCIL_BITS_ARB;
		nFormatAttributes[i++] = 8;
	}


	// Temporarily store the current device and rendering contexts
	HDC hDC = wglGetCurrentDC();
	HGLRC hGLRC = wglGetCurrentContext();

	int nFormat;
	unsigned int nFormats;
	wglChoosePixelFormatARB(hDC, nFormatAttributes, NULL, 1, &nFormat, &nFormats);
	if(nFormats == 0)
	{
		LogError("CPBuffer::Init() - wglChoosePixelFormatARB failed (0x%X)", GetLastError());
		return false;
	}

    if(!(m_hBuffer = wglCreatePbufferARB(hDC, nFormat, nWidth, nHeight, nBufferAttributes)))
	{
		LogError("CPBuffer::Init() - wglCreatePbufferARB failed (0x%X)", GetLastError());
		return false;
	}

	if(!(m_hDC = wglGetPbufferDCARB(m_hBuffer)))
	{
		LogError("CPBuffer::Init() - wglGetPbufferDCARB failed (0x%X)", GetLastError());
		return false;
	}

	if(!(m_hGLRC = wglCreateContext(m_hDC)))
	{
		LogError("CPBuffer::Init() - wglCreateContext failed (0x%X)", GetLastError());
		return false;
	}

	if(!wglShareLists(hGLRC, m_hGLRC))
	{
		LogError("CPBuffer::Init() - wglShareLists failed (0x%X)", GetLastError());
		return false;
	}

	wglQueryPbufferARB(m_hBuffer, WGL_PBUFFER_WIDTH_ARB, &m_nWidth);
	wglQueryPbufferARB(m_hBuffer, WGL_PBUFFER_HEIGHT_ARB, &m_nHeight);
    int nTexFormat = WGL_NO_TEXTURE_ARB;
    wglQueryPbufferARB(m_hBuffer, WGL_TEXTURE_FORMAT_ARB, &nTexFormat);
    if(nTexFormat == WGL_NO_TEXTURE_ARB)
		LogError("CPBuffer::Init() - The pbuffer is not a texture!");

	int iAttributes[] =
	{
		WGL_RED_BITS_ARB,
		WGL_GREEN_BITS_ARB,
		WGL_BLUE_BITS_ARB,
		WGL_ALPHA_BITS_ARB,
		WGL_DEPTH_BITS_ARB,
		WGL_STENCIL_BITS_ARB,
		WGL_SAMPLES_EXT,
		WGL_AUX_BUFFERS_ARB
	};
	int iValues[sizeof(iAttributes) / sizeof(int)];

	if(wglGetPixelFormatAttribivARB(m_hDC, nFormat, 0, sizeof(iAttributes) / sizeof(int), iAttributes, iValues))
		LogInfo("PBuffer::Init() - %dx%d r:%d g:%d b:%d a:%d depth:%d stencil:%d samples:%d aux:%d\n", m_nWidth, m_nHeight, iValues[0], iValues[1], iValues[2], iValues[3], iValues[4], iValues[5], iValues[6], iValues[7]);

	m_shExposure.Load("HDR", m_nTarget == GL_TEXTURE_2D ? "HDRSquare" : "HDRRect");

	return true;
}

void CPBuffer::Cleanup()
{
	if(m_hBuffer)
	{
		glDeleteTextures(1, &m_nTextureID);
		wglDeleteContext(m_hGLRC);
		wglReleasePbufferDCARB(m_hBuffer, m_hDC);
		wglDestroyPbufferARB(m_hBuffer);
		m_hBuffer = NULL;
	}
}

void CPBuffer::HandleModeSwitch()
{
	if(m_hBuffer)
	{
		int nLost = 0;
		wglQueryPbufferARB(m_hBuffer, WGL_PBUFFER_LOST_ARB, &nLost);
		if(nLost)
		{
			Cleanup();
			Init(m_nWidth, m_nHeight, m_nFlags);
		}
	}
}
