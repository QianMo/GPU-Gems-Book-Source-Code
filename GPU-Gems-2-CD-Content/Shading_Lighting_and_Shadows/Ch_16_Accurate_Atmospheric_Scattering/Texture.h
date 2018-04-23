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

#ifndef __Texture_h__
#define __Texture_h__

#include "PixelBuffer.h"
#include "GLUtil.h"

/*******************************************************************************
* Class: CTexture
********************************************************************************
* This class encapsulates OpenGL texture objects. You initialize it with a
* CPixelBuffer instance and a flag indicating whether you want mipmaps to be
* generated.
*******************************************************************************/
class CTexture
{
protected:
	int m_nType;					// GL_TEXTURE_1D, GL_TEXTURE_2D, or GL_TEXTURE_3D
	unsigned int m_nID;				// OpenGL-generated texture ID

	static CTexture m_tCloudCell;		// Shared cloud cell texture
	static CTexture m_t1DGlow;

public:

	CTexture()		{ m_nID = -1; }
	CTexture(CPixelBuffer *pBuffer, bool bMipmap=true)
	{
		m_nID = -1;
		Init(pBuffer, bMipmap);
	}
	~CTexture()		{ Cleanup(); }
	void Cleanup()
	{
		if(m_nID != -1)
		{
			glDeleteTextures(1, &m_nID);
			m_nID = -1;
		}
	}

	static void InitStaticMembers(int nSeed, int nSize);
	static CTexture &GetCloudCell()			{ return m_tCloudCell; }
	static CTexture &Get1DGlow()			{ return m_t1DGlow; }
	static CTexture &Get3DNoise()			{ return m_t1DGlow; }
	static void Enable(int nType)			{ glEnable(nType); }
	static void Disable(int nType)			{ glDisable(nType); }
	
	DWORD GetID()						{ return m_nID; }
	int GetType()						{ return m_nType; }
	void Bind()							{ if(m_nID != -1) glBindTexture(m_nType, m_nID); }
	void Enable()						{ if(m_nID != -1) { Bind(); glEnable(m_nType); } }
	void Disable()						{ if(m_nID != -1) glDisable(m_nType); }

	void Init(CPixelBuffer *pBuffer, bool bClamp=true, bool bMipmap=true);
	void Update(CPixelBuffer *pBuffer, int nLevel=0);

	// Use when rendering to texture (either in the back buffer or a CPBuffer)
	void InitCopy(int x, int y, int nWidth, int nHeight, bool bClamp=true);
	void UpdateCopy(int x, int y, int nWidth, int nHeight, int nOffx=0, int nOffy=0, int nOffz=0);
};


class CTextureArray : public CTexture
{
protected:
	int m_nTextureSize;
	int m_nPartitionSize;
	int m_nChannels;
	int m_nFormat;
	int m_nDataType;
	int m_nArrayWidth;
	int m_nStackSize;
	int m_nStackIndex;
	int *m_pStack;

public:
	CTextureArray()
	{
		m_pStack = NULL;
	}
	~CTextureArray()
	{
		Cleanup();
	}

	void Init(int nTextureSize, int nPartitionSize, int nChannels, int nFormat, int nDataType)
	{
		Cleanup();

		m_nTextureSize = nTextureSize;
		m_nPartitionSize = nPartitionSize;
		m_nChannels = nChannels;
		m_nFormat = nFormat;
		m_nDataType = nDataType;
		m_nArrayWidth = m_nTextureSize / m_nPartitionSize;

		m_nStackIndex = 0;
		m_nStackSize = m_nArrayWidth * m_nArrayWidth;
		m_pStack = new int[m_nStackSize];
		for(int n = 0; n < m_nStackSize; n++)
			m_pStack[n] = n;

		CPixelBuffer pb(nTextureSize, nTextureSize, nChannels, nFormat, nDataType);
		memset(pb.GetBuffer(), 0xFF, pb.GetBufferSize());
		CTexture::Init(&pb, true, false);
	}

	void Cleanup()
	{
		if(m_pStack)
		{
			delete m_pStack;
			m_pStack = NULL;
		}
		CTexture::Cleanup();
	}

	int LockTexture()
	{
		_ASSERT(m_nStackIndex < m_nStackSize);
		if(m_nStackIndex >= m_nStackSize)
			return m_nStackSize;
		return m_pStack[m_nStackIndex++];
	}
	void ReleaseTexture(int nTexture)
	{
		_ASSERT(m_nStackIndex > 0 && nTexture >= 0 && nTexture < m_nStackSize);
		if(m_nStackIndex <= 0 || nTexture < 0 || nTexture >= m_nStackSize)
			return;
		m_pStack[--m_nStackIndex] = nTexture;
	}

	void Update(int nTexture, CPixelBuffer *pBuffer)
	{
		_ASSERT(nTexture >= 0 && nTexture < m_nStackSize);
		if(nTexture < 0 || nTexture >= m_nStackSize)
			return;

		Bind();
		int x = nTexture % m_nArrayWidth;
		int y = nTexture / m_nArrayWidth;
		_ASSERT(pBuffer->GetWidth() == m_nPartitionSize);
		_ASSERT(pBuffer->GetHeight() == m_nPartitionSize);
		_ASSERT(pBuffer->GetFormat() == m_nFormat);
		_ASSERT(pBuffer->GetDataType() == m_nDataType);
		glTexSubImage2D(GL_TEXTURE_2D, 0, x*m_nPartitionSize, y*m_nPartitionSize, pBuffer->GetWidth(), pBuffer->GetHeight(), pBuffer->GetFormat(), pBuffer->GetDataType(), pBuffer->GetBuffer());
	}

	void MapCorners(int nTexture, float fXMin, float fYMin, float fXMax, float fYMax)
	{
		_ASSERT(nTexture >= 0 && nTexture < m_nStackSize);
		if(nTexture < 0 || nTexture >= m_nStackSize)
			return;

		// Sets the corners of the texture coordinates (0-1 in x and y dimensions) to the centers
		// of the texels in the array element being accessed
		float fXScale = (float)(m_nPartitionSize-1) / ((float)m_nTextureSize * (fXMax-fXMin));
		float fYScale = (float)(m_nPartitionSize-1) / ((float)m_nTextureSize * (fYMax-fYMin));
		float fXOffset = ((nTexture % m_nArrayWidth) * m_nPartitionSize + 0.5f) / (float)m_nTextureSize;
		float fYOffset = ((nTexture / m_nArrayWidth) * m_nPartitionSize + 0.5f) / (float)m_nTextureSize;
		CMatrix m;
		m.TranslateMatrix(fXOffset, fYOffset, 0.0f);
		m.Scale(fXScale, fYScale, 1.0f);
		m.Translate(-fXMin, -fYMin, 0.0f);
		glLoadMatrixf(m);
	}
};

#define USE_TEXTURE_ARRAY

#endif // __Texture_h__
