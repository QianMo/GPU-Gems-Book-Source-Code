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

#ifndef __PixelBuffer_h__
#define __PixelBuffer_h__

#include "Matrix.h"

#define ALIGN_SIZE		64
#define ALIGN_MASK		(ALIGN_SIZE-1)
#define ALIGN(x)		(((unsigned int)x+ALIGN_MASK) & ~ALIGN_MASK)

typedef enum
{
	UnsignedByteType = GL_UNSIGNED_BYTE,
	SignedByteType = GL_BYTE,
	UnsignedShortType = GL_UNSIGNED_SHORT,
	SignedShortType = GL_SHORT,
	UnsignedIntType = GL_UNSIGNED_INT,
	SignedIntType = GL_INT,
	FloatType = GL_FLOAT,
	DoubleType = GL_DOUBLE
} BufferDataType;

inline const int GetDataTypeSize(const int nDataType)
{
	int nSize;
	switch(nDataType)
	{
		case UnsignedByteType:
		case SignedByteType:
			nSize = 1;
			break;
		case UnsignedShortType:
		case SignedShortType:
			nSize = 2;
			break;
		case UnsignedIntType:
		case SignedIntType:
		case FloatType:
			nSize = 4;
			break;
		case DoubleType:
			nSize = 8;
			break;
		default:
			nSize = 0;
			break;
	}
	return nSize;
}


class C3DBuffer
{
protected:
	int m_nWidth;				// The width of the buffer (x axis)
	int m_nHeight;				// The height of the buffer (y axis)
	int m_nDepth;				// The depth of the buffer (z axis)
	int m_nDataType;			// The data type stored in the buffer (i.e. GL_UNSIGNED_BYTE, GL_FLOAT)
	int m_nChannels;			// The number of channels of data stored in the buffer
	int m_nElementSize;			// The size of one element in the buffer
	void *m_pAlloc;				// The pointer to the pixel buffer
	void *m_pBuffer;			// A byte-aligned pointer (for faster memory access)

public:
	C3DBuffer()						{ m_pAlloc = m_pBuffer = NULL; }
	C3DBuffer(const C3DBuffer &buf)	{ *this = buf; }
	C3DBuffer(const int nWidth, const int nHeight, const int nDepth, const int nDataType, const int nChannels=1, void *pBuffer=NULL)
	{
		m_pAlloc = m_pBuffer = NULL;
		Init(nWidth, nHeight, nDepth, nDataType, nChannels, pBuffer);
	}
	~C3DBuffer()					{ Cleanup(); }

	void operator=(const C3DBuffer &buf)
	{
		Init(buf.m_nWidth, buf.m_nHeight, buf.m_nDepth, buf.m_nDataType, buf.m_nChannels);
		memcpy(m_pBuffer, buf.m_pBuffer, GetBufferSize());
	}
	bool operator==(const C3DBuffer &buf)
	{
		return (m_nWidth == buf.m_nWidth && m_nHeight == buf.m_nHeight && m_nDepth == buf.m_nDepth && m_nDataType == buf.m_nDataType && m_nChannels == buf.m_nChannels);
	}

	void *operator[](const int n)
	{
		return (void *)((unsigned int)m_pBuffer + n * m_nElementSize);
	}
	void *operator()(const int x, const int y, const int z)
	{
		return (void *)((unsigned int)m_pBuffer + m_nElementSize * (m_nWidth * (m_nHeight * z + y) + x));
	}

	void *operator()(const float x)
	{
		int nX = Min(m_nWidth-1, Max(0, (int)(x*(m_nWidth-1)+0.5f)));
		return (void *)((unsigned long)m_pBuffer + m_nElementSize * nX);
	}
	void *operator()(const float x, const float y)
	{
		int nX = Min(m_nWidth-1, Max(0, (int)(x*(m_nWidth-1)+0.5f)));
		int nY = Min(m_nHeight-1, Max(0, (int)(y*(m_nHeight-1)+0.5f)));
		return (void *)((unsigned long)m_pBuffer + m_nElementSize * (m_nWidth * nY + nX));
	}
	void *operator()(const float x, const float y, const float z)
	{
		int nX = Min(m_nWidth-1, Max(0, (int)(x*(m_nWidth-1)+0.5f)));
		int nY = Min(m_nHeight-1, Max(0, (int)(y*(m_nHeight-1)+0.5f)));
		int nZ = Min(m_nDepth-1, Max(0, (int)(z*(m_nDepth-1)+0.5f)));
		return (void *)((unsigned long)m_pBuffer + m_nElementSize * (m_nWidth * (m_nHeight * nZ + nY) + nX));
	}

	void Interpolate(float *p, const float x)
	{
		float fX = x*(m_nWidth-1);
		int nX = Min(m_nWidth-2, Max(0, (int)fX));
		float fRatioX = fX - nX;
		float *pValue = (float *)((unsigned long)m_pBuffer + m_nElementSize * nX);
		for(int i=0; i<m_nChannels; i++)
		{
			p[i] =	pValue[0] * (1-fRatioX) + pValue[m_nChannels] * (fRatioX);
			pValue++;
		}
	}
	void Interpolate(float *p, const float x, const float y)
	{
		float fX = x*(m_nWidth-1);
		float fY = y*(m_nHeight-1);
		int nX = Min(m_nWidth-2, Max(0, (int)fX));
		int nY = Min(m_nHeight-2, Max(0, (int)fY));
		float fRatioX = fX - nX;
		float fRatioY = fY - nY;
		float *pValue = (float *)((unsigned long)m_pBuffer + m_nElementSize * (m_nWidth * nY + nX));
		for(int i=0; i<m_nChannels; i++)
		{
			p[i] =	pValue[0] * (1-fRatioX) * (1-fRatioY) +
					pValue[m_nChannels*1] * (fRatioX) * (1-fRatioY) +
					pValue[m_nChannels*m_nWidth] * (1-fRatioX) * (fRatioY) +
					pValue[m_nChannels*(m_nWidth+1)] * (fRatioX) * (fRatioY);
			pValue++;
		}
	}
	void Interpolate(float *p, const float x, const float y, const float z)
	{
		float fX = x*(m_nWidth-1);
		float fY = y*(m_nHeight-1);
		float fZ = z*(m_nDepth-1);
		int nX = Min(m_nWidth-2, Max(0, (int)fX));
		int nY = Min(m_nHeight-2, Max(0, (int)fY));
		int nZ = Min(m_nDepth-2, Max(0, (int)fZ));
		float fRatioX = fX - nX;
		float fRatioY = fY - nY;
		float fRatioZ = fZ - nZ;
		float *pValue = (float *)((unsigned long)m_pBuffer + m_nElementSize * (m_nWidth * (m_nHeight * nZ + nY) + nX));
		float *pValue2 = (float *)((unsigned long)m_pBuffer + m_nElementSize * (m_nWidth * (m_nHeight * (nZ+1) + nY) + nX));
		for(int i=0; i<m_nChannels; i++)
		{
			p[i] =	pValue[0] * (1-fRatioX) * (1-fRatioY) * (1-fRatioZ) +
					pValue[m_nChannels*1] * (fRatioX) * (1-fRatioY) * (1-fRatioZ) +
					pValue[m_nChannels*m_nWidth] * (1-fRatioX) * (fRatioY) * (1-fRatioZ) +
					pValue[m_nChannels*(m_nWidth+1)] * (fRatioX) * (fRatioY) * (1-fRatioZ) +
					pValue2[0] * (1-fRatioX) * (1-fRatioY) * (fRatioZ) +
					pValue2[m_nChannels*1] * (fRatioX) * (1-fRatioY) * (fRatioZ) +
					pValue2[m_nChannels*m_nWidth] * (1-fRatioX) * (fRatioY) * (fRatioZ) +
					pValue2[m_nChannels*(m_nWidth+1)] * (fRatioX) * (fRatioY) * (fRatioZ);
			pValue++;
			pValue2++;
		}
	}

	void Init(const int nWidth, const int nHeight, const int nDepth, const int nDataType, const int nChannels=1, void *pBuffer=NULL)
	{
		// If the buffer is already initialized to the specified settings, then nothing needs to be done
		if(m_pAlloc && m_nWidth == nWidth && m_nHeight == nHeight && m_nDataType == nDataType && m_nChannels == nChannels)
			return;

		Cleanup();
		m_nWidth = nWidth;
		m_nHeight = nHeight;
		m_nDepth = nDepth;
		m_nDataType = nDataType;
		m_nChannels = nChannels;
		m_nElementSize = m_nChannels * GetDataTypeSize(m_nDataType);
		if(pBuffer)
			m_pBuffer = pBuffer;
		else
		{
			m_pAlloc = new unsigned char[GetBufferSize() + ALIGN_MASK];
			m_pBuffer = (void *)ALIGN(m_pAlloc);
		}
	}

	void Cleanup()
	{
		if(m_pAlloc)
		{
			delete m_pAlloc;
			m_pAlloc = m_pBuffer = NULL;
		}
	}

	int GetWidth() const 		{ return m_nWidth; }
	int GetHeight() const		{ return m_nHeight; }
	int GetDepth() const		{ return m_nDepth; }
	int GetDataType() const		{ return m_nDataType; }
	int GetChannels() const		{ return m_nChannels; }
	int GetBufferSize() const	{ return m_nWidth * m_nHeight * m_nDepth * m_nElementSize; }
	void *GetBuffer() const		{ return m_pBuffer; }

	void ClearBuffer()			{ memset(m_pBuffer, 0, GetBufferSize()); }
	void SwapBuffers(C3DBuffer &buf)
	{
		void *pTemp;
		ASSERT(*this == buf);
		SWAP(m_pAlloc, buf.m_pAlloc, pTemp);
		SWAP(m_pBuffer, buf.m_pBuffer, pTemp);
	}

	float LinearSample2D(int nChannel, float x, float y)
	{
		x = Min(Max(x, 0.0001f), 0.9999f);
		y = Min(Max(y, 0.0001f), 0.9999f);
		x *= m_nWidth;
		y *= m_nHeight;
		int n[2] = {(int)x, (int)y};
		float fRatio[2] = {x - n[0], y - n[1]};
		float *pBase = (float *)((unsigned int)m_pBuffer + (m_nWidth * n[1] + n[0]) * m_nChannels * sizeof(float));
		//if(n[0] == m_nWidth-1 || n[1] == m_nHeight-1)
			return pBase[nChannel];
		float *p[4] = {
			&pBase[0],
			&pBase[m_nChannels],
			&pBase[m_nWidth*m_nChannels],
			&pBase[(m_nWidth+1)*m_nChannels]
		};
		return p[0][nChannel] * (1-fRatio[0]) * (1-fRatio[1]) +
			p[1][nChannel] * (fRatio[0]) * (1-fRatio[1]) +
			p[2][nChannel] * (1-fRatio[0]) * (fRatio[1]) +
			p[3][nChannel] * (fRatio[0]) * (fRatio[1]);
	}
};

/*******************************************************************************
* Class: CPixelBuffer
********************************************************************************
* This class implements a general-purpose pixel buffer to be used for anything.
* It is often used by CTexture to set up OpenGL textures, so many of the
* parameters you use to initialize it look like the parameters you would pass
* to glTexImage1D or glTexImage2D. Some of the standard pixel buffer routines
* call fast MMX functions implemented in PixelBuffer.asm.
*******************************************************************************/
class CPixelBuffer : public C3DBuffer
{
protected:
	int m_nFormat;				// The format of the pixel data (i.e. GL_LUMINANCE, GL_RGBA)

public:
	CPixelBuffer() : C3DBuffer() {}
	CPixelBuffer(int nWidth, int nHeight, int nDepth, int nChannels=3, int nFormat=GL_RGB, int nDataType=UnsignedByteType) : C3DBuffer(nWidth, nHeight, nDepth, nDataType, nChannels)
	{
		m_nFormat = nFormat;
	}

	int GetFormat()				{ return m_nFormat; }

	void Init(int nWidth, int nHeight, int nDepth, int nChannels=3, int nFormat=GL_RGB, int nDataType=GL_UNSIGNED_BYTE, void *pBuffer=NULL)
	{
		C3DBuffer::Init(nWidth, nHeight, nDepth, nDataType, nChannels, pBuffer);
		m_nFormat = nFormat;
	}

	// Miscellaneous initalization routines
	bool LoadJPEG(const char *pszFile);
	bool SaveJPEG(const char *pszFile, int nQuality);
	void MakeCloudCell(float fExpose, float fSizeDisc);
	void Make3DNoise(int nSeed);
	void MakeGlow1D();
	void MakeGlow2D(float fExposure, float fRadius);
	void MakeOpticalDepthBuffer(float fInnerRadius, float fOuterRadius, float fRayleighScaleHeight, float fMieScaleHeight);
	void MakePhaseBuffer(float ESun, float Kr, float Km, float g);
};

#endif // __PixelBuffer_h__
