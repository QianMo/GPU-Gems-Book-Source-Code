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
#include "PixelBuffer.h"

// LibJPEG headers
extern "C" {
	#include "jpeglib.h"
};


bool CPixelBuffer::LoadJPEG(const char *pszFile)
{
	// Open the file we're reading from
	FILE *pFile = fopen(pszFile, "rb");
	if(pFile == NULL)
		return false;

	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, pFile);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

	Init(cinfo.image_width, cinfo.image_height, 1);
	int nRowSize = cinfo.output_width * cinfo.output_components;
	JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, nRowSize, 1);
	for(int i=0; cinfo.output_scanline < cinfo.output_height; i++)
	{
		(void) jpeg_read_scanlines(&cinfo, buffer, 1);
		memcpy((char *)m_pBuffer + i*nRowSize, *buffer, nRowSize);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	fclose(pFile);
	return true;
}

bool CPixelBuffer::SaveJPEG(const char *pszFile, int nQuality)
{
	if(m_nChannels != 3 || m_nFormat != GL_RGB)
		return false;

	// Open the file we're saving to
	FILE *pFile = fopen(pszFile, "wb");
	if(pFile == NULL)
		return false;

	// Read the pixels and build an array of pointers to the rows
	// The array must be backwards becuase OpenGL goes bottom to top and JPEG goes top to bottom
	unsigned char *pszImage = (unsigned char *)m_pBuffer;
	JSAMPROW *row = new JSAMPROW[m_nHeight];
	for(int i=0; i<m_nHeight; i++)
		row[i] = (JSAMPROW)(pszImage+i*m_nWidth*3);

	// Initialize the JPEG struct
	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, pFile);
	cinfo.image_width = m_nWidth;
	cinfo.image_height = m_nHeight;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, nQuality, TRUE);

	// Flush the image to the JPEG file
	jpeg_start_compress(&cinfo, TRUE);
	jpeg_write_scanlines(&cinfo, row, m_nHeight);
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	// Clean up
	fclose(pFile);
	delete row;
	return true;
}

void CPixelBuffer::MakeCloudCell(float fExpose, float fSizeDisc)
{
	int i;
	int n = 0;
	unsigned char nIntensity;
	for(int y=0; y<m_nHeight; y++)
	{
		float fDy = (y+0.5f)/m_nHeight - 0.5f;
		for(int x=0; x<m_nWidth; x++)
		{
			float fDx = (x+0.5f)/m_nWidth - 0.5f;
			float fDist = sqrtf(fDx*fDx + fDy*fDy);
			float fIntensity = 2.0f - Min(2.0f, powf(2.0f, Max(fDist-fSizeDisc,0.0f)*fExpose));
			switch(m_nDataType)
			{
				case GL_UNSIGNED_BYTE:
					nIntensity = (unsigned char)(fIntensity*255 + 0.5f);
					for(i=0; i<m_nChannels; i++)
						((unsigned char *)m_pBuffer)[n++] = nIntensity;
					break;
				case GL_FLOAT:
					for(i=0; i<m_nChannels; i++)
						((float *)m_pBuffer)[n++] = fIntensity;
					break;
			}
		}
	}
}

void CPixelBuffer::Make3DNoise(int nSeed)
{
	CFractal noise(3, nSeed, 0.5f, 2.0f);
	int n = 0;
	float fValues[3];
	for(int z=0; z<m_nDepth; z++)
	{
		fValues[2] = (float)z * 0.0625f;
		for(int y=0; y<m_nHeight; y++)
		{
			fValues[1] = (float)y * 0.0625f;
			for(int x=0; x<m_nWidth; x++)
			{
				fValues[0] = (float)x * 0.0625f;
				float fIntensity = Abs(noise.fBm(fValues, 4.0f)) - 0.5f;
				if(fIntensity < 0.0)
					fIntensity = 0.0f;
				fIntensity = 1.0f - powf(0.9f, fIntensity*255);
				unsigned char nIntensity = (unsigned char)(fIntensity*255 + 0.5f);
				((unsigned char *)m_pBuffer)[n++] = 255;
				((unsigned char *)m_pBuffer)[n++] = nIntensity;
			}
		}
	}
}

void CPixelBuffer::MakeGlow1D()
{
	int nIndex=0;
	for(int x=0; x<m_nWidth; x++)
	{
		float fIntensity = powf((float)x / m_nWidth, 0.75f);
		for(int i=0; i<m_nChannels-1; i++)
			((unsigned char *)m_pBuffer)[nIndex++] = (unsigned char)255;
		((unsigned char *)m_pBuffer)[nIndex++] = (unsigned char)(fIntensity*255 + 0.5f);
	}
}

void CPixelBuffer::MakeGlow2D(float fExposure, float fRadius)
{
	int nIndex=0;
	for(int y=0; y<m_nHeight; y++)
	{
		for(int x=0; x<m_nWidth; x++)
		{
			float fX = ((m_nWidth-1)*0.5f - x) / (float)(m_nWidth-1);
			float fY = ((m_nHeight-1)*0.5f - y) / (float)(m_nHeight-1);
			float fDist = Max(0.0f, sqrtf(fX*fX + fY*fY) - fRadius);

			float fIntensity = exp(-fExposure * fDist);
			unsigned char c = (unsigned char)(fIntensity*192 + 0.5f);
			for(int i=0; i<m_nChannels; i++)
				((unsigned char *)m_pBuffer)[nIndex++] = c;
		}
	}
}

void CPixelBuffer::MakeOpticalDepthBuffer(float fInnerRadius, float fOuterRadius, float fRayleighScaleHeight, float fMieScaleHeight)
{
	const int nSize = 64;
	const int nSamples = 50;
	const float fScale = 1.0f / (fOuterRadius - fInnerRadius);
	//std::ofstream ofScale1("scale1.txt");
	//std::ofstream ofScale2("scale2.txt");

	Init(nSize, nSize, 1, 4, GL_RGBA, GL_FLOAT);
	int nIndex = 0;
	float fPrev = 0;
	for(int nAngle=0; nAngle<nSize; nAngle++)
	{
		// As the y tex coord goes from 0 to 1, the angle goes from 0 to 180 degrees
		float fCos = 1.0f - (nAngle+nAngle) / (float)nSize;
		float fAngle = acosf(fCos);
		CVector vRay(sinf(fAngle), cosf(fAngle), 0);	// Ray pointing to the viewpoint

		/*char szName[256];
		sprintf(szName, "graph%-2.2d.txt", nAngle);
		std::ofstream ofGraph;
		if(fCos >= 0.0f)
			ofGraph.open(szName);
		ofGraph << "# fCos = " << fCos << std::endl;*/

		float fFirst = 0;
		for(int nHeight=0; nHeight<nSize; nHeight++)
		{
			// As the x tex coord goes from 0 to 1, the height goes from the bottom of the atmosphere to the top
			float fHeight = DELTA + fInnerRadius + ((fOuterRadius - fInnerRadius) * nHeight) / nSize;
			CVector vPos(0, fHeight, 0);				// The position of the camera

			// If the ray from vPos heading in the vRay direction intersects the inner radius (i.e. the planet), then this spot is not visible from the viewpoint
			float B = 2.0f * (vPos | vRay);
			float Bsq = B * B;
			float Cpart = (vPos | vPos);
			float C = Cpart - fInnerRadius*fInnerRadius;
			float fDet = Bsq - 4.0f * C;
			bool bVisible = (fDet < 0 || (0.5f * (-B - sqrtf(fDet)) <= 0) && (0.5f * (-B + sqrtf(fDet)) <= 0));
			float fRayleighDensityRatio;
			float fMieDensityRatio;
			if(bVisible)
			{
				fRayleighDensityRatio = expf(-(fHeight - fInnerRadius) * fScale / fRayleighScaleHeight);
				fMieDensityRatio = expf(-(fHeight - fInnerRadius) * fScale / fMieScaleHeight);
			}
			else
			{
				// Smooth the transition from light to shadow (it is a soft shadow after all)
				fRayleighDensityRatio = ((float *)m_pBuffer)[nIndex - nSize*m_nChannels] * 0.5f;
				fMieDensityRatio = ((float *)m_pBuffer)[nIndex+2 - nSize*m_nChannels] * 0.5f;
			}

			// Determine where the ray intersects the outer radius (the top of the atmosphere)
			// This is the end of our ray for determining the optical depth (vPos is the start)
			C = Cpart - fOuterRadius*fOuterRadius;
			fDet = Bsq - 4.0f * C;
			float fFar = 0.5f * (-B + sqrtf(fDet));

			// Next determine the length of each sample, scale the sample ray, and make sure position checks are at the center of a sample ray
			float fSampleLength = fFar / nSamples;
			float fScaledLength = fSampleLength * fScale;
			CVector vSampleRay = vRay * fSampleLength;
			vPos += vSampleRay * 0.5f;

			// Iterate through the samples to sum up the optical depth for the distance the ray travels through the atmosphere
			float fRayleighDepth = 0;
			float fMieDepth = 0;
			for(int i=0; i<nSamples; i++)
			{
				float fHeight = vPos.Magnitude();
				float fAltitude = (fHeight - fInnerRadius) * fScale;
				//fAltitude = Max(fAltitude, 0.0f);
				fRayleighDepth += expf(-fAltitude / fRayleighScaleHeight);
				fMieDepth += expf(-fAltitude / fMieScaleHeight);
				vPos += vSampleRay;
			}

			// Multiply the sums by the length the ray traveled
			fRayleighDepth *= fScaledLength;
			fMieDepth *= fScaledLength;

			if(!_finite(fRayleighDepth) || fRayleighDepth > 1.0e25f)
				fRayleighDepth = 0;
			if(!_finite(fMieDepth) || fMieDepth > 1.0e25f)
				fMieDepth = 0;

			// Store the results for Rayleigh to the light source, Rayleigh to the camera, Mie to the light source, and Mie to the camera
			((float *)m_pBuffer)[nIndex++] = fRayleighDensityRatio;
			((float *)m_pBuffer)[nIndex++] = fRayleighDepth;
			((float *)m_pBuffer)[nIndex++] = fMieDensityRatio;
			((float *)m_pBuffer)[nIndex++] = fMieDepth;

			/*
			if(nHeight == 0)
			{
				fFirst = fRayleighDepth;
				if(fCos >= 0.0f)
				{
					ofScale1 << 1-fCos << "\t" << logf(fRayleighDepth / fRayleighScaleHeight) << std::endl;
					ofScale2 << 1-fCos << "\t" << logf(fMieDepth) << std::endl;
					fPrev = fRayleighDepth;
				}
			}
			float x = (fHeight-fInnerRadius) / (fOuterRadius-fInnerRadius);
			float y = fRayleighDepth / fFirst;
			ofGraph << x << "\t" << y << std::endl;
			*/
		}
		//ofGraph << std::endl;
	}
}

void CPixelBuffer::MakePhaseBuffer(float ESun, float Kr, float Km, float g)
{
	Km *= ESun;
	Kr *= ESun;
	float g2 = g*g;
	float fMiePart = 1.5f * (1.0f - g2) / (2.0f + g2);

	int nIndex = 0;
	for(int nAngle=0; nAngle<m_nWidth; nAngle++)
	{
		float fCos = 1.0f - (nAngle+nAngle) / (float)m_nWidth;
		float fCos2 = fCos*fCos;
		float fRayleighPhase = 0.75f * (1.0f + fCos2);
		float fMiePhase = fMiePart * (1.0f + fCos2) / powf(1.0f + g2 - 2.0f*g*fCos, 1.5f);
		((float *)m_pBuffer)[nIndex++] = fRayleighPhase * Kr;
		((float *)m_pBuffer)[nIndex++] = fMiePhase * Km;
	}
}

