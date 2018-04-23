#ifndef MP_PBUFFER_H
#define	MP_PBUFFER_H

#include <iostream>

using namespace std;

int PreparePBuffer(HPBUFFERARB &hPBuffer, HDC &hPBufferDC, HGLRC &hPBufferRC, int width, int height, HDC hFBufferDC)
{
	int error=1;	// no error

	int	iPixelAttributes[100], ind;
	int pixelFormat;
	unsigned int numFormats;
			
	// choose a pixel format
	ind=0;
	int precision=32;
	iPixelAttributes[ind++]=WGL_SUPPORT_OPENGL_ARB;
	iPixelAttributes[ind++]=TRUE;
	iPixelAttributes[ind++]=WGL_DRAW_TO_PBUFFER_ARB;
	iPixelAttributes[ind++]=TRUE;
	iPixelAttributes[ind++]=WGL_RED_BITS_ARB;
	iPixelAttributes[ind++]=precision;
	iPixelAttributes[ind++]=WGL_GREEN_BITS_ARB;
	iPixelAttributes[ind++]=0;
	iPixelAttributes[ind++]=WGL_BLUE_BITS_ARB;
	iPixelAttributes[ind++]=0;
	iPixelAttributes[ind++]=WGL_DEPTH_BITS_ARB;
	iPixelAttributes[ind++]=0;
	iPixelAttributes[ind++]=WGL_DOUBLE_BUFFER_ARB;
	iPixelAttributes[ind++]=TRUE;
	iPixelAttributes[ind++]=WGL_FLOAT_COMPONENTS_NV;
	iPixelAttributes[ind++]=TRUE;
	iPixelAttributes[ind++]=WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV;
	iPixelAttributes[ind++]=TRUE;
	iPixelAttributes[ind++]=0;
	if(!wglChoosePixelFormatARB(hFBufferDC, (const int*)iPixelAttributes, NULL, 1, &pixelFormat, &numFormats))
	{
        cerr<<"didn't find a pixel format"<<endl;
		return 0;
	}

	// create the pbuffer
	int iPBufferAttributes[20];
	ind=0;
	iPBufferAttributes[ind++]=WGL_PBUFFER_LARGEST_ARB;
	iPBufferAttributes[ind++]=FALSE;
	iPBufferAttributes[ind++]=WGL_TEXTURE_FORMAT_ARB;
	iPBufferAttributes[ind++]=WGL_TEXTURE_FLOAT_R_NV;
	iPBufferAttributes[ind++]=WGL_TEXTURE_TARGET_ARB;
	iPBufferAttributes[ind++]=WGL_TEXTURE_RECTANGLE_NV;
	iPBufferAttributes[ind++]=0;
	hPBuffer=wglCreatePbufferARB(hFBufferDC, pixelFormat, width, height, iPBufferAttributes);

	// get a device context for the pbuffer
	hPBufferDC=wglGetPbufferDCARB(hPBuffer);

	// create a rendering context for the pbuffer
	hPBufferRC=wglCreateContext(hPBufferDC);

	return error;
}

#endif