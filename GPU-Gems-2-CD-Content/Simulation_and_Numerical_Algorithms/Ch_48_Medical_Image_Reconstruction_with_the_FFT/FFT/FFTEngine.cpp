/***************************************************************************
*        FILE NAME:  FFTEngine.cpp
*
* ONE LINE SUMMARY:
*        This file contains some of the member function for the FFT class,
*        containing OpenGL commands.
*        
*        Thilaka Sumanaweera
*        Siemens Medical Solutions USA, Inc.
*        1230 Shorebird Way
*        Mountain View, CA 94039
*        USA
*        Thilaka.Sumanaweera@siemens.com
*
* DESCRIPTION:
*
*****************************************************************************
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
****************************************************************************/
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "FFT.h"

using namespace std;

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::UploadData
*
* DESCRIPTION:
*	Uploads the floating point data into the VRAM.
*
* FORMAL PARAMETERS:
*	(imageR1, imageI1) and (imageR2, imageI2) are two complex 2D input images.
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::UploadData(float *imageR1, float *imageI1, float *imageR2, float *imageI2)
{
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texReal1);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, imageR1);

	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texImag1);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, imageI1);
	
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texReal2);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, imageR2);

	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texImag2);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, imageI2);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitTextures
*
* DESCRIPTION:
*	Initializes all the textures needed.
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::InitTextures(void)
{
	glClearColor (0.0, 0.0, 0.0, 0.0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glGenTextures(1, &texTmpR1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texTmpR1);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);
	
    glGenTextures(1, &texTmpI1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texTmpI1);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);
	
    glGenTextures(1, &texTmpR2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texTmpR2);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);
	
    glGenTextures(1, &texTmpI2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texTmpI2);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);

	glGenTextures(1, &texReal1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texReal1);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);

	glGenTextures(1, &texImag1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texImag1);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);
	
	glGenTextures(1, &texReal2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texReal2);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);

	glGenTextures(1, &texImag2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, texImag2);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, Height, 0, GL_RED, GL_FLOAT, NULL);

	int i;
	glGenTextures(nButterfliesX, texButterflyLookupI_X);
	glGenTextures(nButterfliesX, texButterflyLookupWR_X);
	glGenTextures(nButterfliesX, texButterflyLookupWI_X);
	GLuint *Iptr0, *Iptr2, *Iptr3;
	GLfloat *Fptr0, *Fptr2, *Fptr3;
	Iptr0 = texButterflyLookupI_X;
	Iptr2 = texButterflyLookupWR_X;
	Iptr3 = texButterflyLookupWI_X;

	Fptr0 = butterflyLookupI_X;
	Fptr2 = butterflyLookupWR_X;
	Fptr3 = butterflyLookupWI_X;
	for (i = 0; i < nButterfliesX; i++) {
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, *Iptr0);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_LUMINANCE_ALPHA_FLOAT16_ATI, Width, 1, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, Fptr0);
		Iptr0++;
		Fptr0 += Width*2;

		glBindTexture(GL_TEXTURE_RECTANGLE_NV, *Iptr2);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, 1, 0, GL_RED, GL_FLOAT, Fptr2);
		Iptr2++;
		Fptr2 += Width;
		
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, *Iptr3);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, Width, 1, 0, GL_RED, GL_FLOAT, Fptr3);
		Iptr3++;
		Fptr3 += Width;
	}
	
	glGenTextures(nButterfliesY, texButterflyLookupI_Y);
	glGenTextures(nButterfliesY, texButterflyLookupWR_Y);
	glGenTextures(nButterfliesY, texButterflyLookupWI_Y);

	Iptr0 = texButterflyLookupI_Y;
	Iptr2 = texButterflyLookupWR_Y;
	Iptr3 = texButterflyLookupWI_Y;

	Fptr0 = butterflyLookupI_Y;
	Fptr2 = butterflyLookupWR_Y;
	Fptr3 = butterflyLookupWI_Y;

	for (i = 0; i < nButterfliesY; i++) {
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, *Iptr0);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_LUMINANCE_ALPHA_FLOAT16_ATI, 1, Height, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, Fptr0);
		Iptr0++;
		Fptr0 += Height*2;
		
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, *Iptr2);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, 1, Height, 0, GL_RED, GL_FLOAT, Fptr2);
		Iptr2++;
		Fptr2 += Height;
		
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, *Iptr3);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_R32_NV, 1, Height, 0, GL_RED, GL_FLOAT, Fptr3);
		Iptr3++;
		Fptr3 += Height;
	}
	GenQuadDisplayList();

	GenDisplayListsX();
	GenDisplayListsY();

}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::CopyFloatBuffersToScreen
*
* DESCRIPTION:
*	Displays *Texs on the screen.
*
* FORMAL PARAMETERS:
*	FromInput_p:			true if you want to display the input textures
*							false if you want to display what is in the draw buffers
*	WinWidth:				window width
*	WinHeight:				window height
*	Texs:					contains (texR1, texI1, texR2, texI2)
*
* RETURNS:
*   none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::CopyFloatBuffersToScreen(bool FromInput_p, int WinWidth, int WinHeight, GLuint* Texs)
{
	if (!FromInput_p) Buffers->PingForDisplay(Texs);

	glDrawBuffer(GL_BACK_LEFT);
	glViewport(0, 0, WinWidth, WinHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, Width, 0.0, Height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	cgGLSetTextureParameter(DispTexR1, Texs[0]);
	CheckCgError();
	cgGLSetTextureParameter(DispTexI1, Texs[1]);
	CheckCgError();
	cgGLSetTextureParameter(DispTexR2, Texs[2]);
	CheckCgError();
	cgGLSetTextureParameter(DispTexI2, Texs[3]);
	CheckCgError();
	EnableDispFragmentProgram();

	glEnable(GL_TEXTURE_RECTANGLE_NV);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex2f(0.0, 0.0);
	glTexCoord2f(Width, 0.0);
	glVertex2f(Width, 0.0);
	glTexCoord2f(Width, Height);
	glVertex2f(Width, Height);
	glTexCoord2f(0.0, Height);
	glVertex2f(0.0, Height);
	glEnd();
	glDisable(GL_TEXTURE_RECTANGLE_NV);

	DisableDispFragmentProgram();

	if (!FromInput_p) Buffers->Pong(false);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DisplayInputImage
*
* DESCRIPTION:
*	Displays the textures bound to (texReal1, texImag1) and 
*	(texReal2, texImag2).
*
* FORMAL PARAMETERS:
*	WinWidth:  width of the window
*	WinHeight: height of the window
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DisplayInputImage(int WinWidth, int WinHeight)
{
	cgGLSetParameter4fv(DispInvEnergy, InvMax);
	CheckCgError();

	GLuint Texs[4];
	bool FromInput_p = true;
	SetTexs(Texs, texReal1, texImag1, texReal2, texImag2);
	CopyFloatBuffersToScreen(FromInput_p, WinWidth, WinHeight, Texs);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DisplayInputImage
*
* DESCRIPTION:
*	Displays the textures bound to (texTmpR1, texTmpI1) and
*	(texTmpR2, texTmpI2).
*
* FORMAL PARAMETERS:
*	WinWidth:  width of the window
*	WinHeight: height of the window
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DisplayOutputImage(int WinWidth, int WinHeight)
{	
	cgGLSetParameter4fv(DispInvEnergy, InvEnergy);
	CheckCgError();

	GLuint Texs[4];
	bool FromInput_p = false;
	SetTexs(Texs, texTmpR1, texTmpI1, texTmpR2, texTmpI2);
	CopyFloatBuffersToScreen(FromInput_p, WinWidth, WinHeight, Texs);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::RenderFFTStageX
*
* DESCRIPTION:
*	Binds various textures, binds draw buffers and renders a quad or a
*	series of quads depending on the method (method 1 or method 2), for
*	doing FFT in x.
*
* FORMAL PARAMETERS:
*	FirstTime_p:			true if running for the first time
*	texButterflyLookupI:	lookup table containing scrambled coordinates
*	texButterflyLookupWR:	lookup table containing weights, real part.
*	texButterflyLookupWI:	lookup table containing weights, real part.
*	Texs:					contains (texR1, texI1, texR2, texI2)
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::RenderFFTStageX(bool FirstTime_p,
						  int texButterflyLookupI,  
						  int texButterflyLookupWR,  
						  int texButterflyLookupWI, 
						  GLuint *Texs)
{
	Buffers->Ping(FirstTime_p, Texs);

    glViewport(0, 0, Width, Height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, Width, 0.0, Height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
	// Check to see we are doing method 1 or method 2
	if (CurrentButterflyStage < xCutOff) {
		cgGLSetTextureParameter(xButterflyLookupI_1, texButterflyLookupI);
		CheckCgError();
		cgGLSetTextureParameter(xButterflyLookupWR_1, texButterflyLookupWR);
		CheckCgError();
		cgGLSetTextureParameter(xButterflyLookupWI_1, texButterflyLookupWI);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexR1_1, Texs[0]);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexI1_1, Texs[1]);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexR2_1, Texs[2]);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexI2_1, Texs[3]);
		CheckCgError();
	}
	else {
		cgGLSetTextureParameter(xBaseTexR1_2, Texs[0]);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexI1_2, Texs[1]);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexR2_2, Texs[2]);
		CheckCgError();
		cgGLSetTextureParameter(xBaseTexI2_2, Texs[3]);
		CheckCgError();
	}

	EnableFFTFragmentProgramX();
	if (CurrentButterflyStage < xCutOff) {
		// Debugging using immediate mode
		// DrawQuadForFFT();
		// Using display lists
		DrawQuadForFFT_List();
	}
	else {
		// Debugging using immediate mode
		// DrawQuadTilesForFFT_X();
		// Using display lists
		DrawQuadTilesForFFT_X_List();
	}
	DisableFFTFragmentProgramX();

	Buffers->Pong(FirstTime_p);
	Buffers->Swap();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::RenderFFTStageY
*
* DESCRIPTION:
*	Binds various textures, binds draw buffers and renders a quad or a
*	series of quads depending on the method (method 1 or method 2), for
*	doing FFT in y.
*
* FORMAL PARAMETERS:
*	FirstTime_p:			true if running for the first time
*	texButterflyLookupI:	lookup table containing scrambled coordinates
*	texButterflyLookupWR:	lookup table containing weights, real part.
*	texButterflyLookupWI:	lookup table containing weights, real part.
*	Texs:					contains (texR1, texI1, texR2, texI2)
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::RenderFFTStageY(bool FirstTime_p,
						  int texButterflyLookupI,  
						  int texButterflyLookupWR,  
						  int texButterflyLookupWI,
						  GLuint *Texs)
{
	Buffers->Ping(FirstTime_p, Texs);

	glViewport(0, 0, Width, Height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, Width, 0.0, Height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
	// Check to see we are doing method 1 or method 2
	if (CurrentButterflyStage < yCutOff) {
		cgGLSetTextureParameter(yButterflyLookupI_1, texButterflyLookupI);
		CheckCgError();
		cgGLSetTextureParameter(yButterflyLookupWR_1, texButterflyLookupWR);
		CheckCgError();
		cgGLSetTextureParameter(yButterflyLookupWI_1, texButterflyLookupWI);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexR1_1, Texs[0]);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexI1_1, Texs[1]);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexR2_1, Texs[2]);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexI2_1, Texs[3]);
		CheckCgError();
	}
	else {
		cgGLSetTextureParameter(yBaseTexR1_2, Texs[0]);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexI1_2, Texs[1]);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexR2_2, Texs[2]);
		CheckCgError();
		cgGLSetTextureParameter(yBaseTexI2_2, Texs[3]);
		CheckCgError();
	}
	EnableFFTFragmentProgramY();

	if (CurrentButterflyStage < yCutOff) {
		// Debugging using immediate mode
		// DrawQuadForFFT();
		// Using display lists
		DrawQuadForFFT_List();
	}
	else {
		// Debugging using immediate mode
		// DrawQuadTilesForFFT_Y();
		// Using display lists
		DrawQuadTilesForFFT_Y_List();
	}
	DisableFFTFragmentProgramY();
		
	Buffers->Pong(FirstTime_p);
	Buffers->Swap();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DrawQuadTilesForFFT_X
*
* DESCRIPTION:
*	Draws quads for method 2 in doing FFT in x. This function used only
*	for debugging. It has been replaced by DrawQuadTilesForFFT_X_List
*	function, which uses display lists and draw arrays.
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DrawQuadTilesForFFT_X(void)
{
	glActiveTextureARB(GL_TEXTURE0_ARB);
	int i = CurrentButterflyStage;	
	int j, k, kk;
	float tllx, turx, llx, urx;
	int nBlocks  = powf(2.0, (float)(nButterfliesX - 1 - i));
	int Inc      = powf(2.0, (float)(i+1));
	int IncB     = powf(2.0, (float)(i)) - 1.0;
	int nHInputs = powf(2.0, (float)(i));
	float BlockSize = (float)(Width)/(float)nBlocks/2.0;
	float Index1Start, Index1End, Index2Start, Index2End;
	float Index1Start_, Index1End_, Index2Start_, Index2End_;
	float AngStart, AngEnd, AngScale;
	float v1, v2, Grad;
	float AngSign = (ForwardFFT_p) ? 1.0 : -1.0;

	if (i == 0) {
		AngStart = 0.0;
		AngEnd   = 0.0;
	}
	else {
		v1 = 0.0;
		v2 = nBlocks*IncB;
		Grad = (v2 - v1)/(float)(BlockSize-1);
		AngScale = -2.0*M_PI/(float)Width;
		AngStart = AngScale*(v1 - 0.5*Grad);
		AngEnd   = AngScale*(v2 + 0.5*Grad);
	}

	glEnable(GL_TEXTURE_RECTANGLE_NV);
	Index1Start = 0;
	Index2Start = powf(2.0, (float)(i));

	for (k = 0; k < nBlocks; k++) {
		Index1End   = Index1Start + IncB;
		Index2End   = Index2Start + IncB;

		if (i == 0) {
			Index1Start_ = BitReverse((int)Index1Start, Width);
			Index2Start_ = BitReverse((int)Index2Start, Width);
			Index1End_   = BitReverse((int)Index1End, Width);
			Index2End_   = BitReverse((int)Index2End, Width);
		}
		else {
			Index1Start_ = Index1Start;
			Index2Start_ = Index2Start;
			Index1End_   = Index1End;
			Index2End_   = Index2End;
		}

		for (j = 0; j < 2; j++) {
			kk = 2*k + j;

			glBegin(GL_QUADS);

			tllx = -0.5 + kk*BlockSize;
			turx = tllx + BlockSize;

			llx = kk*BlockSize;
			urx = llx + BlockSize;

			glTexCoord4f(tllx, -0.5, Index1Start_-0.5, Index2Start_-0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngStart+j*M_PI));
			glVertex2f(llx, 0.0);

			glTexCoord4f(tllx, Height-1+0.5, Index1Start_-0.5, Index2Start_-0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngStart+j*M_PI));
			glVertex2f(llx, Height);

			glTexCoord4f(turx, Height-1+0.5,  Index1End_+0.5, Index2End_+0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngEnd+j*M_PI));
			glVertex2f(urx, Height);

			glTexCoord4f(turx, -0.5, Index1End_+0.5, Index2End_+0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngEnd+j*M_PI));
			glVertex2f(urx, 0.0);

			glEnd();
		}	
		Index1Start += Inc;
		Index2Start += Inc;
	}
	glDisable(GL_TEXTURE_RECTANGLE_NV);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DrawQuadTilesForFFT_X_List
*
* DESCRIPTION:
*	Draws quads for method 2 in doing FFT in x using display lists and
*	draw arrays.
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DrawQuadTilesForFFT_X_List(void)
{
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_RECTANGLE_NV);
	glCallList(ListX+CurrentButterflyStage);
	glDisable(GL_TEXTURE_RECTANGLE_NV);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DrawQuadTilesForFFT_Y
*
* DESCRIPTION:
*	Draws quads for method 2 in doing FFT in y. This function used only
*	for debugging. It has been replaced by DrawQuadTilesForFFT_Y_List
*	function, which uses display lists and draw arrays.
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DrawQuadTilesForFFT_Y(void)
{
	glActiveTextureARB(GL_TEXTURE0_ARB);
	int i = CurrentButterflyStage;	
	int j, k, kk;
	float tlly, tury, lly, ury;
	int nBlocks  = powf(2.0, (float)(nButterfliesY - 1 - i));
	int Inc      = powf(2.0, (float)(i+1));
	int IncB     = powf(2.0, (float)(i)) - 1.0;
	int nHInputs = powf(2.0, (float)(i));
	float BlockSize = (float)(Height)/(float)nBlocks/2.0;
	float Index1Start, Index1End, Index2Start, Index2End;
	float Index1Start_, Index1End_, Index2Start_, Index2End_;	
	float AngStart, AngEnd, AngScale;
	float v1, v2, Grad;
	float AngSign = (ForwardFFT_p) ? 1.0 : -1.0;

	if (i == 0) {
		AngStart = 0.0;
		AngEnd   = 0.0;
	}
	else {
		v1 = 0.0;
		v2 = nBlocks*IncB;
		Grad = (v2 - v1)/(float)(BlockSize-1);
		AngScale = -2.0*M_PI/(float)Height;
		AngStart = AngScale*(v1 - 0.5*Grad);
		AngEnd   = AngScale*(v2 + 0.5*Grad);
	}

	glEnable(GL_TEXTURE_RECTANGLE_NV);
	Index1Start = 0;
	Index2Start = powf(2.0, (float)(i));

	for (k = 0; k < nBlocks; k++) {
		Index1End   = Index1Start + IncB;
		Index2End   = Index2Start + IncB;

		if (i == 0) {
			Index1Start_ = BitReverse((int)Index1Start, Height);
			Index2Start_ = BitReverse((int)Index2Start, Height);
			Index1End_   = BitReverse((int)Index1End, Height);
			Index2End_   = BitReverse((int)Index2End, Height);
		}
		else {
			Index1Start_ = Index1Start;
			Index2Start_ = Index2Start;
			Index1End_   = Index1End;
			Index2End_   = Index2End;
		}

		for (j = 0; j < 2; j++) {
			kk = 2*k + j;

			glBegin(GL_QUADS);

			tlly = -0.5 + kk*BlockSize;
			tury = tlly + BlockSize;

			lly = kk*BlockSize;
			ury = lly + BlockSize;

			glTexCoord4f(-0.5, tlly, Index1Start_-0.5, Index2Start_-0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngStart+j*M_PI));
			glVertex2f(0.0, lly);

			glTexCoord4f(Width-1+0.5, tlly, Index1Start_-0.5, Index2Start_-0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngStart+j*M_PI));
			glVertex2f(Width, lly);

			glTexCoord4f(Width-1+0.5, tury,  Index1End_+0.5, Index2End_+0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngEnd+j*M_PI));
			glVertex2f(Width, ury);

			glTexCoord4f(-0.5, tury, Index1End_+0.5, Index2End_+0.5);
			glMultiTexCoord1f(GL_TEXTURE1_ARB, AngSign*(AngEnd+j*M_PI));
			glVertex2f(0.0, ury);

			glEnd();

		}	
		Index1Start += Inc;
		Index2Start += Inc;
	}
	glDisable(GL_TEXTURE_RECTANGLE_NV);

}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DrawQuadTilesForFFT_Y_List
*
* DESCRIPTION:
*	Draws quads for method 2 in doing FFT in y using display lists and
*	draw arrays.
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DrawQuadTilesForFFT_Y_List(void)
{
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_RECTANGLE_NV);
	glCallList(ListY+CurrentButterflyStage);
	glDisable(GL_TEXTURE_RECTANGLE_NV);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::ComputeWeight
*
* DESCRIPTION:
*	Computes the weights.
*
* FORMAL PARAMETERS:
*	N:	Number of samples
*   k:	Current sample
*
* RETURNS:
*	Wr: real part of the weight
*	Wi: imaginary part of the weight
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::ComputeWeight(int N, int k, float &Wr, float &Wi)
{
	Wr =  cosl(2.0*M_PI*k/(float)N);
	Wi = -sinl(2.0*M_PI*k/(float)N);

	Wi = (ForwardFFT_p == true) ? Wi : -Wi;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::BitReverse
*
* DESCRIPTION:
*	Reverses bits in index
*
* FORMAL PARAMETERS:
*	i:	input index
*   N:	Number of bits in the index
*
* RETURNS:
*	the bit-reversed index
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
int FFT::BitReverse(int i, int N)
{
	int j = i;

	int M   = N;
	int Sum = 0;
	int W   = 1;
	M = M / 2;
	while (M != 0) {
		j = (i & M) > M-1;
		Sum += j*W;
		W *= 2;
		M = M/2;
	}
	return(Sum);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::CreateButterflyLookups
*
* DESCRIPTION:
*	Creates scrambling indices and weights for each butterfly stage
*
* FORMAL PARAMETERS:
*	NButterflies:	number of butterfly stages
*   N:				number of samples
*
* RETURNS:
*	butterflylookupI:	an array containing scrambling lookup table
*	butterflylookupWR:	real part of the weights for each stage
*	butterflylookupWI:	imaginary part of the weights for each stage
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::CreateButterflyLookups(float *butterflylookupI,
								 float *butterflylookupWR, 
								 float *butterflylookupWI, 
								 int NButterflies, int N)
{
	float *ptr0 = butterflylookupI;
	float *ptr1 = butterflylookupWR;
	float *ptr2 = butterflylookupWI;

	int i, j, k, i1, i2, j1, j2;
	int nBlocks, nHInputs;
	float wr, wi;
	float *qtr0, *qtr1, *qtr2;
	float scale = 1.0/((float)(N-1));

	for (i = 0; i < NButterflies; i++) {
		nBlocks  = powf(2.0, (float)(NButterflies - 1 - i));
		nHInputs = powf(2.0, (float)(i));
		qtr0 = ptr0;
		qtr1 = ptr1;
		qtr2 = ptr2;
		for (j = 0; j < nBlocks; j++) {

			for (k = 0; k < nHInputs; k++) {

				if (i == 0) {
					i1 = j*nHInputs*2 + k;
					i2 = j*nHInputs*2 + nHInputs + k;
					j1 = BitReverse(i1, N);
					j2 = BitReverse(i2, N);
				}
				else {
					i1 = j*nHInputs*2 + k;
					i2 = j*nHInputs*2 + nHInputs + k;
					j1 = i1;
					j2 = i2;
				}

				ComputeWeight(N, k*nBlocks, wr, wi);

				*(qtr0 + 2*i1)   = j1;
				*(qtr0 + 2*i1+1) = j2;
				*(qtr1 + i1) = wr;
				*(qtr2 + i1) = wi;

				*(qtr0 + 2*i2)   = j1;
				*(qtr0 + 2*i2+1) = j2;
				*(qtr1 + i2) = -wr;
				*(qtr2 + i2) = -wi;

			}
		}
		ptr0 += 2*N;
		ptr1 += N;
		ptr2 += N;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::ComputeVerticesTexCoordsX
*
* DESCRIPTION:
*	Computes scrambling indices and weights for each sub-quad for method 2
*
* FORMAL PARAMETERS:
*	Stage:	buttefly stage
*
* RETURNS:
*	Vs:		vertices
*	Ts0:	texture coordinates for texture unit 0
*	Ts1:	texture coordinates for texture unit 1
*	Ns:		number of vertices
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::ComputeVerticesTexCoordsX(int Stage, float *Vs, float *Ts0, float *Ts1, int &Ns)
{
	int i = Stage;	
	int j, k, kk;
	float tllx, turx, llx, urx;
	int nBlocks  = powf(2.0, (float)(nButterfliesX - 1 - i));
	int Inc      = powf(2.0, (float)(i+1));
	int IncB     = powf(2.0, (float)(i)) - 1.0;
	int nHInputs = powf(2.0, (float)(i));
	float BlockSize = (float)(Width)/(float)nBlocks/2.0;
	float Index1Start, Index1End, Index2Start, Index2End;
	float Index1Start_, Index1End_, Index2Start_, Index2End_;
	float AngStart, AngEnd, AngScale;
	float v1, v2, Grad;
	float AngSign = (ForwardFFT_p) ? 1.0 : -1.0;

	if (i == 0) {
		AngStart = 0.0;
		AngEnd   = 0.0;
	}
	else {
		v1 = 0.0;
		v2 = nBlocks*IncB;
		Grad = (v2 - v1)/(float)(BlockSize-1);
		AngScale = -2.0*M_PI/(float)Width;
		AngStart = AngScale*(v1 - 0.5*Grad);
		AngEnd   = AngScale*(v2 + 0.5*Grad);
	}
	float *vptr  = Vs;
	float *tptr0 = Ts0;
	float *tptr1 = Ts1;
	
	Ns = 0;
	Index1Start = 0;
	Index2Start = powf(2.0, (float)(i));
	for (k = 0; k < nBlocks; k++) {
		Index1End   = Index1Start + IncB;
		Index2End   = Index2Start + IncB;
		if (i == 0) {
			Index1Start_ = BitReverse((int)Index1Start, Width);
			Index2Start_ = BitReverse((int)Index2Start, Width);
			Index1End_   = BitReverse((int)Index1End, Width);
			Index2End_   = BitReverse((int)Index2End, Width);
		}
		else {
			Index1Start_ = Index1Start;
			Index2Start_ = Index2Start;
			Index1End_   = Index1End;
			Index2End_   = Index2End;
		}
		for (j = 0; j < 2; j++) {
			kk = 2*k + j;
		
			tllx = -0.5 + kk*BlockSize;
			turx = tllx + BlockSize;
			
			llx = kk*BlockSize;
			urx = llx + BlockSize;

			// 1st vertex
			*(tptr0++) = tllx;
			*(tptr0++) = -0.5;
			*(tptr0++) = Index1Start_-0.5;
			*(tptr0++) = Index2Start_-0.5;
			
			*(tptr1++) = AngSign*(AngStart+j*M_PI);

			*(vptr++) = llx;
			*(vptr++) = 0.0;
			Ns++;
					
			// 2nd vertex
			*(tptr0++) = turx;
			*(tptr0++) = -0.5;
			*(tptr0++) = Index1End_+0.5;
			*(tptr0++) = Index2End_+0.5;
			
			*(tptr1++) = AngSign*(AngEnd+j*M_PI);

			*(vptr++) = urx;
			*(vptr++) = 0.0;
			Ns++;
											
			// 3rd vertex
			*(tptr0++) = turx;
			*(tptr0++) = Height-1+0.5;
			*(tptr0++) = Index1End_+0.5;
			*(tptr0++) = Index2End_+0.5;

			*(tptr1++) = AngSign*(AngEnd+j*M_PI);

			*(vptr++) = urx;
			*(vptr++) = Height;
			Ns++;
			
			// 4th vertex
			*(tptr0++) = tllx;
			*(tptr0++) = Height-1+0.5;
			*(tptr0++) = Index1Start_-0.5;
			*(tptr0++) = Index2Start_-0.5;
			
			*(tptr1++) = AngSign*(AngStart+j*M_PI);

			*(vptr++) = llx;
			*(vptr++) = Height;
			Ns++;
		}	
		Index1Start += Inc;
		Index2Start += Inc;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::GenDisplayListsX
*
* DESCRIPTION:
*	Generates display lists containing draw arrays for x FFT method 2
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::GenDisplayListsX(void)
{
	int Ns = 0, i;
	int nButterflies = nButterfliesX;
	int nBlocks  = powf(2.0, (float)(nButterflies - 1));
	int SizeV   = nBlocks*2*4*2;
	int SizeT   = nBlocks*2*4*4;

	float *Vs, *Ts0, *Ts1;
	Vs  = new float [SizeV];
	Ts0 = new float [SizeT];
	Ts1 = new float [SizeT];

	ListX = glGenLists(nButterflies);
	for (i = 0; i < nButterflies; i++) {
		memset(Vs, 0, sizeof(float)*SizeV);
		memset(Ts0, 0, sizeof(float)*SizeT);
		memset(Ts1, 0, sizeof(float)*SizeT);

		ComputeVerticesTexCoordsX(i, Vs, Ts0, Ts1, Ns); 

		glEnableClientState(GL_VERTEX_ARRAY); 
		glVertexPointer(2, GL_FLOAT, 0, Vs);

		glClientActiveTextureARB(GL_TEXTURE0_ARB);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(4 , GL_FLOAT, 0, Ts0); 
		
		glClientActiveTextureARB(GL_TEXTURE1_ARB);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(1 , GL_FLOAT, 0, Ts1);

		glNewList(ListX+i, GL_COMPILE);
		glDrawArrays(GL_QUADS, 0, Ns);
		glEndList();
		glDisableClientState(GL_VERTEX_ARRAY);
		glClientActiveTextureARB(GL_TEXTURE0_ARB);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glClientActiveTextureARB(GL_TEXTURE1_ARB);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	delete [] Vs;
	delete [] Ts0;
	delete [] Ts1;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::ComputeVerticesTexCoordsY
*
* DESCRIPTION:
*	Computes scrambling indices and weights for each sub-quad for method 2
*
* FORMAL PARAMETERS:
*	Stage:	buttefly stage
*
* RETURNS:
*	Vs:		vertices
*	Ts0:	texture coordinates for texture unit 0
*	Ts1:	texture coordinates for texture unit 1
*	Ns:		number of vertices
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::ComputeVerticesTexCoordsY(int Stage, float *Vs, float *Ts0, float *Ts1, int &Ns)
{
	int i = Stage;	
	int j, k, kk;
	float tlly, tury, lly, ury;
	int nBlocks  = powf(2.0, (float)(nButterfliesY - 1 - i));
	int Inc      = powf(2.0, (float)(i+1));
	int IncB     = powf(2.0, (float)(i)) - 1.0;
	int nHInputs = powf(2.0, (float)(i));
	float BlockSize = (float)(Height)/(float)nBlocks/2.0;
	float Index1Start, Index1End, Index2Start, Index2End;
	float Index1Start_, Index1End_, Index2Start_, Index2End_;
	float AngStart, AngEnd, AngScale;
	float v1, v2, Grad;
	float AngSign = (ForwardFFT_p) ? 1.0 : -1.0;

	if (i == 0) {
		AngStart = 0.0;
		AngEnd   = 0.0;
	}
	else {
		v1 = 0.0;
		v2 = nBlocks*IncB;
		Grad = (v2 - v1)/(float)(BlockSize-1);
		AngScale = -2.0*M_PI/(float)Height;
		AngStart = AngScale*(v1 - 0.5*Grad);
		AngEnd   = AngScale*(v2 + 0.5*Grad);
	}
	float *vptr  = Vs;
	float *tptr0 = Ts0;
	float *tptr1 = Ts1;

	Ns = 0;
	Index1Start = 0;
	Index2Start = powf(2.0, (float)(i));
	for (k = 0; k < nBlocks; k++) {
		Index1End   = Index1Start + IncB;
		Index2End   = Index2Start + IncB;
		if (i == 0) {
			Index1Start_ = BitReverse((int)Index1Start, Height);
			Index2Start_ = BitReverse((int)Index2Start, Height);
			Index1End_   = BitReverse((int)Index1End, Height);
			Index2End_   = BitReverse((int)Index2End, Height);
		}
		else {
			Index1Start_ = Index1Start;
			Index2Start_ = Index2Start;
			Index1End_   = Index1End;
			Index2End_   = Index2End;
		}
		for (j = 0; j < 2; j++) {
			kk = 2*k + j;
		
			tlly = -0.5 + kk*BlockSize;
			tury = tlly + BlockSize;
			
			lly = kk*BlockSize;
			ury = lly + BlockSize;

			// 1st vertex
			*(tptr0++) = -0.5;
			*(tptr0++) = tlly;
			*(tptr0++) = Index1Start_-0.5;
			*(tptr0++) = Index2Start_-0.5;
			
			*(tptr1++) = AngSign*(AngStart+j*M_PI);

			*(vptr++) = 0.0;
			*(vptr++) = lly;
			Ns++;
									
			// 2nd vertex
			*(tptr0++) = -0.5;
			*(tptr0++) = tury;
			*(tptr0++) = Index1End_+0.5;
			*(tptr0++) = Index2End_+0.5;
			
			*(tptr1++) = AngSign*(AngEnd+j*M_PI);

			*(vptr++) = 0.0;
			*(vptr++) = ury;
			Ns++;
							
			// 3rd vertex				
			*(tptr0++) = Width-1+0.5;
			*(tptr0++) = tury;
			*(tptr0++) = Index1End_+0.5;
			*(tptr0++) = Index2End_+0.5;

			*(tptr1++) = AngSign*(AngEnd+j*M_PI);

			*(vptr++) = Width;
			*(vptr++) = ury;
			Ns++;
					
			// 4th vertex
			*(tptr0++) = Width-1+0.5;
			*(tptr0++) = tlly;
			*(tptr0++) = Index1Start_-0.5;
			*(tptr0++) = Index2Start_-0.5;

			*(tptr1++) = AngSign*(AngStart+j*M_PI);

			*(vptr++) = Width;
			*(vptr++) = lly;
			Ns++;
		}	
		Index1Start += Inc;
		Index2Start += Inc;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::GenDisplayListsY
*
* DESCRIPTION:
*	Generates display lists containing draw arrays for y FFT method 2
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::GenDisplayListsY(void)
{
	int Ns = 0, i;
	int nButterflies = nButterfliesY;
	int nBlocks  = powf(2.0, (float)(nButterflies - 1));
	int SizeV   = nBlocks*2*4*2;
	int SizeT   = nBlocks*2*4*4;

	float *Vs, *Ts0, *Ts1;
	Vs  = new float [SizeV];
	Ts0 = new float [SizeT];
	Ts1 = new float [SizeT];

	ListY = glGenLists(nButterflies);
	for (i = 0; i < nButterflies; i++) {
		memset(Vs, 0, sizeof(float)*SizeV);
		memset(Ts0, 0, sizeof(float)*SizeT);
		memset(Ts1, 0, sizeof(float)*SizeT);

		ComputeVerticesTexCoordsY(i, Vs, Ts0, Ts1, Ns);

		glEnableClientState(GL_VERTEX_ARRAY); 
		glVertexPointer(2, GL_FLOAT, 0, Vs);

		glClientActiveTextureARB(GL_TEXTURE0_ARB);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(4 , GL_FLOAT, 0, Ts0); 
		
		glClientActiveTextureARB(GL_TEXTURE1_ARB);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(1 , GL_FLOAT, 0, Ts1);

		glNewList(ListY+i, GL_COMPILE);
		glDrawArrays(GL_QUADS, 0, Ns);
		glEndList();

		glDisableClientState(GL_VERTEX_ARRAY);
		glClientActiveTextureARB(GL_TEXTURE0_ARB);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glClientActiveTextureARB(GL_TEXTURE1_ARB);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	delete [] Vs;
	delete [] Ts0;	
	delete [] Ts1;	
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DrawQuadForFFT
*
* DESCRIPTION:
*	Draws a quad
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DrawQuadForFFT(void)
{
	glEnable(GL_TEXTURE_RECTANGLE_NV);	
	glBegin(GL_QUADS);
	glTexCoord2f(-0.5, -0.5);
	glVertex2f(0.0, 0.0);
	glTexCoord2f(Width-1+0.5, -0.5);
	glVertex2f(Width, 0.0);
	glTexCoord2f(Width-1+0.5, Height-1+0.5);
	glVertex2f(Width, Height);
	glTexCoord2f(-0.5, Height-1+0.5);
	glVertex2f(0.0, Height);
	glEnd();	
	glDisable(GL_TEXTURE_RECTANGLE_NV);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::GenQuadDisplayList
*
* DESCRIPTION:
*	Generates a display list for drawing a quad for method 1
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::GenQuadDisplayList(void)
{
	glEnable(GL_TEXTURE_RECTANGLE_NV);	
	glBegin(GL_QUADS);
	glTexCoord2f(-0.5, -0.5);
	glVertex2f(0.0, 0.0);
	glTexCoord2f(Width-1+0.5, -0.5);
	glVertex2f(Width, 0.0);
	glTexCoord2f(Width-1+0.5, Height-1+0.5);
	glVertex2f(Width, Height);
	glTexCoord2f(-0.5, Height-1+0.5);
	glVertex2f(0.0, Height);
	glEnd();	
	glDisable(GL_TEXTURE_RECTANGLE_NV);

	float *Vs, *Ts;
	int i;
	int Size   = 2*4;
	Vs = new float [Size];
	Ts = new float [Size];

	QList = glGenLists(1);
	memset(Vs, 0, sizeof(float)*Size);
	memset(Ts, 0, sizeof(float)*Size);

	glTexCoord2f(-0.5, -0.5);
	glVertex2f(0.0, 0.0);
	glTexCoord2f(Width-1+0.5, -0.5);
	glVertex2f(Width, 0.0);
	glTexCoord2f(Width-1+0.5, Height-1+0.5);
	glVertex2f(Width, Height);
	glTexCoord2f(-0.5, Height-1+0.5);
	glVertex2f(0.0, Height);

	i = 0;
	Vs[i] = 0.0; i++;
	Vs[i] = 0.0; i++;

	Vs[i] = Width; i++;
	Vs[i] = 0.0; i++;

	Vs[i] = Width; i++;
	Vs[i] = Height; i++;

	Vs[i] = 0.0; i++;
	Vs[i] = Height; i++;

	i = 0;
	Ts[i] = -0.5; i++;
	Ts[i] = -0.5; i++;

	Ts[i] = Width-1+0.5; i++;
	Ts[i] = -0.5; i++;

	Ts[i] = Width-1+0.5; i++;
	Ts[i] = Height-1+0.5; i++;

	Ts[i] = -0.5; i++;
	Ts[i] = Height-1+0.5; i++;


	glEnableClientState(GL_VERTEX_ARRAY); 
	glVertexPointer(2, GL_FLOAT, 0, Vs);

	glClientActiveTextureARB(GL_TEXTURE0_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2 , GL_FLOAT, 0, Ts); 

	glNewList(QList, GL_COMPILE);
	glDrawArrays(GL_QUADS, 0, 4);
	glEndList();

	glDisableClientState(GL_VERTEX_ARRAY);
	glClientActiveTextureARB(GL_TEXTURE0_ARB);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	delete [] Vs;
	delete [] Ts;		
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DrawQuadForFFT_List
*
* DESCRIPTION:
*	Draws a quad for method 1 in doing FFT using display lists and
*	draw arrays.
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DrawQuadForFFT_List(void)
{
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_RECTANGLE_NV);
	glCallList(QList);
	glDisable(GL_TEXTURE_RECTANGLE_NV);
}
