/***************************************************************************
*		FILE NAME: PingPong.cpp
*
* ONE LINE SUMMARY:
*		This file contains some of the member function for the PingPong class.
*		This class manages the bbuffers.
*        
*		Thilaka Sumanaweera
*		Siemens Medical Solutions USA, Inc.
*		1230 Shorebird Way
*		Mountain View, CA 94039
*		USA
*		Thilaka.Sumanaweera@siemens.com
*
* DESCRIPTION:
*
*****************************************************************************
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
****************************************************************************/
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "PingPong.h"

using namespace std;
/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::constructor
*
* DESCRIPTION:
*
* FORMAL PARAMETERS:
*   none
*
* RETURNS:
*   none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
PingPong::PingPong(Vendor GPU, PingMethod Method_, int Width, int Height)
{	
	Pbuffer1 = 0;
	Pbuffer2 = 0;
	TexUnits = 0;
	BufSet1  = 0;
	BufSet2  = 0;
	wglBufSet1 = 0;
	wglBufSet2 = 0;

	int i;
	bool quadro_p;
	// SINGLE_BUFFER requires using 1 pbuffer with 8 draw buffers obtained by
	// selecting the stereo mode: GL_FRONT_LEFT, GL_BACK_LEFT, GL_FRONT_RIGHT,
	// GL_BACK_RIGHT, GL_AUX0, GL_AUX1, GL_AUX2 and GL_AUX3. As of 12/15/2004
	// this was only available on the Quadro line of boards. If the current
	// board is not a Quadro NV4x and up, we can use the DOUBLE_BUFFER method,
	// which only requires 1 pbuffer with 4 draw buffers.
	if ((Method_ == SINGLE_PBUFFER) || (Method_ == DOUBLE_PBUFFER)) {
		Method = Method_;
	}
	else {
		quadro_p = Quadro_p();
		if (quadro_p) {
			Method = SINGLE_PBUFFER;
			cerr << "Board is a Quadro" << endl;
		}
		else {
			Method = DOUBLE_PBUFFER;
		}
	}
	switch (Method) {
		case SINGLE_PBUFFER:
			cerr << "Using SINGLE_PBUFFER: *Fastest*" << endl;
			break;
		case DOUBLE_PBUFFER:
			cerr << "Using DOUBLE_PBUFFER: *Not the Fastest* (Use a Quadro board to get *Fastest*)" << endl;
			break;
		default:
			break;
	}

	CreatePbuffer(GPU, Width, Height, Pbuffer1);
	if (Method == DOUBLE_PBUFFER) {
		CreatePbuffer(GPU, Width, Height, Pbuffer2);
	}

	NBuf = 4;
	TexUnits = new GLenum [NBuf];
	i = 0;
	TexUnits[i++] = GL_TEXTURE0_ARB;
	TexUnits[i++] = GL_TEXTURE1_ARB;
	TexUnits[i++] = GL_TEXTURE2_ARB;
	TexUnits[i++] = GL_TEXTURE3_ARB;

	BufSet1 = new GLenum [NBuf];
	i = 0;
	BufSet1[i++] = GL_AUX0;
	BufSet1[i++] = GL_AUX1;
	BufSet1[i++] = GL_AUX2;
	BufSet1[i++] = GL_AUX3;
	BufSet2 = new GLenum [NBuf];
	i = 0;
	BufSet2[i++] = GL_FRONT_LEFT;
	BufSet2[i++] = GL_BACK_LEFT;
	BufSet2[i++] = GL_FRONT_RIGHT;;
	BufSet2[i++] = GL_BACK_RIGHT;
	
	wglBufSet1 = new GLenum [NBuf];
	i = 0;
	wglBufSet1[i++] = WGL_AUX0_ARB;
	wglBufSet1[i++] = WGL_AUX1_ARB;
	wglBufSet1[i++] = WGL_AUX2_ARB;
	wglBufSet1[i++] = WGL_AUX3_ARB;
	wglBufSet2 = new GLenum [NBuf];
	i = 0;
	wglBufSet2[i++] = WGL_FRONT_LEFT_ARB;
	wglBufSet2[i++] = WGL_BACK_LEFT_ARB;
	wglBufSet2[i++] = WGL_FRONT_RIGHT_ARB;
	wglBufSet2[i++] = WGL_BACK_RIGHT_ARB;

	Src = Pbuffer2;
	Dst = Pbuffer1;

	DstBufSet    = BufSet1;
	wglDstBufSet = wglBufSet1;
	switch (Method) {
		case SINGLE_PBUFFER:
			SrcBufSet    = BufSet2;
			wglSrcBufSet = wglBufSet2;
			break;
		case DOUBLE_PBUFFER:
			SrcBufSet    = BufSet1;
			wglSrcBufSet = wglBufSet1;
			break;
		default:
			break;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::destructor
*
* DESCRIPTION:
*
* FORMAL PARAMETERS:
*   none
*
* RETURNS:
*   none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
PingPong::~PingPong(void)
{
	if (wglBufSet2) {
		delete [] wglBufSet2;
		wglBufSet2 = 0;
	}
	if (wglBufSet1) {
		delete [] wglBufSet1;
		wglBufSet1 = 0;
	}	
	if (BufSet2) {
		delete [] BufSet2;
		BufSet2 = 0;
	}
	if (BufSet1) {
		delete [] BufSet1;
		BufSet1 = 0;
	}
	if (Pbuffer1) {
		delete Pbuffer1;
		Pbuffer1 = 0;
	}
	if (Pbuffer2) {
		delete Pbuffer2;
		Pbuffer1 = 0;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::Quadro_p
*
* DESCRIPTION:
*	Returns true if the current board is a Quadro board.
*
* FORMAL PARAMETERS:
*   none
*
* RETURNS:
*   bool
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
bool PingPong::Quadro_p(void)
{
	char *str = strdup((char *)glGetString(GL_RENDERER));
	return((strstr(str, "Quadro") != NULL));
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::CreatePbuffer
*
* DESCRIPTION:
*	Create a pbuffer
*
* FORMAL PARAMETERS:
*	GPU:	the vendor type
*	Width:	the width of the data array
*	height: the height of the data array
*
* RETURNS:
*	pbuf:	the pbuffer created
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void PingPong::CreatePbuffer(Vendor GPU, int Width, int Height, PBuffer *&pbuf)
{
	int RedBits      = sizeof(float)*8;
	int GreenBits    = 0;
	int BlueBits     = 0;
	int AlphaBits    = 0;
	bool isFloat     = true;
	bool dBuffer     = true;
	bool hasDepth    = false;
	bool hasStencil  = false;
	bool hasStereo   = true;
	bool texture     = true;
	bool share       = true;
	int NoAuxBuf     = 4;
	InternalFormat IntForm = PBUF_R;

	pbuf = new PBuffer(GPU, Width, Height, IntForm, RedBits, GreenBits, BlueBits, AlphaBits,
		isFloat, dBuffer, hasDepth, NoAuxBuf, 
		hasStencil, hasStereo, texture, share);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::BindTextures
*
* DESCRIPTION:
*	Binds the a series of draw buffers as textures
*
* FORMAL PARAMETERS:
*	Pbuf:	pbuffer
*	Texs:	the texture names
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void PingPong::BindTextures(PBuffer *Pbuf, GLuint *Texs)
{
	int i = 0;
	for (i = 0; i < NBuf; i++) {
		glActiveTextureARB(TexUnits[i]);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, Texs[i]);
		Pbuf->BindAsTexture(wglSrcBufSet[i]);
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::UnBindTextures
*
* DESCRIPTION:
*	Unbinds the draw buffers as textures
*
* FORMAL PARAMETERS:
*	Pbuf:	pbuffer
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void PingPong::UnBindTextures(PBuffer *Pbuf)
{
	int i = 0;
	for (i = 0; i < NBuf; i++) {
		Pbuf->ReleaseTexture(wglSrcBufSet[i]);
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::PingForDisplay
*
* DESCRIPTION:
*	Bind textures depending on the method for dislay
*
* FORMAL PARAMETERS:
*	Texs:	the texture names
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void PingPong::PingForDisplay(GLuint *Texs)
{
	switch (Method) {
		case SINGLE_PBUFFER:
			BindTextures(Dst, Texs);
			break;
		case DOUBLE_PBUFFER:
			BindTextures(Src, Texs);
			break;
		default:
			break;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::Ping
*
* DESCRIPTION:
*	Bind textures depending on the method
*
* FORMAL PARAMETERS:
*	FirstTime_p:	true for the first time
*	Texs:			the texture names
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void PingPong::Ping(bool FirstTime_p, GLuint *Texs)
{
	switch (Method) {
		case SINGLE_PBUFFER:
			glDrawBuffersATI(NBuf, DstBufSet);
			if (!FirstTime_p) {
				BindTextures(Dst, Texs);
			}
			break;
		case DOUBLE_PBUFFER:
			Dst->Bind();
			glDrawBuffersATI(NBuf, DstBufSet);
			if (!FirstTime_p) {
				BindTextures(Src, Texs);
			}
			break;
		default:
			break;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::Pong
*
* DESCRIPTION:
*	Unbinds textures depending on the method
*
* FORMAL PARAMETERS:
*	FirstTime_p:	true for the first time
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void PingPong::Pong(bool FirstTime_p)
{
	switch (Method) {
		case SINGLE_PBUFFER:
			if (!FirstTime_p) {
				UnBindTextures(Dst);
			}
			break;
		case DOUBLE_PBUFFER:
			if (!FirstTime_p) {
				UnBindTextures(Src);
			}
			Dst->Unbind();
			break;
		default:
			break;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::Swap
*
* DESCRIPTION:
*	Swap various buffers in a ping-pong fashion.
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
void PingPong::Swap(void)
{
	PBuffer *TmpBuf;
	GLenum *TmpSet;
	GLenum *wglTmpSet;
	switch (Method) {
		case SINGLE_PBUFFER:
			TmpSet    = SrcBufSet;
			SrcBufSet = DstBufSet;
			DstBufSet = TmpSet;

			wglTmpSet    = wglSrcBufSet;
			wglSrcBufSet = wglDstBufSet;
			wglDstBufSet = wglTmpSet;
			break;
		case DOUBLE_PBUFFER:
			TmpBuf = Src;
			Src    = Dst;
			Dst    = TmpBuf;
			break;
		default:
			break;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::Bind
*
* DESCRIPTION:
*	Bind the pbuffer if necessary
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
void PingPong::Bind(void)
{
	switch (Method) {
		case SINGLE_PBUFFER:
			Dst->Bind();
			break;
		case DOUBLE_PBUFFER:
			break;
		default:
			break;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   PingPong::UnBind
*
* DESCRIPTION:
*	unbind the pbuffer if necessary
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
void PingPong::Unbind(void)
{
	switch (Method) {
		case SINGLE_PBUFFER:
			Dst->Unbind();
			break;
		case DOUBLE_PBUFFER:
			break;
		default:
			break;
	}
}
