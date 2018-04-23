/***************************************************************************
*		FILE NAME:  FFT.cpp
*
* ONE LINE SUMMARY:
*		This file contains some of the member function for the FFT class.
*
*		This FFT implentation in the NVIDIA GPU is based on the chapter
*		called "Medical Image Reconstruction in GPUs using the FFT" in the
*		book: "GPU Gems 2: Programming Techniques for High-Performance
*		Graphics and General-Purpose Computation", published by Addison
*		Wesley Professional.
*
*		The FFT can be performed by using two methods:
*			Method 1: Mostly loading the fragment processor
*			Method 2: Loading the vertex processor, the rasterizer and
*						fragment processor
*		Method 1 produces faster frame rates for early butterfly stages.
*		Method 2 produces faster frame rates for the later butterfly stages.
*		This implentation picks the optimal transition point between
*		method 1 and method 2 automatically by exhaustively searching
*		all possible transition points.
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
#include "FFT.h"

using namespace std;

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::constructor
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
FFT::FFT(void)
{	
	Buffers = 0;
	xFrameRate = 0;
	yFrameRate = 0;
	butterflyLookupI_X = 0;
	butterflyLookupWR_X = 0;
	butterflyLookupWI_X = 0;
	butterflyLookupI_Y = 0;
	butterflyLookupWR_Y = 0;
	butterflyLookupWI_Y = 0;

	texButterflyLookupI_X  = 0;
	texButterflyLookupWR_X = 0;
	texButterflyLookupWI_X = 0;
	texButterflyLookupI_Y = 0;
	texButterflyLookupWR_Y = 0;
	texButterflyLookupWI_Y = 0;

	// texture names
	texReal1 = 0;
	texImag1 = 0;
	texReal2 = 0;
	texImag2 = 0;
	
	texTmpR1 = 0;
	texTmpI1 = 0;
	texTmpR2 = 0;
	texTmpI2 = 0;

	// Display lists
	QList = 0;
	ListX = 0;
	ListY = 0;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::constructor
*
* DESCRIPTION:
*
* FORMAL PARAMETERS:
*   ForwardFFT_p_:	true if want to do forward FFT. false if reverse FFT
*	Width_:			width of the input complex 2D images
*	Height_:		height of the input complex 2D images
*
* RETURNS:
*   none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
FFT::FFT(bool ForwardFFT_p_, PingMethod PMethod, int Width_, int Height_)
{	
	SetGPUVendor();
	if (GPU != GPU_NVIDIA) {
		cerr << "Currently only supported on NVIDIA GPU (Quadro NV40 and up)" << endl;
		exit(0);
	}

	ForwardFFT_p = ForwardFFT_p_;

	Width  = Width_;
	Height = Height_;
	type = XY;
	
	int i;
	for (i = 0; i < 4; i++) {
		InvEnergy[i] = 0.0;
		InvMax[i]    = 0.0;
	}

	ShowFirstImage_p = true;
	SetDisplayMask();

	nButterfliesX = logf((float)Width)/logf(2.0);
	nButterfliesY = logf((float)Height)/logf(2.0);

	nButterfliesXWorking = nButterfliesX;
	nButterfliesYWorking = nButterfliesY;

	xFrameRate = new float [nButterfliesX];
	memset(xFrameRate, 0.0, sizeof(float)*nButterfliesX);
	yFrameRate = new float [nButterfliesY];
	memset(yFrameRate, 0.0, sizeof(float)*nButterfliesY);

	//xCutOff = nButterfliesX/5;
	//yCutOff = nButterfliesY/2;
	xCutOff = 0;
	yCutOff = 0;

	// Create the buffer manager
	Buffers = new PingPong(GPU, PMethod, Width, Height);

	butterflyLookupI_X  = new float [nButterfliesX*Width*2];
	butterflyLookupWR_X = new float [nButterfliesX*Width];
	butterflyLookupWI_X = new float [nButterfliesX*Width];

	butterflyLookupI_Y  = new float [nButterfliesY*Height*2];
	butterflyLookupWR_Y = new float [nButterfliesY*Height];
	butterflyLookupWI_Y = new float [nButterfliesY*Height];

	texButterflyLookupI_X  = new GLuint [nButterfliesX];
	texButterflyLookupWR_X = new GLuint [nButterfliesX];
	texButterflyLookupWI_X = new GLuint [nButterfliesX];

	texButterflyLookupI_Y  = new GLuint [nButterfliesY];
	texButterflyLookupWR_Y = new GLuint [nButterfliesY];
	texButterflyLookupWI_Y = new GLuint [nButterfliesY];
	
	CreateButterflyLookups(butterflyLookupI_X, 
		butterflyLookupWR_X, 
		butterflyLookupWI_X, nButterfliesX, Width);
	CreateButterflyLookups(butterflyLookupI_Y, 
		butterflyLookupWR_Y, 
		butterflyLookupWI_Y, nButterfliesY, Height);

	InitCg();
	InitTextures();
	InitCgPrograms();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::destructor
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
FFT::~FFT(void)
{
	if (Buffers) {
		delete Buffers;
		Buffers = 0;
	}
	if (xFrameRate) {
		delete [] xFrameRate;
		xFrameRate = 0;
	}
	if (yFrameRate) {
		delete [] yFrameRate;
		yFrameRate = 0;
	}
	if (butterflyLookupI_X) {
		delete [] butterflyLookupI_X;
		butterflyLookupI_X = 0;
	}
	if (butterflyLookupWR_X) {
		delete [] butterflyLookupWR_X;
		butterflyLookupWR_X = 0;
	}
	if (butterflyLookupWI_X) {
		delete [] butterflyLookupWI_X;
		butterflyLookupWI_X = 0;
	}
	if (butterflyLookupI_Y) {
		delete [] butterflyLookupI_Y;
		butterflyLookupI_Y = 0;
	}
	if (butterflyLookupWR_Y) {
		delete [] butterflyLookupWR_Y;
		butterflyLookupWR_Y = 0;
	}
	if (butterflyLookupWI_Y) {
		delete [] butterflyLookupWI_Y;
		butterflyLookupWI_Y = 0;
	}
	if (texButterflyLookupI_X) {
		delete [] texButterflyLookupI_X;
		texButterflyLookupI_X = 0;
	}
	if (texButterflyLookupWR_X) {
		delete [] texButterflyLookupWR_X;
		texButterflyLookupWR_X = 0;
	}
	if (texButterflyLookupWI_X) {
		delete [] texButterflyLookupWI_X;
		texButterflyLookupWI_X = 0;
	}
	if (texButterflyLookupI_Y) {
		delete [] texButterflyLookupI_Y;
		texButterflyLookupI_Y = 0;
	}
	if (texButterflyLookupWR_Y) {
		delete [] texButterflyLookupWR_Y;
		texButterflyLookupWR_Y = 0;
	}
	if (texButterflyLookupWI_Y) {
		delete [] texButterflyLookupWI_Y;
		texButterflyLookupWI_Y = 0;
	}

	// texture names
	if (texReal1) {
		glDeleteTextures(1, &texReal1);
		texReal1 = 0;
	}
	if (texImag1) {
		glDeleteTextures(1, &texImag1);
		texImag1 = 0;
	}
	if (texReal2) {
		glDeleteTextures(1, &texReal2);
		texReal2 = 0;
	}
	if (texImag2) {
		glDeleteTextures(1, &texImag2);
		texImag2 = 0;
	}

	// Display lists
	if (QList) {
		glDeleteLists(QList, 1);
		QList = 0;
	}
	if (ListX) {
		glDeleteLists(ListX, nButterfliesX);
		ListX = 0;
	}
	if (ListY) {
		glDeleteLists(ListY, nButterfliesY);
		ListY = 0;
	}

	// Cg programs
	DestroyCgPrograms();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::ComputeMaxAndEnergy
*
* DESCRIPTION:
*	Computes maximum of the magnitude of the complex image (imageR, imageI).
*	Also computes the Energy of (imageR, imageI).
*
* FORMAL PARAMETERS:
*   imageR:	A 2D array containing the real part of the input image
*	imageI: A 2D array containing the imaginary part of the input image
*
* RETURNS:
*   Max:	The maximum
*	Energy:	The energy
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::ComputeMaxAndEnergy(float *imageR, float *imageI, float &Max, float &Energy)
{
	float *R   = imageR;
	float *I   = imageI;
	
	Max    = 0.0;
	Energy = 0.0;
	int s, t;
	float val, rval, r, i;
	for (t = 0; t < Height; t++) {
		for (s = 0; s < Width; s++) {
			r = *R;
			i = *I;
			val  = r*r + i*i;
			rval = sqrt(val);

			Energy += val;
			Max = (rval > Max) ? rval : Max;

			R++;
			I++;
		}
	}
	Energy = sqrt(Energy)*2.0;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::SetDisplayMask
*
* DESCRIPTION:
*	Computes the DispMask, InvEnergy and InvMax. The last two will be used
*	in the final display shader to only show the magnitude of one of the two
*	input images.
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
void FFT::SetDisplayMask(void)
{
	int i = 0;
	DispMask[i]  = ShowFirstImage_p; i++;
	DispMask[i]  = ShowFirstImage_p; i++;
	DispMask[i]  = !ShowFirstImage_p; i++;
	DispMask[i]  = !ShowFirstImage_p; i++;

	i = 0;
	InvEnergy[i] = ((float)DispMask[i])/Energy1; i++;
	InvEnergy[i] = ((float)DispMask[i])/Energy1; i++;
	InvEnergy[i] = ((float)DispMask[i])/Energy2; i++;
	InvEnergy[i] = ((float)DispMask[i])/Energy2; i++;

	i = 0;
	InvMax[i] = ((float)DispMask[i])/Max1; i++;
	InvMax[i] = ((float)DispMask[i])/Max1; i++;
	InvMax[i] = ((float)DispMask[i])/Max2; i++;
	InvMax[i] = ((float)DispMask[i])/Max2; i++;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::SetMaxAndEnergy
*
* DESCRIPTION:
*	Sets the energies and max values for the two complex 2D images
*
* FORMAL PARAMETERS:
*	Energy1_:	The energy of complex image 1
*	Max1_:		The maximum value of complex image 1
*	Energy2_:	The energy of complex image 2
*	Max2_:		The maximum value of complex image 2
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::SetMaxAndEnergy(float Energy1_, float Max1_, float Energy2_, float Max2_)
{
	Energy1 = Energy1_;
	Energy2 = Energy2_;
	Max1    = Max1_;
	Max2    = Max2_;

	SetDisplayMask();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::ComputeMaxAndEnergy
*
* DESCRIPTION:
*	Compute and sets the energies and max values for the two complex 2D images
*
* FORMAL PARAMETERS:
*   imageR1:	A 2D array containing the real part of the input image 1
*	imageI1:	A 2D array containing the imaginary part of the input image 1
*   imageR2:	A 2D array containing the real part of the input image 2
*	imageI2:	A 2D array containing the imaginary part of the input image 2
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::ComputeMaxAndEnergy(float *imageR1, float *imageI1, float *imageR2, float *imageI2)
{
	ComputeMaxAndEnergy(imageR1, imageI1, Max1, Energy1);
	ComputeMaxAndEnergy(imageR2, imageI2, Max2, Energy2);

	SetDisplayMask();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::PrintArrray
*
* DESCRIPTION:
*	Prints and array (for debugging)
*
* FORMAL PARAMETERS:
*   N:	The number of elements
*   Ar:	The array
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::PrintArrray(int N, float *Ar)
{
	int i;

	for (i = 0; i < N; i++) {
		cerr << i << " " << Ar[i] << endl;
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::GetBestCutOff
*
* DESCRIPTION:
*	Computes the best transition point from method 1 to method 2.
*
* FORMAL PARAMETERS:
*   N:	The number of elements
*   Ar:	The array
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
int FFT::GetBestCutOff(int N, float *Ar)
{
	int i;
	int index = -1;
	float maxf = -1.0;

	for (i = 0; i < N; i++) {
		if (maxf < Ar[i]) {
			index = i;
			maxf = Ar[i];
		}
	}
	return(index);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::SetGPUVendor
*
* DESCRIPTION:
*	Set the vendor
*
* FORMAL PARAMETERS:
*	none
*
* RETURNS:
*	The GPU vendor
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
Vendor FFT::SetGPUVendor(void)
{
	char *str = strdup((char *)glGetString(GL_VENDOR));
	if (strstr(str, "ATI") != NULL) {
		GPU = GPU_ATI;
	}
	else {
		if (strstr(str, "NVIDIA") != NULL) {
			GPU = GPU_NVIDIA;
		}
		else {
			GPU = GPU_NONE;
		}
	}

	if (GPU == GPU_NVIDIA) {
		cerr << "NVIDIA board found" << endl;
		str = strdup((char *)glGetString(GL_RENDERER));
	}
	else {
		cerr << "Currently only implemented on NVIDIA" << endl;
		exit(0);
	}

	return(GPU);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::DO_FFT_GPU_X
*
* DESCRIPTION:
*	Do the FFT in x only
*
* FORMAL PARAMETERS:
*	FirstTime_p:	true for the very first time only
*   Texs:			an array containig (texR1, texI1, texR2, texI2), where
*	texR1:			texture for the real part of image 1
*	texI1:			texture for the imaginary part of image 1
*	texR2:			texture for the real part of image 2
*	texI2:			texture for the imaginary part of image 2
*
* RETURNS:
*	FirstTime_p
*   Texs
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DO_FFT_GPU_X(bool&FirstTime_p, GLuint *Texs)
{
	int i = 0;

	for (i = 0; i < nButterfliesXWorking; i++) {
		CurrentButterflyStage = i;

		RenderFFTStageX(FirstTime_p,
			texButterflyLookupI_X[i],  
			texButterflyLookupWR_X[i],  
			texButterflyLookupWI_X[i], Texs);

		FirstTime_p = false;
		SetTexs(Texs, texTmpR1, texTmpI1, texTmpR2, texTmpI2);
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::DO_FFT_GPU_Y
*
* DESCRIPTION:
*	Do the FFT in y only
*
* FORMAL PARAMETERS:
*	FirstTime_p:	true for the very first time only
*   Texs:			an array containig (texR1, texI1, texR2, texI2), where
*	texR1:			texture for the real part of image 1
*	texI1:			texture for the imaginary part of image 1
*	texR2:			texture for the real part of image 2
*	texI2:			texture for the imaginary part of image 2
*
* RETURNS:
*	FirstTime_p
*   Texs
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DO_FFT_GPU_Y(bool&FirstTime_p, GLuint *Texs)
{
	int i = 0;

	for (i = 0; i < nButterfliesYWorking; i++) {
		CurrentButterflyStage = i;
		RenderFFTStageY(FirstTime_p,
			texButterflyLookupI_Y[i],  
			texButterflyLookupWR_Y[i],  
			texButterflyLookupWI_Y[i], Texs);

		FirstTime_p = false;
		SetTexs(Texs, texTmpR1, texTmpI1, texTmpR2, texTmpI2);
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::DO_FFT_GPU_Y
*
* DESCRIPTION:
*	Do the FFT in y only
*
* FORMAL PARAMETERS:
*	FirstTime_p:	true for the very first time only
*   Texs:			an array containig (texR1, texI1, texR2, texI2), where
*	texR1:			texture for the real part of image 1
*	texI1:			texture for the imaginary part of image 1
*	texR2:			texture for the real part of image 2
*	texI2:			texture for the imaginary part of image 2
*
* RETURNS:
*	FirstTime_p
*   Texs
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::DO_FFT_GPU(bool&FirstTime_p, GLuint *Texs)
{
	DO_FFT_GPU_X(FirstTime_p, Texs);
	DO_FFT_GPU_Y(FirstTime_p, Texs);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::DoFFT
*
* DESCRIPTION:
*	Do the FFT in x, y or xy
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
void FFT::DoFFT(void)
{
	GLuint Texs[4];
	bool FirstTime_p;

	Buffers->Bind();

	switch (type) {
	case X:
		FirstTime_p = true;
		SetTexs(Texs, texReal1, texImag1, texReal2, texImag2);
		DO_FFT_GPU_X(FirstTime_p, Texs);
		break;
	case Y:
		FirstTime_p = true;
		SetTexs(Texs, texReal1, texImag1, texReal2, texImag2);
		DO_FFT_GPU_Y(FirstTime_p, Texs);
		break;
	case XY:
		FirstTime_p = true;
		SetTexs(Texs, texReal1, texImag1, texReal2, texImag2);
		DO_FFT_GPU(FirstTime_p, Texs);
		break;
	default:
		break;
	}
	Buffers->Unbind();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::ComputeFrameRate
*
* DESCRIPTION:
*	Computes the frame rate.
*
* FORMAL PARAMETERS:
*	Count:		Number of frames processed
*	thisClock:	Current clock value
*	prevClock:	Previous clock value
*
* RETURNS:
*	FrameRate:	The computed frame rate
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
float FFT::ComputeFrameRate(int Count, long thisClock, long prevClock)
{
	int NumSimultanuesComplex2DImages = 2;

	float duration  = (double)(thisClock - prevClock)/ CLOCKS_PER_SEC;
	float FrameRate = ((float)Count*NumSimultanuesComplex2DImages)/duration;
	return(FrameRate);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::FindOptimalTransitionPoints
*
* DESCRIPTION:
*	Vary the xCutOff and yCutOff sequentially for all possible combinations
*	and picks the best values for the fastest frame rate. xCutOff is the
*	transition point between doing FFT in x using method 1 and method 2. 
*	Similarly, yCutOff.
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
void FFT::FindOptimalTransitionPoints(void)
{
	long thisClock, startClock;

	int numTimes        = 0;
	int avgCount        = 100;
	float FrameRate     = 0.0;
	float PrevFrameRate = 0.0;
	bool FirstTimeX_p   = true;
	bool DoneX_p        = false;
	bool DoneAdjX_p     = false;
	bool FirstTimeY_p   = true;
	bool DoneY_p        = false;
	bool DoneAdjY_p     = false;

	startClock = clock();

	while (!DoneAdjX_p || !DoneAdjY_p) {
		if (FirstTimeX_p && (numTimes == 0)) {
			cerr << "*** Optimizing Frame Rate ***" << endl;
			cerr << "Do not touch anything until finished" << endl;
			cerr << "Will be done in a few minutes" << endl;
			cerr << endl;
		}

		if (numTimes == avgCount) {
			thisClock = clock();
			FrameRate = ComputeFrameRate(avgCount, thisClock, startClock);
			startClock = thisClock;
			numTimes = 0;

			if (FirstTimeX_p) {
				cerr << "Detecting best X frame rate...";
				cerr << "do not touch anything..." << endl;
				xCutOff = 0;
			}
			else {
				if (!DoneX_p) {
					cerr << xCutOff << " " << FrameRate << endl;
					xFrameRate[xCutOff] = FrameRate;
					xCutOff++;
					DoneX_p = (xCutOff >= nButterfliesX) || (PrevFrameRate > FrameRate);
					PrevFrameRate = FrameRate;
				}
			}
			if (DoneX_p) {
				if (!DoneAdjX_p) {
					xCutOff = GetBestCutOff(nButterfliesX, xFrameRate);
					cerr << "Best xCutOff: " << xCutOff << endl;
					cerr << "Done X" << endl;
					DoneAdjX_p = true;
				}
			}
			FirstTimeX_p = false;

			if (DoneAdjX_p) {
				if (FirstTimeY_p) {
					cerr << "Detecting best Y frame rate...";
					cerr << "do not touch anything..." << endl;
					yCutOff = 0;
					PrevFrameRate = 0.0;
				}
				else {
					if (!DoneY_p) {
						cerr << yCutOff << " " << FrameRate << endl;
						yFrameRate[yCutOff] = FrameRate;
						yCutOff++;
						DoneY_p = (yCutOff >= nButterfliesY) || (PrevFrameRate > FrameRate);
						PrevFrameRate = FrameRate;
					}
				}
				if (DoneY_p) {
					if (!DoneAdjY_p) {
						yCutOff = GetBestCutOff(nButterfliesY, yFrameRate);
						cerr << "Best yCutOff: " << yCutOff << endl;
						cerr << "Done Y" << endl;
						cerr << "*** DONE Optimizing Frame Rate ***" << endl;
						DoneAdjY_p = true;
					}
				}
				FirstTimeY_p = false;
			}

			if (DoneAdjX_p && DoneAdjY_p) {
				cerr << "Optimal Frame Rate = " << FrameRate << endl;
			}
		}
		DoFFT();
		numTimes++;
	}
	cerr << endl;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*   FFT::Print
*
* DESCRIPTION:
*	Prints a string to the display window.
*	Usage: Print(X, Y, "My string", GLUT_BITMAP_HELVETICA_12);
*
* FORMAL PARAMETERS:
*	X:		x coordinate
*	Y:		y coordinate
*	str:	the string to print
*	font:	font to use (e.g.: GLUT_BITMAP_HELVETICA_12)
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::Print(float X, float Y, char *str, void *font)
{
    int i; 
    int fontWidth = glutBitmapWidth(font, 'A'); 
    
	glRasterPos2f(X, Y); 
    for(i = 0; str[i] != '\0'; i++) 
        glutBitmapCharacter(font, str[i]);
} 

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::SetTexs
*
* DESCRIPTION:
*	Sets Texs with 4 texture names.
*
* FORMAL PARAMETERS:
*	texR1, texI1,texR2, texI2: the four texture names
*	Texs: an array
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::SetTexs(GLuint* Texs, GLuint texR1, GLuint texI1, GLuint texR2, GLuint texI2)
{
	int i = 0;
	Texs[i] = texR1; i++;
	Texs[i] = texI1; i++;
	Texs[i] = texR2; i++;
	Texs[i] = texI2; i++;
}
