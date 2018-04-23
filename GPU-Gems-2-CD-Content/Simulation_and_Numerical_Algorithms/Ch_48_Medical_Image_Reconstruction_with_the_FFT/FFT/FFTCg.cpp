/***************************************************************************
*        FILE NAME:  FFTCg.cpp
*
* ONE LINE SUMMARY:
*        This file contains Cg-related member functions for the FFT class.
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
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "FFT.h"

// The shaders
#include "FFTfragmentX_CG1.h"
#include "FFTfragmentX_CG2.h"
#include "FFTfragmentY_CG1.h"
#include "FFTfragmentY_CG2.h"
#include "DispFragment_CG.h"

using namespace std;

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitCg
*
* DESCRIPTION:
*	Initialize Cg variables
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
void FFT::InitCg(void)
{	
	Context = NULL;
	dfProgram = NULL;
	DispTexR1 = NULL;
	DispTexI1 = NULL;
	DispTexR2 = NULL;
	DispTexI2 = NULL;
	DispInvEnergy = NULL;
	dfProfile = CG_PROFILE_FP40;

	xfProgram1 = NULL;
	xBaseTexR1_1 = NULL;
	xBaseTexI1_1 = NULL;
	xBaseTexR2_1 = NULL;
	xBaseTexI2_1 = NULL;
	xButterflyLookupI_1 = NULL;
	xButterflyLookupWR_1 = NULL;
	xButterflyLookupWI_1 = NULL;
	xfProgram2 = NULL;
	xBaseTexR1_2 = NULL;
	xBaseTexI1_2 = NULL;
	xBaseTexR2_2 = NULL;
	xBaseTexI2_2 = NULL;
	xfProfile = CG_PROFILE_FP40;
	
	yfProgram1 = NULL;
	yBaseTexR1_1 = NULL;
	yBaseTexI1_1 = NULL;
	yBaseTexR2_1 = NULL;
	yBaseTexI2_1 = NULL;
	yButterflyLookupI_1 = NULL;
	yButterflyLookupWR_1 = NULL;
	yButterflyLookupWI_1 = NULL;
	
	yfProgram2 = NULL;
	yBaseTexR1_2 = NULL;
	yBaseTexI1_2 = NULL;
	yBaseTexR2_2 = NULL;
	yBaseTexI2_2 = NULL;
	yfProfile = CG_PROFILE_FP40;
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::WriteObject
*
* DESCRIPTION:
*	Writes the object file of a shader into a file
*
* FORMAL PARAMETERS:
*	ofilename:	file name
*	oProgram:	Cg program
*
* RETURNS:
*	none
*
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
**************************[MAN-END]*****************************************/
void FFT::WriteObject(char *ofilename, CGprogram oProgram)
{
	ofstream ofile;
	ofile.open(ofilename, ios::out);
	ofile << cgGetProgramString(oProgram, CG_COMPILED_PROGRAM) << endl;
	ofile.close();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitFFTFragmentCgX1
*
* DESCRIPTION:
*	Initializes the fragment shader for FFT in x for method 1
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
void FFT::InitFFTFragmentCgX1(void)
{
	//cerr << "*** Creating fragment program for X ***" << endl;
	xfProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(xfProfile);
	
	xfProgram1 = cgCreateProgram(Context, CG_SOURCE, FFTfragmentX_CG1, xfProfile, "FragmentProgram", 0);
	CheckCgError();

	/*
	cerr << "----PROGRAM BEGIN----" << endl;
	cerr << cgGetProgramString(xfProgram1, CG_COMPILED_PROGRAM) << endl;
	cerr << "----PROGRAM END  ----" << endl;
	*/

	//char file[200] = "FFTfragmentX_CG1.ocg";
	//WriteObject(file, xfProgram);

	if (xfProgram1 != NULL) {
		cgGLLoadProgram(xfProgram1);
		CheckCgError();
		xBaseTexR1_1 = cgGetNamedParameter(xfProgram1, "Real1");
		CheckCgError();
		xBaseTexI1_1 = cgGetNamedParameter(xfProgram1, "Imag1");
		CheckCgError();
		xBaseTexR2_1 = cgGetNamedParameter(xfProgram1, "Real2");
		CheckCgError();
		xBaseTexI2_1 = cgGetNamedParameter(xfProgram1, "Imag2");
		CheckCgError();
		xButterflyLookupI_1 = cgGetNamedParameter(xfProgram1, "ButterflyLookupI");
		CheckCgError();
		xButterflyLookupWR_1 = cgGetNamedParameter(xfProgram1, "ButterflyLookupWR");
		CheckCgError();
		xButterflyLookupWI_1 = cgGetNamedParameter(xfProgram1, "ButterflyLookupWI");
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitFFTFragmentCgX2
*
* DESCRIPTION:
*	Initializes the fragment shader for FFT in x for method 2
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
void FFT::InitFFTFragmentCgX2(void)
{
	//cerr << "*** Creating fragment program for X ***" << endl;
	xfProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(xfProfile);

	xfProgram2 = cgCreateProgram(Context, CG_SOURCE, FFTfragmentX_CG2, xfProfile, "FragmentProgram", 0);
	CheckCgError();

	/*
	cerr << "----PROGRAM BEGIN----" << endl;
	cerr << cgGetProgramString(xfProgram2, CG_COMPILED_PROGRAM) << endl;
	cerr << "----PROGRAM END  ----" << endl;
	*/

	//char file[200] = "FFTfragmentX_CG2.ocg";
	//WriteObject(file, xfProgram2);

	if (xfProgram2 != NULL) {
		cgGLLoadProgram(xfProgram2);
		CheckCgError();
		xBaseTexR1_2 = cgGetNamedParameter(xfProgram2, "Real1");
		CheckCgError();
		xBaseTexI1_2 = cgGetNamedParameter(xfProgram2, "Imag1");
		CheckCgError();
		xBaseTexR2_2 = cgGetNamedParameter(xfProgram2, "Real2");
		CheckCgError();
		xBaseTexI2_2 = cgGetNamedParameter(xfProgram2, "Imag2");
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitFFTFragmentCgY1
*
* DESCRIPTION:
*	Initializes the fragment shader for FFT in y for method 1
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
void FFT::InitFFTFragmentCgY1(void)
{
	//cerr << "*** Creating fragment program for Y ***" << endl;
	yfProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(yfProfile);
	
	yfProgram1 = cgCreateProgram(Context, CG_SOURCE, FFTfragmentY_CG1, yfProfile, "FragmentProgram", 0);
	CheckCgError();
	
	/*
	cerr << "----PROGRAM BEGIN----" << endl;
	cerr << cgGetProgramString(yfProgram1, CG_COMPILED_PROGRAM) << endl;
	cerr << "----PROGRAM END  ----" << endl;
	*/

	//char file[200] = "FFTfragmentY_CG1.ocg";
	//WriteObject(file, yfProgram);
	
	if (yfProgram1 != NULL) {
		cgGLLoadProgram(yfProgram1);
		CheckCgError();
		yBaseTexR1_1 = cgGetNamedParameter(yfProgram1, "Real1");
		CheckCgError();
		yBaseTexI1_1 = cgGetNamedParameter(yfProgram1, "Imag1");
		CheckCgError();
		yBaseTexR2_1 = cgGetNamedParameter(yfProgram1, "Real2");
		CheckCgError();
		yBaseTexI2_1 = cgGetNamedParameter(yfProgram1, "Imag2");
		CheckCgError();
		yButterflyLookupI_1 = cgGetNamedParameter(yfProgram1, "ButterflyLookupI");
		CheckCgError();
		yButterflyLookupWR_1 = cgGetNamedParameter(yfProgram1, "ButterflyLookupWR");
		CheckCgError();
		yButterflyLookupWI_1 = cgGetNamedParameter(yfProgram1, "ButterflyLookupWI");
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitFFTFragmentCgY2
*
* DESCRIPTION:
*	Initializes the fragment shader for FFT in y for method 2
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
void FFT::InitFFTFragmentCgY2(void)
{
	//cerr << "*** Creating fragment program for Y ***" << endl;
	yfProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(yfProfile);
	
	yfProgram2 = cgCreateProgram(Context, CG_SOURCE, FFTfragmentY_CG2, yfProfile, "FragmentProgram", 0);
	CheckCgError();
	
	/*
	cerr << "----PROGRAM BEGIN----" << endl;
	cerr << cgGetProgramString(yfProgram2, CG_COMPILED_PROGRAM) << endl;
	cerr << "----PROGRAM END  ----" << endl;
	*/

	//char file[200] = "FFTfragmentY_CG2.ocg";
	//WriteObject(file, yfProgram2);
	
	if (yfProgram2 != NULL) {
		cgGLLoadProgram(yfProgram2);
		CheckCgError();
		yBaseTexR1_2 = cgGetNamedParameter(yfProgram2, "Real1");
		CheckCgError();
		yBaseTexI1_2 = cgGetNamedParameter(yfProgram2, "Imag1");
		CheckCgError();
		yBaseTexR2_2 = cgGetNamedParameter(yfProgram2, "Real2");
		CheckCgError();
		yBaseTexI2_2 = cgGetNamedParameter(yfProgram2, "Imag2");
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitDispFragmentCg
*
* DESCRIPTION:
*	Initializes the fragment shader displaying the result.
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
void FFT::InitDispFragmentCg(void)
{
	//cerr << "*** Creating fragment program for display ***" << endl;
	dfProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(dfProfile);
	
	dfProgram = cgCreateProgram(Context, CG_SOURCE, DispFragment_CG, dfProfile, "FragmentProgram", 0);
	CheckCgError();
	
	/*
	cerr << "----PROGRAM BEGIN----" << endl;
	cerr << cgGetProgramString(dfProgram, CG_COMPILED_PROGRAM) << endl;
	cerr << "----PROGRAM END  ----" << endl;
	*/

	//char file[200] = "Dispfragment_CG.ocg";
	//WriteObject(file, dfProgram);

	if (dfProgram != NULL) {
		cgGLLoadProgram(dfProgram);
		CheckCgError();
		DispTexR1 = cgGetNamedParameter(dfProgram, "Real1");
		CheckCgError();
		DispTexI1 = cgGetNamedParameter(dfProgram, "Imag1");
		CheckCgError();
		DispTexR2 = cgGetNamedParameter(dfProgram, "Real2");
		CheckCgError();
		DispTexI2 = cgGetNamedParameter(dfProgram, "Imag2");
		CheckCgError();
		DispInvEnergy = cgGetNamedParameter(dfProgram, "InvEnergy");
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::CheckCgError
*
* DESCRIPTION:
*	Check for Cg errors
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
void FFT::CheckCgError(void)
{
	CGerror err = cgGetError();
	
	if (err != CG_NO_ERROR) {
		cerr << "CG error: " << cgGetErrorString(err) << endl;
		exit(1);
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::EnableFFTFragmentProgramX
*
* DESCRIPTION:
*	Enable all fragment programs for x FFT depending on the method
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
void FFT::EnableFFTFragmentProgramX(void)
{
	cgGLEnableProfile(xfProfile);
	CheckCgError();	
	if (CurrentButterflyStage < xCutOff) {
		cgGLBindProgram(xfProgram1);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexR1_1);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexI1_1);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexR2_1);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexI2_1);
		CheckCgError();
		cgGLEnableTextureParameter(xButterflyLookupI_1);
		CheckCgError();
		cgGLEnableTextureParameter(xButterflyLookupWR_1);
		CheckCgError();
		cgGLEnableTextureParameter(xButterflyLookupWI_1);
		CheckCgError();
	}
	else {
		cgGLBindProgram(xfProgram2);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexR1_2);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexI1_2);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexR2_2);
		CheckCgError();
		cgGLEnableTextureParameter(xBaseTexI2_2);
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DisableFFTFragmentProgramX
*
* DESCRIPTION:
*	Disables all fragment programs for x FFT depending on the method
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
void FFT::DisableFFTFragmentProgramX(void)
{
	if (CurrentButterflyStage < xCutOff) {
		cgGLDisableTextureParameter(xBaseTexR1_1);
		CheckCgError();
		cgGLDisableTextureParameter(xBaseTexI1_1);
		CheckCgError();
		cgGLDisableTextureParameter(xBaseTexR2_1);
		CheckCgError();
		cgGLDisableTextureParameter(xBaseTexI2_1);
		CheckCgError();
		cgGLDisableTextureParameter(xButterflyLookupI_1);
		CheckCgError();
		cgGLDisableTextureParameter(xButterflyLookupWR_1);
		CheckCgError();
		cgGLDisableTextureParameter(xButterflyLookupWI_1);
		CheckCgError();
	}
	else {
		cgGLDisableTextureParameter(xBaseTexR1_2);
		CheckCgError();
		cgGLDisableTextureParameter(xBaseTexI1_2);
		CheckCgError();
		cgGLDisableTextureParameter(xBaseTexR2_2);
		CheckCgError();
		cgGLDisableTextureParameter(xBaseTexI2_2);
		CheckCgError();
	}
	cgGLDisableProfile(xfProfile);
	CheckCgError();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::InitCgPrograms
*
* DESCRIPTION:
*	Initialize all shaders
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
void FFT::InitCgPrograms(void)
{
	Context = cgCreateContext();
	CheckCgError();
	InitFFTFragmentCgX1();
	InitFFTFragmentCgX2();
	InitFFTFragmentCgY1();
	InitFFTFragmentCgY2();
	InitDispFragmentCg();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DestroyCgPrograms
*
* DESCRIPTION:
*	Destroy all shaders
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
void FFT::DestroyCgPrograms(void)
{
	cgDestroyProgram(xfProgram2);
	cgDestroyProgram(yfProgram2);
	cgDestroyProgram(xfProgram1);
	cgDestroyProgram(yfProgram1);
	cgDestroyProgram(dfProgram);
	cgDestroyContext(Context);
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::EnableFFTFragmentProgramY
*
* DESCRIPTION:
*	Enable all fragment programs for y FFT depending on the method
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
void FFT::EnableFFTFragmentProgramY(void)
{
	cgGLEnableProfile(yfProfile);
	CheckCgError();	
	if (CurrentButterflyStage < yCutOff) {
		cgGLBindProgram(yfProgram1);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexR1_1);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexI1_1);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexR2_1);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexI2_1);
		CheckCgError();
		cgGLEnableTextureParameter(yButterflyLookupI_1);
		CheckCgError();
		cgGLEnableTextureParameter(yButterflyLookupWR_1);
		CheckCgError();
		cgGLEnableTextureParameter(yButterflyLookupWI_1);
		CheckCgError();
	}
	else {
		cgGLBindProgram(yfProgram2);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexR1_2);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexI1_2);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexR2_2);
		CheckCgError();
		cgGLEnableTextureParameter(yBaseTexI2_2);
		CheckCgError();
	}
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DisableFFTFragmentProgramY
*
* DESCRIPTION:
*	Disables all fragment programs for y FFT depending on the method
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
void FFT::DisableFFTFragmentProgramY(void)
{
	if (CurrentButterflyStage < yCutOff) {
		cgGLDisableTextureParameter(yBaseTexR1_1);
		CheckCgError();
		cgGLDisableTextureParameter(yBaseTexI1_1);
		CheckCgError();
		cgGLDisableTextureParameter(yBaseTexR2_1);
		CheckCgError();
		cgGLDisableTextureParameter(yBaseTexI2_1);
		CheckCgError();
		cgGLDisableTextureParameter(yButterflyLookupI_1);
		CheckCgError();
		cgGLDisableTextureParameter(yButterflyLookupWR_1);
		CheckCgError();
		cgGLDisableTextureParameter(yButterflyLookupWI_1);
		CheckCgError();
	}
	else {
		cgGLDisableTextureParameter(yBaseTexR1_2);
		CheckCgError();
		cgGLDisableTextureParameter(yBaseTexI1_2);
		CheckCgError();
		cgGLDisableTextureParameter(yBaseTexR2_2);
		CheckCgError();
		cgGLDisableTextureParameter(yBaseTexI2_2);
		CheckCgError();
	}
	cgGLDisableProfile(yfProfile);
	CheckCgError();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::EnableDispFragmentProgram
*
* DESCRIPTION:
*	Enables the fragment program for displaying the result.
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
void FFT::EnableDispFragmentProgram(void)
{
	cgGLEnableProfile(dfProfile);
	CheckCgError();	
	cgGLBindProgram(dfProgram);
	CheckCgError();
	cgGLEnableTextureParameter(DispTexR1);
	CheckCgError();
	cgGLEnableTextureParameter(DispTexI1);
	CheckCgError();
	cgGLEnableTextureParameter(DispTexR2);
	CheckCgError();
	cgGLEnableTextureParameter(DispTexI2);
	CheckCgError();
}

/*************************[MAN-BEG]*****************************************
*
* NAME:
*	FFT::DisableDispFragmentProgram
*
* DESCRIPTION:
*	Disables the fragment program for displaying the result.
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
void FFT::DisableDispFragmentProgram(void)
{
	cgGLDisableTextureParameter(DispTexR1);
	CheckCgError();
	cgGLDisableTextureParameter(DispTexI1);
	CheckCgError();
	cgGLDisableTextureParameter(DispTexR2);
	CheckCgError();
	cgGLDisableTextureParameter(DispTexI2);
	CheckCgError();
	cgGLDisableProfile(dfProfile);
	CheckCgError();
}




