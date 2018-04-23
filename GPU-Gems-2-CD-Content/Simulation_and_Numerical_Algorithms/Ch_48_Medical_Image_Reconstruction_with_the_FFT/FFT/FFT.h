/***************************************************************************
*        FILE NAME:  FFT.h
*
* ONE LINE SUMMARY:
*        FFT class is declared in this file, along with some misc. stuff
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
#ifndef __FFT__
#define __FFT__

const double M_PI = 3.1415926535897932384626433832795029;

enum FftType {X, Y, XY};

#include <PBuffer.h>
#include <GL/glut.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#include "PingPong.h"

/*
C L A S S
*/
class FFT {
public:
    // member functions
	FFT::FFT(void);
	FFT::FFT(bool ForwardFFT_p_, PingMethod PMethod, int Width_, int Height_);
	~FFT(void);

	// Housekeeping
	Vendor FFT::SetGPUVendor(void);
	void FFT::SetDisplayMask(void);
	void FFT::PrintArrray(int N, float *Ar);
	int FFT::GetBestCutOff(int N, float *Ar);
	void FFT::FindOptimalTransitionPoints(void);
	float FFT::ComputeFrameRate(int Count, long thisClock, long prevClock);

	// Initializations
	void FFT::InitTextures(void);
	
	// Data handing
	void FFT::UploadData(float *imageR1, float *imageI1, float *imageR2, float *imageI2);
	void FFT::ComputeMaxAndEnergy(float *imageR, float *imageI, float &Max, float &Energy);
	void FFT::ComputeMaxAndEnergy(float *imageR1, float *imageI1, float *imageR2, float *imageI2);
	void FFT::SetMaxAndEnergy(float Energy1_, float Max1_, float Energy2_, float Max2_);
	
	// Cg stuff
	void FFT::InitCg(void);
	void FFT::WriteObject(char *ofilename, CGprogram oProgram);
	void FFT::InitFFTFragmentCgX1(void);
	void FFT::InitFFTFragmentCgX2(void);
	void FFT::InitFFTFragmentCgY1(void);
	void FFT::InitFFTFragmentCgY2(void);
	void FFT::InitDispFragmentCg(void);
	void FFT::EnableDispFragmentProgram(void);
	void FFT::DisableDispFragmentProgram(void);
	void FFT::EnableFFTFragmentProgramX(void);
	void FFT::DisableFFTFragmentProgramX(void);
	void FFT::EnableFFTFragmentProgramY(void);
	void FFT::DisableFFTFragmentProgramY(void);
	void FFT::CheckCgError(void);
	void FFT::InitCgPrograms(void);
	void FFT::DestroyCgPrograms(void);

	// Computing various tables
	void FFT::ComputeWeight(int N, int k, float &Wr, float &Wi);
	int  FFT::BitReverse(int i, int N);
	void FFT::CreateButterflyLookups(float *butterflylookupI,
								 float *butterflylookupWR, 
								 float *butterflylookupWI, 
								 int NButterflies, int N);
	void FFT::ComputeVerticesTexCoordsX(int Stage, float *Vs, float *Ts0, float *Ts1, int &Ns);
	void FFT::ComputeVerticesTexCoordsY(int Stage, float *Vs, float *Ts0, float *Ts1, int &Ns);

	// Rendering
	void FFT::Print(float X, float Y, char *str, void *font);
	void FFT::SetTexs(GLuint* Texs, GLuint texR1, GLuint texI1, GLuint texR2, GLuint texI2);
	void FFT::CopyFloatBuffersToScreen(bool FromInput_p, int WinWidth, int WinHeight, GLuint* Texs);
	void FFT::RenderFFTStageX(bool FirstTime_p,
						  int texButterflyLookupI,  
						  int texButterflyLookupWR,  
						  int texButterflyLookupWI, 
						  GLuint *Texs);
	void FFT::RenderFFTStageY(bool FirstTime_p,
						  int texButterflyLookupI,  
						  int texButterflyLookupWR,  
						  int texButterflyLookupWI,
						  GLuint *Texs);
	void FFT::DisplayInputImage(int WinWidth, int WinHeight);
	void FFT::DisplayOutputImage(int WinWidth, int WinHeight);
	void FFT::DrawQuadForFFT(void);
	void FFT::DrawQuadTilesForFFT_X(void);
	void FFT::DrawQuadTilesForFFT_Y(void);
	void FFT::GenDisplayListsX(void);
	void FFT::GenDisplayListsY(void);
	void FFT::GenQuadDisplayList(void);
	void FFT::DrawQuadTilesForFFT_X_List(void);
	void FFT::DrawQuadTilesForFFT_Y_List(void);
	void FFT::DrawQuadForFFT_List(void);
	void FFT::DO_FFT_GPU(bool &FirstTime_p, GLuint *Texs);
	void FFT::DO_FFT_GPU_X(bool &FirstTime_p, GLuint *Texs);
	void FFT::DO_FFT_GPU_Y(bool &FirstTime_p, GLuint *Texs);
	void FFT::DoFFT(void);

	Vendor GPU;									// GPU manufacturer
	FftType type;								// X-only (1DFFT), y-only (1DFFT) or XY (2DFFT)

	int Width;									// width and height of the image
	int Height;
	int nButterfliesX;							// number of butterfly stages
	int nButterfliesY;
	int nButterfliesXWorking;					// number of butterfly stages used (for debugging can use less)
	int nButterfliesYWorking;
	int CurrentButterflyStage;					// Current butterfly stage
	float InvEnergy[4];							// Inverse of the energies of the first and second input complex images
	float Energy1;								// Energy of the first input complex image
	float Energy2;								// Energy of the second input complex image
	float InvMax[4];							// Inverse of the maxima of the magnitudes of the first and second input complex images
	float Max1;									// Maximum of the magnitude of the first input complex image
	float Max2;									// Maximum of the magnitude of the second input complex image
	float DispMask[4];							// Display mask to display first or second image
	bool ShowFirstImage_p;						// true if displaying the first image, false if second image.
	bool ForwardFFT_p;							// true if forward FFT, false if inverse FFT

	PingPong *Buffers;

	int xCutOff;								// Transition x butterfly stage where we go from method 1 to method 2
	int yCutOff;								// Transition y butterfly stage where  we go from method 1 to method 2

	float *xFrameRate;							// Measured x frame rate for each value of xCutOff
	float *yFrameRate;							// Measured y frame rate for each value of yCutOff

	GLuint QList;								// Display list for doing method 1
	GLuint ListX;								// Display list for each x butterfly stage when doing method 2
	GLuint ListY;								// Display list for each y butterfly stage when doing method 2

	float *butterflyLookupI_X;					// X: Lookup table containing scrambled coordinates
	float *butterflyLookupWR_X;					// X: Lookup table containing weights, real part.
	float *butterflyLookupWI_X;					// X: Lookup table containing weights, imaginary part.

	float *butterflyLookupI_Y;					// Y: Lookup table containing scrambled coordinates
	float *butterflyLookupWR_Y;					// Y: Lookup table containing weights, real part.
	float *butterflyLookupWI_Y;					// Y: Lookup table containing weights, imaginary part.

	GLuint texReal1;							// Input image 1: real part
	GLuint texImag1;							// Input image 1: imaginary part
	GLuint texReal2;							// Input image 2: real part
	GLuint texImag2;							// Input image 2: imaginary part

	GLuint texTmpR1;							// Image 1: temporary buffer, real part 
	GLuint texTmpI1;							// Image 1: temporary buffer, imaginary part 
	GLuint texTmpR2;							// Image 2: temporary buffer, real part 
	GLuint texTmpI2;							// Image 2: temporary buffer, imaginary part 

	GLuint *texButterflyLookupI_X;				// X: Texture names for scrambled coordinates
	GLuint *texButterflyLookupWR_X;				// X: Texture names for weights, real part.
	GLuint *texButterflyLookupWI_X;				// X: Texture names for weights, imaginary part.

	GLuint *texButterflyLookupI_Y;				// Y: Texture names for scrambled coordinates
	GLuint *texButterflyLookupWR_Y;				// Y: Texture names for weights, real part.
	GLuint *texButterflyLookupWI_Y;				// Y: Texture names for weights, imaginary part.

	CGcontext Context;							// Cg context

	CGprogram xfProgram1;						// Parameters for method 1, shader for X FFT
	CGparameter xBaseTexR1_1;
	CGparameter xBaseTexI1_1;
	CGparameter xBaseTexR2_1;
	CGparameter xBaseTexI2_1;
	CGparameter xButterflyLookupI_1;
	CGparameter xButterflyLookupWR_1;
	CGparameter xButterflyLookupWI_1;
	
	CGprogram xfProgram2;						// Parameters for method 2, shader for X FFT
	CGparameter xBaseTexR1_2;
	CGparameter xBaseTexI1_2;
	CGparameter xBaseTexR2_2;
	CGparameter xBaseTexI2_2;
	CGprofile xfProfile;
	
	CGprogram yfProgram1;						// Parameters for method 1, shader for Y FFT
	CGparameter yBaseTexR1_1;
	CGparameter yBaseTexI1_1;
	CGparameter yBaseTexR2_1;
	CGparameter yBaseTexI2_1;
	CGparameter yButterflyLookupI_1;
	CGparameter yButterflyLookupWR_1;
	CGparameter yButterflyLookupWI_1;
	
	CGprogram yfProgram2;						// Parameters for method 2, shader for Y FFT
	CGparameter yBaseTexR1_2;
	CGparameter yBaseTexI1_2;
	CGparameter yBaseTexR2_2;
	CGparameter yBaseTexI2_2;
	CGprofile yfProfile;

	CGprogram dfProgram;						// Shader parameters for display shader
	CGparameter DispTexR1;
	CGparameter DispTexI1;
	CGparameter DispTexR2;
	CGparameter DispTexI2;
	CGparameter DispInvEnergy;
	CGprofile dfProfile;
	
  private:
};


#endif  /* __FFT__ */

