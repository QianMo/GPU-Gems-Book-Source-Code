/***************************************************************************
*		FILE NAME:  FFTDemo.cpp
*
* ONE LINE SUMMARY:
*		Demostrates 2DFFT in a GPU.
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
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <FFT.h>

using namespace std;

int WinWidth  = 0;
int WinHeight = 0;
int Width     = 0;
int Height    = 0;

char DisplayOriginal_p = 0;
char Debug_p = 0;

float *imageR = 0;
float *imageI = 0;

FFT *Fft;

// Uncomment the lines below to load different data sizes.
// Each file contains a sync function
//
//char *InputFile = "../bin/Data/SyncFunction_8x8.dat";
//char *InputFile = "../bin/Data/SyncFunction_16x16.dat";
//char *InputFile = "../bin/Data/SyncFunction_16x32.dat";
//char *InputFile = "../bin/Data/SyncFunction_32x32.dat";
//char *InputFile = "../bin/Data/SyncFunction_64x64.dat";
//char *InputFile = "../bin/Data/SyncFunction_128x128.dat";
//char *InputFile = "../bin/Data/SyncFunction_256x256.dat";
//char *InputFile = "../bin/Data/SyncFunction_512x512.dat";
char *InputFile = "../bin/Data/SyncFunction_2048x64.dat";
//char *InputFile = "../bin/Data/SyncFunction_2048x128.dat";


// File format:
//
// Width: an integer containing the value
// Height: an integer containing the value
// A stream of 8 bit data with the fastest moving index along width
void ReadData(char *file)
{
	unsigned char *image;

	ifstream ifile;
	ifile.open(file, ios::out | ios_base::binary);
	if (ifile == NULL) {
		cerr << "Could not open the file: " << file << endl;
		cerr << "Exiting." << endl;
		exit(0);
	}
	ifile.read((char *)&Width, sizeof(int));
	ifile.read((char *)&Height, sizeof(int));
	image = new unsigned char [Width*Height];
	ifile.read((char *)image, Width*Height);
	ifile.close();

	// Allocate real and imaginary buffers
	float *R   = imageR = new float [Width*Height];
	float *I   = imageI = new float [Width*Height];
	
	// Fill the real and imaginary buffers with data
	int s, t;
	float val;
	unsigned char *ptr = image;
	for (t = 0; t < Height; t++) {
		for (s = 0; s < Width; s++) {
			val = *ptr;
			*(R++) = val;
			*(I++) = 0;
			ptr++;
		}
	}
	delete [] image;
}

static void Reshape(int w, int h)
{
	WinWidth  = glutGet(GLUT_WINDOW_WIDTH);
	WinHeight = glutGet(GLUT_WINDOW_HEIGHT);

	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, Fft->Width, 0.0, Fft->Height, -1, +1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

static void Display(void)
{
	if (DisplayOriginal_p) {
		Fft->DisplayInputImage(WinWidth, WinHeight);
	}
	else {
		Fft->DoFFT();
		Fft->DisplayOutputImage(WinWidth, WinHeight);
	}
	glutSwapBuffers();
}

static void Idle(void)
{
	static clock_t thisClock, startClock = clock();
	static int numTimes     = 0;
	int avgCount            = 100;
	float FrameRate         = 0.0;
	static bool FirstTime_p = true;

	if (FirstTime_p) {
		// Draw a blue quad to look at.
		Reshape(WinWidth, WinHeight);
		glColor3f(0.0, 0.0, 1.0);
		glBegin(GL_QUADS);
		glVertex2f(0.0, 0.0);
		glVertex2f(Width, 0.0);
		glVertex2f(Width, Height);
		glVertex2f(0.0, Height);
		glEnd();
		glColor3f(1.0, 1.0, 1.0);
		Fft->Print(10, 15, "Optimizing xCutOff and yCutOff. Wait...", GLUT_BITMAP_HELVETICA_12);
		glutSwapBuffers();
		Fft->FindOptimalTransitionPoints();
		FirstTime_p = false;
	}

	if (numTimes == avgCount) {
		thisClock = clock();
		FrameRate = Fft->ComputeFrameRate(avgCount, thisClock, startClock);
		startClock = thisClock;
		numTimes = 0;
		cerr << "Frame Rate = " << FrameRate << endl;
	}

	Display();
	numTimes++;
}

void PrintHelp(void)
{
	cerr << endl;
	cerr << "Demonstrates 2D FFT using the GPU" << endl;
	cerr << "      Thilaka Sumanaweera, 12/15/2004" << endl;
	cerr << "      Siemens Medical Solutions USA, Inc." << endl;
	cerr << "      Mountain View, CA, USA" << endl;
	cerr << endl;
	cerr << " ESC:   quit" << endl;
	cerr << "   1:   toggles showing 1st image or the 2nd image" << endl;
	cerr << "   x:   do FFT in x" << endl;
	cerr << "   y:   do FFT in y" << endl;
	cerr << "   z:   do FFT in x and y" << endl;
	cerr << "   d:   debug mode. Run only one butterfly stage." << endl;
	cerr << "   f:   toggle displaying output or input data" << endl;
	cerr << "   h:   Print this help." << endl;
	cerr << endl;
}

static void Keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case '1':
		Fft->ShowFirstImage_p = !Fft->ShowFirstImage_p;
		Fft->SetDisplayMask();
		if (Fft->ShowFirstImage_p) {
			cerr << "Show first image." << endl;
		}
		else {
			cerr << "Show second image." << endl;
		}
		break;
	case 'f':
		DisplayOriginal_p = !DisplayOriginal_p;
		if (DisplayOriginal_p) {
			cerr << "Showing input." << endl;
		}
		else {
			cerr << "Showing output." << endl;
		}
		break;
	case 'x':
		Fft->type = X;
		cerr << "1DFFT in x" << endl;
		break;
	case 'y':
		Fft->type = Y;
		cerr << "1DFFT in y" << endl;
		break;
	case 'z':
		Fft->type = XY;
		cerr << "2DFFT" << endl;
		break;
	case 'd':
		Debug_p = !Debug_p;
		if (Debug_p) {	
			Fft->nButterfliesXWorking = 1;
			Fft->nButterfliesYWorking = 1;
			cerr << "Debug" << endl;
		}
		else {
			Fft->nButterfliesXWorking = Fft->nButterfliesX;	
			Fft->nButterfliesYWorking = Fft->nButterfliesY;
			cerr << "Not debug" << endl;
		}
		break;
	case 'h':
		PrintHelp();
		break;
	case 27:
		delete [] imageR;
		delete [] imageI;
		delete Fft;
		exit(0);
		break;
	default:
		break;
	}
}

static void InitializeGlut(int *argc, char *argv[])
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WinWidth, WinHeight);
    glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		cerr << "Could not initialize GLEW!" << endl;
		exit(0);
	}
	// Turn off wait for vertical sync
	wglSwapIntervalEXT(0);
	glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	glutKeyboardFunc(Keyboard);
}

int main(int argc, char *argv[])
{	
	ReadData(InputFile);

	InitializeGlut(&argc, argv);

	bool ForwardFFT_p = true;					// Set to false to get reverse FFT
	PingMethod PMethod = AUTO_PBUFFER;			// Use to select the single/double buffer method for your hardware
	// PingMethod PMethod = SINGLE_PBUFFER;		// Select the single buffer method (NVIDIA Quadro board is needed)
	// PingMethod PMethod = DOUBLE_PBUFFER;		// Select the double buffer method

	Fft = new FFT(ForwardFFT_p, PMethod, Width, Height);
	Fft->UploadData(imageR, imageI, imageR, imageI);
	Fft->ComputeMaxAndEnergy(imageR, imageI, imageR, imageI);

	// Resize the display window
	int sh = glutGet(GLUT_SCREEN_HEIGHT)-300;
	int sw = glutGet(GLUT_SCREEN_WIDTH)-300;
	float aspect = (float)(Fft->Height)/(float)(Fft->Width);
	if (aspect > 1.0) {
		WinHeight = sh;
		WinWidth  = (float)WinHeight/aspect;
	}
	else {
		WinWidth = sw;
		WinHeight = (float)WinWidth*aspect;
	}
	WinWidth *= 0.5;
	WinHeight *= 0.5;
	glutReshapeWindow(WinWidth, WinHeight);
	glutPositionWindow(sw-WinWidth, sh-WinHeight); 

	cerr << "Input array: [" << Fft->Width << " x " << Fft->Height << "]" << endl;
	cerr << "Performance is independent of displayed image size," << endl;
	cerr << "which is shrunk to fit the screen." << endl;

	PrintHelp();
	glutMainLoop();

	Fft->DestroyCgPrograms();
	
	return 0;
}

