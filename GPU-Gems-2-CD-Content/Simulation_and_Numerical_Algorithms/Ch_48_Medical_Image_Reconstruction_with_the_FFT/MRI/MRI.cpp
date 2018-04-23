/***************************************************************************
*		FILE NAME:  MRI.cpp
*
* ONE LINE SUMMARY:
*		Demostrates reconcontructing MRI images from acquired Fourier-space
*		data using the GPU.
*
*		It loads two Fourier-domain MRI data streams into the GPU:
*			(a) A mouse heart
*			(b) A human head
*		It then reconstructs the data simultaneously and displays on of the
*		streams on the screen. You can toggle between the two streams by
*		pressing the '1' key. Preass 'h' to get a help line.
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
#include <time.h>
#include <fstream>
#include <iostream>
#include <FFT.h>

using namespace std;

bool DisplayOriginal_p = false;
float SleepTime = 0.0;

GLfloat *Real1 = 0;
GLfloat *Imag1 = 0;
int Width1;
int Height1;
int Nframes1;
GLfloat *Real2 = 0;
GLfloat *Imag2 = 0;
int Width2;
int Height2;
int Nframes2;

int Width;
int Height;
int WinWidth  = 256;
int WinHeight = 256;
FFT *Fft;

// Fourier-space MRI mouse heart data courtesy of Dr. Janaka Wansapura, 
// Assistant Professor, Imaging Research Center, Cincinnati Children's
// Hospital Medical Center, Cincinnati, Ohio, USA
char file1[200] = "../bin/Data/Mouse.dat";

// Fourier-space MRI head data courtesy of Drs. Stephan Kannengiesser and
// Oliver Heid, Siemens Medical Solutions, MR Division, Erlangen, Germany
char file2[200] = "../bin/Data/Head2D.dat";

int ReturnNextPower(int n)
{
    int k;
	float x = (float)n;
    for (k = 0; n > 0; k++) {
        n = n / 2;
		x = x / 2.0f;
    }
    int m = 1;
    for (int i = 0; i < k; i++) {
        m = m * 2;
    }
	if (x == 0.5) {
		return(m/2);
	}
	else {
		return(m);
	}
}

// Read mouse data
void ReadMouseData()
{
	int HeightAcquired;
	int i, j, k, kk;
	int Size;
	float *Data;
	GLfloat *Rtr;
	GLfloat *Itr;
	float *Dtr;
	int hHeightDiff;
	float R, I;

	ifstream ifile;
	ifile.open(file1, ios::in | ios_base::binary);
	if (ifile.fail()) {
		cerr << "Could not open file: " << file1 << endl;
		exit(0);
	}
	else {
		ifile.read((char *)&Nframes1, sizeof(int));
		ifile.read((char *)&Width1, sizeof(int));
		ifile.read((char *)&HeightAcquired, sizeof(int));
		Size = Width1*HeightAcquired*Nframes1*2;
		Data = new GLfloat[Size];
		memset(Data, 0, Size*sizeof(float));
		ifile.read((char *)Data, sizeof(float)*Size);
		ifile.close();

		Height1 = ReturnNextPower(HeightAcquired);
		hHeightDiff = (Height1 - HeightAcquired)/2;
		Size = Width1*Height1*Nframes1;
			
		Real1 = new float [Size];
		Imag1 = new float [Size];
		memset(Real1, 0, Size*sizeof(float));
		memset(Imag1, 0, Size*sizeof(float));

		for (i = 0; i < Nframes1; i++) {
			Rtr = Real1 + i*Width1*Height1;
			Itr = Imag1 + i*Width1*Height1;
			Dtr = Data + i*Width1*HeightAcquired*2;
			for (k = 0; k < HeightAcquired; k++) {
				for (j = 0; j < Width1; j++) {
					R = *(Dtr++);
					I = *(Dtr++);
					kk = k + hHeightDiff;
					*(Rtr + Width1*kk + j) = R;
					*(Itr + Width1*kk + j) = I;
				}
			}
		}
		cerr << endl;
		delete [] Data;
	}
}

// Read head data
void ReadHeadData(void)
{	
	int Size;
	ifstream ifile;
	ifile.open(file2, ios::in | ios_base::binary);
	if (ifile.fail()) {
		cerr << "Could not open file: " << file2 << endl;
	}
	else {
		ifile.read((char *)&Nframes2, sizeof(int));
		ifile.read((char *)&Height2, sizeof(int));
		ifile.read((char *)&Width2, sizeof(int));
		Size = Width2*Height2*Nframes2;
		Real2 = new GLfloat[Size];
		Imag2 = new GLfloat[Size];
		memset(Real2, 0, Size*sizeof(GLfloat));
		memset(Imag2, 0, Size*sizeof(GLfloat));
		ifile.read((char *)Real2, sizeof(float)*Size);
		ifile.read((char *)Imag2, sizeof(float)*Size);
		ifile.close();
		cerr << endl;
	}
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
	static int k1 = 0;
	static int k2 = 0;
	int Size  = Fft->Width * Fft->Height;
	static GLfloat *RPtr1 = Real1;
	static GLfloat *IPtr1 = Imag1;
	static GLfloat *RPtr2 = Real2;
	static GLfloat *IPtr2 = Imag2;
	static bool FirstTime_p = true;

	Fft->UploadData(RPtr1, IPtr1, RPtr2, IPtr2);

	if (FirstTime_p) {
		// Draw a blue bock to look at.
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

	if (DisplayOriginal_p) {
		Fft->DisplayInputImage(WinWidth, WinHeight);
	}
	else {
		Fft->DoFFT();
		Fft->DisplayOutputImage(WinWidth, WinHeight);
	}
	glutSwapBuffers();
		
	k1++;
	RPtr1 += Size;
	IPtr1 += Size;
	if (k1 >= Nframes1) {
		k1 = 0;
		RPtr1 = Real1;
		IPtr1 = Imag1;
	}
	k2++;
	RPtr2 += Size;
	IPtr2 += Size;
	if (k2 >= Nframes2) {
		k2 = 0;
		RPtr2 = Real2;
		IPtr2 = Imag2;
	}
}

static void Idle(void)
{	
	static clock_t thisClock, startClock = clock();
	static int numTimes = 0;
	int avgCount        = 400;
	float duration;

	if (numTimes == avgCount) {
		thisClock = clock();
		duration = (double)(thisClock - startClock)/ CLOCKS_PER_SEC;
		startClock = thisClock;
		numTimes = 0;
		cerr << "Frame Rate (FFT + Uploading) = " << ((float)avgCount)/duration << endl;
	}
	Display();
	if (SleepTime != 0.0) Sleep(SleepTime);
	numTimes++;
}

void PrintHelp(void)
{
	cerr << endl;
	cerr << "Demonstrates MRI image reconstruction using the GPU" << endl;
	cerr << "      Thilaka Sumanaweera, 12/15/2004" << endl;
	cerr << "      Siemens Medical Solutions USA, Inc." << endl;
	cerr << "      Mountain View, CA, USA" << endl;
	cerr << endl;
	cerr << "ESC: quit" << endl;
	cerr << "   1:   toggles showing the mouse data or the head data" << endl;
	cerr << "s/S: make processing slow/fast" << endl;
	cerr << "  f:   toggle displaying output or input data" << endl;
	cerr << "  x:   do FFT in x" << endl;
	cerr << "  y:   do FFT in y" << endl;
	cerr << "  z:   do FFT in x and y" << endl;
	cerr << endl;
}

static void Keyboard(unsigned char key, int x_, int y_)
{
	switch (key) {
	case '1':
		Fft->ShowFirstImage_p = !Fft->ShowFirstImage_p;
		Fft->SetDisplayMask();
		if (Fft->ShowFirstImage_p) {
			cerr << "Show mouse data." << endl;
		}
		else {
			cerr << "Show head data." << endl;
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
	case 's':
		SleepTime += 10.0;
		cerr << "SleepTime = " << SleepTime << " ms." << endl;
		break;
	case 'S':
		SleepTime -= 10.0;
		SleepTime = (SleepTime < 0.0) ? 0.0 : SleepTime;
		cerr << "SleepTime = " << SleepTime << " ms." << endl;
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
	case 'h':
		PrintHelp();
		break;
	case 27:
		delete [] Real1;
		delete [] Imag1;
		delete [] Real2;
		delete [] Imag2;
		delete Fft;
		exit(0);
		break;
	}
}

static void InitializeGlut(int *argc, char *argv[])
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WinWidth, WinHeight);
	glutCreateWindow(argv[0]);
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		cerr << "Could not initialize GLEW!" << endl;
		exit(0);
	}
    glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutIdleFunc(Idle);
	glutKeyboardFunc(Keyboard);
	wglSwapIntervalEXT(0);
	glutReshapeWindow(WinWidth, WinHeight);
}

int main(int argc, char *argv[])
{
	ReadMouseData();
	ReadHeadData();
	if ((Width1 == Width2) && (Height1 == Height2)) {
		Width = Width1;
		Height = Height1;
	}

	InitializeGlut(&argc, argv);

	bool ForwardFFT_p = false;					// Set to false to get reverse FFT
	PingMethod PMethod = AUTO_PBUFFER;			// Use to select the single/double buffer method for your hardware
	// PingMethod PMethod = SINGLE_PBUFFER;		// Select the single buffer method (NVIDIA Quadro board is needed)
	// PingMethod PMethod = DOUBLE_PBUFFER;		// Select the double buffer method

	Fft = new FFT(ForwardFFT_p, PMethod, Width, Height);
	// Some custom settings

	// For the mouse data
	float Energy1 = 3000000.0;
	float Max1    = 9650.0;
	// For the head data
	float Energy2 = 0.1;
	float Max2    = 1.0/5000.0;
	Fft->SetMaxAndEnergy(Energy1, Max1, Energy2, Max2);
	Fft->xCutOff = Fft->nButterfliesX/2;
	Fft->yCutOff = Fft->nButterfliesY/2;

	PrintHelp();

	glutMainLoop();
	return 0;
}
