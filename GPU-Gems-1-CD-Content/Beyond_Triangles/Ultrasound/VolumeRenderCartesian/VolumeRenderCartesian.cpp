/**************************************************************************
*        FILE NAME:  VolumeRenderCartesian.cpp
*
* ONE LINE SUMMARY:
*        Tested on NVIDIA Quadro FX3000
*
* DESCRIPTION:
*        Demonstrates use of 3D textures for volume rendering.
*        Does not require Cg or programmable GPU. Only requires
*        GL_EXT_texture3D OpenGL extension.
*
*    Thilaka Sumanaweera
*    (C) 2002 Siemens Medical Solutions USA, Inc, Ultrasound Division
*    Thilaka.Sumanaweera@siemens.com
*    05/16/2003
*
*****************************************************************************
* REVISION HISTORY:
* Rev     	When      	Who         	What
* A	  	 16May03        sumane		    Created.
****************************************************************************/
#include <fstream.h>
#include <iostream.h>
#include "glut.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include "glext.h"
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>

GLubyte *image;
GLuint texName;
GLuint listName;
int iWidth;
int iHeight;
int iDepth;
float alpha = -85.0;
float theta =  90.0;
float gamma =   0.0;
float scale =   0.7;
char step_p = 0;

PFNGLTEXIMAGE3DEXTPROC glTexImage3DEXT = NULL;

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   ReadHeader
*
* DESCRIPTION:
*
*   This reads in the header of the 3D data volume
*
* FORMAL PARAMETERS:
*   prefix contains the file prefix.
*
* RETURNS:
*   w, h, and d are width, height and depth of the volume
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void ReadHeader(char *prefix, int &w, int &h, int &d)
{
	char file[300];
	ifstream ifile;

	sprintf(file, "%s.hdr", prefix);
	cerr << file << endl;
	ifile.open(file, ios::in | ios::nocreate);

	if (ifile.fail()) {
		cerr << "Could not open " << file << endl;
		exit(0);
	}
	ifile >> w;
	ifile >> h;
	ifile >> d;
	ifile.close();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   ReadVolume
*
* DESCRIPTION:
*
*   This reads in the 3D data volume
*
* FORMAL PARAMETERS:
*   prefix contains the file prefix.
*
* RETURNS:
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void ReadVolume(char *prefix)
{
	char file[100];
	ifstream ifile;
	
	ReadHeader(prefix, iWidth, iHeight, iDepth);
	cerr << "Volume size: [" << iWidth << "x" << iHeight << "x" << iDepth << "]" << endl;
	
	sprintf(file, "%s.img", prefix);
	int Size = iWidth*iHeight*iDepth;

	GLubyte *image_ = new GLubyte [Size];
	image           = new GLubyte [Size*4];

	ifile.open(file, ios::in | ios::nocreate);
	if (ifile.fail()) {
		cerr << "Could not open " << file << endl;
		exit(0);
	}
	ifile.setmode(filebuf::binary);
	ifile.read((signed char *)image_, Size);
	ifile.close();

	// Put a box around the volume so that we can see the outline 
	// of the data.
	//
	GLubyte *ptr = image_;
	int i, j, k;
    for (i = 0; i < iDepth; i++) {
        for (j = 0; j < iHeight; j++) {
            for (k = 0; k < iWidth; k++) {
				if (((i < 4) && (j < 4)) ||
					((j < 4) && (k < 4)) ||
					((k < 4) && (i < 4)) ||
					((i < 4) && (j >  iHeight-5)) ||
					((j < 4) && (k > iWidth-5)) ||
					((k < 4) && (i > iDepth-5)) ||
					((i > iDepth-5) && (j >  iHeight-5)) ||
					((j >  iHeight-5) && (k > iWidth-5)) ||
					((k > iWidth-5) && (i > iDepth-5)) ||
					((i > iDepth-5) && (j < 4)) ||
					((j >  iHeight-5) && (k < 4)) ||
					((k > iWidth-5) && (i < 4))) {
					*ptr = 110;
				}
				ptr++;
			}
		}
	}

	ptr = image_;
	GLubyte *qtr = image;
	GLubyte v, val;
	for (i = 0; i < Size; i++) {
		v = *(ptr++);
		val = (v < 64) ? 0 : v - 64;
		val = val >> 1;
		*(qtr++) = val;
		*(qtr++) = ((float)val)*0.93;
		*(qtr++) = ((float)val)*0.78;
		*(qtr++) = val;
	}
	delete [] image_;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   SetDisplayList
*
* DESCRIPTION:
*
*   Creates a display list containing all the enclosing rectangles 
*   corresponding to the cutplanes.
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void SetDisplayList(void)
{
   float dist;
   int i;
   int nHSlices     = 1.3*(sqrt(float(iWidth*iWidth + iHeight*iHeight + iDepth*iDepth)))/4.0;
   int nSlices      = 2*nHSlices + 1;
   int nElements    = nSlices*4;
   GLfloat *vPoints = new GLfloat [3*nElements];
   GLfloat *ptr     = vPoints;
   float dDist      = sqrt(3.0)/nSlices;
   float s          = 0.5;

   
   for (i = -nHSlices; i <= nHSlices; i++) {
	   dist     = i*dDist;

	   *(ptr++) = -s;
	   *(ptr++) = -s;
	   *(ptr++) =  dist;

	   *(ptr++) =  s;
	   *(ptr++) = -s;
	   *(ptr++) =  dist;

	   *(ptr++) =  s;
	   *(ptr++) =  s;
	   *(ptr++) =  dist;

	   *(ptr++) = -s;
	   *(ptr++) =  s;
	   *(ptr++) =  dist;
   }

   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(3, GL_FLOAT, 0, vPoints);
   listName = glGenLists(1);
   glNewList(listName, GL_COMPILE);
   glDrawArrays(GL_QUADS, 0, nElements); 
   glEndList();
   delete [] vPoints;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Init
*
* DESCRIPTION:
*
*   Initializes OpenGL state machine
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Init(void)
{    
   ReadVolume("../Data/CT/CTData");

   glClearColor (0.0, 0.0, 0.0, 0.0);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glGenTextures(1, &texName);
   glBindTexture(GL_TEXTURE_3D, texName);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   // For automatic texture coordinate generation
   glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
   glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
   glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);

   glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_RGBA, iWidth, iHeight, iDepth, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
			   
   glEnable(GL_TEXTURE_GEN_S);
   glEnable(GL_TEXTURE_GEN_T);
   glEnable(GL_TEXTURE_GEN_R);
   glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
   glEnable(GL_BLEND);

   SetDisplayList();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Display
*
* DESCRIPTION:
*
*   Display routine for OpenGL
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Display(void)
{
   glEnable(GL_TEXTURE_3D);
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
   glBindTexture(GL_TEXTURE_3D, texName);

   glClear(GL_COLOR_BUFFER_BIT);

   // Clip planes
   static GLdouble eqx0[4] = { 1.0, 0.0, 0.0, 0.0};
   static GLdouble eqx1[4] = {-1.0, 0.0, 0.0, 1.0};
   static GLdouble eqy0[4] = {0.0,  1.0, 0.0, 0.0};
   static GLdouble eqy1[4] = {0.0, -1.0, 0.0, 1.0};
   static GLdouble eqz0[4] = {0.0, 0.0,  1.0, 0.0};
   static GLdouble eqz1[4] = {0.0, 0.0, -1.0, 1.0};
   
   // Define equations for automatic texture coordinate generation
   static GLfloat x[] = {1.0, 0.0, 0.0, 0.0};
   static GLfloat y[] = {0.0, 1.0, 0.0, 0.0};
   static GLfloat z[] = {0.0, 0.0, 1.0, 0.0};

   
   glPushMatrix ();
   glScalef(scale, scale, scale);

   glPushMatrix ();

   // Transform the viewing direction
   glRotatef(-gamma, 0.0, 0.0, 1.0);
   glRotatef(-theta, 0.0, 1.0, 0.0);
   glRotatef(-alpha, 1.0, 0.0, 0.0);
   glTranslatef(-0.5, -0.5, -0.5);

   // Take a copy of the model view matrix now shove it in to the GPU 
   // buffer for later use in automatic texture coord generation.
   glTexGenfv(GL_S, GL_EYE_PLANE, x);
   glTexGenfv(GL_T, GL_EYE_PLANE, y);
   glTexGenfv(GL_R, GL_EYE_PLANE, z);

   // Take a copy of the model view matrix and shove it in to the GPU
   // buffer for later use in clipping planes.
   glClipPlane(GL_CLIP_PLANE0, eqx0);
   glClipPlane(GL_CLIP_PLANE1, eqx1);

   glClipPlane(GL_CLIP_PLANE2, eqy0);
   glClipPlane(GL_CLIP_PLANE3, eqy1);

   glClipPlane(GL_CLIP_PLANE4, eqz0);
   glClipPlane(GL_CLIP_PLANE5, eqz1);
   
   glPopMatrix ();

   // Enable clip planes
   glEnable(GL_CLIP_PLANE0);
   glEnable(GL_CLIP_PLANE1);

   glEnable(GL_CLIP_PLANE2);
   glEnable(GL_CLIP_PLANE3);

   glEnable(GL_CLIP_PLANE4);
   glEnable(GL_CLIP_PLANE5);
   
   // Render enclosing rectangles
   glCallList(listName);
 
   glPopMatrix ();
   glutSwapBuffers();

   glDisable(GL_TEXTURE_3D);
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Idle
*
* DESCRIPTION:
*
*   Idle routine for OpenGL
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Idle(void)
{
	static clock_t thisClock, startClock = clock();
	static int numTimes = 0;
	int avgCount        = 20;
	float duration;

	if (numTimes == avgCount) {
		thisClock = clock();
		duration = (double)(thisClock - startClock)/ CLOCKS_PER_SEC;
		startClock = thisClock;
		numTimes = 0;
		cerr << "Frame Rate = " << ((float)avgCount)/duration << endl;
	}

	Display();

	if (!step_p) {
		theta += 5.0;
		theta = (theta >= 360.0) ? 0.0 : theta;
	}

	numTimes++;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Reshape
*
* DESCRIPTION:
*
*   Reshape routine for OpenGL
*
* FORMAL PARAMETERS:
*   w and h are width and height of the window
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Reshape(int w, int h)
{
   glViewport(0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   GLfloat size_ = 1.0;
   GLfloat aspect = (GLfloat) h/(GLfloat) w;
   glOrtho(-size_, size_, -size_*aspect, size_*aspect, -size_, size_);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   PrintHelp
*
* DESCRIPTION:
*
*   Prints a help line
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void PrintHelp(void)
{
	cerr << endl;
	cerr << "Help:" << endl;
	cerr << endl;
	cerr << "x/X: Increase or decrease Alpha" << endl;
	cerr << "y/Y: Increase or decrease Theta" << endl;
	cerr << "g/G: Increase or decrease Gamma" << endl;
	cerr << "z/Z: Zoom in or out" << endl;
	cerr << "s  : Stop/start automatic rotation" << endl;
	cerr << "ESC: Quit" << endl;
	cerr << endl;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Keyboard
*
* DESCRIPTION:
*
*   Handles keybord commands
*
* FORMAL PARAMETERS:
*   key is the key pressed by the user.
*   (x, y) are the coordinates of the mouse
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	  case 'h':
	      PrintHelp();
	      break;
	  case 'x':
		  alpha += 10.0;
		  alpha = (alpha >= 360.0) ? 0.0 : alpha;
		  cerr << "Alpha: " << alpha << endl;
		  break;
	  case 'X':
		  alpha -= 10.0;
		  alpha = (alpha < 0.0) ? 360.0 : alpha;
		  cerr << "Alpha: " << alpha << endl;
		  break;
	  case 'y':
		  theta += 10.0;
		  theta = (theta >= 360.0) ? 0.0 : theta;
		  cerr << "Theta: " << theta << endl;
		  break;
	  case 'Y':
		  theta -= 10.0;
		  theta = (theta < 0.0) ? 360.0 : theta;
		  cerr << "Theta: " << theta << endl;
		  break;
	  case 'g':
		  gamma += 10.0;
		  gamma = (gamma >= 360.0) ? 0.0 : gamma;
		  cerr << "Gamma: " << gamma << endl;
		  break;
	  case 'G':
		  gamma -= 10.0;
		  gamma = (gamma < 0.0) ? 360.0 : gamma;
		  cerr << "Gamma: " << gamma << endl;
		  break;
	  case 'z':
		  scale *= 0.9;
		  cerr << "ZoomFactor: " << scale << endl;
		  break;
	  case 'Z':
		  scale *= 1.1;
		  cerr << "ZoomFactor: " << scale << endl;
		  break;
	  case 's':
		  step_p = !step_p;
		  break;
      case 27:
		  delete [] image;
         exit(0);
         break;
   }
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   main
*
* DESCRIPTION:
*
* This is the main program that starts the app.
*
* FORMAL PARAMETERS:
*  none
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
int main(int argc, char** argv)
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
   glutInitWindowSize(640, 480);
   glutInitWindowPosition(100, 100);
   glutCreateWindow(argv[0]);

   if (glutExtensionSupported("GL_EXT_texture3D")) {
		cerr << "GL_EXT_texture3D supported!" << endl;
		glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC)wglGetProcAddress("glTexImage3DEXT");
   }
   else {
		cerr << "GL_EXT_texture3D unsupported!" << endl;
		exit(0);
   }

   Init();
   glutDisplayFunc(Display);
   glutReshapeFunc(Reshape);
   glutIdleFunc(Idle);
   glutKeyboardFunc (Keyboard);

   glutMainLoop();

   return 0; 
}

