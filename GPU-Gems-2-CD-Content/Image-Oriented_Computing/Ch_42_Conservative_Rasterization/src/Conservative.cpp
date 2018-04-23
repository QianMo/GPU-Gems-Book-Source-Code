/**
 * @file RenderTexture.h
 * 
 * Conservative rasterization demo. 
 * This demo is based on the RenderTexture class by Mark Harris
 */

#include "RenderTexture.h"

#include <GL/glut.h>
#include <Cg/cgGL.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WINDOW_XRES 512
#define WINDOW_YRES 512

#define NUM_PROGRAMS 6
CGprogram   fragProgram[NUM_PROGRAMS];
CGprogram   vtxProgram[NUM_PROGRAMS];

CGcontext   cgContext;
CGprofile   fragProfile;
CGprofile   vtxProfile;

char *progs[] = {	"shaders\\conservativefp.cg","shaders\\conservativevp.cg",
					"shaders\\conservativefpexact.cg","shaders\\conservativevp.cg",
					"shaders\\conservativefpexact.cg","shaders\\conservativevpexact.cg",
					"shaders\\conservativefpod.cg","shaders\\conservativevp.cg",
					"shaders\\conservativefpod.cg","shaders\\conservativevp.cg",
					"shaders\\conservativefpod.cg","shaders\\conservativevpexact.cg"
				};
int useProgram = 0;


#define MAX_TEX_RES 12
int NUM_TEX_RES = 10;
int currentTexRes = 5;
int texRes[MAX_TEX_RES] = {2,4,8,16,32,64,128,256,512,1024,2048,4096};

#define MAX_LIST_RES 8
int NUM_LIST_RES = 7;
int currentListRes = 4;
int listRes[MAX_LIST_RES] = {4,8,16,32,64,128,256,512};

GLuint useList = 1;
GLuint wireList[MAX_LIST_RES];
GLuint consList[MAX_LIST_RES*2];	

int wireframe = 0,grid = 0,conservativewire = 0,overdraw = 0;

RenderTexture *rt = NULL;

float xangle = 0;
float yangle = 0;

int lastX = 0;
int lastY = 0;
int lastState = GLUT_UP;

#define PI 3.1415926535897932384626433832795f

//------------------------------------------------------------------------------
// Function     	 : cgErrorCallback
// Description		: 
//------------------------------------------------------------------------------
void cgErrorCallback()
{
	CGerror lastError = cgGetError();

	if(lastError)
	{
		printf("%s\n\n", cgGetErrorString(lastError));
		printf("%s\n", cgGetLastListing(cgContext));
		printf("Cg error, exiting...\n");

		exit(0);
	}
} 


//------------------------------------------------------------------------------
// Function     	  : PrintGLerror
// Description	    : 
//------------------------------------------------------------------------------
void PrintGLerror( char *msg )
{
	GLenum errCode;
	const GLubyte *errStr;

	if ((errCode = glGetError()) != GL_NO_ERROR) 
	{
		errStr = gluErrorString(errCode);
		fprintf(stderr,"OpenGL ERROR: %s: %s\n", errStr, msg);
	}
}


//------------------------------------------------------------------------------
// Function     	  : Idle
// Description	    : 
//------------------------------------------------------------------------------
void Idle()
{
	glutPostRedisplay();
}

//------------------------------------------------------------------------------
// Function     	  : Reshape
// Description	    : 
//------------------------------------------------------------------------------
void Reshape(int w, int h)
{
	if (h == 0) h = 1;

	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float aspect = (float)h/(float)w;

	glFrustum(-1,1,-1,1,1,1000);
}

//------------------------------------------------------------------------------
// Function     	  : SetupTexture
// Description	    : 
//------------------------------------------------------------------------------
void SetupTexture(int x,int y) {
	// A square, mipmapped, anisotropically filtered 8-bit RGBA texture with
	// depth and stencil.
	rt = new RenderTexture(x, y);
	rt->Initialize(true, true, true, true, true, 8, 8, 8, 8, RenderTexture::RT_COPY_TO_TEXTURE);

	// setup the rendering context for the RenderTexture
	rt->BeginCapture();
	Reshape(x, y);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST); 
	glClearColor(0, 0, 0, 1);
	rt->EndCapture();
}

//------------------------------------------------------------------------------
// Function     	  : SetupTexture
// Description	    : 
//------------------------------------------------------------------------------
void CreateMesh(int x,int y,int listidx) {

	float *MeshCoords = NULL;
	int *MeshIndices = NULL;

	float r = 100.0f;

	MeshCoords = new float[x*y*3];
	MeshIndices = new int[x*(y-1)*4];

	for (int i = 0; i < y; i++) {
		for(int j = 0; j < x; j++) {
			float theta = -((float)j/(float)x)*2.0*PI;
			float phi = ((float)i/(float)(y-1))*(PI-0.3)+0.15;
			float r = cos(fabs(phi-PI/2.0)*10.0)*20+80;

			MeshCoords[(j+i*x)*3] = r*cos(theta)*sin(phi);
			MeshCoords[(j+i*x)*3+1] = r*sin(theta)*sin(phi);
			MeshCoords[(j+i*x)*3+2] = r*cos(phi);
		}
	}

	for (int i = 0; i < y-1; i++) {
		for(int j = 0; j < x; j++) {
			MeshIndices[(j+i*x)*4] = (j+i*x);
			MeshIndices[(j+i*x)*4+1] = ((j+1)%x+i*x);
			MeshIndices[(j+i*x)*4+2] = ((j+1)%x+(i+1)*x);
			MeshIndices[(j+i*x)*4+3] = (j+(i+1)*x);
		}
	}

	// Generate display lists, different formats for normal rendering, exact and hybrid technique
	wireList[listidx] = glGenLists(1);
	consList[listidx*2+0] = glGenLists(1);	
	consList[listidx*2+1] = glGenLists(1);

	int cnt = 0;

	// Generate a displaylist
	glNewList(wireList[listidx],GL_COMPILE);
	glBegin(GL_TRIANGLES);						
	for (int i = 0; i < x*(y-1); i++) {
		for (int u = 0; u < 2; u++) {	
			int Indices[3] = {MeshIndices[i*4]*3, MeshIndices[i*4+u+1]*3, MeshIndices[i*4+u+2]*3};
			for (int j = 0; j < 3; j++)
				glVertex3fv(&MeshCoords[Indices[j]]);
		}
		if ( (cnt++ % 4096) == 0) printf(".");
	}
	glEnd();
	glEndList();

	// Generate a list for the exact method
	glNewList(consList[listidx*2+0],GL_COMPILE);
	for (int i = 0; i < x*(y-1); i++) {
		for (int u = 0; u < 2; u++) {	
			int Indices[3] = {MeshIndices[i*4]*3, MeshIndices[i*4+u+1]*3, MeshIndices[i*4+u+2]*3};

			glBegin(GL_TRIANGLE_FAN);
			for (int j = 0; j < 9; j++) {
				int t = j/3;
				int v = j%3;
				glMultiTexCoord3fvARB(0,&MeshCoords[Indices[(t+2)%3]]);
				glMultiTexCoord3fvARB(1,&MeshCoords[Indices[(t+1)%3]]);
				glMultiTexCoord1fARB(2,v);
				glVertex3fv(&MeshCoords[Indices[t]]);
			}
			glEnd();
		}
		if ( (cnt++ % 4096) == 0) printf(".");
	}
	glEndList();

	// Generate a list for the hybrid method
	glNewList(consList[listidx*2+1],GL_COMPILE);
	glBegin(GL_TRIANGLES);						
	for (int i = 0; i < x*(y-1); i++) {
		for (int u = 0; u < 2; u++) {	
			int Indices[3] = {MeshIndices[i*4]*3, MeshIndices[i*4+u+1]*3, MeshIndices[i*4+u+2]*3};
			for (int j = 0; j < 3; j++) {
				glMultiTexCoord3fvARB(0,&MeshCoords[Indices[(j+2)%3]]);
				glMultiTexCoord3fvARB(1,&MeshCoords[Indices[(j+1)%3]]);
				glVertex3fv(&MeshCoords[Indices[j]]);
			}
		}
		if ( (cnt++ % 4096) == 0) printf(".");
	}
	glEnd();
	glEndList();

	delete MeshCoords;
	delete MeshIndices;

}

//------------------------------------------------------------------------------
// Function     	  : SetupTexture
// Description	    : 
//------------------------------------------------------------------------------
void setProg(CGprogram vtxProg,CGprogram fragProg) {
	cgGLBindProgram(fragProg);
	cgGLEnableProfile(fragProfile);
	cgGLBindProgram(vtxProg);
	cgGLEnableProfile(vtxProfile);			

	CGparameter ModelProjParam,vtxHPixel;
	vtxHPixel = cgGetNamedParameter(vtxProg,"hPixel");
	ModelProjParam = cgGetNamedParameter(vtxProg,"ModelViewProj");

	cgGLSetStateMatrixParameter(ModelProjParam,CG_GL_MODELVIEW_PROJECTION_MATRIX,CG_GL_MATRIX_IDENTITY);
	cgGLSetParameter4f(vtxHPixel, 1.0f/(float)rt->GetWidth(), 1.0f/(float)rt->GetHeight(), 1.0f/(float)rt->GetWidth(), 1.0f/(float)rt->GetHeight());
}

//------------------------------------------------------------------------------
// Function     	  : display
// Description	    : 
//------------------------------------------------------------------------------
void display()
{	
	float Dist = -200;

	float TriA[9] = {-100,-100,0, 10,0,0, -100,100,0};
	float TriB[9] = {100,-100,-10, 100,100,-10, -10,0,-10};
	rt->BeginCapture();
	{ 
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0,0,Dist);
		glRotatef(xangle,1,0,0);
		glRotatef(yangle,0,1,0);

		glEnable(GL_DEPTH_TEST);
		if (overdraw) {
			glEnable(GL_BLEND);
			glDepthFunc(GL_ALWAYS);
			glBlendFunc(GL_ONE,GL_ONE);
		}
		else {
			glDisable(GL_BLEND);
			glDepthFunc(GL_LEQUAL);
		}

		glEnable(GL_CULL_FACE);					

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		setProg(vtxProgram[useProgram + overdraw*3],fragProgram[useProgram + overdraw*3]);

		glColor3f(1,1,1);
		glCallList(consList[currentListRes*2+useList]);

		cgGLDisableProfile(fragProfile);
		cgGLDisableProfile(vtxProfile);

		glDisable(GL_BLEND);
		PrintGLerror("RT Update");

	}    
	rt->EndCapture();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3f(1, 1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glDepthFunc(GL_ALWAYS);

	rt->Bind();
	rt->EnableTextureTarget();

	// Render texture
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(-1, -1, -1);
	glTexCoord2f(1, 0); glVertex3f( 1, -1, -1);
	glTexCoord2f(1, 1); glVertex3f( 1,  1, -1);
	glTexCoord2f(0, 1); glVertex3f(-1,  1, -1);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	// Render coord system grid
	if (grid) {
		glColor3f(1,0,0);
		glBegin(GL_LINES);
		for (int u = 0; u <= rt->GetHeight(); u++) {
			glVertex3f(-1,(float)u / ((float)rt->GetHeight()/2.0f) - 1, -1);
			glVertex3f(1,(float)u / ((float)rt->GetHeight()/2.0f) - 1, -1);
		}
		for (int u = 0; u <= rt->GetWidth(); u++) {
			glVertex3f((float)u / ((float)rt->GetWidth()/2.0f) - 1,-1, -1);
			glVertex3f((float)u / ((float)rt->GetWidth()/2.0f) - 1,1, -1);
		}
		glEnd();
	}

	// Render trianges in full resolution as lines.
	glLoadIdentity();
	glTranslatef(0,0,Dist);
	glRotatef(xangle,1,0,0);
	glRotatef(yangle,0,1,0);

	glLineWidth(1);
	glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glEnable(GL_CULL_FACE);

	if (wireframe) {
		glColor3f(0,0,1);
		glCallList(wireList[currentListRes]);
	}


	if (conservativewire) {

		// conservative but in full resolution emulating the small resolution
		setProg(vtxProgram[useProgram],fragProgram[useProgram]);

		glColor3f(1,1,0);
		glCallList(consList[currentListRes*2+useList]);
	}

	cgGLDisableProfile(fragProfile);
	cgGLDisableProfile(vtxProfile);

	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

	rt->DisableTextureTarget();

	PrintGLerror("display");
	glutSwapBuffers();
}

//------------------------------------------------------------------------------
// Function     	  : KBFunc
// Description	    : 
//------------------------------------------------------------------------------
void KBFunc(unsigned char key, int x, int y) {
	if (key == 'w') {
		wireframe = !wireframe;
	}
	else if (key == 'g') {
		grid = !grid;
	}
	else if (key == 'r') {
		conservativewire = !conservativewire;
	}
	else if (key == 'o') {
		overdraw = !overdraw;
	}
	else if (key == '+') {
		currentTexRes += currentTexRes < NUM_TEX_RES-1 ? 1 : 0;
		delete rt;
		SetupTexture(texRes[currentTexRes],texRes[currentTexRes]);
		printf("Texture res: %dx%d\n",texRes[currentTexRes],texRes[currentTexRes]);
	}
	else if (key == '-') {
		currentTexRes -= currentTexRes > 0 ? 1 : 0;
		delete rt;
		SetupTexture(texRes[currentTexRes],texRes[currentTexRes]);
		printf("Texture res: %dx%d\n",texRes[currentTexRes],texRes[currentTexRes]);
	}
	else if (key == '.') {		
		currentListRes += currentListRes < NUM_LIST_RES-1 ? 1 : 0;
		printf("Mesh res: %dx%d\n",listRes[currentListRes],listRes[currentListRes]);
	}
	else if (key == ',') {
		currentListRes -= currentListRes > 0 ? 1 : 0;
		printf("Mesh res: %dx%d\n",listRes[currentListRes],listRes[currentListRes]);
	}
	else if (key == 'h') {
		useProgram = 0;
		useList = 1;
	}
	else if (key == 'c') {
		useProgram = 1;
		useList = 1;
	}
	else if (key == 'e') {
		useProgram = 2;
		useList = 0;
	}
	else if (key == 'q' || key == 27) {
		exit(0);
	}
}

//------------------------------------------------------------------------------
// Function     	  : MouseMotion
// Description	    : 
//------------------------------------------------------------------------------
void MouseMotion (int x, int y) {
	if (lastState == GLUT_DOWN) {
		xangle = (float)(y-lastY)*0.25f;
		yangle = (float)(x-lastX)*0.25f;
	}
}

//------------------------------------------------------------------------------
// Function     	  : MouseCallback
// Description	    : 
//------------------------------------------------------------------------------
void MouseCallback (int button,int state,int x, int y) {	
	if (state == GLUT_DOWN && button == GLUT_LEFT) {
		lastX = x;
		lastY = y;		
	}
	lastState = state;
}

//------------------------------------------------------------------------------
// Function     	  : main
// Description	    : 
//------------------------------------------------------------------------------
int main(int argv, char **args)
{
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(50, 50);
	glutInitWindowSize(WINDOW_XRES, WINDOW_YRES);
	glutCreateWindow("TestRenderTexture");  

	if (argv >= 2) {  
		int res = strtol(args[1],NULL,10);
		for (int i = 0; i < MAX_TEX_RES; i++)
			if (res == texRes[i]) 
				NUM_TEX_RES = i+1;
		printf("Using max texture res: %d\n",texRes[NUM_TEX_RES-1]);
	}
	if (argv >= 3) {
		int res = strtol(args[2],NULL,10);
		for (int i = 0; i < MAX_LIST_RES; i++)
			if (res == listRes[i]) 
				NUM_LIST_RES = i+1;
		printf("Using max tesselation: %dx%d, %d\n",listRes[NUM_LIST_RES-1],listRes[NUM_LIST_RES-1],NUM_LIST_RES);
	}

	// Init GLEW
	int err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
		exit(-1);
	}  

	// Setup callbacks
	glutDisplayFunc(display);
	glutIdleFunc(Idle);
	glutReshapeFunc(Reshape);
	glutMouseFunc( MouseCallback );
	glutMotionFunc( MouseMotion );
	glutKeyboardFunc( KBFunc );

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST); 

	SetupTexture(texRes[currentTexRes],texRes[currentTexRes]);
	
	// Setup Cg
	cgSetErrorCallback(cgErrorCallback);
	cgContext = cgCreateContext();

	// Get the best profile for this hardware
	fragProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	vtxProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
	assert(fragProfile != CG_PROFILE_UNKNOWN && vtxProfile != CG_PROFILE_UNKNOWN);
	cgGLSetOptimalOptions(fragProfile);
	cgGLSetOptimalOptions(vtxProfile);

	printf("Loading programs\n");
	for (int i = 0; i < NUM_PROGRAMS; i++) {
		fragProgram[i] = cgCreateProgramFromFile(cgContext, CG_SOURCE, progs[i*2], fragProfile, NULL, NULL);
		vtxProgram[i] = cgCreateProgramFromFile(cgContext, CG_SOURCE, progs[i*2+1], vtxProfile, NULL, NULL);
		cgGLLoadProgram(fragProgram[i]);
		cgGLLoadProgram(vtxProgram[i]);
	}

	printf("Tesselating meshes: ");
	for (int i = 0; i < NUM_LIST_RES; i++) {
		printf("\n%dx%d: ",listRes[i],listRes[i]);
		CreateMesh(listRes[i],listRes[i],i);
	}
	printf("\n");
	useList = 1;

	printf("\nKeys:\n");
	printf("--------------------------------------------------------------------\n");
	printf("w - Show wireframe outline\n");
	printf("g - Show pixel cell grid for the texture used as rendertarget\n");
	printf("r - Show wireframe outline of the bounding primitives\n");
	printf("o - Show overdraw factor\n");
	printf("+ - Increase render-texture resolution\n");
	printf("- - Decrease render-texture resolution\n");
	printf(". - Increase mesh resolution\n");
	printf(", - Decrease mesh resolution\n");
	printf("h - Use the hybrid algorithm\n");
	printf("c - Use the hybrid algorithm without the pixel shader clipping-code\n");
	printf("e - Use the algorithm that computes exact bounding primitives\n");


	glutMainLoop();

	return 0;
}

