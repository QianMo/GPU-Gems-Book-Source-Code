//
// StochasticGPU.cpp
// Last Updated:		05.01.07
// 
// Mark Colbert & Jaroslav Krivanek
// colbert@cs.ucf.edu
//
// Copyright (c) 2007.
//
// The following code is freely distributed "as is" and comes with 
// no guarantees or required support by the authors.  Any use of 
// the code for commercial purposes requires explicit written consent 
// by the authors.
//

// GLEW
#include <GL/glew.h>

#include <windows.h>
// Associated Linking for Windows
#ifdef WIN32
#include <GL/wglew.h>
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "cg.lib")
#pragma comment(lib, "cgGL.lib")
#endif

// GLUT
#include <GL/glut.h>

// Cg
#include <Cg/cg.h>
#include <Cg/cgGL.h>
CGcontext	context;
CGprogram	vShader, fShader[3];			///< sampling shaders
CGprogram	envVShader, envFShader;			///< environment shader
CGprofile	VertexProfile, FragmentProfile; 

CGparameter samplesParam;					///< sampling parameters
CGparameter randParam, randcosParam,
			randsinParam, randlogParam;

CGparameter eyeParam;						///< shared parameters
CGparameter specularParam, diffuseParam;
CGparameter keyGammaParam;					///< tonemapping parameters

CGparameter lobeParam;						///< lafortune parameter
CGparameter alphaParam;						///< ward parameter

CGparameter rMParam, gMParam, bMParam;		///< spherical harmonics parameter

CGparameter objectToWorldParam;				///< object matrix rotation

CGparameter sSamplesParam, lScalesParam, lSmplsParam, wScalesParam, wSmplsParam;

//bool useCubeMap = false;
bool showInterface = true;

void cgInit();
void cgDestroy();
void cgError();

// standard libraries
#include <iostream>
#include <iomanip>
#include <cassert>
using namespace std;

// Program Specific Library
#include "alloc2D.h"
#include "wavefront.h"
#include "binaryobj.h"
#include "hdrimage.h"
#include "color.h"
#include "widgets.h"
#include "brdfgraph.h"
#include "sequencegen.h"

/// Calllback Prototypes ///
void init();
void destroy();
void display();
void reshape(int w, int h);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void passiveMotion(int x, int y);
void keyboard(unsigned char key, int x, int y);
void idle();

/// Global Variables ///
// resolution prototypes
int wWidth=0,wHeight=0;

int samples=40;

// rotation variables
static int ox=0, oy=0, rx=0, ry=0, orx=0, ory=0;
static int mouseButton;

void initMesh(const char *filename, unsigned int glId);
void createPositionMap();
void initHDRBackground(const char *fileName, unsigned int front, unsigned int back,
					   float *r, float *g, float *b, float *fBuffer, float *bBuffer, unsigned int mapSize);

void initEnvSphere();

void saveFramebuffer();
void printStats(ostream &fout);

GLuint envId, glMeshId[3];

GLuint hdrTexId[6];
//GLuint hdrCubeTexId[3];
float **r, **g, **b;

int currMesh=0;
int currProbe=0;

void displayEnvSphere();
void displayLafortune();
void displayWard();
void displaySVLafortune();

void updateProbeSH(int probe);

const float xOffset=-0.75f;
float eyeDist=-5.f;
//float eyeDist=-3.f;

const unsigned int meshes=3;
const unsigned int probes=3;
const char *objectFileName[] = {"sphere.obj", "happy.obj", "athena.obj"};
const char *hdrFileName[] = {"grace_probe.hdr", "ennis.hdr", "kitchen_probe.hdr"};

/// Initialization Function ///
void init() {
	cgInit();
	wglSwapIntervalEXT(0);

	glClearColor(0,0,0,0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(xOffset, 0.f, eyeDist);
	cgGLSetParameter3f(eyeParam, 0.f, 0.f, -eyeDist);

	// allocated handles and memory
	// generate the OpenGL handles
	for (int i=0; i < meshes; i++)
		glMeshId[i] = glGenLists(1);

	// generate the texture handles
	glGenTextures(probes*2, hdrTexId);
	//glGenTextures(probes, hdrCubeTexId);
	r = alloc2D<float>(16, probes);
	g = alloc2D<float>(16, probes);
	b = alloc2D<float>(16, probes);


	const unsigned int reflectMapSize = 512;
	float *fBuffer = new float[reflectMapSize*reflectMapSize*3*3];
	float *bBuffer = &fBuffer[reflectMapSize*reflectMapSize*3];

	initWidgets();
	try {
		for (int i=0; i < meshes; i++) {
			cout << "Loading " << objectFileName[i] << "..." << endl;
			initMesh(objectFileName[i], glMeshId[i]);
		}
		
		for (int i=0; i < probes; i++) {
			cout << "Loading " << hdrFileName[i] << "..." << endl;
			initHDRBackground(hdrFileName[i], hdrTexId[i*2], hdrTexId[i*2+1],
							  r[i], g[i], b[i], fBuffer, bBuffer, reflectMapSize);
		}
	} catch (exception e) {
		cerr << e.what() << endl;
		exit(-1);
	}

	initEnvSphere();

	updateProbeSH(currProbe);

	ry=90.f;
	motion(0,0);

	// clear extra memory
	delete[] fBuffer;
}

/// Mesh loader
void initMesh(const char *filename, unsigned int glId) {
	FILE *f = NULL;
	char bobFileName[128];
	strcpy(bobFileName, filename);
	int len = strlen(bobFileName);
	bobFileName[len-3] = 'b';
	bobFileName[len-2] = 'o';
	bobFileName[len-1] = 'b';

	if (!(f = fopen(bobFileName, "r"))) {
		//fclose(f);
		cout << "Creating Cached Binary Object " << bobFileName << "..." << endl;
		Wavefront *wavefront = new Wavefront(filename);
		BinaryObj::saveAsBinaryObj(bobFileName, wavefront);
		delete wavefront;
	} else
		fclose(f);
		

	BinaryObj *mesh = new BinaryObj(bobFileName);	
	
	glNewList(glId, GL_COMPILE);
	{	
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, sizeof(float)*4, &mesh->positions[0]);	
		
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, sizeof(float)*4, &mesh->normals[0]);

		glClientActiveTexture(GL_TEXTURE0_ARB);
		glTexCoordPointer(2, GL_FLOAT, sizeof(float)*4, &mesh->uvs[0]);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

		glClientActiveTexture(GL_TEXTURE6_ARB);
		glTexCoordPointer(3, GL_FLOAT, sizeof(float)*4, &mesh->tangents[0]);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

		glClientActiveTexture(GL_TEXTURE7_ARB);
		glTexCoordPointer(3, GL_FLOAT, sizeof(float)*4, &mesh->binormals[0]);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

		glDrawArrays(GL_TRIANGLES, 0, mesh->vertexCount);

		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glClientActiveTexture(GL_TEXTURE6_ARB);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glClientActiveTexture(GL_TEXTURE0_ARB);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	glEndList();

	delete mesh;
}

void initEnvSphere() {
	envId = glGenLists(1);
	GLUquadric* quadric = gluNewQuadric();

	envVShader = cgCreateProgramFromFile(context, CG_SOURCE, "EnvSphere.cg", VertexProfile, "EnvVShader", NULL);
	cgGLLoadProgram(envVShader);

	envFShader = cgCreateProgramFromFile(context, CG_SOURCE, "EnvSphere.cg", FragmentProfile, "EnvFShader", NULL);
	cgGLLoadProgram(envFShader);

	cgConnectParameter(keyGammaParam, cgGetNamedParameter(envFShader, "keyGamma"));

	glNewList(envId, GL_COMPILE);
		glDepthMask(GL_FALSE);
		glCullFace(GL_FRONT);
		glEnable(GL_CULL_FACE);
		
		gluSphere(quadric, 1.0, 100, 100);

		glDisable(GL_CULL_FACE);
		glDepthMask(GL_TRUE);
	glEndList();

	gluDeleteQuadric(quadric);
}

void initHDRBackground(const char *fileName, unsigned int front, unsigned int back,
					   float *r, float *g, float *b, float *fBuffer, float *bBuffer, unsigned int mapSize)
{
	HDRImage *image = new HDRImage(fileName);
	HDRProbe *probe = new HDRProbe(image);

	probe->ConstrutDualParabolicMap(fBuffer, bBuffer, mapSize, mapSize);

	glBindTexture(GL_TEXTURE_2D, front);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, mapSize, mapSize, 0, GL_RGB, GL_FLOAT, fBuffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, back);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, mapSize, mapSize, 0, GL_RGB, GL_FLOAT, bBuffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	probe->ConstructSHMatrices(r,g,b);

	delete probe;
	delete image;
}

/// Destroy Function ///
void destroy() {
	destroyWidgets();
	cgDestroy();
	
	free2D(r); free2D(g); free2D(b);

	for (int i=0; i < meshes; i++)
		glDeleteLists(glMeshId[i],  1);
	glDeleteLists(envId, 1);

	glDeleteTextures(probes*2, hdrTexId);
}

/// Display Function ///
void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// bind the dual paraboloid
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, hdrTexId[currProbe*2]);

	glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, hdrTexId[currProbe*2+1]);

	cgGLEnableProfile(VertexProfile);
	cgGLEnableProfile(FragmentProfile);

	displayEnvSphere();

	switch (type) {
		case LAFORTUNE:
			displayLafortune(); break;
		case WARD:
			displayWard(); break;
		case SV_LAFORTUNE:
			displaySVLafortune(); break;
	}

	cgGLDisableProfile(FragmentProfile);
	cgGLDisableProfile(VertexProfile);

	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D);

	if (showInterface) displayInterface();

	glutSwapBuffers();
}

void displayLafortune() {
	cgGLBindProgram(vShader);
	cgGLBindProgram(fShader[type]);

	glCallList(glMeshId[currMesh]);
}

void displayWard() {
	cgGLBindProgram(vShader);
	cgGLBindProgram(fShader[type]);
	
	glCallList(glMeshId[currMesh]);
}

void displaySVLafortune() {
	cgGLBindProgram(vShader);
	cgGLBindProgram(fShader[type]);

	glActiveTexture(GL_TEXTURE2);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, lobePainter->GetTexId());

	glCallList(glMeshId[currMesh]);

	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1);
}

void displayEnvSphere() {
	cgGLBindProgram(envVShader);
	cgGLBindProgram(envFShader);
	
	float m[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	m[12]=m[13]=m[14]=0.f;
	glPushMatrix();
	glLoadMatrixf(m);
	glCallList(envId);
	glPopMatrix();
}

void updateProbeSH(int probe) {
	cgGLSetMatrixParameterfc(rMParam, r[probe]);
	cgGLSetMatrixParameterfc(gMParam, g[probe]);
	cgGLSetMatrixParameterfc(bMParam, b[probe]);
}

/// Regenerates a new batch of samples, probably because
/// the number of samples changed
void updateSamples() {
	genSequence(samples);

	cgGLUnbindProgram(FragmentProfile);

	cgGLSetParameter1f(samplesParam, samples/4);
	cgGLSetParameter1f(sSamplesParam, samples);
	cgGLSetParameterArray4f(randParam, 0, samples/4, randnum);
	cgGLSetParameterArray4f(randcosParam, 0, samples/4, randcos);
	cgGLSetParameterArray4f(randsinParam, 0, samples/4, randsin);
	cgGLSetParameterArray4f(randlogParam, 0, samples/4, randlog);

	genLafortuneSamples(samples, cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()));
	genWardSamples(samples, ax->GetValue(), ay->GetValue());

	for (int i=0; i < 3; i++) 
		cgGLLoadProgram(fShader[i]);
}

/// Regenerates a new batch of samples, probably because
/// the sequence changed
void updateSequence() {
	genSequence(samples);

	cgGLSetParameterArray4f(randParam, 0, samples/4, randnum);
	cgGLSetParameterArray4f(randcosParam, 0, samples/4, randcos);
	cgGLSetParameterArray4f(randsinParam, 0, samples/4, randsin);
	cgGLSetParameterArray4f(randlogParam, 0, samples/4, randlog);

	genLafortuneSamples(samples, cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()));
	genWardSamples(samples, ax->GetValue(), ay->GetValue());
}

/// Reshape Function ///
void reshape(int w, int h) {
	wWidth = w;
	wHeight = h;

	glViewport(0,0,wWidth, wHeight);

	reshapeInterface(w,h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.f, ((GLdouble) wWidth)/((GLdouble) wHeight), 0.1, 1000.0);
	glMatrixMode(GL_MODELVIEW);

}

/// Mouse Function ///
void mouse(int button, int state, int x, int y) {
	ox=x; oy=y;
	mouseButton = button;

	if (showInterface) clickInterface(button,state,x,y);

	glutPostRedisplay();
}

/// Motion Function ///
void motion(int x, int y) {
	float m[16];

	if (showInterface && motionInterface(x,y)) {
		glutPostRedisplay();
		ox=x; oy=y;
		return;
	}

	if ((mouseButton == GLUT_LEFT) || (mouseButton == GLUT_MIDDLE_BUTTON)) {
		if (mouseButton == GLUT_LEFT) {
			rx += (y-oy);
			ry += (x-ox);
		} else {
			eyeDist += (y-oy)*0.25;
		}

		glLoadIdentity();
		if (showInterface) glTranslatef(xOffset,0,eyeDist);
		else glTranslatef(0,0,eyeDist);
		glRotatef(rx, 1,0,0);
		glRotatef(ry, 0,1,0);	

		// calculate the new eye position
		glPushMatrix();
		glLoadIdentity();
		
		glRotatef(-ry, 0,1,0);		
		glRotatef(-rx, 1,0,0);
		glTranslatef(0,0,5);

		glGetFloatv(GL_MODELVIEW_MATRIX, m);
		cgGLSetParameter3fv(eyeParam, &m[12]);

		glPopMatrix();
	} else {
		orx += (oy-y);
		ory += (ox-x);

		glPushMatrix();
		glLoadIdentity();
		glRotatef(orx, 1,0,0);
		glRotatef(ory, 0,1,0);
		
		glGetFloatv(GL_MODELVIEW_MATRIX, m);
		cgGLSetMatrixParameterfr(objectToWorldParam, m);

		glPopMatrix();
	}

	ox=x; oy=y;

	glutPostRedisplay();
}

/// Keyboard Function ///
void keyboard(unsigned char key, int x, int y) {
	switch (key) {
		case '1':
			currProbe=0; updateProbeSH(0);
			glutPostRedisplay();
			break;
		case '2':
			currProbe=1; updateProbeSH(1);
			glutPostRedisplay();
			break;
		case '3':
			currProbe=2; updateProbeSH(2);
			glutPostRedisplay();
			break;
		case '4':
			currMesh=0;
			glutPostRedisplay();
			break;
		case '5':
			currMesh=1;
			glutPostRedisplay();
			break;
		case '6':
			currMesh=2;
			glutPostRedisplay();
			break;
		case 'i':
		case 'I':
			showInterface = !showInterface;
			mouseButton  = GLUT_LEFT;
			motion(ox,oy);
			glutPostRedisplay();
			break;
		case 's':
		case 'S':
			saveFramebuffer();
			break;
		case 'p':
		case 'P':
			printStats(cout);
			break;
		case 27:
		case 'q':
		case 'Q':
			destroy();
			exit(0);
			break;
	}
}

/// Cg Initialization Function ///
void cgInit() {
	context = cgCreateContext();
	cgSetErrorCallback(cgError);

	VertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
	FragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);	
	
	// create common params
	samplesParam	= cgCreateParameter(context, CG_INT);
	sSamplesParam	= cgCreateParameter(context, CG_INT);
	cgSetParameterVariability(samplesParam, CG_LITERAL);
	cgSetParameterVariability(sSamplesParam, CG_LITERAL);

	cgGLSetParameter1f(sSamplesParam, samples);

	// truly global params
	keyGammaParam	= cgCreateParameter(context, CG_FLOAT2);
	eyeParam		= cgCreateParameter(context, CG_FLOAT3);
	specularParam	= cgCreateParameter(context, CG_FLOAT3);
	diffuseParam	= cgCreateParameter(context, CG_FLOAT3);

	cgSetAutoCompile(context, CG_COMPILE_MANUAL);

	// spherical harmonics vertex shader
	objectToWorldParam = cgCreateParameter(context, CG_FLOAT4x4);

	vShader = cgCreateProgramFromFile(context, CG_SOURCE, "SHVertShader.cg", VertexProfile, NULL, NULL);

	rMParam			= cgGetNamedParameter(vShader, "rM");
	gMParam			= cgGetNamedParameter(vShader, "gM");
	bMParam			= cgGetNamedParameter(vShader, "bM");

	objectToWorldParam = cgGetNamedParameter(vShader, "objectToWorld");

	const float identity[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0, 0,0,0,1};
	cgGLSetMatrixParameterfr(objectToWorldParam, identity);

	cgCompileProgram(vShader);
	cgGLLoadProgram(vShader);

	// GENERATE SAMPLES
	genSequence(samples);

	// load lafortune shader
	fShader[LAFORTUNE] = cgCreateProgramFromFile(context, CG_SOURCE, "Lafortune.cg", FragmentProfile, NULL, NULL);

	// connect the lafortune parameters
	lSmplsParam = cgGetNamedParameter(fShader[LAFORTUNE], "smpls");
	lScalesParam = cgGetNamedParameter(fShader[LAFORTUNE], "fs");
	lobeParam = cgGetNamedParameter(fShader[LAFORTUNE], "lobe");
	cgConnectParameter(sSamplesParam, cgGetNamedParameter(fShader[LAFORTUNE], "samples"));

	genLafortuneSamples(samples, -1.f, 1.f, powf(10.f,1.2f));
	cgGLSetParameter4f(lobeParam, -1.f, -1.f, 1.f, powf(10.f,1.2f));

	// load ward shader
	fShader[WARD] = cgCreateProgramFromFile(context, CG_SOURCE, "Ward.cg", FragmentProfile, NULL, NULL);

	wSmplsParam = cgGetNamedParameter(fShader[WARD], "smpls");
	wScalesParam = cgGetNamedParameter(fShader[WARD], "fs");
	alphaParam = cgGetNamedParameter(fShader[WARD], "alpha");
	cgConnectParameter(sSamplesParam, cgGetNamedParameter(fShader[WARD], "samples"));

	genWardSamples(samples, 0.1f, 0.1f);


	fShader[SV_LAFORTUNE] = cgCreateProgramFromFile(context, CG_SOURCE, "svlafortune.cg", FragmentProfile, "SVLafortune", NULL);
	
	cgConnectParameter(samplesParam, cgGetNamedParameter(fShader[SV_LAFORTUNE], "sampleGroups"));
		
	randParam = cgGetNamedParameter(fShader[SV_LAFORTUNE], "rand");
	randcosParam = cgGetNamedParameter(fShader[SV_LAFORTUNE], "randcos");
	randsinParam = cgGetNamedParameter(fShader[SV_LAFORTUNE], "randsin");
	randlogParam = cgGetNamedParameter(fShader[SV_LAFORTUNE], "randlog");

	cgGLSetParameter1f(samplesParam, samples/4);
	cgGLSetParameterArray4f(randParam, 0, samples/4, randnum);
	cgGLSetParameterArray4f(randcosParam, 0, samples/4, randcos);
	cgGLSetParameterArray4f(randsinParam, 0, samples/4, randsin);
	cgGLSetParameterArray4f(randlogParam, 0, samples/4, randlog);

	// attach all common params
	for (int i=0; i < 3; i++) {

		cgConnectParameter(specularParam, cgGetNamedParameter(fShader[i], "specularAlbedo"));
		cgConnectParameter(diffuseParam, cgGetNamedParameter(fShader[i], "diffuseAlbedo"));

		cgConnectParameter(eyeParam, cgGetNamedParameter(fShader[i], "eye"));
		cgConnectParameter(keyGammaParam, cgGetNamedParameter(fShader[i], "keyGamma"));

		cgCompileProgram(fShader[i]);
		cgGLLoadProgram(fShader[i]);
	}

	cgGLSetParameter3f(specularParam, 0.5f, 0.5f, 0.5f);
	cgGLSetParameter3f(diffuseParam, 0.15f, 0.15f, 0.15f);

	cgSetAutoCompile(context, CG_COMPILE_IMMEDIATE);
}

/// Cg Destroy Function ///
void cgDestroy() {
	cgDestroyProgram(vShader);
	for (int i=0; i < 3; i++)
		cgDestroyProgram(fShader[i]);

	cgDestroyProgram(envVShader);
	cgDestroyProgram(envFShader);
	cgDestroyParameter(eyeParam);
	cgDestroyParameter(specularParam);
	cgDestroyParameter(diffuseParam);
	cgDestroyParameter(keyGammaParam);

	cgDestroyParameter(samplesParam);
	cgDestroyParameter(sSamplesParam);

	cgDestroyContext(context);
}

/// Cg Error Callback Function
void cgError() {
	cerr << cgGetErrorString(cgGetError()) << endl;
	const char *details = cgGetLastListing(context);
	if (details) cerr << details << endl;
	exit(-1);
}

bool getFilename(char *buffer, size_t bufferSize, bool loading) {
	HWND hwnd = GetActiveWindow();	// owner window	
	OPENFILENAMEA ofn;				// common dialog box structure

	// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = buffer;
	//
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	//
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = (DWORD) bufferSize;
	ofn.lpstrFilter = "HDR - Radiance RGBE files\0*.hdr\0All\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.lpstrDefExt = "hdr";
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 
	if (loading)
		return (GetOpenFileNameA(&ofn) == TRUE);
	else
		return (GetSaveFileNameA(&ofn) == TRUE);
}

void saveFramebuffer() {
	char filename[256];
	if (!getFilename(filename, 256, false)) return;

	GLuint fb, rb, fbTex;

	glGenFramebuffersEXT(1,&fb);
	glGenRenderbuffersEXT(1, &rb);
	glGenTextures(1, &fbTex);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);

	// create texture		
	glBindTexture(GL_TEXTURE_2D, fbTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, wWidth, wHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// attach the color channel to the frame buffer
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbTex, 0);
	
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rb);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, wWidth, wHeight);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, rb);

	display();

	float *data = new float[wWidth*wHeight*3];

	glReadPixels(0,0, wWidth, wHeight, GL_RGB, GL_FLOAT, data);
	
	HDRImage::WriteImage(filename, data, wWidth, wHeight);
	delete[] data;
	
	// clean up the framebuffer
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	glDeleteRenderbuffersEXT(1, &rb);
	glDeleteFramebuffersEXT(1, &fb);
	glDeleteTextures(1, &fbTex);
}

/// Outputs the main 
void printStats(ostream &fout) {
	fout << "Sampling" << endl;
	fout << "Sample Type: " << seqOptions[seqSelector.GetCurrOption()] << endl;
	fout << "Samples:     " << smplOptions[smplSelector.GetCurrOption()] << endl;
	fout << endl;
	
	fout << "Tonemapping" << endl;
	fout << "Exposue Key: " << exposureKey->GetValue() << endl;
	fout << "Gamma:       " << gamma->GetValue() << endl;
	fout << endl;

	fout << "Lafortune" << endl;
	fout << "cxy:         " << cxy->GetValue() << endl;
	fout << "cz:          " << cz->GetValue() << endl;
	fout << "n:           " << powf(10.f,n->GetValue()) << endl;
	fout << "ps_r:        " << ps_r->GetValue() << endl;
	fout << "ps_g:        " << ps_g->GetValue() << endl;
	fout << "ps_b:        " << ps_b->GetValue() << endl;
	fout << endl;

	fout << "Ward" << endl;
	fout << "ax:          " << ax->GetValue() << endl;
	fout << "ay:          " << ay->GetValue() << endl;
	fout << "ps_r:        " << wps_r->GetValue() << endl;
	fout << "ps_g:        " << wps_g->GetValue() << endl;
	fout << "ps_b:        " << wps_b->GetValue() << endl;
	fout << endl;

	fout << "Diffuse" << endl;
	fout << "pd_r:        " << pd_r->GetValue() << endl;
	fout << "pd_g:        " << pd_g->GetValue() << endl;
	fout << "pd_b:        " << pd_b->GetValue() << endl;
	fout << endl;
}

/// Main Function ///
int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowPosition(10,10);
	glutInitWindowSize(512,550);

	if (!glutCreateWindow("StochasticGPU")) {
		cerr << "Unable to Create Window" << endl;
		return -1;
	}

	if (glewInit() != GLEW_OK) {
		cerr << "Unable to Initialze GLEW" << endl;
		return -1;
	}

	init();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);

	glutMainLoop();

	return 0;
}