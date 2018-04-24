//
// lobepainter.cpp
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

#include <GL/glew.h>
#include "lobepainter.h"
#include "brdfgraph.h"

extern CGprofile FragmentProfile, VertexProfile;
extern CGcontext context;
extern GLuint glMeshId[3];
extern int currMesh;

/// Simple function from nVidia for checking the FBO status
void CheckFramebufferStatus() {	
    GLenum status;
    status = (GLenum) glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            printf("Unsupported framebuffer format\n");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            printf("Framebuffer incomplete, missing attachment\n");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            printf("Framebuffer incomplete, attached images must have same dimensions\n");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
            printf("Framebuffer incomplete, attached images must have same format\n");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            printf("Framebuffer incomplete, missing draw buffer\n");
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            printf("Framebuffer incomplete, missing read buffer\n");
            break;
        default:
            exit(-1);
    }
}

LobePainter::LobePainter(BRDFGraph *_brdfGraph, int _offsetX, int y, int displayWidth, int _resWidth) :
						 Container(_offsetX, y, displayWidth, displayWidth), brdfGraph(_brdfGraph),
						 offsetX(_offsetX), resWidth(_resWidth),
						 cxy("c_xy",		30, -1.f, -1.05f, 1.05f,	 5, 15, displayWidth-10, 10),
						 cz("c_z",			30,  1.f,   0.f,  1.f,	 5, 30, displayWidth-10, 10),
						 n("n",				30, 1.2f,   1.f, 3.2f,   5, 45, displayWidth-10, 10),
						 weight("weight",	30, 0.5f,   0.1f, 1.0f,	 5, 60, displayWidth-10, 10),
						 diffuse("diffuse",	30, 0.5f,   0.f,  1.f,	 5, 75, displayWidth-10, 10),
						 size("size",		30, 0.1f,   0.f,  0.5f,	 5, 90, displayWidth-10, 10)
{
	controls = new WidgetGroup("Lobe Painter", offsetX, y+displayWidth+5, displayWidth, 105, WidgetGroup::RIGHT);
	controls->AddWidget(&cxy);
	controls->AddWidget(&cz);
	controls->AddWidget(&n);
	controls->AddWidget(&weight);
	controls->AddWidget(&diffuse);
	controls->AddWidget(&size);
	controls->Compile();
	
	InitCg();
	InitQuad();
	InitFramebuffers();
	
	size.SetValue(1.f);
	Paint(0.5f,0.5f);
	size.SetValue(0.1f);
}

LobePainter::~LobePainter() {
	delete controls;

	glDeleteFramebuffersEXT(2, fb);
	glDeleteTextures(2, fbTex);

	cgDestroyProgram(paintProg);
	cgDestroyProgram(displayProg);
	cgDestroyProgram(uvProg);
}

void LobePainter::Display() {

	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(d.x,fy,d.w,d.h);

	// display the texture
	glMatrixMode(GL_PROJECTION);
	glPushMatrix(); glLoadIdentity();
	gluOrtho2D(0,resWidth,resWidth, 0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix(); glLoadIdentity();

	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, fbTex[currTex]);
	cgGLBindProgram(displayProg);
	cgGLEnableProfile(FragmentProfile);
		glCallList(quadId);
	cgGLDisableProfile(FragmentProfile);

	glDisable(GL_TEXTURE_2D);

	// display the mesh
	glLineWidth(1.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		cgGLBindProgram(uvProg);
		cgGLEnableProfile(VertexProfile);
			glCallList(glMeshId[currMesh]);
		cgGLDisableProfile(VertexProfile);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glPopAttrib();
	
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	// display the controls
	controls->Display();
}

void LobePainter::updateGraph() {
	float exp = powf(10,n.GetValue());
	float scale = powf(weight.GetValue(), 1.f/exp);
	brdfGraph->setLobe(scale*cxy.GetValue(), scale*cz.GetValue(), exp, 1.f);
	brdfGraph->setDiffuse(diffuse.GetValue());
}

bool LobePainter::Click(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		if (Contains(x,y)) {
			float rx = ((float) x-d.x)/((float) d.w);
			float ry = ((float) y-d.y)/((float) d.h);

			Paint(rx,ry);

			return true;
		} else if (controls->Click(button,state,x,y)) {
			updateGraph();
			return true;
		}
	}
	return false;

}
bool LobePainter::Motion(int x, int y) {
	if (Contains(x,y)) {
		float rx = ((float) x-d.x)/((float) d.w);
		float ry = ((float) y-d.y)/((float) d.h);

		Paint(rx,ry);

		return true;
	} else if (controls->Motion(x,y)) {		
		updateGraph();
		return true;
	}
	return false;
}

void LobePainter::Paint(float x, float y) {

	glPushAttrib(GL_VIEWPORT_BIT);

	// set up the framebuffer
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[!currTex]);
	glViewport(0,0,256,256);

	glDisable(GL_BLEND);

	glClear(GL_COLOR_BUFFER_BIT);

	// set up the matrix
	glMatrixMode(GL_PROJECTION);
	glPushMatrix(); glLoadIdentity();
	gluOrtho2D(0,resWidth, 0.f, resWidth);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix(); glLoadIdentity();

	cgGLBindProgram(paintProg);
	cgGLEnableProfile(FragmentProfile);

	// set up the painting parameters
	cgGLSetParameter3f(posParam, x,y,1.f/(0.5f*(size.GetValue()*size.GetValue())));
	float exp = powf(10,n.GetValue());
	float scale = powf(weight.GetValue(), 1.f/exp);
	cgGLSetParameter4f(lobeParam, scale*cxy.GetValue(), scale*cz.GetValue(), exp, diffuse.GetValue());

	glEnable(GL_TEXTURE_2D);

	// bind the previous rendered texture
	glBindTexture(GL_TEXTURE_2D, fbTex[currTex]);
	glCallList(quadId);

	glDisable(GL_TEXTURE_2D);

	cgGLDisableProfile(FragmentProfile);

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glPopAttrib();

	glEnable(GL_BLEND);

	// swap which buffer we read from
	currTex = !currTex;

}

void LobePainter::InitFramebuffers() {
	glGenFramebuffersEXT(2, fb);
	glGenTextures(2, fbTex);

	for (int i=0; i < 2; i++) {
		// create the GPU memory
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb[i]);
		{
			// create texture		
			glBindTexture(GL_TEXTURE_2D, fbTex[i]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, resWidth, resWidth, 0, GL_RGBA, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

			// attach the color channel to the frame buffer
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbTex[i], 0);
		}
		// check for problems
		CheckFramebufferStatus();
	}

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	glClearColor(0,0,0,0);
}

void LobePainter::InitQuad() {
	quadId = glGenLists(1);
	glNewList(quadId, GL_COMPILE);
		glBegin(GL_QUADS);
			glTexCoord2f(0.f,0.f); glVertex2i(0,0);
			glTexCoord2f(0.f,1.f); glVertex2i(0,resWidth);
			glTexCoord2f(1.f,1.f); glVertex2i(resWidth,resWidth);
			glTexCoord2f(1.f,0.f); glVertex2i(resWidth,0);
		glEnd();
	glEndList();
}

void LobePainter::InitCg() {
	displayProg = cgCreateProgramFromFile(context, CG_SOURCE, "lobepaint.cg", FragmentProfile, "FalseColorDisplay", NULL);
	cgGLLoadProgram(displayProg);
	
	paintProg = cgCreateProgramFromFile(context, CG_SOURCE, "lobepaint.cg", FragmentProfile, "Paint", NULL);
	cgGLLoadProgram(paintProg);
	
	lobeParam = cgGetNamedParameter(paintProg, "lobe");
	posParam = cgGetNamedParameter(paintProg, "pos");

	uvProg = cgCreateProgramFromFile(context, CG_SOURCE, "lobepaint.cg", VertexProfile, "UVDisplay", NULL);
	cgGLLoadProgram(uvProg);
}
