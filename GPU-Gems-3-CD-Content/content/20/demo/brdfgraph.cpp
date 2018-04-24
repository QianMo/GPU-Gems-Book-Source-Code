//
// brdfgraph.cpp
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

#define _USE_MATH_DEFINES
#include <cmath>

#include "alloc2d.h"
#include "brdfgraph.h"
#include "gstream.h"

#include <iostream>
using namespace std;

extern CGprofile VertexProfile;
extern CGcontext context;

BRDFGraph::BRDFGraph() : state(ROTATE), active(false), dist(-2.0f), x(128+5), y(0), w(256), h(128), rx(0), ry(0), worot(45.f), ox(0), oy(0), type(0) {
	initHemisphere();
	initCg();
}

BRDFGraph::~BRDFGraph() {
	glDeleteLists(hemiId, 1);
	cgDestroyProgram(vertProg[0]);
	cgDestroyProgram(vertProg[1]);
}

void BRDFGraph::setWo() { 
	float rads = worot/180.f*M_PI;
	wo = Vector3D(cosf(rads), 0.f, sinf(rads));
	cgGLSetParameter3fv(woParam, (float*) &wo);
}

void BRDFGraph::initCg() {
	woParam = cgCreateParameter(context, CG_FLOAT3);

	vertProg[0] = cgCreateProgramFromFile(context, CG_SOURCE, "brdfdisplay.cg", VertexProfile, "Lafortune", NULL);
	cgGLLoadProgram(vertProg[0]);
	
	lobeParam = cgGetNamedParameter(vertProg[0], "lobe");

	vertProg[1] = cgCreateProgramFromFile(context, CG_SOURCE, "brdfdisplay.cg", VertexProfile, "Ward", NULL);
	cgGLLoadProgram(vertProg[1]);
	
	alphaParam = cgGetNamedParameter(vertProg[1], "alphaWeight");

	diffuseParam = cgGetNamedParameter(vertProg[0], "diffuse");

	// shared parameters
	for (int i=0; i < 2; i++) {
		cgConnectParameter(woParam, cgGetNamedParameter(vertProg[i], "wo"));
	}

	setWo();
}

void BRDFGraph::setLobe(float cxy, float cz, float n, float weight) {
	cgGLSetParameter4f(lobeParam, cxy, cz, n, weight);
}

void BRDFGraph::setWardLobe(float ax, float ay, float weight) {
	cgGLSetParameter3f(alphaParam, ax, ay, weight);
}

void BRDFGraph::setDiffuse(float diffuse) {
	cgSetParameter1f(diffuseParam, diffuse);
}

void BRDFGraph::initHemisphere() {
	// build the hemisphere mesh used for rendering the BRDF graph
	const int thetaDivs = 100;
	const int phiDivs = 100;

	float thetaStep = M_PI/2.f/((float) thetaDivs-1);
	float phiStep = 2.f*M_PI/((float) phiDivs-1);

	Vector3D **v = alloc2D<Vector3D>(thetaDivs, phiDivs);

	float thetaPos = 0.f;
	for (int i=0; i < thetaDivs; i++,thetaPos+=thetaStep) {
		float costheta = cosf(thetaPos);
		float sintheta = sinf(thetaPos);

		float phiPos=0.f;
		for	(int j=0; j < phiDivs; j++,phiPos+=phiStep) {
			float cosphi = cosf(phiPos);
			float sinphi = sinf(phiPos);

			v[i][j] = Vector3D(costheta*cosphi, costheta*sinphi, sintheta);
		}
	}

	unsigned short **index = alloc2D<unsigned short>((phiDivs-1)*2+2, thetaDivs-1);

	for (int i=0; i < thetaDivs-1; i++) {
		index[i][0] = (i+1)*phiDivs;
		index[i][1] = i*phiDivs;

		int k=2;
		for (int j=0; j < phiDivs-1; j++) {		
			index[i][k++] = (i+1)*phiDivs + j+1;
			index[i][k++] = i*phiDivs + j+1;	
		}
	}

	// build the 2d function displayererer
	hemiId = glGenLists(1);
	glNewList(hemiId, GL_COMPILE);

		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, sizeof(float)*4, &v[0][0]);

		for (int i=0; i < thetaDivs-1; i++)
			glDrawElements(GL_QUAD_STRIP, (phiDivs-1)*2+2, GL_UNSIGNED_SHORT, index[i]);

		glDisableClientState(GL_VERTEX_ARRAY);

	glEndList();

	free2D(index);
	free2D(v);
}

bool BRDFGraph::Click(int button, int mstate, int mx, int my) {
	if ((mstate == GLUT_DOWN) && Contains(mx,my)) {
		active=true;
		ox=mx; oy=my;
		if (button==GLUT_RIGHT_BUTTON) {
			if (glutGetModifiers() & GLUT_ACTIVE_CTRL) {
				state = ZOOM;
			} else {
				state = ROTATE;
			}
		} else {
			state = ROTWO;
		}

		return true;
	}

	active=false;
	return false;
}

bool BRDFGraph::Motion(int mx, int my) {
	if (active) {
		if (state == ROTATE) {
			rx += my-oy;
			ry += mx-ox;
		} else if (state == ROTWO) {
			worot += my-oy;
			worot = max(0, min(90,worot));
			setWo();
		} else if (state == ZOOM) {
			dist -= (my-oy)*0.1f;			
		}
		ox=mx; oy=my;
		return true;
	}
	return false;
}

void BRDFGraph::Display() {
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(x,fy,w,h);

	glClear(GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(-1.5f,0.5f,0.f,2.f);
	gluPerspective(45., ((float) w)/((float) h), 0.1, 100.);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glTranslatef(0.f,0.f,dist);
	glRotatef(-90.f, 1.f,0.f,0.f);
	glRotatef(ry, 0.f,0.f,1.f);
	
	glColor3f(1,1,0);
	glLineWidth(2.f);
	glBegin(GL_LINES);
		glVertex3f(0.f,0.f,0.f);
		glVertex3fv((float*) &wo);
	glEnd();
	glLineWidth(1.f);

	cgGLBindProgram(vertProg[type]);
	cgGLEnableProfile(VertexProfile);

		glEnable(GL_CULL_FACE);

		// Since the hemisphere is linked backwards, we cull the front face
		glCullFace(GL_FRONT);
		glCallList(hemiId);

		glDisable(GL_CULL_FACE);

	cgGLDisableProfile(VertexProfile);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glPopAttrib();
}
