//
// widgets.cpp
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

// Most of the following code is just the detailed implementation
// of the manual GUI system used in this interface.  Little
// of the code has anything to do with the rendering algorithm.

#include <GL/glew.h>

#include "widgets.h"
#include "color.h"
#include "sequencegen.h"

extern void updateSequence();
extern void updateSamples();

extern CGparameter lobeParam, alphaParam, specularParam, diffuseParam;

BRDFType type = LAFORTUNE;

BRDFGraph	*brdfGraph		= NULL;

/// Tone Mapping Widgets
WidgetGroup	*toneMapGrp		= NULL;
SliderBar	*exposureKey	= NULL;
SliderBar	*gamma			= NULL;

/// Lafortune Widgets
WidgetGroup *lobeGrp;
SliderBar	*cxy;
SliderBar	*cz;
SliderBar	*n;
SliderBar	*ps_r, *ps_g, *ps_b;

WidgetGroup *diffuseGrp		= NULL;
SliderBar	*pd_r, *pd_g, *pd_b;

/// Ward Widgets
WidgetGroup *wardLobeGrp;
SliderBar	*ax;
SliderBar	*ay;
SliderBar	*wps_r, *wps_g, *wps_b;
WidgetGroup *wardDiffuseGrp		= NULL;

SetOptionButton<int> probeButtons[3] = { 
	SetOptionButton<int>("Cathedral (1)",		5,  15,75,20, 0, &currProbe), 
	SetOptionButton<int>("Ennis (2)",		85, 15,75,20, 1, &currProbe),
	SetOptionButton<int>("Kitchen (3)",	165,15,75,20, 2, &currProbe) };
WidgetGroup	*probeGroup = NULL;

SetOptionButton<int> meshButtons[3] = {
	SetOptionButton<int>("Sphere (4)",		5,  15,75,20, 0, &currMesh), 
	SetOptionButton<int>("Buddha (5)",		85, 15,75,20, 1, &currMesh),
	SetOptionButton<int>("Athena (6)",	165, 15,75,20, 2, &currMesh) };
WidgetGroup *meshGroup = NULL;

SetOptionButton<BRDFType> BRDFButtons[3] = {
	SetOptionButton<BRDFType>("Lafortune",	5, 15,60,20, LAFORTUNE, &type),
	SetOptionButton<BRDFType>("Ward",	   70, 15,60,20, WARD, &type),
	SetOptionButton<BRDFType>("SBRDF",	  135, 15,60,20, SV_LAFORTUNE, &type)};
WidgetGroup *brdfGroup = NULL;

LobePainter *lobePainter;

const char *seqOptions[] = {"Halton",
						 "Folded Halton",
						 "Hammersley",
						 "Folded Hammersley",
						 "Possion Disk",
						 "Best Candidate",
						 "Penrose"};
ScrollSelector seqSelector("sequence:", seqOptions, 7, 5,12,190);

const char *smplOptions[] = {"4", "8","12","16","20","24","28","32","36","40"};
ScrollSelector smplSelector("samples:", smplOptions, 10, 5,25,190);

WidgetGroup *smplGroup = NULL;

void initWidgets() {
	brdfGraph = new BRDFGraph();
	lobePainter = new LobePainter(brdfGraph, 5, 145, 200, 256);
	initSamplingWidgets();
	initLafortuneWidgets();
	initWardWidgets();
	initToneMappingWidgets();
	initProbeGroup();
	initMeshGroup();
	initBRDFGroup();
}

#include <iostream>
using namespace std;

void initSamplingWidgets() {
	smplGroup = new WidgetGroup("Sampling", 5,5,200,40, WidgetGroup::RIGHT);
	smplGroup->AddWidget(&seqSelector);
	smplGroup->AddWidget(&smplSelector);

	seqSelector.SetCurrOption((int) currSequenceType);
	smplSelector.SetCurrOption(samples/4-1);

	smplGroup->Compile();
}

void initProbeGroup() {
	probeGroup = new WidgetGroup("Probes", 5,5,245,15+20+5, WidgetGroup::LEFT);
	probeGroup->AddWidget(&probeButtons[0]);
	probeGroup->AddWidget(&probeButtons[1]);
	probeGroup->AddWidget(&probeButtons[2]);
	probeGroup->Compile();
}

void initMeshGroup() {
	meshGroup = new WidgetGroup("Meshes", 5,5,245,15+20+5, WidgetGroup::BOTTOM);
	meshGroup->AddWidget(&meshButtons[0]);
	meshGroup->AddWidget(&meshButtons[1]);
	meshGroup->AddWidget(&meshButtons[2]);
	meshGroup->Compile();
}

void initBRDFGroup() {
	brdfGroup = new WidgetGroup("BRDF Models", 5, 100, 200,15+20+5, WidgetGroup::RIGHT);
	brdfGroup->AddWidget(&BRDFButtons[0]);
	brdfGroup->AddWidget(&BRDFButtons[1]);
	brdfGroup->AddWidget(&BRDFButtons[2]);
	brdfGroup->Compile();
} 

void initLafortuneWidgets() {
	lobeGrp = new WidgetGroup("Lafortune Lobe", 5,145,200,105, WidgetGroup::RIGHT);

	cxy		= new SliderBar("c_xy",		30, -1.f, -1.05f, 1.05f,	 5, 15, 190, 10);	lobeGrp->AddWidget(cxy);
	cz		= new SliderBar("c_z",		30,  1.f,   0.f,  1.f,	 5, 30, 190, 10);	lobeGrp->AddWidget(cz);
	n		= new SliderBar("n",		30, 1.2f,   1.f, 3.2f,	 5, 45, 190, 10);	lobeGrp->AddWidget(n);
	ps_r	= new SliderBar("red",		30,  0.5f,	0.f,  1.f,	 5, 60, 190, 10);	lobeGrp->AddWidget(ps_r);
	ps_g	= new SliderBar("green",	30,  0.5f,	0.f,  1.f,	 5, 75, 190, 10);	lobeGrp->AddWidget(ps_g);
	ps_b	= new SliderBar("blue",		30,  0.5f,	0.f,  1.f,	 5, 90, 190, 10);	lobeGrp->AddWidget(ps_b);

	lobeGrp->Compile();

	diffuseGrp = new WidgetGroup("Diffuse", 5,145+110,200,60, WidgetGroup::RIGHT);
	pd_r		= new SliderBar("red",		30,  0.15f,	0.f,  1.f,	 5, 15, 190, 10);	diffuseGrp->AddWidget(pd_r);
	pd_g		= new SliderBar("green",	30,  0.15f,	0.f,  1.f,	 5, 30, 190, 10);	diffuseGrp->AddWidget(pd_g);
	pd_b		= new SliderBar("blue",		30,  0.15f,	0.f,  1.f,	 5, 45, 190, 10);	diffuseGrp->AddWidget(pd_b);
	diffuseGrp->Compile();

	brdfGraph->setLobe(cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()), 
					   Color3(ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue()).y());
	brdfGraph->setDiffuse(Color3(pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue()).y());

}

void initWardWidgets() {
	wardLobeGrp = new WidgetGroup("Ward Lobe", 5,145,200,90, WidgetGroup::RIGHT);

	ax			= new SliderBar("a_x",		30, 0.1f, 0.001f, 0.2f,	5, 15, 190, 10);	wardLobeGrp->AddWidget(ax);
	ay			= new SliderBar("a_y",		30, 0.1f, 0.001f, 0.2f,	5, 30, 190, 10);	wardLobeGrp->AddWidget(ay);
	wps_r		= new SliderBar("red",		30,  1.f,	0.f,  1.f,	 5, 45, 190, 10);	wardLobeGrp->AddWidget(wps_r);
	wps_g		= new SliderBar("green",	30,  1.f,	0.f,  1.f,	 5, 60, 190, 10);	wardLobeGrp->AddWidget(wps_g);
	wps_b		= new SliderBar("blue",		30,  1.f,	0.f,  1.f,	 5, 75, 190, 10);	wardLobeGrp->AddWidget(wps_b);

	wardLobeGrp->Compile();

	wardDiffuseGrp = new WidgetGroup("Diffuse", 5,145+95,200,60, WidgetGroup::RIGHT);
	wardDiffuseGrp->AddWidget(pd_r);
	wardDiffuseGrp->AddWidget(pd_g);
	wardDiffuseGrp->AddWidget(pd_b);
	wardDiffuseGrp->Compile();	

	brdfGraph->setWardLobe(ax->GetValue(), ay->GetValue(), 
						   Color3(wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue()).y());
}

void initToneMappingWidgets() {
	toneMapGrp		= new WidgetGroup("Tonemapping", 5, 50, 200, 45, WidgetGroup::RIGHT);
	exposureKey		= new SliderBar("key",		30, 0.f,  -3.f, 3.f,	5, 15, 190, 10);	toneMapGrp->AddWidget(exposureKey);
	gamma			= new SliderBar("gamma",	30, 1.8f,  0.5f, 3.f,	5, 30, 190, 10);	toneMapGrp->AddWidget(gamma);
	toneMapGrp->Compile();

	cgGLSetParameter2f(keyGammaParam, powf(10.f,exposureKey->GetValue()), 1.f/gamma->GetValue());
}

void destroyWidgets() {
	delete brdfGraph;
	delete lobePainter;

	delete probeGroup;
	delete meshGroup;
	delete brdfGroup;

	delete lobeGrp;
	delete cxy;
	delete cz;
	delete n;
	delete ps_r;
	delete ps_g;
	delete ps_b;

	delete wardLobeGrp;
	delete ax;
	delete ay;
	delete wps_r;
	delete wps_g;
	delete wps_b;

	delete pd_r;
	delete pd_g;
	delete pd_b;

	delete wardDiffuseGrp;
	delete diffuseGrp;

	delete toneMapGrp;
	delete exposureKey;
	delete gamma;

	delete smplGroup;
}

void displayInterface() {
	// display the interface
	glColor3f(1,1,1);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION); 
	glPushMatrix(); 
	glLoadIdentity();
	gluOrtho2D(0,wWidth, wHeight, 0);
	glMatrixMode(GL_MODELVIEW);

	glDisable(GL_DEPTH_TEST);

	smplGroup->Display();
	probeGroup->Display();
	meshGroup->Display();
	brdfGroup->Display();
	toneMapGrp->Display();
	
	if (type == LAFORTUNE) {
		lobeGrp->Display();
		diffuseGrp->Display();
	} else if (type == WARD) {
		wardLobeGrp->Display();
		wardDiffuseGrp->Display();
	} else if (type == SV_LAFORTUNE) {
		lobePainter->Display();
	}

	glEnable(GL_DEPTH_TEST);

	glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);

	brdfGraph->Display();
}

void reshapeInterface(int w, int h) {
	smplGroup->WindowReshape(w,h);
	toneMapGrp->WindowReshape(w,h);

	lobeGrp->WindowReshape(w,h);
	wardLobeGrp->WindowReshape(w,h);

	diffuseGrp->WindowReshape(w,h);
	wardDiffuseGrp->WindowReshape(w,h);

	brdfGroup->WindowReshape(w,h);
	probeGroup->WindowReshape(w,h);
	meshGroup->WindowReshape(w,h);

	brdfGraph->WindowReshape(w,h);
	lobePainter->WindowReshape(w,h);
}

void clickInterface(int button, int state, int x, int y) {

	if (type == SV_LAFORTUNE && lobePainter->Click(button,state,x,y)) {
		;
	} else if (brdfGraph->Click(button,state,x,y)) {

	} else if (toneMapGrp->Click(button,state,x,y)) {
		cgGLSetParameter2f(keyGammaParam, powf(10.f,exposureKey->GetValue()), 1.f/gamma->GetValue());
	} else if (probeGroup->Click(button,state,x,y)) {
		updateProbeSH(currProbe);
	} else if (meshGroup->Click(button,state,x,y)) {

	} else if (brdfGroup->Click(button,state,x,y)) {		
		if (state==GLUT_UP) return;

		if (type==SV_LAFORTUNE) {
			brdfGraph->setType(0);
			lobePainter->updateGraph();
		} else {
			brdfGraph->setType(type);
			if (type == LAFORTUNE) {
				brdfGraph->setLobe(cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()), 
								   Color3(ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue()).y());
				
				brdfGraph->setDiffuse(Color3(pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue()).y());
				
				cgGLSetParameter3f(specularParam, ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue());
				cgGLSetParameter3f(diffuseParam, pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue());		
			} else if (type == WARD) {
				brdfGraph->setWardLobe(ax->GetValue(), ay->GetValue(),
									   Color3(wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue()).y());
				
				cgGLSetParameter2f(alphaParam, ax->GetValue(), ay->GetValue());
				cgGLSetParameter3f(specularParam, wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue());
				cgGLSetParameter3f(diffuseParam, pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue());		
			}
		}
	} else if (smplGroup->Click(button, state,x,y)) {
		if (state==GLUT_UP) return;

		if (currSequenceType != seqSelector.GetCurrOption()) {
			currSequenceType = (SequenceType) seqSelector.GetCurrOption();
			updateSequence();
		} else if (samples != (smplSelector.GetCurrOption()+1)*4) {
			samples = (smplSelector.GetCurrOption()+1)*4;
			updateSamples();
		}

	} else {
		if (type == LAFORTUNE) {
			if		(lobeGrp->Click(button,state,x,y)) {
				brdfGraph->setLobe(cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()), 
								   Color3(ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue()).y());

				genLafortuneSamples(samples, cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()));

				cgGLSetParameter4f(lobeParam, cxy->GetValue(), cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()));
				cgGLSetParameter3f(specularParam, ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue());

			} else if (diffuseGrp->Click(button,state,x,y)) {
				brdfGraph->setDiffuse(Color3(pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue()).y());
				cgGLSetParameter3f(diffuseParam, pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue());
			}
		} else if (type == WARD)  {
			if		(wardLobeGrp->Click(button,state,x,y)) {
				brdfGraph->setWardLobe(ax->GetValue(), ay->GetValue(), 
									   Color3(wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue()).y());
				
				genWardSamples(samples, ax->GetValue(), ay->GetValue());

				cgGLSetParameter2f(alphaParam, ax->GetValue(), ay->GetValue());
				cgGLSetParameter3f(specularParam, wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue());

			} else if (wardDiffuseGrp->Click(button,state,x,y))
				brdfGraph->setDiffuse(Color3(pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue()).y());
				cgGLSetParameter3f(diffuseParam, pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue());		
		}
	}
}

bool motionInterface(int x, int y) {
	if (brdfGraph->Motion(x,y)) {
		return true;
	}

	if (toneMapGrp->Motion(x,y)) {
		cgGLSetParameter2f(keyGammaParam, powf(10.f,exposureKey->GetValue()), 1.f/gamma->GetValue());
		return true;
	}

	if (type == LAFORTUNE) {
		if (lobeGrp->Motion(x,y)) {
			brdfGraph->setLobe(cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()), 
							   Color3(ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue()).y());
			
			genLafortuneSamples(samples, cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()));

			cgGLSetParameter4f(lobeParam, cxy->GetValue(), cxy->GetValue(), cz->GetValue(), powf(10.f,n->GetValue()));
			cgGLSetParameter3f(specularParam, ps_r->GetValue(), ps_g->GetValue(), ps_b->GetValue());
			
			return true;
		} else if (diffuseGrp->Motion(x,y)) {
			brdfGraph->setDiffuse(Color3(pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue()).y());
			cgGLSetParameter3f(diffuseParam, pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue());

			return true;
		}
	} else if (type == WARD) {
		if (wardLobeGrp->Motion(x,y)) {
			brdfGraph->setWardLobe(ax->GetValue(), ay->GetValue(), 
								   Color3(wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue()).y());
			genWardSamples(samples, ax->GetValue(), ay->GetValue());
			cgGLSetParameter2f(alphaParam, ax->GetValue(), ay->GetValue());
			cgGLSetParameter3f(specularParam, wps_r->GetValue(), wps_g->GetValue(), wps_b->GetValue());

			return true;
		} else if (wardDiffuseGrp->Motion(x,y)) {
			brdfGraph->setDiffuse(Color3(pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue()).y());
			cgGLSetParameter3f(diffuseParam, pd_r->GetValue(), pd_g->GetValue(), pd_b->GetValue());
			return true;
		}
	} else if (type == SV_LAFORTUNE) {
		if (lobePainter->Motion(x,y)) { return true; }
	}

	return false;
}