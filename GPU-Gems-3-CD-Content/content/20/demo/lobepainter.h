//
// lobepainter.h
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

#ifndef _LOBE_DISPLAY_H
#define _LOBE_DISPLAY_H

#include <GL/glew.h>
#include <Cg/cg.h>

#include "guiwidget.h"

class BRDFGraph;

/// Painting class widget for designing the SBRDF
class LobePainter : public Container {
	public:
		LobePainter(BRDFGraph *brdfGraph, int offsetX, int y, int displayWidth, int resWidth);
		virtual ~LobePainter();

		virtual void Display();

		virtual bool Click(int button, int state, int x, int y);
		virtual bool Motion(int x, int y);

		// Since ping-pong texturing is used, this class keeps track of the current
		// framebuffer holding the most up-to-date texture
		unsigned int GetTexId() { return fbTex[currTex]; }
		void updateGraph();

		virtual void WindowReshape(int w, int h) { 
			d.x=w-d.w-5; fy = h-(d.y+d.w);
			controls->WindowReshape(w,h);
		}

	private:
		/// initialize necessary OpenGL components for painting
		void InitCg();
		void InitQuad();
		void InitFramebuffers();
		
		void Paint(float x, float y);
		
		// CG & OpenGL data parameters
		CGprogram paintProg, displayProg, uvProg;
		CGparameter lobeParam, posParam;

		unsigned int fb[2], fbTex[2], quadId;
		bool currTex;

		// Updates BRDF grapher when widget is active
		BRDFGraph *brdfGraph;
		
		WidgetGroup *controls;
		SliderBar cxy, cz, n, weight, diffuse, size;

		int offsetX, fy, resWidth;

};

#endif
