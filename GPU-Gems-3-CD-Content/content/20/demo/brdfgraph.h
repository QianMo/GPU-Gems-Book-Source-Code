//
// brdfgraph.h
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


#ifndef _BRDF_DISPLAY_H
#define _BRDF_DISPLAY_H

#include <GL/glut.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "vectors.h"

/// Class for displaying either a Lafortune or Ward Lobe in the lower
/// right corner of a window
class BRDFGraph {
	public:
		BRDFGraph();
		~BRDFGraph();

		void Display();
		void WindowReshape(int ww, int wh) { x = ww-w; y = wh-h; fy=0; }
	
		bool Click(int button, int state, int x, int y);
		bool Motion(int x, int y);

		/// Set Lafortune or Ward lobe for displaying
		inline void setType(int _type) { type=_type; }

		void setLobe(float cxy, float cz, float n, float weight);
		void setWardLobe(float ax, float ay, float weight);
		void setDiffuse(float weight);

	private:
		void initHemisphere();
		void initCg();
		
		void setWo();

		inline bool Contains(int ax, int ay) { 
			int relx=ax-x,rely=ay-y; return ((relx>0)&&(relx<w)&&(rely>0)&&(rely<h));
		}

		CGprogram vertProg[2];
		CGparameter alphaParam;					///< Ward parameters
		CGparameter lobeParam;					///< Lafortune parameters
		CGparameter woParam, diffuseParam;		///< Shared parameters
		GLuint hemiId;							///< Hemisphere OpenGL Id

		int type;

		// UI variables
		enum State { ROTATE, ROTWO, ZOOM };
		State state;

		// current outgoing direction being displayed
		Vector3D wo;
		bool active;

		// window/graph/mouse positions
		int x,y,fy,w,h;
		int ox, oy;
		float dist;
		int rx,ry,worot;
};

#endif
