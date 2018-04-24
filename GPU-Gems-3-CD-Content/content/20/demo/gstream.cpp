//
// gstream.cpp
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

#include "gstream.h"

#include <GL/glut.h>
#include <iostream>
using namespace std;

gstream gout;

void gstream::draw(int x, int y, bool newLineDown) {
	string str = oss.str();
	glRasterPos2i(x,y);

	for (unsigned int i=0; i < str.length(); i++) {
		if (str[i] == '\n') {
			y += ((newLineDown)?1:-1) * 12;
			glRasterPos2i(x, y);
		}
		else {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, str[i]);
		}
	}
	clear();	
}
