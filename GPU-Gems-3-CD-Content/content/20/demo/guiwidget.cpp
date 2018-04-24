//
// guiwidget.cpp
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

#include <GL/glut.h>
#include "guiwidget.h"
#include "gstream.h"

#include <iostream>
using namespace std;

void SliderBar::Compile() {
	// non varying components 

	if (sliderName) {
		glColor3ub(255,255,255);
		gout << sliderName;
		gout.draw(d.x,d.y+9);
	}

	int xoffset = d.x+offset;
	int xpos = (int) (((float) (d.w-offset))*(value-minValue)/(maxValue-minValue)) + xoffset;

	glColor4ub(255,255,255,128);
	glBegin(GL_LINE_LOOP);
		glVertex2i(xoffset,	 d.y);
		glVertex2i(d.x+d.w, d.y);
		glVertex2i(d.x+d.w, d.y+d.h);
		glVertex2i(xoffset,	 d.y+d.h);
	glEnd();
}

void SliderBar::Display() {
	// possibly varying components

	int xoffset = d.x+offset;
	int xpos = (int) (((float) (d.w-offset))*(value-minValue)/(maxValue-minValue)) + xoffset;

	glBegin(GL_QUADS);
		glColor4ub(128,128,128,100); glVertex2i(xoffset, d.y);
		glColor4ub(128,128,128,200); glVertex2i(xpos,	 d.y);
		glColor4ub(128,128,128,200); glVertex2i(xpos,	 d.y+d.h);
		glColor4ub(128,128,128,100); glVertex2i(xoffset, d.y+d.h);
	glEnd();
}

bool SliderBar::Click(int button, int state, int x, int y) {
	if (Contains(x,y)) {
		if (state == GLUT_DOWN) {
			if (button == GLUT_LEFT_BUTTON) {
				//active = true;
				value = min(((float) max(0,x-d.x-offset))/((float) (d.w-offset)), 1.f)*(maxValue-minValue) + minValue;
			}
		} else {
			//active = false;
		}
		return true;
	}
	return false;
}

bool SliderBar::Motion(int x, int y) {
	if (Contains(x,y)) {
		value = min(max(0.f,((float) (x-d.x-offset))/((float) (d.w-offset))), 1.f)*(maxValue-minValue) + minValue;
		return true;
	}
	return false;
}

void WidgetGroup::Compile() {
	glId = glGenLists(1);
	glNewList(glId, GL_COMPILE);

		glBegin(GL_QUADS);
			glColor4ub(37,101,146,128); glVertex2i(0,0);
										glVertex2i(d.w,0);
			glColor4ub(37,101,146,210); glVertex2i(d.w, d.h);
										glVertex2i(0, d.h);
		glEnd();

		glColor4ub(95,166,216,200);
		glBegin(GL_LINE_LOOP);
			glVertex2i(0,0);
			glVertex2i(d.w,0);
			glVertex2i(d.w, d.h);
			glVertex2i(0, d.h);
		glEnd();

		glColor3ub(255,255,255);
		gout << grpName;
		gout.draw(2, 10);

		for (vector<Container*>::iterator widgetIter = widgets.begin();
			widgetIter != widgets.end(); widgetIter++)
		{
			(*widgetIter)->Compile();
		}

	glEndList();	
}

void WidgetGroup::Display() {
	glPushMatrix();
		glTranslatef(d.x, d.y, 0.f);
		glCallList(glId);

		for (vector<Container*>::iterator widgetIter = widgets.begin();
			widgetIter != widgets.end(); widgetIter++)
		{
			(*widgetIter)->Display();
		}
	glPopMatrix();
}

bool WidgetGroup::Click(int button, int state, int x, int y) {
	if (Contains(x,y)) {
		int nx = x-d.x;
		int ny = y-d.y;

		for (vector<Container*>::iterator widgetIter = widgets.begin();
			widgetIter != widgets.end(); widgetIter++)
		{
			if ((*widgetIter)->Contains(nx,ny)) (*widgetIter)->Click(button,state,nx,ny);
		}
		return true;
	}

	return false;
}

bool WidgetGroup::Motion(int x, int y) {
	int nx = x-d.x;
	int ny = y-d.y;

	if (Contains(x,y)) {
		for (vector<Container*>::iterator widgetIter = widgets.begin();
			widgetIter != widgets.end(); widgetIter++)
		{
			if ((*widgetIter)->Motion(nx,ny)) return true;
		}
		return true;
	}
	return false;
}

void WidgetGroup::WindowReshape(int w, int h) {
	if (aType == RIGHT) {
		// find out the old relative distance
		d.x = w - d.w - offsetX;
		d.y = offsetY;
		//d.y = h - d.h - offsetY;
	} else if (aType == CENTER) {
		d.x = (w - d.w)/2;
	} else if (aType == BOTTOM) {
		d.y = h-d.h-offsetY;
	}
}

void Button::Compile() {
	glColor4ub(128,128,128,128);
	glBegin(GL_QUADS);
		glVertex2i(d.x, d.y);
		glVertex2i(d.x+d.w, d.y);
		glVertex2i(d.x+d.w, d.y+d.h);
		glVertex2i(d.x, d.y+d.h);
	glEnd();

	glColor3ub(255,255,255);
	gout << buttonName;
	gout.draw(d.x+5,d.y+d.h/2+5);
}

void Button::Display() {
	if (state != UP) {
		switch (state) {
			case DISABLED:
				glColor4ub(5,5,5,100);
				break;
			case UP:
				glColor4ub(128,128,128,128);
				break;
			case DOWN:
				glColor4ub(15,15,15,200);
				break;
			case HOVER:
				glColor4ub(128,15,15,128);
				break;
		}

		glBegin(GL_QUADS);
			glVertex2i(d.x, d.y);
			glVertex2i(d.x+d.w, d.y);
			glVertex2i(d.x+d.w, d.y+d.h);
			glVertex2i(d.x, d.y+d.h);
		glEnd();
	}
}

bool Button::Click(int button, int wState, int x, int y) {
	if (state != DISABLED) {
		if (wState == GLUT_DOWN && Contains(x,y)) {
			if (button == GLUT_LEFT_BUTTON) {
				state = DOWN;
				this->OnClick();
			}
			return true;
		} else {
			state = UP;
		}
	}

	return false;

}

void Button::PassiveMotion(int x, int y) {
	if (state != DISABLED) {
		state = (Contains(x,y))?HOVER:UP;
	}
}

void ScrollSelector::Compile() {
	// widget display name
	glColor3ub(255,255,255);
	gout << name;
	gout.draw(d.x,d.y+9);

	glBegin(GL_TRIANGLES);
		glVertex2i(d.x+d.w-18, d.y+d.h);
		glVertex2i(d.x+d.w-15, d.y);
		glVertex2i(d.x+d.w-12, d.y+d.h);

		glVertex2i(d.x+d.w- 8, d.y);
		glVertex2i(d.x+d.w- 5, d.y+d.h);
		glVertex2i(d.x+d.w- 2, d.y);
	glEnd();
}

void ScrollSelector::Display() {
	glColor3ub(255,255,255);
	gout << options[currOption];
	gout.draw(d.x+50,d.y+9);
}


bool ScrollSelector::Click(int button, int state, int x, int y) {
	if (Contains(x,y) && (state==GLUT_DOWN)) {
		if ((x >= (d.x+d.w-18)) && (x <= (d.x+d.w-12)))
			currOption = (currOption+1)%numOptions;
		else if ((x >= (d.x+d.w-8)) && (x <= (d.x+d.w-2))) {
			currOption--;
			currOption = (currOption < 0)?numOptions-1:currOption;
		}

		return true;
	}
	return false;
}
