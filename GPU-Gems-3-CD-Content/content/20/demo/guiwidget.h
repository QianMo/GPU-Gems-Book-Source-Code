//
// guiwidget.h
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

#ifndef _GUI_WIDGET_H
#define _GUI_WIDGET_H

#include <vector>

typedef struct Dimension {
	Dimension() : x(0), y(0), w(0), h(0) {}
	Dimension(int _x, int _y, int _w, int _h) : x(_x), y(_y), w(_w), h(_h) {}
	int x, y, w, h;
} Dimension;

/// Base-class for all other widgets
class Widget {
	public:
		Widget() {}
		virtual ~Widget() {}

		/// Compile the object into the OpenGL driver
		virtual void Compile() {}

		/// display the object onto the screen
		virtual void Display() {}

		// messages functions from window
		virtual bool Click(int button, int state, int x, int y)		{ return false; }
		virtual bool Motion(int x, int y)							{ return false; }
		virtual void PassiveMotion(int x, int y)					{}
		virtual void Keyboard(unsigned char key, int x, int y)		{}
		virtual void WindowReshape(int w, int h)					{}

		virtual bool Contains(int x, int y)							{ return false; }
};

/// Base-class for widgets containing space on the screen
class Container : public Widget {
	public:
		Container() {}
		Container(int x, int y, int w, int h) : d(x,y,w,h) {}
		virtual ~Container() {}

		virtual void SetPosition(int x, int y)				{ d.x = x; d.y = y; }
		virtual void SetDimensions(int w, int h)			{ d.w = w; d.h = h; }
		virtual Dimension GetDimensions()				{ return d; }
		virtual bool Contains(int x, int y)					{ return ((x >= d.x) && (x <= (d.x+d.w)) &&
																	  (y >= d.y) && (y <= (d.y+d.h))); }
	protected:
		Dimension d;
};

/// Base-class for slider-bars with informative text
class SliderBar : public Container {
	public:
		SliderBar(float _value, float _minValue, float _maxValue, int x, int y, int w, int h) :
				  Container(x,y,w,h), active(false), sliderName(NULL), offset(0), value(_value), minValue(_minValue), maxValue(_maxValue) {}
		SliderBar(const char *_sliderName, int _offset, float _value, float _minValue, float _maxValue, int x, int y, int w, int h) :
				  Container(x,y,w,h), active(false), sliderName(_sliderName), offset(_offset), value(_value), minValue(_minValue), maxValue(_maxValue) {}
	    
		virtual ~SliderBar() { }

		inline float GetValue() { return value; }
		inline void  SetValue(float _value) { value=_value; }

		virtual void Compile();
		virtual void Display();
		virtual bool Click(int button, int state, int x, int y);
		virtual bool Motion(int x, int y);

		virtual bool Contains(int x, int y)					{ return ((x >= (d.x+offset)) && (x <= (d.x+d.w)) &&
																	  (y >= d.y) && (y <= (d.y+d.h))); }
	protected:

		bool active;
		const char *sliderName;
		int offset;
		float value, minValue, maxValue;
};

/// Base-class for buttons
class Button : public Container {
	public:
		Button(const char *_buttonName, int x, int y, int w, int h) : Container(x,y,w,h), buttonName(_buttonName), state(UP) {}
		
		virtual void Compile();
		virtual void Display();
		virtual bool Click(int button, int state, int x, int y);
		virtual void PassiveMotion(int x, int y);

		virtual void OnClick() = 0;

		enum ButtonState { DOWN, UP, HOVER, DISABLED};
	protected:
		const char *buttonName;
		ButtonState state;

};

/// Base-class for the pseudo-scroll bars
class ScrollSelector : public Container {
	public:
		ScrollSelector(const char *_name, const char **_options, int _numoptions, int x, int y, int w) : 
		  Container(x,y,w,8), name(_name), options(_options), numOptions(_numoptions), currOption(0) {}

		virtual void Compile();
		virtual void Display();
		virtual bool Click(int button, int state, int x, int y);
		
		inline int	GetCurrOption() { return currOption; }
		inline void SetCurrOption(int option) { currOption = option; }

	private:
		const char *name;
		const char **options;
		int numOptions, currOption;

};

/// Class for putting multiple widgets together into a larger container
class WidgetGroup : public Container {
	public:
		enum AlignType {LEFT, RIGHT, CENTER, BOTTOM};

		WidgetGroup(const char *_grpName, int _offsetX, int _offsetY, int w, int h, AlignType _type) : 
					Container(_offsetX,_offsetY,w,h), grpName(_grpName), aType(_type), offsetX(_offsetX), offsetY(_offsetY) {}
		virtual ~WidgetGroup() {}

		virtual void AddWidget(Container *c) { widgets.push_back(c); }

		virtual void Compile();
		virtual void Display();
		virtual bool Click(int button, int state, int x, int y);
		virtual bool Motion(int x, int y);

		virtual void WindowReshape(int w, int h);
	protected:
		const char *grpName;
		AlignType aType;
		int offsetX, offsetY;
		std::vector<Container*> widgets;
		unsigned int glId;
};

#endif
