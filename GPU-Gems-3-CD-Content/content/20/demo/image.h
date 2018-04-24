//
// image.h
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

#ifndef _IMAGE_H
#define _IMAGE_H
		
/// Base-class for all images
template<class T> class Image {
	public:
		Image() : data(0), width(0), height(0), components(-1) {}
		Image(T *_data, int _width, int _height, char _components) : 
			  data(_data), width(_width), height(_height), components(_components)
		{}
		virtual ~Image() {}
		
		inline T* GetData()		{ return data; }
		inline int GetWidth()	{ return width; }
		inline int GetHeight()	{ return height; }
		inline int GetComponents() { return components; }

	protected:
		T *data;
		int width, height;
		char components;
};

#endif
