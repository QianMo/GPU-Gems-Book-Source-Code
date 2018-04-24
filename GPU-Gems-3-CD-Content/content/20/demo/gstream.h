//
// gstream.h
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

#ifndef _GSTREAM_H
#define _GSTREAM_H

#include <sstream>
#include <string>

/// Class for generating GLUT text strings via the gout
/// variable, which acts exactly like the cout variable, but
/// with a draw command
class gstream {
	public:
		gstream() {}
		~gstream() {}
		
		template <typename T>
			inline gstream& operator<<(const T &val) { oss << val; return (*this); }
		
		void draw(int x, int y, bool newLineDown = true);
		inline void clear() { oss.str(""); }		
		
	private:
		std::ostringstream oss;
};

const char gendl = '\n';
extern gstream gout;

#endif
