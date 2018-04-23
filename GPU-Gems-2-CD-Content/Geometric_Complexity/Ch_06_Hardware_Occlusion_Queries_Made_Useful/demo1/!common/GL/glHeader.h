//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.

#ifndef GlInterfaceH
#define GlInterfaceH

//includes an openGL extension header file
//http://glew.sourceforge.net/
#include <GL/glew.h>
//includes the glut header file
#include <GL/glut.h>

//includes  windows specific stuff
#ifdef _WIN32
	#include <GL/wglew.h>
#else
//include linux specific stuff
	#include <GL/glxew.h>
#endif	

#endif
