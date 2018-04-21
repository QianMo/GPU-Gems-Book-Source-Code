/**
  @file getOpenGLState.h

  @maintainer Morgan McGuire, matrix@graphics3d.com
  @cite       Created by Morgan McGuire & Seth Block
  @created 2001-08-05
  @edited  2003-04-07
*/

#ifndef GETOPENGLSTATE_H
#define GETOPENGLSTATE_H

#include "graphics3D.h"
#include "GLG3D/glheaders.h"
#include "GLG3D/glcalls.h"

namespace G3D {

/** 
 Returns all OpenGL state as a formatted string of C++
 code that will reproduce that state.   Leaves all OpenGL
 state in exactly the same way it found it.   Use this for 
 debugging when OpenGL doesn't seem to be in the same state 
 that you think it is in. 

  A common idiom is:
  {std::string s = getOpenGLState(false); debugPrintf("%s", s.c_str();}

 @param showDisabled if false, state that is not affecting
  rendering is not shown (e.g. if lighting is off,
  lighting information is not shown).
 */
std::string getOpenGLState(bool showDisabled=true);


/**
 Returns the number of bytes occupied by a value in
 an OpenGL format (e.g. GL_FLOAT).  Returns 0 for unknown
 formats.
 */
size_t sizeOfGLFormat(GLenum format);

/**
 Pretty printer for GLenums.  Useful for debugging OpenGL
 code.
 */
const char* GLenumToString(GLenum i);

}

#endif
