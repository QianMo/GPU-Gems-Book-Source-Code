/* 
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <cstdarg>
#include <cstdio>
#include "gprintf.h"

int gprintf(int x, int y, bool shadow, char* fmt, ...)
{
  if (shadow) {
    glPushAttrib(GL_CURRENT_BIT);
    glColor4f(0.0, 0.0, 0.0, 1.0);
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        va_list va;
        va_start(va, fmt);
        vgprintf(x + dx, y + dy, fmt, va);
        va_end(va);
      }
    }
    glPopAttrib();
  }
  
  va_list va;
  va_start(va, fmt);
  int r = vgprintf(x, y, fmt, va);
  va_end(va);
  return r;
}

int vgprintf(int x, int y, char* fmt, va_list va)
{
  char temp[1024];
  vsprintf(temp, fmt, va);

  // setup the matrices for a direct
  // screen coordinate transform when
  // using glRasterPos
  int vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, vp[2], 0, vp[3], -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // just in case, turn lighting and
  // texturing off and disable depth testing
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glDisable(GL_VERTEX_PROGRAM_ARB);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);

  // render the character through glut
  char* p = temp;
  glRasterPos2f(x, y);
  while(*p) {
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, (*p));
    p++;
  }

  // reset OpenGL to what is was
  // before we started
  glPopAttrib();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  return p-temp;
}
