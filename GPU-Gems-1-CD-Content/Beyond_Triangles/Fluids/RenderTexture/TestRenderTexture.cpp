#include "RenderTexture.h"

#include <GL/glut.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GLuint iTextureProgram = 0;
GLuint iPassThroughProgram = 0;

RenderTexture *rt = NULL;

float maxS = 1;
float maxT = 1;

float angle = 0;

//------------------------------------------------------------------------------
// Function     	  : PrintGLerror
// Description	    : 
//------------------------------------------------------------------------------
void PrintGLerror( char *msg )
{
 GLenum errCode;
 const GLubyte *errStr;

 if ((errCode = glGetError()) != GL_NO_ERROR) 
 {
    errStr = gluErrorString(errCode);
    fprintf(stderr,"OpenGL ERROR: %s: %s\n", errStr, msg);
 }
}


//------------------------------------------------------------------------------
// Function     	  : Idle
// Description	    : 
//------------------------------------------------------------------------------
void Idle()
{
  angle += 1;
  glutPostRedisplay();
}

void Reshape(int w, int h)
{
  if (h == 0) h = 1;
  
  glViewport(0, 0, w, h);
  
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  
  gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 0.1, 1000.0);
}


//------------------------------------------------------------------------------
// Function     	  : display
// Description	    : 
//------------------------------------------------------------------------------
void display()
{
  rt->BeginCapture();
  { 
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    {
      glRotatef(angle, 1, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1,1,0);
      
      if (rt->IsFloatTexture() && rt->IsRectangleTexture())
      {
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, iPassThroughProgram);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
      }
      glutWireTorus(0.25, 1, 32, 64);
      
      if (rt->IsFloatTexture() && rt->IsRectangleTexture())
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }
    glPopMatrix();
    PrintGLerror("RT Update");
  }    
  rt->EndCapture();
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glColor3f(1, 1, 1);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glRotatef(angle / 10, 0, 1, 0);

  if (rt->IsFloatTexture())
  {    
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, iTextureProgram);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glActiveTexture(GL_TEXTURE0_ARB);
  }
  rt->Bind();
  rt->EnableTextureTarget();
    
  glBegin(GL_QUADS);
  glTexCoord2f(0,       0); glVertex3f(-1, -1, -0.5f);
  glTexCoord2f(maxS,    0); glVertex3f( 1, -1, -0.5f);
  glTexCoord2f(maxS, maxT); glVertex3f( 1,  1, -0.5f);
  glTexCoord2f(0,    maxT); glVertex3f(-1,  1, -0.5f);
  glEnd();
      
  if (rt->IsFloatTexture())
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
  
  rt->DisableTextureTarget();
  
  glPopMatrix();
  
  PrintGLerror("display");
  glutSwapBuffers();
}


//------------------------------------------------------------------------------
// Function     	  : main
// Description	    : 
//------------------------------------------------------------------------------
int main()
{
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowPosition(50, 50);
  glutInitWindowSize(512, 512);
  glutCreateWindow("TestRenderTexture");  

  int err = glewInit();
  if (GLEW_OK != err)
  {
    // problem: glewInit failed, something is seriously wrong
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    exit(-1);
  }  

  glutDisplayFunc(display);
  glutIdleFunc(Idle);
  glutReshapeFunc(Reshape);
  
  Reshape(512, 512);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);
  glDisable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST); 
  glClearColor(0.4, 0.6, 0.8, 1);
  
  int texWidth = 256, texHeight = 256;
  rt = new RenderTexture(texWidth, texHeight);
  // (Below, POTD = Power-of-two-dimenioned, NPOTD = Non-Power-of-two-dimenioned).

  // Default: RGBA8, Sharing enabled, no depth, stencil, mipmaps, or aniso filtering
  // (Can be POTD or NPOTD)
  //rt->Initialize();

  // Try these for other types of render textures.

  // RGBA8 with depth, stencil, mipmaps, and anisotropic filtering. (Must be POTD)  
  rt->Initialize(true, true, true, true, true, 8, 8, 8, 8);

  // The same as above, but updates are performed via copies rather than RTT.
  //rt->Initialize(true, true, true, true, true, 8, 8, 8, 8, RenderTexture::RT_COPY_TO_TEXTURE);
  
  // RGB8 with depth and stencil. (Can be POTD or NPOTD)
  //rt->Initialize(true, true, true, false, false, 8, 8, 8, 0);
  
  // R5G6B5 without depth and stencil
  //rt->Initialize(true, false, false, false, false, 5, 6, 5, 0);

  // 32-bit float texture.with depth and stencil. (must be POTD)
  //rt->Initialize(true, true, true, false, false, 32, 32, 32, 32);

  // 16-bit float texture.with depth and stencil. (must be POTD)
  //rt->Initialize(true, true, true, false, false, 16, 16, 16, 16);
  
  // single channel float texture.without depth or stencil. (must be POTD)
  //rt->Initialize(true, false, false, false, false, 32, 0, 0, 0);

  // setup the rendering context for the RenderTexture
  rt->BeginCapture();
  Reshape(texWidth, texHeight);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);
  glDisable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glDisable(GL_DEPTH_TEST); 
  glEnable(GL_DEPTH_TEST); 
  glClearColor(0.2, 0.2, 0.2, 1);
  rt->EndCapture();
  
  if (rt->IsFloatTexture())
  {
    // rectangle textures need texture coordinates in [0, resolution] rather than [0, 1]
    maxS = rt->GetWidth();
    maxT = rt->GetHeight();

    glGenProgramsARB(1, &iTextureProgram);
    glGenProgramsARB(1, &iPassThroughProgram);

    const char* textureProgram = rt->IsRectangleTexture() ?
                                 "!!ARBfp1.0\n"
                                 "TEX result.color, fragment.texcoord[0], texture[0], RECT;\n"
                                 "END\n"
                                 :
                                 "!!ARBfp1.0\n"
                                 "TEX result.color, fragment.texcoord[0], texture[0], 2D;\n"
                                 "END\n";
    const char* passThroughProgram = "!!ARBfp1.0\n"
                                     "MOV result.color, fragment.color.primary;\n"
                                     "END\n";

    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, iTextureProgram);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                       strlen(textureProgram), textureProgram);

    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, iPassThroughProgram);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                       strlen(passThroughProgram), passThroughProgram);
  }
  
  glutMainLoop();

  return 0;
}
