#include "RenderTexture.h"

#include <GL/glut.h>
#include <assert.h>
#include <stdio.h>

GLuint iTextureProgram = 0;
GLuint iPassThroughProgram = 0;

RenderTexture *rt = NULL;

float angle = 0;
bool bShowDepthTexture = true;

float maxS = 1;
float maxT = 1;

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

void Keyboard(unsigned char key, int x, int y)
{
  switch(key)
  {
  case ' ':
    bShowDepthTexture = !bShowDepthTexture;
    return;
  default:
    return;
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
  
  gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 0.5, 10.0);
}


//------------------------------------------------------------------------------
// Function     	  : Display
// Description	    : 
//------------------------------------------------------------------------------
void Display()
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

      glutSolidTorus(0.25, 1, 32, 64);
      
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

  if(bShowDepthTexture)
    rt->BindDepth();
  else
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
void main()
{
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowPosition(50, 50);
  glutInitWindowSize(512, 512);
  glutCreateWindow("TestRenderDepthTexture");  

  int err = glewInit();
  if (GLEW_OK != err)
  {
    // problem: glewInit failed, something is seriously wrong
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    exit(-1);
  }  

  glutDisplayFunc(Display);
  glutIdleFunc(Idle);
  glutReshapeFunc(Reshape);
  glutKeyboardFunc(Keyboard);
  
  Reshape(512, 512);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);
  glDisable(GL_LIGHTING);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST); 
  glClearColor(0.4, 0.6, 0.8, 1);
  
  int texWidth = 256, texHeight = 256;
  // A square, mipmapped, anisotropically filtered 8-bit RGBA texture with
  // depth and stencil.
  // Note that RT_COPY_TO_TEXTURE is required for depth textures on ATI hardware
  rt = new RenderTexture(texWidth, texHeight, true, true);
  rt->Initialize(true, true, true, true, true, 8, 8, 8, 8);
  // for using the depth texture for shadow mapping we would still have to bind it and set
  // the correct texture parameters using the SGI_shadow or ARB_shadow extension

  // Test with Copy
  //rt = new RenderTexture(texWidth, texHeight, true, true);
  //rt->Initialize(true, true, false, true, true, 8, 8, 8, 8, RenderTexture::RT_COPY_TO_TEXTURE);

  // Try these for other types of render textures.

  // A rectangle 8-bit RGB texture with depth and stencil.
  //texWidth = 200; texHeight = 231;
  //rt = new RenderTexture(texWidth, texHeight, true, true);
  //rt->Initialize(true, true, true, false, false, 8, 8, 8, 0);//, RenderTexture::RT_COPY_TO_TEXTURE);
  
  // A square float texture.with depth and stencil.
  //rt = new RenderTexture(texWidth, texHeight, true, true);
  //rt->Initialize(true, true, true, false, false, 32, 32, 32, 32);//, RenderTexture::RT_COPY_TO_TEXTURE);
  
  // A square single channel float texture.without stencil.
  //rt = new RenderTexture(texWidth, texHeight, true, true);
  //rt->Initialize(true, false, false, false, false, 32, 0, 0, 0);//, RenderTexture::RT_COPY_TO_TEXTURE);

  // setup the rendering context for the RenderTexture
  rt->BeginCapture();
  Reshape(texWidth, texHeight);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST); 
  glClearColor(0.2, 0.2, 0.2, 1);
  rt->EndCapture();
  
  if (rt->IsRectangleTexture())
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
}
