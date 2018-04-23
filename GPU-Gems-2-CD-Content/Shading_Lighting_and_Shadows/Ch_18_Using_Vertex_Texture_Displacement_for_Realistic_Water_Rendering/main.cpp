
//----------------------------------------------------------------------------
// USER-PROVIDED INCLUDES
//----------------------------------------------------------------------------
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CRenderer.hpp"

CRenderer g_renderer;
bool bLeftButtonDown    = false;
int  iOldMouseXPosition = 0;
int  iOldMouseYPosition = 0;
int  iMouseXPosition    = 0;
int  iMouseYPosition    = 0;
int  wireframe = 0;
float rRotationX        = 0;
float rRotationY        = 0;
float CamX  = +200;
float CamY  = +650;
float CamZ  = 1.5;
float Weather = 0;

#define GL_CLAMP_TO_EDGE                  0x812F
bool bUseDisplayFragmentProgram = true;
int iFrameCount = 0;

void Keyboard(/*unsigned char*/int key, int x, int y)
{

  switch (key)
  {
  //case 'q':
  case 27:
    //rd.Shutdown();
    exit(0);
    break;
  case '>': rRotationX += 2.0; break;

  case '<': rRotationX -= 2.0; break;
  //case 'w': case 'W': Weather = Weather+0.1; if (Weather > 1) Weather = 0.0f;

  case GLUT_KEY_PAGE_UP: rRotationY += 2.0; break;

  case GLUT_KEY_PAGE_DOWN: rRotationY -= 2.0; break;

  case GLUT_KEY_HOME: CamZ       += 10.0; break;
  case GLUT_KEY_END : CamZ       -= 10.0; break;

  case GLUT_KEY_RIGHT:CamX       += 10.0; break;
  case GLUT_KEY_LEFT: CamX       -= 10.0; break;
  
  
  case GLUT_KEY_UP:   CamY       -= 10.0; break;
  case GLUT_KEY_DOWN: CamY       += 10.0; break;


  

  default:
    rRotationX = rRotationY = 0;
    break;
  }
}

void KeyboardA(unsigned char key, int x, int y)
{

  switch (key)
  {
  //case 'q':
  case 'w':	case 'W':
		wireframe = !wireframe;
		glPolygonMode( GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL );
	break;

  case 27:
    //rd.Shutdown();
    exit(0);
    break;
  case ',': rRotationX += 2.0; break;
  case '.': rRotationX -= 2.0; break;

  default:
    rRotationX = rRotationY = 0;
    break;
  }
}

//----------------------------------------------------------------------------
// USER-PROVIDED METHOD OVERRIDES
//----------------------------------------------------------------------------
void Display()
{
  //rd.Update();
 
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  //gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);
  
  

  


  
  //glRotatef(-90.0, 1.0, 0.0, 0);    
  
  glRotatef(+90.0, 1.0, 0.0, 0);    

  glScalef(1,1,-1);

  glRotatef(rRotationY, 1, 0, 0);
  glRotatef(rRotationX, 0, 0, 1);

  glTranslatef(-CamX, -CamY, -CamZ);
  

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glColor3f(1.0f, 1.0f, 1.0f);

  // Render reflection to texture
  struct {
     GLint x, y, width, height;
  } GLViewPort;

  glGetIntegerv(GL_VIEWPORT, (GLint*)&GLViewPort.x);


  static int InitReflectionTex = 0;
  int ReflTexSize = 512;

  glViewport(0, 0, ReflTexSize, ReflTexSize);

  

  glPushMatrix(); 
    glScalef(1,1, -1);
    g_renderer.Render(1);  
  glPopMatrix();
  
    
  glBindTexture(GL_TEXTURE_2D, g_renderer.GetReflID());  

  if (!InitReflectionTex) {
      InitReflectionTex = 1;
    
      glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 0, 0, ReflTexSize, ReflTexSize, 0);

      //glTexParameteri(GL_TEXTURE_2D,GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    }
   
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, ReflTexSize, ReflTexSize);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(GLViewPort.x, GLViewPort.y, GLViewPort.width, GLViewPort.height);

    /*DWORD *TestBuf = new DWORD[ReflTexSize*ReflTexSize];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, TestBuf);


    FILE *F = fopen("RawFN.raw","wb");
    fwrite(TestBuf, ReflTexSize*ReflTexSize, 4, F);
    fclose(F);
    delete[] TestBuf;*/
    
    glPushMatrix(); 
      g_renderer.Render(0);

    glPopMatrix();

  glutSwapBuffers();

}


void Idle()
{
  glutPostRedisplay();
}


void Reshape(int w, int h)
{
    if (h == 0) h = 1;
    
    glViewport(0, 0, w, h);

    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 0.5, 20000.0);
    //gluPerspective(72,1.0,0.1,5000.0);
    //glOrtho(-5, 5, -5, 5, 0.01, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


//----------------------------------------------------------------------------
// MAIN ROUTINE. INIT USER OBJECTS AND ACTIVATE VIEWER.
//----------------------------------------------------------------------------
void main()
{ 
  printf("Keys:\n");
  printf("'Esc': Quit\n");
  
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
  //glutInitWindowPosition(50, 50);
  glutInitWindowSize(800, 600);
//glutInitWindowSize(400, 300);
  glutCreateWindow("A Shader");

  glutIdleFunc(Idle);
  glutDisplayFunc(Display);
  glutKeyboardFunc(KeyboardA);
glutSpecialFunc(Keyboard);
  glutReshapeFunc(Reshape);

  glDisable(GL_LIGHTING);
  
  
  g_renderer.Initialize();

  //Reshape(800, 600);
Reshape(800, 600);

  glutMainLoop();
}
