/* ----------------------------------------------------------

Octree Textures on the GPU - source code - GPU Gems 2 release
                                                   2004-11-21

Updates on http://www.aracknea.net/octreetex
--
(c) 2004 Sylvain Lefebvre - all rights reserved
--
The source code is provided 'as it is', without any warranties. 
Use at your own risk. The use of any part of the source code in a
commercial or non commercial product without explicit authorisation
from the author is forbidden. Use for research and educational
purposes is allowed and encouraged, provided that a short notice
acknowledges the author's work.
---------------------------------------------------------- */
/* -------------------------------------------------------- */
/*
/*
        Liquid flowing along a surface

*/


#ifdef WIN32
#include <windows.h>
#endif

#include "config.h"
#include "CTexture.h"
#include "common.h"

/* -------------------------------------------------------- */

#define FPPROG      "simul/fp_simul_tree.cg"

/* -------------------------------------------------------- */

#include <iostream>
#include <fstream>
#include <GL/gl.h>
#include <stdio.h>
#include <GL/glut.h>
#include <time.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>

#include <glux.h>
#include <gltex.h>

#include "cg_load.h"

#include <glfont.h>

#include "CProfiler.h"
#include "CVertex.h"
#include "CBox.h"
#include "CSimulTree.h"
#include "CGL3DSMesh.h"
#include "CQuaternion.h"
#include "pbuffer.h"
#include "help.h"

/* -------------------------------------------------------- */

void draw();
void report();
void transform();
bool getLeafAt(double x,double y,int& _i,int& _j);

/* -------------------------------------------------------- */

#include <list>
using namespace std;

/* -------------------------------------------------------- */

int             g_iMainWindow;       // Glut Window Id
int             g_iFrame      =0;    // Frame counter
int             g_iW          =SCREEN_SIZE;
int             g_iH          =SCREEN_SIZE;
int             g_iOldX       =-1;
int             g_iOldY       =-1;
int             g_iBtn        =0;
double          g_dRotX       =30.0;
double          g_dRotZ       =0.0;
double          g_dDist       =2.0;
GLfont         *g_Font        =NULL;
bool            g_bRun        =true;
bool            g_bSpil       =false;

int             g_iTimerDraw   =-1;
int             g_iTimerCommit =-1;
double          g_dFPS;
int             g_iNbFrames  =0;
int             g_iLastTime  =0;

bool            g_bDrawTree       =false;
bool            g_bDrawEmpty      =true;
bool            g_bDrawObject     =true;
bool            g_bDrawSprites    =false;
bool            g_bDrawWireframe  =false;
bool            g_bDrawTexSprite  =true;
bool            g_bFullScreenDebug=false;
bool            g_bRain           =false;
bool            g_bHUD            =false;
bool            g_bDebug          =false;
int             g_LastSelected[2] ={-1,-1};
double          g_dDropSize       =2.0;

PBuffer         g_Screen("rgb depth");
GLuint          g_uiScreenTex=0;
GLuint          g_uiTex=0;

CGL3DSMesh     *g_Object      =NULL;

CSimulTree     *g_Tree       =NULL;

#ifdef WIN32
double drand48()
{
  return ((rand() % RAND_MAX)/(double)(RAND_MAX-1.0));
}
#endif

/* -------------------------------------------------------- */

CGprogram   g_cgVPRender;
CGparameter g_cgVPViewProj;
CGparameter g_cgVPITView;
CGparameter g_cgVPView;

CGprogram   g_cgFPRender;
CGparameter g_cgFPScreenBuffer;
CGparameter g_cgFPEnvmap;
CGparameter g_cgFPTex;

CCubeMap   *g_Envmap;

/* -------------------------------------------------------- */

bool             g_bRecord=false;

#ifdef WIN32
#include "MovieMaker.h"
MovieMaker       g_VCR;

static void* getCurrentHWND()   
{     // get current DC from wgl     
  return WindowFromDC(wglGetCurrentDC());   
}
#endif

/* -------------------------------------------------------- */

void draw();
void draw_texspace();
void transform();
bool getPosNormalAt(double x,double y,CVertex&,CVertex&);

/* -------------------------------------------------------- */

void quad()
{
  glBegin(GL_QUADS);
  glTexCoord2d(1,1);
  glVertex2i(1,1);
  glTexCoord2d(0,1);
  glVertex2i(0,1);
  glTexCoord2d(0,0);
  glVertex2i(0,0);
  glTexCoord2d(1,0);
  glVertex2i(1,0);
  glEnd();
}

/* -------------------------------------------------------- */

void mainKeyboard(unsigned char key, int x, int y) 
{
  if (key == 'q')
  {
#ifdef WIN32
    if (g_bRecord)
      g_VCR.EndCapture();
#endif
    exit (0);
  }
  else if (key == 'c')
  {
    g_Tree->clear();
  }
  else if (key == 'h')
  {
    g_bHUD=!g_bHUD;
  }
  else if (key == 'H')
  {
    g_bDebug=!g_bDebug;
  }
  else if (key == 'g')
  {
    g_bRain=!g_bRain;
  }
  else if (key == 'o')
  {
    g_dDist+=0.1;
  }
  else if (key == 'r')
  {
    g_bRun=!g_bRun;
  }
  else if (key == 'p')
  {
    g_dDist-=0.1;
  }  
  else if (key == 't')
  {
    g_bDrawTree=!g_bDrawTree;
  }  
  else if (key == 'e')
  {
    g_bDrawEmpty=!g_bDrawEmpty;
  }  
  else if (key == ' ')
  {
    g_bDrawObject=!g_bDrawObject;
  }  
  else if (key == 'f')
  {
    g_bFullScreenDebug=!g_bFullScreenDebug;
  }  
  else if (key == 'w')
  {
    g_bDrawWireframe=!g_bDrawWireframe;
  }  
  else if (key == 'w')
  {
    g_bDrawWireframe=!g_bDrawWireframe;
  }  
  else if (key == '+')
  {
    g_dDropSize*=2.0;
  }  
  else if (key == '-')
  {
    g_dDropSize/=2.0;
  }  
  else if (key == '*')
  {
    static int n=0;
    unsigned char *rgb_data=new unsigned char[g_iW*g_iH*3];
    glFlush();
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,g_iW,g_iH,GL_RGB,GL_UNSIGNED_BYTE,rgb_data);
    CTexture *tex=new CTexture("",g_iW,g_iH,false,rgb_data);
    static char name[128];
    sprintf(name,"shot%04d.tga",n++);
    CTexture::saveTexture(tex,name);
    delete [](rgb_data);
  }
#ifdef WIN32
  else if (key == 'R')
  {
    if (!g_bRecord)
    {
      g_bRecord=true;
      g_VCR.StartCapture((HWND)getCurrentHWND(),g_iW,g_iH,"movie.avi");
    }
    else
    {
      g_bRecord=false;      
      g_VCR.EndCapture();
    }
  }
#endif
  else if (key == '/')
  {
    PROFILER.normalize();
  }
}

/* -------------------------------------------------------- */

void mainMouse(int btn, int state, int x, int y) 
{
  g_iBtn=btn;

  if (state == GLUT_DOWN)
  {
    g_iOldX=x;
    g_iOldY=y;
    if (btn == GLUT_LEFT_BUTTON)
    {
      g_bSpil=true;      
    }
  }
  if (state == GLUT_UP)
  {
    if (btn == GLUT_LEFT_BUTTON)
    {
      g_bSpil=false;      
    }
  }

  if (btn == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
  {
    //    cerr << "Leaf selection" << endl;
    int i,j;
    //for (int si=-5;si<5;si++)
    //  for (int sj=-5;sj<5;sj++)
    int si=0,sj=0;
    if (getLeafAt(x+si,y+sj,i,j))
    {
      if (i == 255 && j == 255)
        cerr << "OUPS !" << endl;
      else
      {
        g_Tree->addDensity(g_Tree->getLeafCenter(i,j),
          1.0+2.0*drand48(),
          g_dDropSize*(0.005+drand48()*0.003));
        g_LastSelected[0]=i;
        g_LastSelected[1]=j;
      }
    }
    else
      g_LastSelected[0]=g_LastSelected[1]=-1;
  }
}

/* -------------------------------------------------------- */

void mainMotion(int x,int y)
{
  if (g_iBtn == GLUT_RIGHT_BUTTON)
  {
    g_dRotZ+=(g_iOldX-x)*360.0/800.0;
    g_dRotX+=(g_iOldY-y)*360.0/800.0;
    g_iOldX=x;
    g_iOldY=y;
  }
  if (g_iBtn == GLUT_LEFT_BUTTON)
  {
    g_iOldX=x;
    g_iOldY=y;    
  }
}

/* -------------------------------------------------------- */

void mainReshape(int w,int h)
{
  glutSetWindow(g_iMainWindow);
  glViewport(0,0,w,h);
  g_iW=w;
  g_iH=h;
}

/* -------------------------------------------------------- */

void loadCg()
{
  // =============================================
  // load vertex program
  g_cgVPRender=cg_loadVertexProgram("simul/vp_simul_render.cg");
  g_cgVPViewProj=cgGetNamedParameter(g_cgVPRender,"ViewProj");
  g_cgVPITView=cgGetNamedParameter(g_cgVPRender,"ITView");
  g_cgVPView=cgGetNamedParameter(g_cgVPRender,"View");

  // =============================================
  // load fragment program
  g_cgFPRender=cg_loadFragmentProgram("simul/fp_simul_render.cg");
  g_cgFPScreenBuffer=cgGetNamedParameter(g_cgFPRender,"ScreenBuffer");
  cgGLSetTextureParameter(g_cgFPScreenBuffer,g_uiScreenTex);
}

// ------------------------------------------------------------

void render_tree()
{
  // render  
  glClearColor(0.0,0.0,0.0,0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(FOV,
    (double)g_iW/(double)g_iH,
    0.1,100.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  transform();

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_COLOR_MATERIAL);

  glColor3d(234.0/255.0,220.0/255.0,161.0/255.0);
  g_Tree->bind();
  draw();
  g_Tree->unbind();
}

/* -------------------------------------------------------- */

void render_final()
{
  // render  
  glClearColor(0.5,0.5,0.5,0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(FOV,
    (double)g_iW/(double)g_iH,
    0.1,100.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  transform();

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glDisable(GL_BLEND);

  glEnable(GL_COLOR_MATERIAL);

  glColor3d(234.0/255.0,220.0/255.0,161.0/255.0);

  cgGLBindProgram(g_cgFPRender);
  cgGLBindProgram(g_cgVPRender);
  cgGLEnableProfile(g_cgVertexProfile);
  cgGLEnableProfile(g_cgFragmentProfile);
  cgGLEnableTextureParameter(g_cgFPScreenBuffer);

  draw();

  cgGLDisableTextureParameter(g_cgFPScreenBuffer);
  cgGLDisableProfile(g_cgVertexProfile);
  cgGLDisableProfile(g_cgFragmentProfile);
}

/* -------------------------------------------------------- */

void renderScreenBuffer()
{
  g_Screen.Activate();

  glViewport(0,0,SCREEN_BUFFER_SIZE,SCREEN_BUFFER_SIZE);

  render_tree();

  glBindTexture(GL_TEXTURE_2D,g_uiScreenTex);
  glTexParameteri(GL_TEXTURE_2D,
    GL_GENERATE_MIPMAP_SGIS,
    GL_TRUE);
  glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,
    0,0,SCREEN_BUFFER_SIZE,SCREEN_BUFFER_SIZE);

  g_Screen.Deactivate();
}

/* -------------------------------------------------------- */

void mainRender()
{
  glViewport(0,0,SCREEN_SIZE,SCREEN_SIZE);
  // simulation step
  static int cnt=0;
  if (g_bRun)// && (cnt++)%10 == 0)
  {
    g_Tree->simulstep();
    // g_Tree->simulstep();
  }
  // rain ... hmm not very convincing
  if (g_bRain)
  {
    double x=drand48();
    double z=drand48();
    double y=drand48();
    g_Tree->addDensity(CVertex(x,y,z),1.0+2.0*drand48(),0.01+drand48()*0.04);
  }
  // spil blood
  if (g_bSpil)
  {
    int i,j;
    if (getLeafAt(g_iOldX,g_iOldY,i,j))
    {
      g_Tree->addDensity(g_Tree->getLeafCenter(i,j),
        1.0+2.0*drand48(),
        g_dDropSize*(0.005+drand48()*0.003));
      g_LastSelected[0]=i;
      g_LastSelected[1]=j;
    }
    else
    {
      g_LastSelected[0]=-1;
      g_LastSelected[1]=-1;
    }
  }

  if (!g_bFullScreenDebug)
  {

    if (g_bDrawObject)
    {
      renderScreenBuffer();
      render_final();
    }

    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_POLYGON_OFFSET_FILL);

    if (g_bDrawWireframe)
    {
      // draw wireframe object
      glEnable(GL_LINE_SMOOTH);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
      glPolygonOffset(-2.0,0.0);
      glEnable(GL_POLYGON_OFFSET_LINE);
      glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
      glColor3d(0.0,0.0,0.0);
      draw();
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      glDisable(GL_POLYGON_OFFSET_LINE);
      glDisable(GL_LINE_SMOOTH);
      glDisable(GL_BLEND);
    }

    glDepthFunc(GL_LESS);
  }

  // draw debug data
  if (g_bFullScreenDebug || 
    g_bDrawSprites
    || g_bDrawTree)
  {
    // =================
    // insert
    if (g_bFullScreenDebug)
    {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glViewport(0,0,g_iW,g_iH);
    }
    else
    {
      glClear(GL_DEPTH_BUFFER_BIT);
      glViewport(0,g_iH-g_iH/2,g_iW/2,g_iH/2);

      // bkg
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0,1.0,1.0,0.0);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glDepthMask(0);
      glColor4d(1.0,1.0,1.0,0.2);
      glBegin(GL_QUADS);
      glVertex2i(0,0);
      glVertex2i(0,1);
      glVertex2i(1,1);
      glVertex2i(1,0);
      glEnd();
      glDepthMask(0xFF);
    }

    // 3d tex space
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV,
      (double)g_iW/(double)g_iH,
      0.1,100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    transform();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);

    if (g_bDrawObject)
    {
      glColor3d(234.0/255.0,220.0/255.0,161.0/255.0);
      g_Tree->bind();
      draw_texspace();
      g_Tree->unbind();
      glPolygonOffset(-1.0,0.0);
      glEnable(GL_POLYGON_OFFSET_FILL);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
      glDisable(GL_POLYGON_OFFSET_FILL);
    }

    glDisable(GL_LIGHTING);

    if (g_LastSelected[0] >= 0)
      g_Tree->draw_nodes(g_LastSelected[0],g_LastSelected[1]);

    // draw tree structure
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    if (g_bDrawTree)
    {
      glEnable(GL_FOG);
      GLfloat fs=g_dDist-0.9;
      glFogfv(GL_FOG_START,&fs);
      GLfloat fe=g_dDist+0.5;
      glFogfv(GL_FOG_END,&fe);
      GLint fm=GL_LINEAR;
      glFogiv(GL_FOG_MODE,&fm);
      g_Tree->draw_structure(g_bDrawEmpty);
      glDisable(GL_FOG);
    }
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_BLEND);
  }

  if (g_bDebug)
  {
    // show screen buffer texture
    glViewport(g_iW-g_iW/3,0,g_iW/3,g_iH/4);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0,1.0,0.0,1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,g_uiScreenTex);
    glColor3d(1,1,1);
    quad();
    glDisable(GL_TEXTURE_2D);      
  }

  PROFILER.step();

  glViewport(0,0,g_iW,g_iH);

  // draw help
  glPushAttrib(GL_ENABLE_BIT);

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0,1.0,1.0,0.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  drawHelp();

  glPopAttrib();

  if (!g_bRecord)
  {
    // draw profiler
    if (g_bHUD)
    {
      glPushAttrib(GL_ENABLE_BIT);

      glDisable(GL_DEPTH_TEST);
      glEnable(GL_LINE_SMOOTH);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

      glViewport(0,0,g_iW/3,g_iH/4);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0,1.0,1.0,0.0);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glColor4d(1.0,1.0,1.0,0.2);
      glBegin(GL_QUADS);
      glVertex2i(0,0);
      glVertex2i(0,1);
      glVertex2i(1,1);
      glVertex2i(1,0);
      glEnd();

      PROFILER.draw();
    }
  }
  // fps
  double tm;
  if ((tm=(PROFILER.getRealTime()-g_iLastTime)) > 1000)
  {
    g_dFPS=(g_iNbFrames*1000.0/(double)tm);
    g_iNbFrames=0;
    g_iLastTime=(int)PROFILER.getRealTime();
  }
  else
    g_iNbFrames++;
  // display fps
  static char strfps[16];
  sprintf(strfps,"%3d FPS",(int)g_dFPS);
  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glColor3d(1,1,1);
  g_Font->printString(0.0,0.0,0.03,strfps);


  glViewport(0,0,g_iW,g_iH);

  glPopAttrib();

  // swap
  glutSwapBuffers();

#ifdef WIN32
  if (g_bRecord)
    g_VCR.Snap();
#endif

}

/* -------------------------------------------------------- */

bool getLeafAt(double x,double y,int& _i,int& _j)
{
  int                 oldv[4];
  unsigned char       pix[4];

  glGetError();

  y=g_iH-y;

  glGetIntegerv(GL_VIEWPORT,oldv);

  glViewport(0,0,4,4);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPickMatrix(x,y,4.0,4.0,oldv);
  gluPerspective(FOV,
    (double)g_iW/(double)g_iH,
    0.1,100.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  transform();

  // read position
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // -> bind
  g_Tree->getHrdwTree()->bind_leafselect();
  // -> draw
  if (g_bFullScreenDebug)
    draw_texspace();
  else
    draw();
  // -> undind
  g_Tree->getHrdwTree()->unbind_leafselect();
  // -> flush
  glFlush();
  // -> read pixel
  glReadPixels(0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE,pix);
  // done
  glViewport(oldv[0],oldv[1],oldv[2],oldv[3]);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (pix[3] == 0)
    return (false);

  int i=CPU_DECODE_INDEX8_U(pix[0],pix[1],pix[2]);
  int j=CPU_DECODE_INDEX8_V(pix[0],pix[1],pix[2]);

  //  cerr << "Leaf at " << i << ',' << j << endl;

  _i=i;
  _j=j;

  if (glGetError())
    cerr << "GL Error (getLeafAt)" << endl;

  return (true);
}

/* -------------------------------------------------------- */

void idle()
{
  glutSetWindow(g_iMainWindow);  
  glutPostRedisplay();
}

/* -------------------------------------------------------- */

void transform()
{
  glTranslated(0.0,0.0,-g_dDist);
  glRotated(g_dRotX,1.0,0.0,0.0);
  glRotated(g_dRotZ,0.0,1.0,0.0);
  //  glTranslated(-0.5,-0.5,-0.5);

  glTranslated(-g_Object->center().x(),
    -g_Object->center().y(),
    -g_Object->center().z());

}

/* -------------------------------------------------------- */

void draw()
{
  glPushMatrix();

  g_Tree->setCgTransform();

  cgGLSetStateMatrixParameter(g_cgVPViewProj,
    CG_GL_MODELVIEW_PROJECTION_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(g_cgVPView,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(g_cgVPITView,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_INVERSE_TRANSPOSE);
  g_Object->draw();

  glPopMatrix();
}

/* -------------------------------------------------------- */

void draw_texspace()
{
  glPushMatrix();

  g_Tree->setCgTransform();

  cgGLSetStateMatrixParameter(g_cgVPViewProj,
    CG_GL_MODELVIEW_PROJECTION_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(g_cgVPView,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_IDENTITY);
  cgGLSetStateMatrixParameter(g_cgVPITView,
    CG_GL_MODELVIEW_MATRIX,
    CG_GL_MATRIX_INVERSE_TRANSPOSE);

  g_Object->draw();

  glPopMatrix();
}

/* -------------------------------------------------------- */

int main(int argc, char **argv)
{
  if (argc < 2)
  {
      stringstream str;
      str << "simul "
			  << " <geom.3ds>" 
				<< endl;
#ifdef WIN32
      MessageBox(NULL,str.str().c_str(),"Please provide the following arguments",MB_OK |MB_ICONINFORMATION);
#else
      cerr << str;
#endif
			exit (-1);
  }

  // GLUT
  glutInit(&argc, argv);
  glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
  // main window
  g_iMainWindow=glutCreateWindow("Liquid flowing along a surface");
  glutMouseFunc(mainMouse);
  glutMotionFunc(mainMotion);
  glutKeyboardFunc(mainKeyboard);
  glutDisplayFunc(mainRender);
  glutReshapeFunc(mainReshape);
  glutIdleFunc(idle);
  glutSetWindow(g_iMainWindow);

  char *cgpath="../cg";
  for (int i=1;i<argc;i++)
  {
    if (!strcmp(argv[i],"--cg_path"))
    {
      if (i+1<argc)
      {
        cgpath=argv[i+1];
      }
    }
  }

  try
  {
    // init glux
    gluxInit();

    // check error

    if (glGetError())
      cerr << "main - GL Error (0)" << endl;

    // init gl
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);
    glClearColor(1.0,1.0,1.0,0);

    float diff[]={234.0/255.0,220.0/255.0,161.0/255.0,1.0};
    glMaterialfv(GL_FRONT,GL_DIFFUSE,diff);
    float spec[]={0.3,0.3,0.3,1.0};
    glMaterialfv(GL_FRONT,GL_SPECULAR,spec);
    float shin[]={30.0,30.0,30.0,1.0};
    glMaterialfv(GL_FRONT,GL_SHININESS,shin);

    if (glGetError())
      cerr << "main - GL Error (1)" << endl;

    // init Cg
    cg_init();
    cg_set_path(cgpath);

    if (glGetError())
      cerr << "main - GL Error (2)" << endl;

    // font
    g_Font=new GLfont("default.tga");

    // profiler
    PROFILER.init(g_Font);
    PROFILER.addVar(&g_dFPS,"FPS",CProfiler::red,1000);
    g_iTimerDraw=PROFILER.createTimer("draw",CProfiler::yellow);
    g_iTimerCommit=PROFILER.createTimer("commit",CProfiler::green);

    // init pbuffer and back screen texture
    g_Screen.Initialize(SCREEN_BUFFER_SIZE,SCREEN_BUFFER_SIZE,false,true);
    glGenTextures(1,&g_uiScreenTex);
    glBindTexture(GL_TEXTURE_2D,g_uiScreenTex);
    glTexImage2D(GL_TEXTURE_2D,0,
      GL_RGBA,
      SCREEN_BUFFER_SIZE,SCREEN_BUFFER_SIZE,0,
      GL_RGBA,GL_UNSIGNED_BYTE,NULL);
    glTexParameteri(GL_TEXTURE_2D,
      GL_TEXTURE_MAG_FILTER,
      GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,
      GL_TEXTURE_MIN_FILTER,
      GL_LINEAR_MIPMAP_LINEAR);

    // load programs
    loadCg();

    // doc
    printf("[q]     - quit\n");

    if (glGetError())
      cerr << "main - GL Error (3)" << endl;

    char *order="yzx";
    if (argc > 2)
      order=argv[2];
    g_Object=new CGL3DSMesh(argv[1],order);

    // warning
    cerr << endl;
    cerr << " ---------------- " << endl << endl;
    cerr << " Loading may be quite long ..." << endl;
    cerr << "       please wait" << endl;
    cerr << endl;
    cerr << " ---------------- " << endl << endl;

    // tree
    std::list<CPolygon> polys;
    g_Object->polygons(polys);
    cerr << "Creating tree ... " << endl;
    g_Tree=new CSimulTree(polys,FPPROG);
    cerr << "... done." << endl;
    g_Tree->report();

    if (glGetError())
      cerr << "main - GL Error (6)" << endl;

    // let's go
    glutMainLoop();

  }
  // handle exceptions
  catch (CLibTextureException& e)
  {
#ifdef WIN32
    MessageBox(NULL,e.getMsg(),"FATAL Error",MB_OK | MB_ICONSTOP);
#else
    cerr << "[FATAL] " << e.getMsg() << endl;
#endif
  }
  catch (CLibOctreeGPUException& e)
  {
#ifdef WIN32
    MessageBox(NULL,e.getMsg(),"FATAL Error",MB_OK | MB_ICONSTOP);
#else
    cerr << "[FATAL] " << e.getMsg() << endl;
#endif
  }
  catch (CCoreException& e)
  {
#ifdef WIN32
    MessageBox(NULL,e.getMsg(),"FATAL Error",MB_OK | MB_ICONSTOP);
#else
    cerr << "[FATAL] " << e.getMsg() << endl;
#endif
  }

  return (0);
}

/* -------------------------------------------------------- */
