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
        Painting on unparameterized meshes
  --

  This file is mostly about user interface.

  To use an octree in your app, see the two main steps:    

    A) Create an octree texture                 (main)
    B) Drawing an object with an octree texture (mainRender)

  To see how the octree texture is built upon liboctreegpu, 
  look into CPaintNode.h/.cpp and CPaintTree.h/.cpp

*/
/* -------------------------------------------------------- */

#ifdef WIN32
#include <windows.h>
#else
#include <qapplication.h>
#include <qcolor.h>
#include <qcolordialog.h>
#endif

#include "config.h"
#include "help.h"
#include "CTexture.h"
#include "CCoreException.h"

/* -------------------------------------------------------- */

#define SCREEN_SIZE 512
#define FOV         45.0

/* -------------------------------------------------------- */

// Fragment programs to use
#define FPPROG             "paint/fp_diffus_color_tree.cg"
#define FPPROG_INTERP      "paint/fp_diffus_interp_color_tree.cg"

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

#include "cg_load.h"

#include <glfont.h>

#include "CProfiler.h"

#include "CVertex.h"
#include "CQuaternion.h"
#include "CBox.h"

#include "CPaintTree.h"
#include "CGL3DSMesh.h"
#include "CGenAtlas.h"

/* -------------------------------------------------------- */

#include <list>
using namespace std;

/* -------------------------------------------------------- */

// Window, viewpoint and mouse

int             g_iMainWindow;               // Glut Window Id
int             g_iFrame      = 0;           // Frame counter
int             g_iW          = SCREEN_SIZE; // Window width
int             g_iH          = SCREEN_SIZE; // Window height
int             g_iOldX       = -1;          // Old mouse X pos
int             g_iOldY       = -1;          // Old mouse Y pos
int             g_iBtn        = 0;           // Mouse button pressed
double          g_dRotX       = 90.0;        // X axis rotation
double          g_dRotZ       = 0.0;         // Z axis rotation
double          g_dDist       = 2.0;         // Viewing distance
CVertex         g_Center(0.5,0.5,0.5);       // View center
GLfont         *g_Font        = NULL;        // Font

// Performance measures

int             g_iTimerDraw   = -1;
int             g_iTimerCommit = -1;
double          g_dFPS;         
int             g_iNbFrames    = 0;
int             g_iLastTime    = 0;

// Display states

bool            g_bDrawTree      = false;
bool            g_bDrawEmpty     = true;
bool            g_bDrawObject    = true;
bool            g_bDrawWireframe = false;
bool            g_bFullScreen    = false;
bool            g_bDrawUVMap     = false;

// Painting

bool            g_bPaint              = false;
int             g_iPaintX             = 0;
int             g_iPaintY             = 0;
bool            g_bRefine             = false;
bool            g_bUpdatePaintPos     = false;
int             g_iDrawSubset         = 0;
GLuint          g_uiSubsetDisplayList = 0;
double          g_dPaintRadius        = 0.05;
CVertex         g_PaintPos(1.0,1.0,1.0);
double          g_dPaintOpacity       = 1.0;
unsigned char   g_Color[3]={0,0,255};

// Mesh, tree and atlas

CGL3DSMesh     *g_Object         = NULL;
CPaintTree     *g_Tree           = NULL;
CGenAtlas      *g_Atlas          = NULL;
int             g_iAtlasMIPLevel = 0;

// Filename for saving and loading

char           *g_FileName="out.octree";

// VCR and color chooser

#ifdef WIN32
#include "MovieMaker.h"
MovieMaker       g_VCR;
bool             g_bRecord=false;

static void* getCurrentHWND()   
{     // get current DC from wgl     
	return WindowFromDC(wglGetCurrentDC());   
}
#else
QApplication *g_QApp=NULL; // just for the color chooser
#endif

/* -------------------------------------------------------- */

void    draw();
void    draw_atlas();
void    report();
void    transform();
bool    getPosNormalAt(double x,double y,CVertex&,CVertex&);
CVertex getPointAt(int x,int y);

/* -------------------------------------------------------- */

/**
Draw a simple quad
*/
void quad()
{
	glBegin(GL_QUADS);

	glTexCoord2f(0,0);
	glVertex2i(0,0);

	glTexCoord2f(0,1);
	glVertex2i(0,1);

	glTexCoord2f(1,1);
	glVertex2i(1,1);

	glTexCoord2f(1,0);
	glVertex2i(1,0);

	glEnd();
}

/* -------------------------------------------------------- */

/**
Keyboard callback
*/
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
	else if (key == 'o')
	{
		g_dDist+=0.1;
	}
	else if (key == 'p')
	{
		g_dDist-=0.1;
	}  
	else if (key == 'O')
	{
		g_dDist+=0.001;
	}
	else if (key == 'P')
	{
		g_dDist-=0.001;
	}  
	else if (key == 't')
	{
		g_bDrawTree=!g_bDrawTree;
	}  
	else if (key == '!')
	{
		if (g_iDrawSubset > 0)
			g_iDrawSubset--;
		else
			g_iDrawSubset=3;
	}  
	else if (key == ' ')
	{
		g_Center=g_PaintPos;
	}  
	else if (key == 'c')
	{
#ifdef WIN32
		CHOOSECOLOR     cc;             // common dialog box structure 
		static COLORREF acrCustClr[16]; // array of custom colors 

		// Initialize CHOOSECOLOR 
		ZeroMemory(&cc, sizeof(cc));
		cc.lStructSize = sizeof(cc);
		cc.hwndOwner = (HWND)getCurrentHWND();
		cc.lpCustColors = (LPDWORD) acrCustClr;
		cc.rgbResult = ((int)g_Color[2] << 16) + ((int)g_Color[1] << 8) + (int)g_Color[0];
		cc.Flags = CC_FULLOPEN | CC_RGBINIT;

		if (ChooseColor(&cc) == TRUE) 
		{
			g_Color[2] = (cc.rgbResult >> 16) & 255;
			g_Color[1] = (cc.rgbResult >>  8) & 255;
			g_Color[0] = (cc.rgbResult      ) & 255;
		}
#else
		// use QT
		QColor clr=QColorDialog::getColor(QColor(g_Color[0],g_Color[1],g_Color[2]),NULL,NULL);
		g_Color[0]=clr.red();
		g_Color[1]=clr.green();
		g_Color[2]=clr.blue();
#endif 
	}  
	else if (key == 'f')
	{
		g_bFullScreen=!g_bFullScreen;
	}  
	else if (key == 'w')
	{
		g_bDrawWireframe=!g_bDrawWireframe;
	}  
	else if (key == 'u')
	{
		g_bDrawUVMap=!g_bDrawUVMap;
	}  
	else if (key == 'y')
	{
		g_iAtlasMIPLevel=(g_iAtlasMIPLevel+1)%g_Atlas->nbLevels();
	}
	else if (key == 'e')
	{
		static bool swap=false;
		g_Atlas->setExtrapolate(swap);
		swap=!swap;
	}
	else if (key == 'i')
	{
    static bool swap=true;
    if (swap)
      g_Tree->changePrograms(FPPROG_INTERP);
    else
      g_Tree->changePrograms(FPPROG);
    swap=!swap;
  }
	else if (key == '+')
	{
		g_dPaintRadius+=0.001;
	}
	else if (key == '-')
	{
		g_dPaintRadius-=0.001;
	}
	else if (key == '9')
	{
		g_dPaintRadius-=0.00001;
	}
	else if (key == '6')
	{
		g_dPaintRadius+=0.00001;
	}
	else if (key == 'l')
	{
		g_dPaintOpacity+=0.01;
		g_dPaintOpacity=min(g_dPaintOpacity,1.0);
	}
	else if (key == 'k')
	{
		g_dPaintOpacity-=0.01;
		g_dPaintOpacity=max(g_dPaintOpacity,0.0);
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
	else if (key == 'r')
	{
		g_bRefine=!g_bRefine;
	}
	else if (key == 's')
	{
		g_Tree->save(g_FileName);
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
		//    PROFILER.normalize();
		g_Tree->report();
	}
}

/* -------------------------------------------------------- */

/**
Mouse callback
*/
void mainMouse(int btn, int state, int x, int y) 
{
	g_iBtn=btn;

	if (state == GLUT_DOWN)
	{
		g_iOldX=x;
		g_iOldY=y;

		if (btn == GLUT_LEFT_BUTTON)
		{
      // start painting
			g_bUpdatePaintPos=true;
			g_iPaintX=x;
			g_iPaintY=y;
			g_Tree->startDrawing();
		}
	}
}

/* -------------------------------------------------------- */

/**
Mouse motion callback
*/
void mainMotion(int x,int y)
{
	if (g_iBtn == GLUT_RIGHT_BUTTON)
	{
    // rotate view
		g_dRotZ+=(g_iOldX-x)*360.0/800.0;
		g_dRotX+=(g_iOldY-y)*360.0/800.0;
		g_iOldX=x;
		g_iOldY=y;
	}
	else if (g_iBtn == GLUT_LEFT_BUTTON)
	{
    // paint
    g_bUpdatePaintPos=true;
		g_iPaintX=x;
		g_iPaintY=y;
		if (g_iDrawSubset >= 2)
		{
			// subset selection - nothing to do
		}
		else
		{
			// painting
			g_bPaint=true;
		}
	}
}

/* -------------------------------------------------------- */

/**
Reshape callback
*/
void mainReshape(int w,int h)
{
	glutSetWindow(g_iMainWindow);
	glViewport(0,0,w,h);
	g_iW=w;
	g_iH=h;
}

/* -------------------------------------------------------- */

/**
Display tree structure for visualization purpose
*/
void debugRender()
{
	if (g_bFullScreen)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0,0,g_iW,g_iH);
	}
 	else
	{
    // draw in a small window

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
		quad();
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
		draw();
		glPolygonOffset(-1.0,0.0);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_POLYGON_OFFSET_FILL);
	}

	glDisable(GL_LIGHTING);

	// draw tree structure
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_LIGHTING);
	if (g_bDrawTree || g_bFullScreen)
	{
		glEnable(GL_FOG);
		GLfloat fs=g_dDist-0.9;
		glFogfv(GL_FOG_START,&fs);
		GLfloat fe=g_dDist+0.5;
		glFogfv(GL_FOG_END,&fe);
		GLint fm=GL_LINEAR;
		glFogiv(GL_FOG_MODE,&fm);
		glLineWidth(1.0);
		glColor3d(1,1,1);
		if (g_bDrawEmpty)
			g_Tree->getRoot()->draw_structure(true);
		else
			g_Tree->getRoot()->box().draw_box_line();
		glDisable(GL_FOG);
	}
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);
}

/* -------------------------------------------------------- */

/**
Draw texture atlas (when converting octree into texture)
*/
void atlasRender()
{
	glColor3f(1,1,1);

	// create atlas
	g_Atlas->begin();
	g_Tree->bind();
	draw_atlas();
	g_Tree->unbind();
	g_Atlas->end();

	// bind the atlas
	glEnable(GL_TEXTURE_2D);
	g_Atlas->bind();
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_LOD_SGIS,g_iAtlasMIPLevel);

	// render mesh with atlas texture
	if (g_bFullScreen)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0,0,g_iW,g_iH);
	}
	else
		glViewport(g_iW-g_iW/3,0,g_iW/3,g_iH/3);
	g_Object->draw_uv();

	// display atlas texture
	glViewport(0,0,g_iW/3,g_iH/3);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0,1.0,1.0,0.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	quad();
	glDisable(GL_TEXTURE_2D);  
}

/* -------------------------------------------------------- */

/**
Render the main view - textured mesh
*/
void mainRender()
{
  // update point pos ?
	if (g_bUpdatePaintPos)
	{
    // retrieve paint pos
		g_PaintPos=getPointAt(g_iPaintX,g_iPaintY);
		g_bUpdatePaintPos=false;

    // draw only a mesh subset (useful on slow computers)
		if (g_iDrawSubset >= 1)
		{
			if (g_uiSubsetDisplayList == 0)
				g_uiSubsetDisplayList=glGenLists(1);
			glNewList(g_uiSubsetDisplayList,GL_COMPILE);
			g_Tree->drawSubset(g_PaintPos,g_dPaintRadius,6);
			glEndList();
			if (g_iDrawSubset == 3)
				g_iDrawSubset=2;
		}

    // painting ?
		if (g_bPaint)
		{
			if (!g_bRefine)
			{
        // apply paint
				g_Tree->paint(g_PaintPos,
					g_dPaintRadius,
					g_dPaintOpacity,
					g_Color[0],g_Color[1],g_Color[2]);
			}
			else
			{
        // refinment brush
				std::list<CPolygon> polys;
				g_Object->polygons(polys);
				g_Tree->refine(g_PaintPos,
					g_dPaintRadius,
					PAINT_MAX_DEPTH,
					polys);
			}
			PROFILER.startTimer(g_iTimerCommit);
			g_Tree->commit();
			PROFILER.stopTimer(g_iTimerCommit);
			g_bPaint=false;
		}
	}
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glDisable(GL_BLEND);

	// render  
	glViewport(0,0,g_iW,g_iH);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOV,
		(double)g_iW/(double)g_iH,
		0.1,100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	transform();

	glEnable(GL_COLOR_MATERIAL);

	if (!g_bFullScreen)
	{    
  	// draw object
		if (g_bDrawObject)
		{
      //---------------------------------------------
      //  B) Drawing an object with an octree texture
      //
      // Bind octree texture
      g_Tree->bind();
      // draw object
			draw();
      // Unbind octree texture
			g_Tree->unbind();
		}

		glDisable(GL_BLEND);
		glDisable(GL_ALPHA_TEST);
		glDisable(GL_POLYGON_OFFSET_FILL);

		// draw paint tool
		glDepthMask(0x00);
		glPushMatrix();
		glTranslated(g_PaintPos.x(),g_PaintPos.y(),g_PaintPos.z());
		if (g_iDrawSubset >= 2)
		{
			glColor3f(1,1,0);
			glutWireSphere(g_dPaintRadius,10,10);
		}
		else if (g_bRefine)
		{
			glColor3f(1,0,0);
			glutWireSphere(g_dPaintRadius,10,10);
		}
		else
		{
			glutWireSphere(g_dPaintRadius,10,10);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
			glColor4f((float)g_Color[0]/255.0f,(float)g_Color[1]/255.0f,(float)g_Color[2]/255.0f,g_dPaintOpacity);
			glutSolidSphere(g_dPaintRadius,10,10);
			glDisable(GL_BLEND);
		}
		glPopMatrix();
		glDepthMask(0xFF);

    // draw wireframe
		if (g_bDrawWireframe)
		{
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

  // draw debug (tree structure)
	if (g_bDrawTree || (g_bFullScreen && !g_bDrawUVMap))
	{
		debugRender();
	}
	// draw atlas
	else if (g_bDrawUVMap)
	{
		atlasRender();
	}

  // fps
  int tm;
	if ((tm=((int)PROFILER.getRealTime()-g_iLastTime)) > 1000)
	{
		g_dFPS=(g_iNbFrames*1000.0/(double)tm);
		g_iNbFrames=0;
		g_iLastTime=(int)PROFILER.getRealTime();
	}
	else
		g_iNbFrames++;
	PROFILER.step();
  // display fps
  static char strfps[16];
  sprintf(strfps,"%3d FPS",(int)g_dFPS);
  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glColor3d(1,1,1);
  g_Font->printString(0.0,0.0,0.03,strfps);

	// swap
	glutSwapBuffers();

#ifdef WIN32
	if (g_bRecord)
		g_VCR.Snap();
#endif

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

	glRotated(-g_dRotX,1.0,0.0,0.0);
	glRotated(-g_dRotZ,0.0,0.0,1.0);
	glTranslated(-g_Center.x(),-g_Center.y(),-g_Center.z());
}

/* -------------------------------------------------------- */

void draw()
{
	glPushMatrix();

	g_Tree->setCgTransform();

	if (g_iDrawSubset > 0 && g_iDrawSubset < 3)
		glCallList(g_uiSubsetDisplayList);
	else
		g_Object->draw();
	
	glPopMatrix();
}

/* -------------------------------------------------------- */

void draw_atlas()
{
	glPushMatrix();

	g_Tree->setCgTransform();

	g_Object->draw_atlas();

	glPopMatrix();
}

/* -------------------------------------------------------- */
/**
Retrieve 3D coordinates of point under mouse cursor
*/
CVertex getPointAt(int x,int y)
{
	static  GLfloat prevpix=1.0f;
	GLfloat pix=0.9f;

	glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
	glClear(GL_DEPTH_BUFFER_BIT);

	glViewport(0,0,g_iW,g_iH);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOV,
		(double)g_iW/(double)g_iH,
		0.1,100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	transform();

	draw();

	glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

	// -> read pixel
	glReadBuffer(GL_BACK);
	glReadPixels(x,g_iH-y-1,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&pix);

	int      vp[4];
	GLdouble mdl[16],prj[16];
	GLdouble px,py,pz;

	glGetIntegerv(GL_VIEWPORT,vp);
	glGetDoublev(GL_MODELVIEW_MATRIX,mdl);
	glGetDoublev(GL_PROJECTION_MATRIX,prj);

	if (pix == 1.0f)
		pix=prevpix;
	else
		prevpix=pix;

	gluUnProject(x,g_iH-y-1,pix,
		mdl,prj,vp,
		&px,&py,&pz);

	return (CVertex(px,py,pz));
}


/* -------------------------------------------------------- */

int main(int argc, char **argv) 
{
	try
	{
#ifndef WIN32
		g_QApp=new QApplication(argc,argv); // required by the color chooser
#endif
		if (argc < 2)
		{
      stringstream str;
      str << "paint "
			  << " <geom.3ds>" 
        << " [<texture.octree>]" 
				<< endl;
#ifdef WIN32
      MessageBox(NULL,str.str().c_str(),"Please provide the following arguments",MB_OK |MB_ICONINFORMATION);
#else
      cerr << str;
#endif
			exit (-1);
		}

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

		// GLUT
		glutInit(&argc, argv);
		glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);

    // main window
		g_iMainWindow=glutCreateWindow("Painting on Unparametrized Meshes");
		glutMouseFunc(mainMouse);
		glutMotionFunc(mainMotion);
		glutKeyboardFunc(mainKeyboard);
		glutDisplayFunc(mainRender);
		glutReshapeFunc(mainReshape);
		glutIdleFunc(idle);
		glutSetWindow(g_iMainWindow);

		gluxInit();

		if (glGetError())
			cerr << "main - GL Error (0)" << endl;

		// init gl
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable(GL_NORMALIZE);
		glClearColor(0.5,0.5,0.5,0);

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

		// doc
		printf("[q]     - quit\n");

		if (glGetError())
			cerr << "main - GL Error (3)" << endl;

		// load geometry
		g_Object=new CGL3DSMesh(argv[1],"xyz");
    g_Center=g_Object->center();

		// atlas
		g_Atlas=new CGenAtlas(512);

    // warning
    cerr << endl;
    cerr << " ---------------- " << endl << endl;
    cerr << " Loading may be quite long ..." << endl;
    cerr << "       please wait" << endl;
    cerr << endl;
    cerr << " ---------------- " << endl << endl;

    //---------------------------------------------
    //  A) Create an octree texture
    //
    // retrieve object polygons
    std::list<CPolygon> polys;
		g_Object->polygons(polys);

		if (argc > 2)
		{
			g_FileName=argv[2];
      // load tree from file
			g_Tree=new CPaintTree(g_FileName,polys,FPPROG);
		}
		else
      // create na ew tree
			g_Tree=new CPaintTree(polys,FPPROG);
    // display tree information
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
