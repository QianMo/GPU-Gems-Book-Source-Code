//-----------------------------------------------------------------------------
// File : main.cpp
//----------------------------------------------------------------------------
// Copyright 2003 Mark J. Harris and
//     The University of North Carolina at Chapel Hill
//----------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software and its 
// documentation for any purpose is hereby granted without fee, provided that 
// the above copyright notice appear in all copies and that both that 
// copyright notice and this permission notice appear in supporting 
// documentation.  Binaries may be compiled with this software without any 
// royalties or restrictions. 
//
// The author(s) and The University of North Carolina at Chapel Hill make no 
// representations about the suitability of this software for any purpose. 
// It is provided "as is" without express or implied warranty.
/**
 * @file main.cpp
 * 
 * Flo main file.
 */
#include <gl/glew.h>
#include <gl/glut.h>
#include <cstdio>
#include <stdlib.h>
#include "stopwatch.hpp"
#include "flo.h"
#include "paramgl/paramgl.h"
#include <assert.h>

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------
// The Cg context
CGcontext g_cgContext = NULL;

// The UI.
ParamListGL *paramlist;  // parameter list
// The User interface (displayed when true).
bool g_bDisplaySliders = false;

// interaction stuff:
int  g_iMouseButton = 0;
bool g_bMouseDown = false;
bool g_bCtrlDown = false; 

int g_iMouseX = 0;
int g_iMouseY = 0;

// The simulator
Flo *g_pFlo = 0;

// simulation constants
int g_iFloWidth  = 64;
int g_iFloHeight = 64;

// Booleans for simulation behavior
bool	  g_bDisplayFluid	  = true;
bool      g_bBilerp           = true;
bool      g_bMakeTex          = false;
bool      g_bArbitraryBC      = false;
bool      g_bClearPressure    = true;
bool      g_bComputeVorticity = true;
// Pause / Simulation stepping.
bool	  g_bPause			  = false;
bool	  g_bSingleStep		  = false;

// For tracking framerate.
Stopwatch g_timer;
int       g_iFrameCount = 0;

Stopwatch g_perfTimer;
bool      g_bTiming = false;

Flo::DisplayMode g_displayMode = Flo::DISPLAY_INK;

// parameters

float     g_rVCScale         = 0.025f;
int       g_iNumPoissonSteps = 50;

float     g_rViscosity       = 0;
float     g_rInkRGB[3]     = { 0.54f, 0.2f, 0.0f };
float     g_rInkLongevity  = 0.997;
float	  g_rBrushRadius		= 0.1;
float     g_rTimestep        = 1;
float     g_rGridScale       = 1;
//----------------------------------------------------------------------------


// forward decls
void Shutdown();

// because we can't rely on everyone to have a good STL installed...
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) (((x) < (y)) ? (y) : (x))
#endif

//----------------------------------------------------------------------------
// Function     	  : cgErrorCallback
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn cgErrorCallback()
 * @brief Relays errors reported by CG.
 */ 
void cgErrorCallback()
{
  CGerror lastError = cgGetError();
  
  if(lastError)
  {
    printf("%s\n\n", cgGetErrorString(lastError));
    printf("%s\n", cgGetLastListing(g_cgContext));
    printf("Cg error, exiting...\n");
    
    exit(0);
  }
} 

//----------------------------------------------------------------------------
// Function     	: InitParameters
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn InitParameters()
 * @brief Initializes the UI Sliders
 */ 
void InitParameters()
{
  // create a new parameter list
  paramlist = new ParamListGL("misc");
  paramlist->bar_col_outer[0] = 0.8f;
  paramlist->bar_col_outer[1] = 0.8f;
  paramlist->bar_col_outer[2] = 0.0f;
  paramlist->bar_col_inner[0] = 0.8f;
  paramlist->bar_col_inner[1] = 0.8f;
  paramlist->bar_col_inner[2] = 0.0f;
  
  // add some parameters to the list

  // How many iterations to run the poisson solvers.  
  paramlist->AddParam(new Param<int>("Solver Iterations", g_iNumPoissonSteps, 
                                     1, 100, 1, &g_iNumPoissonSteps));
  // The size of the time step taken by the simulation
  paramlist->AddParam(new Param<float>("Time step", g_rTimestep, 
									   0.1f, 10, 0.1f, &g_rTimestep));
  // The Grid Cell Size
  paramlist->AddParam(new Param<float>("Grid Scale", g_rGridScale, 
									   0.1f, 100, 0.1f, &g_rGridScale));
  // Scales the vorticity confinement force.
  paramlist->AddParam(new Param<float>("Vort. Conf. Scale", g_rVCScale, 
									   0, .25, 0.005f, &g_rVCScale));
  // The viscosity ("thickness") of the fluid.
  paramlist->AddParam(new Param<float>("Viscosity", g_rViscosity, 
									   0, 0.005f, 0.0001f, &g_rViscosity));
  
  // How slow or fast the Ink fades.  1 = does not fade.
  paramlist->AddParam(new Param<float>("Ink Longevity", g_rInkLongevity, 
									   0.99f, 1, 0.0001, &g_rInkLongevity));
  // The size of the "brush" the user draws with
  paramlist->AddParam(new Param<float>("Brush Radius", g_rBrushRadius, 
									   0.005, .25, .005, &(g_rBrushRadius)));
   // The Ink color, RGB.
  paramlist->AddParam(new Param<float>("Ink Red", g_rInkRGB[0], 
									   0.0, 1.0, 0.01f, &(g_rInkRGB[0])));
  paramlist->AddParam(new Param<float>("    Green", g_rInkRGB[1], 
									   0.0, 1.0, 0.01f, &(g_rInkRGB[1])));
  paramlist->AddParam(new Param<float>("    Blue", g_rInkRGB[2], 
									   0.0, 1.0, 0.01f, &(g_rInkRGB[2])));
  
  g_perfTimer.Reset();
}


//----------------------------------------------------------------------------
// Function     	  : Display
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Display()
 * @brief GLUT display callback.
 */ 
void Display()
{
  //g_timer.Reset();

  if (!g_bPause || g_bSingleStep)
  {
    g_bSingleStep = false;
     
    // set parameters that may have changed
    g_pFlo->SetViscosity(g_rViscosity);
    g_pFlo->EnablePressureClear(g_bClearPressure);
    g_pFlo->SetNumPoissonSteps(g_iNumPoissonSteps);
    g_pFlo->SetMassColor(g_rInkRGB);
    g_pFlo->SetInkLongevity(g_rInkLongevity);
    g_pFlo->SetTimeStep(g_rTimestep);
    g_pFlo->SetGridScale(g_rGridScale);
    g_pFlo->SetVorticityConfinementScale(g_rVCScale);
    
    if (g_displayMode == Flo::DISPLAY_VORTICITY || g_bComputeVorticity)
      g_pFlo->EnableVorticityComputation(true);
    else
      g_pFlo->EnableVorticityComputation(false);
    
	// For benchmarking...
    if (g_bTiming)
    {
      if (g_perfTimer.GetNumStarts() == 100)
      {
        g_bTiming = false;
        g_perfTimer.Stop();
        printf("Average iteration time: %f\n", g_perfTimer.GetAvgTime());
      }
      g_perfTimer.Start();
    }

	// Take a simulation timestep.
    g_pFlo->Update();
  }
  
  if (g_bDisplayFluid)
  {
	// Display the fluid.
	g_pFlo->Display(g_displayMode, g_bBilerp, g_bMakeTex, g_bArbitraryBC);
	  
	// Display user interface.
	if (g_bDisplaySliders)
	{
	  paramlist->Render(0, 0);
	}
	  
	glutSwapBuffers();
  }
  
  // Frame rate update
  g_iFrameCount++;
  
  if (g_timer.GetTime() > 0.5)
  {  
    char title[100];
    sprintf(title, "Flo Fluid Simulator: %f FPS", 
			g_iFrameCount / g_timer.GetTime());
    glutSetWindowTitle(title);

    g_iFrameCount = 0;
    g_timer.Reset();
  }

}



//----------------------------------------------------------------------------
// Function     	  : Special
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Special(int key, int x, int y)
 * @brief GLUT Special key callback. 
 */ 
void Special(int key, int x, int y)
{
  paramlist->Special(key, x, y);
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
// Function     	  : Idle
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Idle()
 * @brief GLUT idle callback.
 */ 
void Idle()
{
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
// Function     	  : Reshape
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Reshape(int w, int h)
 * @brief GLUT reshape callback.
 */ 
void Reshape(int w, int h)
{
  if (h == 0) h = 1;
  
  glViewport(0, 0, w, h);
  
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

//----------------------------------------------------------------------------
// Function     	  : Keyboard
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Keyboard(unsigned char key, int x, int y)
 * @brief GLUT keyboard callback.
 */ 
void Keyboard(unsigned char key, int x, int y)
{
  switch(key) 
  {
  case 'q':
  case 'Q':
  case 27:
    Shutdown();
    exit(0);
  	break;
  case '.':
    g_bSingleStep = true;
    break;
  case 'p':
    g_bPause = !g_bPause;
    break;
  case 'a':
    g_bArbitraryBC = !g_bArbitraryBC;
    g_pFlo->EnableArbitraryBC(g_bArbitraryBC);
    break;
  case 'c':
    g_bClearPressure = !g_bClearPressure;
    g_pFlo->EnablePressureClear(g_bClearPressure);
    break;
  case 'r':
    g_pFlo->Reset();
    break;
  case 'R':
    g_pFlo->Reset(true);
    break;
  case 'v':
    g_bComputeVorticity = !g_bComputeVorticity;
    g_pFlo->EnableVCForce(g_bComputeVorticity);
    break;
  case '~':
  case '`':
    g_bDisplaySliders = !g_bDisplaySliders;
    break;
  case 't':
    g_displayMode = static_cast<Flo::DisplayMode>(((g_displayMode + 1) 
												  % Flo::DISPLAY_COUNT));
    break;
  case 'T':
	g_bDisplayFluid = !g_bDisplayFluid;
	break;
  case 'b':
    g_bBilerp = !g_bBilerp;
    break;
  case 'm':
    g_bMakeTex = !g_bMakeTex;
    break;
  case 'L':
      g_perfTimer.Stop();
      g_perfTimer.Reset();
      g_bTiming = true;
      break;
  default:
    break;
  }
}

//----------------------------------------------------------------------------
// Function     	  : Motion
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Motion(int x, int y)
 * @brief GLUT mouse motion callback.
 */ 
void Motion(int x, int y)
{
  const int maxRadius = 1; // in pixels

  if (g_bDisplaySliders && g_bMouseDown) 
  {
    // call parameter list motion function
    paramlist->Motion(x, y);
    glutPostRedisplay();
    return;
  }
  
  y = glutGet(GLUT_WINDOW_HEIGHT) - y - 1;
  if(g_bMouseDown)
  {
    // Force is applied by dragging with the left mouse button
    // held down.
    
    float dx, dy;
    dx = x - g_iMouseX;
    dy = y - g_iMouseY;
    
    // clamp to some range
    float strength[3] = 
    {
      MIN( MAX(dx, -maxRadius * g_rGridScale), maxRadius * g_rGridScale),
      MIN( MAX(dy, -maxRadius * g_rGridScale), maxRadius * g_rGridScale),
      0
    };

    float pos[2] = 
    {
      x / (float) glutGet(GLUT_WINDOW_WIDTH),
      y / (float) glutGet(GLUT_WINDOW_HEIGHT)
    };
    
    if (g_iMouseButton == GLUT_LEFT_BUTTON)
    {
      if (g_bArbitraryBC && g_bCtrlDown)
        g_pFlo->DrawBoundary(pos, 1.0 / (float)g_iFloWidth);
      else
        g_pFlo->DrawImpulse(strength, pos, g_rBrushRadius, true);
    }
    else if (g_iMouseButton == GLUT_RIGHT_BUTTON)
      g_pFlo->DrawImpulse(strength, pos, g_rBrushRadius, false);

        
    g_iMouseX = x;
    g_iMouseY = y;
  }

}

//----------------------------------------------------------------------------
// Function     	  : Mouse
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Mouse(int x, int y)
 * @brief GLUT mouse button callback.
 */ 
void Mouse(int button, int state, int x, int y)
{
  // We get the y coordinate in normal window system coordinates,
  // which is upside down from the point of view of OpenGL,
  // so subtract it from the height of the window.  
  if( state == GLUT_DOWN )
  {
    g_iMouseButton = button;
    // If the left mouse button has just been pressed down,
    // set our global variables that keep track of the coordinates.
    g_iMouseX = x;
    g_iMouseY = glutGet(GLUT_WINDOW_HEIGHT) - y - 1;
    g_bMouseDown = true;
  }
  else
  {
    g_iMouseButton = 0;
    g_bMouseDown = false;
  }

  if (g_bDisplaySliders && g_bMouseDown) 
  {
    // call list mouse function
    paramlist->Mouse(x, y);
  }

  g_bCtrlDown = (state == GLUT_DOWN) && (glutGetModifiers() == GLUT_ACTIVE_CTRL);
  if (g_bCtrlDown)
    glutSetCursor(GLUT_CURSOR_CROSSHAIR);
  else
    glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
}

//----------------------------------------------------------------------------
// Function     	  : Initialize
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Initialize()
 * @brief Setup the simulation.
 */ 
void Initialize()
{
  // Initialize the UI
  InitParameters();
  
  // First initialize extensions.
  int err = glewInit();

  if (GLEW_OK != err)
  {
    /* problem: glewInit failed, something is seriously wrong */
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    exit(-1);
  }

  // Create the Cg context
  cgSetErrorCallback(cgErrorCallback);

  // Create cgContext.
  g_cgContext = cgCreateContext();
  
  // Create and initialize the Flo simulator object
  g_pFlo = new Flo(g_iFloWidth, g_iFloHeight);
  g_pFlo->Initialize(g_cgContext);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
}

//----------------------------------------------------------------------------
// Function     	: Shutdown
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Shutdown()
 * @brief Shut down the simulation.
 */ 
void Shutdown()
{
  g_pFlo->Shutdown();
 
  delete g_pFlo;
  g_pFlo = 0;
}

//----------------------------------------------------------------------------
// Function     	: main
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn main()
 * @brief The main method.
 */ 
int main(int argc, char **argv)
{
  // optionally set the resolution at startup
  if (argc > 1)
  {
    g_iFloWidth = g_iFloHeight = atoi(argv[1]);
  }

  // Window size.
  const int width = 512, height = 512;

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA);
  glutInitWindowPosition(50, 50);
  glutInitWindowSize(width, height);
  glutCreateWindow("Mark's Fluid Simulator");

  glutIdleFunc(Idle);
  glutDisplayFunc(Display);
  glutKeyboardFunc(Keyboard);
  glutMotionFunc(Motion);
  glutMouseFunc(Mouse);
  glutReshapeFunc(Reshape);
  glutSpecialFunc(Special);

  Reshape(width, height); 

  // Init the simulation, UI, and graphics state.
  Initialize();

  // Print some instructions.
  system("cls");
  printf("\nFlo: Mark's GPU Fluid Simulator\n\n");
  printf("Keys:\n");
  printf("[`]: (tilde key) Display parameter sliders\n"
         "[t]: Cycle display mode (Velocity, Scalar, Pressure).\n"
		 "[T]: Toggle display (turn it off for more accurate \n"
		 "     simulation timing.\n"
         "[r]: Reset fluid simulation\n"
         "[b]: Toggle bilinear interpolation in fragment program.\n"
		 "     (High Quality, Lower Performance)\n"
		 "[m]: Toggle interpolated texture generation.\n"
		 "     (Lower Quality, Higher Performance)\n"
		 "[v]: Toggle vorticity confinement.\n"
         "[a]: Toggle arbitrary boundaries.\n"
         "[c]: Toggle clearing of pressure each iteration. \n"
         "     (With this off, solution requires fewer iterations, but\n"
         "     has more oscillatory behavior.)\n"
         "[L]: Time 100 iterations and print out the result.\n"
         "[q]/[ESC]: Quit\n"
         "\n"
         "Mouse:\n"
         "Left Button: Inject Ink and perturb the fluid.\n"
         "Right Button: perturb the fluid w/o injecting Ink.\n"
         "CTRL + Left Button: Draw boundaries.\n"
         "\n\n\n"
        );

  // start the framerate timer.
  g_timer.Start();

  // Go!
  glutMainLoop();

  return 0;
}
