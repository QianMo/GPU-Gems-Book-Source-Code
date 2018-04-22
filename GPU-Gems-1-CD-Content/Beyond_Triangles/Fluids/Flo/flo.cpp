//----------------------------------------------------------------------------
// File : flo.cpp
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
 * @file flo.cpp
 * 
 * Flo fluid simulation class implementation.
 */

#include <gl/glew.h>
#include "flo.h"
#include <gl/glut.h>
#include <assert.h>
#include "RenderTexture.h"
#include "tga.hpp"

//----------------------------------------------------------------------------
// Function     	: Flo::Flo
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Flo()
 * @brief Default Constructor
 */ 
Flo::Flo(int width, int height) 
: _iWidth(width),
  _iHeight(height),
  _dx(1),
  _rTimestep(1),
  _iNumPoissonSteps(50),
  _rInkLongevity(0.9995f),
  _rViscosity(0),
  _bArbitraryBC(false),
  _bClearPressureEachStep(true),
  _bComputeVorticity(true),
  _bApplyVCForce(true),
  _rVorticityConfinementScale(0.035f),
  _bImpulseToProcess(false),
  _bInkToAdd(false),
  _pOffscreenBuffer(0),
  _iVelocityOffsetTexture(0),
  _iPressureOffsetTexture(0),
  _iDisplayTexture(0),
  _iBCTexture(0),
  _iBCDisplayTexture(0),
  _iBCDetailTexture(0),
  _zeros(0)
{
  _rInkColor[0] = _rInkColor[1] = _rInkColor[2] = 1;
  memset(_iTextures, 0, TEXTURE_COUNT * sizeof(GLuint));

  _zeros = new float[_iWidth * _iHeight * 4];
  memset(_zeros, 0, _iWidth * _iHeight * 4 * sizeof(float));
}

//----------------------------------------------------------------------------
// Function     	: Flo::~Flo
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::~Flo()
 * @brief Destructor
 */ 
Flo::~Flo()
{
  delete [] _zeros;
}

//----------------------------------------------------------------------------
// Function     	: Flo::Initialize
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Initialize()
 * @brief Initializes the Flo simulator.
 * 
 * Here we initialize the simulation.
 */ 
void Flo::Initialize(CGcontext context)
{
  // create the offscreen buffer -- this has to be a float buffer because
  // we need float precision for the simulation.
  _pOffscreenBuffer = new RenderTexture(_iWidth, _iHeight, false);
  
  // Offscreen buffer does not update a texture object, it calls 
  // wglShareLists, it has no depth/stencil, mipmaps, or aniso filtering.  
  // 32 bits per channel.
  _pOffscreenBuffer->Initialize(true, false, false, false, false, 
                                32, 32, 32, 32);

  // Set the constant state for the pbuffer.
  _pOffscreenBuffer->BeginCapture();
  {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, 1, 0, 1);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0, 0, 0, 0);
  }
  _pOffscreenBuffer->EndCapture();
  
  // create texture objects -- there are four: velocity, pressure, divergence,
  // and density.  All are float textures.
  int iTex = 0;
  glGenTextures(TEXTURE_COUNT, _iTextures);
  for (iTex = 0; iTex < TEXTURE_COUNT; ++iTex)
  {
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iTextures[iTex]);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV, 
                 _iWidth, _iHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                    GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                    GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                    GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }

  glGenTextures(1, &_iBCTexture);
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iBCTexture);
  glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8,
               _iWidth, _iHeight, 0, GL_RGBA, GL_FLOAT, NULL);
  
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                  GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                  GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                  GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
                    GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  _ClearTexture(_iBCTexture);

  glGenTextures(1, &_iBCDisplayTexture);
  glBindTexture(GL_TEXTURE_2D, _iBCDisplayTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_FLOAT, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  _ClearTexture(_iBCDisplayTexture, GL_TEXTURE_2D);
  
  // create the display texture
  glGenTextures(1, &_iDisplayTexture);
  glBindTexture(GL_TEXTURE_2D, _iDisplayTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _iWidth, _iHeight, 
			   0, GL_RGBA, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


  unsigned char *bcColor = NULL;
  int bcWidth, bcHeight, bcChannels;
  LoadTGA("bc.tga", bcColor, bcWidth, bcHeight, bcChannels);

  glGenTextures(1, &_iBCDetailTexture);
  glBindTexture(GL_TEXTURE_2D, _iBCDetailTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, bcChannels == 3 ? GL_RGB8 : GL_RGBA8, 
               bcWidth, bcHeight, 0, bcChannels == 3 ? GL_RGB : GL_RGBA, 
               GL_UNSIGNED_BYTE, bcColor);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);


  _CreateOffsetTextures();

  // Initialize the slabops...
  _InitializeSlabOps(context);

  // clear the textures
  Reset();
}

//----------------------------------------------------------------------------
// Function     	: Flo::_CreateOffsetTexture()
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Initialize()
 * @brief Generates the boundary offset lookup table
 * 
 * See the comments in the file flo.cg.
 */ 
void Flo::_CreateOffsetTextures()
{
  float velocityData[136] = 
  {
     // This cell is a fluid cell
     1,  0,  1,  0,   // Free (no neighboring boundaries)
     0,  0, -1,  1,   // East (a boundary to the east)
     1,  0,  1,  0,   // Unused
     1,  0,  0,  0,   // North
     0,  0,  0,  0,   // Northeast
     1,  0,  1,  0,   // South
     0,  0,  1,  0,   // Southeast
     1,  0,  1,  0,   // West
     1,  0,  1,  0,   // Unused
     0,  0,  0,  0,   // surrounded (3 neighbors)
     1,  0,  0,  0,   // Northwest
     0,  0,  0,  0,   // surrounded (3 neighbors)
     1,  0,  1,  0,   // Southwest 
     0,  0,  0,  0,   // surrounded (3 neighbors)
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // surrounded (3 neighbors)
     0,  0,  0,  0,   // surrounded (4 neighbors)
     // This cell is a boundary cell (the inverse of above!)
     1,  0,  1,  0,   // No neighboring boundaries (Error)
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // Unused
    -1, -1, -1, -1,   // Southwest 
     0,  0,  0,  0,   // Unused
    -1,  1,  0,  0,   // Northwest
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // Unused
     0,  0, -1, -1,   // West
     0,  0, -1,  1,   // Southeast
    -1, -1,  0,  0,   // South
     0,  0,  0,  0,   // Northeast
    -1,  1,  0,  0,   // North
     0,  0,  0,  0,   // Unused
     0,  0, -1,  1,   // East (a boundary to the east)
     0,  0,  0,  0    // Unused
  };
  
  glGenTextures(1, &_iVelocityOffsetTexture);
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iVelocityOffsetTexture);

  glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV, 34, 1,
			   0, GL_RGBA, GL_FLOAT, velocityData);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  float pressureData[136] = 
  {
    // This cell is a fluid cell
     0,  0,  0,  0,   // Free (no neighboring boundaries)
     0,  0,  0,  0,   // East (a boundary to the east)
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // North
     0,  0,  0,  0,   // Northeast
     0,  0,  0,  0,   // South
     0,  0,  0,  0,   // Southeast
     0,  0,  0,  0,   // West
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // Landlocked (3 neighbors)
     0,  0,  0,  0,   // Northwest
     0,  0,  0,  0,   // Landlocked (3 neighbors)
     0,  0,  0,  0,   // Southwest 
     0,  0,  0,  0,   // Landlocked (3 neighbors)
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // Landlocked (3 neighbors)
     0,  0,  0,  0,   // Landlocked (4 neighbors)
     // This cell is a boundary cell (the inverse of above!)
     0,  0,  0,  0,   // no neighboring boundaries
     0,  0,  0,  0,   // unused
     0,  0,  0,  0,   // unused
     0,  0,  0,  0,   // unused
    -1,  0,  0, -1,   // Southwest 
     0,  0,  0,  0,   // unused
    -1,  0,  0,  1,   // Northwest
     0,  0,  0,  0,   // Unused
     0,  0,  0,  0,   // Unused
    -1,  0, -1,  0,   // West
     0, -1,  1,  0,   // Southeast
     0, -1,  0, -1,   // South
     0,  1,  1,  0,   // Northeast
     0,  1,  0,  1,   // North
     0,  0,  0,  0,   // Unused
     1,  0,  1,  0,   // East (a boundary to the east)
     0,  0,  0,  0   // Unused
  };
  
  glGenTextures(1, &_iPressureOffsetTexture);
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iPressureOffsetTexture);

  glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV, 34, 1,
			   0, GL_RGBA, GL_FLOAT, pressureData);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

//----------------------------------------------------------------------------
// Function     	: Flo::_InitializeSlabOps
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::InitializeSlabOps()
 * @brief Initializes the SlabOps used in the fluid simulation
 * 
 * Here we initialize the slab operations, passing them parameters such
 * as input textures, and fragment programs.
 */ 
void Flo::_InitializeSlabOps(CGcontext context)
{
  // Compute the minimum and maximum vertex and texture coordinates for 
  // rendering the slab interiors (sans boundaries).
  float xMin  = 1.0f / (float)_iWidth;         // one pixel from left
  float xMax  = 1.0f - 1.0f / (float)_iWidth;  // one pixel from right
  float yMin  = 1.0f / (float)_iHeight;        // on pixel from bottom
  float yMax  = 1.0f - 1.0f / (float)_iHeight; // one pixel from top
  
  float sMin  = 1.0f;                          // the same, this time in 
  float sMax  = (float)_iWidth - 1.0f;         // texture coordinates
  float tMin  = 1.0f;
  float tMax  = (float)_iHeight - 1.0f;
  
  // Add Impulse.  This renders a Gaussian "splat" into the specified texture,
  // with the specified position, radius, and color.
  _addImpulse.InitializeFP(context, "../geep/programs/splat.cg");
  _addImpulse.SetTextureParameter("base",  _iTextures[TEXTURE_VELOCITY]);
  _addImpulse.SetFragmentParameter2f("windowDims", _iWidth, _iHeight);
  _addImpulse.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _addImpulse.SetSlabRect(xMin, yMin, xMax, yMax);
  _addImpulse.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], 
                               _iWidth, _iHeight);

  // Advection:  This advects a field by the moving velocity field.  This is 
  // used to advect both velocity and scalar values, such as mass.  The result
  // of applying this to the velocity field is a moving but divergent velocity
  // field.  The next few steps correct that divergence to give a divergence-
  // free velocity field.
  _advect.InitializeFP(context, "flo.cg", "advect");
  _advect.SetTextureParameter("u", _iTextures[TEXTURE_VELOCITY]);
  _advect.SetTextureParameter("x",    _iTextures[TEXTURE_VELOCITY]);
  _advect.SetFragmentParameter1f("rdx", 1.0f / _dx);
  _advect.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _advect.SetSlabRect(xMin, yMin, xMax, yMax);
  _advect.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], _iWidth, _iHeight);

  // Divergence of velocity: This computes how divergent the velocity field is
  // (how much in/out flow there is at every point).  Used as input to the 
  // Poisson solver, below.
  _divergence.InitializeFP(context, "flo.cg", "divergence");
  _divergence.SetTextureParameter("w", _iTextures[TEXTURE_VELOCITY]);
  _divergence.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _divergence.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _divergence.SetSlabRect(xMin, yMin, xMax, yMax);
  _divergence.SetOutputTexture(_iTextures[TEXTURE_DIVERGENCE], 
                               _iWidth, _iHeight);

  // Poisson-pressure solver: By running this Jacobi Relaxation solver for 
  // multiple iterations, this solves for the pressure disturbance in the 
  // fluid given the divergence of the velocity.
  _poissonSolver.InitializeFP(context, "flo.cg", "jacobi");
  _poissonSolver.SetTextureParameter("x", _iTextures[TEXTURE_PRESSURE]);
  _poissonSolver.SetTextureParameter("b", _iTextures[TEXTURE_DIVERGENCE]);
  _poissonSolver.SetFragmentParameter1f("alpha", -_dx * _dx);
  _poissonSolver.SetFragmentParameter1f("rBeta", 0.25f);
  _poissonSolver.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _poissonSolver.SetSlabRect(xMin, yMin, xMax, yMax);
  _poissonSolver.SetOutputTexture(_iTextures[TEXTURE_PRESSURE], 
                                  _iWidth, _iHeight);

  // Subtract Gradient.  After solving for the pressure disturbance, this 
  // subtracts the pressure gradient from the divergent velocity field to 
  // give a divergence-free field.
  _subtractGradient.InitializeFP(context, "flo.cg", "gradient");
  _subtractGradient.SetTextureParameter("p", _iTextures[TEXTURE_PRESSURE]);
  _subtractGradient.SetTextureParameter("w", _iTextures[TEXTURE_VELOCITY]);
  _subtractGradient.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _subtractGradient.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _subtractGradient.SetSlabRect(xMin, yMin, xMax, yMax);
  _subtractGradient.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], 
                                     _iWidth, _iHeight);

  // vorticity computation.
  _vorticity.InitializeFP(context, "flo.cg", "vorticity");
  _vorticity.SetTextureParameter("u", _iTextures[TEXTURE_VELOCITY]);
  _vorticity.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _vorticity.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _vorticity.SetSlabRect(xMin, yMin, xMax, yMax);
  _vorticity.SetOutputTexture(_iTextures[TEXTURE_VORTICITY], 
                              _iWidth, _iHeight);

  // vorticity confinement force computation.
  _vorticityForce.InitializeFP(context, "flo.cg", "vortForce");
  _vorticityForce.SetTextureParameter("vort", _iTextures[TEXTURE_VORTICITY]);
  _vorticityForce.SetTextureParameter("u", _iTextures[TEXTURE_VELOCITY]);
  _vorticityForce.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _vorticityForce.SetFragmentParameter2f("dxscale", 
                                         _rVorticityConfinementScale * _dx, 
                                         _rVorticityConfinementScale * _dx);
  _vorticityForce.SetTexCoordRect(0, sMin, tMin, sMax, tMax);
  _vorticityForce.SetSlabRect(xMin, yMin, xMax, yMax);
  _vorticityForce.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], 
                                   _iWidth, _iHeight);
  
  // This applies pure neumann boundary conditions (see floPoissonBC.cg) to 
  // the pressure field once per iteration of the poisson-pressure jacobi 
  // solver.  Also no-slip BCs to velocity once per time step.
  _boundaries.InitializeFP(context, "flo.cg", "boundary");
  _boundaries.SetTexCoordRect(0, 0, _iWidth, _iHeight);
  _boundaries.SetTexResolution(_iWidth, _iHeight);


  _updateOffsets.InitializeFP(context, "flo.cg", 
                                      "updateOffsets");
  _updateOffsets.SetTextureParameter("b", _iBCTexture);
  _updateOffsets.SetTextureParameter("offsetTable", _iVelocityOffsetTexture);
  _updateOffsets.SetTexCoordRect(0, 0, 0, _iWidth, _iHeight);
  _updateOffsets.SetSlabRect(xMin, yMin, xMax, yMax);
  _updateOffsets.SetOutputTexture(_iTextures[TEXTURE_VELOCITY_OFFSETS],
                                  _iWidth, _iHeight);

  // setup the arbitrary boundary operations.
  _arbitraryVelocityBC.InitializeFP(context, "flo.cg", 
                                    "arbitraryVelocityBoundary");
  _arbitraryVelocityBC.SetTextureParameter("u", _iTextures[TEXTURE_VELOCITY]);
  _arbitraryVelocityBC.SetTextureParameter("offsets", 
                                           _iTextures[TEXTURE_VELOCITY_OFFSETS]);
  _arbitraryVelocityBC.SetTexCoordRect(0, 0, 0, _iWidth, _iHeight);
  _arbitraryVelocityBC.SetSlabRect(xMin, yMin, xMax, yMax);
  _arbitraryVelocityBC.SetOutputTexture(_iTextures[TEXTURE_VELOCITY],
                                        _iWidth, _iHeight);
  

  _arbitraryPressureBC.InitializeFP(context, "flo.cg", 
                                    "arbitraryPressureBoundary");
  _arbitraryPressureBC.SetTextureParameter("p", _iTextures[TEXTURE_PRESSURE]);
  _arbitraryPressureBC.SetTextureParameter("offsets", _iTextures[TEXTURE_PRESSURE_OFFSETS]);
  _arbitraryPressureBC.SetTexCoordRect(0, 0, 0, _iWidth, _iHeight);
  _arbitraryPressureBC.SetSlabRect(xMin, yMin, xMax, yMax);
  _arbitraryPressureBC.SetOutputTexture(_iTextures[TEXTURE_PRESSURE],
                                        _iWidth, _iHeight);
  

  // Display: These ops are used to display scalar and vector fields with
  // and without bilinear interpolation.
  _displayVector.InitializeFP(context, "flo.cg", "displayVector");
  _displayVector.SetTexCoordRect(0, 0, _iWidth, _iHeight);
  _displayVectorBilerp.InitializeFP(context, "flo.cg", "displayVectorBilerp");
  _displayVectorBilerp.SetTexCoordRect(0, 0, _iWidth, _iHeight);

  _displayScalar.InitializeFP(context, "flo.cg", "displayScalar");
  _displayScalar.SetTexCoordRect(0, 0, _iWidth, _iHeight);
  _displayScalarBilerp.InitializeFP(context, "flo.cg", "displayScalarBilerp");
  _displayScalarBilerp.SetTexCoordRect(0, 0, _iWidth, _iHeight);

  // to ensure it gets set in the advection operation
  SetTimeStep(_rTimestep); 
}

//----------------------------------------------------------------------------
// Function     	: Flo::_ClearTexture
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::_ClearTexture()
 * @brief Clears the texture specified by @a id to all zeros.
 */ 
void Flo::_ClearTexture(unsigned int id, GLenum target /* = GL_TEXTURE_RECTANGLE_NV*/)
{
  glBindTexture(target, id);
  glTexSubImage2D(target, 0, 0, 0, _iWidth, _iHeight, GL_RGBA, GL_FLOAT, _zeros);
}


//----------------------------------------------------------------------------
// Function     	: Flo::Shutdown
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Shutdown()
 * @brief Shuts down the Flo simulator.
 */ 
void Flo::Shutdown()
{
  glDeleteTextures(TEXTURE_COUNT, _iTextures);
  glDeleteTextures(1, &_iDisplayTexture);

  delete _pOffscreenBuffer;
  _pOffscreenBuffer = 0;

  _advect.ShutdownFP();
  _divergence.ShutdownFP();
  _poissonSolver.ShutdownFP();
  _subtractGradient.ShutdownFP();
  _vorticity.ShutdownFP();
  _vorticityForce.ShutdownFP();
  _boundaries.ShutdownFP();
  _arbitraryPressureBC.ShutdownFP();
  _arbitraryVelocityBC.ShutdownFP();
  
  _displayVector.ShutdownFP();
  _displayVectorBilerp.ShutdownFP();
  _displayScalar.ShutdownFP();
  _displayScalarBilerp.ShutdownFP();
}

//----------------------------------------------------------------------------
// Function     	: Flo::Update
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Update(float timestep)
 * @brief Runs a fluid simulation step. 
 * 
 * Update solves the incompressible Navier-Stokes equations for a single 
 * time step.  It consists of four main steps:
 *
 * 1. Add Impulse
 * 2. Advect
 * 3. Apply Vorticity Confinement
 * 4. Diffuse (if viscosity > 0)
 * 5. Project (computes divergence-free velocity from divergent field).
 *
 * 1. Advect: pulls the velocity and Ink fields forward along the velocity
 *            field.  This results in a divergent field, which must be 
 *            corrected in step 4.
 * 2. Add Impulse: simply add an impulse to the velocity (and optionally,Ink)
 *    field where the user has clicked and dragged the mouse.
 * 3. Apply Vorticity Confinement: computes the amount of vorticity in the 
 *            flow, and applies a small impulse to account for vorticity lost 
 *            to numerical dissipation.
 * 4. Diffuse: viscosity causes the diffusion of velocity.  This step solves
 *             a poisson problem: (I - nu*dt*Laplacian)u' = u.
 * 5. Project: In this step, we correct the divergence of the velocity field
 *             as follows.
 *        a.  Compute the divergence of the velocity field, div(u)
 *        b.  Solve the Poisson equation, Laplacian(p) = div(u), for p using 
 *            Jacobi iteration.
 *        c.  Now that we have p, compute the divergence free velocity:
 *            u = gradient(p)
 */ 
void Flo::Update()
{
  int i;

  // Activate rendering to the offscreen buffer
  _pOffscreenBuffer->BeginCapture();

  //---------------
  // 1.  Advect
  //---------------
  // Advect velocity (velocity advects itself, resulting in a divergent
  // velocity field.  Later, correct this divergence).

  // Set the no-slip velocity...
  // This sets the scale to -1, so that v[0, j] = -v[1, j], so that at 
  // the boundary between them, the avg. velocity is zero. 
  _boundaries.SetTextureParameter("x", _iTextures[TEXTURE_VELOCITY]);
  _boundaries.SetFragmentParameter1f("scale", -1); 
  _boundaries.Compute();

  // Set the no-slip velocity on arbitrary interior boundaries if they are enabled
  if (_bArbitraryBC)
    _arbitraryVelocityBC.Compute();
  
  // Advect velocity.
  _advect.SetFragmentParameter1f("dissipation", 1);
  _advect.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], _iWidth, _iHeight);
  _advect.SetTextureParameter("x", _iTextures[TEXTURE_VELOCITY]);
  _advect.Compute();

  // Set ink boundaries to zero
  _boundaries.SetTextureParameter("x", _iTextures[TEXTURE_DENSITY]);
  _boundaries.SetFragmentParameter1f("scale", 0); 
  _boundaries.Compute();

  // Advect "ink", a passive scalar carried by the flow.
  _advect.SetFragmentParameter1f("dissipation", _rInkLongevity);
  _advect.SetOutputTexture(_iTextures[TEXTURE_DENSITY], _iWidth, _iHeight);
  _advect.SetTextureParameter("x", _iTextures[TEXTURE_DENSITY]);
  _advect.Compute();

  //---------------
  // 2. Add Impulse
  //---------------
  if (_bImpulseToProcess)
  {
    _addImpulse.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], 
                                 _iWidth, _iHeight);
    _addImpulse.SetTextureParameter("base", _iTextures[TEXTURE_VELOCITY]);   
    _addImpulse.Compute();  

    _bImpulseToProcess = false;
    
    if (_bInkToAdd)
    {
      _addImpulse.SetOutputTexture(_iTextures[TEXTURE_DENSITY], 
                                   _iWidth, _iHeight);
      _addImpulse.SetTextureParameter("base", _iTextures[TEXTURE_DENSITY]);
      _addImpulse.SetFragmentParameter3fv("color", _rInkColor);
      _addImpulse.Compute();      
      _bInkToAdd = false;
    }
  }

  //---------------
  // 3. Apply Vorticity Confinement
  //---------------
  if (_bComputeVorticity)
  {
    _vorticity.Compute();
  }

  if (_bApplyVCForce)
  {
    _boundaries.SetTextureParameter("x", _iTextures[TEXTURE_VELOCITY]);
    _boundaries.SetFragmentParameter1f("scale", -1); 
    _boundaries.Compute();
    _vorticityForce.Compute();
  }

  //--------------- 
  // 3. Diffuse (if viscosity is > 0)
  //---------------
  // If this is a viscous fluid, solve the poisson problem for the viscous
  // diffusion
  if (_rViscosity > 0)
  {
    float centerFactor = _dx * _dx / (_rViscosity * _rTimestep);
    float stencilFactor = 1.0f / (4.0f + centerFactor);
    _poissonSolver.SetTextureParameter("x", _iTextures[TEXTURE_VELOCITY]);
    _poissonSolver.SetTextureParameter("b", _iTextures[TEXTURE_VELOCITY]);
    _poissonSolver.SetFragmentParameter1f("alpha", centerFactor);
    _poissonSolver.SetFragmentParameter1f("rBeta", stencilFactor);
    _poissonSolver.SetTexCoordRect(0, 0, 0, _iWidth, _iHeight);
    _poissonSolver.SetOutputTexture(_iTextures[TEXTURE_VELOCITY], 
                                    _iWidth, _iHeight);
    for (i = 0; i < _iNumPoissonSteps; ++i)
      _poissonSolver.Compute();
  }
 
  //---------------
  // 4. Project divergent velocity into divergence-free field
  //---------------
  
  //---------------
  // 4a. compute divergence
  //---------------
  // Compute the divergence of the velocity field
  _divergence.Compute();

  //---------------
  // 4b. Compute pressure disturbance
  //---------------
  
  // Solve for the pressure disturbance caused by the divergence, by solving
  // the poisson problem Laplacian(p) = div(u)
  _poissonSolver.SetTextureParameter("x", _iTextures[TEXTURE_PRESSURE]);
  _poissonSolver.SetTextureParameter("b", _iTextures[TEXTURE_DIVERGENCE]);
  _poissonSolver.SetFragmentParameter1f("alpha", -_dx * _dx);
  _poissonSolver.SetFragmentParameter1f("rBeta", 0.25f);
  _poissonSolver.SetOutputTexture(_iTextures[TEXTURE_PRESSURE], 
                                  _iWidth, _iHeight);
  _boundaries.SetTextureParameter("x", _iTextures[TEXTURE_PRESSURE]);
  _boundaries.SetFragmentParameter1f("scale", 1);

  // Clear the pressure texture, to initialize the pressure disturbance to 
  // zero before iterating.  If this is disabled, the solution converges 
  // faster, but tends to have oscillations.
  if (_bClearPressureEachStep)
    _ClearTexture(_iTextures[TEXTURE_PRESSURE]);

  for (i = 0; i < _iNumPoissonSteps; ++i)
  {
    _boundaries.Compute();    // Apply pure neumann boundary conditions

    // Apply pure neumann boundary conditions to arbitrary 
    // interior boundaries if they are enabled.
    if (_bArbitraryBC)
      _arbitraryPressureBC.Compute();

    _poissonSolver.Compute(); // perform one Jacobi iteration
  }
    
  // Set the no-slip velocity...
  // We do this here again because the advection step modified the velocity,
  // and we want to avoid instabilities at the boundaries.

  // This sets the scale to -1, so that v[0, j] = -v[1, j], so that at 
  // the boundary between them, the avg. velocity is zero.
  _boundaries.SetTextureParameter("x", _iTextures[TEXTURE_VELOCITY]);
  _boundaries.SetFragmentParameter1f("scale", -1); 
  _boundaries.Compute();
  
  if (_bArbitraryBC)
    _arbitraryVelocityBC.Compute();

  //---------------
  // 4c. Subtract gradient(p) from u
  //---------------
  // This gives us our final, divergence free velocity field.
  _subtractGradient.Compute();   
    
  // End rendering to the offscreen buffer.
  _pOffscreenBuffer->EndCapture();
}


//----------------------------------------------------------------------------
// Function     	: Flo::Reset
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Reset()
 * @brief Resets the simulation.
 */ 
void Flo::Reset(bool resetBC /* = false */)
{
  // create some zeros to clear textures with.
  float *zeros = new float[_iWidth * _iHeight * 4];
  memset(zeros, 0, _iWidth * _iHeight * 4 * sizeof(float));

  int iTex = 0;
  // clear all textures to zero.
  for (iTex = 0; iTex < TEXTURE_COUNT; ++iTex)
  {
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iTextures[iTex]);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, _iWidth, _iHeight, 
                    GL_RGBA, GL_FLOAT, zeros);
  }

  if (resetBC)
  {
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iBCTexture);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, _iWidth, _iHeight, 
                    GL_RGBA, GL_FLOAT, zeros);
    glBindTexture(GL_TEXTURE_2D, _iBCDisplayTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 64, 
                    GL_RGBA, GL_FLOAT, zeros);
  }

  _pOffscreenBuffer->BeginCapture();
  _updateOffsets.SetTextureParameter("offsetTable", 
                                     _iVelocityOffsetTexture);
  _updateOffsets.SetOutputTexture(_iTextures[TEXTURE_VELOCITY_OFFSETS],
                                  _iWidth, _iHeight);
  _updateOffsets.Compute();
  
  _updateOffsets.SetTextureParameter("offsetTable", 
                                     _iPressureOffsetTexture);
  _updateOffsets.SetOutputTexture(_iTextures[TEXTURE_PRESSURE_OFFSETS],
                                  _iWidth, _iHeight);
  _updateOffsets.Compute();

  _pOffscreenBuffer->EndCapture();

  delete [] zeros;
}

//----------------------------------------------------------------------------
// Function     	: Flo::Display
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::Display()
 * @brief Displays the simulation on a screen quad.
 */ 
void Flo::Display(DisplayMode mode, 
                  bool bilerp /* = false */, 
                  bool makeTex /*= false*/,
                  bool displayBoundary /* = false */)
{
  FloDisplayOp *display;

  // If makeTex is true, then we render to a viewport the size of the 
  // simulation, rather than the whole window.  Then (see below), we copy the 
  // viewport to a texture. This is a regular (non-float) 2D texture, and we 
  // can enable bilinear filtering when rendering it.  This is faster than 
  // performing the bilinear filtering directly on the float texture in a 
  // fragment program, but the quality of the filtering is lower.

  // Also note that makeTex causes the scaled and biased fields, such as the 
  // velocity, pressure, and vorticity, to lose their scaling and biasing on 
  // display...
  if (makeTex)
  {
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, _iWidth, _iHeight);
  }

  static DisplayMode previous = DISPLAY_COUNT;
  if (mode != previous)
  {
    switch(mode) 
    {
    default:
    case DISPLAY_INK:
      display = (bilerp) ? &_displayVectorBilerp : &_displayVector;
      display->SetFragmentParameter4f("bias", 0, 0, 0, 0);
      display->SetFragmentParameter4f("scale", 1, 1, 1, 1);
      display->SetTextureParameter("texture", _iTextures[TEXTURE_DENSITY]);
      break;
    case DISPLAY_VELOCITY:
      display = (bilerp) ? &_displayVectorBilerp : &_displayVector;
      display->SetFragmentParameter4f("bias", 0.5f, 0.5f, 0.5f, 0.5f);
      display->SetFragmentParameter4f("scale", 0.5f, 0.5f, 0.5f, 0.5f);
      display->SetTextureParameter("texture", _iTextures[TEXTURE_VELOCITY]);
      break;
    case DISPLAY_PRESSURE:
      display = (bilerp) ? &_displayScalarBilerp : &_displayScalar;
      display->SetFragmentParameter4f("bias", 0, 0, 0, 0);
      display->SetFragmentParameter4f("scale", 2, -1, -2, 1);
      display->SetTextureParameter("texture", _iTextures[TEXTURE_PRESSURE]);
      break;
    case DISPLAY_VORTICITY:
      display = (bilerp) ? &_displayScalarBilerp : &_displayScalar;
      display->SetFragmentParameter4f("bias", 0, 0, 0, 0);
      display->SetFragmentParameter4f("scale", 1, 1, -1, 1);
      display->SetTextureParameter("texture", _iTextures[TEXTURE_VORTICITY]);
      break;
    }
  }
  display->Compute();

  // See the comments about makeTex at the top of this method.
  if (makeTex)
  {
    glBindTexture(GL_TEXTURE_2D, _iDisplayTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, _iWidth, _iHeight);

    glPopAttrib(); // GL_VIEWPORT_BIT

    glEnable(GL_TEXTURE_2D);

    // now display
    glBegin(GL_QUADS);
    {
      glTexCoord2f(0, 0); glVertex2f(0, 0);
      glTexCoord2f(1, 0); glVertex2f(1, 0);
      glTexCoord2f(1, 1); glVertex2f(1, 1);
      glTexCoord2f(0, 1); glVertex2f(0, 1);
    }
    glEnd();

    glDisable(GL_TEXTURE_2D);
  }

  if (displayBoundary)
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glBindTexture(GL_TEXTURE_2D, _iBCDisplayTexture);
    glEnable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glTexEnvf(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBindTexture(GL_TEXTURE_2D, _iBCDetailTexture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    {
      glColor4f(1, 1, 1, 1);
      glTexCoord2f(0, 0); glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 0, 0); glVertex2f(0, 0);
      glTexCoord2f(1, 0); glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 1, 0); glVertex2f(1, 0);
      glTexCoord2f(1, 1); glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 1, 1); glVertex2f(1, 1);
      glTexCoord2f(0, 1); glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 0, 1); glVertex2f(0, 1);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
  }
}

//----------------------------------------------------------------------------
// Function     	: Flo::SetImpulse
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::SetImpulse(const float strength[3], const float position[3], float radius)
 * @brief Sets the current interaction impulse.
 */ 
void Flo::DrawImpulse(const float strength[3], 
                      const float position[2], 
                      float radius,
                      bool bAddInk /* = true */)
{
  // don't add impulses if the mouse is outside the window!
  if (position[0] < 0 || position[0] > 1 || 
      position[1] < 0 || position[1] > 1)
    return;

  _addImpulse.SetFragmentParameter3fv("color", strength);
  _addImpulse.SetFragmentParameter2fv("position", position);
  _addImpulse.SetFragmentParameter1f("radius", radius);
  _bImpulseToProcess = true;
  _bInkToAdd        = bAddInk;
}

//----------------------------------------------------------------------------
// Function     	: Flo::DrawBoundary
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::SetImpulse(const float strength[3], const float position[3], float radius)
 * @brief Called when the user is drawing in arbitrary boundaries.
 */ 
void Flo::DrawBoundary(const float position[2], float radius)
{
  // don't add boundaries if the mouse is outside the window!
  if (position[0] < 0 || position[0] > 1 || 
      position[1] < 0 || position[1] > 1)
    return;

  float xMin = position[0] - radius;
  float xMax = xMin + 2 * radius;
  float yMin = position[1] - radius;
  float yMax = yMin + 2 * radius;
  
  int vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  glViewport(0, 0, _iWidth, _iHeight);
  glClear(GL_COLOR_BUFFER_BIT);
    
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iBCTexture);
  glEnable(GL_TEXTURE_RECTANGLE_NV);
  glBegin(GL_QUADS);
  {
    glColor4f(1, 1, 1, 1);
    glTexCoord2f(      0,        0); glVertex2f(0, 0);
    glTexCoord2f(_iWidth,        0); glVertex2f(1, 0);
    glTexCoord2f(_iWidth, _iHeight); glVertex2f(1, 1);
    glTexCoord2f(      0, _iHeight); glVertex2f(0, 1);
  }
  glEnd();
  glDisable(GL_TEXTURE_RECTANGLE_NV);
  
  glBlendFunc(GL_ONE, GL_ONE);
  glEnable(GL_BLEND);
  glBegin(GL_QUADS);
  {
    glColor4f(1, 1, 1, 1);
    glVertex3f(xMin, yMin, 0);
    glVertex3f(xMax, yMin, 0);
    glVertex3f(xMax, yMax, 0);
    glVertex3f(xMin, yMax, 0);
  }
  glEnd();
  glDisable(GL_BLEND);

  glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, _iWidth, _iHeight);

  glViewport(0, 0, 64, 64);
  glClear(GL_COLOR_BUFFER_BIT);
    
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iBCTexture);
  glEnable(GL_TEXTURE_RECTANGLE_NV);
  glBegin(GL_QUADS);
  {
    glColor4f(1, 1, 1, 1);
    glTexCoord2f(      0,        0); glVertex2f(0, 0);
    glTexCoord2f(_iWidth,        0); glVertex2f(1, 0);
    glTexCoord2f(_iWidth, _iHeight); glVertex2f(1, 1);
    glTexCoord2f(      0, _iHeight); glVertex2f(0, 1);
  }
  glEnd();
  glDisable(GL_TEXTURE_RECTANGLE_NV);
  
  glBindTexture(GL_TEXTURE_2D, _iBCDisplayTexture);
  glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, 64, 64);
  
  glViewport(vp[0], vp[1], vp[2], vp[3]);

  _pOffscreenBuffer->BeginCapture();
  _updateOffsets.SetTextureParameter("offsetTable", 
                                     _iVelocityOffsetTexture);
  _updateOffsets.SetOutputTexture(_iTextures[TEXTURE_VELOCITY_OFFSETS],
                                          _iWidth, _iHeight);
  _updateOffsets.Compute();
 
  _updateOffsets.SetTextureParameter("offsetTable", 
                                     _iPressureOffsetTexture);
  _updateOffsets.SetOutputTexture(_iTextures[TEXTURE_PRESSURE_OFFSETS],
                                          _iWidth, _iHeight);
  _updateOffsets.Compute();
  
  _pOffscreenBuffer->EndCapture();
}

//----------------------------------------------------------------------------
// Function     	: Flo::SetTimeStep
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::SetTimeStep(float t)
 * @brief Sets the simulation timestep
 */ 
void Flo::SetTimeStep(float t)
{
  _rTimestep = t;
  _advect.SetFragmentParameter1f("timestep", t);
  _vorticityForce.SetFragmentParameter1f("timestep", _rTimestep);
}

//----------------------------------------------------------------------------
// Function     	: Flo::SetGridScale
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn Flo::SetGridScale(float dx)
 * @brief Sets the size of each grid cell (they are assumed to be square).
 */ 
void Flo::SetGridScale(float dx)
{
  _dx = dx;
  _advect.SetFragmentParameter1f("rdx", 1.0f / _dx);
  _divergence.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _subtractGradient.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _vorticity.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _vorticityForce.SetFragmentParameter1f("halfrdx", 0.5f / _dx);
  _vorticityForce.SetFragmentParameter2f("dxscale", 
                                         _rVorticityConfinementScale * _dx, 
                                         _rVorticityConfinementScale * _dx);
}
