//----------------------------------------------------------------------------
// File : flo.h
//----------------------------------------------------------------------------
// Copyright 2003 Mark J. Harris and
// The University of North Carolina at Chapel Hill
//----------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software and its 
// documentation for any purpose is hereby granted without fee, provided that 
// the above copyright notice appear in all copies and that both that copyright 
// notice and this permission notice appear in supporting documentation. 
// Binaries may be compiled with this software without any royalties or 
// restrictions. 
//
// The author(s) and The University of North Carolina at Chapel Hill make no 
// representations about the suitability of this software for any purpose. 
// It is provided "as is" without express or implied warranty.
/**
 * @file flo.h
 * 
 * Flo fluid simulation class definition.
 */
#ifndef __FLO_H__
#define __FLO_H__

#include <geep/slabop.h>
#include <geep/slabopGL.h>
#include <geep/slabopCgGL.h>

using namespace geep;

// forward decl
class RenderTexture;

//----------------------------------------------------------------------------
/**
 * @class Flo
 * @brief A gpu implementation of fluid simulation.
 */
class Flo
{
public: // enumerations

  // Names for the textures used in the simulation
  enum Texture
  {
    TEXTURE_VELOCITY,
    TEXTURE_DENSITY,
    TEXTURE_DIVERGENCE,
    TEXTURE_PRESSURE,
    TEXTURE_VORTICITY,
    TEXTURE_VELOCITY_OFFSETS,
    TEXTURE_PRESSURE_OFFSETS,
    TEXTURE_COUNT
  };

  enum Texture8Bit
  {
    TEXTURE_BOUNDARIES,
    TEXTURE_8BIT_COUNT
  };

  // Names for the display modes
  enum DisplayMode
  {
    DISPLAY_VELOCITY,
    DISPLAY_INK,
    DISPLAY_PRESSURE,
    DISPLAY_VORTICITY,
    DISPLAY_COUNT
  };

public:
  Flo(int width, int height);
  ~Flo();
  
  void  Initialize(CGcontext context);
  void  Shutdown();

  void  Update();   // updates the simulation by one time step
  void  Reset(bool resetBC = false);    // resets the state of the simulation

  // Renders the simulation
  void  Display(DisplayMode mode, bool bilerp = false, bool makeTex = false,
                bool displayBoundary = false);

  // Used to draw an impulse into the fluid with the mouse
  void  DrawImpulse(const float strength[3], const float position[2], 
                    float radius, bool bAddMass = true);

  void  DrawBoundary(const float position[2], float radius);
 
  // setters / getters / enablers / queries
  void  SetTimeStep(float t);
  float GetTimeStep() const           { return _rTimestep; }

  void  SetGridScale(float dx);
  float GetGridScale() const          { return _dx; }

  void  SetNumPoissonSteps(int steps) { _iNumPoissonSteps = steps; }
  int   GetNumPoissonSteps() const    { return _iNumPoissonSteps;  }

  void  SetViscosity(float viscosity) { _rViscosity = viscosity;   }
  float GetViscosity() const          { return _rViscosity;        }

  void  SetInkLongevity(float longev) { _rInkLongevity = longev; }

  void  EnableArbitraryBC(bool bEnable) { _bArbitraryBC = bEnable; }
  bool  IsArbitraryBCEnabled() const    { return _bArbitraryBC;    }
  
  void  EnablePressureClear(bool bEnable) { _bClearPressureEachStep = bEnable; }
  bool  IsPressureClearEnabled() const    { return _bClearPressureEachStep; }

  void  EnableVorticityComputation(bool bEnable) { _bComputeVorticity = bEnable; }
  bool  IsVorticityComputationEnabled() const    { return _bComputeVorticity; }

  void  EnableVCForce(bool bEnable) { _bApplyVCForce = bEnable; }
  bool  IsVCForceEnabled() const    { return _bApplyVCForce; }

  void  SetVorticityConfinementScale(float scale) { _rVorticityConfinementScale = scale; }
  float GetVorticityConfinementScale() const      { return _rVorticityConfinementScale;  }
  
  void  SetMassColor(float color[3])  
  { _rInkColor[0] = color[0]; _rInkColor[1] = color[1]; _rInkColor[2] = color[2]; }
  void  GetMassColor(float color[3]) const         
  { color[0] = _rInkColor[0]; color[1] = _rInkColor[1]; color[2] = _rInkColor[2]; }

protected: // methods
  void _CreateOffsetTextures();
  void _InitializeSlabOps(CGcontext context);
  void _ClearTexture(unsigned int id, GLenum target = GL_TEXTURE_RECTANGLE_NV);
  
protected: // types

  // These "SlabOp" (Slab Operation) types are the workhorses of the 
  // simulation.  They abstract the computation through a generic interface.
  // This uses the "geep" library.
  
  // The standard SlabOp: just performs a computation by binding a fragment
  // program and rendering a viewport-filling quad.
  typedef SlabOp < NoopRenderTargetPolicy, NoopGLStatePolicy,
                   NoopVertexPipePolicy, GenericCgGLFragmentPipePolicy, 
                   MultiTextureGLComputePolicy, CopyTexGLUpdatePolicy >
  FloSlabOp;  

  // The boundary condition slabop.  Performs a computation only at boundaries
  // (one pixel wide) by binding a fragment program and rendering 4 
  // 1-pixel-wide quads at the edges of the viewport.
  typedef SlabOp < NoopRenderTargetPolicy, NoopGLStatePolicy, 
                   NoopVertexPipePolicy, GenericCgGLFragmentPipePolicy,
                   BoundaryGLComputePolicy, NoopUpdatePolicy >
  FloBCOp;

  // The display Slabop.  This simply displays a float texture to the screen,
  // using a specified fragment program.
  typedef SlabOp < NoopRenderTargetPolicy, NoopGLStatePolicy, 
                   NoopVertexPipePolicy, GenericCgGLFragmentPipePolicy,
                   SingleTextureGLComputePolicy, NoopUpdatePolicy >
  FloDisplayOp; 

protected:
  // constants
  int             _iWidth;
  int             _iHeight;

  float           _dx;               // grid scale (assumes square cells)
  
  float           _rTimestep;
  int             _iNumPoissonSteps; // number of steps used in the jacobi solver

  float           _rInkColor[3];
  float           _rInkLongevity;   // how long the "ink" lasts

  float           _rViscosity;

  bool            _bArbitraryBC;

  bool            _bClearPressureEachStep;

  bool            _bImpulseToProcess;
  bool            _bInkToAdd;

  bool            _bComputeVorticity;
  bool            _bApplyVCForce;
  float           _rVorticityConfinementScale;
  
  // Simulation operations
  FloSlabOp       _addImpulse;
  FloSlabOp       _advect;
  FloSlabOp       _divergence;
  FloSlabOp       _poissonSolver;
  FloSlabOp       _subtractGradient;
  FloSlabOp       _vorticity;
  FloSlabOp       _vorticityForce;  

  FloSlabOp       _updateOffsets;
  FloSlabOp       _arbitraryVelocityBC;
  FloSlabOp       _arbitraryPressureBC;

  FloBCOp         _boundaries;
  
  // Display operations
  FloDisplayOp    _displayScalar;
  FloDisplayOp    _displayVector;
  FloDisplayOp    _displayScalarBilerp;
  FloDisplayOp    _displayVectorBilerp;

  RenderTexture   *_pOffscreenBuffer;

  // Textures
  unsigned int    _iTextures[TEXTURE_COUNT];
    
  unsigned int    _iVelocityOffsetTexture;
  unsigned int    _iPressureOffsetTexture;

  unsigned int    _iDisplayTexture;
  unsigned int    _iBCTexture;
  unsigned int    _iBCDisplayTexture;
  unsigned int    _iBCDetailTexture;


  float           *_zeros; // used for clearing textures
  
private:
  Flo(); // private default constructor prevents default construction.
};

#endif //__FLO_H__
