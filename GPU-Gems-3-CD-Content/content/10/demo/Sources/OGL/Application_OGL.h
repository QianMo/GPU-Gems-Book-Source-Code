#pragma once

#include <gl/gl.h>
#include <gl/glu.h>
#include "wglext.h"
#include "glext.h"

#include "Extensions.h"

#pragma comment(lib,"opengl32.lib")

#include "../Framework/Application.h"

// Application mostly handles creating a window and OGL
// There can only be one instance, use GetApp()
//
class Application_OGL : public Application
{
public:

  // create window and initialize D3D
  bool Create(const CreationParams &cp);

  // run and call framefunc every frame
  typedef void (*FrameFunction) (void);
  void Run(FrameFunction framefunc);

  // destroy window and D3D
  void Destroy(void);

  // render target aspect ratio
  float GetAspectRatio(void);

  Application_OGL();
  ~Application_OGL();

public:
  bool m_bTextureArraysSupported;
  bool m_bGeometryShadersSupported;
  bool m_bInstancingSupported;

public:
  HDC m_hDC;
  HGLRC m_hRC;
};

inline Application_OGL *GetApp(void) { return (Application_OGL *)g_pApplication; }

// Creates an application and returns it
extern Application *CreateApplication(void);
