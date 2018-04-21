/**
 @file GLG3D.h

 This header includes all of the GLG3D libraries in 
 appropriate namespaces.

 @maintainer Morgan McGuire, matrix@graphics3d.com

 @created 2002-08-07
 @edited  2004-01-07

 Copyright 2001-2003, Morgan McGuire.
 All rights reserved.
*/

#ifndef G3D_GLG3D_H
#define G3D_GLG3D_H

#include "graphics3D.h"

// Set up the linker on Windows
#ifdef G3D_WIN32
    #ifndef GLG3D_BUILDING_LIBRARY_DLL
        #ifdef _DEBUG
            #pragma comment(lib, "GLG3D-debug.lib")        
        #else
            #pragma comment(lib, "GLG3D.lib")
        #endif
    #endif

    #pragma comment(lib, "opengl32.lib")
    #pragma comment(lib, "glut32.lib")
    #pragma comment(lib, "glu32.lib")
    #pragma comment(lib, "sdl.lib")
    #pragma comment(lib, "sdlmain.lib")
#endif

#include "GLG3D/glheaders.h"
#include "GLG3D/glcalls.h"
#include "GLG3D/getOpenGLState.h"
#include "GLG3D/TextureFormat.h"
#include "GLG3D/Texture.h"
#include "GLG3D/glFormat.h"
#include "GLG3D/Milestone.h"
#include "GLG3D/RenderDevice.h"
#include "GLG3D/VARArea.h"
#include "GLG3D/VAR.h"
#include "GLG3D/GFont.h"
#include "GLG3D/LightingParameters.h"
#include "GLG3D/Sky.h"
#include "GLG3D/UserInput.h"
#include "GLG3D/ManualCameraController.h"
#include "GLG3D/VertexProgram.h"
#include "GLG3D/Draw.h"
#include "GLG3D/tesselate.h"
#include "GLG3D/GApp.h"
#include "GLG3D/PosedModel.h"
#include "GLG3D/IFSModel.h"
#include "GLG3D/MD2Model.h"
#include "GLG3D/TextureManager.h"
#include "GLG3D/shadowVolume.h"

#endif
