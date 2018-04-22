//----------------------------------------------------------------------------
// File : slabopCgGL.cpp
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
 * @file SlabOpCgGL.cpp
 * 
 * SlabOp policy class implementations that use Cg and GL.
 */

#pragma warning(disable:4786)

#include "SlabOpCgGL.h"
#include <assert.h>

namespace geep
{

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::Initialize
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::Initialize(CGcontext context, string fpFilename)
 * @brief Initializes the vertex program.
 */ 
void GenericCgGLVertexPipePolicy::InitializeVP(CGcontext context, 
                                               string fpFilename)
{
  ShutdownVP();

  // TODO: support user-selected profiles?
  _profile = cgGLGetLatestProfile(CG_GL_VERTEX);
  assert(_profile != CG_PROFILE_UNKNOWN);

  // Load and initialize the Reaction-Diffusion Vertex program
  _vertexProgram = cgCreateProgramFromFile(context, CG_SOURCE, 
                                           fpFilename.c_str(),
                                           _profile, NULL, NULL);
  
  if(_vertexProgram != NULL)
  {
    cgGLLoadProgram(_vertexProgram);
  }
}


//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::Shutdown
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::Shutdown()
 * @brief Shuts down the vertex program.
 */ 
void GenericCgGLVertexPipePolicy::ShutdownVP()
{
  if (_vertexProgram)
  {
    cgDestroyProgram(_vertexProgram);
    _vertexProgram = 0;
  }
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter1fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter1fv(string name, const float *v)
 * @brief Passes an array of 1 value to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter1fv(string name, 
                                                        const float *v)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter1fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter1f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter1f(string name, float x)
 * @brief Passes the value to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter1f(string name, float x)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter1f(p, x);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter2fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter2fv(string name, const float *v)
 * @brief Passes an array of 2 values to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter2fv(string name, 
                                                        const float *v)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter2fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter2f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter2f(string name, float x, float y)
 * @brief Passes the 2 values to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter2f(string name, 
                                                       float x, float y)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter2f(p, x, y);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter3fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter3fv(string name, const float *v)
 * @brief Passes an array of 3 values to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter3fv(string name, 
                                                        const float *v)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter3fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter3f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter3f(string name, float x, float y, float z)
 * @brief Passes the 3 values to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter3f(string name, 
                                                       float x, float y, 
                                                       float z)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter3f(p, x, y, z);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter4fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter4fv(string name, const float *v)
 * @brief Passes an array of 4 values to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter4fv(string name, 
                                                        const float *v)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter4fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetVertexParameter4f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetVertexParameter4f(string name, float x, float y, float z, float w)
 * @brief Passes the 4 values to the given named Cg parameter.
 */ 
void GenericCgGLVertexPipePolicy::SetVertexParameter4f(string name, 
                                                       float x, float y, 
                                                       float z, float w)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetParameter4f(p, x, y, z, w);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLVertexPipePolicy::SetStateMatrixParameter
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLVertexPipePolicy::SetStateMatrixParameter(string name, GLenum matrix, GLenum transform)
 * @brief Sets the specified parameter to the specified transformed GL state matrix.
 */ 
void GenericCgGLVertexPipePolicy::SetStateMatrixParameter(string name, 
                                                          CGGLenum matrix, 
                                                          CGGLenum transform)
{
  CGparameter p = cgGetNamedParameter(_vertexProgram, name.c_str());
  assert(p);
  cgGLSetStateMatrixParameter(p, matrix, transform);
}

//----------------------------------------------------------------------------
// Function     	  : FloAddForcesVertexPipePolicy::SetState
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn FloAddForcesVertexPipePolicy::SetState()
 * @brief Sets Vertex program state.
 */ 
void GenericCgGLVertexPipePolicy::SetState()
{
  cgGLBindProgram(_vertexProgram);
  cgGLEnableProfile(_profile);
}

//----------------------------------------------------------------------------
// Function     	  : FloAddForcesVertexPipePolicy::ResetState
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn FloAddForcesVertexPipePolicy::ResetState()
 * @brief Restores Vertex program state.
 */ 
void GenericCgGLVertexPipePolicy::ResetState()
{
  cgGLDisableProfile(_profile);
}

  
//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::Initialize
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::Initialize(CGcontext context, string fpFilename)
 * @brief Initializes the fragment program.
 */ 
void GenericCgGLFragmentPipePolicy::InitializeFP(CGcontext context, 
                                                 string fpFilename,
                                                 string entryPoint)
{
  ShutdownFP();

  // TODO: support user-selected profiles?
  _profile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
  assert(_profile != CG_PROFILE_UNKNOWN);

  // Load and initialize the Reaction-Diffusion fragment program
  _fragmentProgram = cgCreateProgramFromFile(context, CG_SOURCE, 
                                             fpFilename.c_str(),
                                             _profile,
                                             entryPoint.empty() ? 
                                                NULL : entryPoint.c_str(),
                                             NULL);
  
  if(_fragmentProgram != NULL)
  {
    cgGLLoadProgram(_fragmentProgram);
  }
}


//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::Shutdown
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::Shutdown()
 * @brief Shuts down the fragment program.
 */ 
void GenericCgGLFragmentPipePolicy::ShutdownFP()
{
  if (_fragmentProgram)
  {
    cgDestroyProgram(_fragmentProgram);
    _fragmentProgram = NULL;
  }
  _texParams.clear();
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter1fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter1fv(string name, const float *v)
 * @brief Passes an array of 1 value to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter1fv(string name, 
                                                            const float *v)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter1fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter1f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter1f(string name, float x)
 * @brief Passes the value to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter1f(string name, 
                                                           float x)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter1f(p, x);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter2fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter2fv(string name, const float *v)
 * @brief Passes an array of 2 values to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter2fv(string name, 
                                                            const float *v)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter2fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter2f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter2f(string name, float x, float y)
 * @brief Passes the 2 values to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter2f(string name, 
                                                           float x, float y)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter2f(p, x, y);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter3fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter3fv(string name, const float *v)
 * @brief Passes an array of 3 values to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter3fv(string name, 
                                                            const float *v)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter3fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter3f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter3f(string name, float x, float y, float z)
 * @brief Passes the 3 values to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter3f(string name, 
                                                           float x, float y, 
                                                           float z)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter3f(p, x, y, z);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter4fv
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter4fv(string name, const float *v)
 * @brief Passes an array of 4 values to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter4fv(string name, 
                                                            const float *v)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter4fv(p, v);
}

//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetFragmentParameter4f
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetFragmentParameter4f(string name, float x, float y, float z, float w)
 * @brief Passes the 4 values to the given named Cg parameter.
 */ 
void GenericCgGLFragmentPipePolicy::SetFragmentParameter4f(string name, 
                                                           float x, float y, 
                                                           float z, float w)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  cgGLSetParameter4f(p, x, y, z, w);
}


//----------------------------------------------------------------------------
// Function     	  : GenericCgGLFragmentPipePolicy::SetTextureParameter
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn GenericCgGLFragmentPipePolicy::SetTextureParameter(string name, GLuint texObj)
 * @brief Sets the named Cg texture parameter to the specified texture object ID.
 */ 
void GenericCgGLFragmentPipePolicy::SetTextureParameter(string name, 
                                                        GLuint texObj)
{
  CGparameter p = cgGetNamedParameter(_fragmentProgram, name.c_str());
  assert(p);
  // first erase any existing copy of this parameter, since maps don't allow 
  // duplicates (and attempts to insert them leave the map unchanged).
  _texParams.erase(p);
  _texParams.insert(TexSet::value_type(p, texObj));
}

//----------------------------------------------------------------------------
// Function     	  : FloAddForcesFragmentPipePolicy::SetState
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn FloAddForcesFragmentPipePolicy::SetState()
 * @brief Sets fragment program state.
 */ 
void GenericCgGLFragmentPipePolicy::SetState()
{
  cgGLBindProgram(_fragmentProgram);
 
  cgGLEnableProfile(_profile);

  for (TexSet::iterator iter = _texParams.begin(); 
       iter != _texParams.end(); 
       iter++)
  {
    CGparameter p = iter->first;
    GLuint i = iter->second;
    cgGLSetTextureParameter(iter->first, iter->second);
    cgGLEnableTextureParameter(iter->first);  
  }
}

//----------------------------------------------------------------------------
// Function     	  : FloAddForcesFragmentPipePolicy::ResetState
// Description	    : 
//----------------------------------------------------------------------------
/**
 * @fn FloAddForcesFragmentPipePolicy::ResetState()
 * @brief Restores fragment program state.
 */ 
void GenericCgGLFragmentPipePolicy::ResetState()
{
  for (TexSet::iterator iter = _texParams.begin(); 
       iter != _texParams.end(); 
       iter++)
  {
    cgGLDisableTextureParameter(iter->first);
  }

  cgGLDisableProfile(_profile);
}

}; // namespace geep