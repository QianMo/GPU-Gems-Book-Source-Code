//----------------------------------------------------------------------------
// File : slabopCgGL.h
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
 * @file SlabOpCgGL.h
 * 
 * SlabOp policy classes that use Cg and OpenGL.
 */
#ifndef __SLABOPCGGL_H__
#define __SLABOPCGGL_H__

#include <gl/glew.h>
#include <cg/cggl.h>
#include <string>
#include <map>

#pragma warning(disable:4786)

using namespace std;

namespace geep
{

//----------------------------------------------------------------------------
/**
 * @class GenericCgGLVertexPipePolicy
 * @brief Manages vertex program state using Cg and OpenGL.
 * 
 * This class supports a single vertex program and its parameters.
 */
class GenericCgGLVertexPipePolicy
{
public:
  void InitializeVP(CGcontext context, string vpFilename);
  void ShutdownVP();

  void SetVertexParameter1fv(string name, const float *v);
  void SetVertexParameter1f(string name, float x);
  void SetVertexParameter2fv(string name, const float *v);
  void SetVertexParameter2f(string name, float x, float y);
  void SetVertexParameter3fv(string name, const float *v);
  void SetVertexParameter3f(string name, float x, float y, float z);
  void SetVertexParameter4fv(string name, const float *v);
  void SetVertexParameter4f(string name, float x, float y, float z, float w);

  void SetStateMatrixParameter(string name, CGGLenum matrix, 
                               CGGLenum transform);
  
protected:
  GenericCgGLVertexPipePolicy() : _vertexProgram(0), 
                                  _profile(CG_PROFILE_UNKNOWN) {}
  virtual ~GenericCgGLVertexPipePolicy() {}

  void SetState();   // called by SlabOp::Compute()
  void ResetState(); // called by SlabOp::Compute()

protected: // data
  CGprogram _vertexProgram;
  CGprofile _profile;
};

//----------------------------------------------------------------------------
/**
 * @class GenericCgGLFragmentPipePolicy
 * @brief Manages fragment program state using Cg and OpenGL.
 * 
 * This class supports a single fragment program and its parameters.
 */
class GenericCgGLFragmentPipePolicy
{
public: 
  void InitializeFP(CGcontext context, string fpFilename,
                    string entryPoint = "");
  void ShutdownFP();

  void SetFragmentParameter1fv(string name, const float *v);
  void SetFragmentParameter1f(string name, float x);
  void SetFragmentParameter2fv(string name, const float *v);
  void SetFragmentParameter2f(string name, float x, float y);
  void SetFragmentParameter3fv(string name, const float *v);
  void SetFragmentParameter3f(string name, float x, float y, float z);
  void SetFragmentParameter4fv(string name, const float *v);
  void SetFragmentParameter4f(string name, float x, float y, float z, float w);

  void SetTextureParameter(string name, GLuint texObj);
      
protected: // methods
  GenericCgGLFragmentPipePolicy() : _fragmentProgram(0), 
                                    _profile(CG_PROFILE_UNKNOWN) {}
  virtual ~GenericCgGLFragmentPipePolicy() {}
 
  void SetState();   // called by SlabOp::Compute()
  void ResetState(); // called by SlabOp::Compute()

protected: // types
  
  // We maintain a map of CG parameters to texture objects, so that SetState()
  // and ResetState() can enable and disable them.  A std::map is used because
  // it doesn't allow duplicates.  Since inserting a duplicate key leaves the
  // map unchanged, we must first erase any pair with the parameter we are 
  // setting, so that we reflect the most recent call to SetTextureParameter.
  typedef map<CGparameter, GLuint> TexSet;

protected: // data
  CGprogram _fragmentProgram;
  CGprofile _profile;

  TexSet    _texParams;
};

}; // namespace geep
#endif //__SLABOPCGGL_H__