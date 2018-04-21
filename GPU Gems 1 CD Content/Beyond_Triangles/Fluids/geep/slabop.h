//----------------------------------------------------------------------------
// File : slabop.h
//----------------------------------------------------------------------------
// Copyright 2003 Mark J. Harris and
// The University of North Carolina at Chapel Hill
//----------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software and its 
// documentation for any purpose is hereby granted without fee, provided that 
// the above copyright notice appear in all copies and that both that copyrigh
// notice and this permission notice appear in supporting documentation. 
// Binaries may be compiled with this software without any royalties or 
// restrictions. 
//
// The author(s) and The University of North Carolina at Chapel Hill make no 
// representations about the suitability of this software for any purpose. 
// It is provided "as is" without express or implied warranty.
/**
 * @file slabop.h
 * 
 * A generic abstraction for the slab computations commonly performed when 
 * a GPU is used for general purpose computation (GPGPU).
 */
#ifndef __SLABOP_H__
#define __SLABOP_H__

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

// "geep" is the sound you make when you pronounce "GPGPU".
namespace geep
{

//----------------------------------------------------------------------------
/**
 * @class SlabOp
 * @brief A generic abstraction of a GPGPU slab computation.
 * 
 * A slab computation is a computation performed in parallel on a 2D lattice
 * or a single slice of a 2D lattice.  SlabOp serves as an abstraction of 
 * the typical idiom of such a computation.  It is a template (host) class, 
 * using policy-based class design [Alexandrescu 2001].  The template 
 * parameters specify policy classes that implement small pieces of a SlabOp's
 * functionality.  This way, by providing a set of simple policy 
 * implementations, a wide variety of slab ops can be created by mixing and 
 * matching policies.  This is very flexible.
 *
 * The policies that must be provided are: 
 *   * RenderTargetPolicy -- sets up / shuts down any special render target 
 *       functionality needed by the SlabOp.
 *   * GLStatePolicy -- sets and unsets the GL state needed for the SlabOp.
 *   * VertexPipePolicy -- Sets up / shuts down vertex programs.
 *   * FragmentPipePolicy -- Sets up / shuts down fragment programs.
 *   * ComputePolicy -- Performs the computation (usually via rendering).
 *   * UpdatePolicy -- performs any copies or other update functions after 
 *       the compution has been performed.
 *
 * The interfaces of the policies are not strictly defined.  They each have
 * one or two methods that are required for SlabOp to compile.  These are 
 * shown in the "Noop" policies given below.  A policy may (and often will) 
 * have a richer interface than just the methods given in the Noops below.
 * See SlabOpGL.h or SlabOpCGGl.h for examples.
 */
template 
<
  class RenderTargetPolicy,
  class GLStatePolicy,
  class VertexPipePolicy,
  class FragmentPipePolicy,
  class ComputePolicy,
  class UpdatePolicy
>
class SlabOp : public RenderTargetPolicy,
               public GLStatePolicy,
               public VertexPipePolicy,
               public FragmentPipePolicy,
               public ComputePolicy,
               public UpdatePolicy
{
public:
  SlabOp()  {}
  ~SlabOp() {}

  // The only method of the SlabOp host class is Compute(), which 
  // uses the inherited policy methods to perform the slab computation.
  // Note that this also defines the interfaces that the policy classes 
  // must have.
  void Compute()
  {
    // Activate the output slab, if necessary
    ActivateRenderTarget();

    // Set the necessary state for the slab operation
    GLStatePolicy::SetState();
    VertexPipePolicy::SetState();
    FragmentPipePolicy::SetState();
   
    SetViewport();

    // Perform the slab operation
    ComputePolicy::Compute();

    // Put the results of the operation into the output slab.
    UpdateOutputSlab();
    
    // Reset state
    FragmentPipePolicy::ResetState();
    VertexPipePolicy::ResetState();
    GLStatePolicy::ResetState();

    // Deactivate the output slab, if necessary
    DeactivateRenderTarget();
  }
};

//----------------------------------------------------------------------------
// "Noop" policies -- these define interfaces for the various SlabOp policies.
//----------------------------------------------------------------------------

struct NoopRenderTargetPolicy
{
protected:
  void ActivateRenderTarget()   {}
  void DeactivateRenderTarget() {}
};

struct NoopGLStatePolicy
{
protected:
  void SetState()   {}
  void ResetState() {}
};

struct NoopVertexPipePolicy
{
protected:
  void SetState()   {}
  void ResetState() {}
};

struct NoopFragmentPipePolicy
{
protected:
  void SetState()   {}
  void ResetState() {}
};

struct NoopComputePolicy
{
protected:
  void Compute() {}
};

struct NoopUpdatePolicy
{
protected:
  void SetViewport() {}
  void UpdateOutputSlab() {}
};

// An example (trivial and useless) SlabOp definition.
typedef SlabOp
< 
NoopRenderTargetPolicy, 
NoopGLStatePolicy,
NoopVertexPipePolicy,
NoopFragmentPipePolicy,
NoopComputePolicy,
NoopUpdatePolicy 
>
SlabNop;

};

#endif //__SLABOP_H__