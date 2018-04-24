/*! \file GpuOcclusion.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for approximating
 *         ambient occlusion on the gpu.
 */

#ifndef GPU_OCCLUSION_H
#define GPU_OCCLUSION_H

#include <gl++/texture/Texture.h>
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>
#include <gl++/framebuffer/Framebuffer.h>
#include "OcclusionTree.h"

class GpuOcclusion
{
  public:
    void init(const OcclusionTree &tree);

    virtual void reloadShaders(void);

    void initTextures(const OcclusionTree &tree);

    /*! This method computes a pass of occlusion and stores it in the current disc occlusion Texture.
     *  \param epsilon Error-control parameter.
     *  \param distanceAttenuation Factor to use when attenuating occlusion by distance.
     */
    void computeOcclusion(const float epsilon, const float distanceAttenuation = 0.0f);

    /*! Since the occlusion solution often oscillates between two values rather than
     *  converge to a single solution, we include this method which computes a weighted
     *  minimum of the two mDiscOcclusion Textures.  We compute a minimum since
     *  this method tends to overestimate occlusion resulting in a dark image.
     */
    void computeWeightedMinimumOcclusion(void);

    /*! This method binds the program and textures to compute per-fragment occlusion.
     *  \param epsilon Error-control parameter.
     *  \param distanceAttenuation Factor to use when attenuating occlusion by distance.
     */
    void bind(const float epsilon, const float distanceAttenuation = 0.0f);

    /*! This method unbinds the program and textures.
     */
    void unbind(void);

    /*! This method binds the program and textures to compute robust per-fragment occlusion.
     *  \param epsilon Error-control parameter.
     *  \param triangleAttenuation How much to attenuate the contribution from triangles.
     */
    void bindRobust(const float epsilon, const float distanceAttenuation, const float triangleAttenuation);

    /*! This method unbinds the robust program and textures.
     */
    void unbindRobust(void);

  protected:
    /*! This method binds the program to compute disc-to-disc occlusion.
     *  \param epsilon Error-control parameter.
     *  \param distanceAttenuation Factor to use when attenuating occlusion by distance.
     */
    void bindDiscOcclusionProgram(const float epsilon, const float distanceAttenuation);

    /*! This method unbinds the program to compute disc-to-disc occlusion.
     */
    void unbindDiscOcclusionProgram(void);

    /*! This method sets up the OpenGL state to prepare to compute a pass of occlusion.
     */
    void setupGLState(void);

    /*! This method restores the OpenGL state after computing a pass of occlusion.
     */
    void restoreGLState(void);

    /*! A Texture stores disc centers.
     */
    Texture mDiscCenters;

    /*! A Texture stores normals and areas.
     */
    Texture mDiscNormalsAndAreas;

    /*! A Texture stores pointers to encode the tree structure.
     */
    Texture mNodePointers;

    /*! The texture coordinates of the root of the tree in mNodePointers.
     */
    float2 mTreeRoot;

    /*! Two Textures (for ping-ponging) store occlusion.
     */
    Texture mDiscOcclusion[2];

    /*! Records which Texture (0 or 1) of mOcclusion is the more
     *  current data.
     */
    unsigned int mCurrentOcclusionTexture;

    /*! A Texture stores triangle vertices.
     */
    Texture mVertices;

    /*! Two Textures pack triangle vertex "indices".
     */
    Texture mTriangles[2];

    /*! A fragment shader to compute disc-to-disc occlusion.
     */
    Shader mDiscOcclusionShader;

    /*! A program to compute disc-to-disc occlusion.
     */
    Program mDiscOcclusionProgram;

    /*! A vertex shader to pass world position & normal to
     *  the fragment shader for disc-to-fragment occlusion.
     */
    Shader mWorldAndNormalPassthrough;

    /*! A fragment shader to compute disc-to-fragment occlusion.
     */
    Shader mFragmentOcclusionShader;

    /*! A program to compute disc-to-fragment occlusion.
     */
    Program mFragmentOcclusionProgram;

    /*! A fragment shader to compute robust disc/triangle-to-fragment occlusion.
     */
    Shader mRobustOcclusionShader;

    /*! A program to compute robust dist/triangle-to-fragment occlusion.
     */
    Program mRobustOcclusionProgram;

    /*! A fragment shader to compute a weighted minimum of two Textures.
     */
    Shader mWeightedMinimumShader;

    /*! A program to compute a weighted minimum of two Textures.
     */
    Program mWeightedMinimumProgram;

    /*! A Framebuffer for computation.
     */
    Framebuffer mFramebuffer;
}; // end GpuOcclusion

#endif // GPU_OCCLUSION_H
