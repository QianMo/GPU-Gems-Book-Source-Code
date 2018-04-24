/*! \file GpuScattering.h
 *  \author Yuntao Jia, refers to GpuOcclusion class by Jared Hoberock
 *  \brief Defines the interface to a class for approximating
 *         multiple scattering on the gpu.
 */

#ifndef GPU_OCCLUSION_H
#define GPU_OCCLUSION_H

#include <gl++/texture/Texture.h>
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>
#include <gl++/framebuffer/Framebuffer.h>
#include <buffer/Buffer.h>
#include <math.h>
#include <vector>
#include <string>
#include "DiscTree.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

// this class manages the material parameters for gpu scattering. 
// details refers to Henrik Wann Jensen's papers at 
// http://graphics.ucsd.edu/~henrik/papers/bssrdf/
class ScatteringParameters
{
public:
	ScatteringParameters(void)
	{
		// jade 
		_names.push_back("Jade");		
 		_albedo_p.push_back(float3(0.067307688f, 0.99009901f, 0.92105263f));
 		_sig_tr.push_back(float3(1.7396553f, 0.10444137f, 0.18493241f));
 		_zr.push_back(float3(0.96153837f, 1.6501650f, 2.6315789f));
 		_zv.push_back(float3(6.0378160f, 10.361929f, 16.524551f));

		// marble 
		_names.push_back("Marble");
		ComputeParams(float3(2.19f, 2.62f, 3.00f),float3(0.0021f, 0.0041f, 0.0071f),1.5f);

		// milk 
		_names.push_back("Wholemilk");
		ComputeParams(float3(2.55f,3.21f,3.77f),float3(0.0011f,0.0024f,0.014f),1.3f);

		// skin1	
		_names.push_back("Skin1");
		ComputeParams(float3(0.74f,0.88f,1.01f),float3(0.032f,0.17f,0.48f),1.3f);

		// skin2
		_names.push_back("Skin2");
		ComputeParams(float3(1.09f,1.59f,1.79f),float3(0.013f,0.070f,0.145f),1.3f);

		// Apple 
		_names.push_back("Apple");
		ComputeParams(float3(2.29f,2.39f,1.97f),float3(0.0030f,0.0034f,0.046f),1.3f);

		// Chicken1 
		_names.push_back("Chicken1");
		ComputeParams(float3(0.15f,0.21f,0.38f),float3(0.015f,0.077f,0.19f),1.3f);

		// Chicken2 
		_names.push_back("Chicken2");
		ComputeParams(float3(0.19f,0.25f,0.32f),float3(0.018f,0.088f,0.20f),1.3f);

		// Cream 
		_names.push_back("Cream");
		ComputeParams(float3(7.38f,5.47f,3.15f),float3(0.0002f,0.0028f,0.0163f),1.3f);

		// Ketchup 
		_names.push_back("Ketchup");
		ComputeParams(float3(0.18f,0.07f,0.03f),float3(0.061f,0.97f,1.45f),1.3f);

		// Potato 
		_names.push_back("Potato");
		ComputeParams(float3(0.68f,0.70f,0.55f),float3(0.0024f,0.0090f,0.12f),1.3f);

		// Skimmilk 
		_names.push_back("Skimmilk");
		ComputeParams(float3(0.70f,1.22f,1.90f),float3(0.0014f,0.0025f,0.0142f),1.3f);

		// Spectralon 
		_names.push_back("Spectralon");
		ComputeParams(float3(11.6f,20.4f,14.9f),float3(0.00f,0.00f,0.00f),1.3f);

		mIndex = 0;
	} // end ScatteringParameters::ScatteringParameters()	

public:
	// refer to equations in Wann Jensen's papers at 
	// http://graphics.ucsd.edu/~henrik/papers/bssrdf/
	void ComputeParams(float3 sig_sP, float3 sig_a, float ior)
	{	
		float Fdr = -1.440f / ior / ior + 0.710f / ior + 0.668f + 0.0636f * ior;		
		float A = (1.f + Fdr) / (1.f - Fdr);
		float3 sig_tP = sig_sP + sig_a;
		float3 sig_tr2 = ((sig_a * sig_tP) * 3.f);
		float3 sig_tr = float3(sqrtf(sig_tr2[0]), sqrtf(sig_tr2[1]), sqrtf(sig_tr2[2]));
		float3 zr = float3(1,1,1) / sig_tP;
		float3 zv = zr * (1.f + 4.f / 3.f * A);
		float3 albedo_p = sig_sP * (float3(1,1,1) / sig_tP);	

		_albedo_p.push_back(albedo_p);
		_sig_tr.push_back(sig_tr);
		_zr.push_back(zr);
		_zv.push_back(zv);
	} // end ScatteringParameters::ComputeParams()

public:	
	void GetScatteringParams(float3& albedo_p, float3& sig_tr, float3& zr, float3& zv)
	{
		albedo_p = _albedo_p[mIndex];
		sig_tr   = _sig_tr[mIndex];
		zr       = _zr[mIndex];
		zv       = _zv[mIndex];		
	} // end ScatteringParameters::GetScatteringParams()

	void Next()
	{
		mIndex = (mIndex+1)%(int)_albedo_p.size();
	} // end ScatteringParameters::Next()

	std::string GetScatteringParamName()
	{
		return _names[mIndex];
	} // end ScatteringParameters::GetScatteringParamName()

public:	
	std::vector<std::string> _names;
	std::vector<float3> _albedo_p;
	std::vector<float3> _sig_tr;
	std::vector<float3> _zr;
	std::vector<float3> _zv;
	int mIndex;
}; // end ScatteringParameters

class GpuScattering
{
  public:
    void init(const DiscTree &tree);

    virtual void reloadShaders(void);

    void initTextures(const DiscTree &tree);

    /*! This method computes a pass of accessibility and stores it in the current disc occlusion Texture.
     *  \param epsilon Error-control parameter.
     */
    void computeAccessibility(const float epsilon);

    /*! This method computes a pass of scattering and stores it in the disc scattering Texture.
     *  \param epsilon Error-control parameter.
     *  \param lightPosition The position of the light source.
     */
    void computeScattering(const float epsilon,
                           const float3 &lightPosition);

    /*! This method copies the contents of the current accessibility
     *  texture into the given buffer object.
     *  \param buffer The buffer to write RGBA to. (bent normal and occlusion)
     *  \param n The number of elements to read
     */
    void copyAccessibility(Buffer &buffer, const unsigned int n);

    /*! This method copies the contents of the scattering
     *  texture into the given buffer object.
     *  \param buffer The buffer to write to.
     *  \param n The number of elements to read
     */
    void copyScattering(Buffer &buffer, const unsigned int n);

    /*! This method changes the scattering materials parameters 	 
     */
    void ChangeScatteringParams(void);

    /*! This method return the name of the scattering materials
     */
    std::string GetScatteringParamName(void);

  protected:
    /*! This method binds the program to compute disc-to-disc accessibility.
     *  \param epsilon Error-control parameter.
     */
    void bindDiscAccessibilityProgram(const float epsilon);

    /*! This method unbinds the program to compute disc-to-disc accessibility.
     */
    void unbindDiscAccessibilityProgram(void);

    /*! This method binds the program and textures to compute per-vertex scattering.
     *  \param epsilon Error-control parameter.
     *  \param lightPosition The position of the light source.
     */
    void bindDiscScatteringProgram(const float epsilon,
                                   const float3 &lightPosition);

    /*! This method unbinds the program and textures.
     */
    void unbindDiscScatteringProgram(void);

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

    /*! Two Textures (for ping-ponging) store bent normals and accessibility.
     */
    Texture mBentNormalsAndAccessibility[2];

    /*! Textures store scattering.
     */
    Texture mDiscScattering;

    /*! Records which Texture (0 or 1) of m mBentNormalsAndAccessibility the more
     *  current data.
     */
    unsigned int mCurrentAccessibilityTexture;

    /*! A fragment shader to compute disc-to-disc accessibility.
     */
    Shader mDiscAccessibilityShader;

    /*! A program to compute disc-to-disc accessibility.
     */
    Program mDiscAccessibilityProgram;

    /*! A fragment shader to compute disc-to-disc scattering.
     */
    Shader mDiscScatteringShader;

    /*! A program to compute disc-to-disc scattering.
     */
    Program mDiscScatteringProgram;

    /*! A Framebuffer for computation.
     */
    Framebuffer mFramebuffer;

    /*! A scattering material parameters manager.
     */
    ScatteringParameters mScatteringParams;
}; // end GpuScattering

#endif // GPU_OCCLUSION_H
