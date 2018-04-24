//
// hdrimage.h
// Last Updated:		05.01.07
// 
// Mark Colbert & Jaroslav Krivanek
// colbert@cs.ucf.edu
//
// Copyright (c) 2007.
//
// The following code is freely distributed "as is" and comes with 
// no guarantees or required support by the authors.  Any use of 
// the code for commercial purposes requires explicit written consent 
// by the authors.
//

#ifndef _HDR_IMAGE_H
#define _HDR_IMAGE_H

#include "image.h"

class Color3;
class Vector2D;
class Vector3D;

/// Class wrapping Bruce Walter's implementation of the RGBE file format.
/// The additional benefit is it will automatically handle writing the
/// proper header when writing an HDR file to be read in Photoshop.
class HDRImage : public Image<float> {
	public:
		explicit HDRImage(const char *_name);
		~HDRImage();

		/// HDR writing function
		/// Data must be in a floating-point RGB array with no buffer space between the color values.
		static void WriteImage(const char *name, float *data, int w, int h);

		friend class HDRProbe;
	protected:
		void ReadImage(const char *name);
		bool selfAlloc;
};

/// Class to handle constructing various lighting environments when using
/// light probes defined in a angular map.
class HDRProbe {
	public:
		explicit HDRProbe(HDRImage *_image) : image(_image) {}

		void ConstructLatLongMap(float *buffer, int w, int h);
		void ConstrutDualParabolicMap(float *_fBuffer, float *_bBuffer, int w, int h);
		void ConstructCubeMap(float *posx, float *negx, float *posy, float *negy, float *posz, float *negz, int w, int h);

		/// Computation of the spherical harmonic matricies for the irradiance map
		/// > This code is directly from the code provided by Ravi Ramamoorthi
		void ConstructSHMatrices(float *r, float *g, float *b);
		float GetAverageLuminance();

		/// Computes the cube map for each MIP-map level by finding the spherical
		/// coordinates of each point around the center of the pixel, and filtering
		/// the surrounding points via a Gaussian
		/// NOTE: While this is almost an ideal dessimation filter, the linear-filtering
		///		  on the reconstruction filter will still cause edges to appear if using
		///		  lower MIP-map levels
		void ConstructFilteredCubeMap(float *posx, float *negx, float *posy, float *negy, float *posz, float *negz, int res, int level);

	private:
		/// Filtering functions for accessing the angular map light probe
		Color3 BilinearInterpolate(Color3 *buffer, float x, float y, int w, int h);
		Vector2D ComputeSphericalCoord(const Vector3D &v);

		/// supporting functions for computing the MIP-mapped cube map
		Color3 ComputeAverageColor();
		Vector3D ConvertTexcoordToCubeMap(int face, float px, float py);
		Vector2D ComputeAngles(const Vector3D &v);
		void ComputeFilteredArea(int face, float px, float py, float inc, Vector2D *angles);
		Color3 FilterArea(Vector2D &currAngle, Vector2D *angles, int steps, float swf, float sfh);

		/// supporting function for spherical harmonics convolution
		void UpdateCoeffs(float coeff[][3], float hdr[3], float domega, const Vector3D &v);
		void ConvertCoeffsToMatrix(float coeff[][3], float *r, float *g, float *b);

		HDRImage *image;
		float avgLum;
		
};

#endif
