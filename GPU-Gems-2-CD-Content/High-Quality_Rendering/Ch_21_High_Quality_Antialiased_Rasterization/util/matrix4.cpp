/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)
/////////////////////////////////////////////////////////////////////////////


#include <cmath>
#include <iostream>

#include "fmath.h"
#include "vecmat.h"

namespace Gelato {



inline void
swap (float& s1, float& s2)
{
    register float temp = s1;
    s1 = s2;
    s2 = temp;
}


// Implementation of Matrix4 inverse.  Note that this is a standard
// matrix inversion routine adapted from Numerical Recipes in C (but
// obviously changed around quite a bit to adhere to C++ conventions).
Matrix4
Matrix4::inverse (float *determinant) const
{
    Matrix4 invary = *this;
    int  pivlst[8];
    bool pivchk[4];
    // Need to initialize lerow,lecol in case the matrix is all zeros and NaNs.
    int i, j, l, lerow = 0, lecol = 0;
    float d = 1;   // determinant

    memset (pivchk, 0, 4*sizeof(bool));
    for (i = 0;  i < 4;  i++) {
	float leval = 0.0f;
	for (j = 0;  j < 4;  j++) {
	    if (! pivchk[j])
		for (int k = 0;  k < 4;  k++)
		    if (! pivchk[k]  &&  fabsf(invary[j][k]) > leval) {
			lerow = j;
			lecol = k;
			leval = fabsf(invary[j][k]);
                    }
	}
	pivchk[lecol] = 1;
	pivlst[i*2] = lerow;
	pivlst[i*2+1] = lecol;
	if (lerow != lecol) {
	    d = -d;
	    for (l = 0;  l < 4;  ++l)
		swap (invary[lerow][l], invary[lecol][l]);
        }
	float piv = invary[lecol][lecol];
	d *= piv;
	invary[lecol][lecol] = 1.0f;
	for (l = 0;  l < 4;  l++)
	    invary[lecol][l] /= piv;
	for (int l1 = 0;  l1 < 4;  l1++)
	    if (l1 != lecol) {
		float t = invary[l1][lecol];
		invary[l1][lecol] = 0.0f;
		for (l = 0; l < 4; l++)
		    invary[l1][l] -= (invary[lecol][l] * t);
            }
    }
    for (i = 0;  i < 4;  i++) {
	l = 4 - i - 1;
	int l2 = l * 2;
	if (pivlst[l2] != pivlst[l2+1]) {
	    lerow = pivlst[l2];
	    lecol = pivlst[l2+1];
	    for (int k = 0;  k < 4;  k++)
		swap (invary[k][lerow], invary[k][lecol]);
        }
    }
    if (determinant)
	*determinant = d;
    return invary;
}



// Return the matrix which translates by the given vector
Matrix4
Matrix4::TranslationMatrix (const Vector3& trans)
{
    return Matrix4 (1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 0.0f,
		    trans[0], trans[1], trans[2], 1.0f);
}



// Return the matrix which rotates theta (in radians) about the axis
Matrix4
Matrix4::RotationMatrix (float angle, const Vector3& axis)
{
    Vector3 A = normalize (axis);
    const float TWOPI = float (2*M_PI);
    angle -= TWOPI * float (trunc(angle/TWOPI));
    float cosang = cosf(angle);
    float sinang = sinf(angle);
    float cosang1 = 1.0f - cosang;
    return Matrix4 (A[0] * A[0] + (1.0f - A[0] * A[0]) * cosang,
		    A[0] * A[1] * cosang1 + A[2] * sinang,
		    A[0] * A[2] * cosang1 - A[1] * sinang,
		    0.0f,
		    A[0] * A[1] * cosang1 - A[2] * sinang,
		    A[1] * A[1] + (1.0f - A[1] * A[1]) * cosang,
		    A[1] * A[2] * cosang1 + A[0] * sinang,
		    0.0f,
		    A[0] * A[2] * cosang1 + A[1] * sinang,
		    A[1] * A[2] * cosang1 - A[0] * sinang,
		    A[2] * A[2] + (1.0f - A[2] * A[2]) * cosang,
		    0.0f,
		    0.0f, 0.0f, 0.0f, 1.0f);
}


// Return the matrix which translates by the given vector
Matrix4
Matrix4::ScaleMatrix (float sx, float sy, float sz)
{
    return Matrix4 (sx, 0.0f, 0.0f, 0.0f,
		    0.0f, sy, 0.0f, 0.0f,
		    0.0f, 0.0f, sz, 0.0f,
		    0.0f, 0.0f, 0.0f, 1.0f);
}


void
Matrix4::lookat(const Vector3& lookat, const Vector3& up)
{

    // Pick out the translation, find viewing vector.
    Vector3 p(m[12], m[13], m[14]);
    Vector3 z = lookat - p;
    z.normalize();
    Vector3 x = cross (up, z);
    x.normalize();
    Vector3 y = cross(z, x);
    
    // Construct the matrix from the basis vectors in left handed space
    m[0]  = x[0];
    m[1]  = x[1];
    m[2]  = x[2];
    m[3]  = 0;
    
    m[4]  = y[0];
    m[5]  = y[1];
    m[6]  = y[2];
    m[7]  = 0;
    
    m[8]  = z[0];
    m[9]  = z[1];
    m[10] = z[2];
    m[11] = 0;
    
    m[12] = p[0];
    m[13] = p[1];
    m[14] = p[2];
    m[15] = 1;
}



Matrix4
Matrix4::LookatMatrix (const Vector3& camera,
    const Vector3& lookat, const Vector3& up)
{
    Matrix4 c = TranslationMatrix(camera);
    c.lookat (lookat, up);
    return c;
}



Matrix4
Matrix4::PerspectiveMatrix (float fov, float aspect, float hither, float yon)
{
    float tanhalffov = tanf (0.5f * fov * float (M_PI)/180.0f);
    float depth = float ((double)yon - (double)hither);
    return Matrix4 (
        1/tanhalffov,    0,          0,         0,
        0,      aspect/tanhalffov,   0,         0,
        0,               0,      yon/depth,     1,
        0,               0,    -yon*hither/depth, 0);
}



// Taken from GL Reference manual.  Note that the matrix is transposed
// and that the glOrtho man page is incorrect (or displayed incorrectly).
Matrix4
Matrix4::OrthoMatrix (float left, float right, float bottom, float top,
    float hither, float yon)
{
    float tx = -(right+left)/(right-left);
    float ty = -(top+bottom)/(top-bottom);
    float tz = -(yon+hither)/(yon-hither);
    
    return Matrix4(
        2/(right-left), 0, 0, 0,
        0, 2/(top-bottom), 0, 0,
        0, 0, -2/(yon-hither), 0,
        tx, ty, tz, 1);
}



void
transformp (const Matrix4& M, Vector3 *P, int n)
{
    // For now, just do a simple loop.  Someday, this may be something
    // that can be sped up with SSE or some such (handwave, handwave).
    do {
	transformp (M, *P++);
    } while (--n);
}



void
transformv (const Matrix4& M, Vector3 *P, int n)
{
    // For now, just do a simple loop.  Someday, this may be something
    // that can be sped up with SSE or some such (handwave, handwave).
    do {
	transformv (M, *P++);
    } while (--n);
}



void
transformvT (const Matrix4& M, Vector3 *P, int n)
{
    // For now, just do a simple loop.  Someday, this may be something
    // that can be sped up with SSE or some such (handwave, handwave).
    do {
	transformvT (M, *P++);
    } while (--n);
}



// Transform a whole array of points in-place.
void
transformp (const Matrix4& M, Vector4 *P, int n)
{
    // For now, just do a simple loop.  Someday, this may be something
    // that can be sped up with SSE or some such (handwave, handwave).
    do {
	transformp (M, *P++);
    } while (--n);
}



// Transform a whole array of points into a new array.
void
transformp (const Matrix4& M, const Vector3 *P, Vector3 *R, int n)
{
    // For now, just do a simple loop.  Someday, this may be something
    // that can be sped up with SSE or some such (handwave, handwave).
    do {
	transformp (M, *P++, *R++);
    } while (--n);
}

// Transform a whole array of hpoints into a new array.
void
transformp (const Matrix4& M, const Vector4 *P, Vector4 *R, int n)
{
    // For now, just do a simple loop.  Someday, this may be something
    // that can be sped up with SSE or some such (handwave, handwave).
    do {
	transformp (M, *P++, *R++);
    } while (--n);
}

};  /* end namespace Gelato */
