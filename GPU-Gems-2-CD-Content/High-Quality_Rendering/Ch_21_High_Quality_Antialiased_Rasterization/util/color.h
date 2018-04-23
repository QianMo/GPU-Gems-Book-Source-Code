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



#ifndef GELATO_COLOR_H
#define GELATO_COLOR_H


// Some includes we always need
#include <iostream>
#include <cassert>



namespace Gelato {


// The Color3 class is for generic 3-component color values.
// It is very similar to the Vector3 class in vecmat.h, and in
// any case is very self-documenting.
//
// We GUARANTEE that the memory layout and size of a Color3 is
// identical to a generic float[3], and thus it's presumed relatively
// safe to cast between a (float *) and a (Color3 *).
//
// Semantics are as expected: construction from one or three floats or
// a float pointer (presumed to point to 3 floats), [] for accessing
// individual components, the usual arithmetic (+, -, *, /).  Note
// that math ops between two points are always component-by-component.
// 

class Color3 {

public:

    // Constructors
    Color3 (void) { }
    Color3 (float x) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] = x;
#else
        v[0] = x; v[1] = x; v[2] = x;
#endif
    }
    Color3 (float x, float y, float z) {
	v[0] = x; v[1] = y; v[2] = z;
    }
    explicit Color3 (const float *xyz) {
	assert (xyz != NULL);
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] = xyz[i];
#else
	v[0] = xyz[0]; v[1] = xyz[1]; v[2] = xyz[2];
#endif
    }

    // Direct access to the data -- use with caution
    const float *data(void) const { return v; }

    // Simple assignment
    const Color3& operator= (float x) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] = x;
#else
	v[0] = x; v[1] = x; v[2] = x;
#endif
	return *this;
    }
    // Use the default assignment operator (bitwise copy)


    // Component access
    float operator[] (int i) const {
	assert(i>=0 && i < 3);  // range check -- only in DEBUG mode
	return v[i];
    }
    float& operator[] (int i) {
	assert(i>=0 && i < 3);  // range check -- only in DEBUG mode
	return v[i];
    }


    // Comparisons between colors, and with floats
    friend bool operator== (const Color3 &a, const Color3 &b) {
	return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
    friend bool operator!= (const Color3 &a, const Color3 &b) {
	return a[0] != b[0] || a[1] != b[1] || a[2] != b[2];
    }
    friend bool operator== (const Color3 &a, float b) {
	return a[0] == b && a[1] == b && a[2] == b;
    }
    friend bool operator!= (const Color3 &a, float b) {
	return a[0] != b || a[1] != b || a[2] != b;
    }
    friend bool operator== (float b, const Color3 &a) { return (a==b); }
    friend bool operator!= (float b, const Color3 &a) { return (a!=b); }

    // less-than operator: this isn't mathematically meaningful,
    // but lets you use certain STL classes.
    friend bool operator< (const Color3 &a, const Color3 &b) {
	if (a[0] < b[0]) return true;
	else if (a[0] > b[0]) return false;
	if (a[1] < b[1]) return true;
	else if (a[1] > b[1]) return false;
	if (a[2] < b[2]) return true;
	else return false;
    }

    // Stream output -- very handy for debugging
    friend std::ostream& operator<< (std::ostream& out, const Color3& a) {
	return (out << '[' << a[0] << ' ' << a[1] << ' ' << a[2] << ']');
    }


    //////////
    // The obvious color operations:
    // 

    // Addition
    friend Color3 operator+ (const Color3& a, const Color3& b) {
#ifdef VECMAT_LOOPS
        Color3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = a[i] + b[i];
        return r;
#else
	return Color3(a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2]);
#endif
    }
    const Color3& operator+= (const Color3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] += b.v[i];
#else
	v[0] += b.v[0];  v[1] += b.v[1];  v[2] += b.v[2];
#endif
	return *this;
    }
    friend Color3 operator+ (const Color3& a, float b) {
	return Color3(a.v[0]+b, a.v[1]+b, a.v[2]+b);
    }
    friend Color3 operator+ (float a, const Color3& b) {
	return Color3(a+b.v[0], a+b.v[1], a+b.v[2]);
    }
    const Color3& operator+= (float b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] += b;
#else
	v[0] += b;  v[1] += b;  v[2] += b;
#endif
	return *this;
    }

    // Subtraction
    friend Color3 operator- (const Color3& a, const Color3& b) {
#ifdef VECMAT_LOOPS
        Color3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = a[i] - b[i];
        return r;
#else
	return Color3(a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2]);
#endif
    }
    const Color3& operator-= (const Color3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] -= b.v[i];
#else
	v[0] -= b.v[0];  v[1] -= b.v[1];  v[2] -= b.v[2];
#endif
	return *this;
    }
    friend Color3 operator- (const Color3& a, float b) {
	return Color3(a.v[0]-b, a.v[1]-b, a.v[2]-b);
    }
    friend Color3 operator- (float a, const Color3& b) {
	return Color3(a-b.v[0], a-b.v[1], a-b.v[2]);
    }
    const Color3& operator-= (float b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] -= b;
#else
	v[0] -= b;  v[1] -= b;  v[2] -= b;
#endif
	return *this;
    }

    // Negation
    friend Color3 operator- (const Color3& b) {
	return Color3(-b.v[0], -b.v[1], -b.v[2]);
    }

    // Scalar multiplication
    friend Color3 operator* (const Color3& a, float b) {
	return Color3(a.v[0]*b, a.v[1]*b, a.v[2]*b);
    }
    friend Color3 operator* (float b, const Color3& a) {
	return Color3(a.v[0]*b, a.v[1]*b, a.v[2]*b);
    }
    const Color3& operator*= (float b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] *= b;
#else
	v[0] *= b;  v[1] *= b;  v[2] *= b;
#endif
	return *this;
    }

    // Component-by-component multiplication
    friend Color3 operator* (const Color3& a, const Color3& b) {
#ifdef VECMAT_LOOPS
        Color3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = a[i] * b[i];
        return r;
#else
	return Color3(a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2]);
#endif
    }
    const Color3& operator*= (const Color3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] *= b.v[i];
#else
	v[0] *= b.v[0];  v[1] *= b.v[1];  v[2] *= b.v[2];
#endif
	return *this;
    }

    // Scalar division -- watch out for divide-by-zero
    friend Color3 operator/ (const Color3& a, float b) {
	if (b == 0.0f)
	    return Color3(0.0f);
	else
	    return Color3(a.v[0]/b, a.v[1]/b, a.v[2]/b);
    }
    const Color3& operator/= (float b) {
	if (b == 0.0f)
	    *this = 0.0f;
	else {
	    v[0] /= b;  v[1] /= b;  v[2] /= b;
	}
	return *this;
    }

    // Component-by-component division -- watch out for divide-by-zero
    friend Color3 operator/ (const Color3& a, const Color3& b) {
	return Color3((b.v[0] == 0.0f) ? 0.0f : a.v[0]/b.v[0],
		      (b.v[1] == 0.0f) ? 0.0f : a.v[1]/b.v[1],
		      (b.v[2] == 0.0f) ? 0.0f : a.v[2]/b.v[2]);
    }
    const Color3& operator/= (const Color3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] = (b.v[i] == 0.0f) ? 0.0f : v[i]/b.v[i];
#else
	v[0] = (b.v[0] == 0.0f) ? 0.0f : v[0]/b.v[0];
	v[1] = (b.v[1] == 0.0f) ? 0.0f : v[1]/b.v[1];
	v[2] = (b.v[2] == 0.0f) ? 0.0f : v[2]/b.v[2];
#endif
	return *this;
    }

    // a += b*c
    friend void mad (Color3 &a, float b, const Color3& c) {
#ifdef VECMAT_LOOPS
        for (int i = 0;  i < 3;  ++i)
            a[i] += b * c[i];
#else
        a[0] += b * c[0];
        a[1] += b * c[1];
        a[2] += b * c[2];
#endif
    }

    friend Color3 lerp (const Color3& a, const Color3& b, float t) {
	float t1 = 1.0f - t;
	return Color3 (a[0]*t1 + b[0]*t, a[1]*t1 + b[1]*t, a[2]*t1 + b[2]*t);
    }

    // Return the average of the three components
    float average (void) const {
        return (v[0] + v[1] + v[2]) / 3.0f;
    }

    // Return the equivalent luminance, assuming standard RGB.
    // (These numbers are from Haeberli, "Matrix Operations for
    // Image Processing", 1993.  He says the numbers one usually sees
    // are for gamma 2.2, while these numbers are better for linear RGB.)
    float luminance (void) const {
        return 0.3086f * v[0] + 0.6094f * v[1] + 0.0820f * v[2];
    }

    // Color transformations
    friend Color3 hsv_to_rgb (const Color3 &hsv);
    friend Color3 rgb_to_hsv (const Color3 &rgb);
    friend Color3 hsl_to_rgb (const Color3 &hsl);
    friend Color3 rgb_to_hsl (const Color3 &rgb);
    friend Color3 YIQ_to_rgb (const Color3 &yiq);
    friend Color3 rgb_to_YIQ (const Color3 &rgb);
    friend Color3 xyz_to_rgb (const Color3 &xyz);
    friend Color3 rgb_to_xyz (const Color3 &rgb);

private:
    float v[3];

};


};  /* end namespace Gelato */

#endif /* !defined(GELATO_COLOR_H) */
