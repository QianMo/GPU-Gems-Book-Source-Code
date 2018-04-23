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


#ifndef GELATO_VECMAT_H
#define GELATO_VECMAT_H

// Some includes we always need
#include <iostream>
#include <cassert>
#include <cmath>


namespace Gelato {


// The Vector3 class is for generic 3-component spatial values.
//
// We GUARANTEE that the memory layout and size of a Vector3 is
// identical to a generic float[3], and thus it's presumed relatively
// safe to cast between a (float *) and a (Vector3 *).
//
// Semantics are as expected: construction from one or three floats or
// a float pointer (presumed to point to 3 floats), [] for accessing
// individual components, the usual arithmetic (+, -, *, /).  Note
// that math ops between two vectors are always component-by-component.
// 
// There are a handful of useful friends -- routines for normalizing,
// taking distance between two points, dot and cross products, etc.
//
// It's up to the user to keep track of whether they represent
// points, vectors, or normals, and to only use the operators where
// their semantics make sense.
//
// See also the Matrix4 class for other handy friends.

class Vector3 {

public:

    // Constructors
    Vector3 (void) { }
    explicit Vector3 (float x) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] = x;
#else
        v[0] = x; v[1] = x; v[2] = x;
#endif
    }
    Vector3 (float x, float y, float z) {
	v[0] = x; v[1] = y; v[2] = z;
    }
    explicit Vector3 (const float *xyz) {
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
    const Vector3& operator= (float x) {
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


    // Comparisons between points, and with floats
    friend bool operator== (const Vector3 &a, const Vector3 &b) {
	return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
    friend bool operator!= (const Vector3 &a, const Vector3 &b) {
	return a[0] != b[0] || a[1] != b[1] || a[2] != b[2];
    }
    friend bool operator== (const Vector3 &a, float b) {
	return a[0] == b && a[1] == b && a[2] == b;
    }
    friend bool operator!= (const Vector3 &a, float b) {
	return a[0] != b || a[1] != b || a[2] != b;
    }
    friend bool operator== (float b, const Vector3 &a) { return (a==b); }
    friend bool operator!= (float b, const Vector3 &a) { return (a!=b); }

    // less-than operator: this isn't mathematically meaningful,
    // but lets you use certain STL classes.
    friend bool operator< (const Vector3 &a, const Vector3 &b) {
	if (a[0] < b[0]) return true;
	else if (a[0] > b[0]) return false;
	if (a[1] < b[1]) return true;
	else if (a[1] > b[1]) return false;
	if (a[2] < b[2]) return true;
	else return false;
    }

    // Stream output -- very handy for debugging
    friend std::ostream& operator<< (std::ostream& out, const Vector3& a) {
	return (out << '[' << a[0] << ' ' << a[1] << ' ' << a[2] << ']');
    }


    //////////
    // The obvious vector operations:
    // 

    // Addition
    friend Vector3 operator+ (const Vector3& a, const Vector3& b) {
#ifdef VECMAT_LOOPS
        Vector3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = a[i] + b[i];
        return r;
#else
	return Vector3(a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2]);
#endif
    }
    const Vector3& operator+= (const Vector3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] += b.v[i];
#else
	v[0] += b.v[0];  v[1] += b.v[1];  v[2] += b.v[2];
#endif
	return *this;
    }

    // Subtraction
    friend Vector3 operator- (const Vector3& a, const Vector3& b) {
#ifdef VECMAT_LOOPS
        Vector3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = a[i] - b[i];
        return r;
#else
	return Vector3(a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2]);
#endif
    }
    const Vector3& operator-= (const Vector3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] -= b.v[i];
#else
	v[0] -= b.v[0];  v[1] -= b.v[1];  v[2] -= b.v[2];
#endif
	return *this;
    }

    // Negation
    friend Vector3 operator- (const Vector3& b) {
	return Vector3(-b.v[0], -b.v[1], -b.v[2]);
    }

    // Scalar multiplication
    friend Vector3 operator* (const Vector3& a, float b) {
	return Vector3(a.v[0]*b, a.v[1]*b, a.v[2]*b);
    }
    friend Vector3 operator* (float b, const Vector3& a) {
	return Vector3(a.v[0]*b, a.v[1]*b, a.v[2]*b);
    }
    const Vector3& operator*= (float b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] *= b;
#else
	v[0] *= b;  v[1] *= b;  v[2] *= b;
#endif
	return *this;
    }

    // Component-by-component multiplication
    friend Vector3 operator* (const Vector3& a, const Vector3& b) {
#ifdef VECMAT_LOOPS
        Vector3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = a[i] * b[i];
        return r;
#else
	return Vector3(a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2]);
#endif
    }
    const Vector3& operator*= (const Vector3& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            v[i] *= b.v[i];
#else
	v[0] *= b.v[0];  v[1] *= b.v[1];  v[2] *= b.v[2];
#endif
	return *this;
    }

    // Scalar division -- watch out for divide-by-zero
    friend Vector3 operator/ (const Vector3& a, float b) {
	if (b == 0.0f)
	    return Vector3(0.0f);
	else {
            float binv = 1.0f/b;
	    return Vector3(a.v[0]*binv, a.v[1]*binv, a.v[2]*binv);
        }
    }
    friend Vector3 operator/ (float a, const Vector3& b) {
        Vector3 dst;
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 3;  ++i)
            if (b[i] == 0.0f) dst[i] = 0.0f;
            else dst[i] = a / b[i];
#else
        if (b[0] == 0.0f) dst[0] = 0;
        else dst[0] = a / b[0];
        if (b[1] == 0.0f) dst[1] = 0;
        else dst[1] = a / b[1];
        if (b[2] == 0.0f) dst[2] = 0;
        else dst[2] = a / b[2];
#endif
        return dst;
    }
    const Vector3& operator/= (float b) {
	if (b == 0.0f)
	    *this = 0.0f;
	else {
            float binv = 1.0f/b;
#ifdef VECMAT_LOOPS
            for (int i = 0;  i < 3;  ++i)
                v[i] *= binv;
#else
	    v[0] *= binv;  v[1] *= binv;  v[2] *= binv;
#endif
	}
	return *this;
    }

    // Component-by-component division -- watch out for divide-by-zero
    friend Vector3 operator/ (const Vector3& a, const Vector3& b) {
	return Vector3((b.v[0] == 0.0f) ? 0.0f : a.v[0]/b.v[0],
		      (b.v[1] == 0.0f) ? 0.0f : a.v[1]/b.v[1],
		      (b.v[2] == 0.0f) ? 0.0f : a.v[2]/b.v[2]);
    }
    const Vector3& operator/= (const Vector3& b) {
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

    // a += b*c, but faster because it avoids any extra temporary copies
    friend void mad (Vector3 &a, float b, const Vector3& c) {
#ifdef VECMAT_LOOPS
        for (int i = 0;  i < 3;  ++i)
            a[i] += b * c[i];
#else
        a[0] += b * c[0];
        a[1] += b * c[1];
        a[2] += b * c[2];
#endif
    }

    // Dot product
    friend float dot (const Vector3& a, const Vector3& b) {
	return a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2];
    }

    // Cross product
    friend Vector3 cross (const Vector3& a, const Vector3& b) {
	return Vector3 (a.v[1]*b.v[2] - b.v[1]*a.v[2],
		       a.v[2]*b.v[0] - b.v[2]*a.v[0],
		       a.v[0]*b.v[1] - b.v[0]*a.v[1]);
    }

    friend Vector3 lerp (const Vector3& a, const Vector3& b, float t) {
	float t1 = 1.0f - t;
	return Vector3 (a[0]*t1 + b[0]*t, a[1]*t1 + b[1]*t, a[2]*t1 + b[2]*t);
    }

    friend void lerp (const Vector3& a, const Vector3& b, Vector3& d, float t) {
	float t1 = 1.0f - t;
#ifdef VECMAT_LOOPS
        for (int i = 0;  i < 3;  ++i)
            d[i] = a[i]*t1 + b[i]*t;
#else
        d[0] = a[0]*t1 + b[0]*t;
        d[1] = a[1]*t1 + b[1]*t;
        d[2] = a[2]*t1 + b[2]*t;
#endif
    }

    // Array form
    friend void lerp (Vector3 *a, Vector3 *b, Vector3 *c, int n, float t) {
        float t1 = 1.0f - t;
        for ( ;  n--;  ++a, ++b, ++c) {
            (*c)[0] = (*a)[0]*t1 + (*b)[0]*t;
            (*c)[1] = (*a)[1]*t1 + (*b)[1]*t;
            (*c)[2] = (*a)[2]*t1 + (*b)[2]*t;
        }
    }


    friend Vector3 bilerp (const Vector3& a, const Vector3& b, 
                           const Vector3& c, const Vector3& d,
                           float s, float t)
    {
	float s1 = 1.0f - s;
	float t1 = 1.0f - t;
        Vector3 r;
        for (int i = 0;  i < 3;  ++i)
            r[i] = t1*(s1*a[i] + s*b[i]) + t*(s1*c[i] + s*d[i]);
        return r;
    }

    // Return the Euclidean length
    friend float length (const Vector3& a) {
	return sqrtf (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }
    friend float squaredLength (const Vector3& a) {
	return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }

    // Member normalize alters the Vector3, returns original length
    float normalize (void) {
	float len = length(*this);
	if (len != 0.0f) {
#ifdef VECMAT_LOOPS
            for (int i = 0;  i < 3;  ++i)
                v[i] /= len;
#else
	    v[0] /= len; v[1] /= len; v[2] /= len;
#endif
	}
	return len;
    }
    // Functional normalize returns a normalized Vector3, leaves alone original
    friend Vector3 normalize (const Vector3& P) {
	float len = length(P);
	return (len == 0.0f) ? Vector3(0.0f)
	                     : Vector3(P[0]/len, P[1]/len, P[2]/len);
    }

    // Return the Euclidean distance between two points
    // NOTE: must be 'dist' since 'distance' is something else in STL
    friend float distSquared (const Vector3 &a, const Vector3 &b) {
#ifdef VECMAT_LOOPS
        float x[3];
        for (int i = 0;  i < 3;  ++i) {
            x[i] = a[i] - b[i];
            x[i] *= x[i];
        }
        return x[0] + x[1] + x[2];
#else
	float x = a[0] - b[0];
	float y = a[1] - b[1];
	float z = a[2] - b[2];
	return (x*x + y*y + z*z);
#endif
    }
    friend float dist (const Vector3& a, const Vector3 &b) {
	return sqrtf (distSquared (a, b));
    }

    // Make two unit vectors that are orthogonal to yourself and each
    // other.  This assumes that *this is already normalized.  We get
    // the first orthonormal by taking the cross product of *this and
    // (1,1,1), unless *this is 1,1,1, in which case we cross with
    // (-1,1,1).  Either way, we get something orthogonal.  Then
    // cross(this,a) is mutually orthogonal to the other two.
    void make_orthonormals (Vector3 &a, Vector3 &b) const {
        if (v[0] != v[1] || v[0] != v[2])
            a = Vector3 (v[2]-v[1], v[0]-v[2], v[1]-v[0]);  // (1,1,1)X(this)
        else
            a = Vector3 (v[2]-v[1], v[0]+v[2], -v[1]-v[0]);  // (-1,1,1)X(this)
        a.normalize ();
        b = cross (*this, a);
    }

private:
    float v[3];

};






// The Vector4 class is for 4-component spatial values.
//
// We GUARANTEE that the memory layout and size of a Vector4 is
// identical to a generic float[4], and thus it's presumed relatively
// safe to cast between a (float *) and a (Vector4 *).
//
// Semantics are as expected: construction from one or four floats or
// a float pointer (presumed to point to 4 floats), [] for accessing
// individual components, the usual arithmetic (+, -, *, /).  Note
// that math ops between two vectors are always component-by-component.
// 
// There are a handful of useful friends -- routines for normalizing,
// taking distance between two points, dot and cross products, etc.
//
// It's up to the user to keep track of whether they represent
// points, vectors, or normals, and to only use the operators where
// their semantics make sense.
//
// See also the Matrix4 class for other handy friends.

class Vector4 {

public:

    // Constructors
    Vector4 (void) { }
    explicit Vector4 (float x) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] = x;
#else
        v[0] = x; v[1] = x; v[2] = x; v[3] = x;
#endif
    }
    Vector4 (float x, float y, float z, float w) {
	v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    explicit Vector4 (const float *xyzw) {
	assert (xyzw != NULL);
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] = xyzw[i];
#else
	v[0] = xyzw[0]; v[1] = xyzw[1]; v[2] = xyzw[2]; v[3] = xyzw[3];
#endif
    }

    // Direct access to the data -- use with caution
    const float *data(void) const { return v; }

    // Simple assignment
    const Vector4& operator= (float x) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] = x;
#else
	v[0] = x; v[1] = x; v[2] = x; v[3] = x;
#endif
	return *this;
    }
    // Use the default assignment operator (bitwise copy)


    // Component access
    float operator[] (int i) const {
	assert(i>=0 && i < 4);  // range check -- only in DEBUG mode
	return v[i];
    }
    float& operator[] (int i) {
	assert(i>=0 && i < 4);  // range check -- only in DEBUG mode
	return v[i];
    }


    // Comparisons between points, and with floats
    friend bool operator== (const Vector4 &a, const Vector4 &b) {
	return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
    }
    friend bool operator!= (const Vector4 &a, const Vector4 &b) {
	return a[0] != b[0] || a[1] != b[1] || a[2] != b[2] || a[3] != b[3];
    }
    friend bool operator== (const Vector4 &a, float b) {
	return a[0] == b && a[1] == b && a[2] == b && a[3] == b;
    }
    friend bool operator!= (const Vector4 &a, float b) {
	return a[0] != b || a[1] != b || a[2] != b || a[3] != b;
    }
    friend bool operator== (float b, const Vector4 &a) { return (a==b); }
    friend bool operator!= (float b, const Vector4 &a) { return (a!=b); }

    // less-than operator: this isn't mathematically meaningful,
    // but lets you use certain STL classes.
    friend bool operator< (const Vector4 &a, const Vector4 &b) {
	if (a[0] < b[0]) return true;
	else if (a[0] > b[0]) return false;
	if (a[1] < b[1]) return true;
	else if (a[1] > b[1]) return false;
	if (a[2] < b[2]) return true;
	else if (a[2] > b[2]) return false;
	if (a[3] < b[3]) return true;
	else return false;
    }

    // Stream output -- very handy for debugging
    friend std::ostream& operator<< (std::ostream& out, const Vector4& a) {
	return (out << '[' << a[0] << ' ' << a[1] << ' ' 
                    << a[2] << ' ' << a[3] << ']');
    }


    //////////
    // The obvious vector operations:
    // 

    // Addition
    friend Vector4 operator+ (const Vector4& a, const Vector4& b) {
#ifdef VECMAT_LOOPS
        Vector4 r;
        for (int i = 0;  i < 4;  ++i)
            r[i] = a[i] + b[i];
        return r;
#else
	return Vector4(a.v[0]+b.v[0], a.v[1]+b.v[1],
                       a.v[2]+b.v[2], a.v[3]+b.v[3]);
#endif
    }
    const Vector4& operator+= (const Vector4& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] += b.v[i];
#else
	v[0] += b.v[0];  v[1] += b.v[1];  v[2] += b.v[2];  v[3] += b.v[3];
#endif
	return *this;
    }

    // Subtraction
    friend Vector4 operator- (const Vector4& a, const Vector4& b) {
#ifdef VECMAT_LOOPS
        Vector4 r;
        for (int i = 0;  i < 4;  ++i)
            r[i] = a[i] - b[i];
        return r;
#else
	return Vector4(a.v[0]-b.v[0], a.v[1]-b.v[1],
                       a.v[2]-b.v[2], a.v[3]-b.v[3]);
#endif
    }
    const Vector4& operator-= (const Vector4& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] -= b.v[i];
#else
	v[0] -= b.v[0];  v[1] -= b.v[1];  v[2] -= b.v[2];  v[3] -= b.v[3];
#endif
	return *this;
    }

    // Negation
    friend Vector4 operator- (const Vector4& b) {
	return Vector4(-b.v[0], -b.v[1], -b.v[2], -b.v[3]);
    }

    // Scalar multiplication
    friend Vector4 operator* (const Vector4& a, float b) {
	return Vector4(a.v[0]*b, a.v[1]*b, a.v[2]*b, a.v[3]*b);
    }
    friend Vector4 operator* (float b, const Vector4& a) {
	return Vector4(a.v[0]*b, a.v[1]*b, a.v[2]*b, a.v[3]*b);
    }
    const Vector4& operator*= (float b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] *= b;
#else
	v[0] *= b;  v[1] *= b;  v[2] *= b;  v[3] *= b;
#endif
	return *this;
    }

    // Component-by-component multiplication
    friend Vector4 operator* (const Vector4& a, const Vector4& b) {
#ifdef VECMAT_LOOPS
        Vector4 r;
        for (int i = 0;  i < 4;  ++i)
            r[i] = a[i] * b[i];
        return r;
#else
	return Vector4(a.v[0]*b.v[0], a.v[1]*b.v[1],
                       a.v[2]*b.v[2], a.v[3]*b.v[3]);
#endif
    }
    const Vector4& operator*= (const Vector4& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] *= b.v[i];
#else
	v[0] *= b.v[0];  v[1] *= b.v[1];  v[2] *= b.v[2];  v[3] *= b.v[3];
#endif
	return *this;
    }

    // Scalar division -- watch out for divide-by-zero
    friend Vector4 operator/ (const Vector4& a, float b) {
	if (b == 0.0f)
	    return Vector4(0.0f);
	else {
            float binv = 1.0f/b;
	    return Vector4(a.v[0]*binv, a.v[1]*binv, a.v[2]*binv, a.v[3]*binv);
        }
    }
    friend Vector4 operator/ (float a, const Vector4& b) {
        Vector4 dst;
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            if (b[i] == 0.0f) dst[i] = 0.0f;
            else dst[i] = a / b[i];
#else
        if (b[0] == 0.0f) dst[0] = 0;
        else dst[0] = a / b[0];
        if (b[1] == 0.0f) dst[1] = 0;
        else dst[1] = a / b[1];
        if (b[2] == 0.0f) dst[2] = 0;
        else dst[2] = a / b[2];
        if (b[3] == 0.0f) dst[3] = 0;
        else dst[3] = a / b[3];
#endif
        return dst;
    }
    const Vector4& operator/= (float b) {
	if (b == 0.0f)
	    *this = 0.0f;
	else {
            float binv = 1.0f/b;
#ifdef VECMAT_LOOPS
            for (int i = 0;  i < 4;  ++i)
                v[i] *= binv;
#else
	    v[0] *= binv;  v[1] *= binv;  v[2] *= binv;  v[3] *= binv;
#endif
	}
	return *this;
    }

    // Component-by-component division -- watch out for divide-by-zero
    friend Vector4 operator/ (const Vector4& a, const Vector4& b) {
	return Vector4((b.v[0] == 0.0f) ? 0.0f : a.v[0]/b.v[0],
                       (b.v[1] == 0.0f) ? 0.0f : a.v[1]/b.v[1],
                       (b.v[2] == 0.0f) ? 0.0f : a.v[2]/b.v[2],
                       (b.v[3] == 0.0f) ? 0.0f : a.v[3]/b.v[3]);
    }
    const Vector4& operator/= (const Vector4& b) {
#ifdef VECMAT_LOOPS
	for (int i = 0;  i < 4;  ++i)
            v[i] = (b.v[i] == 0.0f) ? 0.0f : v[i]/b.v[i];
#else
	v[0] = (b.v[0] == 0.0f) ? 0.0f : v[0]/b.v[0];
	v[1] = (b.v[1] == 0.0f) ? 0.0f : v[1]/b.v[1];
	v[2] = (b.v[2] == 0.0f) ? 0.0f : v[2]/b.v[2];
	v[3] = (b.v[3] == 0.0f) ? 0.0f : v[3]/b.v[3];
#endif
	return *this;
    }

    // Dot product
    friend float dot (const Vector4& a, const Vector4& b) {
	return a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2] + a.v[3]*b.v[3];
    }

    // Return the Euclidean length
    friend float length (const Vector4& a) {
	return sqrtf (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
    }
    friend float squaredLength (const Vector4& a) {
	return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
    }

    // Member normalize alters the Vector4, returns original length
    float normalize (void) {
	float len = length(*this);
	if (len != 0.0f) {
            float s = 1.0f/len;
	    v[0] *= s; v[1] *= s; v[2] *= s;  v[3] *= s;
	}
	return len;
    }
    // Functional normalize returns a normalized Vector4, leaves alone original
    friend Vector4 normalize (const Vector4& P) {
	float len = length(P);
	return (len == 0.0f) ? Vector4(0.0f)
	                     : Vector4(P[0]/len, P[1]/len, P[2]/len, P[3]/len);
    }

    // Return the Euclidean distance between two points
    // NOTE: must be 'dist' since 'distance' is something else in STL
    friend float distSquared (const Vector4& a, const Vector4 &b) {
#ifdef VECMAT_LOOPS
        float x[4];
        for (int i = 0;  i < 4;  ++i) {
            x[i] = a[i] - b[i];
            x[i] *= x[i];
        }
        return x[0] + x[1] + x[2] + x[3];
#else
	float x = a[0] - b[0];
	float y = a[1] - b[1];
	float z = a[2] - b[2];
	float w = a[3] - b[3];
	return (x*x + y*y + z*z + w*w);
#endif
    }
    friend float dist (const Vector4& a, const Vector4 &b) {
	return sqrtf (distSquared (a, b));
    }

private:
    float v[4];

};




// The Matrix4 class is for a 4x4 matrix.
//
// We GUARANTEE that the memory layout and size of a Matrix4 is
// identical to a float[4][4], and thus it's presumed relatively safe
// to cast between a (float *) and a (Matrix4 *).  We also presume
// that the components of matrices (in either Matrix4 or float[]) has
// the translation portion in elements 12,13,14 -- this is as expected
// by both OpenGL and RenderMan.
//
// There are two Matrix4 constants, given by Matrix4::Zero() and
// Matrix4::Ident().  It may be handy to use these by name to
// provide fastest copying of the values.
//
// Component access is by [int][int].
//
// Multiplication with floats or matrices works as expected.
// Routines for transpose and inverse, as well as point/vector
// (implied 1 and 0 as 4th component, respectively) transformation
// are supplied.
//

class Matrix4 {

public:

    // Special Matrix4's -- Matrix4::Zero(), Matrix4::Ident()
    static const Matrix4& Zero() { 
        static const Matrix4 _zero (0.0);
	return _zero;
    }
    static const Matrix4& Ident() { 
        static const Matrix4 _ident (1.0);
	return _ident;
    }

    //////////////
    // Constructors

    // Constructor with no args - leave uninitialized
    Matrix4 (void) { }

    // Construct from a float f makes f*Ident
    Matrix4 (float x) {
	m[ 0] = x;     m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
	m[ 4] = 0.0f;  m[ 5] = x;     m[ 6] = 0.0f;  m[ 7] = 0.0f;
	m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = x;     m[11] = 0.0f;
	m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = x;
    }

    // Construct directly from 16 floats
    Matrix4 (float m0, float m1, float m2, float m3,
	     float m4, float m5, float m6, float m7,
	     float m8, float m9, float m10, float m11,
	     float m12, float m13, float m14, float m15)
    {
	m[ 0] = m0;  m[ 1] = m1;  m[ 2] = m2;  m[ 3] = m3;
	m[ 4] = m4;  m[ 5] = m5;  m[ 6] = m6;  m[ 7] = m7;
	m[ 8] = m8;  m[ 9] = m9;  m[10] = m10; m[11] = m11;
	m[12] = m12; m[13] = m13; m[14] = m14; m[15] = m15;
    }

    // Construct from float* presumes 16 contigious floats
    Matrix4 (const float *xarray) {
	memcpy (m, xarray, 16*sizeof(float));
    }

    // Simple assignment
    const Matrix4& operator= (float x) {
	m[ 0] = x;     m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
	m[ 4] = 0.0f;  m[ 5] = x;     m[ 6] = 0.0f;  m[ 7] = 0.0f;
	m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = x;     m[11] = 0.0f;
	m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = x;
	return *this;
    }
    // Use the default assignment operator (bitwise copy)

    // Direct access to the data -- use with caution
    const float *data(void) const { return m; }


    // Component access -- [] addresses a row as a float*, 
    // so [][] addresses a float
    const float* operator[] (int i) const {
	assert(i>=0 && i < 4);  // range check -- only in DEBUG mode
	return m+i*4;
    }
    float* operator[] (int i) {
	assert(i>=0 && i < 4);  // range check -- only in DEBUG mode
	return m+i*4;
    }

    // Comparisons between matrices
    friend bool operator== (const Matrix4 &a, const Matrix4 &b) {
	return memcmp (&a, &b, sizeof(Matrix4)) == 0;
    }
    friend bool operator!= (const Matrix4 &a, const Matrix4 &b) {
	return memcmp (&a, &b, sizeof(Matrix4)) != 0;
    }

    // Stream output -- very handy for debugging
    friend std::ostream& operator<< (std::ostream& out, const Matrix4& a) {
	out << '[';
	for (int i = 0;  i < 15;  ++i)
	    out << a.m[i] << ' ';
	out << a.m[15] << ']';
	return out;
    }

    // Matrix multiplication
    friend Matrix4 operator* (const Matrix4& a, const Matrix4& b) {
	Matrix4 result;
	for (int j = 0;  j < 16;  ++j) {
	    int col = (j & 0x03);
	    int row = (j & 0x0c);
#ifdef VECMAT_LOOPS
            result.m[j] = 0.0f;
            for (int i = 0;  i < 4;  ++i)
                result.m[j] += a.m[row+i] * b.m[col+4*i];
#else
	    result.m[j] = a.m[row] * b.m[col]
		        + a.m[row+1] * b.m[col+4]
		        + a.m[row+2] * b.m[col+8]
		        + a.m[row+3] * b.m[col+12];
#endif
	}
	return result;
    }

    // Matrix multiplication by a scalar
    friend Matrix4 operator* (const Matrix4& M, float f) {
	Matrix4 result;
	for (int i = 0;  i < 16;  ++i)
	    result.m[i] = M.m[i] * f;
	return result;
    }
    friend Matrix4 operator* (float f, const Matrix4& M) { return M*f; }
    const Matrix4& operator*= (float f) {
	for (int i = 0;  i < 16;  ++i)
	    m[i] *= f;
	return *this;
    }

    // Matrix division by a scalar -- up to the caller to beware of 
    // division-by-zero
    friend Matrix4 operator/ (const Matrix4& M, float f) {
	Matrix4 result;
	for (int i = 0;  i < 16;  ++i)
	    result.m[i] = M.m[i] / f;
	return result;
    }
    const Matrix4& operator/= (float f) {
	for (int i = 0;  i < 16;  ++i)
	    m[i] /= f;
	return *this;
    }

    // float/matrix is the same as float*inverse(matrix)
    friend Matrix4 operator/ (float f, const Matrix4& M) {
	return f * M.inverse();
    }

    // matrix1/matrix2 is the same as matrix1*inverse(matrix2)
    friend Matrix4 operator/ (const Matrix4 &M1, const Matrix4& M2) {
	return M1 * M2.inverse();
    }

    friend Matrix4 lerp (const Matrix4& a, const Matrix4& b, float t) {
        Matrix4 M;
        float t1 = 1.0f - t;
        for (int i = 0;  i < 16;  ++i)
            M.m[i] = a.m[i]*t1 + b.m[i]*t;
	return M;
    }

    friend void lerp (const Matrix4& a, const Matrix4& b, Matrix4& d, float t) {
        float t1 = 1.0f - t;
        for (int i = 0;  i < 16;  ++i)
            d.m[i] = a.m[i]*t1 + b.m[i]*t;
    }

    // Transpose the rows & columns
    Matrix4 transpose (void) const { 
	return Matrix4 (m[0], m[4], m[ 8], m[12],
			m[1], m[5], m[ 9], m[13],
			m[2], m[6], m[10], m[14],
			m[3], m[7], m[11], m[15]);
    }

    // Matrix inverse
    Matrix4 inverse (float *determinant=NULL) const;

    // Point the Z axis of the current matrix toward the lookat point
    void lookat (const Vector3& lookat, const Vector3& up);
    
    ///////////////////
    // Generate commonly-used transformation matrices
    static Matrix4 TranslationMatrix (const Vector3& trans);
    static Matrix4 RotationMatrix (float angle, const Vector3& axis);
    static Matrix4 ScaleMatrix (float sx, float sy, float sz);
    static Matrix4 LookatMatrix (const Vector3& camera,
        const Vector3& lookat, const Vector3& up);
    static Matrix4 PerspectiveMatrix (float fov, float aspect,
        float hither, float yon);
    static Matrix4 OrthoMatrix (float left, float right, float bottom,
        float top, float hither, float yon);
    
    friend void
    transformp (const Matrix4& M, const Vector4 &P, Vector4 &result) {
#ifdef VECMAT_LOOPS
        assert (&P != &result);
        for (int i = 0;  i < 4;  ++i)
            result[i] = P[0] * M.m[i] + P[1] * M.m[i+4] + P[2] * M.m[i+8] + P[3] * M.m[i+12];
#else
	register float x, y, z, w;
	x = P[0] * M.m[0] + P[1] * M.m[4] + P[2] * M.m[8] + P[3] * M.m[12];
	y = P[0] * M.m[1] + P[1] * M.m[5] + P[2] * M.m[9] + P[3] * M.m[13];
	z = P[0] * M.m[2] + P[1] * M.m[6] + P[2] * M.m[10] + P[3] * M.m[14];
	w = P[0] * M.m[3] + P[1] * M.m[7] + P[2] * M.m[11] + P[3] * M.m[15];
	result[0] = x;
	result[1] = y;
	result[2] = z;
	result[3] = w;
#endif
    }
    friend void transformp (const Matrix4& M, Vector4 &P) {
        Vector4 r;
	transformp (M, P, r);
        P = r;
    }

    // transformp transforms a point (with assumed 4th component == 1)
    // by a matrix.
    friend void
    transformp (const Matrix4& M, const Vector3 &P, Vector3 &result) {
#ifdef VECMAT_LOOPS
        float r[4];
#if 1
        for (int i = 0;  i < 4;  ++i)
            r[i] = P[0] * M.m[i] + P[1] * M.m[4+i] + P[2] * M.m[8+i] + M.m[12+i];
#else  /* alternate - may be better on certain architectures */
        for (int i = 0;  i < 4;  ++i)
            r[i] = M.m[12+i];
        for (int j = 0; j < 3;  ++j)
            for (int i = 0;  i < 4;  ++i)
                r[i] += P[j] * M.m[i+4*j];
#endif

	if (r[3] != 1.0f) {
	    float w = 1.0f/r[3];
            for (int i = 0;  i < 3;  ++i)
                result[i] = r[i] * w;
        } else {
            for (int i = 0;  i < 3;  ++i)
                result[i] = r[i];
        }
#else
	register float x, y, z, w;
	x = P[0] * M.m[0] + P[1] * M.m[4] + P[2] * M.m[8] + M.m[12];
	y = P[0] * M.m[1] + P[1] * M.m[5] + P[2] * M.m[9] + M.m[13];
	z = P[0] * M.m[2] + P[1] * M.m[6] + P[2] * M.m[10] + M.m[14];
	w = P[0] * M.m[3] + P[1] * M.m[7] + P[2] * M.m[11] + M.m[15];
	if (w != 1.0f) {
	    w = 1.0f/w;
	    x *= w;
	    y *= w;
	    z *= w;
	}
	result[0] = x;
	result[1] = y;
	result[2] = z;
#endif
    }

    friend void transformp (const Matrix4& M, Vector3 &P) {
        Vector3 r;
	transformp (M, P, r);
        P = r;
    }

    // transformv transforms a vector (with assumed 4th component == 0)
    // by a matrix.
    friend void
    transformv (const Matrix4& M, const Vector3 &P, Vector3 &result) {
#ifdef VECMAT_LOOPS
	float r[4];
#if 1
        for (int i = 0;  i < 3;  ++i)
            r[i] = P[0] * M.m[i] + P[1] * M.m[4+i] + P[2] * M.m[8+i];
#else  /* alternate - may be better on some architectures */
        for (int i = 0;  i < 4;  ++i)
            r[i] = 0.0f;
        for (int j = 0; j < 3;  ++j)
            for (int i = 0;  i < 3;  ++i)
                r[i] += P[j] * M.m[i+4*j];
#endif
        for (int i = 0;  i < 3;  ++i)
            result[i] = r[i];
#else
	register float x, y, z;
	x = P[0] * M.m[0] + P[1] * M.m[4] + P[2] * M.m[8];
	y = P[0] * M.m[1] + P[1] * M.m[5] + P[2] * M.m[9];
	z = P[0] * M.m[2] + P[1] * M.m[6] + P[2] * M.m[10];
	result[0] = x;
	result[1] = y;
	result[2] = z;
#endif
    }

    friend void transformv (const Matrix4& M, Vector3 &P) {
	transformv (M, P, P);
    }

    // transformvT transforms a vector (with assumed 4th component == 0)
    // by the transpose of a matrix (handy for transforming normals).
    friend void
    transformvT (const Matrix4& M, const Vector3 &P, Vector3 &result) {
#ifdef VECMAT_LOOPS
        float r[4];
        for (int i = 0;  i < 3;  ++i)
            r[i] = P[0] * M.m[i*4] + P[1] * M.m[1+i*4] + P[2] * M.m[2+i*4];
        for (int i = 0;  i < 3;  ++i)
            result[i] = r[i];
#else
	register float x, y, z;
	x = P[0] * M.m[0] + P[1] * M.m[1] + P[2] * M.m[2];
	y = P[0] * M.m[4] + P[1] * M.m[5] + P[2] * M.m[6];
	z = P[0] * M.m[8] + P[1] * M.m[9] + P[2] * M.m[10];
	result[0] = x;
	result[1] = y;
	result[2] = z;
#endif
    }

    friend void transformvT (const Matrix4& M, Vector3 &P) {
	transformvT (M, P, P);
    }

    // Transform normal -- must use the inverse transpose
    friend void
    transformn (const Matrix4& M, const Vector3 &P, Vector3 &result) {
        Matrix4 Minv = M.inverse();
        transformvT (Minv, P, result);
    }

    friend void transformn (const Matrix4& M, Vector3 &P) {
        Matrix4 Minv = M.inverse();
        transformvT (Minv, P);
    }

    // Transform an array of points or vectors in place
    friend void transformp (const Matrix4& M, Vector3 *P, int n);
    friend void transformv (const Matrix4& M, Vector3 *P, int n);
    friend void transformvT (const Matrix4& M, Vector3 *P, int n);
    friend void transformp (const Matrix4& M, Vector4 *P, int n);

    // Transform an array of points into a new array.
    friend void transformp (const Matrix4& M, const Vector3 *P, Vector3 *R,
                            int n);
    friend void transformp (const Matrix4& M, const Vector4 *P, Vector4 *R,
                            int n);

    // Return just the z component of point P transformed by M.  If you
    // only need z, this is cheaper than the whole matrix multiply.
    friend float
    transformp_z (const Matrix4& M, const Vector3 &P) {
	register float z, w;
	z = P[0] * M.m[2] + P[1] * M.m[6] + P[2] * M.m[10] + M.m[14];
	w = P[0] * M.m[3] + P[1] * M.m[7] + P[2] * M.m[11] + M.m[15];
	if (w != 1.0f)
	    z /= w;
        return z;
    }

    // Return just the z component of vector P transformed by M.  If you
    // only need z, this is cheaper than the whole matrix multiply.
    friend float
    transformv_z (const Matrix4& M, const Vector3 &P) {
	return P[0] * M.m[2] + P[1] * M.m[6] + P[2] * M.m[10];
    }

    // Return just the xy components of point P transformed by M.  If you
    // only need x and y, this is cheaper than the whole matrix multiply.
    // CAVEAT: result[2] is not modified.
    friend void
    transformp_xy (const Matrix4& M, const Vector3 &P, Vector3 &result) {
	register float x, y, w;
	x = P[0] * M.m[0] + P[1] * M.m[4] + P[2] * M.m[8] + M.m[12];
	y = P[0] * M.m[1] + P[1] * M.m[5] + P[2] * M.m[9] + M.m[13];
	w = P[0] * M.m[3] + P[1] * M.m[7] + P[2] * M.m[11] + M.m[15];
	if (w != 1.0f) {
	    w = 1.0f/w;
	    x *= w;
	    y *= w;
	}
	result[0] = x;
	result[1] = y;
    }

    friend float transform_length (const Matrix4& M, const float length) {
        // transform the length as three separate vectors, (length, 0, 0),
        // (0, length, 0) and (0, 0, length) and compute the average length
        // of the three transformed vectors.  Note that v*M for each of
        // the above vectors corresponds to a row in the matrix.
        float d = ( M.m[0] * M.m[0] + M.m[1] * M.m[1] + M.m[2] * M.m[2] +
                    M.m[4] * M.m[4] + M.m[5] * M.m[5] + M.m[6] * M.m[6] +
                    M.m[8] * M.m[8] + M.m[9] * M.m[9] + M.m[10] * M.m[10] ) /3;
        return length * sqrt (d);
    }

private:
    float m[16];

};





// The Bbox3 class is for axis-aligned 3D bounding boxes.
//
// We GUARANTEE that the memory layout and size of a Bbox3 is
// identical to a float[6], and thus it's presumed relatively
// safe to cast between a (float *) and a (Bbox3 *).
//
// Semantics are as described below: construction from six floats
// or two Vector3's, unions and tests against points, boxes, and spheres.
// Plus queries for volume, diagonal length, and surface area,
// and methods for transformation by a Matrix4.

enum bboxindex { XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX };



class Bbox3 {

public:

    // Default constructor leaves contents undefined
    Bbox3 (void) { }

    // Constructor from six floats fill in the bbox values
    Bbox3 (float xmin, float xmax, float ymin, float ymax,
           float zmin, float zmax) {
        b[0] = xmin;  b[1] = xmax;
        b[2] = ymin;  b[3] = ymax;
        b[4] = zmin;  b[5] = zmax;
    }

    // Constructor from a Vector3 creates a box that exactly contains
    // the point
    Bbox3 (const Vector3 &v) {
        b[0] = v[0];  b[1] = v[0];
        b[2] = v[1];  b[3] = v[1];
        b[4] = v[2];  b[5] = v[2];
    }

    // Special Bbox3's -- Bbox3::Empty() contains nothing (the next point
    // added will be exactly contained), Bbox3::All() contains everything.
    static const Bbox3& Empty() { 
	static const Bbox3 _empty ( 1.0e38f, -1.0e38f,  1.0e38f,
                                   -1.0e38f,  1.0e38f, -1.0e38f);
	return _empty;
    }
    static const Bbox3& All() { 
        static const Bbox3 _all (-1.0e38f,  1.0e38f, -1.0e38f, 
                                  1.0e38f, -1.0e38f,  1.0e38f);
	return _all;
    }

    // Direct access to the data -- use with caution
    const float *data(void) const { return b; }

    // Component access
    float operator[] (int i) const {
	assert(i>=0 && i < 6);  // range check -- only in DEBUG mode
	return b[i];
    }
    float& operator[] (int i) {
	assert(i>=0 && i < 6);  // range check -- only in DEBUG mode
	return b[i];
    }

    void set_corners(const Vector3& min, const Vector3& max) {
        b[0] = min[0]; b[1] = max[0];
        b[2] = min[1]; b[3] = max[1];
        b[4] = min[2]; b[5] = max[2];
    }

    // Bbox3 = Bbox3   uses the default assignment operator (bitwise copy)

    // Bbox3 = Vector3   makes a Bbox that exactly contains one point
    const Bbox3& operator= (const Vector3 &v) {
        b[0] = v[0];  b[1] = v[0];
        b[2] = v[1];  b[3] = v[1];
        b[4] = v[2];  b[5] = v[2];
	return *this;
    }

    // Bbox3 | Bbox3   gives a new bbox that is the union
    friend const Bbox3 operator| (const Bbox3& bb1, const Bbox3& bb2) {
	Bbox3 b;
	register float *b1 = (float *)&(bb1), *b2 = (float *)&(bb2);
	b[0] = (*b1 < *b2 ? *b1 : *b2);  ++b1;  ++b2;
	b[1] = (*b1 > *b2 ? *b1 : *b2);  ++b1;  ++b2;
	b[2] = (*b1 < *b2 ? *b1 : *b2);  ++b1;  ++b2;
	b[3] = (*b1 > *b2 ? *b1 : *b2);  ++b1;  ++b2;
	b[4] = (*b1 < *b2 ? *b1 : *b2);  ++b1;  ++b2;
	b[5] = (*b1 > *b2 ? *b1 : *b2);
	return b;
    }
    const Bbox3& operator|= (const Bbox3& b2) {
	if (b[0] > b2.b[0]) b[0] = b2.b[0];
	if (b[1] < b2.b[1]) b[1] = b2.b[1];
	if (b[2] > b2.b[2]) b[2] = b2.b[2];
	if (b[3] < b2.b[3]) b[3] = b2.b[3];
	if (b[4] > b2.b[4]) b[4] = b2.b[4];
	if (b[5] < b2.b[5]) b[5] = b2.b[5];
	return *this;
    }

    // Bbox3 | Vector3  includes both the box and the point
    const Bbox3& operator|= (const Vector3& p) {
	if (p[0] < b[0]) b[0] = p[0];
	if (p[0] > b[1]) b[1] = p[0];
	if (p[1] < b[2]) b[2] = p[1];
	if (p[1] > b[3]) b[3] = p[1];
	if (p[2] < b[4]) b[4] = p[2];
	if (p[2] > b[5]) b[5] = p[2];
	return *this;
    }
    friend const Bbox3 operator| (const Bbox3& box, const Vector3& p) {
        Bbox3 b(box);
	if (p[0] < b[0]) b[0] = p[0];
	if (p[0] > b[1]) b[1] = p[0];
	if (p[1] < b[2]) b[2] = p[1];
	if (p[1] > b[3]) b[3] = p[1];
	if (p[2] < b[4]) b[4] = p[2];
	if (p[2] > b[5]) b[5] = p[2];
	return b;
    }
    friend const Bbox3 operator| (const Vector3& p, const Bbox3& box) {
        return (box | p);
    }

    // update the box to include the sphere centered at p with radius
    const Bbox3& bound_sphere (const Vector3& p, float radius) {
        if (p[0] - radius < b[0]) b[0] = p[0] - radius;
        if (p[0] + radius > b[1]) b[1] = p[0] + radius;
        if (p[1] - radius < b[2]) b[2] = p[1] - radius;
        if (p[1] + radius > b[3]) b[3] = p[1] + radius;
        if (p[2] - radius < b[4]) b[4] = p[2] - radius;
        if (p[2] + radius > b[5]) b[5] = p[2] + radius;
        return *this;
    }
    
    // Set this bound to the smallest bound containing p[0..n-1]
    void bound_points (int n, const Vector3 *p);
    // Merge p[0..n-1] into our bounding box.
    void bound_more_points (int n, const Vector3 *p);

    // Comparisons between bounds
    friend bool operator== (const Bbox3 &a, const Bbox3 &b);
    friend bool operator!= (const Bbox3 &a, const Bbox3 &b);

    // Stream output -- very handy for debugging
    friend std::ostream& operator<< (std::ostream& out, const Bbox3& a);

    // Grow the bounds by a certain amount
    void pad (float f) {
	b[0] -= f;  b[1] += f;
        b[2] -= f;  b[3] += f;
        b[4] -= f;  b[5] += f;
    }

    // Does the bound include a point?
    bool includes (const Vector3& p) const {
	return (p[0] >= b[0] && p[0] <= b[1] &&
                p[1] >= b[2] && p[1] <= b[3] &&
		p[2] >= b[4] && p[2] <= b[5]);
    }

    // Does the bound completely enclose the passed-in bound?
    bool includes (const Bbox3& a) const {
        return a.b[0] >= b[0] && a.b[1] <= b[1] &&
            a.b[2] >= b[2] && a.b[3] <= b[3] &&
            a.b[4] >= b[4] && a.b[5] <= b[5];
    }
                
    // Return true if the bounding boxes overlap at all
    friend bool overlaps (const Bbox3& b1, const Bbox3& b2);

    // Return the volume of the box (clamp at float max)
    float volume (void) const;

    // Return the surface area of the box (clamp at float max)
    float area (void) const;

    // Return the length, width, height of the box
    // (not clamped, so dangerous for empty boxes)
    Vector3 diagonal (void) const {
	return Vector3 (b[1]-b[0], b[3]-b[2], b[5]-b[4]);
    }

    // Return the length of the diagonal of the box
    float diaglen (void) const {
	return sqrtf ((b[1]-b[0])*(b[1]-b[0]) +
                      (b[3]-b[2])*(b[3]-b[2]) +
		      (b[5]-b[4])*(b[5]-b[4]));
    }

    // Return the center of the box
    Vector3 center (void) const {
	return Vector3 ((b[0]+b[1])*0.5f, (b[2]+b[3])*0.5f, (b[4]+b[5])*0.5f);
    }

    // Return one of the 8 corners of the box
    // 0:min,min,min 1:max,min,min 2:min,max,min 3:max,max,min
    // 4:min,min,max 5:max,min,max 6:min,max,max 7:max,max,max
    const Vector3 corner (int i) const {
        assert (i >= 0 && i < 8);
	return Vector3(b[i&1], b[2+((i/2)&1)], b[4+((i/4)&1)]);
    }

    // Retrieve all 8 corners of the box
    void getcorners (Vector3 *corners) const;

    // btransform transforms a Bbox3 by a matrix.
    friend void btransform (const Matrix4& M, const Bbox3 &b, Bbox3 &result);
    friend void btransform (const Matrix4& M, Bbox3 &b) { btransform(M,b,b); }


private:
    float b[6];     // xmin, xmax, ymin, ymax, zmin, zmax
};


};  /* end namespace Gelato */


#endif /* !defined(GELATO_VECMAT_H) */
