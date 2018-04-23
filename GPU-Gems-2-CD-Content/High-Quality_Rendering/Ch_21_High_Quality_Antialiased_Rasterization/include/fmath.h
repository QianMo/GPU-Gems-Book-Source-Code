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


// fmath.h
//
// Contains definitions for floating-point math routines, for platforms
// that do not support them natively.

#ifndef FMATH_H
#define FMATH_H

#include <cmath>
#include <limits>

#ifdef __ICC
#include <mathimf.h>
#endif


// Define math constants that aren't on all systems
#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#ifndef M_TWO_PI
#define M_TWO_PI (M_PI * 2.0)
#endif

#ifndef M_PI_2
#define M_PI_2 (M_PI / 2.0)
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI / 4.0)
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41413562373095
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678117654752440
#endif

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

#define HUGE_FLOAT ((float)1.0e38)
#define UNINITIALIZED_FLOAT (- std::numeric_limits<float>::max())
inline bool huge (float f) { return (f >= HUGE_FLOAT/2); }
inline bool uninitialized (float f) { return (f == UNINITIALIZED_FLOAT); }

#ifdef WINNT
#define hypotf(x,y) (float)hypot(x,y)
#define isnanf(x) _isnan(x)
#define isinff(x) (!_finite(x))
#define finitef(x) _finite(x)

inline double trunc(double x) { return x < 0.0  ? ceil  (x) : floor  (x); }
inline float  truncf(float x) { return x < 0.0f ? ceilf (x) : floorf (x); }

inline double round(double x)
{
    return x < 0.0  ? ceil (x - .5) : floor (x + .5);
}
inline float  roundf(float x)
{
    return x < 0.0f ? ceilf (x - .5f) : floorf (x + .5f);
}

#endif // WINNT



#ifdef MACOSX

#define cosf(x) (float)cos(x)
#define sinf(x) (float)sin(x)
#define sqrtf(x) (float)sqrt(x)
#define acosf(x) (float)acos(x)
#define asinf(x) (float)asin(x)
#define atan2f(x,y) (float)atan2(x,y)
#define logf(x) (float)log(x)

#endif // MACOSX



inline float radians (float deg)
{
    return float (M_PI*deg/180.0);
}



inline float degrees (float rad)
{
    return float (rad*180.0/M_PI);
}



inline float lerp (float a, float b, float t)
{
    // NOTE: a*(t-1) + b*t is much more numerically stable than a+t*(b-a)
    float t1 = 1.0f - t;
    return a*t1 + b*t;
}



// Array form
inline void lerp (float *a, float *b, float *c, int n, float t) {
    float t1 = 1.0f - t;
    for (int i=0; i<n; i++)
        c[i] = a[i]*t1 + b[i]*t;
}



inline float bilerp (float v0, float v1, float v2, float v3, float s, float t)
{
    // NOTE: a*(t-1) + b*t is much more numerically stable than a+t*(b-a)
    float s1 = 1.0f - s;
    return (1.0f-t)*(v0*s1 + v1*s) + t*(v2*s1 + v3*s);
}



inline void bilerp (const float *v0, const float *v1,
                    const float *v2, const float *v3,
                    float s, float t, int n, float *result)
{
    float s1 = 1.0f - s;
    float t1 = 1.0f - t;
    for (int i = 0;  i < n;  ++i)
        result[i] = t1*(v0[i]*s1 + v1[i]*s) + t*(v2[i]*s1 + v3[i]*s);
}



inline float clamp (float x, float minval, float maxval)
{
    if (x < minval)
        return minval;
    if (x > maxval)
        return maxval;
    return x;
}



inline int clamp (int x, int minval, int maxval)
{
    if (x <= minval)
        return minval;
    if (x >= maxval)
        return maxval;
    return x;
}



// "Safe" acosf -- no possible exception, but it clamps its input to the
// valid domain of acosf.
inline float safe_acosf (float x)
{
    if (x >= 1.0f)
	return 0.0f;
    if (x <= -1.0f)
	return (float) M_PI;
    return acosf (x);
}


// "Safe" asinf -- no possible exception, but it clamps its input to the
// valid domain of asinf.
inline float safe_asinf (float x)
{
    if (x >= 1.0f)
	return float (0.5f * M_PI);
    if (x <= -1.0f)
	return float (-0.5f * M_PI);
    return asinf (x);
}



// "Safe" powf -- avoid exceptions for non-integral negative powers (by
// returning 0), and protect against log underflow for very small numbers
// and large exponents.
// 
// This is based on the principle that if r = x^y, then ln r = y ln x.
// But if (y ln x) is too small, it'll be invalid input to the necessary
// exp().
inline float safe_powf (float x, float y)
{
    if (x <= 0.0f) {  // Handle negative numbers
	if (y == floorf(y) && y >= 1.0f) {
            // OK to take positive integral powers of negative numbers
	    return powf (x,y);
	}
	return 0.0f;  // But return 0 for invalid input
    }
    // Compute y * ln(x)
    float ylnx = y * logf(x);
    // small_log was determined empirically to be about the smallest we
    // could reliably take the expf of.
    const float small_log = -87.0f;
    return (ylnx <= small_log) ? 0.0f : expf(ylnx);
}



// Return (x-floor(x)) and put (int)floor(x) in *xi.
// This is similar to the built-in modf, but with a full int.
// Also note that modf rounds the whole nuber toward 0 (whereas we
// always round down), and modf has frac<0 if x<0, whereas we always
// have frac>=0.
inline float
floorfrac (float x, int *xi)
{
    // Multiple implementations are presented below.  The simple one,
    // interestingly, appears to be the fastest.  Because this function
    // is used so pervasively in texture mapping and noise, it can have
    // a noticeable impact on overall runtime.  So it's worth periodically
    // checking which of the implementations is most efficient for each
    // platform.
#if 1
    // The obvious implementation
    int i = (x>=0.0f) ? (int)x : (int)x - 1;
    *xi = i;
    return x-i;
#endif
#if 0
    // Using built-in modf.  
    float xx, frac;
    frac = modff (x, &xx);
    if (x >= 0.0f) {
        *xi = (int)xx;
        return frac;
    } else {
        // Negative is tricky because modf makes the fraction negative also,
        // and it's the integer portion rather than the floor.
        *xi = (int)xx - 1;
        return -frac;
    }
    return frac;
#endif
}



// Class for eight bit to float conversion.  You don't want this initialized
// every time you use it, so declare as 'static' any of these you make.
class EightBitToFloat {
public:
    EightBitToFloat () {
        for (int i = 0;  i < 256;  ++i)
            val[i] = (float)i / 255.0f;
    }
    float operator() (unsigned char c) const { return val[c]; }
private:
    float val[256];
};




inline bool
ispow2 (int x)
{
    // x is a power of 2 iff x == 1<<b iff x-1 is 1 in all bits < b
    return (x & (x-1)) == 0;
}



// FIXME this can be a whole lot more efficient
inline int
pow2roundup (int n)
{
    int m = 1;
    while (m < n)
        m <<= 1;
    return m;
}


#endif /* FMATH_H */

