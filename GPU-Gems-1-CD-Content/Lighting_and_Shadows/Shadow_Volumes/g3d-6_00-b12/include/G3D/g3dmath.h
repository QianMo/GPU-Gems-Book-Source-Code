/**
 @file g3dmath.h
 
 Math util class.
 
 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite Portions by Laura Wollstadt 
 @cite Portions based on Dave Eberly's Magic Software Library
        at <A HREF="http://www.magic-software.com">http://www.magic-software.com</A>
 @cite highestBit by Jukka Liimatta
 
 @created 2001-06-02
 @edited  2004-01-06

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
 */

#ifndef G3DMATH_H
#define G3DMATH_H

// Prevent MSVC from defining min and max macros
#define NOMINMAX

#include "G3D/platform.h"
#include <ctype.h>
#include <float.h>
#include <limits>
#include <math.h>

#undef min
#undef max

namespace G3D {

#ifdef _MSC_VER
    // disable: "conversion from 'double' to 'float', possible loss of
    // data
    #pragma warning (disable : 4244)

    // disable: "truncation from 'double' to 'float'
    #pragma warning (disable : 4305)

    // disable: "C++ exception handler used"
    #pragma warning (disable : 4530)

#endif // _MSC_VER

#define G3D_FLOAT 1
#define G3D_DOUBLE 2


const double fuzzyEpsilon = 0.0000001;

#ifdef _MSC_VER

const double inf = (std::numeric_limits<double>::infinity());
const double nan = (std::numeric_limits<double>::quiet_NaN());
const double NAN = (std::numeric_limits<double>::quiet_NaN());

#else

// On Linux, the std constants are incorrect at compile time, so we have
// to trick the compiler into producing inf and nan without
// producing a compile-time warning.  Since sin(0.0) == 0.0,
// dividing by it hides the divide by zero at compile time
// but produces the correct constants.
const double inf = 1.0/sin(0.0);
const double nan = 0.0/sin(0.0);

#endif

#define G3D_PI      (3.1415926535898)
#define G3D_HALF_PI (1.5707963267949)
#define G3D_TWO_PI  (6.283185)

typedef signed char		int8;
typedef unsigned char	uint8;
typedef short			int16;
typedef unsigned short	uint16;
typedef int				int32;
typedef unsigned int	uint32;

#ifdef _MSC_EXTENSIONS
    typedef __int64			   int64;
    typedef unsigned __int64   uint64;
#else
    typedef long long		   int64;
    typedef unsigned long long uint64;
#endif

typedef float			float32;
typedef double			float64;

int iAbs(int iValue);
int iCeil(double fValue);

/**
 Clamps the value to the range [low, hi] (inclusive)
 */
int iClamp(int val, int low, int hi);
double clamp(double val, double low, double hi);

/**
 Returns a + (b - a) * f;
 */
inline double lerp(double a, double b, double f) {
    return a + (b - a) * f;
}

/**
 Wraps the value to the range [0, hi) (exclusive
 on the high end).  This is like the clock arithmetic
 produced by % (modulo) except the result is guaranteed
 to be positive.
 */
int iWrap(int val, int hi);

int iFloor(double fValue);
int iSign(int iValue);
int iSign(double fValue);
int iRound(double fValue);

/**
 Returns a random number uniformly at random between low and hi
 (inclusive).
 */
int iRandom(int low, int hi);

double abs (double fValue);
double aCos (double fValue);
double aSin (double fValue);
double aTan (double fValue);
double aTan2 (double fY, double fX);
double sign (double fValue);
double square (double fValue);

/**
 Returns true if the argument is a finite real number.
 */
bool isFinite(double x);

/**
 Returns true if the argument is NaN (not a number).
 You can't use x == nan to test this because all
 comparisons against nan return false.
 */
bool isNaN(double x);

/**
 Computes x % 3.
 */
int iMod3(int x);

/** [0, 1] */
double unitRandom ();

/**
 Uniform random number between low and hi, inclusive.
 */
double random(double low, double hi);

/** [-1, 1] */
double symmetricRandom ();
double min(double x, double y);
double max(double x, double y);
int iMin(int x, int y);
int iMax(int x, int y);

double square(double x);
double sumSquares(double x, double y);
double sumSquares(double x, double y, double z);
double distance(double x, double y);
double distance(double x, double y, double z);

/**
  Returnes the 0-based index of the highest 1 bit from
  the left.  -1 means the number was 0.

  @cite Based on code by jukka@liimatta.org
 */ 
int highestBit(uint32 x);

/**
 Note that fuzzyEq(a, b) && fuzzyEq(b, c) does not imply
 fuzzyEq(a, c), although that will be the case on some
 occasions.
 */
bool fuzzyEq(double a, double b);

bool fuzzyNe(double a, double b);

bool fuzzyGt(double a, double b);

bool fuzzyGe(double a, double b);

bool fuzzyLt(double a, double b);

bool fuzzyLe(double a, double b);

/**
 Computes 1 / sqrt(x) using SSE instructions for efficiency.
 @cite Nick nicolas@capens.net
 */
inline float rsq(float x) {
    return 1.0f / sqrt(x);
}

/**
 Uses SSE to implement rsq.
 */
inline float SSErsq(float x) {

    #ifdef SSE
        __asm {
           movss xmm0, x
           rsqrtss xmm0, xmm0
           movss x, xmm0
        }
        return x;
    #else
        return 1.0f / sqrt(x);
    #endif
}

/**
 Return the next power of 2 higher than the input
 If the input is already a power of 2, the output will be the same 
 as the input.
 */
int ceilPow2(unsigned int in);

/**
 * True if num is a power of two.
 */
bool isPow2(int num);

bool isOdd(int num);
bool isEven(int num);

double toRadians(double deg);
double toDegrees(double rad);


/**
 Interpolates a property according to a piecewise linear spline.  This provides
 C0 continuity but the derivatives are not smooth.  
 <P>
 Example:
 <CODE>
    const double times[] = {MIDNIGHT,               SUNRISE - HOUR,         SUNRISE,              SUNRISE + sunRiseAndSetTime / 4, SUNRISE + sunRiseAndSetTime, SUNSET - sunRiseAndSetTime, SUNSET - sunRiseAndSetTime / 2, SUNSET,               SUNSET + HOUR/2,     DAY};
    const Color3 color[] = {Color3(0, .0, .1),      Color3(0, .0, .1),      Color3::BLACK,        Color3::BLACK,                   Color3::WHITE * .25,         Color3::WHITE * .25,        Color3(.5, .2, .2),             Color3(.05, .05, .1),   Color3(0, .0, .1), Color3(0, .0, .1)};
    ambient = linearSpline(time, times, color, 10);
 </CODE>

  @param x         The spline is a function of x; this is the sample to choose.
  @param controlX  controlX[i], controlY[i] is a control points.  It is assumed
                   that controlX are strictly increasing.  XType must support
                   the "<" operator and a subtraction operator that returns
                   a number.
  @param controlY  YType must support multiplication and addition.
  @param numControl The number of control points.
 */
template<class XType, class YType>
YType linearSpline(double x, const XType* controlX, const YType* controlY, int numControl) {
    debugAssert(numControl >= 1);

    // Off the beginning
    if ((numControl == 1) || (x < controlX[0])) {
        return controlY[0];
    }

    for (int i = 1; i < numControl; ++i) {
        if (x < controlX[i]) {
            const double alpha = (double)(controlX[i] - x) / (controlX[i] - controlX[i - 1]);
            return controlY[i] * (1 - alpha) + controlY[i - 1] * alpha;
        }
    }

    // Off the end
    return controlY[numControl - 1];
}


/**
 Returns true if x is not exactly equal to 0.0f.
 */
inline bool any(float x) {
    return x != 0;
}

/**
 Returns true if x is not exactly equal to 0.0f.
 */
inline bool all(float x) {
    return x != 0;
}

/**
 v / v (for DirectX/Cg support)
 */
inline float normalize(float v) {
    return v / v;
}

/**
 a * b (for DirectX/Cg support)
 */
inline float dot(float a, float b) {
    return a * b;
}


/**
 a * b (for DirectX/Cg support)
 */
inline float mul(float a, float b) {
    return a * b;
}

/**
 2^x
 */
inline double exp2(double x) {
    return pow(2.0, x);
}

inline double rsqrt(double x) {
    return 1.0 / sqrt(x);
}


} // namespace


#include "g3dmath.inl"

#endif

