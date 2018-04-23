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


#include <math.h>
#include <string.h>

#include "fmath.h"
#include "filter.h"


// Below are the implementations of several 2D filters.  They all
// inherit their interface from Filter2D.  Each must redefine two
// virtual functions:
//    char *name()                     Return the filter name
//    float operator(float,float)      Evaluate the filter
//

class FilterBox2D : public Filter2D {
public:
    FilterBox2D (float width, float height) : Filter2D(width,height) { }
    ~FilterBox2D (void) { }
    float operator() (float x, float y) const {
        if (fabsf(x) <= w*0.5f && fabsf(y) <= h*0.5f)
            return 1.0f;
        else return 0.0f;
    }
    const char *name (void) const { return "box"; }
};



class FilterTriangle2D : public Filter2D {
public:
    FilterTriangle2D (float width, float height) : Filter2D(width,height) { }
    ~FilterTriangle2D (void) { }
    float operator() (float x, float y) const {
        return tri1d (x / (w*0.5f)) * tri1d (y / (h*0.5f));
    }
    const char *name (void) const { return "triangle"; }
private:
    static float tri1d (float x) {
        x = fabsf(x);
        return (x < 1.0f) ? (1.0f - x) : 0.0f;
    }
};



class FilterGaussian2D : public Filter2D {
public:
    FilterGaussian2D (float width, float height) : Filter2D(width,height) { }
    ~FilterGaussian2D (void) { }
    float operator() (float x, float y) const {
        x = 2.0f * fabsf(x) / w;
        y = 2.0f * fabsf(y) / h;
        return (x < 1.0 && y < 1.0) ? expf (-2.0f * (x*x+y*y)) : 0.0f;
    }
    const char *name (void) const { return "gaussian"; }
};



class FilterCatmullRom2D : public Filter2D {
public:
    FilterCatmullRom2D (float width, float height) : Filter2D(width,height) { }
    ~FilterCatmullRom2D (void) { }
    float operator() (float x, float y) const {
        return catrom1d(x) * catrom1d(y);
    }
    const char *name (void) const { return "catmull-rom"; }
private :
    static float catrom1d (float x) {
        x = fabsf(x);
        float x2 = x * x;
        float x3 = x * x2;
        return (x >= 2.0f) ? 0.0f :  ((x < 1.0f) ?
                                      (3.0f * x3 - 5.0f * x2 + 2.0f) :
                                      (-x3 + 5.0f * x2 - 8.0f * x + 4.0f) );
    }
};



class FilterBlackmanHarris2D : public Filter2D {
public:
    FilterBlackmanHarris2D (float width, float height) 
        : Filter2D(width,height) { }
    ~FilterBlackmanHarris2D (void) { }
    float operator() (float x, float y) const {
        return bh1d (x / (w*0.5f)) * bh1d (y / (h*0.5f));
    }
    const char *name (void) const { return "blackman-harris"; }
private:
    static float bh1d (float x) {
	if (x < -1.0f || x > 1.0f)  // Early out if outside filter range
            return 0.0f;
        // Compute BH.  Straight from classic BH paper, but the usual
        // formula assumes that the filter is centered at 0.5, so scale:
        x = (x + 1.0f) * 0.5f;
        const float A0 =  0.35875f;
        const float A1 = -0.48829f;
        const float A2 =  0.14128f;
        const float A3 = -0.01168f;
        const float m_pi = float (M_PI);
        return A0 + A1 * cosf(2.f * m_pi * x) 
             + A2 * cosf(4.f * m_pi * x) + A3 * cosf(6.f * m_pi * x);
    }
};



class FilterSinc2D : public Filter2D {
public:
    FilterSinc2D (float width, float height) : Filter2D(width,height) { }
    ~FilterSinc2D (void) { }
    float operator() (float x, float y) const {
        if (fabsf(x) > 0.5f*w || fabsf(y) > 0.5f*h)
             return 0.0f;
        else return sinc1d(x) * sinc1d(y);
    }
    const char *name (void) const { return "sinc"; }
private:
    static float sinc1d (float x) {
        x = float (fabsf(x));
        const float m_pi = float (M_PI);
        return (x < 0.0001f) ? 1.0f : sinf (m_pi*x)/(m_pi*x);
    }
};



class FilterMitchell2D : public Filter2D {
public:
    FilterMitchell2D (float width, float height) : Filter2D(width,height) { }
    ~FilterMitchell2D (void) { }
    float operator() (float x, float y) const {
        return mitchell1d (x / (w*0.5f)) * mitchell1d (y / (h*0.5f));
    }
    const char *name (void) const { return "mitchell"; }
private:
    static float mitchell1d (float x) {
        // Computation stright out of the classic Mitchell paper.
        // In the paper, the range is -2 to 2, so we rescale:
        if (x < -1.0f || x > 1.0f)
            return 0.0f;
        x = fabsf (2.0f * x);
        float x2 = x*x;
        const float B = 1.0f/3.0f;
        const float C = 1.0f/3.0f;
        const float SIXTH = 1.0f/6.0f;
        if (x >= 1.0f)
            return ((-B - 6.0f*C)*x*x2 + (6.0f*B + 30.0f*C)*x2 +
                    (-12.0f*B - 48.0f*C)*x + (8.0f*B + 24.0f*C)) * SIXTH;
        else
            return ((12.0f - 9.0f*B - 6.0f*C)*x*x2 + 
                    (-18.0f + 12.0f*B + 6.0f*C)*x2 + (6.0f - 2.0f*B)) * SIXTH;
    }
};



class FilterDisk2D : public Filter2D {
public:
    FilterDisk2D (float width, float height) : Filter2D(width,height) { }
    ~FilterDisk2D (void) { }
    float operator() (float x, float y) const {
        x /= (w*0.5f);
        y /= (h*0.5f);
        float x2 = x*x;
        float y2 = y*y;
        return (x2 < 1.0f && y2 < 1.0f) ? 1.0f : 0.0f;
    }
    const char *name (void) const { return "disk"; }
};



// Filter2D::MakeFilter is the static method that, given a filter name,
// width, and height, returns an allocated and instantiated filter of
// the correct implementation.  If the name is not recognized, return
// NULL.
Filter2D *
Filter2D::MakeFilter (const char *filtername, float width, float height)
{
    if (! strcmp (filtername, "box"))
        return new FilterBox2D (width, height);
    if (! strcmp (filtername, "triangle"))
        return new FilterTriangle2D (width, height);
    if (! strcmp (filtername, "gaussian"))
        return new FilterGaussian2D (width, height);
    if (! strcmp (filtername, "catmull-rom"))
        return new FilterCatmullRom2D (width, height);
    if (! strcmp (filtername, "blackman-harris"))
        return new FilterBlackmanHarris2D (width, height);
    if (! strcmp (filtername, "sinc"))
        return new FilterSinc2D (width, height);
    if (! strcmp (filtername, "mitchell"))
        return new FilterMitchell2D (width, height);
    if (! strcmp (filtername, "disk"))
        return new FilterDisk2D (width, height);
    return NULL;
}
