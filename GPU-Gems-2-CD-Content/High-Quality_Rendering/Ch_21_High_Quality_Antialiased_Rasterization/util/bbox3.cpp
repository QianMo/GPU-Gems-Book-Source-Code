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
#include <climits>
#include <iostream>

#include "vecmat.h"
#include "fmath.h"


namespace Gelato {



bool
operator== (const Bbox3 &a, const Bbox3 &b)
{
    for (int i = 0;  i < 6;  ++i)
        if (a[i] != b[i])
            return false;
    return true;
}



bool
operator!= (const Bbox3 &a, const Bbox3 &b)
{
    for (int i = 0;  i < 6;  ++i)
        if (a[i] != b[i])
            return true;
    return false;
}



std::ostream&
operator<< (std::ostream& out, const Bbox3& a)
{
    return (out << '[' << a[0] << ' ' << a[1] << ' ' << a[2]
            << ' ' << a[3] << ' ' << a[4] << ' ' << a[5] << ']');
}



bool
overlaps (const Bbox3& b1, const Bbox3& b2)
{
    // If any of {x,y,z} do not overlap in range, they don't overlap
    for (int i = 0;  i < 3;  ++i)
        if (b1[2*i+1] < b2[2*i] || b1[2*i] > b2[2*i+1])
            return false;
    return true;
}



float
Bbox3::volume (void) const
{
    // Use doubles in case of overflow of big bounds
    double x = b[1] - b[0];
    double y = b[3] - b[2];
    double z = b[5] - b[4];
    double vol = x * y * z;
    // Protect against overflow when we cast back to float
    if (vol > std::numeric_limits<float>::max())
        vol = std::numeric_limits<float>::max();
    return (float)vol;
}



float
Bbox3::area (void) const
{
    // Use doubles in case of overflow of big bounds
    double x = b[1] - b[0];
    double y = b[3] - b[2];
    double z = b[5] - b[4];
    double area = (2.0 * (x * (y+z) + y*z));
    // Protect against overflow when we cast back to float
    if (area > std::numeric_limits<float>::max())
        area = std::numeric_limits<float>::max();
    return (float)area;
}



void
Bbox3::getcorners (Vector3 *corners) const
{
    for (int i = 0;  i < 8;  ++i)
        corners[i] = corner(i);
}



void
btransform (const Matrix4& M, const Bbox3 &b, Bbox3 &result)
{
    Bbox3 t(b);
    result = Bbox3::Empty();
    for (int i = 0;  i < 8;  ++i) {
        Vector3 P = t.corner(i);
        transformp (M, P);
        result |= P;
    }
}



// Merge p[0..n-1] into our bounding box.
void
Bbox3::bound_more_points (int n, const Vector3 *p)
{
    assert (p != NULL);
    for (int i = 0;  i < n;  ++i)
        *this |= p[i];
}



// Set this bound to the smallest bound containing p[0..n-1]
void
Bbox3::bound_points (int n, const Vector3 *p)
{
    assert (p != NULL && n > 0);
    *this = p[0];
    bound_more_points (n-1, p+1);
}

};  /* end namespace Gelato */
