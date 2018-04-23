/*
s_p_oneil@hotmail.com
Copyright (c) 2000, Sean O'Neil
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of this project nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include "Master.h"
#include "Matrix.h"


void CMatrix::operator=(const CQuaternion &q)
{
	// 9 muls, 15 adds
	float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
	float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
	float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
	float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;

	f14 = f24 = f34 = f41 = f42 = f43 = 0; f44 = 1;
	f11 = 1-(yy+zz);	f21 = xy-wz;		f31 = xz+wy;
	f12 = xy+wz;		f22 = 1-(xx+zz);	f32 = yz-wx;
	f13 = xz-wy;		f23 = yz+wx;		f33 = 1-(xx+yy);
}

CMatrix CMatrix::operator*(const CMatrix &m) const
{
		// 36 muls, 27 adds
		// | f11 f21 f31 f41 |   | m.f11 m.f21 m.f31 m.f41 |   | f11*m.f11+f21*m.f12+f31*m.f13 f11*m.f21+f21*m.f22+f31*m.f23 f11*m.f31+f21*m.f32+f31*m.f33 f11*m.f41+f21*m.f42+f31*m.f43+f41 |
		// | f12 f22 f32 f42 |   | m.f12 m.f22 m.f32 m.f42 |   | f12*m.f11+f22*m.f12+f32*m.f13 f12*m.f21+f22*m.f22+f32*m.f23 f12*m.f31+f22*m.f32+f32*m.f33 f12*m.f41+f22*m.f42+f32*m.f43+f42 |
		// | f13 f23 f33 f43 | * | m.f13 m.f23 m.f33 m.f43 | = | f13*m.f11+f23*m.f12+f33*m.f13 f13*m.f21+f23*m.f22+f33*m.f23 f13*m.f31+f23*m.f32+f33*m.f33 f13*m.f41+f23*m.f42+f33*m.f43+f43 |
		// | 0   0   0   1   |   | 0     0     0     1     |   | 0                             0                             0                             1                                 |
		CMatrix mRet;
		mRet.f11 = f11*m.f11+f21*m.f12+f31*m.f13;
		mRet.f21 = f11*m.f21+f21*m.f22+f31*m.f23;
		mRet.f31 = f11*m.f31+f21*m.f32+f31*m.f33;
		mRet.f41 = f11*m.f41+f21*m.f42+f31*m.f43+f41;
		mRet.f12 = f12*m.f11+f22*m.f12+f32*m.f13;
		mRet.f22 = f12*m.f21+f22*m.f22+f32*m.f23;
		mRet.f32 = f12*m.f31+f22*m.f32+f32*m.f33;
		mRet.f42 = f12*m.f41+f22*m.f42+f32*m.f43+f42;
		mRet.f13 = f13*m.f11+f23*m.f12+f33*m.f13;
		mRet.f23 = f13*m.f21+f23*m.f22+f33*m.f23;
		mRet.f33 = f13*m.f31+f23*m.f32+f33*m.f33;
		mRet.f43 = f13*m.f41+f23*m.f42+f33*m.f43+f43;
		mRet.f14 = mRet.f24 = mRet.f34 = 0;
		mRet.f44 = 1;
		return mRet;
}

void CQuaternion::operator=(const CMatrix &m)
{
	// Check the sum of the diagonal
	float tr = m(0, 0) + m(1, 1) + m(2, 2);
	if(tr > 0.0f)
	{
		// The sum is positive
		// 4 muls, 1 div, 6 adds, 1 trig function call
		float s = sqrtf(tr + 1.0f);
		w = s * 0.5f;
		s = 0.5f / s;
		x = (m(1, 2) - m(2, 1)) * s;
		y = (m(2, 0) - m(0, 2)) * s;
		z = (m(0, 1) - m(1, 0)) * s;
	}
	else
	{
		// The sum is negative
		// 4 muls, 1 div, 8 adds, 1 trig function call
		const int nIndex[3] = {1, 2, 0};
		int i, j, k;
		i = 0;
		if(m(1, 1) > m(i, i))
			i = 1;
		if(m(2, 2) > m(i, i))
			i = 2;
		j = nIndex[i];
		k = nIndex[j];

		float s = sqrtf((m(i, i) - (m(j, j) + m(k, k))) + 1.0f);
		(*this)[i] = s * 0.5f;
		if(s != 0.0)
			s = 0.5f / s;
		(*this)[j] = (m(i, j) + m(j, i)) * s;
		(*this)[k] = (m(i, k) + m(k, i)) * s;
		(*this)[3] = (m(j, k) - m(k, j)) * s;
	}
}

CQuaternion CQuaternion::operator*(const CQuaternion &q) const
{
	// 12 muls, 30 adds
	float E = (x + z)*(q.x + q.y);
	float F = (z - x)*(q.x - q.y);
	float G = (w + y)*(q.w - q.z);
	float H = (w - y)*(q.w + q.z);
	float A = F - E;
	float B = F + E;
	return CQuaternion(
		(w + x)*(q.w + q.x) + (A - G - H) * 0.5f,
		(w - x)*(q.y + q.z) + (B + G - H) * 0.5f,
		(y + z)*(q.w - q.x) + (B - G + H) * 0.5f,
		(z - y)*(q.y - q.z) + (A + G + H) * 0.5f);
}

// Spherical linear interpolation between two quaternions
CQuaternion Slerp(const CQuaternion &q1, const CQuaternion &q2, const float t)
{
	// Calculate the cosine of the angle between the two
	float fScale0, fScale1;
	double dCos = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

	// If the angle is significant, use the spherical interpolation
	if((1.0 - ABS(dCos)) > DELTA)
	{
		double dTemp = acos(ABS(dCos));
		double dSin = sin(dTemp);
		fScale0 = (float)(sin((1.0 - t) * dTemp) / dSin);
		fScale1 = (float)(sin(t * dTemp) / dSin);
	}
	// Else use the cheaper linear interpolation
	else
	{
		fScale0 = 1.0f - t;
		fScale1 = t;
	}
	if(dCos < 0.0)
		fScale1 = -fScale1;

	// Return the interpolated result
	return (q1 * fScale0) + (q2 * fScale1);
}
