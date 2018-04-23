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

#ifndef __Matrix_h__
#define __Matrix_h__

#include "Noise.h"	// Has some useful defines and inline functions

class CMatrix;
class CQuaternion;
class C3DObject;

/*******************************************************************************
* Template Class: TVector
********************************************************************************
* This template class implements a simple 3D vector with x, y, and z coordinates.
* Several functions and operators are defined to make working with vectors easier,
* and because it's templatized, any numeric type can be used with it. Macros are
* defined for the most common types.
*******************************************************************************/
#define CVector			TVector<float>
#define CDoubleVector	TVector<double>
#define CIntVector		TVector<int>
#define CByteVector		TVector<unsigned char>
template <class T> class TVector
{
public:
	T x, y, z;

	// Constructors
	TVector()									{}
	TVector(const T a, const T b, const T c)	{ x = a; y = b; z = c; }
	TVector(const T t)							{ *this = t; }
	TVector(const T *pt)						{ *this = pt; }
	TVector(const TVector<T> &v)				{ *this = v; }

	// Casting and unary operators
	operator TVector<float>()					{ return TVector<float>((float)x, (float)y, (float)z); }
	operator TVector<double>()					{ return TVector<double>((double)x, (double)y, (double)z); }
	operator T*()								{ return &x; }
	T &operator[](const int n)					{ return (&x)[n]; }
	operator const T*() const					{ return &x; }
	T operator[](const int n) const				{ return (&x)[n]; }
	TVector<T> operator-() const				{ return TVector<T>(-x, -y, -z); }

	// Equal and comparison operators
	void operator=(const T t)					{ x = y = z = t; }
	void operator=(const T *pt)					{ x = pt[0]; y = pt[1]; z = pt[2]; }
	void operator=(const TVector<T> &v)			{ x = v.x; y = v.y; z = v.z; }
	bool operator==(TVector<T> &v) const		{ return (Abs(x - v.x) <= (T)0.001f && Abs(y - v.y) <= 0.001f && Abs(z - v.z) <= 0.001f); }
	bool operator!=(TVector<T> &v) const		{ return !(*this == v); }

	// Arithmetic operators (vector with scalar)
	TVector<T> operator+(const T t) const		{ return TVector<T>(x+t, y+t, z+t); }
	TVector<T> operator-(const T t) const		{ return TVector<T>(x-t, y-t, z-t); }
	TVector<T> operator*(const T t) const		{ return TVector<T>(x*t, y*t, z*t); }
	TVector<T> operator/(const T t) const		{ return TVector<T>(x/t, y/t, z/t); }
	void operator+=(const T t)					{ x += t; y += t; z += t; }
	void operator-=(const T t)					{ x -= t; y -= t; z -= t; }
	void operator*=(const T t)					{ x *= t; y *= t; z *= t; }
	void operator/=(const T t)					{ x /= t; y /= t; z /= t; }

	// Arithmetic operators (vector with vector)
	TVector<T> operator+(const TVector<T> &v) const	{ return TVector<T>(x+v.x, y+v.y, z+v.z); }
	TVector<T> operator-(const TVector<T> &v) const	{ return TVector<T>(x-v.x, y-v.y, z-v.z); }
	TVector<T> operator*(const TVector<T> &v) const	{ return TVector<T>(x*v.x, y*v.y, z*v.z); }
	TVector<T> operator/(const TVector<T> &v) const	{ return TVector<T>(x/v.x, y/v.y, z/v.z); }
	void operator+=(const TVector<T> &v)		{ x += v.x; y += v.y; z += v.z; }
	void operator-=(const TVector<T> &v)		{ x -= v.x; y -= v.y; z -= v.z; }
	void operator*=(const TVector<T> &v)		{ x *= v.x; y *= v.y; z *= v.z; }
	void operator/=(const TVector<T> &v)		{ x /= v.x; y /= v.y; z /= v.z; }

	// Dot and cross product operators
	T operator|(const TVector<T> &v) const		{ return x*v.x + y*v.y + z*v.z; }
	TVector<T> operator^(const TVector<T> &v) const	{ return TVector<T>(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x); }
	void operator^=(const TVector<T> &v)		{ *this = *this ^ v; }

	// Magnitude/distance methods
	T MagnitudeSquared() const					{ return x*x + y*y + z*z; }
	T Magnitude() const							{ return (T)sqrtf((float)MagnitudeSquared()); }
	T DistanceSquared(const TVector<T> &v) const{ return (*this - v).MagnitudeSquared(); }
	T Distance(const TVector<T> &v) const		{ return (*this - v).Magnitude(); }
	TVector<T> Midpoint(const TVector<T> &v) const	{ return CVector((*this - v) / 2 + v); }
	TVector<T> Average(const TVector<T> &v) const	{ return CVector((*this + v) / 2); }

	// Advanced methods (should only be used with float or double types)
	void Normalize()							{ *this /= Magnitude(); }
	double Angle(const TVector<T> &v) const		{ return acos(*this | v); }
	TVector<T> Reflect(const TVector<T> &n) const
	{
		T t = (T)Magnitude();
		TVector<T> v = *this / t;
		return (v - n * (2 * (v | n))) * t;
	}
	TVector<T> Rotate(const T tAngle, const CVector &n) const
	{
		T tCos = (T)cos(tAngle);
		T tSin = (T)sin(tAngle);
		return TVector<T>(*this * tCos + ((n * *this) * (1 - tCos)) * n + (*this ^ n) * tSin);
	}
};

// Returns the normal vector of two vectors (the normalized cross product)
template <class T> inline TVector<T> NormalVector(const TVector<T> &v1, const TVector<T> &v2)
{
	TVector<T> v = v1 ^ v2;
	v.Normalize();
	return v;
}

// Returns the normal vector of a triangle or 3 points on a plane (assumes counter-clockwise orientation)
template <class T> inline TVector<T> NormalVector(const TVector<T> &p1, const TVector<T> &p2, const TVector<T> &p3)
{
	return NormalVector(p2-p1, p3-p1);
}

// Returns the direction vector between two points
template <class T> inline TVector<T> DirectionVector(const TVector<T> &p1, const TVector<T> &p2)
{
	TVector<T> v = p2 - p1;
	v.Normalize();
	return v;
}

template <class T> inline float dot(const TVector<T> &v1, const TVector<T> &v2)
{
	return v1 | v2;
}

template <class T> inline float length(const TVector<T> &v1)
{
	return v1.Magnitude();
}


/*******************************************************************************
* Template Class: TVector4
********************************************************************************
* This template class implements a simple 4D vector with x, y, z, and w
* coordinates. Like TVector, it is templatized and macros are defined for the
* most common types.
*******************************************************************************/
#define CVector4		TVector4<float>
#define CDoubleVector4	TVector4<double>
#define CIntVector4		TVector4<int>
#define CByteVector4	TVector4<unsigned char>
template <class T> class TVector4
{
public:
	T x, y, z, w;

	// Constructors
	TVector4()									{}
	TVector4(const T a, const T b, const T c, const T d)	{ x = a; y = b; z = c; w = d; }
	TVector4(const T t)							{ *this = t; }
	TVector4(const T *pt)						{ *this = pt; }
	TVector4(const TVector<T> &v)				{ *this = v; }
	TVector4(const TVector4<T> &v)				{ *this = v; }

	// Casting and unary operators
	operator T*()								{ return &x; }
	T &operator[](const int n)					{ return (&x)[n]; }
	operator const T*() const					{ return &x; }
	T operator[](const int n) const				{ return (&x)[n]; }
	TVector4<T> operator-() const				{ return TVector4<T>(-x, -y, -z, -w); }

	// Equal and comparison operators
	void operator=(const T t)					{ x = y = z = w = t; }
	void operator=(const T *pt)					{ x = pt[0]; y = pt[1]; z = pt[2]; w = pt[3]; }
	void operator=(const TVector<T> &v)			{ x = v.x; y = v.y; z = v.z; w = 0; }
	void operator=(const TVector4<T> &v)		{ x = v.x; y = v.y; z = v.z; w = v.w; }
	bool operator==(TVector4<T> &v) const		{ return (Abs(x - v.x) <= (T)DELTA && Abs(y - v.y) <= (T)DELTA && Abs(z - v.z) <= (T)DELTA && Abs(w - v.w) <= (T)DELTA); }
	bool operator!=(TVector4<T> &v) const		{ return !(*this == v); }

	// Arithmetic operators (vector with scalar)
	TVector4<T> operator+(const T t) const		{ return TVector4<T>(x+t, y+t, z+t, w+t); }
	TVector4<T> operator-(const T t) const		{ return TVector4<T>(x-t, y-t, z-t, w-t); }
	TVector4<T> operator*(const T t) const		{ return TVector4<T>(x*t, y*t, z*t, w*t); }
	TVector4<T> operator/(const T t) const		{ return TVector4<T>(x/t, y/t, z/t, w/t); }
	void operator+=(const T t)					{ x += t; y += t; z += t; w += t; }
	void operator-=(const T t)					{ x -= t; y -= t; z -= t; w -= t; }
	void operator*=(const T t)					{ x *= t; y *= t; z *= t; w *= t; }
	void operator/=(const T t)					{ x /= t; y /= t; z /= t; w /= t; }

	// Arithmetic operators (vector with vector)
	TVector4<T> operator+(const TVector4<T> &v) const	{ return TVector4<T>(x+v.x, y+v.y, z+v.z, w+v.w); }
	TVector4<T> operator-(const TVector4<T> &v) const	{ return TVector4<T>(x-v.x, y-v.y, z-v.z, w-v.w); }
	TVector4<T> operator*(const TVector4<T> &v) const	{ return TVector4<T>(x*v.x, y*v.y, z*v.z, w*v.w); }
	TVector4<T> operator/(const TVector4<T> &v) const	{ return TVector4<T>(x/v.x, y/v.y, z/v.z, w/v.w); }
	void operator+=(const TVector4<T> &v)		{ x += v.x; y += v.y; z += v.z; w += v.w; }
	void operator-=(const TVector4<T> &v)		{ x -= v.x; y -= v.y; z -= v.z; w -= v.w; }
	void operator*=(const TVector4<T> &v)		{ x *= v.x; y *= v.y; z *= v.z; w *= v.w; }
	void operator/=(const TVector4<T> &v)		{ x /= v.x; y /= v.y; z /= v.z; w /= v.w; }

	// Magnitude/normalize methods
	T MagnitudeSquared() const					{ return x*x + y*y + z*z + w*w; }
	T Magnitude() const							{ return (T)sqrt(MagnitudeSquared()); }
	void Normalize()							{ *this /= Magnitude(); }
};


/*******************************************************************************
* Class: CQuaternion
********************************************************************************
* This class implements a 4D quaternion. Several functions and operators are
* defined to make working with quaternions easier. Quaternions are often used to
* represent rotations, and have a number of advantages over other constructs.
* Their main disadvantage is that they are unintuitive.
*
* Note: This class is not templatized because integral data types don't make sense
*       and there's no need for double-precision.
*******************************************************************************/
class CQuaternion
{
public:
	float x, y, z, w;

	// Constructors
	CQuaternion()								{}
	CQuaternion(const float a, const float b, const float c, const float d)	{ x = a; y = b; z = c; w = d; }
	CQuaternion(const CVector &v, const float f){ SetAxisAngle(v, f); }
	CQuaternion(const CVector &v)				{ *this = v; }
	CQuaternion(const CQuaternion &q)			{ *this = q; }
	CQuaternion(const CMatrix &m)				{ *this = m; }
	CQuaternion(const float *p)					{ *this = p; }

	// Casting and unary operators
	operator float*()							{ return &x; }
	float &operator[](const int n)				{ return (&x)[n]; }
	operator const float*() const				{ return &x; }
	float operator[](const int n) const			{ return (&x)[n]; }
	CQuaternion operator-() const				{ return CQuaternion(-x, -y, -z, -w); }

	// Equal and comparison operators
	void operator=(const CVector &v)			{ x = v.x; y = v.y; z = v.z; w = 0; }
	void operator=(const CQuaternion &q)		{ x = q.x; y = q.y; z = q.z; w = q.w; }
	void operator=(const CMatrix &m);
	void operator=(const float *p)				{ x = p[0]; y = p[1]; z = p[2]; w = p[3]; }

	// Arithmetic operators (quaternion and scalar)
	CQuaternion operator+(const float f) const	{ return CQuaternion(x+f, y+f, z+f, w+f); }
	CQuaternion operator-(const float f) const	{ return CQuaternion(x-f, y-f, z-f, w-f); }
	CQuaternion operator*(const float f) const	{ return CQuaternion(x*f, y*f, z*f, w*f); }
	CQuaternion operator/(const float f) const	{ return CQuaternion(x/f, y/f, z/f, w/f); }
	void operator+=(const float f)				{ x+=f; y+=f; z+=f; w+=f; }
	void operator-=(const float f)				{ x-=f; y-=f; z-=f; w-=f; }
	void operator*=(const float f)				{ x*=f; y*=f; z*=f; w*=f; }
	void operator/=(const float f)				{ x/=f; y/=f; z/=f; w/=f; }

	// Arithmetic operators (quaternion and quaternion)
	CQuaternion operator+(const CQuaternion &q) const	{ return CQuaternion(x+q.x, y+q.y, z+q.z, w+q.w); }
	CQuaternion operator-(const CQuaternion &q) const	{ return CQuaternion(x-q.x, y-q.y, z-q.z, w-q.w); }
	CQuaternion operator*(const CQuaternion &q) const;	// Multiplying quaternions is a special operation
	void operator+=(const CQuaternion &q)	{ x+=q.x; y+=q.y; z+=q.z; w+=q.w; }
	void operator-=(const CQuaternion &q)	{ x-=q.x; y-=q.y; z-=q.z; w-=q.w; }
	void operator*=(const CQuaternion &q)	{ *this = *this * q; }

	// Magnitude/normalize methods
	float MagnitudeSquared() const			{ return x*x + y*y + z*z + w*w; }
	float Magnitude() const					{ return sqrtf(MagnitudeSquared()); }
	void Normalize()						{ *this /= Magnitude(); }

	// Advanced quaternion methods
	CQuaternion Conjugate() const			{ return CQuaternion(-x, -y, -z, w); }
	CQuaternion Inverse() const				{ return Conjugate() / MagnitudeSquared(); }
	CQuaternion UnitInverse() const			{ return Conjugate(); }
	CVector RotateVector(const CVector &v) const	{ return (*this * CQuaternion(v) * UnitInverse()); }

	void SetAxisAngle(const CVector &vAxis, const float fAngle)
	{
		// 4 muls, 2 trig function calls
		float f = fAngle * 0.5f;
		*this = vAxis * sinf(f);
		w = cosf(f);
	}
	void GetAxisAngle(CVector &vAxis, float &fAngle) const
	{
		// 4 muls, 1 div, 2 trig function calls
		fAngle = acosf(w);
		vAxis = *this / sinf(fAngle);
		fAngle *= 2.0f;
	}

	void Rotate(const CQuaternion &q)			{ *this = q * *this; }
	void Rotate(const CVector &vAxis, const float fAngle)
	{
		CQuaternion q;
		q.SetAxisAngle(vAxis, fAngle);
		Rotate(q);
	}
	void Rotate(const CVector &vFrom, const CVector &vTo)
	{
		CVector vAxis = vFrom ^ vTo;
		vAxis.Normalize();
		float fCos = vFrom | vTo;
		Rotate(vAxis, acosf(fCos));
	}

	CVector GetViewAxis() const
	{
		// 6 muls, 7 adds
		float x2 = x + x, y2 = y + y, z2 = z + z;
		float xx = x * x2, xz = x * z2;
		float yy = y * y2, yz = y * z2;
		float wx = w * x2, wy = w * y2;
		return -CVector(xz+wy, yz-wx, 1-(xx+yy));
	}
	CVector GetUpAxis() const
	{
		// 6 muls, 7 adds
		float x2 = x + x, y2 = y + y, z2 = z + z;
		float xx = x * x2, xy = x * y2;
		float yz = y * z2, zz = z * z2;
		float wx = w * x2, wz = w * z2;
		return CVector(xy-wz, 1-(xx+zz), yz+wx);
	}
	CVector GetRightAxis() const
	{
		// 6 muls, 7 adds
		float x2 = x + x, y2 = y + y, z2 = z + z;
		float xy = x * y2, xz = x * z2;
		float yy = y * y2, zz = z * z2;
		float wy = w * y2, wz = w * z2;
		return CVector(1-(yy+zz), xy+wz, xz-wy);
	}
};

extern CQuaternion Slerp(const CQuaternion &q1, const CQuaternion &q2, const float t);

/*******************************************************************************
* Class: CMatrix
********************************************************************************
* This class implements a 4x4 matrix. Several functions and operators are
* defined to make working with matrices easier. The values are kept in column-
* major order to make it easier to use with OpenGL. For performance reasons,
* most of the functions assume that all matrices are orthogonal, which means the
* bottom row is [ 0 0 0 1 ]. Since I plan to use the GL_PROJECTION matrix to
* handle the projection matrix, I should never need to use any other kind of
* matrix, and I get a decent performance boost by ignoring the bottom row.
*
* Note: This class is not templatized because integral data types don't make sense
*       and there's no need for double-precision.
*******************************************************************************/
class CMatrix
{
public:
	// This class uses column-major order, as used by OpenGL
	// Here are the ways in which the matrix values can be accessed:
	// | f11 f21 f31 f41 |   | f1[0] f1[4] f1[8]  f1[12] |   | f2[0][0] f2[1][0] f2[2][0] f2[3][0] |
	// | f12 f22 f32 f42 |   | f1[1] f1[5] f1[9]  f1[13] |   | f2[0][1] f2[1][1] f2[2][1] f2[3][1] |
	// | f13 f23 f33 f43 | = | f1[2] f1[6] f1[10] f1[14] | = | f2[0][2] f2[1][2] f2[2][2] f2[3][2] |
	// | f14 f24 f34 f44 |   | f1[3] f1[7] f1[11] f1[15] |   | f2[0][3] f2[1][3] f2[2][3] f2[3][3] |
	union
	{
		struct { float f11, f12, f13, f14, f21, f22, f23, f24, f31, f32, f33, f34, f41, f42, f43, f44; };
		float f1[16];
		float f2[4][4];
	};

	CMatrix()							{}
	CMatrix(const float f)				{ *this = f; }
	CMatrix(const float *pf)			{ *this = pf; }
	CMatrix(const CQuaternion &q)		{ *this = q; }

	// Init functions
	void ZeroMatrix()
	{
		f11 = f12 = f13 = f14 = f21 = f22 = f23 = f24 = f31 = f32 = f33 = f34 = f41 = f42 = f43 = f44 = 0;
	}
	void IdentityMatrix()
	{
		f12 = f13 = f14 = f21 = f23 = f24 = f31 = f32 = f34 = f41 = f42 = f43 = 0;
		f11 = f22 = f33 = f44 = 1;
	}

	operator float*()								{ return f1; }
	float &operator[](const int n)					{ return f1[n]; }
	float &operator()(const int i, const int j)		{ return f2[i][j]; }
	operator const float*() const					{ return f1; }
	float operator[](const int n) const				{ return f1[n]; }
	float operator()(const int i, const int j) const{ return f2[i][j]; }

	void operator=(const float f)					{ for(register int i=0; i<16; i++) f1[i] = f; }
	void operator=(const float *pf)					{ for(register int i=0; i<16; i++) f1[i] = pf[i]; }
	void operator=(const CQuaternion &q);

	CMatrix operator*(const CMatrix &m) const;
	void operator*=(const CMatrix &m)				{ *this = *this * m; }
	CVector operator*(const CVector &v) const		{ return TransformVector(v); }

	CVector TransformVector(const CVector &v) const
	{
		// 9 muls, 9 adds
		// | f11 f21 f31 f41 |   | v.x |   | f11*v.x+f21*v.y+f31*v.z+f41 |
		// | f12 f22 f32 f42 |   | v.y |   | f12*v.x+f22*v.y+f32*v.z+f42 |
		// | f13 f23 f33 f43 | * | v.z | = | f13*v.x+f23*v.y+f33*v.z+f43 |
		// | 0   0   0   1   |   | 1   |   | 1                           |
		return CVector((f11*v.x+f21*v.y+f31*v.z+f41),
					   (f12*v.x+f22*v.y+f32*v.z+f42),
					   (f13*v.x+f23*v.y+f33*v.z+f43));
	}
	CVector TransformNormal(const CVector &v) const
	{
		// 9 muls, 6 adds
		// (Transpose rotation vectors, ignore position)
		// | f11 f12 f13 0 |   | v.x |   | f11*v.x+f12*v.y+f13*v.z |
		// | f21 f22 f23 0 |   | v.y |   | f21*v.x+f22*v.y+f23*v.z |
		// | f31 f32 f33 0 | * | v.z | = | f31*v.x+f32*v.y+f33*v.z |
		// | 0   0   0   1 |   | 1   |   | 1                       |
		return CVector((f11*v.x+f12*v.y+f13*v.z),
					   (f21*v.x+f22*v.y+f23*v.z),
					   (f31*v.x+f32*v.y+f33*v.z));
	}

	// Translate functions
	void TranslateMatrix(const float x, const float y, const float z)
	{
		// | 1  0  0  x |
		// | 0  1  0  y |
		// | 0  0  1  z |
		// | 0  0  0  1 |
		f12 = f13 = f14 = f21 = f23 = f24 = f31 = f32 = f34 = 0;
		f11 = f22 = f33 = f44 = 1;
		f41 = x; f42 = y; f43 = z;
	}
	void TranslateMatrix(const float *pf)		{ TranslateMatrix(pf[0], pf[1], pf[2]); }
	void Translate(const float x, const float y, const float z)
	{
		// 9 muls, 9 adds
		// | f11 f21 f31 f41 |   | 1  0  0  x |   | f11 f21 f31 f11*x+f21*y+f31*z+f41 |
		// | f12 f22 f32 f42 |   | 0  1  0  y |   | f12 f22 f32 f12*x+f22*y+f32*z+f42 |
		// | f13 f23 f33 f43 | * | 0  0  1  z | = | f13 f23 f33 f13*x+f23*y+f33*z+f43 |
		// | 0   0   0   1   |   | 0  0  0  1 |   | 0   0   0   1                     |
		f41 = f11*x+f21*y+f31*z+f41;
		f42 = f12*x+f22*y+f32*z+f42;
		f43 = f13*x+f23*y+f33*z+f43;
	}
	void Translate(const float *pf)				{ Translate(pf[0], pf[1], pf[2]); }

	// Scale functions
	void ScaleMatrix(const float x, const float y, const float z)
	{
		// | x  0  0  0 |
		// | 0  y  0  0 |
		// | 0  0  z  0 |
		// | 0  0  0  1 |
		f12 = f13 = f14 = f21 = f23 = f24 = f31 = f32 = f34 = f41 = f42 = f43 = 0;
		f11 = x; f22 = y; f33 = z; f44 = 1;
	}
	void ScaleMatrix(const float *pf)			{ ScaleMatrix(pf[0], pf[1], pf[2]); }
	void Scale(const float x, const float y, const float z)
	{
		// 9 muls
		// | f11 f21 f31 f41 |   | x  0  0  0 |   | f11*x f21*y f31*z f41 |
		// | f12 f22 f32 f42 |   | 0  y  0  0 |   | f12*x f22*y f32*z f42 |
		// | f13 f23 f33 f43 | * | 0  0  z  0 | = | f13*x f23*y f33*z f43 |
		// | 0   0   0   1   |   | 0  0  0  1 |   | 0     0     0     1   |
		f11 *= x; f21 *= y; f31 *= z;
		f12 *= x; f22 *= y; f32 *= z;
		f13 *= x; f23 *= y; f33 *= z;
	}
	void Scale(const float *pf)					{ Scale(pf[0], pf[1], pf[2]); }

	// Rotate functions
	void RotateXMatrix(const float fRadians)
	{
		// | 1 0    0     0 |
		// | 0 fCos -fSin 0 |
		// | 0 fSin fCos  0 |
		// | 0 0    0     1 |
		f12 = f13 = f14 = f21 = f24 = f31 = f34 = f41 = f42 = f43 = 0;
		f11 = f44 = 1;

		float fCos = cosf(fRadians);
		float fSin = sinf(fRadians);
		f22 = f33 = fCos;
		f23 = fSin;
		f32 = -fSin;
	}
	void RotateX(const float fRadians)
	{
		// 12 muls, 6 adds, 2 trig function calls
		// | f11 f21 f31 f41 |   | 1 0    0     0 |   | f11 f21*fCos+f31*fSin f31*fCos-f21*fSin f41 |
		// | f12 f22 f32 f42 |   | 0 fCos -fSin 0 |   | f12 f22*fCos+f32*fSin f32*fCos-f22*fSin f42 |
		// | f13 f23 f33 f43 | * | 0 fSin fCos  0 | = | f13 f23*fCos+f33*fSin f33*fCos-f23*fSin f43 |
		// | 0   0   0   1   |   | 0 0    0     1 |   | 0   0                 0                 1   |
		float fTemp, fCos, fSin;
		fCos = cosf(fRadians);
		fSin = sinf(fRadians);
		fTemp = f21*fCos+f31*fSin;
		f31 = f31*fCos-f21*fSin;
		f21 = fTemp;
		fTemp = f22*fCos+f32*fSin;
		f32 = f32*fCos-f22*fSin;
		f22 = fTemp;
		fTemp = f23*fCos+f33*fSin;
		f33 = f33*fCos-f23*fSin;
		f23 = fTemp;
	}
	void RotateYMatrix(const float fRadians)
	{
		// | fCos  0 fSin  0 |
		// | 0     1 0     0 |
		// | -fSin 0 fCos  0 |
		// | 0     0 0     1 |
		f12 = f14 = f21 = f23 = f24 = f32 = f34 = f41 = f42 = f43 = 0;
		f22 = f44 = 1;

		float fCos = cosf(fRadians);
		float fSin = sinf(fRadians);
		f11 = f33 = fCos;
		f13 = -fSin;
		f31 = fSin;
	}
	void RotateY(const float fRadians)
	{
		// 12 muls, 6 adds, 2 trig function calls
		// | f11 f21 f31 f41 |   | fCos  0 fSin  0 |   | f11*fCos-f31*fSin f21 f11*fSin+f31*fCos f41 |
		// | f12 f22 f32 f42 |   | 0     1 0     0 |   | f12*fCos-f32*fSin f22 f12*fSin+f32*fCos f42 |
		// | f13 f23 f33 f43 | * | -fSin 0 fCos  0 | = | f13*fCos-f33*fSin f23 f13*fSin+f33*fCos f43 |
		// | 0   0   0   1   |   | 0     0 0     1 |   | 0                 0   0                 1   |
		float fTemp, fCos, fSin;
		fCos = cosf(fRadians);
		fSin = sinf(fRadians);
		fTemp = f11*fCos-f31*fSin;
		f31 = f11*fSin+f31*fCos;
		f11 = fTemp;
		fTemp = f12*fCos-f32*fSin;
		f32 = f12*fSin+f32*fCos;
		f12 = fTemp;
		fTemp = f13*fCos-f33*fSin;
		f33 = f13*fSin+f33*fCos;
		f13 = fTemp;
	}
	void RotateZMatrix(const float fRadians)
	{
		// | fCos -fSin 0 0 |
		// | fSin fCos  0 0 |
		// | 0    0     1 0 |
		// | 0    0     0 1 |
		f13 = f14 = f23 = f24 = f31 = f32 = f34 = f41 = f42 = f43 = 0;
		f33 = f44 = 1;

		float fCos = cosf(fRadians);
		float fSin = sinf(fRadians);
		f11 = f22 = fCos;
		f12 = fSin;
		f21 = -fSin;
	}
	void RotateZ(const float fRadians)
	{
		// 12 muls, 6 adds, 2 trig function calls
		// | f11 f21 f31 f41 |   | fCos -fSin 0 0 |   | f11*fCos+f21*fSin f21*fCos-f11*fSin f31 f41 |
		// | f12 f22 f32 f42 |   | fSin fCos  0 0 |   | f12*fCos+f22*fSin f22*fCos-f12*fSin f32 f42 |
		// | f13 f23 f33 f43 | * | 0    0     1 0 | = | f13*fCos+f23*fSin f23*fCos-f13*fSin f33 f43 |
		// | 0   0   0   1   |   | 0    0     0 1 |   | 0                 0                 0   1   |
		float fTemp, fCos, fSin;
		fCos = cosf(fRadians);
		fSin = sinf(fRadians);
		fTemp = f11*fCos+f21*fSin;
		f21 = f21*fCos-f11*fSin;
		f11 = fTemp;
		fTemp = f12*fCos+f22*fSin;
		f22 = f22*fCos-f12*fSin;
		f12 = fTemp;
		fTemp = f13*fCos+f23*fSin;
		f23 = f23*fCos-f13*fSin;
		f13 = fTemp;
	}
	void RotateMatrix(const CVector &v, const float f)
	{
		// 15 muls, 10 adds, 2 trig function calls
		float fCos = cosf(f);
		CVector vCos = v * (1 - fCos);
		CVector vSin = v * sinf(f);

		f14 = f24 = f34 = f41 = f42 = f43 = 0;
		f44 = 1;

		f11 = (v.x * vCos.x) + fCos;
		f21 = (v.x * vCos.y) - (vSin.z);
		f31 = (v.x * vCos.z) + (vSin.y);
		f12 = (v.y * vCos.x) + (vSin.z);
		f22 = (v.y * vCos.y) + fCos;
		f32 = (v.y * vCos.z) - (vSin.x);
		f13 = (v.z * vCos.x) - (vSin.y);
		f32 = (v.z * vCos.y) + (vSin.x);
		f33 = (v.z * vCos.z) + fCos;
	}
	void Rotate(const CVector &v, const float f)
	{
		// 51 muls, 37 adds, 2 trig function calls
		CMatrix mat;
		mat.RotateMatrix(v, f);
		*this *= mat;
	}

	void ModelMatrix(const CQuaternion &q, const CVector &vFrom)
	{
		*this = q;
		f41 = vFrom.x;
		f42 = vFrom.y;
		f43 = vFrom.z;
	}
	void ModelMatrix(const CVector &vFrom, const CVector &vView, const CVector &vUp, const CVector &vRight)
	{
		f11 = vRight.x;	f21 = vUp.x;	f31 = -vView.x;	f41 = vFrom.x;
		f12 = vRight.y;	f22 = vUp.y;	f32 = -vView.y;	f42 = vFrom.y;
		f13 = vRight.z;	f23 = vUp.z;	f33 = -vView.z;	f43 = vFrom.z;
		f14 = 0;		f24 = 0;		f34 = 0;		f44 = 1;
	}
	void ModelMatrix(const CVector &vFrom, const CVector &vAt, const CVector &vUp)
	{
		CVector vView = vAt - vFrom;
		vView.Normalize();
		CVector vRight = vView ^ vUp;
		vRight.Normalize();
		CVector vTrueUp = vRight ^ vView;
		vTrueUp.Normalize();
		ModelMatrix(vFrom, vView, vTrueUp, vRight);
	}

	void ViewMatrix(const CQuaternion &q, const CVector &vFrom)
	{
		*this = q;
		Transpose();
		f41 = -(vFrom.x*f11 + vFrom.y*f21 + vFrom.z*f31);
		f42 = -(vFrom.x*f12 + vFrom.y*f22 + vFrom.z*f32);
		f43 = -(vFrom.x*f13 + vFrom.y*f23 + vFrom.z*f33);
	}
	void ViewMatrix(const CVector &vFrom, const CVector &vView, const CVector &vUp, const CVector &vRight)
	{
		// 9 muls, 9 adds
		f11 = vRight.x;	f21 = vRight.y;	f31 = vRight.z;	f41 = -(vFrom | vRight);
		f12 = vUp.x;	f22 = vUp.y;	f32 = vUp.z;	f42 = -(vFrom | vUp);
		f13 = -vView.x;	f23 = -vView.y;	f33 = -vView.z;	f43 = -(vFrom | -vView);
		f14 = 0;		f24 = 0;		f34 = 0;		f44 = 1;
	}
	void ViewMatrix(const CVector &vFrom, const CVector &vAt, const CVector &vUp)
	{
		CVector vView = vAt - vFrom;
		vView.Normalize();
		CVector vRight = vView ^ vUp;
		vRight.Normalize();
		CVector vTrueUp = vRight ^ vView;
		vTrueUp.Normalize();
		ViewMatrix(vFrom, vView, vTrueUp, vRight);
	}

	void ProjectionMatrix(const float fNear, const float fFar, const float fFOV, const float fAspect)
	{
		// 2 muls, 3 divs, 2 adds, 1 trig function call
		float h = 1.0f / tanf(DEGTORAD(fFOV * 0.5f));
		float Q = fFar / (fFar - fNear);
		f12 = f13 = f14 = f21 = f23 = f24 = f31 = f32 = f41 = f42 = f44 = 0;
		f11 = h / fAspect;
		f22 = h;
		f33 = Q;
		f34 = 1;
		f43 = -Q*fNear;
	}

	// For orthogonal matrices, I belive this also gives you the inverse.
	void Transpose()
	{
		float f;
		SWAP(f12, f21, f);
		SWAP(f13, f31, f);
		SWAP(f14, f41, f);
		SWAP(f23, f32, f);
		SWAP(f24, f42, f);
		SWAP(f34, f43, f);
	}
};


/*
class CRay
{
public:
	CDoubleVector m_vOrigin;
	CDoubleVector m_vDirection;

	CRay()		{}
	CRay(CDoubleVector p1, CDoubleVector p2)	{ Init(p1, p2); }
	void Init(CDoubleVector p1, CDoubleVector p2)
	{
		m_vOrigin = p1;
		m_vDirection = p2 - p1;
		m_vDirection.Normalize();
	}

	bool Intersect(CDoubleVector vCenter, double dRadius)
	{
		CDoubleVector vDir = vCenter - m_vOrigin;
		double v = m_vDirection | vDir;
		double b2 = (vDir|vDir) - v*v;
		double r2 = dRadius * dRadius;
		if(b2 > r2)
			return false;
		//vIntersection = m_vOrigin + m_vDirection * (v - sqrtf(r2 - b2));
		return true;
	}
};
*/
/*******************************************************************************
* Class: CLine
********************************************************************************
* This class implements a 3-dimensional line. It is initialized with two points
* on the line. It can be used to find the shortest distance between two lines or
* between a point and a line, which can be useful for things like collision
* detection.
*******************************************************************************/
class CLine
{
public:
	CVector m_vStart;
	CVector m_vDir;

	CLine()		{}
	CLine(CVector &p1, CVector &p2)	{ Init(p1, p2); }
	void Init(CVector &p1, CVector &p2)
	{
		m_vStart = p1;
		m_vDir = p2 - p1;
		m_vDir.Normalize();
	}

	CVector ClosestPoint(CVector &p)	{ return m_vStart + (m_vDir * ((m_vDir | p) - (m_vDir | m_vStart))); }
	float Distance(CVector &p)			{ return p.Distance(ClosestPoint(p)); }
	float Distance(CLine &l)
	{
		CVector v = m_vDir ^ l.m_vDir;
		v.Normalize();
		return Abs((m_vStart - l.m_vStart) | v);
	}
/*
	// Calculate the shortest distance between a point to a line
	(known) point p = p, or (px, py, pz)
	(known) point on line = a, or (ax, ay, az)
	(known) direction vector of line = n, or (nx, ny, nz)
	(unknown) point on line closest to p = P, or (Px, Py, Pz)

	// The vector from p to P must be perpendicular to n, which gives us:
	(px-Px, py-Py, pz-Pz) . (nx, ny, nz) = 0
	// To make sure P is actually on the line, it must also fit this line equation:
	(ax, ay, az) + k * (nx, ny, nz) = (Px, Py, Pz)

	// Expanding those equations gives us this:
	nx * (px - Px) + ny * (py - Py) + nz * (pz - Pz) = 0
	// and these:
	Px = ax + k * nx
	Py = ay + k * ny
	Pz = az + k * nz

	// Substituting and solving for k gives us:
	nx * (px - (ax + k * nx)) + ny * (py - (ay + k * ny)) + nz * (pz - (az + k * nz)) = 0
	nx*px - nx*ax - k*nx2 + ny*py - ny*ay - k*ny2 + nz*pz - nz*az - k*nz2 = 0
	k * (nx2 + ny2 + nz2) = nx*px - nx*ax + ny*py - ny*ay + nz*pz - nz*az
	k = (nx*px + ny*py + nz*pz - nx*ax - ny*ay - nz*az) / (nx2 + ny2 + nz2)
	k = (n.p - n.a) / (n.n)
	
	// If n is normalized then n.n will always equal 1, so...
	P = a + (n.p - n.a) * n
	distance = | P - p |
*/

/*
	// Calculate the shortest distance between two lines (call them line A and line B)
	(known) point on A = AP, or (APx, APy, APz)
	(known) direction vector of A = AN, or (ANx, ANy, ANz)
	(known) point on B = BP, or (BPx, BPy, BPz)
	(known) direction vector of B = BN, or (BNx, BNy, BNz)
	(unknown) point on A closest to B = P1, or (P1x, P1y, P1z)
	(unknown) point on B closest to A = P2, or (P2x, P2y, P2z)

	// The vector from P1 to P2 is perpendicular to both lines:
	(P2x - P1x, P2y - P1y, P2z - P1z) . (ANx, ANy, ANz) = 0
	(P2x - P1x, P2y - P1y, P2z - P1z) . (BNx, BNy, BNz) = 0
	// To find P1 and P2, we need to find values k and l for the line equations:
	(APx, APy, APz) + k * (ANx, ANy, ANz) = (P1x, P1y, P1z)
	(BPx, BPy, BPz) + l * (BNx, BNy, BNz) = (P2x, P2y, P2z)

	// Expanding these equations gives us these:
	ANx * (P2x - P1x) + ANy * (P2y - P1y) + ANz * (P2z - P1z) = 0
	BNx * (P2x - P1x) + BNy * (P2y - P1y) + BNz * (P2z - P1z) = 0
	// And these:
	P1x = APx + k * ANx
	P1y = APy + k * ANy
	P1z = APz + k * ANz
	P2x = BPx + l * BNx
	P2y = BPy + l * BNy
	P2z = BPz + l * BNz

	// Substituting and solving for k and l gives us:
	ANx * ((BPx + l * BNx) - (APx + k * ANx)) + ANy * ((BPy + l * BNy) - (APy + k * ANy)) + ANz * ((BPz + l * BNz) - (APz + k * ANz)) = 0
	BNx * ((BPx + l * BNx) - (APx + k * ANx)) + BNy * ((BPy + l * BNy) - (APy + k * ANy)) + BNz * ((BPz + l * BNz) - (APz + k * ANz)) = 0

	(ANx*BPx + l*ANx*BNx - ANx*APx - k*ANx2) + (ANy*BPy + l*ANy*BNy - ANy*APy - k*ANy2) + (ANz*BPz + l*ANz*BNz - ANz*APz - k*ANz2) = 0
	(BNx*BPx + l*BNx2 - BNx*APx - k*ANx*BNx) + (BNy*BPy + l*BNy2 - BNy*APy - k*ANy*BNy) + (BNz*BPz + l*BNz2 - BNz*APz - k*ANz*BNz) = 0

	k = (ANx*BPx + l*ANx*BNx - ANx*APx + ANy*BPy + l*ANy*BNy - ANy*APy + ANz*BPz + l*ANz*BNz - ANz*APz) / (ANx2 + ANy2 + ANz2)
	k = (BNx*BPx + l*BNx2 - BNx*APx + BNy*BPy + l*BNy2 - BNy*APy + BNz*BPz + l*BNz2 - BNz*APz) / (ANx*BNx + ANy*BNy + ANz*BNz)

	(ANx*BNx + ANy*BNy + ANz*BNz) * (ANx*BPx + l*ANx*BNx - ANx*APx + ANy*BPy + l*ANy*BNy - ANy*APy + ANz*BPz + l*ANz*BNz - ANz*APz)  = (ANx2 + ANy2 + ANz2) * (BNx*BPx + l*BNx2 - BNx*APx + BNy*BPy + l*BNy2 - BNy*APy + BNz*BPz + l*BNz2 - BNz*APz)

	(ANx*BNx*ANy*BPy + ANx*BNx*l*ANy*BNy - ANx*BNx*ANy*APy + ANx*BNx*ANz*BPz + ANx*BNx*l*ANz*BNz - ANx*BNx*ANz*APz) + 
	(ANy*BNy*ANx*BPx + ANy*BNy*l*ANx*BNx - ANy*BNy*ANx*APx + ANy*BNy*ANz*BPz + ANy*BNy*l*ANz*BNz - ANy*BNy*ANz*APz) +
	(ANz*BNz*ANx*BPx + ANz*BNz*l*ANx*BNx - ANz*BNz*ANx*APx + ANz*BNz*ANy*BPy + ANz*BNz*l*ANy*BNy - ANz*BNz*ANy*APy) =
	(ANx2*BNy*BPy + ANx2*l*BNy2 - ANx2*BNy*APy + ANx2*BNz*BPz + ANx2*l*BNz2 - ANx2*BNz*APz) +
	(ANy2*BNx*BPx + ANy2*l*BNx2 - ANy2*BNx*APx + ANy2*BNz*BPz + ANy2*l*BNz2 - ANy2*BNz*APz) +
	(ANz2*BNx*BPx + ANz2*l*BNx2 - ANz2*BNx*APx + ANz2*BNy*BPy + ANz2*l*BNy2 - ANz2*BNy*APy)

	l * (2*ANx*BNx*ANy*BNy + 2*ANz*BNz*ANx*BNx + 2*ANy*BNy*ANz*BNz - ANx2*BNy2 - ANy2*BNx2 - ANz2*BNx2 - ANx2*BNz2 - ANy2*BNz2 - ANz2*BNy2) = 
	(ANx2*BNy*BPy - ANx2*BNy*APy + ANx2*BNz*BPz - ANx2*BNz*APz) +
	(ANy2*BNx*BPx - ANy2*BNx*APx + ANy2*BNz*BPz - ANy2*BNz*APz) +
	(ANz2*BNx*BPx - ANz2*BNx*APx + ANz2*BNy*BPy - ANz2*BNy*APy) +
	(-ANx*BNx*ANy*BPy + ANx*BNx*ANy*APy - ANx*BNx*ANz*BPz + ANx*BNx*ANz*APz) +
	(-ANy*BNy*ANx*BPx + ANy*BNy*ANx*APx - ANy*BNy*ANz*BPz + ANy*BNy*ANz*APz) +
	(-ANz*BNz*ANx*BPx + ANz*BNz*ANx*APx - ANz*BNz*ANy*BPy + ANz*BNz*ANy*APy)

	// Next think about using AN ^ BN to get the direction vector between P1 and P2
	// Normalize that vector and call it n
	// The distance between the two lines might be the absolute value of (P2-P1).n
	// That still doesn't tell us where the closest points on the line are, but it might help find them
*/
};


/*******************************************************************************
* Class: CPlane
********************************************************************************
* This class implements a 3-dimensional plane equation. It can be initialized
* with 3 points on the plane or a normal and a point on the plane, and it can
* be used to find the distance from a point to a plane. I plan to add more to
* this class later to make it more useful.
*******************************************************************************/
class CPlane
{
public:
	CVector m_vNormal;
	float D;

	CPlane()		{}
	void Init(CVector &p1, CVector &p2, CVector &p3)
	{	// Initializes the plane based on three points in the plane
		Init(NormalVector(p1, p2, p3), p1);
	}
	void Init(CVector &vNormal, CVector &p)
	{	// Initializes the plane based on a normal and a point in the plane
		m_vNormal = vNormal;
		D = -(p | m_vNormal);
	}

	bool operator==(CPlane &p)	{ return (Abs(D - p.D) <= DELTA && m_vNormal == p.m_vNormal); }
	bool operator!=(CPlane &p)	{ return !(*this == p); }

	float Distance(const CVector &p) const
	{	// Returns the distance between the plane and point p
		return (m_vNormal | p) + D;	// A positive, 0, or negative result indicates the point is in front of, on, or behind the plane
	}

	bool Intersection(CVector &vPos, CVector &vDir)
	{	// Returns true if the line intersects the plane and changes vPos to the location of the intersection
		float f = m_vNormal | vDir;
		if(ABS(f) < DELTA)
			return false;
		vPos -= vDir * (Distance(vPos) / f);
		return true;
	}
};


class CFrustum
{
protected:
	CPlane m_plFrustum[6];

public:
	CFrustum() {}
	void Init()
	{
		float   proj[16];
		float   modl[16];
		float   clip[16];
		float   t;

		/* Get the current PROJECTION matrix from OpenGL */
		glGetFloatv(GL_PROJECTION_MATRIX, proj);

		/* Get the current MODELVIEW matrix from OpenGL */
		glGetFloatv(GL_MODELVIEW_MATRIX, modl);

		/* Combine the two matrices (multiply projection by modelview) */
		clip[ 0] = modl[ 0] * proj[ 0] + modl[ 1] * proj[ 4] + modl[ 2] * proj[ 8] + modl[ 3] * proj[12];
		clip[ 1] = modl[ 0] * proj[ 1] + modl[ 1] * proj[ 5] + modl[ 2] * proj[ 9] + modl[ 3] * proj[13];
		clip[ 2] = modl[ 0] * proj[ 2] + modl[ 1] * proj[ 6] + modl[ 2] * proj[10] + modl[ 3] * proj[14];
		clip[ 3] = modl[ 0] * proj[ 3] + modl[ 1] * proj[ 7] + modl[ 2] * proj[11] + modl[ 3] * proj[15];

		clip[ 4] = modl[ 4] * proj[ 0] + modl[ 5] * proj[ 4] + modl[ 6] * proj[ 8] + modl[ 7] * proj[12];
		clip[ 5] = modl[ 4] * proj[ 1] + modl[ 5] * proj[ 5] + modl[ 6] * proj[ 9] + modl[ 7] * proj[13];
		clip[ 6] = modl[ 4] * proj[ 2] + modl[ 5] * proj[ 6] + modl[ 6] * proj[10] + modl[ 7] * proj[14];
		clip[ 7] = modl[ 4] * proj[ 3] + modl[ 5] * proj[ 7] + modl[ 6] * proj[11] + modl[ 7] * proj[15];

		clip[ 8] = modl[ 8] * proj[ 0] + modl[ 9] * proj[ 4] + modl[10] * proj[ 8] + modl[11] * proj[12];
		clip[ 9] = modl[ 8] * proj[ 1] + modl[ 9] * proj[ 5] + modl[10] * proj[ 9] + modl[11] * proj[13];
		clip[10] = modl[ 8] * proj[ 2] + modl[ 9] * proj[ 6] + modl[10] * proj[10] + modl[11] * proj[14];
		clip[11] = modl[ 8] * proj[ 3] + modl[ 9] * proj[ 7] + modl[10] * proj[11] + modl[11] * proj[15];

		clip[12] = modl[12] * proj[ 0] + modl[13] * proj[ 4] + modl[14] * proj[ 8] + modl[15] * proj[12];
		clip[13] = modl[12] * proj[ 1] + modl[13] * proj[ 5] + modl[14] * proj[ 9] + modl[15] * proj[13];
		clip[14] = modl[12] * proj[ 2] + modl[13] * proj[ 6] + modl[14] * proj[10] + modl[15] * proj[14];
		clip[15] = modl[12] * proj[ 3] + modl[13] * proj[ 7] + modl[14] * proj[11] + modl[15] * proj[15];

		/* Extract the numbers for the RIGHT plane */
		m_plFrustum[0].m_vNormal.x = clip[ 3] - clip[ 0];
		m_plFrustum[0].m_vNormal.y = clip[ 7] - clip[ 4];
		m_plFrustum[0].m_vNormal.z = clip[11] - clip[ 8];
		m_plFrustum[0].D = clip[15] - clip[12];
		t = m_plFrustum[0].m_vNormal.Magnitude();
		m_plFrustum[0].m_vNormal /= t;
		m_plFrustum[0].D /= t;

		/* Extract the numbers for the LEFT plane */
		m_plFrustum[1].m_vNormal.x = clip[ 3] + clip[ 0];
		m_plFrustum[1].m_vNormal.y = clip[ 7] + clip[ 4];
		m_plFrustum[1].m_vNormal.z = clip[11] + clip[ 8];
		m_plFrustum[1].D = clip[15] + clip[12];
		t = m_plFrustum[1].m_vNormal.Magnitude();
		m_plFrustum[1].m_vNormal /= t;
		m_plFrustum[1].D /= t;

		/* Extract the BOTTOM plane */
		m_plFrustum[2].m_vNormal.x = clip[ 3] + clip[ 1];
		m_plFrustum[2].m_vNormal.y = clip[ 7] + clip[ 5];
		m_plFrustum[2].m_vNormal.z = clip[11] + clip[ 9];
		m_plFrustum[2].D = clip[15] + clip[13];
		t = m_plFrustum[2].m_vNormal.Magnitude();
		m_plFrustum[2].m_vNormal /= t;
		m_plFrustum[2].D /= t;

		/* Extract the TOP plane */
		m_plFrustum[3].m_vNormal.x = clip[ 3] - clip[ 1];
		m_plFrustum[3].m_vNormal.y = clip[ 7] - clip[ 5];
		m_plFrustum[3].m_vNormal.z = clip[11] - clip[ 9];
		m_plFrustum[3].D = clip[15] - clip[13];
		t = m_plFrustum[3].m_vNormal.Magnitude();
		m_plFrustum[3].m_vNormal /= t;
		m_plFrustum[3].D /= t;

		/* Extract the FAR plane */
		m_plFrustum[4].m_vNormal.x = clip[ 3] - clip[ 2];
		m_plFrustum[4].m_vNormal.y = clip[ 7] - clip[ 6];
		m_plFrustum[4].m_vNormal.z = clip[11] - clip[10];
		m_plFrustum[4].D = clip[15] - clip[14];
		t = m_plFrustum[4].m_vNormal.Magnitude();
		m_plFrustum[4].m_vNormal /= t;
		m_plFrustum[4].D /= t;

		/* Extract the NEAR plane */
		m_plFrustum[5].m_vNormal.x = clip[ 3] + clip[ 2];
		m_plFrustum[5].m_vNormal.y = clip[ 7] + clip[ 6];
		m_plFrustum[5].m_vNormal.z = clip[11] + clip[10];
		m_plFrustum[5].D = clip[15] + clip[14];
		t = m_plFrustum[5].m_vNormal.Magnitude();
		m_plFrustum[5].m_vNormal /= t;
		m_plFrustum[5].D /= t;
	}

	bool IsInFrustum(const CVector &vPos, const float fRadius) const
	{
		for(int i=0; i<4; i++)
		{
			if(m_plFrustum[i].Distance(vPos) < -fRadius)
				return false;
		}
		return true;
	}
};

/****************************************************************************
* Class: C3DObject
*****************************************************************************
* This class represents a basic 3D object in the scene. It has a 3D position,
* orientation, velocity, and a parent which provides its frame of reference
* in the scene.
* Note: This class is derived from CQuaternion so it will inherit useful
*       functions like Rotate(), GetViewAxis(), GetUpAxis(), and GetRightAxis().
****************************************************************************/
#define MIN_DISTANCE	0.01
#define MAX_DISTANCE	1000.0				// Distance to desired far clipping plane
#define MAX_DISCERNABLE	1000000.0			// Beyond this distance, everything is rendered at MAX_DISTANCE
#define HALF_MAX		(MAX_DISTANCE*0.5)	// Everything between HALF_MAX and MAX_DISCERNABLE is scaled exponentially between HALF_MAX and MAX_DISTANCE
class C3DObject : public CQuaternion
{
protected:
	C3DObject *m_pParent;		// The object's parent
	float m_fBoundingRadius;	// The object's bounding radius
	float m_fMass;				// The object's mass (kg)
	CDoubleVector m_vPosition;	// The object's position (km)
	CVector m_vVelocity;		// The object's velocity (km/s)

public:
	C3DObject() : CQuaternion(0.0f, 0.0f, 0.0f, 1.0f), m_vPosition(0.0f), m_vVelocity(0.0f)
	{
		m_fMass = 0.0f;
	}
	C3DObject(const CMatrix &m) : CQuaternion(m), m_vPosition(m.f41, m.f42, m.f43), m_vVelocity(0.0f)
	{
		m_fMass = 0.0f;
	}

	void operator=(const CQuaternion &q)		{ CQuaternion::operator=(q); }
	void operator=(const CMatrix &m)
	{
		CQuaternion::operator=(m);
		SetPosition(CDoubleVector(m.f41, m.f42, m.f43));
	}
	
	void SetParent(C3DObject *p)		{ m_pParent = p; }
	C3DObject *GetParent()				{ return m_pParent; }
	void SetMass(float f)				{ m_fMass = f; }
	float GetMass()						{ return m_fMass; }
	void SetBoundingRadius(float f)		{ m_fBoundingRadius = f; }
	float GetBoundingRadius()			{ return m_fBoundingRadius; }
	void SetPosition(CDoubleVector &v)	{ m_vPosition = v; }
	CDoubleVector GetPosition()			{ return m_vPosition; }
	void SetVelocity(CVector &v)		{ m_vVelocity = v; }
	CVector GetVelocity()				{ return m_vVelocity; }

	CMatrix GetViewMatrix()
	{
		// Don't use the normal view matrix because it causes precision problems if the camera is too far away from the origin.
		// Instead, pretend the camera is at the origin and offset all model matrices by subtracting the camera's position.
		CMatrix m = *this;
		m.Transpose();
		return m;
	}
	CMatrix GetModelMatrix(C3DObject *pCamera)
	{
		// Don't use the normal model matrix because it causes precision problems if the camera and model are too far away from the origin.
		// Instead, pretend the camera is at the origin and offset all model matrices by subtracting the camera's position.
		CMatrix m;
		m.ModelMatrix(*this, m_vPosition-pCamera->m_vPosition);
		return m;
	}

	static float GetScalingFactor(double dDistance, double dFactor=1.0)
	{
		//return 1.0f;
		if(dDistance > HALF_MAX)
		{
			dFactor *= (dDistance >= MAX_DISCERNABLE) ? MAX_DISTANCE : HALF_MAX + HALF_MAX * (1.0 - exp(-2.5 * dDistance / MAX_DISCERNABLE));
			dFactor /= dDistance;
		}
		return (float)dFactor;
	}

	float ScaleModelMatrix(CMatrix &m, double dDistance, double dFactor=1.0)
	{
		// This code scales the object's size and distance to the camera down when it's too far away.
		// This solves a problem with many video card drivers where objects too far away aren't rendering properly.
		// It also alleviates the Z-buffer precision problem caused by having your near and far clipping planes too far apart.
		float fFactor = GetScalingFactor(dDistance, dFactor);
		if(fFactor <= 1.0f)
		{
			m.f41 *= fFactor;
			m.f42 *= fFactor;
			m.f43 *= fFactor;
			m.Scale(fFactor, fFactor, fFactor);
		}
		return fFactor;
	}

	float GetScaledModelMatrix(CMatrix &m, C3DObject *pCamera, double dFactor=1.0)
	{
		CDoubleVector vPos = m_vPosition - pCamera->m_vPosition;
		double dDistance = vPos.Magnitude();
		m.ModelMatrix(*this, vPos);
		return ScaleModelMatrix(m, dDistance, dFactor);
	}

	void GetBillboardMatrix(CMatrix &m, C3DObject *pCamera, float fSize)
	{
		CDoubleVector vPos = m_vPosition - pCamera->m_vPosition;
		double dDistance = vPos.Magnitude();

		CVector vView = vPos / dDistance;
		CVector vRight = vView ^ CVector(0, 1, 0);
		vRight.Normalize();
		CVector vUp = vRight ^ vView;
		vUp.Normalize();

		m.ModelMatrix(vPos, vView, vUp, vRight);
		m.Scale(fSize, fSize, fSize);
	}

	float GetScaledBillboardMatrix(CMatrix &m, C3DObject *pCamera, float fSize, double dFactor=1.0)
	{
		CDoubleVector vPos = m_vPosition - pCamera->m_vPosition;
		double dDistance = vPos.Magnitude();

		CVector vView = vPos / dDistance;
		CVector vRight = vView ^ CVector(0, 1, 0);
		vRight.Normalize();
		CVector vUp = vRight ^ vView;
		vUp.Normalize();

		m.ModelMatrix(vPos, vView, vUp, vRight);
		float fFactor = ScaleModelMatrix(m, dDistance, dFactor);
		m.Scale(fSize, fSize, fSize);
		return fFactor;
	}

	float DistanceTo(C3DObject &obj)		{ return (float)m_vPosition.Distance(obj.m_vPosition); }
	CVector VectorTo(C3DObject &obj)		{ return obj.m_vPosition - m_vPosition; }
	float GravityPull(float fDistSquared)	{ return (GRAVCONST * m_fMass) / fDistSquared; }
	CVector GravityVector(C3DObject &obj)
	{
		CVector vGravity = obj.VectorTo(*this);
		float fDistSquared = vGravity.MagnitudeSquared();
		vGravity *= GravityPull(fDistSquared) / sqrtf(fDistSquared);
		return vGravity;
	}

	void Accelerate(CVector &vAccel, float fSeconds, float fResistance=0)
	{
		m_vVelocity += vAccel * fSeconds;
		if(fResistance > DELTA)
			m_vVelocity *= 1.0f - fResistance * fSeconds;
		m_vPosition += m_vVelocity * fSeconds;
	}

	// Kinetic energy (Joules) = 1/2 * mass * velocity^2 (mass in kg, velocity in m/s)
	// Kinetic energy (kilotons) = (KE in Joules) / (4.185e12 joules/KT)
	//float GetKEJoules(C3DObject &obj)		{ return 500000.0f * m_fMass * (m_vVelocity-obj.m_vVelocity).MagnitudeSquared(); }
	//float GetKEKilotons(C3DObject &obj)		{ return 1.19474e-7f * m_fMass * (m_vVelocity-obj.m_vVelocity).MagnitudeSquared(); }
};

#define COLOR_CTOF(x)	(x/256.0f)
#define COLOR_FTOC(x)	(ColorClamp(x*256))

inline unsigned char ColorClamp(int n)				{ return (unsigned char)(n < 0 ? 0 : n > 255 ? 255 : n); }
inline unsigned char ColorClamp(float f)			{ return (unsigned char)(f < 0 ? 0 : f > 255 ? 255 : f); }

class CColor
{
public:
	unsigned char r, g, b, a;

	CColor() {}
	CColor(int r, int g, int b, int a=255)
	{
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}
	CColor(float r, float g, float b, float a=1.0f)
	{
		this->r = COLOR_FTOC(r);
		this->g = COLOR_FTOC(g);
		this->b = COLOR_FTOC(b);
		this->a = COLOR_FTOC(a);
	}

	CColor operator*(const float f) const		{ return CColor(ColorClamp(r*f), ColorClamp(g*f), ColorClamp(b*f), ColorClamp(a*f)); }
	CColor operator+(const CColor c) const		{ return CColor(ColorClamp(r+c.r), ColorClamp(g+c.g), ColorClamp(b+c.b), ColorClamp(a+c.a)); }
	operator unsigned char*()					{ return &r; }
};

class CRay
{
public:
	CDoubleVector m_vOrigin;
	CDoubleVector m_vDirection;

	CRay()		{}
	CRay(const CDoubleVector p1, const CDoubleVector p2)	{ Init(p1, p2); }
	CRay(C3DObject *pCamera, const CVector &vPointer)	{ Init(pCamera, vPointer); }
	void Init(const CDoubleVector p1, const CDoubleVector p2)
	{
		m_vOrigin = p1;
		m_vDirection = p2 - p1;
		m_vDirection.Normalize();
	}
	void Init(C3DObject *pCamera, const CVector &vPointer)
	{
		double dModelView[16], dProjection[16];
		int nViewport[4];

		CMatrix mView;
		mView.ViewMatrix(*pCamera, pCamera->GetPosition());
		for(int i=0; i<16; i++)
			dModelView[i] = mView[i];
		glGetDoublev(GL_PROJECTION_MATRIX, dProjection);
		glGetIntegerv(GL_VIEWPORT, nViewport);

		CDoubleVector vFrom, vTo;
		gluUnProject(vPointer.x, vPointer.y, 0.0, dModelView, dProjection, nViewport, &vFrom.x, &vFrom.y, &vFrom.z);
		gluUnProject(vPointer.x, vPointer.y, 1.0, dModelView, dProjection, nViewport, &vTo.x, &vTo.y, &vTo.z);
		Init(vFrom, vTo);
	}

	bool GetNearestIntersection(CDoubleVector vCenter, double dRadius, CDoubleVector &vIntersection)
	{
		CDoubleVector vDir = vCenter - m_vOrigin;
		double v = m_vDirection | vDir;
		double b2 = (vDir|vDir) - v*v;
		double r2 = dRadius * dRadius;
		if(b2 > r2)
		{
			CDoubleVector vRight = vDir ^ m_vDirection;
			CDoubleVector vUp = vRight ^ vDir;
			vUp.Normalize();
			//vIntersection = vCenter + vUp * dRadius;

			double dDistance = sqrt(vDir | vDir);							// Distance to center of sphere
			double dAltitude = dDistance - dRadius;							// Distance to nearest edge of sphere
			double dHorizon = sqrt(dAltitude*dAltitude + 2.0f*dAltitude*dRadius);	// Distance to horizon
			double dCos = dHorizon / dDistance;
			double dAngle = HALF_PI - acos(dCos);
			CDoubleVector vOut = vDir / -dDistance;
			vIntersection = vCenter + (vUp*sin(dAngle)) + (vOut*cos(dAngle));
		}
		else
			vIntersection = m_vOrigin + m_vDirection * (v - sqrtf((float)(r2 - b2)));
		return true;
	}
	bool GetIntersection(CDoubleVector vCenter, double dRadius, CDoubleVector &vIntersection)
	{
		CDoubleVector vDir = vCenter - m_vOrigin;
		double v = m_vDirection | vDir;
		double b2 = (vDir|vDir) - v*v;
		double r2 = dRadius * dRadius;
		if(b2 > r2)
			return false;
		vIntersection = m_vOrigin + m_vDirection * (v - sqrtf((float)(r2 - b2)));
		return true;
	}
};

#endif //__Matrix_h__
