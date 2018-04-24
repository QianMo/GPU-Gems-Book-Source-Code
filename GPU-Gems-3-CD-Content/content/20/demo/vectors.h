//
// vectors.h
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

#ifndef _VECTORS_H
#define _VECTORS_H

#include <iostream>
#include <cmath>

class Vector2D;
class Vector3D;
class Vector4D;

/// 2-dimensional vectors
class Vector2D {
	public:
		inline Vector2D() : x(0), y(0) {}
		inline Vector2D(float f) : x(f), y(f) {}
		inline Vector2D(float _x, float _y) : x(_x), y(_y) {}
		explicit inline Vector2D(const Vector3D &v);
		inline ~Vector2D() {}

		inline Vector2D operator + (const Vector2D &v) const { return Vector2D(x+v.x, y+v.y); }
		inline Vector2D operator - (const Vector2D &v) const { return Vector2D(x-v.x, y-v.y); }
		
		inline float operator * (const Vector2D &v) const { return x*v.x + y*v.y; }
		inline Vector3D operator %(const Vector2D &v) const;

		inline Vector2D& operator += (const Vector2D &v) { x+=v.x; y+=v.y; return (*this); }
		inline Vector2D& operator -= (const Vector2D &v) { x-=v.x; y-=v.y; return (*this); }
		
		inline Vector2D& operator += (const float &f) { x+=f; y+=f; return (*this); }
		inline Vector2D& operator -= (const float &f) { x-=f; y-=f; return (*this); }
		inline Vector2D& operator *= (const float &f) { x*=f; y*=f; return (*this); }
		inline Vector2D& operator /= (const float &f) { x/=f; y/=f; return (*this); }

		inline Vector2D operator + (const float &f) const { return Vector2D(x+f, y+f); }
		inline Vector2D operator - (const float &f) const { return Vector2D(x-f, y-f); }
		inline Vector2D operator * (const float &f) const { return Vector2D(x*f, y*f); }
		inline Vector2D operator / (const float &f) const { return Vector2D(x/f, y/f); }

		inline Vector2D operator -() const { return Vector2D(-x, -y); }

		friend std::ostream& operator<< (std::ostream& fout, const Vector2D& v) {
			fout << "( " << v.x << " " << v.y << " " << " )";
			return fout;
		}

		friend Vector2D operator* (const float &f, const Vector2D &v) {
			return v*f;
		}

		inline static Vector2D vmin(const Vector2D &v1, const Vector2D &v2) { return Vector2D(((v1.x<v2.x)?v1.x:v2.x), ((v1.y<v2.y)?v1.y:v2.y)); }
		inline static Vector2D vmax(const Vector2D &v1, const Vector2D &v2) { return Vector2D(((v1.x>v2.x)?v1.x:v2.x), ((v1.y>v2.y)?v1.y:v2.y)); }

		inline float distance2() const { return x*x+y*y; }
		inline float distance() const { return sqrtf(distance2()); }
		
		inline static Vector2D normalize(const Vector2D&v) { return v/v.distance(); }

		float x,y;
};

inline Vector2D mul(const Vector2D &a, const Vector2D &b) { return Vector2D(a.x*b.x, a.y*b.y); }

/// 3-dimensional vectors
class Vector3D {
	public:
		inline Vector3D() : x(0), y(0), z(0) {}
		inline Vector3D(float f) : x(f), y(f), z(f) {}
		inline Vector3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
		inline Vector3D(const Vector2D &v) : x(v.x), y(v.y), z(0) {}
		explicit inline Vector3D(const Vector4D &v);
		~Vector3D() {}
		
		inline Vector3D operator + (const Vector3D &v) const { return Vector3D(x+v.x, y+v.y, z+v.z); }
		inline Vector3D operator - (const Vector3D &v) const { return Vector3D(x-v.x, y-v.y, z-v.z); }

		// dot product
		inline float operator * (const Vector3D &v) const { return x*v.x + y*v.y + z*v.z; }

		// cross product
		inline Vector3D operator % (const Vector3D&) const;

		inline Vector3D& operator+= (const Vector3D &v) { x+=v.x; y+=v.y; z+=v.z; return (*this); }
		inline Vector3D& operator-= (const Vector3D &v) { x-=v.x; y-=v.y; z-=v.z; return (*this); }
		
		inline Vector3D& operator += (const float &f) { x+=f; y+=f; z+=f; return (*this); }
		inline Vector3D& operator -= (const float &f) { x-=f; y-=f; z-=f; return (*this); }
		inline Vector3D& operator *= (const float &f) { x*=f; y*=f; z*=f; return (*this); }
		inline Vector3D& operator /= (const float &f) { x/=f; y/=f; z/=f; return (*this); }

		inline Vector3D operator - () const { return Vector3D(-x,-y,-z); }
		
		inline Vector3D operator + (const float &f) const { return Vector3D(x+f, y+f, z+f); }
		inline Vector3D operator - (const float &f) const { return Vector3D(x-f, y-f, z-f); }
		inline Vector3D operator * (const float &f) const { return Vector3D(x*f, y*f, z*f); }
		inline Vector3D operator / (const float &f) const { return Vector3D(x/f, y/f, z/f); }
		
		friend std::ostream& operator<< (std::ostream& fout, const Vector3D& v) {
			fout << "( " << v.x << " " << v.y << " " << v.z << " )";
			return fout;
		}
		
		friend Vector3D operator*  (const float& f, const Vector3D& v) {
			return v * f;
		}

		inline float distance2() const { return x*x+y*y+z*z; }
		inline float distance() const { return sqrt(distance2()); }
		
		inline static Vector3D normalize(const Vector3D &v) { return v/v.distance(); }
		inline static Vector3D minv(const Vector3D &a, const Vector3D &b) { return Vector3D(((a.x<b.x)?a.x:b.x), ((a.y<b.y)?a.y:b.y), ((a.z<b.z)?a.z:b.z)); }
		inline static Vector3D maxv(const Vector3D &a, const Vector3D &b) { return Vector3D(((a.x>b.x)?a.x:b.x), ((a.y>b.y)?a.y:b.y), ((a.z>b.z)?a.z:b.z)); }

		float x,y,z;

	private:
		char padding[4]; // buffer bytes
};

/// 4-dimensional vectors
class Vector4D {
	public:
		inline Vector4D() : x(0), y(0), z(0), w(1) {}
		inline Vector4D(float f) : x(f), y(f), z(f), w(1) {}
		inline Vector4D(float _x, float _y, float _z) : x(_x), y(_y), z(_z), w(1) {}
		inline Vector4D(const Vector2D &v, float _z, float _w) : x(v.x), y(v.y), z(_z), w(_w) {}
		inline Vector4D(const Vector2D &v, const Vector2D &w) : x(v.x), y(v.y), z(w.x), w(w.y) {}
		inline Vector4D(const Vector3D &v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) {}
		inline Vector4D(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
		explicit inline Vector4D(const Vector2D &v) : x(v.x), y(v.y), z(0), w(1) {}
		explicit inline Vector4D(const Vector3D &v) : x(v.x), y(v.y), z(v.z), w(1) {}
		~Vector4D() {}
		
		inline Vector4D operator + (const Vector4D &v) const { return Vector4D(x+v.x, y+v.y, z+v.z, w+v.w); }
		inline Vector4D operator - (const Vector4D &v) const { return Vector4D(x-v.x, y-v.y, z-v.z, w+v.w); }

		// dot product
		inline float operator * (const Vector4D &v) const { return x*v.x + y*v.y + z*v.z + w*v.w; }

		inline Vector4D& operator+= (const Vector4D &v) { x+=v.x; y+=v.y; z+=v.z; w+=v.w; return (*this); }
		inline Vector4D& operator-= (const Vector4D &v) { x-=v.x; y-=v.y; z-=v.z; w-=v.w; return (*this); }
		
		inline Vector4D& operator += (const float &f) { x+=f; y+=f; z+=f; w+=f; return (*this); }
		inline Vector4D& operator -= (const float &f) { x-=f; y-=f; z-=f; w-=f; return (*this); }
		inline Vector4D& operator *= (const float &f) { x*=f; y*=f; z*=f; w*=f; return (*this); }
		inline Vector4D& operator /= (const float &f) { x/=f; y/=f; z/=f; w/=f; return (*this); }

		inline Vector4D operator - () const { return Vector4D(-x,-y,-z,-w); }
		
		inline Vector4D operator + (const float &f) const { return Vector4D(x+f, y+f, z+f, w+f); }
		inline Vector4D operator - (const float &f) const { return Vector4D(x-f, y-f, z-f, w-f); }
		inline Vector4D operator * (const float &f) const { return Vector4D(x*f, y*f, z*f, w*f); }
		inline Vector4D operator / (const float &f) const { return Vector4D(x/f, y/f, z/f, w/f); }
		
		friend std::ostream& operator<< (std::ostream& fout, const Vector4D& v) {
			fout << "( " << v.x << " " << v.y << " " << v.z << " " << v.w << " )";
			return fout;
		}
		
		friend Vector4D operator*  (const float& f, const Vector4D& v) { return v * f; }
		
		inline Vector3D homogeneous() { return Vector3D(x/w, y/w, z/w); }

		float x,y,z,w;
};

inline Vector3D mul(const Vector3D &a, const Vector3D &b) { return Vector3D(a.x*b.x, a.y*b.y, a.z*b.z); }

inline Vector3D Vector3D::operator % (const Vector3D& v) const {
	Vector3D v1, v2, v3, v4;
	float* v1ptr = (float*) &v1;
	float* v2ptr = (float*) &v2;
	float* v3ptr = (float*) &v3;
	float* v4ptr = (float*) &v4;
	
	v1ptr[0] = y;	v1ptr[1] = z;	v1ptr[2] = x;
	v2ptr[0] = v.z;	v2ptr[1] = v.x;	v2ptr[2] = v.y;	
	v3ptr[0] = z;	v3ptr[1] = x;	v3ptr[2] = y;	
	v4ptr[0] = v.y;	v4ptr[1] = v.z;	v4ptr[2] = v.x;
	
	return mul(v1,v2) - mul(v3,v4);
}

inline Vector2D::Vector2D(const Vector3D &v) : x(v.x), y(v.y) {}
inline Vector3D::Vector3D(const Vector4D &v) : x(v.x), y(v.y), z(v.z) {}
inline Vector3D Vector2D::operator %(const Vector2D &v) const { return Vector3D(0,0,x*v.y-y*v.x); }


#endif
