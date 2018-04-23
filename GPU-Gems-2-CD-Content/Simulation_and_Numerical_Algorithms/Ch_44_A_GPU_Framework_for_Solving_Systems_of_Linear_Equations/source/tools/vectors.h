/**************************************************************
 *                                                            *
 * description: simple vector templates						  *
 * version    : 1.10                                          *
 * date       : 01.Jul.2003                                   *
 * modified   : 16.Dec.2004                                   *
 * author     : Jens Krüger                                   *
 * e-mail     : mail@jens-krueger.com                         *
 *                                                            *
 **************************************************************/
#pragma once

#ifndef MAX
	#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
	#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

typedef unsigned int uint;


template <class T=int> class VECTOR2 {
public:
	T x;
	T y;

	VECTOR2<T>(): x(0), y(0) {}
	VECTOR2<T>(const VECTOR2<T> &other): x(other.x), y(other.y) {}
	VECTOR2<T>(const T _x, const T _y) : x(_x), y(_y) {}

	bool operator == ( const VECTOR2<T>& other ) const {return (other.x==x && other.y==y); }
	bool operator != ( const VECTOR2<T>& other ) const {return (other.x!=x || other.y!=y); } 

    // binary operators with scalars
	VECTOR2<T> operator + ( T scalar ) const {return VECTOR2<T>(x+scalar,y+scalar);}
	VECTOR2<T> operator - ( T scalar ) const {return VECTOR2<T>(x-scalar,y-scalar);}
	VECTOR2<T> operator * ( T scalar ) const {return VECTOR2<T>(x*scalar,y*scalar);}
	VECTOR2<T> operator / ( T scalar ) const {return VECTOR2<T>(x/scalar,y/scalar);}

	// binaray operpators  with intvectors
	VECTOR2<T> operator + ( const VECTOR2& other ) const {return VECTOR2<T>(x+other.x,y+other.y);}
	VECTOR2<T> operator - ( const VECTOR2& other ) const {return VECTOR2<T>(x-other.x,y-other.y);}
	VECTOR2<T> operator * ( const VECTOR2& other ) const {return VECTOR2<T>(x*other.x,y*other.y);}
	VECTOR2<T> operator / ( const VECTOR2& other ) const {return VECTOR2<T>(x/other.x,y/other.y);}

	int area() const {return x*y;}
	float length() const {return sqrt(float(x*x+y*y));}
	void normalize() {float len = length(); x/=len;y/=len;}
	T maxVal() const {return MAX(x,y);}
	T minVal() const {return MIN(x,y);}

	VECTOR2<T> makepow2() const {
		VECTOR2<T> vOut;
		vOut.x = T(1<<int(ceil(log(T(x))/log(2.0))));
		vOut.y = T(1<<int(ceil(log(T(y))/log(2.0))));
		return vOut;
	}

	#ifdef __D3DX9MATH_H__
		VECTOR2<T>(const D3DXVECTOR2 &other): x(T(other.x)), y(T(other.y)) {}
		D3DXVECTOR2 toD2DXVEC() const {return D3DXVECTOR2(float(x),float(y));}
		bool operator == ( const D3DXVECTOR2& other ) const {return (other.x==T(x) && other.y== T(y)); }
		bool operator != ( const D3DXVECTOR2& other ) const {return (other.x!=T(x) || other.y!= T(y)); }
	#endif
};

template <class T=int> class VECTOR3 {
public:
	T x;
	T y;
	T z;

	VECTOR3(): x(0), y(0),z(0) {}
	VECTOR3(const VECTOR3<T> &other): x(other.x), y(other.y), z(other.z) {}
	VECTOR3(const T _x, const T _y, const T _z) : x(_x), y(_y), z(_z) {}

	bool operator == ( const VECTOR3<T>& other ) const {return (other.x==x && other.y==y && other.z==z); }
	bool operator != ( const VECTOR3<T>& other ) const {return (other.x!=x || other.y!=y || other.z!=z); } 

    // binary operators with scalars
	VECTOR3<T> operator + ( T scalar ) const {return VECTOR3<T>(x+scalar,y+scalar,z+scalar);}
	VECTOR3<T> operator - ( T scalar ) const {return VECTOR3<T>(x-scalar,y-scalar,z-scalar);}
	VECTOR3<T> operator * ( T scalar ) const {return VECTOR3<T>(x*scalar,y*scalar,z*scalar);}
	VECTOR3<T> operator / ( T scalar ) const {return VECTOR3<T>(x/scalar,y/scalar,z/scalar);}

	// binaray operpators  with vectors
	VECTOR3<T> operator + ( const VECTOR3<T>& other ) const {return VECTOR3<T>(x+other.x,y+other.y,z+other.z);}
	VECTOR3<T> operator - ( const VECTOR3<T>& other ) const {return VECTOR3<T>(x-other.x,y-other.y,z-other.z);}
	VECTOR3<T> operator * ( const VECTOR3<T>& other ) const {return VECTOR3<T>(x*other.x,y*other.y,z*other.z);}
	VECTOR3<T> operator / ( const VECTOR3<T>& other ) const {return VECTOR3<T>(x/other.x,y/other.y,z/other.z);}

	T maxVal() const {return MAX(x,MAX(y,z));}
	T minVal() const {return MIN(x,MIN(y,z));}
	T volume() const {return x*y*z;}
	float length() const {return sqrt(float(x*x+y*y+z*z));}
	void normalize() {float len = length(); x/=len;y/=len;z/=len;}
	void normalize(float epsilon, const VECTOR3<T> replacement=VECTOR3<T>(T(0),T(0),T(1))) {
		float len = length();
		if (len > epsilon) {
			x/=len;
			y/=len;
			z/=len;
		} else { // specify some arbitrary normal
			x = replacement.x;
			y = replacement.y;
			z = replacement.z;
		}
	}

	VECTOR2<T> xx() {return VECTOR2<T>(x,x);}
	VECTOR2<T> xy() {return VECTOR2<T>(x,y);}
	VECTOR2<T> xz() {return VECTOR2<T>(x,z);}
	VECTOR2<T> yx() {return VECTOR2<T>(y,x);}
	VECTOR2<T> yy() {return VECTOR2<T>(y,y);}
	VECTOR2<T> yz() {return VECTOR2<T>(y,z);}
	VECTOR2<T> zx() {return VECTOR2<T>(z,x);}
	VECTOR2<T> zy() {return VECTOR2<T>(z,y);}
	VECTOR2<T> zz() {return VECTOR2<T>(z,z);}

	VECTOR3<T> makepow2() const  {
		VECTOR3<T> vOut;
		vOut.x = T(1<<int(ceil(log(float(x))/log(2.0))));
		vOut.y = T(1<<int(ceil(log(float(y))/log(2.0))));
		vOut.z = T(1<<int(ceil(log(float(z))/log(2.0))));
		return vOut;
	}

	#ifdef __D3DX9MATH_H__
		VECTOR3(const D3DXVECTOR3 &other): x(T(other.x)), y(T(other.y)), z(T(other.z)) {}
		D3DXVECTOR3 toD3DXVEC() {return D3DXVECTOR3(float(x),float(y),float(z));}
		bool operator == ( const D3DXVECTOR3& other ) const {return (other.x==T(x) && other.y== T(y) && other.z== T(z)); }
		bool operator != ( const D3DXVECTOR3& other ) const {return (other.x!=T(x) || other.y!= T(y) || other.z!= T(z)); }
	#endif
};

template <class T=int> class VECTOR4 {
public:
	T x;
	T y;
	T z;
	T w;

	VECTOR4<T>(): x(0), y(0),z(0), w(0) {}
	VECTOR4<T>(const VECTOR4<T> &other): x(other.x), y(other.y), z(other.z), w(other.w) {}
	VECTOR4<T>(const T _x, const T _y, const T _z, const T _w) : x(_x), y(_y), z(_z), w(_w) {}

	bool operator == ( const VECTOR4<T>& other ) const {return (other.x==x && other.y==y && other.z==z && other.w==w); }
	bool operator != ( const VECTOR4<T>& other ) const {return (other.x!=x || other.y!=y || other.z!=z || other.w!=w); } 

    // binary operators with scalars
	VECTOR4<T> operator + ( T scalar ) const {return VECTOR4<T>(x+scalar,y+scalar,z+scalar,w+scalar);}
	VECTOR4<T> operator - ( T scalar ) const {return VECTOR4<T>(x-scalar,y-scalar,z-scalar,w-scalar);}
	VECTOR4<T> operator * ( T scalar ) const {return VECTOR4<T>(x*scalar,y*scalar,z*scalar,w*scalar);}
	VECTOR4<T> operator / ( T scalar ) const {return VECTOR4<T>(x/scalar,y/scalar,z/scalar,w/scalar);}

	// binaray operpators  with intvectors
	VECTOR4<T> operator + ( const VECTOR4<T>& other ) const {return VECTOR4<T>(x+other.x,y+other.y,z+other.z,w+other.w);}
	VECTOR4<T> operator - ( const VECTOR4<T>& other ) const {return VECTOR4<T>(x-other.x,y-other.y,z-other.z,w-other.w);}
	VECTOR4<T> operator * ( const VECTOR4<T>& other ) const {return VECTOR4<T>(x*other.x,y*other.y,z*other.z,w*other.w);}
	VECTOR4<T> operator / ( const VECTOR4<T>& other ) const {return VECTOR4<T>(x/other.x,y/other.y,z/other.z,w/other.w);}

	VECTOR4<T> makepow2() const  {
		VECTOR4<T> vOut;
		vOut.x = T(1<<int(ceil(log(float(x))/log(2.0))));
		vOut.y = T(1<<int(ceil(log(float(y))/log(2.0))));
		vOut.z = T(1<<int(ceil(log(float(z))/log(2.0))));
		vOut.w = T(1<<int(ceil(log(float(w))/log(2.0))));
		return vOut;
	}

	#ifdef __D3DX9MATH_H__
		VECTOR4<T>(const D3DXVECTOR4 &other): x(T(other.x)), y(T(other.y)), z(T(other.z)), w(T(other.w)){}
		D3DXVECTOR4 toD4DXVEC() {return D3DXVECTOR4(float(x),float(y),float(z),float(w));}

		bool operator == ( const D3DXVECTOR4& other ) const {return (other.x==T(x) && other.y==T(y) && other.z==T(z) && other.w==T(w)); }
		bool operator != ( const D3DXVECTOR4& other ) const {return (other.x!=T(x) || other.y!=T(y) || other.z!=T(z) || other.w!=T(w)); }
	#endif

};

#ifdef __D3DX9MATH_H__
	typedef VECTOR4<D3DXFLOAT16> D3DXVECTOR4_16;
#endif

typedef VECTOR4<> INTVECTOR4;
typedef VECTOR3<> INTVECTOR3;
typedef VECTOR2<> INTVECTOR2;

typedef VECTOR4<float> FLOATVECTOR4;
typedef VECTOR3<float> FLOATVECTOR3;
typedef VECTOR2<float> FLOATVECTOR2;