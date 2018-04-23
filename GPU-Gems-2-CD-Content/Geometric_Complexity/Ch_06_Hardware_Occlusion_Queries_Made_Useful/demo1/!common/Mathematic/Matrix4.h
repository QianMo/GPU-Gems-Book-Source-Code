//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef Matrix4H
#define Matrix4H
#include "Vector3.h"
#include "Matrix.h"
//---------------------------------------------------------------------------
namespace Math {

template<class REAL = float>
class Matrix4 : public Matrix<REAL,4,4> {
protected:
	//array[16];    // 00  04  08  12
					// 01  05  09  13
					// 02  06  10  14
					// 03  07  11  15
	typedef Vector3<REAL> V3;
	typedef Matrix<REAL,4,4> M4;

	Matrix4& mulUnSave(const Matrix4& a, const Matrix4& b) {
		const unsigned SIZE = 4;
		for(unsigned iCol = 0; iCol < SIZE; iCol++) {
			const unsigned cID = iCol*SIZE;
			for(unsigned iRow = 0; iRow < SIZE; iRow++) {
				const unsigned id = iRow+cID;
				array[id] = a.array[iRow]*b.array[cID];
				for(unsigned k = 1; k < SIZE; k++) {
					array[id] += a.array[iRow+k*SIZE]*b.array[k+cID];
				}
			}
		}
		return *this;
	}

public:
	Matrix4() { /* empty -> performance */ }
	Matrix4(const Matrix4& m): M4(m) { }
	Matrix4(const M4& m): M4(m) { }
	Matrix4(const Tuppel<REAL,16>& m): M4(m.addr()) { }
	Matrix4(const REAL p[16]): M4(p) { }

	Matrix4(
				const REAL&  m11, const REAL& m12, const REAL&  m13, const REAL&  m14,
				const REAL&  m21, const REAL& m22, const REAL&  m23, const REAL&  m24,
				const REAL&  m31, const REAL& m32, const REAL&  m33, const REAL&  m34,
				const REAL&  m41, const REAL& m42, const REAL&  m43, const REAL&  m44) {
		array[0] = m11;  array[4] = m12;  array[ 8] = m13;  array[12] = m14;
		array[1] = m21;  array[5] = m22;  array[ 9] = m23;  array[13] = m24;
		array[2] = m31;  array[6] = m32;  array[10] = m33;  array[14] = m34;
		array[3] = m41;  array[7] = m42;  array[11] = m43;  array[15] = m44;
	}

	REAL& a11() { return array[0]; }
	const REAL& a11() const { return array[0]; }
	REAL& a21() { return array[1]; }
	const REAL& a21() const { return array[1]; }
	REAL& a31() { return array[2]; }
	const REAL& a31() const { return array[2]; }
	REAL& a41() { return array[3]; }
	const REAL& a41() const { return array[3]; }

	REAL& a12() { return array[4]; }
	const REAL& a12() const { return array[4]; }
	REAL& a22() { return array[5]; }
	const REAL& a22() const { return array[5]; }
	REAL& a32() { return array[6]; }
	const REAL& a32() const { return array[6]; }
	REAL& a42() { return array[7]; }
	const REAL& a42() const { return array[7]; }

	REAL& a13() { return array[8]; }
	const REAL& a13() const { return array[8]; }
	REAL& a23() { return array[9]; }
	const REAL& a23() const { return array[9]; }
	REAL& a33() { return array[10]; }
	const REAL& a33() const { return array[10]; }
	REAL& a43() { return array[11]; }
	const REAL& a43() const { return array[11]; }

	REAL& a14() { return array[12]; }
	const REAL& a14() const { return array[12]; }
	REAL& a24() { return array[13]; }
	const REAL& a24() const { return array[13]; }
	REAL& a34() { return array[14]; }
	const REAL& a34() const { return array[14]; }
	REAL& a44() { return array[15]; }
	const REAL& a44() const { return array[15]; }

	Matrix4& transpose() { 
		std::swap(array[ 1],array[ 4]);
		std::swap(array[ 2],array[ 8]);
		std::swap(array[ 3],array[12]);
		std::swap(array[ 6],array[ 9]);
		std::swap(array[ 7],array[13]);
		std::swap(array[11],array[14]);
		return *this;
	}

	Matrix4& operator*=(const REAL& r) {
		M4::operator*=(r);
		return *this;
	}

	Matrix4& operator*=(const Matrix4& lValue) {
		if(lValue.addr() == addr()) {
			Matrix4 tmp(lValue);
			return mulUnSave(tmp,tmp);
		}
		else {
			return mulUnSave(Matrix4(*this),lValue);
		}
	}

	Matrix4 operator*(const Matrix4& a) {
		return Matrix4().mulUnSave(*this,a);
	}

	
	//output = i^(-1)
	void invert(const Matrix4& i) {
		double a11 =  det3x3(i[5],i[6],i[7],i[9],i[10],i[11],i[13],i[14],i[15]);
		double a21 = -det3x3(i[1],i[2],i[3],i[9],i[10],i[11],i[13],i[14],i[15]);
		double a31 =  det3x3(i[1],i[2],i[3],i[5],i[6],i[7],i[13],i[14],i[15]);
		double a41 = -det3x3(i[1],i[2],i[3],i[5],i[6],i[7],i[9],i[10],i[11]);

		double a12 = -det3x3(i[4],i[6],i[7],i[8],i[10],i[11],i[12],i[14],i[15]);
		double a22 =  det3x3(i[0],i[2],i[3],i[8],i[10],i[11],i[12],i[14],i[15]);
		double a32 = -det3x3(i[0],i[2],i[3],i[4],i[6],i[7],i[12],i[14],i[15]);
		double a42 =  det3x3(i[0],i[2],i[3],i[4],i[6],i[7],i[8],i[10],i[11]);

		double a13 =  det3x3(i[4],i[5],i[7],i[8],i[9],i[11],i[12],i[13],i[15]);
		double a23 = -det3x3(i[0],i[1],i[3],i[8],i[9],i[11],i[12],i[13],i[15]);
		double a33 =  det3x3(i[0],i[1],i[3],i[4],i[5],i[7],i[12],i[13],i[15]);
		double a43 = -det3x3(i[0],i[1],i[3],i[4],i[5],i[7],i[8],i[9],i[11]);

		double a14 = -det3x3(i[4],i[5],i[6],i[8],i[9],i[10],i[12],i[13],i[14]);
		double a24 =  det3x3(i[0],i[1],i[2],i[8],i[9],i[10],i[12],i[13],i[14]);
		double a34 = -det3x3(i[0],i[1],i[2],i[4],i[5],i[6],i[12],i[13],i[14]);
		double a44 =  det3x3(i[0],i[1],i[2],i[4],i[5],i[6],i[8],i[9],i[10]);

		double det = (i[0]*a11) + (i[4]*a21) + (i[8]*a31) + (i[12]*a41);
		double oodet = 1/det;

		array[ 0] = a11*oodet;
		array[ 1] = a21*oodet;
		array[ 2] = a31*oodet;
		array[ 3] = a41*oodet;

		array[ 4] = a12*oodet;
		array[ 5] = a22*oodet;
		array[ 6] = a32*oodet;
		array[ 7] = a42*oodet;

		array[ 8] = a13*oodet;
		array[ 9] = a23*oodet;
		array[10] = a33*oodet;
		array[11] = a43*oodet;

		array[12] = a14*oodet;
		array[13] = a24*oodet;
		array[14] = a34*oodet;
		array[15] = a44*oodet;
	}

	void invert() { invert(*this); }
	Matrix4 getInverse() const { return Matrix4().invert(*this); }

	//calc matrix-vector product; input has assumed homogenous component w = 1
	//before the output is  written homogen division is performed (w = 1)
	V3 mulHomogenPoint(const V3& v) const {
		//if v == output -> overwriting problems -> so store in temp
		double x = array[0]*v[0] + array[4]*v[1] + array[ 8]*v[2] + array[12];
		double y = array[1]*v[0] + array[5]*v[1] + array[ 9]*v[2] + array[13];
		double z = array[2]*v[0] + array[6]*v[1] + array[10]*v[2] + array[14];
		double w = array[3]*v[0] + array[7]*v[1] + array[11]*v[2] + array[15];

		return V3(x/w, y/w, z/w);
	}

	static const Matrix4 IDENTITY;
	static const Matrix4 ONE;
	static const Matrix4 T05_S05;
	static const Matrix4 ZERO;
};

typedef Matrix4<float> Matrix4f;
typedef Matrix4<double> Matrix4d;

//namespace
}
#endif
