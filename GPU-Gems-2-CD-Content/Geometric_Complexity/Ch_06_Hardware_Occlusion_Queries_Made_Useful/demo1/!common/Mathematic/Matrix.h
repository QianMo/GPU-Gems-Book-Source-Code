//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef MatrixH
#define MatrixH
#include "Tuppel.h"
#include "Vector4.h"
//---------------------------------------------------------------------------
namespace Math {

template<class REAL = float, const unsigned COL = 4, const unsigned ROW = 4>
struct Matrix : public Tuppel<REAL,COL*ROW> {
	//array[COL*ROW];   // 00    ROW .  .  (COL-1)*ROW
						// 01     .  .  .       .
						// .      .  .  .       .
						// ROW-1  .  .  .   COL*ROW-1

	Matrix() { /* empty -> performance */ }
	Matrix(const Matrix& m): Tuppel<REAL,COL*ROW>(m) { }
	Matrix(const REAL p[COL*ROW]): Tuppel<REAL,COL*ROW>(p) { }

	static const unsigned colCount() { return COL; }
	static const unsigned rowCount() { return ROW; }

	const REAL operator()(const unsigned col, const unsigned row) const { return array[col-1+ROW*(row-1)]; }
	REAL& operator()(const unsigned col, const unsigned row) { return array[col-1+ROW*(row-1)]; }

	Matrix<REAL,ROW,COL> transpose(const Matrix<REAL,COL,ROW>& m) { }
};

//namespace
};
#endif
