// from C4Dfx by Jörn Loviscach, www.l7h.cn
// some matrix functions, inspired by NVIDIA's Cg demos

#if !defined(MATRIXMATH_H)
#define MATRIXMATH_H

namespace MatrixMath
{
	void Transpose(const float A[16], float B[16]);
	float Det(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3);
	void Invert(const float A[16], float B[16]);
	void Mult(const float A[16], const float B[16], float C[16]);
};

#endif