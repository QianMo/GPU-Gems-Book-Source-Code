// from C4Dfx by Jörn Loviscach, www.l7h.cn
// some matrix functions, inspired by NVIDIA's Cg demos

#include "Matrixmath.h"
#include <math.h>

void MatrixMath::Transpose(const float A[16], float B[16])
{
	int i, j;
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            B[i*4+j] = A[j*4+i];
}

float MatrixMath::Det(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3)
{
    return a1 * (b2*c3 - b3*c2) - b1 * (a2*c3 - a3*c2) + c1 * (a2*b3 - a3*b2);
}

void MatrixMath::Invert(const float A[16], float B[16])
{
    B[0] =  Det(A[5], A[6], A[7], A[9], A[10], A[11], A[13], A[14], A[15]);
    B[1] = -Det(A[1], A[2], A[3], A[9], A[10], A[11], A[13], A[14], A[15]);
    B[2] =  Det(A[1], A[2], A[3], A[5], A[6], A[7], A[13], A[14], A[15]);
    B[3] = -Det(A[1], A[2], A[3], A[5], A[6], A[7], A[9], A[10], A[11]);
    B[4] = -Det(A[4], A[6], A[7], A[8], A[10], A[11], A[12], A[14], A[15]);
    B[5] =  Det(A[0], A[2], A[3], A[8], A[10], A[11], A[12], A[14], A[15]);
    B[6] = -Det(A[0], A[2], A[3], A[4], A[6], A[7], A[12], A[14], A[15]);
    B[7] =  Det(A[0], A[2], A[3], A[4], A[6], A[7], A[8], A[10], A[11]);
    B[8] =  Det(A[4], A[5], A[7], A[8], A[9], A[11], A[12], A[13], A[15]);
    B[9] = -Det(A[0], A[1], A[3], A[8], A[9], A[11], A[12], A[13], A[15]);
    B[10] =  Det(A[0], A[1], A[3], A[4], A[5], A[7], A[12], A[13], A[15]);
    B[11] = -Det(A[0], A[1], A[3], A[4], A[5], A[7], A[8], A[9], A[11]);
    B[12] = -Det(A[4], A[5], A[6], A[8], A[9], A[10], A[12], A[13], A[14]);
    B[13] =  Det(A[0], A[1], A[2], A[8], A[9], A[10], A[12], A[13], A[14]);
    B[14] = -Det(A[0], A[1], A[2], A[4], A[5], A[6], A[12], A[13], A[14]);
    B[15] =  Det(A[0], A[1], A[2], A[4], A[5], A[6], A[8], A[9], A[10]);
    float det = A[0] * B[0] + A[4] * B[1] + A[8] * B[2] + A[12] * B[3];
    det = 1.0f / det;
    B[0] *= det;
    B[1] *= det;
    B[2] *= det;
    B[3] *= det;
    B[4] *= det;
    B[5] *= det;
    B[6] *= det;
    B[7] *= det;
    B[8] *= det;
    B[9] *= det;
    B[10] *= det;
    B[11] *= det;
    B[12] *= det;
    B[13] *= det;
    B[14] *= det;
    B[15] *= det;
}

void MatrixMath::Mult(const float A[15], const float B[15], float C[15])
//for column-major matrices
{
    C[0] = A[0] * B[0] + A[4] * B[1] + A[8] * B[2] + A[12] * B[3];
    C[1] = A[1] * B[0] + A[5] * B[1] + A[9] * B[2] + A[13] * B[3];
    C[2] = A[2] * B[0] + A[6] * B[1] + A[10] * B[2] + A[14] * B[3];
    C[3] = A[3] * B[0] + A[7] * B[1] + A[11] * B[2] + A[15] * B[3];
    C[4] = A[0] * B[4] + A[4] * B[5] + A[8] * B[6] + A[12] * B[7];
    C[5] = A[1] * B[4] + A[5] * B[5] + A[9] * B[6] + A[13] * B[7];
    C[6] = A[2] * B[4] + A[6] * B[5] + A[10] * B[6] + A[14] * B[7];
    C[7] = A[3] * B[4] + A[7] * B[5] + A[11] * B[6] + A[15] * B[7];
    C[8] = A[0] * B[8] + A[4] * B[9] + A[8] * B[10] + A[12] * B[11];
    C[9] = A[1] * B[8] + A[5] * B[9] + A[9] * B[10] + A[13] * B[11];
    C[10] = A[2] * B[8] + A[6] * B[9] + A[10] * B[10] + A[14] * B[11];
    C[11] = A[3] * B[8] + A[7] * B[9] + A[11] * B[10] + A[15] * B[11];
    C[12] = A[0] * B[12] + A[4] * B[13] + A[8] * B[14] + A[12] * B[15];
    C[13] = A[1] * B[12] + A[5] * B[13] + A[9] * B[14] + A[13] * B[15];
    C[14] = A[2] * B[12] + A[6] * B[13] + A[10] * B[14] + A[14] * B[15];
    C[15] = A[3] * B[12] + A[7] * B[13] + A[11] * B[14] + A[15] * B[15];
}