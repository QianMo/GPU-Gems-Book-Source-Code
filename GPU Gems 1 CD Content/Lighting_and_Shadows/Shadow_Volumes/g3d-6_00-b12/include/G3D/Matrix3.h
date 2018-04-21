/**
  @file Matrix3.h
 
  3x3 matrix class
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @cite Portions based on Dave Eberly's Magic Software Library at <A HREF="http://www.magic-software.com">http://www.magic-software.com</A>
 
  @created 2001-06-02
  @edited  2003-11-22
 */

#ifndef G3D_MATRIX3_H
#define G3D_MATRIX3_H

#include "G3D/Vector3.h"
#include "G3D/Vector4.h"
#include "G3D/debugAssert.h"

namespace G3D {

/**
  3x3 matrix.  Do not subclass.
 */
class Matrix3 {
private:
    /**
     Constructor.  Private so there is no confusion about whether
     if is initialized, zero, or identity.
     */
    Matrix3 ();

public:

    Matrix3(class BinaryInput& b);
    Matrix3 (const float aafEntry[3][3]);
    Matrix3 (const Matrix3& rkMatrix);
    Matrix3 (float fEntry00, float fEntry01, float fEntry02,
             float fEntry10, float fEntry11, float fEntry12,
             float fEntry20, float fEntry21, float fEntry22);

    void serialize(class BinaryOutput& b) const;
    void deserialize(class BinaryInput& b);

    /**
     Sets all elements.
     */
    void set(float fEntry00, float fEntry01, float fEntry02,
             float fEntry10, float fEntry11, float fEntry12,
             float fEntry20, float fEntry21, float fEntry22);

    /**
     * member access, allows use of construct mat[r][c]
     */
    float* operator[] (int iRow) const;
    operator float* ();
    Vector3 getColumn (int iCol) const;
    Vector3 getRow (int iRow) const;
    void setColumn(int iCol, const Vector3 &vector);
    void setRow(int iRow, const Vector3 &vector);

    // assignment and comparison
    Matrix3& operator= (const Matrix3& rkMatrix);
    bool operator== (const Matrix3& rkMatrix) const;
    bool operator!= (const Matrix3& rkMatrix) const;

    // arithmetic operations
    Matrix3 operator+ (const Matrix3& rkMatrix) const;
    Matrix3 operator- (const Matrix3& rkMatrix) const;
    Matrix3 operator* (const Matrix3& rkMatrix) const;
    Matrix3 operator- () const;

    /**
     * matrix * vector [3x3 * 3x1 = 3x1]
     */
    Vector3 operator* (const Vector3& rkVector) const;

    /**
     * vector * matrix [1x3 * 3x3 = 1x3]
     */
    friend Vector3 operator* (const Vector3& rkVector,
                              const Matrix3& rkMatrix);

    /**
     * matrix * scalar
     */
    Matrix3 operator* (float fScalar) const;

    /** scalar * matrix */
    friend Matrix3 operator* (float fScalar, const Matrix3& rkMatrix);

    // utilities
    Matrix3 transpose () const;
    bool inverse (Matrix3& rkInverse, float fTolerance = 1e-06) const;
    Matrix3 inverse (float fTolerance = 1e-06) const;
    float determinant () const;

    /** singular value decomposition */
    void singularValueDecomposition (Matrix3& rkL, Vector3& rkS,
                                     Matrix3& rkR) const;
    /** singular value decomposition */
    void singularValueComposition (const Matrix3& rkL,
                                   const Vector3& rkS, const Matrix3& rkR);

    /** Gram-Schmidt orthonormalization (applied to columns of rotation matrix) */
    void orthonormalize ();

    /** orthogonal Q, diagonal D, upper triangular U stored as (u01,u02,u12) */
    void qDUDecomposition (Matrix3& rkQ, Vector3& rkD,
                           Vector3& rkU) const;

    float spectralNorm () const;

    /** matrix must be orthonormal */
    void toAxisAngle (Vector3& rkAxis, float& rfRadians) const;
    void fromAxisAngle (const Vector3& rkAxis, float fRadians);

    /**
     * The matrix must be orthonormal.  The decomposition is yaw*pitch*roll
     * where yaw is rotation about the Up vector, pitch is rotation about the
     * right axis, and roll is rotation about the Direction axis.
     */
    bool toEulerAnglesXYZ (float& rfYAngle, float& rfPAngle,
                           float& rfRAngle) const;
    bool toEulerAnglesXZY (float& rfYAngle, float& rfPAngle,
                           float& rfRAngle) const;
    bool toEulerAnglesYXZ (float& rfYAngle, float& rfPAngle,
                           float& rfRAngle) const;
    bool toEulerAnglesYZX (float& rfYAngle, float& rfPAngle,
                           float& rfRAngle) const;
    bool toEulerAnglesZXY (float& rfYAngle, float& rfPAngle,
                           float& rfRAngle) const;
    bool toEulerAnglesZYX (float& rfYAngle, float& rfPAngle,
                           float& rfRAngle) const;
    void fromEulerAnglesXYZ (float fYAngle, float fPAngle, float fRAngle);
    void fromEulerAnglesXZY (float fYAngle, float fPAngle, float fRAngle);
    void fromEulerAnglesYXZ (float fYAngle, float fPAngle, float fRAngle);
    void fromEulerAnglesYZX (float fYAngle, float fPAngle, float fRAngle);
    void fromEulerAnglesZXY (float fYAngle, float fPAngle, float fRAngle);
    void fromEulerAnglesZYX (float fYAngle, float fPAngle, float fRAngle);

    /** eigensolver, matrix must be symmetric */
    void eigenSolveSymmetric (float afEigenvalue[3],
                              Vector3 akEigenvector[3]) const;

    static void tensorProduct (const Vector3& rkU, const Vector3& rkV,
                               Matrix3& rkProduct);

    static const float EPSILON;
    static const Matrix3 ZERO;
    static const Matrix3 IDENTITY;

protected:
    // support for eigensolver
    void tridiagonal (float afDiag[3], float afSubDiag[3]);
    bool qLAlgorithm (float afDiag[3], float afSubDiag[3]);

    // support for singular value decomposition
    static const float ms_fSvdEpsilon;
    static const int ms_iSvdMaxIterations;
    static void bidiagonalize (Matrix3& kA, Matrix3& kL,
                               Matrix3& kR);
    static void golubKahanStep (Matrix3& kA, Matrix3& kL,
                                Matrix3& kR);

    // support for spectral norm
    static float maxCubicRoot (float afCoeff[3]);

    float m_aafEntry[3][3];
};


} // namespace

#endif

