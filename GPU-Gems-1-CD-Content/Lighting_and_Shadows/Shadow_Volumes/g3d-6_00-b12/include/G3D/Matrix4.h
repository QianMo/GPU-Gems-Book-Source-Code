/**
  @file Matrix4.h
 
  4x4 matrix class
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2003-10-02
  @edited  2004-01-04
 */

#ifndef G3D_MATRIX4_H
#define G3D_MATRIX4_H

namespace G3D {

/**
 Not full featured.  Consider G3D::CoordinateFrame instead.
 */
class Matrix4 {
private:

    float elt[4][4];

    /**
      Computes the determinant of the 3x3 matrix that lacks excludeRow
      and excludeCol. 
    */
    double subDeterminant(int excludeRow, int excludeCol) const;

public:
    Matrix4(
        float r1c1, float r1c2, float r1c3, float r1c4,
        float r2c1, float r2c2, float r2c3, float r2c4,
        float r3c1, float r3c2, float r3c3, float r3c4,
        float r4c1, float r4c2, float r4c3, float r4c4);

    /**
     init should be <B>row major</B>.
     */
    Matrix4(const float* init);

    Matrix4(const class CoordinateFrame& c);

    Matrix4(const double* init);

    Matrix4();

    static const Matrix4 IDENTITY;
    static const Matrix4 ZERO;

    const float* operator[](int r) const;
    float* operator[](int r);

    Matrix4 operator*(const Matrix4& other) const;

    /**
     Constructs an orthogonal projection matrix from the given parameters.
     */
    static Matrix4 orthogonalProjection(
        double            left,
        double            right,
        double            bottom,
        double            top,
        double            nearval,
        double            farval);

    static Matrix4 perspectiveProjection(
        double            left,
        double            right,
        double            bottom,
        double            top,
        double            nearval,
        double            farval);

    void setRow(int r, const class Vector4& v);
    void setColumn(int c, const Vector4& v);
    Vector4 getRow(int r) const;
    Vector4 getColumn(int c) const;

    Matrix4 operator*(const double s) const;
    Vector4 operator*(const Vector4& vector) const;

    Matrix4 transpose() const;

    bool operator!=(const Matrix4& other) const;
    bool operator==(const Matrix4& other) const;

    double determinant() const;
    Matrix4 inverse() const;
    Matrix4 adjoint() const;
    Matrix4 cofactor() const;
};

} // namespace

#endif

