#pragma once

// 3D vector class
typedef struct Vector3
{
  float x, y, z;

  inline Vector3() {}
  inline Vector3(float fX, float fY, float fZ) : x(fX), y(fY), z(fZ) {}

  inline bool operator == (const Vector3 &vOther) { return (x == vOther.x && y == vOther.y && z == vOther.z); }
  inline bool operator != (const Vector3 &vOther) { return (x != vOther.x || y != vOther.y || z != vOther.z); }
  inline Vector3 &operator /= (const float fScalar) { x /= fScalar; y /= fScalar; z /= fScalar; return (*this); }
  inline Vector3 &operator *= (const float fScalar) { x *= fScalar; y *= fScalar; z *= fScalar; return (*this); }
  inline Vector3 &operator += (const Vector3 &vOther) { x += vOther.x;  y += vOther.y; z += vOther.z; return (*this); }
  inline Vector3 &operator -= (const Vector3 &vOther) { x -= vOther.x;  y -= vOther.y; z -= vOther.z; return (*this);  }

  inline Vector3 operator * (float fScalar) const { return Vector3(x * fScalar, y * fScalar, z * fScalar); }
  inline Vector3 operator / (float fScalar) const { return Vector3(x / fScalar, y / fScalar, z / fScalar); }
  inline Vector3 operator + (const Vector3 &vOther) const { return Vector3(x + vOther.x, y + vOther.y, z + vOther.z); }
  inline Vector3 operator - (const Vector3 &vOther) const { return Vector3(x - vOther.x, y - vOther.y, z - vOther.z); }
  inline Vector3 operator - (void) const { return Vector3(-x, -y, -z); }

  // calculate length of vector
  inline float Length(void) const { return (float) sqrt(x * x + y * y + z * z); }

  // normalize vector (returns length)
  inline float Normalize(void)
  {
    float fLength = Length();
    if (fLength == 0.0f) return 0.0f;
    float fInvLength = 1.0f / fLength;
    (*this) *= fInvLength;
    return fLength;
  }
} Vector3;

// normalize vector (returns length)
inline Vector3 Normalize(const Vector3 &vOther)
{
  float fLength = vOther.Length();
  if (fLength == 0.0f) return vOther;
  float fInvLength = 1.0f / fLength;
  return vOther * fInvLength;
}

// calculate cross product
inline Vector3 Cross(const Vector3 &vA, const Vector3 &vB)
{
  return Vector3( (vA.y * vB.z) - (vA.z * vB.y),
                  (vA.z * vB.x) - (vA.x * vB.z),
                  (vA.x * vB.y) - (vA.y * vB.x) );
}

// calculate dot product
inline float Dot(const Vector3 &vA, const Vector3 &vB)
{
  return (vA.x * vB.x) + (vA.y * vB.y) + (vA.z * vB.z);
}

//////////////////////////////////////////////////////////////////////////////////////////

// 4D vector class
typedef struct Vector4
{
  float x, y, z, w;

  inline Vector4() {}
  inline Vector4(float fX, float fY, float fZ, float fW) : x(fX), y(fY), z(fZ), w(fW) {}
  inline Vector4(const Vector3 &v, float fW) : x(v.x), y(v.y), z(v.z), w(fW) {}

  inline bool operator == (const Vector4 &vOther) { return (x == vOther.x && y == vOther.y && z == vOther.z && w == vOther.w); }
  inline bool operator != (const Vector4 &vOther) { return (x != vOther.x || y != vOther.y || z != vOther.z || w != vOther.w); }
  inline Vector4 &operator /= (const float fScalar) { x /= fScalar; y /= fScalar; z /= fScalar; w /= fScalar; return (*this); }
  inline Vector4 &operator *= (const float fScalar) { x *= fScalar; y *= fScalar; z *= fScalar; w *= fScalar; return (*this); }
  inline Vector4 &operator += (const Vector4 &vOther) { x += vOther.x;  y += vOther.y; z += vOther.z; w += vOther.w; return (*this); }
  inline Vector4 &operator -= (const Vector4 &vOther) { x -= vOther.x;  y -= vOther.y; z -= vOther.z; w -= vOther.w; return (*this);  }

  inline Vector4 operator * (float fScalar) const { return Vector4(x * fScalar, y * fScalar, z * fScalar, w * fScalar); }
  inline Vector4 operator / (float fScalar) const { return Vector4(x / fScalar, y / fScalar, z / fScalar, w / fScalar); }
  inline Vector4 operator + (const Vector4 &vOther) const { return Vector4(x + vOther.x, y + vOther.y, z + vOther.z, w + vOther.w); }
  inline Vector4 operator - (const Vector4 &vOther) const { return Vector4(x - vOther.x, y - vOther.y, z - vOther.z, z + vOther.z); }
  inline Vector4 operator - (void) const { return Vector4(-x, -y, -z, -w); }

  // calculate length of vector
  inline float Length(void) const { return (float) sqrt(x * x + y * y + z * z + w * w); }

  // normalize vector (returns length)
  inline float Normalize(void)
  {
    float fLength = Length();
    if (fLength == 0.0f) return 0.0f;
    float fInvLength = 1.0f / fLength;
    (*this) *= fInvLength;
    return fLength;
  }
} Vector4;

// normalize vector (returns length)
inline Vector4 Normalize(const Vector4 &vOther)
{
  float fLength = vOther.Length();
  if (fLength == 0.0f) return vOther;
  float fInvLength = 1.0f / fLength;
  return vOther * fInvLength;
}

// calculate dot product
inline float Dot(const Vector4 &vA, const Vector4 &vB)
{
  return (vA.x * vB.x) + (vA.y * vB.y) + (vA.z * vB.z) + (vA.w * vB.w);
}

//////////////////////////////////////////////////////////////////////////////////////////////

// 4x4 matrix class
typedef struct Matrix
{
  union
  {
    struct
    {
      float _11, _12, _13, _14;
      float _21, _22, _23, _24;
      float _31, _32, _33, _34;
      float _41, _42, _43, _44;
    };
    float m[4][4];
  };

  inline Matrix() {}
  inline Matrix( float f11, float f12, float f13, float f14,
                  float f21, float f22, float f23, float f24,
                  float f31, float f32, float f33, float f34,
                  float f41, float f42, float f43, float f44 ) :
    _11(f11), _12(f12), _13(f13), _14(f14),
    _21(f21), _22(f22), _23(f23), _24(f24),
    _31(f31), _32(f32), _33(f33), _34(f34),
    _41(f41), _42(f42), _43(f43), _44(f44) {}

  // set matrix to identity
  inline void SetIdentity(void) { (*this) = Matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1); }

  // set rotation from given euler angles (clears _11 to _33)
  inline void SetRotation(const Vector3 &vEuler);

  // set translation elements of matrix
  inline void SetTranslation(const Vector3 &vPos);

  // scale matrix by given multipliers
  inline void Scale(const Vector3 &vMultiplier);

  // multiply two matrices together
  inline Matrix operator * (const Matrix &B) const;

} Matrix;

// transforms vector with matrix
inline Vector4 Transform(const Vector4 &A, const Matrix &M)
{
  return Vector4((A.x * M._11) + (A.y * M._21) + (A.z * M._31) + (A.w * M._41),
                 (A.x * M._12) + (A.y * M._22) + (A.z * M._32) + (A.w * M._42),
                 (A.x * M._13) + (A.y * M._23) + (A.z * M._33) + (A.w * M._43),
                 (A.x * M._14) + (A.y * M._24) + (A.z * M._34) + (A.w * M._44));
}

// transforms vector with matrix
inline Vector4 Transform(const Vector3 &A, const Matrix &M)
{
  return Transform(Vector4(A.x,A.y,A.z,1), M);
}

// multiply two matrices together
inline Matrix Matrix::operator * (const Matrix &B) const
{
  const Matrix &A = (*this);
  return Matrix(
    (A._11 * B._11) + (A._12 * B._21) + (A._13 * B._31) + (A._14 * B._41),
    (A._11 * B._12) + (A._12 * B._22) + (A._13 * B._32) + (A._14 * B._42),
    (A._11 * B._13) + (A._12 * B._23) + (A._13 * B._33) + (A._14 * B._43),
    (A._11 * B._14) + (A._12 * B._24) + (A._13 * B._34) + (A._14 * B._44),

    (A._21 * B._11) + (A._22 * B._21) + (A._23 * B._31) + (A._24 * B._41),
    (A._21 * B._12) + (A._22 * B._22) + (A._23 * B._32) + (A._24 * B._42),
    (A._21 * B._13) + (A._22 * B._23) + (A._23 * B._33) + (A._24 * B._43),
    (A._21 * B._14) + (A._22 * B._24) + (A._23 * B._34) + (A._24 * B._44),

    (A._31 * B._11) + (A._32 * B._21) + (A._33 * B._31) + (A._34 * B._41),
    (A._31 * B._12) + (A._32 * B._22) + (A._33 * B._32) + (A._34 * B._42),
    (A._31 * B._13) + (A._32 * B._23) + (A._33 * B._33) + (A._34 * B._43),
    (A._31 * B._14) + (A._32 * B._24) + (A._33 * B._34) + (A._34 * B._44),

    (A._41 * B._11) + (A._42 * B._21) + (A._43 * B._31) + (A._44 * B._41),
    (A._41 * B._12) + (A._42 * B._22) + (A._43 * B._32) + (A._44 * B._42),
    (A._41 * B._13) + (A._42 * B._23) + (A._43 * B._33) + (A._44 * B._43),
    (A._41 * B._14) + (A._42 * B._24) + (A._43 * B._34) + (A._44 * B._44));
}

// set translation elements of matrix
inline void Matrix::SetTranslation(const Vector3 &vPos)
{
  _41 = vPos.x;
  _42 = vPos.y;
  _43 = vPos.z;
}

// scale matrix by given multipliers
inline void Matrix::Scale(const Vector3 &vMultiplier)
{
  _11 *= vMultiplier.x;
  _22 *= vMultiplier.y;
  _33 *= vMultiplier.z;
}

// set rotation from given euler angles (clears _11 to _33)
inline void Matrix::SetRotation(const Vector3 &vEuler)
{
  // calculate from euler angles
  float ch = cosf(vEuler.x);
  float sh = sinf(vEuler.x);
  float cp = cosf(vEuler.y);
  float sp = sinf(vEuler.y);
  float cb = cosf(vEuler.z);
  float sb = sinf(vEuler.z);

  _11 = (cb * ch) + (sp * sh * sb);
  _12 = cp * sb;
  _13 = (-cb * sh) + (sp * ch * sb);
  _21 = (-sb * ch) + (sp * sh * cb);
  _22 = cp * cb;
  _23 = (sb * sh) + (sp * ch * cb);
  _31 = sh * cp;
  _32 = -sp;
  _33 = cp * ch;
}

// make a left-handed look-at matrix
inline Matrix MatrixLookAtLH(const Vector3 &vEye, const Vector3 &vAt, const Vector3 &vUp)
{
  Vector3 vZ = Normalize(vAt - vEye);
  Vector3 vX = Normalize(Cross(vUp, vZ));
  Vector3 vY = Cross(vZ, vX);

  return Matrix( vX.x, vY.x, vZ.x, 0,
                 vX.y, vY.y, vZ.y, 0,
                 vX.z, vY.z, vZ.z, 0,
                 -Dot(vX, vEye), -Dot(vY, vEye), -Dot(vZ, vEye),  1);
}

// make a left-handed perspective projection matrix
inline Matrix MatrixPerspectiveFovLH(float fFov, float fAspect, float fNear, float fFar)
{
  float fHeight = 1.0f / tanf(fFov * 0.5f);
  float fWidth = fHeight / fAspect;

  return Matrix(  fWidth,        0,                          0,  0,
                       0,  fHeight,                          0,  0,
                       0,        0,        fFar/(fFar - fNear),  1,
                       0,        0, -fNear*fFar/(fFar - fNear),  0   );

}

// make a left-handed orthographic projection matrix
inline Matrix MatrixOrthoLH(float fWidth, float fHeight, float fNear, float fFar)
{
  return Matrix(  2.0f/fWidth,             0,                     0,  0,
                            0,  2.0f/fHeight,                     0,  0,
                            0,             0,      1/(fFar - fNear),  0,
                            0,             0, -fNear/(fNear - fFar),  1  );

}
