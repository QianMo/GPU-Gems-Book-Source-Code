///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Matrix.h
//  Desc : Generic matrix class implementations
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

static float g_pIdentityMatrix[] =
{
  1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 1.0f
};

class CMatrix44f
{
public:

  // Matrix data (line/column)
  union
  {
    struct 
    {
      float m_f11, m_f12, m_f13, m_f14;
      float m_f21, m_f22, m_f23, m_f24;
      float m_f31, m_f32, m_f33, m_f34;
      float m_f41, m_f42, m_f43, m_f44;
    };

    float m[4][4];
  };
  
  CMatrix44f() 
  {
    // by default matrix is identity matrix
    memcpy(m, g_pIdentityMatrix, sizeof(float)<<4);
  }

  CMatrix44f(float *pMatrix)
  { 
    memcpy(m, pMatrix, sizeof(float)<<4);
  }

  CMatrix44f(float m11, float m12, float m13, float m14,
             float m21, float m22, float m23, float m24,
             float m31, float m32, float m33, float m34,
             float m41, float m42, float m43, float m44)
  {
    m_f11 = m11; m_f12 = m12; m_f13 = m13; m_f14 = m14;
    m_f21 = m21; m_f22 = m22; m_f23 = m23; m_f24 = m24;
    m_f31 = m31; m_f32 = m32; m_f33 = m33; m_f34 = m34;
    m_f41 = m41; m_f42 = m42; m_f43 = m43; m_f44 = m44; 
  }

  // Copy matrices
  CMatrix44f &operator=(const CMatrix44f &pArg)
  {
    memcpy(m, pArg.m, sizeof(float)<<4);
    return *this; 
  }

  // Add matrix
  CMatrix44f &operator+=(const CMatrix44f &pArg)
  {
    m_f11+= pArg.m_f11; m_f12+= pArg.m_f12; m_f13+= pArg.m_f13; m_f14+= pArg.m_f14;
    m_f21+= pArg.m_f21; m_f22+= pArg.m_f22; m_f23+= pArg.m_f23; m_f24+= pArg.m_f24;
    m_f31+= pArg.m_f31; m_f32+= pArg.m_f32; m_f33+= pArg.m_f33; m_f34+= pArg.m_f34;
    m_f41+= pArg.m_f41; m_f42+= pArg.m_f42; m_f43+= pArg.m_f43; m_f44+= pArg.m_f44;
    return *this;
  }

  // Subtract matrix
  CMatrix44f &operator-=(const CMatrix44f &pArg)
  {
    m_f11-= pArg.m_f11; m_f12-= pArg.m_f12; m_f13-= pArg.m_f13; m_f14-= pArg.m_f14;
    m_f21-= pArg.m_f21; m_f22-= pArg.m_f22; m_f23-= pArg.m_f23; m_f24-= pArg.m_f24;
    m_f31-= pArg.m_f31; m_f32-= pArg.m_f32; m_f33-= pArg.m_f33; m_f34-= pArg.m_f34;
    m_f41-= pArg.m_f41; m_f42-= pArg.m_f42; m_f43-= pArg.m_f43; m_f44-= pArg.m_f44;
    return *this;
  }

  // Multiply matrix
  CMatrix44f &operator *= (const CMatrix44f &pArg)
  {
    CMatrix44f pSrc = *this;
    m_f11 = pSrc.m_f11 * pArg.m_f11 + pSrc.m_f12 * pArg.m_f21 + pSrc.m_f13 * pArg.m_f31 + pSrc.m_f14 * pArg.m_f41;
    m_f12 = pSrc.m_f11 * pArg.m_f12 + pSrc.m_f12 * pArg.m_f22 + pSrc.m_f13 * pArg.m_f32 + pSrc.m_f14 * pArg.m_f42;
    m_f13 = pSrc.m_f11 * pArg.m_f13 + pSrc.m_f12 * pArg.m_f23 + pSrc.m_f13 * pArg.m_f33 + pSrc.m_f14 * pArg.m_f43;
    m_f14 = pSrc.m_f11 * pArg.m_f14 + pSrc.m_f12 * pArg.m_f24 + pSrc.m_f13 * pArg.m_f34 + pSrc.m_f14 * pArg.m_f44;

    m_f21 = pSrc.m_f21 * pArg.m_f11 + pSrc.m_f22 * pArg.m_f21 + pSrc.m_f23 * pArg.m_f31 + pSrc.m_f24 * pArg.m_f41;
    m_f22 = pSrc.m_f21 * pArg.m_f12 + pSrc.m_f22 * pArg.m_f22 + pSrc.m_f23 * pArg.m_f32 + pSrc.m_f24 * pArg.m_f42;
    m_f23 = pSrc.m_f21 * pArg.m_f13 + pSrc.m_f22 * pArg.m_f23 + pSrc.m_f23 * pArg.m_f33 + pSrc.m_f24 * pArg.m_f43;
    m_f24 = pSrc.m_f21 * pArg.m_f14 + pSrc.m_f22 * pArg.m_f24 + pSrc.m_f23 * pArg.m_f34 + pSrc.m_f24 * pArg.m_f44;

    m_f31 = pSrc.m_f31 * pArg.m_f11 + pSrc.m_f32 * pArg.m_f21 + pSrc.m_f33 * pArg.m_f31 + pSrc.m_f34 * pArg.m_f41;
    m_f32 = pSrc.m_f31 * pArg.m_f12 + pSrc.m_f32 * pArg.m_f22 + pSrc.m_f33 * pArg.m_f32 + pSrc.m_f34 * pArg.m_f42;
    m_f33 = pSrc.m_f31 * pArg.m_f13 + pSrc.m_f32 * pArg.m_f23 + pSrc.m_f33 * pArg.m_f33 + pSrc.m_f34 * pArg.m_f43;
    m_f34 = pSrc.m_f31 * pArg.m_f14 + pSrc.m_f32 * pArg.m_f24 + pSrc.m_f33 * pArg.m_f34 + pSrc.m_f34 * pArg.m_f44;

    m_f41 = pSrc.m_f41 * pArg.m_f11 + pSrc.m_f42 * pArg.m_f21 + pSrc.m_f43 * pArg.m_f31 + pSrc.m_f44 * pArg.m_f41;
    m_f42 = pSrc.m_f41 * pArg.m_f12 + pSrc.m_f42 * pArg.m_f22 + pSrc.m_f43 * pArg.m_f32 + pSrc.m_f44 * pArg.m_f42;
    m_f43 = pSrc.m_f41 * pArg.m_f13 + pSrc.m_f42 * pArg.m_f23 + pSrc.m_f43 * pArg.m_f33 + pSrc.m_f44 * pArg.m_f43;
    m_f44 = pSrc.m_f41 * pArg.m_f14 + pSrc.m_f42 * pArg.m_f24 + pSrc.m_f43 * pArg.m_f34 + pSrc.m_f44 * pArg.m_f44;
    return *this;
  }

  // Multiply by scalar
  CMatrix44f &operator *= (float fArg)
  {
    m_f11*= fArg; m_f12*= fArg; m_f13*= fArg; m_f14*= fArg;
    m_f21*= fArg; m_f22*= fArg; m_f23*= fArg; m_f24*= fArg;
    m_f31*= fArg; m_f32*= fArg; m_f33*= fArg; m_f34*= fArg;
    m_f41*= fArg; m_f42*= fArg; m_f43*= fArg; m_f44*= fArg;
    return *this;
  };

  //! Divide by scalar
  CMatrix44f &operator /= (const float fArg)
  {
    (*this)*=(1.0f/fArg);
    return *this;
  }

  //Return pointer to 1st line/column
  operator float *()
  {
    return &m_f11;
  }

  // Return pointer to 1st line/column
  operator const float *() const
  {
    return (const float *)&m_f11;
  }

  // Return matrix
  CMatrix44f operator +()const
  {
    return *this;
  }

  CMatrix44f operator -() const
  {
    return CMatrix44f(-m_f11, -m_f12, -m_f13, -m_f14,
                      -m_f21, -m_f22, -m_f23, -m_f24,
                      -m_f31, -m_f32, -m_f33, -m_f34,
                      -m_f41, -m_f42, -m_f43, -m_f44);
  }

  // Add matrixes
  CMatrix44f operator+(const CMatrix44f &pArg) const
  {
    return CMatrix44f( m_f11+pArg.m_f11, m_f12+pArg.m_f12, m_f13+pArg.m_f13, m_f14+pArg.m_f14,
                       m_f21+pArg.m_f21, m_f22+pArg.m_f22, m_f23+pArg.m_f23, m_f24+pArg.m_f24,
                       m_f31+pArg.m_f31, m_f32+pArg.m_f32, m_f33+pArg.m_f33, m_f34+pArg.m_f34,
                       m_f41+pArg.m_f41, m_f42+pArg.m_f42, m_f43+pArg.m_f43, m_f44+pArg.m_f44);
  }

  // Subtract matrixes
  CMatrix44f operator - (const CMatrix44f &pArg) const
  {
    return CMatrix44f( m_f11-pArg.m_f11, m_f12-pArg.m_f12, m_f13-pArg.m_f13, m_f14-pArg.m_f14,
                       m_f21-pArg.m_f21, m_f22-pArg.m_f22, m_f23-pArg.m_f23, m_f24-pArg.m_f24,
                       m_f31-pArg.m_f31, m_f32-pArg.m_f32, m_f33-pArg.m_f33, m_f34-pArg.m_f34,
                       m_f41-pArg.m_f41, m_f42-pArg.m_f42, m_f43-pArg.m_f43, m_f44-pArg.m_f44);
  }

  // Multiply matrixes
  CMatrix44f operator * (const CMatrix44f &pArg) const
  {
    return CMatrix44f(m_f11 * pArg.m_f11 + m_f12 * pArg.m_f21 + m_f13 * pArg.m_f31 + m_f14 * pArg.m_f41,
                      m_f11 * pArg.m_f12 + m_f12 * pArg.m_f22 + m_f13 * pArg.m_f32 + m_f14 * pArg.m_f42,
                      m_f11 * pArg.m_f13 + m_f12 * pArg.m_f23 + m_f13 * pArg.m_f33 + m_f14 * pArg.m_f43,
                      m_f11 * pArg.m_f14 + m_f12 * pArg.m_f24 + m_f13 * pArg.m_f34 + m_f14 * pArg.m_f44,

                      m_f21 * pArg.m_f11 + m_f22 * pArg.m_f21 + m_f23 * pArg.m_f31 + m_f24 * pArg.m_f41,
                      m_f21 * pArg.m_f12 + m_f22 * pArg.m_f22 + m_f23 * pArg.m_f32 + m_f24 * pArg.m_f42,
                      m_f21 * pArg.m_f13 + m_f22 * pArg.m_f23 + m_f23 * pArg.m_f33 + m_f24 * pArg.m_f43,
                      m_f21 * pArg.m_f14 + m_f22 * pArg.m_f24 + m_f23 * pArg.m_f34 + m_f24 * pArg.m_f44,

                      m_f31 * pArg.m_f11 + m_f32 * pArg.m_f21 + m_f33 * pArg.m_f31 + m_f34 * pArg.m_f41,
                      m_f31 * pArg.m_f12 + m_f32 * pArg.m_f22 + m_f33 * pArg.m_f32 + m_f34 * pArg.m_f42,
                      m_f31 * pArg.m_f13 + m_f32 * pArg.m_f23 + m_f33 * pArg.m_f33 + m_f34 * pArg.m_f43,
                      m_f31 * pArg.m_f14 + m_f32 * pArg.m_f24 + m_f33 * pArg.m_f34 + m_f34 * pArg.m_f44,

                      m_f41 * pArg.m_f11 + m_f42 * pArg.m_f21 + m_f43 * pArg.m_f31 + m_f44 * pArg.m_f41,
                      m_f41 * pArg.m_f12 + m_f42 * pArg.m_f22 + m_f43 * pArg.m_f32 + m_f44 * pArg.m_f42,
                      m_f41 * pArg.m_f13 + m_f42 * pArg.m_f23 + m_f43 * pArg.m_f33 + m_f44 * pArg.m_f43,
                      m_f41 * pArg.m_f14 + m_f42 * pArg.m_f24 + m_f43 * pArg.m_f34 + m_f44 * pArg.m_f44);
  }

  // Multiply by scalar
  CMatrix44f operator *(float fArg) const
  {
    return CMatrix44f(fArg * m_f11, fArg * m_f12, fArg * m_f13, fArg * m_f14,
                      fArg * m_f21, fArg * m_f22, fArg * m_f23, fArg * m_f24,
                      fArg * m_f31, fArg * m_f32, fArg * m_f33, fArg * m_f34, 
                      fArg * m_f41, fArg * m_f42, fArg * m_f43, fArg * m_f44);
  }

  // Multiply by scalar
  friend CMatrix44f operator * (float fArg, const CMatrix44f &pArg)
  {
    return CMatrix44f(fArg * pArg.m_f11, fArg * pArg.m_f12, fArg * pArg.m_f13, fArg * pArg.m_f14,
                      fArg * pArg.m_f21, fArg * pArg.m_f22, fArg * pArg.m_f23, fArg * pArg.m_f24,
                      fArg * pArg.m_f31, fArg * pArg.m_f32, fArg * pArg.m_f33, fArg * pArg.m_f34, 
                      fArg * pArg.m_f41, fArg * pArg.m_f42, fArg * pArg.m_f43, fArg * pArg.m_f44);
  }

  // Multiply by scalar
  friend CMatrix44f operator * (const CMatrix44f &pArg, float fArg)
  {
    return CMatrix44f(fArg * pArg.m_f11, fArg * pArg.m_f12, fArg * pArg.m_f13, fArg * pArg.m_f14,
                      fArg * pArg.m_f21, fArg * pArg.m_f22, fArg * pArg.m_f23, fArg * pArg.m_f24,
                      fArg * pArg.m_f31, fArg * pArg.m_f32, fArg * pArg.m_f33, fArg * pArg.m_f34, 
                      fArg * pArg.m_f41, fArg * pArg.m_f42, fArg * pArg.m_f43, fArg * pArg.m_f44);
  }

  // Check if matrixes are equal/notequal
  bool operator ==(const CMatrix44f &pArg) const
  {
    return ((m_f11==pArg.m_f11)&&(m_f12==pArg.m_f12)&&(m_f13==pArg.m_f13)&&(m_f14==pArg.m_f14) &&
            (m_f21==pArg.m_f21)&&(m_f22==pArg.m_f22)&&(m_f23==pArg.m_f23)&&(m_f24==pArg.m_f24) &&
            (m_f31==pArg.m_f31)&&(m_f32==pArg.m_f32)&&(m_f33==pArg.m_f33)&&(m_f34==pArg.m_f34) &&
            (m_f41==pArg.m_f41)&&(m_f42==pArg.m_f42)&&(m_f43==pArg.m_f43)&&(m_f44==pArg.m_f44));
  }

  bool operator !=(const CMatrix44f &pArg) const 
  {
    return ((m_f11!=pArg.m_f11)||(m_f12!=pArg.m_f12)||(m_f13!=pArg.m_f13)||(m_f14!=pArg.m_f14) &&
            (m_f21!=pArg.m_f21)||(m_f22!=pArg.m_f22)||(m_f23!=pArg.m_f23)||(m_f24!=pArg.m_f24) &&
            (m_f31!=pArg.m_f31)||(m_f32!=pArg.m_f32)||(m_f33!=pArg.m_f33)||(m_f34!=pArg.m_f34) &&
            (m_f41!=pArg.m_f41)||(m_f42!=pArg.m_f42)||(m_f43!=pArg.m_f43)||(m_f44!=pArg.m_f44));
  }

  // Set matrix
  CMatrix44f &Set(float m11, float m12, float m13, float m14,
                  float m21, float m22, float m23, float m24,
                  float m31, float m32, float m33, float m34,
                  float m41, float m42, float m43, float m44)
  {
    m_f11 = m11; m_f12 = m12; m_f13 = m13; m_f14 = m14;
    m_f21 = m21; m_f22 = m22; m_f23 = m23; m_f24 = m24;
    m_f31 = m31; m_f32 = m32; m_f33 = m33; m_f34 = m34;
    m_f41 = m41; m_f42 = m42; m_f43 = m43; m_f44 = m44;
    return *this;
  };

  // Reset matrix
  CMatrix44f &Reset()
  {
    memset(m, 0, sizeof(float)*16);    
    return *this;
  }

  // Set identity
  CMatrix44f &Identity()
  {
    memcpy(m, g_pIdentityMatrix, sizeof(float)<<4);
    return *this;
  }

  // Set translation
  CMatrix44f &SetTranslation(float tx, float ty, float tz)
  {
    m_f41=tx; m_f42=ty; m_f43=tz;
    return *this;
  }

  // Set rotation 
  CMatrix44f &SetRotation(float rx, float ry, float rz)
  {
    float cx = cosf(rx);
    float cy = cosf(ry);
    float cz = cosf(rz);

    float sx = sinf(rx);
    float sy = sinf(ry);
    float sz = sinf(rz);

    float sxsy = sx*sy;
    float cxsy = cx*sy;

    m_f11 = cy * cz;
    m_f12 = cy * sz;
    m_f13 = -sy;

    m_f21 = sxsy *cz - cx * sz;
    m_f22 = sxsy *sz + cx * cz;
    m_f23 = sx * cy;

    m_f31 = cxsy * cz + sx * sz;
    m_f32 = cxsy * sz - sx * cz;
    m_f33 = cx * cy;  
    return *this;
  }

  // Compute transpose matrix
  CMatrix44f &Transpose()
  {
    CMatrix44f pTemp=(*this);
    m_f12 = pTemp.m_f21; m_f21 = pTemp.m_f12;
    m_f13 = pTemp.m_f31; m_f31 = pTemp.m_f13;
    m_f14 = pTemp.m_f41; m_f41 = pTemp.m_f14;
    m_f23 = pTemp.m_f32; m_f32 = pTemp.m_f23;
    m_f24 = pTemp.m_f42; m_f42 = pTemp.m_f24;
    m_f34 = pTemp.m_f43; m_f43 = pTemp.m_f34;
    m_f11 = pTemp.m_f11; m_f22 = pTemp.m_f22;
    m_f33 = pTemp.m_f33; m_f44 = pTemp.m_f44;
    return *this;
  }

  // Compute determinant
  float Determinant() const
  {
    // compute simple 2x2 matrixes determinants
    float M3344 = m_f33*m_f44 - m_f43*m_f34;
    float M2344 = m_f23*m_f44 - m_f43*m_f24;
    float M2334 = m_f23*m_f34 - m_f33*m_f24;
    float M3244 = m_f32*m_f44 - m_f42*m_f34;
    float M2244 = m_f22*m_f44 - m_f42*m_f24;
    float M2234 = m_f22*m_f34 - m_f32*m_f24;
    float M3243 = m_f32*m_f43 - m_f42*m_f33;
    float M2243 = m_f22*m_f43 - m_f42*m_f23;
    float M2233 = m_f22*m_f33 - m_f32*m_f23;
    float M1344 = m_f13*m_f44 - m_f43*m_f14;
    float M1334 = m_f13*m_f34 - m_f33*m_f14;
    float M1244 = m_f12*m_f44 - m_f42*m_f14;
    float M1234 = m_f12*m_f34 - m_f32*m_f14;
    float M1243 = m_f12*m_f43 - m_f42*m_f13;
    float M1233 = m_f12*m_f33 - m_f32*m_f13;
    float M1324 = m_f13*m_f24 - m_f23*m_f14;
    float M1224 = m_f12*m_f24 - m_f22*m_f14;
    float M1223 = m_f12*m_f23 - m_f22*m_f13;

    // create transposed co-factor matrix
    CMatrix44f pTemp(m_f22 * M3344 - m_f32 * M2344 + m_f42 * M2334,
                      -m_f21 * M3344 + m_f31 * M2344 - m_f41 * M2334,
                      m_f21 * M3244 - m_f31 * M2244 + m_f41 * M2234,
                      -m_f21 * M3243 + m_f31 * M2243 - m_f41 * M2233,

                      -m_f12 * M3344 + m_f32 * M1344 - m_f42 * M1334,
                      m_f11 * M3344 - m_f31 * M1344 + m_f41 * M1334,
                      -m_f11 * M3244 + m_f31 * M1244 - m_f41 * M1234,
                      m_f11 * M3243 - m_f31 * M1243 + m_f41 * M1233,

                      m_f12 * M2344 - m_f22 * M1344 + m_f42 * M1324,
                      -m_f11 * M2344 + m_f21 * M1344 - m_f41 * M1324,
                      m_f11 * M2244 - m_f21 * M1244 + m_f41 * M1224,
                      -m_f11 * M2243 + m_f21 * M1243 - m_f41 * M1223,

                      -m_f12 * M2334 + m_f22 * M1334 - m_f32 * M1324,
                      m_f11 * M2334 - m_f21 * M1334 + m_f31 * M1324,
                      -m_f11 * M2234 + m_f21 * M1234 - m_f31 * M1224,
                      m_f11 * M2233 - m_f21 * M1233 + m_f31 * M1223);

    // compute determinant
    return (m_f11*pTemp.m_f11+m_f21*pTemp.m_f12+m_f31*pTemp.m_f13+m_f41*pTemp.m_f14);
  }

  // Compute inverse matrix
  CMatrix44f &Invert()
  {
    // compute simple 2x2 matrixes determinants
    float M3344 = m_f33*m_f44 - m_f43*m_f34;
    float M2344 = m_f23*m_f44 - m_f43*m_f24;
    float M2334 = m_f23*m_f34 - m_f33*m_f24;
    float M3244 = m_f32*m_f44 - m_f42*m_f34;
    float M2244 = m_f22*m_f44 - m_f42*m_f24;
    float M2234 = m_f22*m_f34 - m_f32*m_f24;
    float M3243 = m_f32*m_f43 - m_f42*m_f33;
    float M2243 = m_f22*m_f43 - m_f42*m_f23;
    float M2233 = m_f22*m_f33 - m_f32*m_f23;
    float M1344 = m_f13*m_f44 - m_f43*m_f14;
    float M1334 = m_f13*m_f34 - m_f33*m_f14;
    float M1244 = m_f12*m_f44 - m_f42*m_f14;
    float M1234 = m_f12*m_f34 - m_f32*m_f14;
    float M1243 = m_f12*m_f43 - m_f42*m_f13;
    float M1233 = m_f12*m_f33 - m_f32*m_f13;
    float M1324 = m_f13*m_f24 - m_f23*m_f14;
    float M1224 = m_f12*m_f24 - m_f22*m_f14;
    float M1223 = m_f12*m_f23 - m_f22*m_f13;

    // create transposed co-factor matrix
    CMatrix44f pTemp(m_f22 * M3344 - m_f32 * M2344 + m_f42 * M2334,
                     -m_f21 * M3344 + m_f31 * M2344 - m_f41 * M2334,
                     m_f21 * M3244 - m_f31 * M2244 + m_f41 * M2234,
                     -m_f21 * M3243 + m_f31 * M2243 - m_f41 * M2233,

                     -m_f12 * M3344 + m_f32 * M1344 - m_f42 * M1334,
                     m_f11 * M3344 - m_f31 * M1344 + m_f41 * M1334,
                     -m_f11 * M3244 + m_f31 * M1244 - m_f41 * M1234,
                     m_f11 * M3243 - m_f31 * M1243 + m_f41 * M1233,

                     m_f12 * M2344 - m_f22 * M1344 + m_f42 * M1324,
                     -m_f11 * M2344 + m_f21 * M1344 - m_f41 * M1324,
                     m_f11 * M2244 - m_f21 * M1244 + m_f41 * M1224,
                     -m_f11 * M2243 + m_f21 * M1243 - m_f41 * M1223,

                     -m_f12 * M2334 + m_f22 * M1334 - m_f32 * M1324,
                     m_f11 * M2334 - m_f21 * M1334 + m_f31 * M1324,
                     -m_f11 * M2234 + m_f21 * M1234 - m_f31 * M1224,
                     m_f11 * M2233 - m_f21 * M1233 + m_f31 * M1223);
    
    pTemp.Transpose();

    // compute determinant
    float fDetA=m_f11*pTemp.m_f11+m_f21*pTemp.m_f12+m_f31*pTemp.m_f13+m_f41*pTemp.m_f14;

    // NOTE: i am assuming that input matrix as always invertible
    fDetA=1.0f/fDetA;

    // inverse matrix is: (1/|A|)*transposed(cof(A))
    (*this)=pTemp*fDetA;

    return *this;
  }

  // Compute right handed perspective matrix
  CMatrix44f &PerspectiveFovRH(float fFov, float fAspect, float fZNear, float fZFar)
  {
    Identity();
    // compute near and far plane x and y values
    float fXmin, fXmax, fYmin, fYmax;
    fYmax = fZNear * tanf(fFov * 0.5f);
    fYmin = -fYmax;
    fXmin = fYmin * fAspect;
    fXmax = -fXmin;

    m_f11= 2.0f*fZNear/(fXmax-fXmin);
    m_f22= 2.0f*fZNear/(fYmax-fYmin);

    m_f31= (fXmax+fXmin)/(fXmax-fXmin);
    m_f32= (fYmax+fYmin)/(fYmax-fYmin);
    m_f33= (fZFar+fZNear)/(fZNear-fZFar);
    m_f34= -1.0f;

    m_f43=fZNear * fZFar/(fZNear-fZFar);
    return *this;
  }

  //! Compute a right handed look at pivot matrix
  CMatrix44f &LookAtRH(const CVector3f &pEye, const CVector3f &pAt, const CVector3f &pUp)
  {
    // compute axis
    CVector3f pZaxis  = (pEye-pAt).Normalize();
    CVector3f pXaxis  = (pUp.Cross(pZaxis)).Normalize();
    CVector3f pYaxis  = (pZaxis.Cross(pXaxis)).Normalize();
    CVector3f pDot    = CVector3f(pXaxis.Dot(pEye), pYaxis.Dot(pEye), pZaxis.Dot(pEye));

    m_f11=pXaxis.m_fX; m_f12=pYaxis.m_fX; m_f13=pZaxis.m_fX; m_f14=0.0f;
    m_f21=pXaxis.m_fY; m_f22=pYaxis.m_fY; m_f23=pZaxis.m_fY; m_f24=0.0f;
    m_f31=pXaxis.m_fZ; m_f32=pYaxis.m_fZ; m_f33=pZaxis.m_fZ; m_f34=0.0f;
    m_f41=-pDot.m_fX;  m_f42=-pDot.m_fY;  m_f43=-pDot.m_fZ;  m_f44=1.0f;
    return *this;
  }

  // Create scaling matrix
  CMatrix44f &CreateScalingMatrix(float sx, float sy, float sz)
  { 
    Identity();
    m_f11=sx; m_f22=sy; m_f33=sz;
    return *this;
  }

  // Create translation matrix
  CMatrix44f &CreateTranslationMatrix(float tx, float ty, float tz)
  {
    Identity();
    m_f41=tx; m_f42=ty; m_f43=tz;
    return *this;
  }

  // Create rotation matrix
  CMatrix44f &CreateRotationMatrix(float rx, float ry, float rz) 
  {
    Identity();
    SetRotation(rx, ry, rz);
    return *this;
  }

  // Create generic tranformation matrix
  CMatrix44f &CreateTransformationMatrix(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz)
  {
    Identity();
    SetRotation(rx, ry, rz);
    SetTranslation(tx, ty, tz);

    if(sx!=1.0f || sy!=1.0f || sz!=1.0) 
    {
      CMatrix44f pScale;
      (*this)*=(pScale.CreateScalingMatrix(sx, sy, sz));
    }

    return *this;
  }
};

// Transform vector by matrix
inline void  MatrixTransformPlane(float *pPlaneOut, const float *pPlane, const CMatrix44f &pMatrix)
 {
  pPlaneOut[0]=pPlane[0]*pMatrix.m_f11+pPlane[1]*pMatrix.m_f21+pPlane[2]*pMatrix.m_f31+pPlane[3]*pMatrix.m_f41;
  pPlaneOut[1]=pPlane[0]*pMatrix.m_f12+pPlane[1]*pMatrix.m_f22+pPlane[2]*pMatrix.m_f32+pPlane[3]*pMatrix.m_f42;
  pPlaneOut[2]=pPlane[0]*pMatrix.m_f13+pPlane[1]*pMatrix.m_f23+pPlane[2]*pMatrix.m_f33+pPlane[3]*pMatrix.m_f43;
  pPlaneOut[3]=pPlane[0]*pMatrix.m_f14+pPlane[1]*pMatrix.m_f24+pPlane[2]*pMatrix.m_f34+pPlane[3]*pMatrix.m_f44;
}

// Transform vector by matrix
inline void  MatrixTransformVec3(CVector3f &pOut, CVector3f pVector, const CMatrix44f &pMatrix)
{
  pOut.m_fX=pVector.m_fX*pMatrix.m_f11+pVector.m_fY*pMatrix.m_f21+pVector.m_fZ*pMatrix.m_f31+pMatrix.m_f41;
  pOut.m_fY=pVector.m_fX*pMatrix.m_f12+pVector.m_fY*pMatrix.m_f22+pVector.m_fZ*pMatrix.m_f32+pMatrix.m_f42;
  pOut.m_fZ=pVector.m_fX*pMatrix.m_f13+pVector.m_fY*pMatrix.m_f23+pVector.m_fZ*pMatrix.m_f33+pMatrix.m_f43;
}

// Only apply matrix rotation to vector
inline void  MatrixRotateVec3(CVector3f &pOut, CVector3f pVector, const CMatrix44f &pMatrix)
{
  pOut.m_fX=pVector.m_fX*pMatrix.m_f11+pVector.m_fY*pMatrix.m_f21+pVector.m_fZ*pMatrix.m_f31;
  pOut.m_fY=pVector.m_fX*pMatrix.m_f12+pVector.m_fY*pMatrix.m_f22+pVector.m_fZ*pMatrix.m_f32;
  pOut.m_fZ=pVector.m_fX*pMatrix.m_f13+pVector.m_fY*pMatrix.m_f23+pVector.m_fZ*pMatrix.m_f33;
}

// Only apply matrix translation to vector
inline void  MatrixTranslateVec3(CVector3f &pOut, CVector3f pVector, const CMatrix44f &pMatrix)
{
  pOut.m_fX=pVector.m_fX+pMatrix.m_f41;
  pOut.m_fY=pVector.m_fY+pMatrix.m_f42;
  pOut.m_fZ=pVector.m_fZ+pMatrix.m_f43;
}

// Inverse transformation
inline void MatrixInverseTransformVec3(CVector3f &pOutVector, CVector3f pVector, const CMatrix44f &pMatrix)
{
  pOutVector.m_fX=(pVector.m_fX-pMatrix.m_f41)*pMatrix.m_f11+(pVector.m_fY-pMatrix.m_f42)*pMatrix.m_f12+(pVector.m_fZ-pMatrix.m_f43)*pMatrix.m_f13;
  pOutVector.m_fY=(pVector.m_fX-pMatrix.m_f41)*pMatrix.m_f21+(pVector.m_fY-pMatrix.m_f42)*pMatrix.m_f22+(pVector.m_fZ-pMatrix.m_f43)*pMatrix.m_f23;
  pOutVector.m_fZ=(pVector.m_fX-pMatrix.m_f41)*pMatrix.m_f31+(pVector.m_fY-pMatrix.m_f42)*pMatrix.m_f32+(pVector.m_fZ-pMatrix.m_f43)*pMatrix.m_f33;
}

// Inverse rotation
inline void MatrixInverseRotateVec3(CVector3f &pOutVector, CVector3f pVector, const CMatrix44f &pMatrix)
{
  pOutVector.m_fX=pVector.m_fX*pMatrix.m_f11+pVector.m_fY*pMatrix.m_f12+pVector.m_fZ*pMatrix.m_f13;
  pOutVector.m_fY=pVector.m_fX*pMatrix.m_f21+pVector.m_fY*pMatrix.m_f22+pVector.m_fZ*pMatrix.m_f23;
  pOutVector.m_fZ=pVector.m_fX*pMatrix.m_f31+pVector.m_fY*pMatrix.m_f32+pVector.m_fZ*pMatrix.m_f33;
}

// Inverse translation
inline void MatrixInverseTranslateVec3(CVector3f &pOutVector, CVector3f pVector, const CMatrix44f &pMatrix)
{
  pOutVector.m_fX=pVector.m_fX-pMatrix.m_f41;
  pOutVector.m_fY=pVector.m_fY-pMatrix.m_f42;
  pOutVector.m_fZ=pVector.m_fZ-pMatrix.m_f43;
}
