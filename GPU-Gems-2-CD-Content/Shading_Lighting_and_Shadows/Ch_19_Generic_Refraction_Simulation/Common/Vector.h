///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Vector.h
//  Desc : Generic vector class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CVector3f 
{
public:

  // Coordinates
  union
  {
    struct 
    {
      float m_fX, m_fY, m_fZ;
    };

    float v[3];
  };

  CVector3f():m_fX(0), m_fY(0), m_fZ(0)
  {
  }

  CVector3f(const float fX, const float fY, const float fZ): m_fX(fX), m_fY(fY), m_fZ(fZ)
  {
  }

  // Set vector coordinates
  CVector3f &Set(float fX, float fY, float fZ)
  {
    m_fX=fX; m_fY=fY; m_fZ=fZ;
    return *this; 
  }

  // Copy vector
  CVector3f &operator =(const CVector3f &pVec)
  {
    memcpy(v, pVec.v, sizeof(float)*3);
    return *this; 
  }

  // Sum vectors
  CVector3f &operator +=(const CVector3f &pArg)
  {
    m_fX+= pArg.m_fX; m_fY+= pArg.m_fY; m_fZ+=pArg.m_fZ;
    return *this;
  }

  // Subtract vectors
  CVector3f &operator -=(const CVector3f &pArg)
  {
    m_fX-= pArg.m_fX; m_fY -= pArg.m_fY; m_fZ -= pArg.m_fZ;
    return *this;
  }

  // Multiply vectors
  CVector3f &operator *=(const CVector3f &pArg)
  {
    m_fX *= pArg.m_fX; m_fY *= pArg.m_fY; m_fZ *= pArg.m_fZ;
    return *this;
  }

  // Multiply vector by scalar
  CVector3f &operator *=(float fArg)
  {
    m_fX *= fArg; m_fY *= fArg; m_fZ *= fArg;
    return *this; 
  }

  // Divide vector by scalar
  CVector3f &operator /=(float fArg)
  {
    float fInv = 1.0f / fArg; 
    m_fX *= fInv; m_fY *= fInv; m_fZ *= fInv;
    return *this; 
  }

  // Return vector
  operator float *()
  {
    return v;
  }

  // Return vector
  operator const float *() const
  {
    return v;
  }

  // Mul by scalar
  CVector3f operator *(float fArg)
  {
    return CVector3f(m_fX * fArg, m_fY * fArg, m_fZ * fArg);
  }

  // Mul vectors
  CVector3f operator *(const CVector3f &pArg) const
  {
    return CVector3f(m_fX * pArg.m_fX, m_fY * pArg.m_fY, m_fZ * pArg.m_fZ);
  }

  // Return vector
  CVector3f operator + ()const
  {
    return *this;
  }

  // Return simetric vector
  CVector3f operator - ()const
  {
    return CVector3f(-m_fX, -m_fY, -m_fZ);
  }

  // Sum vectors
  CVector3f operator +(const CVector3f &pArg) const
  {
    return CVector3f(m_fX + pArg.m_fX, m_fY + pArg.m_fY, m_fZ + pArg.m_fZ);
  }

  // Sub vectors
  CVector3f operator -(const CVector3f &pArg) const
  {
    return CVector3f(m_fX - pArg.m_fX, m_fY - pArg.m_fY, m_fZ - pArg.m_fZ);
  }

  // Div by scalar
  CVector3f operator /(float fArg) const
  {
    float fRecip = 1.0f / fArg;
    return CVector3f(m_fX*fRecip, m_fY*fRecip, m_fZ*fRecip);
  }

  // Equal ?
  bool operator ==(const CVector3f &pArg) const
  {
    return ((m_fX == pArg.m_fX) && (m_fY == pArg.m_fY) && (m_fZ == pArg.m_fZ)); 
  }

  // Not equal ?
  bool operator !=(const CVector3f &pArg) const
  {
    return ((m_fX != pArg.m_fY) || (m_fY != pArg.m_fY) || (m_fZ != pArg.m_fZ));
  }

  // Compute vector length/size
  float Lenght() const
  {
    return sqrtf(m_fX*m_fX + m_fY*m_fY + m_fZ*m_fZ);
  }

  // Normalize vector
  CVector3f &Normalize()
  {    
    float fLen=1.0f/sqrtf(m_fX*m_fX + m_fY*m_fY + m_fZ*m_fZ);
    m_fX*=fLen; m_fY*=fLen; m_fZ*=fLen;
    return *this;
  }

  // Return dot product between two vectors
  float Dot(const CVector3f &pArg) const
  {
    return (m_fX*pArg.m_fX + m_fY*pArg.m_fY + m_fZ*pArg.m_fZ);
  }

  // Compute cross product, returns orthogonal vector
  CVector3f Cross(const CVector3f &pArg) const
  {
    return CVector3f( m_fY*pArg.m_fZ - pArg.m_fY*m_fZ,
                      m_fZ*pArg.m_fX - pArg.m_fZ*m_fX,
                      m_fX*pArg.m_fY - pArg.m_fX*m_fY);
  }
};

// Mul by scalar
inline CVector3f operator *(float fArg, const CVector3f &pArg)
{
  return CVector3f(pArg.m_fX * fArg, pArg.m_fY * fArg, pArg.m_fZ * fArg);
}

// Mul by scalar
inline CVector3f operator *(const CVector3f &pArg, float fArg)
{
  return CVector3f(pArg.m_fX * fArg, pArg.m_fY * fArg, pArg.m_fZ * fArg);
}