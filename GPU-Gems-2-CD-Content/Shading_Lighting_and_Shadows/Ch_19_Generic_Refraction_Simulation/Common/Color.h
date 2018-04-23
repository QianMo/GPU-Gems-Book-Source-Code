///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Color.h
//  Desc : Generic color class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CColor
{
public:

  // Color coordinates
  union
  {
    struct 
    {
      float m_fR, m_fG, m_fB, m_fA;
    };

    float c[4];
  };

  CColor():m_fR(1), m_fG(1), m_fB(1), m_fA(1)
  {
  }

  CColor(float fR, float fG, float fB, float fA): m_fR(fR),  m_fG(fG), m_fB(fB), m_fA(fA)
  {
  }

  // Set color
  CColor &Set(float fR, float fG, float fB, float fA)
  {
    m_fR=fR; m_fG=fG; m_fB=fB; m_fA=fA;
    return *this; 
  }

  // copy color
  CColor &operator =(const CColor &pCol)
  {
    memcpy(c, pCol.c, sizeof(float)*4);
    return *this; 
  }

  // sum colors
  CColor &operator +=(const CColor &pArg)
  {
    m_fR+= pArg.m_fR; m_fG+= pArg.m_fG; m_fB+=pArg.m_fB; m_fA+=pArg.m_fA;
    return *this;
  }

  // subtract colors
  CColor &operator -=(const CColor &pArg)
  {
    m_fR-= pArg.m_fR; m_fG -= pArg.m_fG; m_fB -= pArg.m_fB; m_fA -= pArg.m_fA;
    return *this;
  }

  // multiply colors
  CColor &operator *=(const CColor &pArg)
  {
    m_fR *= pArg.m_fR; m_fG *= pArg.m_fG; m_fB *= pArg.m_fB; m_fA *= pArg.m_fA;
    return *this;
  }

  // multiply color by scalar
  CColor &operator *=(float fArg)
  {
    m_fR *= fArg; m_fG *= fArg; m_fB *= fArg; m_fA *= fArg;
    return *this; 
  }

  // divide color by scalar
  CColor &operator /=(float fArg)
  {
    float fInv = 1.0f / fArg; 
    m_fR *= fInv; m_fG *= fInv; m_fB *= fInv; m_fA *= fInv;
    return *this; 
  }

  // return color
  operator float *()
  {
    return c;
  }

  // return color
  operator const float *() const
  {
    return c;
  }

  // mul by scalar
  CColor operator *(float fArg)
  {
    return CColor(m_fR * fArg, m_fG * fArg, m_fB * fArg, m_fA * fArg);
  }

  // mul colors
  CColor operator *(const CColor &pArg) const
  {
    return CColor(m_fR * pArg.m_fR, m_fG * pArg.m_fG, m_fB * pArg.m_fB, m_fA * pArg.m_fB);
  }

  // return color
  CColor operator + ()const
  {
    return *this;
  }

  // return simetric color
  CColor operator - ()const
  {
    return CColor(-m_fR, -m_fG, -m_fB, -m_fA);
  }

  // sum colors
  CColor operator +(const CColor &pArg) const
  {
    return CColor(m_fR + pArg.m_fR, m_fG + pArg.m_fG, m_fB + pArg.m_fB, m_fA + pArg.m_fA);
  }

  // sub colors
  CColor operator -(const CColor &pArg) const
  {
    return CColor(m_fR - pArg.m_fR, m_fG - pArg.m_fG, m_fB - pArg.m_fB, m_fA - pArg.m_fA);
  }

  // div by scalar
  CColor operator /(float fArg) const
  {
    float fRecip = 1.0f / fArg;
    return CColor(m_fR*fRecip, m_fG*fRecip, m_fB*fRecip, m_fA*fRecip);
  }

  // is equal ?
  bool operator ==(const CColor &pArg) const
  {
    return ((m_fR == pArg.m_fR) && (m_fG == pArg.m_fG) && (m_fB == pArg.m_fB) && (m_fA == pArg.m_fA)); 
  }

  // not equal ?
  bool operator !=(const CColor &pArg) const
  {
    return ((m_fR != pArg.m_fG) || (m_fG != pArg.m_fG) || (m_fB != pArg.m_fB) || (m_fA != pArg.m_fA));
  }

  // clamp color to min/max values
  CColor &Clamp(float fMin=0.0f, float fMax=1.0f)
  {
    if(m_fR>fMax) m_fR=fMax;
    else
    if(m_fR<fMin) m_fR=fMin;

    if(m_fG>fMax) m_fG=fMax;
    else
    if(m_fG<fMin) m_fG=fMin;

    if(m_fB>fMax) m_fB=fMax;
    else
    if(m_fB<fMin) m_fB=fMin;

    if(m_fA>fMax) m_fA=fMax;
    else
    if(m_fA<fMin) m_fA=fMin;          
    return *this;
  }
};

// Mul by scalar
inline const CColor operator *(float fArg, const CColor &pArg)
{
  return CColor(pArg.m_fR * fArg, pArg.m_fG * fArg, pArg.m_fB * fArg, pArg.m_fA * fArg);
}

// Mul by scalar
inline const CColor operator *(const CColor &pArg, float fArg)
{
  return CColor(pArg.m_fR * fArg, pArg.m_fG * fArg, pArg.m_fB * fArg, pArg.m_fA * fArg);
}