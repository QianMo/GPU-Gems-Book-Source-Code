/*
    Copyright (C) 1998,1999,2000 by Jorrit Tyberghein
    Largely rewritten by Ivan Avramovic <ivan@avramovic.com>
  
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.
  
    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.
  
    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include <math.h>
#include <float.h>
#include "cs_compat.h"
#include "qint.h"
#include "csgeom/matrix3.h"


//---------------------------------------------------------------------------

csMatrix3& csMatrix3::operator+= (const csMatrix3& m)
{
  m11 += m.m11; m12 += m.m12; m13 += m.m13;
  m21 += m.m21; m22 += m.m22; m23 += m.m23;
  m31 += m.m31; m32 += m.m32; m33 += m.m33;
  return *this;
}

csMatrix3& csMatrix3::operator-= (const csMatrix3& m)
{
  m11 -= m.m11; m12 -= m.m12; m13 -= m.m13;
  m21 -= m.m21; m22 -= m.m22; m23 -= m.m23;
  m31 -= m.m31; m32 -= m.m32; m33 -= m.m33;
  return *this;
}

csMatrix3& csMatrix3::operator*= (const csMatrix3& m)
{
  float old_m11 = m11;
  m11 = m11*m.m11 + m12*m.m21 + m13*m.m31;
  float old_m12 = m12;
  m12 = old_m11*m.m12 + m12*m.m22 + m13*m.m32;
  m13 = old_m11*m.m13 + old_m12*m.m23 + m13*m.m33;
  float old_m21 = m21;
  m21 = m21*m.m11 + m22*m.m21 + m23*m.m31;
  float old_m22 = m22;
  m22 = old_m21*m.m12 + m22*m.m22 + m23*m.m32;
  m23 = old_m21*m.m13 + old_m22*m.m23 + m23*m.m33;
  float old_m31 = m31;
  m31 = m31*m.m11 + m32*m.m21 + m33*m.m31;
  float old_m32 = m32;
  m32 = old_m31*m.m12 + m32*m.m22 + m33*m.m32;
  m33 = old_m31*m.m13 + old_m32*m.m23 + m33*m.m33;
  return *this;
}

csMatrix3& csMatrix3::operator*= (float s)
{
  m11 *= s; m12 *= s; m13 *= s;
  m21 *= s; m22 *= s; m23 *= s;
  m31 *= s; m32 *= s; m33 *= s;
  return *this;
}

void csMatrix3::Identity ()
{
  m11 = m22 = m33 = 1.0;
  m12 = m13 = m21 = m23 = m31 = m32 = 0.0;
}

bool csMatrix3::IsIdentity () const
{
  return (m11 == 1.0) && (m22 == 1.0) && (m33 == 1.0)
      && (m12 == 0.0) && (m13 == 0.0) && (m21 == 0.0)
      && (m23 == 0.0) && (m31 == 0.0) && (m32 == 0.0);
}

void csMatrix3::Transpose ()
{
  float swap;
  swap = m12; m12 = m21; m21 = swap;
  swap = m13; m13 = m31; m31 = swap;
  swap = m23; m23 = m32; m32 = swap;
}

csMatrix3 csMatrix3::GetTranspose () const
{
  csMatrix3 t;
  t.m12 = m21; t.m21 = m12;
  t.m13 = m31; t.m31 = m13;
  t.m23 = m32; t.m32 = m23;
  t.m11 = m11; t.m22 = m22; t.m33 = m33;
  return t;
}

float csMatrix3::Determinant () const
{
  return
    m11 * (m22*m33 - m23*m32)
   -m12 * (m21*m33 - m23*m31)
   +m13 * (m21*m32 - m22*m31);
}


csMatrix3 operator+ (const csMatrix3& m1, const csMatrix3& m2) 
{
  return csMatrix3 (m1.m11+m2.m11, m1.m12+m2.m12, m1.m13+m2.m13,
                    m1.m21+m2.m21, m1.m22+m2.m22, m1.m23+m2.m23,
                    m1.m31+m2.m31, m1.m32+m2.m32, m1.m33+m2.m33);
}
                  
csMatrix3 operator- (const csMatrix3& m1, const csMatrix3& m2)
{
  return csMatrix3 (m1.m11-m2.m11, m1.m12-m2.m12, m1.m13-m2.m13,
                    m1.m21-m2.m21, m1.m22-m2.m22, m1.m23-m2.m23,
                    m1.m31-m2.m31, m1.m32-m2.m32, m1.m33-m2.m33);
}
csMatrix3 operator* (const csMatrix3& m1, const csMatrix3& m2)
{
  return csMatrix3 (
   m1.m11*m2.m11 + m1.m12*m2.m21 + m1.m13*m2.m31,
   m1.m11*m2.m12 + m1.m12*m2.m22 + m1.m13*m2.m32,
   m1.m11*m2.m13 + m1.m12*m2.m23 + m1.m13*m2.m33,
   m1.m21*m2.m11 + m1.m22*m2.m21 + m1.m23*m2.m31,
   m1.m21*m2.m12 + m1.m22*m2.m22 + m1.m23*m2.m32,
   m1.m21*m2.m13 + m1.m22*m2.m23 + m1.m23*m2.m33,
   m1.m31*m2.m11 + m1.m32*m2.m21 + m1.m33*m2.m31,
   m1.m31*m2.m12 + m1.m32*m2.m22 + m1.m33*m2.m32,
   m1.m31*m2.m13 + m1.m32*m2.m23 + m1.m33*m2.m33 );
}

csMatrix3 operator* (const csMatrix3& m, float f)
{
  return csMatrix3 (m.m11*f, m.m12*f, m.m13*f,
                    m.m21*f, m.m22*f, m.m23*f,
                    m.m31*f, m.m32*f, m.m33*f);
}

csMatrix3 operator* (float f, const csMatrix3& m)
{
  return csMatrix3 (m.m11*f, m.m12*f, m.m13*f,
                    m.m21*f, m.m22*f, m.m23*f,
                    m.m31*f, m.m32*f, m.m33*f);
}

csMatrix3 operator/ (const csMatrix3& m, float f)
{
  float inv_f = 1 / f;
  return csMatrix3 (m.m11*inv_f, m.m12*inv_f, m.m13*inv_f,
                    m.m21*inv_f, m.m22*inv_f, m.m23*inv_f,
                    m.m31*inv_f, m.m32*inv_f, m.m33*inv_f);
}

bool operator== (const csMatrix3& m1, const csMatrix3& m2)
{ 
  if (m1.m11 != m2.m11 || m1.m12 != m2.m12 || m1.m13 != m2.m13) return false;
  if (m1.m21 != m2.m21 || m1.m22 != m2.m22 || m1.m23 != m2.m23) return false;
  if (m1.m31 != m2.m31 || m1.m32 != m2.m32 || m1.m33 != m2.m33) return false;
  return true;
}

bool operator!= (const csMatrix3& m1, const csMatrix3& m2)
{
  if (m1.m11 != m2.m11 || m1.m12 != m2.m12 || m1.m13 != m2.m13) return true;
  if (m1.m21 != m2.m21 || m1.m22 != m2.m22 || m1.m23 != m2.m23) return true;
  if (m1.m31 != m2.m31 || m1.m32 != m2.m32 || m1.m33 != m2.m33) return true;
  return false;
}

bool operator< (const csMatrix3& m, float f)
{
  return ABS(m.m11)<f && ABS(m.m12)<f && ABS(m.m13)<f &&
         ABS(m.m21)<f && ABS(m.m22)<f && ABS(m.m23)<f &&
         ABS(m.m31)<f && ABS(m.m32)<f && ABS(m.m33)<f;
}

bool operator> (float f, const csMatrix3& m)
{
  return ABS(m.m11)<f && ABS(m.m12)<f && ABS(m.m13)<f &&
         ABS(m.m21)<f && ABS(m.m22)<f && ABS(m.m23)<f &&
         ABS(m.m31)<f && ABS(m.m32)<f && ABS(m.m33)<f;
}

//---------------------------------------------------------------------------

csXRotMatrix3::csXRotMatrix3 (float angle)
{
  m11 = 1; m12 = 0;           m13 = 0;
  m21 = 0; m22 = cos (angle); m23 = -sin(angle);
  m31 = 0; m32 = sin (angle); m33 =  cos(angle);
}

csYRotMatrix3::csYRotMatrix3 (float angle)
{
  m11 = cos (angle); m12 = 0; m13 = -sin(angle);
  m21 = 0;           m22 = 1; m23 = 0;
  m31 = sin (angle); m32 = 0; m33 =  cos(angle);
}

csZRotMatrix3::csZRotMatrix3 (float angle)
{
  m11 = cos (angle); m12 = -sin(angle); m13 = 0;
  m21 = sin (angle); m22 =  cos(angle); m23 = 0;
  m31 = 0;           m32 = 0;           m33 = 1;
}

//---------------------------------------------------------------------------
