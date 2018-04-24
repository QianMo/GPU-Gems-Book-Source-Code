/*! \file floatmxn.h
 *  \author Jared Hoberock
 *  \brief Defines some float Matrix types.
 */

#ifndef FLOATMXN_H
#define FLOATMXN_H

#include "Matrix.h"
#include "Vector.h"

//typedef Matrix<float,2,2> float2x2;
typedef Matrix<float,2,3> float2x3;
typedef Matrix<float,3,2> float3x2;
typedef Matrix<float,3,3> float3x3;
typedef Matrix<float,3,4> float3x4;
typedef Matrix<float,4,4> float4x4;
typedef Matrix<float,5,2> float5x2;

/*! XXX remove after Siggraph */
class float2x2 : public Matrix<float,2,2>
{
  public:
  inline float2x2(void){;}
  inline float2x2(const Matrix<float,2,2> &m)
  {
    float2x2 &a = *this;
    a(0,0) = m(0,0); a(0,1) = m(0,1);
    a(1,0) = m(1,0); a(1,1) = m(1,1);
  } // end float2x2()
  
  inline float determinant(void) const
  {
    const float2x2 &m = *this;
    return m(0,0)*m(1,1) - m(0,1)*m(1,0);
  } // end determinant()
}; // end class float2x2

#endif // FLOATMXN_H

