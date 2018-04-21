/**
  @file Viewport.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#include <G3DAll.h>
#include "Viewport.h"

Viewport::Viewport(
        double                              screenWidth,
        double                              screenHeight, 
        double                              fieldOfView,
        double                              nearPlane)
{

    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;
    m_fieldOfView = fieldOfView;
    m_nearPlane = nearPlane;
}


void Viewport::getNearXYZ(
        double&                             x,
        double&                             y,
        double&                             z) const
{
    y = m_nearPlane * tan(m_fieldOfView / 2);
    x = y * m_screenWidth / m_screenHeight;
    z = m_nearPlane;
}

void Viewport::getInfiniteFrustumMatrix(
        double*                             mat) const
{
        double x, y, z;

    getNearXYZ(x, y, z);

    double left = -x;
    double right = x;
    double top = y;
    double bottom = -y;
    double nearval = z;

    double a, b, c;

    x = (2.0*nearval) / (right-left);
    y = (2.0*nearval) / (top-bottom);
    a = (right+left) / (right-left);
    b = (top+bottom) / (top-bottom);
    c = -2.0*nearval;

#define M(row,col)  mat[row + col*4]
    M(0,0) = x;    M(0,1) = 0.0;  M(0,2) = a;      M(0,3) = 0.0;
    M(1,0) = 0.0;  M(1,1) = y;    M(1,2) = b;      M(1,3) = 0.0;
    M(2,0) = 0.0;  M(2,1) = 0.0;  M(2,2) = -1.0;   M(2,3) = c;
    M(3,0) = 0.0;  M(3,1) = 0.0;  M(3,2) = -1.0;   M(3,3) = 0.0;
#undef M

}


void Viewport::setInfiniteFrustum()
{
    double m[16];
    getInfiniteFrustumMatrix(m);
    glLoadMatrixd(m);
}

