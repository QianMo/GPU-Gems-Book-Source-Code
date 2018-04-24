#pragma once

#include "Point.hpp"

//--------------------------------------------------------------------------------------
// Impelements percentage closer filtering (using a box region)
// Same as Point, just with a different shader
class PCF : public Point
{
public:
  PCF(ID3D10Device* d3dDevice,
      ID3D10Effect* Effect,
      int Width, int Height);

  virtual ~PCF();

protected:
  ID3D10EffectScalarVariable*         m_EffectConeBias;
};
