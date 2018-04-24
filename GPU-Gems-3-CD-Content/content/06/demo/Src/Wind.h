#ifndef __WIND_H
#define __WIND_H

#include "Cfg.h"

///////////////////////////////////////////////////////////////////////////////
// Wind simulation functions
//

// 		y = cos(PI*x) * cos(PI*3*x) * cos(PI*5*x) * cos(PI*7*x) - 0.1*sin(PI*25*x)
float windWithPseudoTurbulence(float t);
// 		y = (sin(PI*x) + sin(PI*3*x) + sin(PI*5*x) + sin(PI*7*x)) / 4
float windWithPseudoTurbulence2(float t);
// 		y = (cos(PI*x)^2 * cos(PI*3*x) * cos(PI*5*x) - 0.02*sin(PI*25*x)
float windSmoothWithSlightNoise(float t);
// 		y = (cos(PI*x)^2 * cos(PI*3*x) * cos(PI*5*x) - 0.1*sin(PI*25*x)
float windPeriodicWithNoise(float t);

// sample rotation of the tree while affected by wind
D3DXQUATERNION calcWindRotation(D3DXVECTOR2 const& windDirection, D3DXVECTOR2 const& windPower);

#endif