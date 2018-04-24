#ifndef __UTILS_H
#define __UTILS_H

#include "Cfg.h"

inline float deg2rad(float deg)
{
	return deg * D3DX_PI / 180.0f;
}

template <typename T>
inline float lerp(T a, T b, float t)
{
	return a*(1.0f - t) + b*t;
}

#endif
