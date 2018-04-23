///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : MathLib.h
//  Desc : Generic math functions implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <math.h>

#ifdef PI
#undef PI
#endif

#ifdef HALF_PI
#undef HALF_PI
#endif

#define PI      (3.14159265358979323846f)
#define HALF_PI	(1.57079632679489661923f)

#define DEGTORAD(alpha) ((alpha)*(PI / 180.0f))
#define RADTODEG(alpha) ((alpha)*(180.0f / PI))
#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)
#define ABS(x) ((x >= 0) ? (x) : -(x))
#define SGN(x) ((x < 0) ? (-1) : (1))
