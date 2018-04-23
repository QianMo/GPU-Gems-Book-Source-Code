/*
s_p_oneil@hotmail.com
Copyright (c) 2000, Sean O'Neil
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of this project nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#define interface struct

//#pragma inline_depth( 255 )
//#pragma inline_recursion( on )
//#pragma auto_inline( on )
#pragma warning(disable:4786)

// C Runtime headers
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// Windows headers
#define STRICT
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>

// C++ runtime headers
#include <string>
#include <iostream>
#include <fstream>
#include <strstream>
#include <vector>
#include <list>
#include <map>
#include <algorithm>

// OpenGL headers
#include <GL\gl.h>
#include <GL\glu.h>
#include "glprocs.h"	// GLSDK extension library


// Defines and macros
#define _MAX_NAME			31
#define GAME_SECTION		"Game Engine"
#define DISPLAY_SECTION		"Display Engine"
#define SOUND_SECTION		"Sound Engine"
#define INPUT_SECTION		"Input Engine"

// Some physics defines
#define GRAVCONST		6.67259e-20f	// Gravitational constant of the universe (km3 / kg sec^2)
#define LIGHTSPEED		299792.458f		// Speed of light (km / sec)
#define SUN_RADIUS		695000.0f		// The radius of the sun (km)
#define SUN_STRENGTH	1370.0f			// The strength of the sun at 1 AU (Watts/m^2 or Joules/second/meter^2)
#define SUN_MASS		1.989e30f		// The mass of the sun (kg)
#define EARTH_RADIUS	6378.0f			// The radius of Earth (km)
#define EARTH_MASS		5.9736e24f		// The mass of Earth (kg)
#define EARTH_ORBIT		149600000.0f	// The Earth's distance from the sun (km), also equal to 1 AU
//#define PLUTO_ORBIT	5913520000.0f	// Pluto's distance from the sun (km)
#define MOON_RADIUS		1738.0f			// The radius of Earth's moon (km)
#define MOON_MASS		7.35e22f		// The mass of Earth's moon (kg)
#define MOON_ORBIT		384400.0f		// The moon's distance from Earth (km)

//             Distance  Radius    Mass
// Planet      (000 km)   (km)     (kg)
// ---------  ---------  ------  -------
// Mercury       57,910    2439  3.30e23
// Venus        108,200    6052  4.87e24
// Earth        149,600    6378  5.98e24
// Mars         227,940    3397  6.42e23
// Jupiter      778,330   71492  1.90e27
// Saturn     1,426,940   60268  5.69e26
// Uranus     2,870,990   25559  8.69e25
// Neptune    4,497,070   24764  1.02e26
// Pluto      5,913,520    1160  1.31e22


// My includes
#include "Log.h"
#include "WndClass.h"
