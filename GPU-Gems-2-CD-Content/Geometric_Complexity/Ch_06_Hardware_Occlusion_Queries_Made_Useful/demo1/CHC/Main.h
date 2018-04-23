//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef MainH
#define MainH

#include "MathStuff.h"
#include "SceneData.h"
#include <GL/glHeader.h>

extern double nearDist; // eye near plane distance (often 1)
extern V3 eyePos;  // eye position 
extern V3 viewDir;  // eye view dir 
extern V3 lightDir;  // light dir 

extern Math::Matrix4d eyeView; // eye view matrix
extern Math::Matrix4d eyeProjection; // eye projection matrix
extern Math::Matrix4d eyeProjView; //= eyeProjection*eyeView
extern Math::Matrix4d invEyeProjView; //= eyeProjView^(-1)

extern Math::Matrix4d lightView; // light view matrix
extern Math::Matrix4d lightProjection; // light projection matrix

extern void updateLightMtx(const AABox& sceneAABox); // updates lightView & lightProjection

#endif
