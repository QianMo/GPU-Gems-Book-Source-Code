//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include "Main.h"
#include <Mathematic/Mathematic.h>
#include <Mathematic/Matrix4Xform.h>

V3 calcNewDir(const VecPoint& B) {
	V3 dir(V3::ZERO);
	for(unsigned i = 0; i < B.size(); i++) {
		dir += B[i]-eyePos;
	}
	dir.normalize();
	return dir;
}

//calculates the up vector for the light coordinate frame
V3 calcUpVec(const V3& viewDir, const V3& lightDir) {
	//we do what gluLookAt does...
	//left is the normalized vector perpendicular to lightDir and viewDir
	//this means left is the normalvector of the yz-plane from the paper
	V3 left;
	left.cross(lightDir,viewDir);
	//we now can calculate the rotated(in the yz-plane) viewDir vector
	//and use it as up vector in further transformations
	V3 up;
	up.unitCross(left,lightDir);
	return up;
}

//this is the algorithm discussed in the LispSM paper
void calcLispSMMtx(VecPoint& B) {
	VecPoint Bcopy(B);
	V3 up;

	//CHANGED
	V3 newDir = calcNewDir(B);
	up = calcUpVec(newDir,lightDir);

	//temporal light View
	//look from position(eyePos)
	//into direction(lightDir) 
	//with up vector(up)
	Math::look(lightView,eyePos,lightDir,up);

	//transform the light volume points from world into light space
	transformVecPoint(B,lightView);

	V3 min, max;
	//calculate the cubic hull (an AABB) 
	//of the light space extents of the intersection body B
	//and save the two extreme points min and max
	calcCubicHull(min,max,B);

	const double dotProd = viewDir.dot(lightDir);
	const double sinGamma = sqrt(1.0-dotProd*dotProd);

	//use the formulas of the paper to get n (and f)
	const double factor = 1.0/sinGamma;
	const double z_n = factor;//*nearDist; //often 1  //todo
	const double d = Math::abs(max[1]-min[1]); //perspective transform depth //light space y extents
	const double z_f = z_n + d*sinGamma;
	const double n = (z_n+sqrt(z_f*z_n))/sinGamma;
	const double f = n+d;

	//new observer point n-1 behind eye position
	V3 pos = Math::Vector<double,3>(eyePos-(n-nearDist)*up);

	Math::look(lightView,pos,lightDir,up);

	//one possibility for a simple perspective transformation matrix
	//with the two parameters n(near) and f(far) in y direction
	Math::Matrix4d lispMtx(Math::Matrix4d::IDENTITY);// a = (f+n)/(f-n); b = -2*f*n/(f-n);
	lispMtx(2,2) = (f+n)/(f-n);		 // [ 1 0 0 0] 
	lispMtx(2,4) = -2*f*n/(f-n);	 // [ 0 a 0 b]
	lispMtx(4,2) = 1;				 // [ 0 0 1 0]
	lispMtx(4,4) = 0;				 // [ 0 1 0 0]

	//temporal arrangement for the transformation of the points to post-perspective space
	lightProjection = lispMtx*lightView;
	
	//transform the light volume points from world into the distorted light space
	transformVecPoint(Bcopy,lightProjection);

	//calculate the cubic hull (an AABB) 
	//of the light space extents of the intersection body B
	//and save the two extreme points min and max
	calcCubicHull(min,max,Bcopy);

	//refit to unit cube
	//this operation calculates a scale translate matrix that
	//maps the two extreme points min and max into (-1,-1,-1) and (1,1,1)
	Math::scaleTranslateToFit(lightProjection,min,max);

	//together
	lightProjection *= lispMtx; // ligthProjection = scaleTranslate*lispMtx
}

void updateLightMtx(const AABox& sceneAABox) {
	//the intersection Body B
	VecPoint B;

	//calculates the ViewFrustum Object; clippes this Object By the sceneAABox and
	//extrudes the object into -lightDir and clippes by the sceneAABox
	//the defining points are returned
	calcFocusedLightVolumePoints(B,invEyeProjView,lightDir,sceneAABox);

	//do Light Space Perspective shadow mapping
	calcLispSMMtx(B);

	//transform from right handed into left handed coordinate system
	{
		Math::Matrix4d rh2lf;
		Math::makeScaleMtx(rh2lf,1.0,1.0,-1.0);
		lightProjection = rh2lf*lightProjection;
	}
}
