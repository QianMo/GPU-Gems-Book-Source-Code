//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef Matrix4XformH
#define Matrix4XformH
#include "Matrix4.h"
//---------------------------------------------------------------------------
namespace Math {

//[x,0,0,0]
//[0,y,0,0]
//[0,0,z,0]
//[0,0,0,1]
template<class REAL>
void makeScaleMtx(Matrix4<REAL>& output, const REAL& x, const REAL& y, const REAL& z) {
	output[ 0] = x;
	output[ 1] = 0.0;
	output[ 2] = 0.0;
	output[ 3] = 0.0;

	output[ 4] = 0.0;
	output[ 5] = y;
	output[ 6] = 0.0;
	output[ 7] = 0.0;

	output[ 8] = 0.0;
	output[ 9] = 0.0;
	output[10] = z;
	output[11] = 0.0;

	output[12] = 0.0;
	output[13] = 0.0;
	output[14] = 0.0;
	output[15] = 1.0;
}

//make a scaleTranslate matrix that includes the two values vMin and vMax
template<class REAL>
void scaleTranslateToFit(Matrix4<REAL>& output, const Vector3<REAL>& vMin, const Vector3<REAL>& vMax) {
	output[ 0] = 2/(vMax[0]-vMin[0]);
	output[ 4] = 0;
	output[ 8] = 0;
	output[12] = -(vMax[0]+vMin[0])/(vMax[0]-vMin[0]);

	output[ 1] = 0;
	output[ 5] = 2/(vMax[1]-vMin[1]);
	output[ 9] = 0;
	output[13] = -(vMax[1]+vMin[1])/(vMax[1]-vMin[1]);

	output[ 2] = 0;
	output[ 6] = 0;
	output[10] = 2/(vMax[2]-vMin[2]);
	output[14] = -(vMax[2]+vMin[2])/(vMax[2]-vMin[2]);

	output[ 3] = 0;
	output[ 7] = 0;
	output[11] = 0;
	output[15] = 1;
}

//output = look from position:pos into direction:dir with up-vector:up
template<class REAL>
void look(Matrix4<REAL>& output, const Vector3<REAL>& pos, const Vector3<REAL>& dir, const Vector3<REAL>& up) {
	Vector3<REAL> dirN;
	Vector3<REAL> upN;
	Vector3<REAL> lftN;

	lftN.unitCross(dir,up);
	upN.unitCross(lftN,dir);
	dirN = dir;
	dirN.normalize();

	output[ 0] = lftN[0];
	output[ 1] = upN[0];
	output[ 2] = -dirN[0];
	output[ 3] = 0.0;

	output[ 4] = lftN[1];
	output[ 5] = upN[1];
	output[ 6] = -dirN[1];
	output[ 7] = 0.0;

	output[ 8] = lftN[2];
	output[ 9] = upN[2];
	output[10] = -dirN[2];
	output[11] = 0.0;

	output[12] = -lftN.dot(pos);
	output[13] = -upN.dot(pos);
	output[14] = dirN.dot(pos);
	output[15] = 1.0;
}

//output is initialized with the same result as glPerspective vFovy in rad
template<class REAL>
void perspectiveRad(Matrix4<REAL>& output, const REAL& vFovy, const REAL& vAspect,
					const REAL& vNearDis, const REAL& vFarDis) {
	const REAL f = Math::coTan(vFovy/2.0);
	const REAL dif = 1.0/(vNearDis-vFarDis);

	output[ 0] = f/vAspect;
	output[ 4] = 0;
	output[ 8] = 0;
	output[12] = 0;

	output[ 1] = 0;
	output[ 5] = f;
	output[ 9] = 0;
	output[13] = 0;

	output[ 2] = 0;
	output[ 6] = 0;
	output[10] = (vFarDis+vNearDis)*dif;
	output[14] = 2*vFarDis*vNearDis*dif;

	output[ 3] = 0;
	output[ 7] = 0;
	output[11] = -1;
	output[15] = 0;
}

//output is initialized with the same result as glPerspective vFovy in degrees
template<class REAL>
void perspectiveDeg(Matrix4<REAL>& output, const REAL& vFovy, const REAL& vAspect,
						  const REAL& vNearDis, const REAL& vFarDis) {
	perspectiveRad(output,vFovy*Math::Const<REAL>::pi_180(),vAspect,vNearDis,vFarDis);
}


//namespace
}
#endif
