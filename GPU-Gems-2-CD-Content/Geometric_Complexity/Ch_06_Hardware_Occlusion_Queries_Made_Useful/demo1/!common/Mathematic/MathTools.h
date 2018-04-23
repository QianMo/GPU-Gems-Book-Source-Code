//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef MathToolsH
#define MathToolsH
//---------------------------------------------------------------------------
#include "Vector3.h"
#include "Matrix4.h"
#include <Types/StaticArray.h>
#include <Types/DynamicArray.h>
//---------------------------------------------------------------------------
namespace Math {

// transform each vector in the array by the matrix and homogenize
template<class REAL>
inline void transformHomogenVecsByFrame(Array<Vector3<REAL> >& v, const Matrix4<REAL>& frame) {
	for(unsigned i = 0; i < v.size(); i++) {
		v[i] = frame.mulHomogenPoint(v[i]);
	}
}

//calculates the world coordinates of the 8 view frustum corner points
//input matrix is the (eyeProj*eyeView)^(-1) matrix
template<class REAL>
inline void calcViewFrustumWorldCoord(StaticArray<Vector3<REAL>,8>& points, const Matrix4<REAL>& invEyeProjView) {
	const AABox box(Vector3<REAL>(-1, -1, -1), Vector3<REAL>(1, 1, 1));
	box.computeVerticesLeftHanded(points); //calc unit cube corner points
	transformHomogenVecsByFrame(points,invEyeProjView); //camera to world frame
	//v := viewFrustumWorldCoord ->
	//viewFrustumWorldCoord[0] == near-bottom-left
	//viewFrustumWorldCoord[1] == near-bottom-right
	//viewFrustumWorldCoord[2] == near-top-right
	//viewFrustumWorldCoord[3] == near-top-left
	//viewFrustumWorldCoord[4] == far-bottom-left
	//viewFrustumWorldCoord[5] == far-bottom-right
	//viewFrustumWorldCoord[6] == far-top-right
	//viewFrustumWorldCoord[7] == far-top-left
}

// Return value is 'true' if line segment intersects the current test
// plane.  Otherwise 'false' is returned in which case the line segment
// is entirely clipped.
template<class REAL>
bool clipTest(const REAL& p, const REAL& q, REAL& u1, REAL& u2) {
	if(p < 0.0) {
		const REAL r = q/p;
		if(r > u2) {
			return false;
		}
		else {
			if(r > u1) {
				u1 = r;
			}
			return true;
		}
	}
	else {
		if(p > 0.0)	{
			const REAL r = q/p;
			if(r < u1) {
				return false;
			}
			else {
				if(r < u2) {
					u2 = r;
				}
				return true;
			}
		}
		else {
			return q >= 0.0;
		}
	}
}

void createSingularRandomNumbers(DynamicArray<unsigned>& output, const unsigned count, const unsigned max);

template<class T>
void randomizeArray(Array<T>& io) {
	const unsigned lastID = io.size()-1;
	const double fact = (1.0/RAND_MAX)*lastID;
	for(unsigned i = 0; i < io.size(); i++) {
		const unsigned a = Math::clamp<unsigned>(rand()*fact,0,lastID);
		const unsigned b = Math::clamp<unsigned>(rand()*fact,0,lastID);
		std::swap(io[a],io[b]);
	}
}

//namespace
}
#endif
