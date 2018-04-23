//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef FrustumH
#define FrustumH
//---------------------------------------------------------------------------
#include <vector>
#include <bitset>
#include "../Vector3.h"
#include "../Matrix4.h"
#include "Plane.h"
#include "AABox.h"
//---------------------------------------------------------------------------
namespace Math {
namespace Geometry {

template<class REAL = float>
class Frustum {
protected:
	typedef Plane<REAL> Plane;
	typedef AABox<REAL> AABox;
	typedef Vector3<REAL> V3;
public:
	typedef std::vector<Plane> VecPlane;
	typedef std::bitset<32> ActivePlanes;
	typedef Matrix4<REAL> Matrix4;

protected:
	VecPlane vecPlane;
	ActivePlanes activePlanes;

public:
	Frustum() { activePlanes.reset(); }

	Frustum(const Matrix4& projectionCamera) {
		const Matrix4& m = projectionCamera;
		activePlanes.reset();
		// Extract the numbers for the LEFT plane
		// the n vectors are pointing inside
		addClipPlane(Plane(m.a41()+m.a11(), m.a42()+m.a12(), m.a43()+m.a13(), -m.a14()-m.a44()));

		// Extract the numbers for the RIGHT plane
		addClipPlane(Plane(m.a41()-m.a11(), m.a42()-m.a12(), m.a43()-m.a13(), m.a14()-m.a44()));

		// Extract the BOTTOM plane
		addClipPlane(Plane(m.a41()+m.a21(), m.a42()+m.a22(), m.a43()+m.a23(), -m.a24()-m.a44()));

		// Extract the TOP plane
		addClipPlane(Plane(m.a41()-m.a21(), m.a42()-m.a22(), m.a43()-m.a23(), m.a24()-m.a44()));

		// Extract the NEAR plane 
		addClipPlane(Plane(m.a41()+m.a31(), m.a42()+m.a32(), m.a43()+m.a33(), -m.a34()-m.a44()));

		// Extract the FAR plane
		addClipPlane(Plane(m.a41()-m.a31(), m.a42()-m.a32(), m.a43()-m.a33(), m.a34()-m.a44()));
	}

	void addClipPlane(const Plane& p){
		if(vecPlane.size() < activePlanes.size()) {
			vecPlane.push_back(p);
			activePlanes[vecPlane.size()-1] = true;
		}
		else {
			throw GeometryException("Frustum: to many ClipPlanes");
		}
	}

	const Plane& getPlaneUnChecked(cuint id) const { return vecPlane[id]; }

	const Plane& getPlane(cuint id) const {
		if(id < vecPlane.size()) {
			return getPlaneUnChecked(id);
		}
		else {
			throw GeometryException("Frustum::getPlane: invalid plane");
		}
	}

	const VecPlane& getVecPlanes() const { return vecPlane; }

	const ActivePlanes& getActivePlanes() const { return activePlanes; }
	ActivePlanes& getActivePlanes() { return activePlanes; }

	bool contains(const V3& p) const{
		for(uint i = 0; i < vecPlane.size(); i++) {
			if(activePlanes[i] && vecPlane[i].inFront(p)) {
				return false;
			}
		}
		return true;
	}

	bool contains(const V3& p, const ActivePlanes& ap) const {
		unsigned size;
		Math::minimum(size,ap.size(),vecPlane.size());
		for(unsigned i = 0; i < size; i++) {
			if(vecPlane[i].inFront(p)) {
				return false;
			}
		}
		return true;
	}

	bool inside(const AABox& box, const ActivePlanes& ap) const {
		Math::Vector3x8 p;
		box.computeVerticesRightHanded(p);
		for(unsigned i = 0; i < 8; i++) {
			if(!contains(p[i],ap)) {
				return false;
			}
		}
		return true;
	}

	//bool visible(const OBox&) const;
	bool visible(const AABox& box) const {
		if(activePlanes.none()) {
			return true;
		}
		ActivePlanes help = activePlanes;
		for(unsigned i = 0; help.any(); i++, help >>= 1) { 
			// i < vecPlane.size test is superflue if the active planes are valid(only contains 1s where planes are existent)
			if(help[0]) {
				if(BEHIND == box.getPlaneSide(vecPlane[i])) {
					return false;
				}
			}
		}
		return true;
	}

	//returns true if the box can be culled
	//updates activePlanes if the AABox lies in front of a plane and therefor 
	//child boxes don't need to check this plane
	bool cull(const AABox& box) {
		if(activePlanes.none()) {
			return false;
		}
		ActivePlanes help = activePlanes;
		for(uint i = 0; /*i < vecPlane.size() &&*/ help.any(); i++, help >>= 1) { 
			// vecPlane.size test is superflue if the active planes only contains 1s where planes are existent
			if(help[0]) {
				const PlaneSide ps = box.getPlaneSide(vecPlane[i]);
				if(ps == BEHIND) {
					return true;
				}
				if(ps == BEFORE) {
					activePlanes[i] = false;
				}
			}
		}
		return false;
	}
};

//namespace
}
//namespace
}
#endif
