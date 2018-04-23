//---------------------------------------------------------------------------
#ifndef BaseGLH
#define BaseGLH

#include <Types/StaticArray.h>
#include <Mathematic/Vector3.h>
#include <Mathematic/Geometry/AABox.h>
#include <GL/glHeader.h>
//---------------------------------------------------------------------------
inline void sendVertex3(const Math::Vector3f& v) {
	glVertex3fv(v.addr());
}

inline void sendVertex3(const Math::Vector3d& v) {
	glVertex3dv(v.addr());
}

//draws the edges of a volume with 8 corner points, like a box or a frustum (GL_LINES)
template<class T>
inline void drawLineBoxVolume(const Array<T>& v) {
	if(8 > v.size()) {
		throw BaseException("drawLineBoxVolume with Array smaller 8 called");
	}
	glBegin(GL_LINE_LOOP);
		sendVertex3(v[0]);
		sendVertex3(v[1]);
		sendVertex3(v[5]);
		sendVertex3(v[6]);
		sendVertex3(v[2]);
		sendVertex3(v[3]);
		sendVertex3(v[7]);
		sendVertex3(v[4]);
	glEnd();
	glBegin(GL_LINES);
		sendVertex3(v[0]);
		sendVertex3(v[3]);
		sendVertex3(v[1]);
		sendVertex3(v[2]);
		sendVertex3(v[4]);
		sendVertex3(v[5]);
		sendVertex3(v[6]);
		sendVertex3(v[7]);
	glEnd();
}


//draws a volume with 8 corner points, like a box or a frustum (GL_TRIANGLE_STRIP)
template<class T>
inline void drawBoxVolume(const Array<T>& v) {
	if(8 > v.size()) {
		throw BaseException("drawBoxVolume with Array smaller 8 called");
	}
	glBegin(GL_TRIANGLE_STRIP);
		sendVertex3(v[0]);
		sendVertex3(v[1]);
		sendVertex3(v[3]);
		sendVertex3(v[2]);
		sendVertex3(v[7]);
		sendVertex3(v[6]);
		sendVertex3(v[4]);
		sendVertex3(v[5]);
	glEnd();

	glBegin(GL_TRIANGLE_STRIP);
		sendVertex3(v[6]);
		sendVertex3(v[2]);
		sendVertex3(v[5]);
		sendVertex3(v[1]);
		sendVertex3(v[4]);
		sendVertex3(v[0]);
		sendVertex3(v[7]);
		sendVertex3(v[3]);
	glEnd();
}

template<class T>
inline drawLineAABox(const Math::Geometry::AABox<T>& box) {
	typedef Math::Geometry::AABox<T> AABox;
	StaticArray<AABox::V3,8> p;
	box.computeVerticesLeftHanded(p);
	drawLineBoxVolume(p);
}

template<class T>
inline drawAABox(const Math::Geometry::AABox<T>& box) {
	typedef Math::Geometry::AABox<T> AABox;
	StaticArray<AABox::V3,8> p;
	box.computeVerticesLeftHanded(p);
	drawBoxVolume(p);
}


#endif
