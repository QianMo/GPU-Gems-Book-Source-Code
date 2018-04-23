//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef SceneDataH
#define SceneDataH

#include "MathStuff.h"
#include "FrameState.h"
#include <GL/glHeader.h>
#include "Geometry.h"
#include "GeometryObjects.h"
#include "Occlusion.h"

struct Scene {
protected:
	struct HeightFieldState : public State {
		virtual void begin() {
			glEnable(GL_TEXTURE_1D);
			glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,Math::Vector4f::ONE.addr());
		}

		virtual void end() {
			glDisable(GL_TEXTURE_1D);
		}
	} heightFieldState;

	struct TreeState : public State {
		virtual void begin() {
			glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,Math::Vector4f::UNIT_Z.addr());
		}

		virtual void end() {
		}
	} treeState;

	Occlusion* cull;

	typedef DynamicArray<GLubyte> DynByte;
	typedef std::list<SmpGeometry> Objects;

	DynamicArray<float> texCoord;
	DynVector3 normal;
	DynVector3 vertex;

	BaseTree baseTree;
	unsigned triangleCount;

	Objects scene;
	Objects caster;
	
	//Axis-Aligned Bounding Box of the scene
	AABox sceneAABox;

	void makeOTexture(void);
	void makeHeightTexture(void);
	inline unsigned getID(const unsigned u, const unsigned v, const unsigned mapSize) const;
	void createFaceTriangleStrip(ArrID& vertexID, const unsigned mapSize,
							 const unsigned U, const unsigned V,
							 const unsigned deltaU, const unsigned deltaV);
	void createHeightFieldObjects(const DynVector3& vertex, const unsigned mapSize, const unsigned objVertexDelta);
	void createHeightField(const DynByte& heightMap, const unsigned mapSize, const float& scale, const float& sY);
	void createTrees(const DynByte& heightMap, const DynVector3& v, const unsigned count, const unsigned treeSize, const unsigned MAP_SIZE, const float terrainSize);
	
public:
	Scene();
	~Scene();
	
	//returns Axis-Aligned Bounding Box of the scene
	const AABox& getAABox() const { return sceneAABox; }

	//draws the scene :)
	void draw(FrameState&);
	//draws the shadow casters
	void drawShadowCasters(FrameState&);
	//visualizes the KDTree
	void drawKDTree(FrameState&);
	
	const Math::Vector3f::ElementType getHeight(const Math::Vector3f& pos) const;
	const unsigned getSceneTriangleCount() const { return triangleCount; }
	const unsigned getObjectCount() const { return scene.size(); }
	const unsigned getNodeCount() const { return cull->getNodeCount(); }
};


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
