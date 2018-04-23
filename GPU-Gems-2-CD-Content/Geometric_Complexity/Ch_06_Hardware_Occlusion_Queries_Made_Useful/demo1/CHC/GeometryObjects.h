//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.
#ifndef GeometryObjectsH
#define GeometryObjectsH

#include <Types/DynamicArray.h>
#include <Mathematic/Vector3.h>
#include "Geometry.h"

typedef DynamicArray<Math::Vector3f> DynVector3;
typedef DynamicArray<unsigned>  ArrID;

class BaseTree : public Geometry {
protected:
	const static unsigned complex = 32;
	GLuint list;
	AABox bb;

public:
	BaseTree(State& neededState): Geometry(neededState), bb(V3::ZERO,V3::ZERO) {
		V3 pos = V3::ZERO;
		const float size = 1.0f;
		const float trans = size/1.5;
		const float trunkThick = size/7.0;
		const float outR = size/5.0;
		const float inR = size/20.0;
		pos[1] -= trunkThick;
		const V3 min = V3(pos[0]-outR,pos[1],pos[2]-outR);
		const V3 max = V3(pos[0]+outR,pos[1]+size,pos[2]+outR);
		bb = AABox(min,max);

		list = glGenLists(1);
		glNewList(list,GL_COMPILE);
		glPushMatrix();
			glTranslatef(pos[0],pos[1],pos[2]);
			glRotatef(-90.0,1.0,0.0,0.0);
			glutSolidCone(trunkThick,size,complex,2);
			glTranslatef(0.0,0.0,trans);
			glutSolidTorus(inR,outR,8,complex);	
		glPopMatrix();
		glEndList();
	}
	virtual ~BaseTree() {
		glDeleteLists(list,1);
	}

	virtual void drawGeometry() const {
		glCallList(list);
	}
	virtual const AABox& getBoundingBox() const { return bb; }
	virtual const unsigned triangleCount() const { return complex*2*2+complex*8*2; }
};

class Tree : public Geometry {
protected:
	AABox box;
	const BaseTree& bt;
	const V3 pos;
	float size;

public:
	Tree(State& neededState, const V3& position, const float& vSize, const BaseTree& baseTree): 
		Geometry(neededState), box(position,position), bt(baseTree), pos(position), size(vSize) {
		Math::clamp(size,1.0f,9.0f);
		AABox aBox = bt.getBoundingBox();
		aBox.scale(size);
		aBox.translate(position);
		box = aBox;
	}

	virtual void drawGeometry() const {
		glPushMatrix();
			glTranslatef(pos[0],pos[1],pos[2]);
			glScalef(size,size,size);
			bt.drawGeometry();
		glPopMatrix();
	}
	virtual const AABox& getBoundingBox() const { return box; }
	virtual const unsigned triangleCount() const { return bt.triangleCount(); }
};


class HeightField : public Geometry {
protected:
	const AABox bb;
	const ArrID vertexID;
	static AABox calcAABox(const ArrID& ids, const DynVector3& vertex) {
		if(ids.size() < 3)
			throw BaseException("Height field too small");
		AABox box(vertex[ids[0]],vertex[ids[1]]);
		for(unsigned i = 2; i < ids.size(); i++) {
			box.expandToContain(vertex[ids[i]]);
		}
		return box;
	}

public:
	HeightField(State& neededState, const ArrID& ids, const DynVector3& vertex): 
		Geometry(neededState), vertexID(ids), bb(calcAABox(ids,vertex)) { }
	virtual void drawGeometry() const {
		glDrawElements(GL_TRIANGLE_STRIP,vertexID.size(),GL_UNSIGNED_INT,&vertexID);
	}
	virtual const AABox& getBoundingBox() const { return bb; }
	virtual const unsigned triangleCount() const { return vertexID.size()-2; }
};

#endif
