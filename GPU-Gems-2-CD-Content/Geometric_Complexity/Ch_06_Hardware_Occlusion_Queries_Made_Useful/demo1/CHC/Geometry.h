//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef GeometryH
#define GeometryH

#include <Base/Base.h>
#include <Base/SmartPointer.h>
#include <Mathematic/Geometry/AABox.h>
#include "FrameState.h"

class Geometry : public Base {
protected:
	State& state;
public:
	typedef Math::Geometry::AABox<float> AABox;
	typedef Math::Vector3<float> V3;

	Geometry(State& neededState);
	virtual ~Geometry() { }
	void draw(FrameState&);
	void drawGeometryCulled(FrameState&);
	virtual void drawGeometry() const = 0;
	virtual const AABox& getBoundingBox() const = 0;
	virtual const unsigned triangleCount() const = 0;
};
typedef SmartPointer<Geometry> SmpGeometry;

#endif
