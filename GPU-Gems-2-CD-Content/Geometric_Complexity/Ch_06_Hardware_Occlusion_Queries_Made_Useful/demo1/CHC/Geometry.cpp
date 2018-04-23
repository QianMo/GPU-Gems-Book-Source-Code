//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include "Geometry.h"

Geometry::Geometry(State& neededState): state(neededState) { 
}

void Geometry::draw(FrameState& fs) {
	fs.stateManager.setState(state);
	drawGeometry();
}

void Geometry::drawGeometryCulled(FrameState& fs) {
	const AABox& box = getBoundingBox();
	if(fs.frustum.visible(box)) {
		drawGeometry();
	}
}
