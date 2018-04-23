//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef FrameStateH
#define FrameStateH

#include <Mathematic/Vector3.h>
#include <Mathematic/Geometry/Frustum.h>
#include <Engine/Renderer/State.h>

class FrameState {
protected:
	unsigned frame;
	Math::Vector3f pos;
public:
	Math::Geometry::Frustum<float> frustum;
	StateManager stateManager;
	unsigned object_cnt;
	unsigned query_cnt;
	unsigned triangle_cnt;
	unsigned traversed_nodes_cnt;
	unsigned frustum_culled_nodes_cnt;
	unsigned mode; //culling mode: view frustum(0), stop and wait(1) or coherent occlusion culling(2)
	
	FrameState(): frame(0), pos(Math::Vector3f::ZERO), 
		object_cnt(0), query_cnt(0), triangle_cnt(0), mode(0),
		traversed_nodes_cnt(0), frustum_culled_nodes_cnt(0) { } 

	FrameState(const unsigned frameID,
				const Math::Vector3f& p,
				Math::Geometry::Frustum<float>& f,
				const unsigned vMode): 
		frame(frameID), pos(p), frustum(f), 
		object_cnt(0), query_cnt(0), triangle_cnt(0), mode(vMode),
		traversed_nodes_cnt(0), frustum_culled_nodes_cnt(0) { }
	
	FrameState& operator=(const FrameState& fs) {
		frame = fs.frame;
		pos = fs.pos;
		frustum = fs.frustum;
		stateManager = fs.stateManager;
		object_cnt = fs.object_cnt;
		query_cnt = fs.query_cnt;
		triangle_cnt = fs.triangle_cnt;
		traversed_nodes_cnt = fs.traversed_nodes_cnt;
		frustum_culled_nodes_cnt = fs.frustum_culled_nodes_cnt;
		mode = fs.mode;
		return *this;
	}

	const unsigned getFrame() const { return frame; }
	const Math::Vector3f getPos() const { return pos; }
};

#endif
