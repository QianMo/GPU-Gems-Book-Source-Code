//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include <algorithm>
#include <stdlib.h>
#include "Occlusion.h"
#include "main.h"
#include <Mathematic/Vector3.h>
#include <Mathematic/Vector4.h>
#include <Mathematic/perlin.h>
#include <Mathematic/MathTools.h>
#include <Types/DynamicArray.h>
#include <Types/StaticArray.h>
#include <Base/Base.h>
#include <Base/SmartPointer.h>
#include <Engine/Renderer/State.h>
#include "Geometry.h"

Occlusion::Occlusion(const std::list<SmartPointer<Geometry> >& data) {
	class NodeCount : public KDNode::const_Traversal {
		mutable unsigned count;
	protected:
		virtual void traverseChildren(const KDNode::Between& n) const {
			n.getChildSmall().traversal(*this);
			n.getChildBig().traversal(*this);
		}

		virtual void onNodeBetween(const KDNode::Between& n) const {
			count++;
			traverseChildren(n);
		}

		virtual void onLeafNode(const KDNode::Leaf& n) const {
			count++;
		}
	public:
		NodeCount(): count(0) { }
		unsigned getCount() const { return count; }
	};

	kdTree = createKDTree<Geometry,TreeNodeData>(data);
	NodeCount n;
	kdTree->traversal(n);
	nodeCount = n.getCount();
}

struct OcclusionQueryState : public State {
	virtual void begin() {
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDepthMask(GL_FALSE);
	}

	virtual void end() {
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDepthMask(GL_TRUE);
	}
} queryState;

void Occlusion::issueOcclusionQuery(FrameState& fs, const KDNode::Node& n) {
	const OcclusionQuery& q = n.nodeData.q;
	//disable texturing and any fancy shaders writing to depth and color buffer
	fs.stateManager.setState(queryState);
	fs.query_cnt++;
	q.begin();
	// render bounding box for object i
	drawAABox(n.getBoundingBox());
	q.end();
}

void Occlusion::ViewFrustumCullingTraversal::traverseChildren(KDNode::Between& n) const {
	typedef Math::Vector3f V3;
	const float distSmall = n.getChildSmall().getBoundingBox().squaredDistance(fs.getPos());
	const float distBig = n.getChildBig().getBoundingBox().squaredDistance(fs.getPos());
	//traverse nearer child first
	if(distSmall < distBig) {
		n.getChildSmallPtr()->traversal(*this);
		n.getChildBigPtr()->traversal(*this);
	}
	else {
		n.getChildBigPtr()->traversal(*this);
		n.getChildSmallPtr()->traversal(*this);
	}
}

void Occlusion::ViewFrustumCullingTraversal::onNodeBetween(KDNode::Between& n) const {
	fs.traversed_nodes_cnt++;
	Frustum::ActivePlanes a(fs.frustum.getActivePlanes());
	if(!fs.frustum.cull(n.getBoundingBox())) {
		traverseChildren(n);
	}
	else {
		fs.frustum_culled_nodes_cnt++;
	}
	fs.frustum.getActivePlanes() = a;
}

void Occlusion::ViewFrustumCullingTraversal::onLeafNode(KDNode::Leaf& n) const {
	fs.traversed_nodes_cnt++;
	Geometry& g = *n.data;
	if(fs.frustum.visible(g.getBoundingBox())) {
		fs.object_cnt++;
		fs.triangle_cnt += g.triangleCount();
		g.draw(fs);
	}
	else {
		fs.frustum_culled_nodes_cnt++;
	}
}

bool Occlusion::SnWTraversal::isVisible(FrameState& fs, const KDNode::Node& n) {
	issueOcclusionQuery(fs,n);
	return n.nodeData.q.getResult() > visibilityTreshold;
}

void Occlusion::SnWTraversal::onLeafNode(KDNode::Leaf& n) const {
	fs.traversed_nodes_cnt++;
	Geometry& g = *n.data;
	if(fs.frustum.visible(g.getBoundingBox())) {
		if(isVisible(fs,n)) {
			fs.object_cnt++;
			fs.triangle_cnt += g.triangleCount();
			g.draw(fs);
		}
	}
	else {
		fs.frustum_culled_nodes_cnt++;
	}
}

void Occlusion::SnWTraversal::onNodeBetween(KDNode::Between& n) const {
	fs.traversed_nodes_cnt++;
	Frustum::ActivePlanes a(fs.frustum.getActivePlanes());
	if(!fs.frustum.cull(n.getBoundingBox())) {
		n.nodeData.visible = isVisible(fs,n);
		if(n.nodeData.visible) {
			traverseChildren(n);
		}
	}
	else {
		fs.frustum_culled_nodes_cnt++;
	}
	fs.frustum.getActivePlanes() = a;
}

void Occlusion::pullUpVisibility(KDNode& node) {
	KDNode* n = &node;
	while(0 != n) {
		if(!n->nodeData.visible) {
			n->nodeData.visible = true;
			n = n->getParent();
		}
		else {
			break;
		}
	}
}

void Occlusion::traverseNode(PquKDNode& stack, FrameState& fs, KDNode& n) {
	class ToStackTraversal : public KDNode::Traversal {
		PquKDNode& stack;
		FrameState& fs;
		virtual void onNodeBetween(KDNode::Between& n) const {
			fs.traversed_nodes_cnt++;
			stack.push(n.getChildSmallPtr());
			stack.push(n.getChildBigPtr());
		}

		virtual void onLeafNode(KDNode::Leaf& n) const {
			fs.traversed_nodes_cnt++;
			if(n.nodeData.frameID != fs.getFrame()) {
				n.nodeData.frameID = fs.getFrame();
				fs.object_cnt++;
				fs.triangle_cnt += n.data->triangleCount();
				n.data->draw(fs);
			}
		}

	public:
		ToStackTraversal(PquKDNode& vStack, FrameState& vR): stack(vStack), fs(vR) { }
	};

	n.traversal(ToStackTraversal(stack,fs));
}

bool Occlusion::enterIfFinishedOrEmptyStack(const PquKDNode& s, const QueKDNode& q) {
	if(q.empty()) {
		return false;
	}
	else {
		return q.front()->nodeData.q.finished() || s.empty();
	}
}


bool Occlusion::insideViewFrustum(FrameState& fs, const KDNode& n) {
	bool result = fs.frustum.visible(n.getBoundingBox());
	if(!result) {
		fs.frustum_culled_nodes_cnt++;
	}
	return result;
}

//this is the algorithm from the paper
void Occlusion::CHCtraversal(FrameState& fs) {
	PquKDNode traversalStack(PquKDNode_Comp(fs.getPos()));
	std::queue<KDNode*> queryQueue;
	traversalStack.push(kdTree);
	while(!traversalStack.empty() || !queryQueue.empty()) {
		//first part
		while(enterIfFinishedOrEmptyStack(traversalStack,queryQueue)) {
			KDNode& node = *(queryQueue.front());
			queryQueue.pop();
			//wait if result not available
			if(node.nodeData.q.getResult() > visibilityTreshold) {
				pullUpVisibility(node);
				traverseNode(traversalStack,fs,node);
			}
		}
		//2nd part
		if(!traversalStack.empty()) {
			KDNode& node = *traversalStack.top();
			traversalStack.pop();
			if(insideViewFrustum(fs,node)) {
				// identify previously visible nodes
				bool wasVisible = node.nodeData.visible && (fs.getFrame()-1 == node.nodeData.lastVisited);
				// identify nodes that we cannot skip queries for
				bool leafOrWasInvisible = !wasVisible || node.isLeaf();
				//reset node's visibility classification
				node.nodeData.visible = false;
				//update node's visited flag
				node.nodeData.lastVisited = fs.getFrame();
				// skip testing previously visible interior nodes
				if(leafOrWasInvisible) {
					issueOcclusionQuery(fs,node);
					queryQueue.push(&node);
				}
				//always traverse a node if it was visible
				if(wasVisible) {
					traverseNode(traversalStack,fs,node);
				}
			}
		}
	}
}

void Occlusion::viewFrustumCulling(FrameState& fs) {
	kdTree->traversal(ViewFrustumCullingTraversal(fs));
}

void Occlusion::stopAndWait(FrameState& fs) {
	kdTree->traversal(SnWTraversal(fs));
}

void Occlusion::KDTreeOnly(FrameState& fs) {
	class TreeOnly : public KDNode::const_Traversal {
	protected:
		FrameState& fs;
		virtual void traverseChildren(const KDNode::Between& n) const {
			n.getChildSmall().traversal(*this);
			n.getChildBig().traversal(*this);
		}

		virtual void onNodeBetween(const KDNode::Between& n) const {
			Frustum::ActivePlanes a(fs.frustum.getActivePlanes());
			if(!fs.frustum.cull(n.getBoundingBox())) {
				glColor3f(1.0,1.0,1.0);
				drawLineAABox(n.getBoundingBox());
				traverseChildren(n);
			}
			fs.frustum.getActivePlanes() = a;
		}

		virtual void onLeafNode(const KDNode::Leaf& n) const {
			if(fs.frustum.visible(n.getBoundingBox())) {
				glColor3f(1.0,1.0,1.0);
				drawLineAABox(n.getBoundingBox());
			}
		}

	public:
		TreeOnly(FrameState& vfs): fs(vfs) { }
	};

	class CHCTreeOnly : public TreeOnly {
	protected:
		virtual void onNodeBetween(const KDNode::Between& n) const {
			Frustum::ActivePlanes a(fs.frustum.getActivePlanes());
			if(!fs.frustum.cull(n.getBoundingBox())) {
				if(n.nodeData.visible) {
//					glColor3f(1.0,1.0,1.0);
//					drawLineAABox(n.getBoundingBox());
					traverseChildren(n);
				}
				else {
					glColor3f(1.0,0.0,0.0);
					drawLineAABox(n.getBoundingBox());
				}

			}
			fs.frustum.getActivePlanes() = a;
		}

		virtual void onLeafNode(const KDNode::Leaf& n) const {
			if(fs.frustum.visible(n.getBoundingBox())) {
				if(n.nodeData.visible) {
					glColor3f(1.0,1.0,1.0);
					drawLineAABox(n.getBoundingBox());
				}
				else {
					glColor3f(1.0,0.0,0.0);
					drawLineAABox(n.getBoundingBox());
				}
			}
		}

	public:
		CHCTreeOnly(FrameState& fs): TreeOnly(fs) { }
	};


	if(0 == fs.mode) {
		kdTree->traversal(TreeOnly(fs));
	}
	else {
		kdTree->traversal(CHCTreeOnly(fs));
	}
}
