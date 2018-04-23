//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#ifndef OcclusionH
#define OcclusionH

#include <stack>
#include <queue>
#include "MathStuff.h"
#include "FrameState.h"
#include <Engine/GlInterface/OcclusionQuery.h>
#include <Engine/EngineBase/KDTree.h>
#include "Geometry.h"

struct Occlusion {
protected:
	static const unsigned visibilityTreshold = 0;
	
	unsigned nodeCount;

	//contains the additional data needed for the chc
	struct TreeNodeData {
		bool visible;
		unsigned lastVisited;
		unsigned frameID;
		OcclusionQuery q;
		TreeNodeData(): visible(true), lastVisited(0), frameID(0) { }
	};

	//a KDTree with Geometry objects in the leafes and a TreeNodeData struct at every node
	typedef KDTreeNode<Geometry,TreeNodeData> KDNode;

	//comparator for priority queue
	class PquKDNode_Comp {
		const KDNode::AABox::V3 pos;
	public:
		PquKDNode_Comp(const KDNode::AABox::V3& position): pos(position) { }
		bool operator()(const KDNode*& a, const KDNode*& b) const {
			return a->getBoundingBox().squaredDistance(pos) > b->getBoundingBox().squaredDistance(pos);
		}
	};

	//priority queue for the nodes to traversal
	typedef std::priority_queue<KDNode*,std::vector<KDNode*>,PquKDNode_Comp> PquKDNode;
	//queue for the nodes with running occlusion queries
	typedef std::queue<KDNode*> QueKDNode;

	//the KD-Tree data structure
	KDNode *kdTree;

	//KDTree traversal used for View Frustum Culling
	class ViewFrustumCullingTraversal : public KDNode::Traversal {
	protected:
		FrameState& fs;
		virtual void traverseChildren(KDNode::Between&) const;
		virtual void onNodeBetween(KDNode::Between&) const;
		virtual void onLeafNode(KDNode::Leaf&) const;
	public:
		ViewFrustumCullingTraversal(FrameState& vfs): fs(vfs) { }
	};

	//KDTree traversal used for Stop and Wait Culling
	class SnWTraversal : public ViewFrustumCullingTraversal {
		virtual void onNodeBetween(KDNode::Between&) const;
		virtual void onLeafNode(KDNode::Leaf&) const;
		static bool isVisible(FrameState&, const KDNode::Node&);
	public:
		SnWTraversal(FrameState& fs): ViewFrustumCullingTraversal(fs) { }
	};

	//routines used for Coherent Occlusion Culling
	static void pullUpVisibility(KDNode&);
	static void traverseNode(PquKDNode&, FrameState&, KDNode&);
	static void issueOcclusionQuery(FrameState&, const KDNode&);
	static bool enterIfFinishedOrEmptyStack(const PquKDNode&, const QueKDNode&); 
	static bool insideViewFrustum(FrameState&, const KDNode&);

public:
	Occlusion(const std::list<SmartPointer<Geometry> >&);

	//this is the algorithm from the paper
	void CHCtraversal(FrameState&);

	void viewFrustumCulling(FrameState&);
	void stopAndWait(FrameState&);
	void KDTreeOnly(FrameState&);

	const unsigned getNodeCount() const { return nodeCount; }
};


#endif