#ifndef NO_PRAGMA_ONCE
#pragma once
#endif

#ifndef RENDERTRAVERSER_H
#define RENDERTRAVERSER_H

#include "glInterface.h"
#include "HierarchyNode.h"
#include <queue>
//#include <priority_queue>
#include <stack>

using namespace std;

typedef stack<HierarchyNode *> TraversalStack;
typedef queue<HierarchyNode *> QueryQueue;
typedef priority_queue<HierarchyNode *, vector<HierarchyNode *>, myless<vector<HierarchyNode *>::value_type> > PriorityQueue;

/**
	this is the class that actually traverses the hierarchy and renders
	the associated geometry. nodes of the hierarchy are culled depending on the chosen 
	rendering mode. one of the modii (RENDER_COHERENT) is the algorithm described in the book.
	Additionally, some statistics are calculated.
*/

class RenderTraverser
{
public:
	enum {RENDER_CULL_FRUSTUM, RENDER_STOP_AND_WAIT, RENDER_COHERENT, NUM_RENDERMODES};

	RenderTraverser();
	//! Renders the scene with the specified mode
	/**
		The mode is one of
		RENDER_CULL_FRUSTUM: renders the scene with view frustum culling only
		RENDER_STOP_AND_WAIT: renders the scene with the hierarchical stop and wait algorithm
		RENDER_COHERENT: renders the scene with the coherent hierarchical algorithm
	*/
	void Render(int mode=RENDER_CULL_FRUSTUM);
	//! sets the scene hierarchy.
	void SetHierarchy(HierarchyNode *sceneRoot);
	//! sets viewpoint
	void SetViewpoint(Vector3 const &viewpoint);
	//! sets view projection matrix
	void SetProjViewMatrix(Matrix4x4 const &projViewMatrix);
	//! returns root of hierarchy
	HierarchyNode *GetHierarchy();
	//! sets visible pixels threshold for visibility classification
	void SetVisibilityThreshold(int threshold);
	//! returns visibility threshold
	int GetVisibilityThreshold();

	// several statistics for a rendering pass

	//! returns rendering time of the specified algorihtm
	long GetRenderTime();
	//! returns number of traversed nodes
	int  GetNumTraversedNodes();
	//! returns the number of hierarchy nodes culled by the occlusion query
	int GetNumQueryCulledNodes();
	//! returns the number of hierarchy nodes culled by the frustum culling only
	int GetNumFrustumCulledNodes();
	//! returns number of rendered geometric objects (e.g., teapots, ...)
	int GetNumRenderedGeometry();

	//! renders a visualization of the hierarchy
	void RenderVisualization();

	//! use optimization to take leaf nodes instead of bounding box for occlusion queries	
	void SetUseOptimization(bool useOptimization);

protected:

	//! renders the scene with view frustum culling only
	void RenderCullFrustum();
	//! renders the scene with the hierarchical stop and wait algorithm
	void RenderStopAndWait();
	//! renders the scene with the coherent hierarchical algorithm and the query queye
	void RenderCoherentWithQueue();
	//! does some importand initialisations
	void Preprocess();
	
	//! returns occlusion query result for specified node
	int GetOcclusionQueryResult(HierarchyNode *node);
	//! the node is traversed as usual
	void TraverseNode(HierarchyNode *node);
	//! visibility is pulled up from visibility of children 
	void PullUpVisibility(HierarchyNode *node);
	//! is result available from query queue?
	bool ResultAvailable(HierarchyNode *node);
	//! issues occlusion query for specified node
	void IssueOcclusionQuery(HierarchyNode *node, bool wasVisible);
	//! resets occlusion queries after a traversal
	void ResetQueries();
	//! true if bounding box is culled by view frustum culling
	/**
		intersectsNearplane returns true if bounding box intersects the near plane.
		additionally stores the distance from the near plane to the center of the 
		current node with the node. this will for front-to-back ordering
	*/
	bool InsideViewFrustum(HierarchyNode *node, bool &intersects);
	//! switches to normal render mode
	void Switch2GLRenderState();
	//! switches to occlusion query mode (geometry not rendered on the screen)
	void Switch2GLQueryState();

protected:

	// the current clip planes of the view frustum
	VecPlane mClipPlanes;
	// the indices of the np-vertices of the bounding box for view frustum culling
	int mNPVertexIndices[12];

	Vector3 mViewpoint;
	Matrix4x4 mProjViewMatrix;
	HierarchyNode *mHierarchyRoot;
	
	//! we use a priority queue rather than a renderstack
	PriorityQueue mDistanceQueue; 
	
	int mFrameID;
	int mVisibilityThreshold;
	unsigned int *mOcclusionQueries;
	int mCurrentTestIdx;
	bool mIsQueryMode;
		
	// statistics
	int mNumTraversedNodes;
	int mNumQueryCulledNodes;
	int mNumFrustumCulledNodes;
	int mNumRenderedGeometry;

	long mRenderTime;

	bool mUseOptimization;
};

#endif // RENDERTRAVERSER_H