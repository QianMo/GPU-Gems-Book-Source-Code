#include "RenderTraverser.h"
#include "glInterface.h"
#include "Timers.h"

extern "C"
{
	#include "MathStuff.h"
}


RenderTraverser::RenderTraverser(): mFrameID(1), mVisibilityThreshold(0),
mHierarchyRoot(NULL), mOcclusionQueries(NULL), mCurrentTestIdx(0), mIsQueryMode(false),
mNumTraversedNodes(0), mNumQueryCulledNodes(0), mNumFrustumCulledNodes(0),
mRenderTime(0), mNumRenderedGeometry(0), mUseOptimization(false)
{
}


RenderTraverser::~RenderTraverser()
{
	if(mOcclusionQueries) 
		delete [] mOcclusionQueries;
}


void RenderTraverser::Render(int mode)
{
	mDistanceQueue.push(mHierarchyRoot);

	long startTime = getTime();

	Preprocess();

	switch(mode)
	{
		case RENDER_CULL_FRUSTUM:
			RenderCullFrustum();
            break;
		case RENDER_STOP_AND_WAIT:
			RenderStopAndWait();
			break;
		case RENDER_COHERENT:
			RenderCoherentWithQueue();
			break;
		default:
			RenderCullFrustum();
			break;
	}

	mFrameID ++;
	ResetQueries();

	long endTime = getTime();
	mRenderTime = endTime - startTime;

	finishTiming();
}

/**
	this is the standard render traversal algorithm doing only frustum culling
*/
void RenderTraverser::RenderCullFrustum()
{
	while(!mDistanceQueue.empty())
	{
		HierarchyNode *node = mDistanceQueue.top();
		mDistanceQueue.pop();
	
		// interesting for the visualization, so rest and set
		node->SetVisible(false);
		mNumTraversedNodes ++;

		// We don't need to know about near plane intersection 
		// for frustum culling only, but we have to pass a parameter
		bool intersectsNearplane;
		if(InsideViewFrustum(node, intersectsNearplane))
		{
			// update node's visited flag => needed for rendering
			// so set it also here
			node->SetLastVisited(mFrameID);
			node->SetVisible(true);
			TraverseNode(node);
		}
		else
		{
			mNumFrustumCulledNodes ++;
		}
	}
}


/**
	this is the naive algorithm always waiting for the query to finish
*/
void RenderTraverser::RenderStopAndWait()
{
	while(! mDistanceQueue.empty())
	{
		HierarchyNode *node = mDistanceQueue.top();
		mDistanceQueue.pop();
		mNumTraversedNodes ++;
		// interesting for the visualization, so rest and set
		node->SetVisible(false);

		bool intersectsNearplane;
		
		if(InsideViewFrustum(node, intersectsNearplane))
		{
			// update node's visited flag
			node->SetLastVisited(mFrameID);

			// for near plane intersecting AABs possible 
			// wrong results => skip occlusion query
			if(intersectsNearplane)
			{
				node->SetVisible(true);
				TraverseNode(node);
			}
			else
			{
				IssueOcclusionQuery(node, false);
				
				// wait if result not available
				int visiblePixels = GetOcclusionQueryResult(node);
				
				// node visible
				if(visiblePixels > mVisibilityThreshold)
				{
					node->SetVisible(true);
					TraverseNode(node);
				}
				else
				{
					mNumQueryCulledNodes ++;
				}
			}					
		}
		else
		{
			mNumFrustumCulledNodes ++;
		}
	}
}

/**
	this is the algorithm as it is described in the book. It uses
	a query queue and frame-to-frame coherence in order to prevent 
	stalls and avoid unnecessary queries.
*/
void RenderTraverser::RenderCoherentWithQueue()
{
	QueryQueue queryQueue;

	//-- PART 1: process finished occlusion queries
	while(!mDistanceQueue.empty() || !queryQueue.empty())
	{
		while(!queryQueue.empty() && 
			  (ResultAvailable(queryQueue.front()) || 	mDistanceQueue.empty()))
		{
			HierarchyNode *node = queryQueue.front();
			queryQueue.pop();
			
			// wait until result available
			int visiblePixels = GetOcclusionQueryResult(node);

			if(visiblePixels > mVisibilityThreshold)
			{
				PullUpVisibility(node);
				TraverseNode(node);
			}
			else
			{
				mNumQueryCulledNodes ++;
			}
		}	

		//-- PART 2: hierarchical traversal
		if(! mDistanceQueue.empty())
		{
			HierarchyNode *node = mDistanceQueue.top();

			mDistanceQueue.pop();
	
			mNumTraversedNodes ++;

			bool intersectsNearplane;
			
			if(InsideViewFrustum(node, intersectsNearplane))
			{
				// for near plane intersecting AABs possible 
				// wrong results => skip occlusion query
				if(intersectsNearplane)
				{
					// update node's visited flag
					node->SetLastVisited(mFrameID);

					PullUpVisibility(node);
					TraverseNode(node);
				}
				else
				{		
					// identify previously visible nodes
					bool wasVisible = node->Visible() && (node->LastVisited() == mFrameID - 1);
					
					// identify nodes that we cannot skip queries for
					bool leafOrWasInvisible = !wasVisible || node->IsLeaf();

					// reset node's visibility classification 
					node->SetVisible(false);

					// update node's visited flag
					node->SetLastVisited(mFrameID);
				
					// skip testing previously visible interior nodes
					if(leafOrWasInvisible)
					{
						IssueOcclusionQuery(node, wasVisible);
						queryQueue.push(node);
					}
					
					// always traverse a node if it was visible
					if(wasVisible)
						TraverseNode(node);
				}
			}
			else
			{
				// for stats
				mNumFrustumCulledNodes ++;
			}
		}
	}
}
	

void RenderTraverser::TraverseNode(HierarchyNode *node)
{
	if(node->IsLeaf())
		mNumRenderedGeometry += node->Render();
	else // internal node: add children to priority queue for further processing
	{
		mDistanceQueue.push(node->GetLeftChild());
		mDistanceQueue.push(node->GetRightChild());
	}
}


void RenderTraverser::RenderVisualization()
{
	mDistanceQueue.push(mHierarchyRoot);

	while(!	mDistanceQueue.empty())
	{
		HierarchyNode *node = mDistanceQueue.top();
		mDistanceQueue.pop();

		// identify previously visible nodes
		bool wasVisible = node->Visible() && (node->LastVisited() == mFrameID - 1);

		if(wasVisible)
			TraverseNode(node);
		else
		{
			// also render culled nodes
			glColor3f(1.0,0.0,0.0);
			node->RenderBoundingVolumeForVisualization();		
		}
	}
}


void RenderTraverser::PullUpVisibility(HierarchyNode *node)
{
	while(node && !node->Visible())
	{
		node->SetVisible(true);
		node = node->GetParent();
	}
}

bool RenderTraverser::ResultAvailable(HierarchyNode *node)
{
	int result;

	glGetQueryivARB(node->GetOcclusionQuery(),
					GL_QUERY_RESULT_AVAILABLE_ARB, &result);

	return (result == GL_TRUE);
}

void RenderTraverser::SetHierarchy(HierarchyNode *sceneRoot)
{
	mHierarchyRoot = sceneRoot;

	// not valid anymore for new hierarchy => delete
	delete [] mOcclusionQueries;
	mOcclusionQueries = NULL;
}

HierarchyNode *RenderTraverser::GetHierarchy()
{
	return mHierarchyRoot;
}

int RenderTraverser::GetOcclusionQueryResult(HierarchyNode *node)
{
	unsigned int result;
	
	glGetQueryObjectuivARB(node->GetOcclusionQuery(), GL_QUERY_RESULT_ARB, &result);

	return (int)result;
}


void RenderTraverser::Switch2GLQueryState()
{	
	// boolean used to avoid unnecessary state changes
	if(!mIsQueryMode)
	{
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDepthMask(GL_FALSE);
		glDisable(GL_LIGHTING);
		mIsQueryMode = true;
	}
}


void RenderTraverser::Switch2GLRenderState()
{
	// boolean used to avoid unnecessary state changes
	if(mIsQueryMode)
	{
		// switch back to rendermode		
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDepthMask(GL_TRUE);
		glEnable(GL_LIGHTING);
		mIsQueryMode = false;
	}
}

void RenderTraverser::IssueOcclusionQuery(HierarchyNode *node, bool wasVisible)
{
	// get next available test id
	unsigned int occlusionQuery = mOcclusionQueries[mCurrentTestIdx++];
	
	node->SetOcclusionQuery(occlusionQuery);
	// do the actual occlusion query for this node
	glBeginQueryARB(GL_SAMPLES_PASSED_ARB, occlusionQuery);
	
	// if leaf and was visible => will be rendered anyway, thus we
	// can also test with the real geometry 
	if(node->IsLeaf() && wasVisible && mUseOptimization)
	{
		mNumRenderedGeometry += node->Render();
	}
	else
	{
		// change state so the bounding box gets not actually rendered on the screen
		Switch2GLQueryState();
		node->RenderBoundingVolume();
		Switch2GLRenderState();
	}

	glEndQueryARB(GL_SAMPLES_PASSED_ARB);
}

void RenderTraverser::Preprocess()
{
	if(!mOcclusionQueries)
	{
		mOcclusionQueries = new unsigned int[mHierarchyRoot->GetNumHierarchyNodes()];
	}

	// view frustum planes for view frustum culling
	calcViewFrustumPlanes(&mClipPlanes, mProjViewMatrix);
	calcAABNPVertexIndices(mNPVertexIndices, mClipPlanes);
	// generate ids for occlusion test
	glGenQueriesARB(mHierarchyRoot->GetNumHierarchyNodes(), mOcclusionQueries);
	mCurrentTestIdx = 0;

	// reset statistics
	mNumTraversedNodes = 0;
	mNumQueryCulledNodes = 0;
	mNumFrustumCulledNodes = 0;
	mNumRenderedGeometry = 0;
}


void RenderTraverser::ResetQueries()
{
	// tell the driver that the occlusion queries won't be needed any more
	glDeleteQueriesARB(mHierarchyRoot->GetNumHierarchyNodes(), mOcclusionQueries);
}


void RenderTraverser::SetViewpoint(Vector3 const &viewpoint)
{
	copyVector3(mViewpoint, viewpoint);
}
	

void RenderTraverser::SetProjViewMatrix(Matrix4x4 const &projViewMatrix)
{
	copyMatrix(mProjViewMatrix, projViewMatrix);
}


bool RenderTraverser::InsideViewFrustum(HierarchyNode *node, bool &intersectsNearplane)
{
	Vector3x8 vertices;
	
	calcAABoxPoints(vertices, node->GetBoundingVolume());

	// test all 6 clip planes if a bouning box vertex is outside
	// only need the n and p vertices of the bouding box to determine this
	for (int i = 0; i < 6; i++)
	{		
		// test the n-vertex
		// note: the calcAABNearestVertexId should be preprocessed
		if(!pointBeforePlane(mClipPlanes.plane[i], vertices[mNPVertexIndices[i * 2]]))
		{
			// outside
			return false;
		}
	}

	// test if bounding box is intersected by nearplane (using the p-vertex)
	intersectsNearplane = (!pointBeforePlane(mClipPlanes.plane[5], vertices[mNPVertexIndices[11]]));

	// -- get vector from viewpoint to center of bounding volume
	Vector3 vec;
	calcAABoxCenter(vec, node->GetBoundingVolume());
	diffVector3(vec, vec, mViewpoint);

	// compute distance from nearest point to viewpoint
	diffVector3(vec, vertices[calcAABNearestVertexIdx(vec)], mViewpoint);
	node->SetDistance(squaredLength(vec));
	
	return true;
}


void RenderTraverser::SetVisibilityThreshold(int threshold)
{
	mVisibilityThreshold = threshold;
}

long RenderTraverser::GetRenderTime()
{
	return mRenderTime;
}

int RenderTraverser::GetNumTraversedNodes()
{
	return mNumTraversedNodes;
}

int RenderTraverser::GetNumQueryCulledNodes()
{
	return mNumQueryCulledNodes;
}

int RenderTraverser::GetNumFrustumCulledNodes()
{
	return mNumFrustumCulledNodes;
}


int RenderTraverser::GetNumRenderedGeometry()
{
	return mNumRenderedGeometry;
}


int RenderTraverser::GetVisibilityThreshold()
{
	return mVisibilityThreshold;
}

void RenderTraverser::SetUseOptimization(bool useOptimization)
{
	mUseOptimization = useOptimization;
}