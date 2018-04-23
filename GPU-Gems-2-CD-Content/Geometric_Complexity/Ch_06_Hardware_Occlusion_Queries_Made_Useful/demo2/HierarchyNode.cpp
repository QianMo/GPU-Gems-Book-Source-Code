#include "HierarchyNode.h"
#include "glInterface.h"
#include <math.h>
#include <limits.h>
#include <float.h>

typedef vector<Geometry *> GeometryList;

// criteria and values partly taken from Piringer's thesis

// values used as termination criteria for the tree generation
// the overall surface
float HierarchyNode::sSurfaceThreshold;
// the maximum tree depth
int HierarchyNode::sMaxDepth;
// maximum number of objects in a node
int HierarchyNode::sGeometryThreshold;

// percentage of allowed deviation from the center split plane
float HierarchyNode::sSplitBandwith;

// also render bounding volume
bool HierarchyNode::sRenderBoundingVolume = false;

HierarchyNode::HierarchyNode(): mVisible(false), mLastVisited(0), 
mParent(NULL), mOcclusionQuery(0), mAABValid(false), mLeftChild(NULL), mRightChild(NULL),
mNumHierarchyNodes(1), mSplitAxis(X_AXIS), mLastRendered(-1), mSplitValue(0),
mDepth(0), mEnclosedSpaceValid(false), mDistance(0)
{
	copyVector3Values(mBoundingBox.min, 0, 0, 0);
	copyVector3Values(mBoundingBox.max, 0, 0, 0);

	copyVector3Values(mEnclosedSpace.min, 0, 0, 0);
	copyVector3Values(mEnclosedSpace.max, 0, 0, 0);

	// bounding box color
	mBoxColor[0] = mBoxColor[2] = 0; mBoxColor[1] = 1.0;
}

HierarchyNode::HierarchyNode(const Vector3 boundLower, const Vector3 boundUpper, 
							 HierarchyNode *parent, int depth)
:mNumHierarchyNodes(1), mOcclusionQuery(0), mLeftChild(NULL), 
mRightChild(NULL), mSplitAxis(X_AXIS), mVisible(false), 
mLastVisited(0), mParent(parent), mAABValid(false), 
mLastRendered(-1), mSplitValue(0), mDepth(depth), 
mEnclosedSpaceValid(true), mDistance(0)
{
	copyVector3Values(mBoundingBox.min, 0, 0, 0);
	copyVector3Values(mBoundingBox.max, 0, 0, 0);

	copyVector3(mEnclosedSpace.min, boundLower);
	copyVector3(mEnclosedSpace.max, boundUpper);

	float vol = calcAABoxSurface(mEnclosedSpace);

	mBoxColor[0] = mBoxColor[2] = 0; mBoxColor[1] = 1.0;
}


void HierarchyNode::InitKdTree(HierarchyNode *root)
{
	sMaxDepth = (int)((log((float) root->GetGeometry().size())/log(2.0f)) * 20.0f);

	sSurfaceThreshold = FLT_MAX;

	// factor times mininal surface as determination criterium
	const float factor = 2.5;

	for (GeometryList::const_iterator it = root->GetGeometry().begin(); it != root->GetGeometry().end(); it++)
	{
		// same geometry can possible be in to or more nodes => also test for them
		float surface = calcAABoxSurface((*it)->GetBoundingVolume()) * factor;

		if(surface < sSurfaceThreshold) sSurfaceThreshold = surface;
	}

	// number of objects in a leaf
	sGeometryThreshold = 1;

	// percentage of allowed deviation from the center split plane
	sSplitBandwith = 0.15;
}


HierarchyNode::~HierarchyNode()
{
	if(mLeftChild)
		delete mLeftChild;

	if(mRightChild)
		delete mRightChild;
}

int HierarchyNode::Render()
{
	int renderedGeometry = 0;

	if(sRenderBoundingVolume)
	{
		glColor3fv(mBoxColor);
        RenderBoundingVolumeForVisualization();
	}

	// prevent the geometry to be rendered several times in the same frame
	if(mLastRendered != mLastVisited)
	{
		for (GeometryList::const_iterator it = mGeometry.begin(); it != mGeometry.end(); it++)
		{
			// same geometry can possible be in to or more nodes => also test for them
			if((*it)->GetLastRendered() != mLastVisited)
			{
				(*it)->SetLastRendered(mLastVisited);
				(*it)->Render();
				renderedGeometry ++;
			}
		}
		mLastRendered = mLastVisited;
	}

	return renderedGeometry;
}

bool HierarchyNode::Visible()
{
	return mVisible;
}

void HierarchyNode::SetVisible(bool visible)
{
	mVisible = visible;
}

void HierarchyNode::SetLeftChild(HierarchyNode *child)
{
	mLeftChild = child;
}

void HierarchyNode::SetRightChild(HierarchyNode *child)
{
	mRightChild = child;
}

HierarchyNode *HierarchyNode::GetLeftChild()
{
	return mLeftChild;
}

HierarchyNode *HierarchyNode::GetRightChild()
{
	return mRightChild;
}

int HierarchyNode::LastVisited()
{
	return mLastVisited;
}

void HierarchyNode::SetLastVisited(int lastVisited)
{
	mLastVisited = lastVisited;
}

bool HierarchyNode::IsLeaf()
{
	return (!mLeftChild && !mRightChild);
}

void HierarchyNode::AddGeometry(Geometry *geometry)
{
	mGeometry.push_back(geometry);
		
	if(!mAABValid)
	{
		copyVector3(mBoundingBox.min, geometry->GetBoundingVolume().min);
		copyVector3(mBoundingBox.max, geometry->GetBoundingVolume().max);

		mAABValid = true;
	}
	else
		combineAABoxes(&mBoundingBox, geometry->GetBoundingVolume());

	// root node
	if(mDepth == 0)
	{
		if(!mEnclosedSpaceValid)
		{
			copyVector3(mEnclosedSpace.min, geometry->GetBoundingVolume().min);
			copyVector3(mEnclosedSpace.max, geometry->GetBoundingVolume().max);

			mEnclosedSpaceValid = true;
		}
		else 
			combineAABoxes(&mEnclosedSpace, geometry->GetBoundingVolume());	
	}
	else
	{
		// cut boxes so they fit into the enclosed space
		clipAABoxByAABox(&mBoundingBox, mEnclosedSpace);
	}
}

HierarchyNode *HierarchyNode::GetParent()
{
	return mParent;
}

int HierarchyNode::GetOcclusionQuery()
{
	return mOcclusionQuery;
}

void HierarchyNode::SetOcclusionQuery(int occlusionQuery)
{
	mOcclusionQuery = occlusionQuery;
}

void HierarchyNode::RenderBoundingVolume()
{
	Vector3x8 vertices;

	calcAABoxPoints(vertices, mBoundingBox);

	//glPolygonMode(GL_FRONT, GL_LINE);
	//     7+------+6
	//     /|     /|
	//    / |    / |
	//   / 4+---/--+5
	// 3+------+2 /    y   z
	//  | /    | /     |  /
	//  |/     |/      |/
	// 0+------+1      *---x

	//---- render AABB
	glBegin(GL_TRIANGLE_FAN);
		glVertex3dv(vertices[6]);
		glVertex3dv(vertices[5]);
		glVertex3dv(vertices[4]);
		glVertex3dv(vertices[7]);
		glVertex3dv(vertices[3]);
		glVertex3dv(vertices[2]);
		glVertex3dv(vertices[1]);
		glVertex3dv(vertices[5]);
	glEnd();

	//---- render second half of AABB
	glBegin(GL_TRIANGLE_FAN);
		glVertex3dv(vertices[0]);
		glVertex3dv(vertices[1]);
		glVertex3dv(vertices[2]);
		glVertex3dv(vertices[3]);
		glVertex3dv(vertices[7]);
		glVertex3dv(vertices[4]);
		glVertex3dv(vertices[5]);
		glVertex3dv(vertices[1]);
	glEnd();
}


void HierarchyNode::RenderBoundingVolumeForVisualization()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);
		
	RenderBoundingVolume();

	glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}



const AABox &HierarchyNode::GetBoundingVolume()
{
	return mBoundingBox;
}

int	HierarchyNode::GenerateKdTree()
{
	// check the termination criterium (a heuristic)
	if (SimpleEnough()) 
	{
		return 1;
	}

	// largest dimension will be split
	Vector3 size;
	diffVector3(size, mBoundingBox.max, mBoundingBox.min);
    
	if (size[X_AXIS] > size[Y_AXIS])	
		mSplitAxis	= (size[X_AXIS] > size[Z_AXIS]) ? X_AXIS : Z_AXIS;
	else					
		mSplitAxis	= (size[Y_AXIS] > size[Z_AXIS]) ? Y_AXIS : Z_AXIS;

	// select the value of the split plane
	mSplitValue = ComputeSplitPlane();
	
	// generate the children
	Vector3 changedLower;
	Vector3 changedUpper;

	copyVector3(changedLower, mEnclosedSpace.min);
	copyVector3(changedUpper, mEnclosedSpace.max);
	   	
	changedLower[mSplitAxis] = mSplitValue;
	changedUpper[mSplitAxis] = mSplitValue;

	mLeftChild	= new HierarchyNode(mEnclosedSpace.min, changedUpper, this, mDepth + 1);
	mRightChild	= new HierarchyNode(changedLower, mEnclosedSpace.max, this, mDepth + 1);

	// add the geometry to the according children
	for (GeometryList::iterator it = mGeometry.begin(); it != mGeometry.end(); it++)
	{
		if ((*it)->GetBoundingVolume().min[mSplitAxis] >= mSplitValue)
		{
			// box lies completely within right part
			mRightChild->AddGeometry(*it);
		}
		else if ((*it)->GetBoundingVolume().max[mSplitAxis] <= mSplitValue)
		{
			// box lies completely within left part
			mLeftChild->AddGeometry((*it));
		}
		else
		{
			//---- box intersects both parts
			mLeftChild->AddGeometry((*it));
			mRightChild->AddGeometry((*it));
		}
	}

	//---- we continue with the children
	int leftSize  = mLeftChild->GenerateKdTree();
	int rightSize = mRightChild->GenerateKdTree();

	// since the geometry is now referenced by the children
	mGeometry.clear();
	
	mNumHierarchyNodes = leftSize + rightSize + 1;

	return mNumHierarchyNodes;
}

bool HierarchyNode::SimpleEnough()
{
	return ((mGeometry.size() <= (unsigned int)sGeometryThreshold) ||
		    (calcAABoxSurface(mBoundingBox) <= sSurfaceThreshold) ||
			(sMaxDepth <= mDepth));
}

float HierarchyNode::ComputeSplitPlane()
{
	float left	= mBoundingBox.min[mSplitAxis];
	float right	= mBoundingBox.max[mSplitAxis];

	// the smaller the value returned from the heuristic, the better => big starting value
	float bestValue	= FLT_MAX; 
	float result = 0.0f;

	bool found	= false;

	// calculate the borders of the band
	float currLeft, currRight;
	currLeft = currRight = (left + right) / 2.0f;
	
	currLeft  -= (right - left) * sSplitBandwith;
	currRight += (right - left) * sSplitBandwith;

	// check all geometry within that node
	for (GeometryList::const_iterator it = mGeometry.begin(); it != mGeometry.end(); it++)
	{
		// one border of the geometry's AABB
		float leftPlane	 = (*it)->GetBoundingVolume().min[mSplitAxis];
		// the other border of the geometry's AABB
		float rightPlane = (*it)->GetBoundingVolume().max[mSplitAxis];
		
		// only consider planes that lie within the band
		if ((leftPlane > currLeft) && (leftPlane < currRight))
		{
			// compute the heuristic for the left plane and note the value if it was good
			float currValue	= ComputeHeuristics(leftPlane);

			if (currValue < bestValue)
			{
				bestValue = currValue;
				result	  = leftPlane;
				found	  = true;
			}
		}

		if ((rightPlane > currLeft) && (rightPlane < currRight))
		{
			// compute the heuristic for the right plane and note the value if it was good
			float currValue = ComputeHeuristics(rightPlane);

			if (currValue < bestValue)
			{
				bestValue	= currValue;
				result		= rightPlane;
				found		= true;
			}
		}
	}

	// in case we haven't found any proper plane, we simply take the center
	if (!found)	
		result = (left + right) / 2.0f;
			
	return result;
}

float HierarchyNode::ComputeHeuristics(float pos)
{
	// this implements a very simple heuristic: it simply counts the nodes being intersected by the splitplane
	float result = 0.0f;

	for (GeometryList::const_iterator it = mGeometry.begin(); it != mGeometry.end(); it++)
	{
		if (((*it)->GetBoundingVolume().min[mSplitAxis] < pos) &&
			((*it)->GetBoundingVolume().max[mSplitAxis] > pos))
		{
			result += 1.0f;
		}
	}

	return result;
}

int HierarchyNode::GetNumHierarchyNodes()
{
	return mNumHierarchyNodes;
}

void HierarchyNode::PushChildrenOrdered(const Vector3 viewpoint, TraversalStack &traversalStack)
{
	if(viewpoint[mSplitAxis] > mSplitValue)
	{
		traversalStack.push(mLeftChild);
		traversalStack.push(mRightChild);
	}
	else
	{
		traversalStack.push(mRightChild);
		traversalStack.push(mLeftChild);
	}
}


void HierarchyNode::SetRenderBoundingVolume(bool renderBoundingVolume)
{
	sRenderBoundingVolume = renderBoundingVolume;
}


GeometryList &HierarchyNode::GetGeometry()
{
	return mGeometry;
}


float HierarchyNode::GetDistance()
{
	return mDistance;
}


void HierarchyNode::SetDistance(float distance)
{
	mDistance = distance;
}