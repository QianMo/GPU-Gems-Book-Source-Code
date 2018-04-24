#include "DXUT.h"
#include "RenderableTree.h"
#include "DXUTShapes.h"

#include <list>
#include "RenderableTreeBlueprint.h"

void generateRenderableTree(
	IDirect3DDevice& device,
	size_t hierarchyDepth,
	BranchDesc const* branchDescArray,
	RenderableTree& dest)
{
	assert(branchDescArray);

	size_t descCount = 1;
	for(BranchDesc const* bDescIt = branchDescArray; bDescIt->childrenCount != 0; ++bDescIt, ++descCount);

	std::vector<float> branchBendExps(descCount);
	for(size_t q = 0; q < descCount; ++q)
		branchBendExps[q] = branchDescArray[q].bendExp;

	assert(hierarchyDepth <= descCount);

	typedef std::list<BranchBlueprint> TreeBlueprintT;
	TreeBlueprintT treeBlueprint;

	D3DXMATRIX xform;
	D3DXMatrixIdentity(&xform);
	BranchBlueprint::ParentsT parents;
	BranchBlueprint::OffsetsT offsets;
	unsigned index = 0U;
	unsigned parentIndex = ~0U;
	generateTreeBlueprint(branchDescArray, xform, index, parents, 0.0f, offsets, parentIndex, std::back_inserter(treeBlueprint));

	if(treeBlueprint.empty())
		return;

	std::vector<com_ptr<IMesh> > managedMeshes(treeBlueprint.size());
	std::vector<IMesh*> branchMeshes(treeBlueprint.size());
	std::vector<D3DXMATRIX> branchXforms(treeBlueprint.size());
	dest.originsAndPhases.resize(treeBlueprint.size());
	dest.directions.resize(treeBlueprint.size());
	dest.branchPerHierarchyLevelCount.resize(4);
	std::fill(dest.branchPerHierarchyLevelCount.begin(), dest.branchPerHierarchyLevelCount.end(), 0);

	size_t q = 0;
	BranchBlueprint::ParentsT affectingBranches; affectingBranches.reserve(4);
	BranchBlueprint::OffsetsT branchOffsets; branchOffsets.reserve(4);
	TreeBlueprintT::const_iterator firstBranchIt = treeBlueprint.begin();
	for(TreeBlueprintT::const_iterator branchIt = firstBranchIt; branchIt != treeBlueprint.end(); ++branchIt, ++q)
	{
		D3DXMATRIX branchLocalMatrix;
		D3DXMatrixTranslation(&branchLocalMatrix, 0, 0, branchIt->length*0.0f);
		
		affectingBranches.resize(0);
		BranchBlueprint::ParentsT parents = branchIt->parents;
		parents.push_back(branchIt->index);
		std::copy(parents.rbegin(), parents.rend(), std::back_inserter(affectingBranches));

		branchOffsets.resize(0);
		BranchBlueprint::OffsetsT offsets = branchIt->offsets;

		com_ptr<IMesh> branchMesh = generateBranchGeometry(device, 
			branchIt->radius0, branchIt->radius1, branchIt->length,
			(affectingBranches.size())?&affectingBranches[0]: 0,
			(offsets.size())?&offsets[0]: 0,
			&branchBendExps[0],
			affectingBranches.size());
		managedMeshes[q] = branchMesh;
		branchMeshes[q] = branchMesh.get();
		branchXforms[q] = branchLocalMatrix * branchIt->xform;

		dest.originsAndPhases[q].x = branchXforms[q]._41;
		dest.originsAndPhases[q].y = branchXforms[q]._42;
		dest.originsAndPhases[q].z = branchXforms[q]._43;
		dest.originsAndPhases[q].w = cosf(float(q*0.1f));

		D3DXVECTOR3 branchDir(branchXforms[q]._31, branchXforms[q]._32, branchXforms[q]._33);
		branchDir.z = 0.0f;
		D3DXVec3Normalize(&branchDir, &branchDir);
		branchDir *= -1.0f;

		dest.directions[q].x = branchDir.x;
		dest.directions[q].y = branchDir.y;
		dest.directions[q].z = branchDir.z;
		dest.directions[q].w = 0.0f;

		++dest.branchPerHierarchyLevelCount[parents.size()];
	}
	dest.branchCount = treeBlueprint.size();
	dest.mesh = concatenateMeshes(device, &branchMeshes[0], &branchXforms[0], branchMeshes.size());
}