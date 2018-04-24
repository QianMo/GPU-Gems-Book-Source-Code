#ifndef __RENDERABLE_TREE_H
#define __RENDERABLE_TREE_H

#include "Cfg.h"
#include "Platform.h"

#include <vector>
#include "Tree.h"
#include "ComPtr.h"

///////////////////////////////////////////////////////////////////////////////
// Renderable tree contains data suitable for rendering
//
struct RenderableTree
{
	com_ptr<IMesh> mesh;
	std::vector<D3DXVECTOR4> originsAndPhases;
	std::vector<D3DXVECTOR4> directions;
	size_t branchCount;

	std::vector<int> branchPerHierarchyLevelCount;
};

struct BranchVertex
{
	D3DXVECTOR3 pos;
	float bendWeight[4];
	DWORD branchIndices;
	D3DXVECTOR3 normal;
};

void generateRenderableTree(IDirect3DDevice& device,
	size_t hierarchyDepth, BranchDesc const* branchDescArray, RenderableTree& dest);
void calcBoundingSphere(RenderableTree const& tree, D3DXVECTOR3& center, float& radius);

com_ptr<IMesh> concatenateMeshes(IDirect3DDevice& device, IMesh * const* meshes, D3DXMATRIX const* transforms, size_t meshCount);
com_ptr<IMesh> generateBranchGeometry(IDirect3DDevice& device, 
	float radius0, float radius1, float length, unsigned* branchIndices = 0, float* branchOffsets = 0, float* branchBendExps = 0, size_t branchCount = 0);

#endif
