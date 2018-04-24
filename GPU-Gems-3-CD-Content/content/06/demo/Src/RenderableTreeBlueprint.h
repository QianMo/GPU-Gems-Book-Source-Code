#ifndef __RENDERABLE_TREE_BLUEPRINT_H
#define __RENDERABLE_TREE_BLUEPRINT_H

#include "Cfg.h"
#include <vector>
#include <d3dx9.h>
#include "Tree.h"
#include "Utils.h"

///////////////////////////////////////////////////////////////////////////////
// Branch blueprint defines how to generate a branch for a renderable tree
// Array of branch blueprints define to generate a renderable tree
//
struct BranchBlueprint
{
	float radius0;
	float radius1;
	float length;
	D3DXMATRIX xform;
	unsigned index;
	unsigned parentIndex;

	typedef std::vector<unsigned> ParentsT;
	typedef std::vector<float> OffsetsT;
	ParentsT parents;
	OffsetsT offsets;
};

// fills in array of branch blueprints according to branch descriptions
template <typename OI>
void generateTreeBlueprint(
	BranchDesc const* branchDescArray, D3DXMATRIX const& xform, unsigned& index, BranchBlueprint::ParentsT parents, float offset, BranchBlueprint::OffsetsT offsets, unsigned& parentIndex, OI dest,
	float radiusModifier = 1.0f, float lengthModifier = 1.0f)
{
	assert(branchDescArray);

	BranchDesc branchDesc = branchDescArray[0];
	branchDesc.radiusFrom *= radiusModifier;
	branchDesc.radiusTo *= radiusModifier;
	branchDesc.length *= lengthModifier;

	BranchBlueprint branchBlueprint;
	branchBlueprint.radius0 = branchDesc.radiusFrom;
	branchBlueprint.radius1 = branchDesc.radiusTo;
	branchBlueprint.length = branchDesc.length;
	branchBlueprint.xform = xform;
	branchBlueprint.index = index;
	branchBlueprint.parentIndex = parentIndex;
	branchBlueprint.parents = parents;
	parents.push_back(index);
	offsets.push_back(offset);
	branchBlueprint.offsets = offsets;
	++index;
	++dest = branchBlueprint;
	++parentIndex;


	if(branchDesc.childrenCount == 0)
		return;

	D3DXMATRIX childRotationMatrix0;
	D3DXVECTOR3 const oX(1.0f, 0.0f, 0.0f);
	D3DXMatrixRotationAxis(&childRotationMatrix0, &oX, branchDesc.childrenDirAngle);
	float invChildCount = 1.0f / static_cast<float>(branchDesc.childrenCount - 1);

	for(size_t q = 0; q < branchDesc.childrenCount; ++q)
	{
		float childOffset = lerp( branchDesc.childrenPosFrom, branchDesc.childrenPosTo, invChildCount * q);
		D3DXVECTOR3 childPos = D3DXVECTOR3(0.0f, 0.0f, branchDesc.length * childOffset);
		float childModifier = lerp(1.0f, branchDesc.childModifier, invChildCount * q);

		D3DXMATRIX childXform;
		D3DXMatrixTranslation(&childXform, childPos.x, childPos.y, childPos.z);

		D3DXMATRIX childRotationMatrix1;
		D3DXVECTOR3 const oZ(0.0f, 0.0f, 1.0f);
		D3DXMatrixRotationAxis(&childRotationMatrix1, &oZ, branchDesc.childrenPosAngle * q);

		childXform = childRotationMatrix0 * childRotationMatrix1 * childXform * xform;

		generateTreeBlueprint(branchDescArray + 1, childXform, index, parents, childOffset, offsets, parentIndex, dest,
			childModifier * radiusModifier, childModifier * lengthModifier);
	}
}

#endif