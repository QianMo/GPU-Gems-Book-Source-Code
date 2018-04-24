#include "DXUT.h"
#include "RenderableTree.h"

#include <limits>

com_ptr<IMesh> generateBranchGeometry(IDirect3DDevice& device,
	float radius0, float radius1, float length,
	unsigned* branchIndices, float* branchOffsets, float* branchBendExps, size_t branchCount)
{
	const size_t Slices = 5;
	const size_t Stacks = 8;

	HRESULT hr;
	com_ptr<IMesh> cylinderMesh;
	V(D3DXCreateCylinder(&device, radius0, radius1, length, Slices, Stacks, &cylinderMesh, 0));

	com_ptr<IMesh> outputMesh;
	static const DWORD FVF = D3DFVF_XYZB5 | D3DFVF_LASTBETA_UBYTE4 | D3DFVF_NORMAL;
	V(cylinderMesh->CloneMeshFVF(D3DXMESH_SYSTEMMEM, FVF, &device, &outputMesh));

	DWORD branchIndicesPacked = 0;
	assert(branchCount <= 4);

	for(size_t q = 0; q < branchCount; ++q)
	{
		assert(branchIndices);
		assert(branchIndices[q] < 256);
		branchIndicesPacked <<= 8;
		branchIndicesPacked |= (branchIndices[q] & 0xff);
	}

	BranchVertex* vertexBuffer = 0;
	V(outputMesh->LockVertexBuffer(0, reinterpret_cast<void**>(&vertexBuffer)));
	for(size_t q = 0; q < outputMesh->GetNumVertices(); ++q)
	{
		vertexBuffer->pos.z += length * 0.5f;
		vertexBuffer->branchIndices = branchIndicesPacked;

		for(size_t w = 0; w < 4; ++w)
		{
			if(w < 3)
			{
				if(branchCount > w + 1)
					vertexBuffer->bendWeight[w] = pow(branchOffsets[w + 1], branchBendExps[w]);
				else
					vertexBuffer->bendWeight[w] = pow(vertexBuffer->pos.z / length, branchBendExps[w]);
			}
			else
				vertexBuffer->bendWeight[w] = 1.0f;
		}
		++vertexBuffer;
	}
	V(outputMesh->UnlockVertexBuffer());

	return outputMesh;
}

com_ptr<IMesh> concatenateMeshes(IDirect3DDevice& device, IMesh * const* meshes, D3DXMATRIX const* transforms, size_t meshCount)
{
	HRESULT hr;
	com_ptr<IMesh> outputMesh;
	V(D3DXConcatenateMeshes(const_cast<IMesh**>(meshes), static_cast<UINT>(meshCount), D3DXMESH_MANAGED,
		transforms, 0, 0,
		&device, &outputMesh));

	DWORD* attributeBuffer = 0;
	V(outputMesh->LockAttributeBuffer(0, &attributeBuffer));
	for(size_t faceIt = 0; faceIt < outputMesh->GetNumFaces(); ++faceIt)
		attributeBuffer[faceIt] = 0;
	V(outputMesh->UnlockAttributeBuffer());

	std::vector<DWORD> adjacency(outputMesh->GetNumFaces() * 3);
	V(outputMesh->GenerateAdjacency(std::numeric_limits<float>::epsilon(), &adjacency[0]));
	V(outputMesh->OptimizeInplace(D3DXMESHOPT_COMPACT | D3DXMESHOPT_VERTEXCACHE | D3DXMESHOPT_ATTRSORT, &adjacency[0], 0, 0, 0));

	return outputMesh;
}

void calcBoundingSphere(RenderableTree const& tree, D3DXVECTOR3& center, float& radius)
{
	if(tree.mesh)
	{
		HRESULT hr;
		D3DXVECTOR3* data = 0; 
		V(tree.mesh->LockVertexBuffer(0, reinterpret_cast<void**>(&data)));
		V(D3DXComputeBoundingSphere(data, tree.mesh->GetNumVertices(), D3DXGetFVFVertexSize(tree.mesh->GetFVF()), &center, &radius));
		V(tree.mesh->UnlockVertexBuffer());
	}
}