#include "DXUT.h"
//#include "dxstdafx.h"
#include "DXUTShapes.h"
#include "RenderableTree.h"

#include <limits>
#include <algorithm>
#include <functional>

namespace 
{

	struct UTVertex
	{
		D3DXVECTOR3 pos;
		D3DXVECTOR3 normal;
	};

	template <typename T>
	class BufferLock
	{
	protected:
		HRESULT hr;
		com_ptr<ID3DX10MeshBuffer> mBuffer;

	protected:
		BufferLock() {}
		BufferLock(BufferLock const& c);
		BufferLock& operator= (BufferLock const& rhs);
		void lock(ID3DX10MeshBuffer& buffer, T*& data)
		{
			V(buffer.Map(reinterpret_cast<void**>(&data), 0));
		}

	public:
		~BufferLock() { V(mBuffer->Unmap()); }
		size_t bufferSize() const { return mBuffer->GetSize(); }
	};

	template <typename T>
	class VertexBufferLock : public BufferLock<T>
	{
	public:
		VertexBufferLock(IMesh& mesh, T*& data, unsigned int streamIndex = 0)
		{
 			V(mesh.GetVertexBuffer(streamIndex, &mBuffer));
			lock(*mBuffer, data);
		}
	};

	template <typename T>
	class IndexBufferLock : public BufferLock<T>
	{
	public:
		IndexBufferLock(IMesh& mesh, T*& data)
		{
			V(mesh.GetIndexBuffer(&mBuffer));
			lock(*mBuffer, data);
		}
	};
};


com_ptr<IMesh> generateBranchGeometry(IDirect3DDevice& device,
	float radius0, float radius1, float length,
	unsigned* branchIndices, float* branchOffsets, float* branchBendExps, size_t branchCount)
{
	const size_t Slices = 5;
	const size_t Stacks = 8;

	HRESULT hr;
	com_ptr<IMesh> cylinderMesh;
	V(DXUTCreateCylinder(&device, radius0, radius1, length, Slices, Stacks, &cylinderMesh));

    const D3D10_INPUT_ELEMENT_DESC declaration[] =
    {
        { "POSITION",     0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,    D3D10_INPUT_PER_VERTEX_DATA, 0 },
//        { "NORMAL",       0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 3*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "BLENDWEIGHT",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 3*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT,      0, 7*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",       0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 8*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };

	com_ptr<IMesh> outputMesh;
	V(D3DX10CreateMesh(
		&device,
		declaration, 4, declaration[0].SemanticName, 
		cylinderMesh->GetVertexCount(), cylinderMesh->GetFaceCount(),
		0,
		&outputMesh));

	DWORD branchIndicesPacked = 0;
	assert(branchCount <= 4);

	for(size_t q = 0; q < branchCount; ++q)
	{
		assert(branchIndices);
		assert(branchIndices[q] < 256);
		branchIndicesPacked <<= 8;
		branchIndicesPacked |= (branchIndices[q] & 0xff);
	}

	typedef UTVertex SrcVertexT;
	typedef BranchVertex DstVertexT;

	{
		SrcVertexT* srcVertices = 0;
		VertexBufferLock<SrcVertexT> vbSrcLock(*cylinderMesh, srcVertices);

		DstVertexT* dstVertices = 0;
		VertexBufferLock<DstVertexT> vbDstLock(*outputMesh, dstVertices);

		for(size_t q = 0; q < outputMesh->GetVertexCount(); ++q)
		{
			dstVertices->pos = srcVertices->pos + D3DXVECTOR3(0.0f, 0.0f, length * 0.5f);
			dstVertices->normal = srcVertices->normal;
			dstVertices->branchIndices = branchIndicesPacked;

			for(size_t w = 0; w < 4; ++w)
			{
				if(w < 3)
				{
					if(branchCount > w + 1)
						dstVertices->bendWeight[w] = pow(branchOffsets[w + 1], branchBendExps[w]);
					else
						dstVertices->bendWeight[w] = pow(dstVertices->pos.z / length, branchBendExps[w]);
				}
				else
					dstVertices->bendWeight[w] = 1.0f;
			}
			++dstVertices;
			++srcVertices;
		}
	}

	{
		void* indices = 0;
		IndexBufferLock<void> ibSrcLock(*cylinderMesh, indices);
		V(outputMesh->SetIndexData(indices, cylinderMesh->GetFaceCount() * 3));
	}

	V(outputMesh->CommitToDevice());
	return outputMesh;
}

com_ptr<IMesh> concatenateMeshes(IDirect3DDevice& device, IMesh * const* meshes, D3DXMATRIX const* transforms, size_t meshCount)
{
	assert(meshes);
	assert(transforms);

	HRESULT hr;
	unsigned int faceCount = 0;
	unsigned int vertexCount = 0;
	for(size_t meshIt = 0; meshIt < meshCount; ++meshIt)
	{
		assert(meshes[meshIt]);
		faceCount += meshes[meshIt]->GetFaceCount();
		vertexCount += meshes[meshIt]->GetVertexCount();
	}

	D3D10_INPUT_ELEMENT_DESC const* declaration = 0; 
	UINT declCount = 0; 
	V(meshes[0]->GetVertexDescription(&declaration, &declCount));
	assert(declaration);

	com_ptr<IMesh> outputMesh;
	V(D3DX10CreateMesh(
		&device, 
		declaration,
		declCount,
		declaration[0].SemanticName,
		vertexCount,
		faceCount,
		D3DX10_MESH_32_BIT,
		&outputMesh));
	
	typedef BranchVertex/*UTVertex*/ SrcVertexT;
	typedef BranchVertex/*UTVertex*/ DstVertexT;
	typedef WORD SrcIndexT;
	typedef DWORD DstIndexT;

	{
		DstVertexT* dstVertices = 0;
		VertexBufferLock<DstVertexT> dstVbLock(*outputMesh, dstVertices);

		DstIndexT* dstIndices = 0;
		IndexBufferLock<DstIndexT> dstIbLock(*outputMesh, dstIndices);

		DstIndexT indexOffset = 0;
		for(size_t meshIt = 0; meshIt < meshCount; ++meshIt)
		{
			assert(meshes[meshIt]);
			unsigned int srcVertexCount = meshes[meshIt]->GetVertexCount();
			unsigned int srcIndexCount = meshes[meshIt]->GetFaceCount() * 3;

			{ // copy and transform vertices
				SrcVertexT* srcVertices = 0;
				VertexBufferLock<SrcVertexT> srcLock(*meshes[meshIt], srcVertices);

				for(size_t vertexIt = 0; vertexIt < srcVertexCount; ++vertexIt, ++dstVertices)
				{
					*dstVertices = srcVertices[vertexIt];
					D3DXVec3TransformCoord(&dstVertices->pos, &dstVertices->pos, &transforms[meshIt]);
					D3DXVec3TransformNormal(&dstVertices->normal, &dstVertices->normal, &transforms[meshIt]);
				}
			}

			{ // concat indices
				SrcIndexT* srcIndices = 0;
				IndexBufferLock<SrcIndexT> srcLock(*meshes[meshIt], srcIndices);

				std::transform(srcIndices, srcIndices + srcIndexCount, dstIndices,
					std::bind2nd(std::plus<DstIndexT>(), indexOffset));

				dstIndices += srcIndexCount;
			}

			indexOffset += srcVertexCount;
		}
	}

	V(outputMesh->CommitToDevice());
	return outputMesh;
}

void calcBoundingSphere(RenderableTree const& tree, D3DXVECTOR3& center, float& radius)
{
	if(tree.mesh)
	{
		HRESULT hr;
		D3DXVECTOR3* data = 0;
		VertexBufferLock<D3DXVECTOR3> vbLock(*tree.mesh, data);

		DWORD stride = static_cast<DWORD>(vbLock.bufferSize()) / tree.mesh->GetVertexCount();
		V(D3DXComputeBoundingSphere(data, tree.mesh->GetVertexCount(), stride, &center, &radius));
	}
}