//----------------------------------------------------------------------------------
// File:   AnimatedCharacter.cpp
// Author: Bryan Dudash
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

#include "DXUT.h"
#include "AnimatedCharacter.h"
#include "CharacterAnimation.h"
#include "MultiAnimation.h"

AnimatedCharacter::AnimatedCharacter()
{
     animationsTexture = NULL;
     numAttachments = 0;
     fWalkSpeed = 6.f;
}
AnimatedCharacter::~AnimatedCharacter()
{
    for(int i=0;i<(int)m_animations.size();i++)
    {
        CharacterAnimation *anim = m_animations[i];
        delete anim;
    }
    m_animations.clear();
    m_characterMeshes.clear();
}

void AnimatedCharacter::CreateDefaultBuffer(ID3D10Device* pd3dDevice,void *data, UINT size,UINT bindFlags, ID3D10Buffer **ppBuffer)
{
    HRESULT hr;

    // Vertex Buffer
    D3D10_BUFFER_DESC bufferDesc;
    ZeroMemory(&bufferDesc,sizeof(D3D10_BUFFER_DESC));
    bufferDesc.ByteWidth = size;
    bufferDesc.Usage = D3D10_USAGE_DEFAULT;
    bufferDesc.BindFlags = bindFlags;

    D3D10_SUBRESOURCE_DATA subresourceData;
    subresourceData.pSysMem = (void*)data;
    V( pd3dDevice->CreateBuffer( &bufferDesc, &subresourceData, ppBuffer ) );
}

void AnimatedCharacter::CreateEmptyDefaultBuffer(ID3D10Device* pd3dDevice, UINT size,UINT bindFlags, ID3D10Buffer **ppBuffer)
{
    HRESULT hr;

    // Vertex Buffer
    D3D10_BUFFER_DESC bufferDesc;
    ZeroMemory(&bufferDesc,sizeof(D3D10_BUFFER_DESC));
    bufferDesc.ByteWidth = size;
    bufferDesc.Usage = D3D10_USAGE_DEFAULT;
    bufferDesc.BindFlags = bindFlags;

    V( pd3dDevice->CreateBuffer( &bufferDesc, NULL, ppBuffer ) );
}


HRESULT AnimatedCharacter::Initialize(ID3D10Device* pd3dDevice)
{
    fWalkSpeed = 6.f;
    return S_OK;
}

void AnimatedCharacter::addSingleDrawMesh(ID3D10Device* pd3dDevice,Vertex *vertices,UINT vtxCount, DWORD *indices, UINT idxCount,int meshSet,UINT nameHash, int texture)
{
    // First look to see if there isn a mesh with the same set in there already...
    SubmeshList::iterator itExistantMesh;
    for(itExistantMesh=m_characterMeshes.begin();itExistantMesh != m_characterMeshes.end();itExistantMesh++)
    {
        if((*itExistantMesh).meshSet == meshSet)
            break;
    }

    ID3D10Buffer *pIndexBuffer = NULL;
    ID3D10Buffer *pVertexBuffer = NULL;

    if(itExistantMesh == m_characterMeshes.end())
    {
        CreateDefaultBuffer(pd3dDevice,(void*)vertices,vtxCount*sizeof(Vertex),D3D10_BIND_VERTEX_BUFFER,&pVertexBuffer);

        if(idxCount > 0)
            CreateDefaultBuffer(pd3dDevice,(void*)indices,idxCount*sizeof(DWORD),D3D10_BIND_INDEX_BUFFER,&pIndexBuffer);

    }
    else
    {
        SubMesh subMesh = *itExistantMesh;

        // Make aggregate sized buffers
        CreateEmptyDefaultBuffer(pd3dDevice,(subMesh.numVertices + vtxCount)*sizeof(Vertex),D3D10_BIND_VERTEX_BUFFER,&pVertexBuffer);

        if(idxCount > 0)
            CreateEmptyDefaultBuffer(pd3dDevice,(subMesh.numIndices + idxCount)*sizeof(DWORD),D3D10_BIND_INDEX_BUFFER,&pIndexBuffer);

        // Copy in vertex data

        // first copy in original meshSet
        pd3dDevice->CopySubresourceRegion(pVertexBuffer,D3D10CalcSubresource(0,0,1),0,0,0,subMesh.pVB,D3D10CalcSubresource(0,0,1),NULL);
        
        // Now update with new data.
        D3D10_BUFFER_DESC subMeshDesc;
        subMesh.pVB->GetDesc(&subMeshDesc);

        D3D10_BOX updateBox;
        ZeroMemory(&updateBox,sizeof(D3D10_BOX));
        updateBox.left = subMeshDesc.ByteWidth;
        updateBox.right = updateBox.left + vtxCount*sizeof(Vertex);
        updateBox.top = 0; updateBox.bottom = 1;
        updateBox.front = 0; updateBox.back = 1;

        pd3dDevice->UpdateSubresource(pVertexBuffer,D3D10CalcSubresource(0,0,1),&updateBox,(const void *)vertices,sizeof(Vertex),sizeof(Vertex));
     
        // Now do index buffer, offsetting by number of vertices in original vertex buffer, but must work on a local copy to avoid changing original data

        // Now copy like vb
        // first copy in origianl meshSet
        pd3dDevice->CopySubresourceRegion(pIndexBuffer,D3D10CalcSubresource(0,0,1),0,0,0,subMesh.pIB,D3D10CalcSubresource(0,0,1),NULL);

        // Then new data
        // Now update with new data.
        subMesh.pIB->GetDesc(&subMeshDesc);

        ZeroMemory(&updateBox,sizeof(D3D10_BOX));
        updateBox.left = subMeshDesc.ByteWidth;
        updateBox.right = updateBox.left + idxCount*sizeof(DWORD);
        updateBox.top = 0; updateBox.bottom = 1;
        updateBox.front = 0; updateBox.back = 1;

        DWORD *localIndices = new DWORD[idxCount];
        memcpy((void*)localIndices,(void*)indices,idxCount*sizeof(DWORD));
        for(UINT iIndex=0;iIndex < idxCount;iIndex++)
        {
            localIndices[iIndex] += subMesh.numVertices;
        }

        pd3dDevice->UpdateSubresource(pIndexBuffer,D3D10CalcSubresource(0,0,1),&updateBox,(const void *)localIndices,sizeof(DWORD),sizeof(DWORD));

        delete [] localIndices;

        // update counts
        vtxCount += subMesh.numVertices;
        idxCount += subMesh.numIndices;

        // remove old entry
        SAFE_RELEASE(subMesh.pIB);
        SAFE_RELEASE(subMesh.pVB);
        m_characterMeshes.erase(itExistantMesh);
    }

    // Add in the buffers we made
    addSingleDrawMesh(pVertexBuffer,vtxCount,pIndexBuffer,idxCount,meshSet,nameHash,texture);

}

void AnimatedCharacter::addSingleDrawMesh(ID3D10Buffer*pVB,int numVerts,ID3D10Buffer*pIB,int numIndices,int meshSet,UINT nameHash, int texture)
{
    SubMesh subMesh;
    subMesh.pVB = pVB;
    subMesh.pIB = pIB;
    subMesh.numIndices = numIndices;
    subMesh.numVertices = numVerts;
    subMesh.meshSet = meshSet;
    subMesh.nameHash = nameHash;
    subMesh.texture = texture;

    SubmeshList::iterator it=m_characterMeshes.begin();
    for(;it!= m_characterMeshes.end();it++)
    {
        if(subMesh.nameHash < it->nameHash)
        {
            break;
        }
    }
    m_characterMeshes.insert(it,subMesh);
}

void AnimatedCharacter::Release()
{
    SAFE_RELEASE(animationsTexture);
    while(m_characterMeshes.size())
    {
        SubMesh subMesh = m_characterMeshes.back();
        SAFE_RELEASE(subMesh.pIB);
        SAFE_RELEASE(subMesh.pVB);
        m_characterMeshes.pop_back();
    }

    while(m_animations.size())
    {
        delete m_animations.back();
        m_animations.pop_back();
    }
}
