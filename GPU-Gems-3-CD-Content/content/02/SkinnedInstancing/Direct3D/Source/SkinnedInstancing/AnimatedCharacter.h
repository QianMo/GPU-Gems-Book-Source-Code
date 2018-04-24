//----------------------------------------------------------------------------------
// File:   AnimatedCharacter.h
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

#pragma once 

#include <vector>

class CharacterAnimation;

/*
    This just wraps up DX10 vars for a single character mesh.  It also holds a list of animations that this character can run through.

    This class is filled in by the animation loader, so there is a dependency on this class.
*/
class AnimatedCharacter
{
public:

    struct Vertex
    {
        D3DXVECTOR3 position;
        D3DXVECTOR3 normal;
        D3DXVECTOR2 uv;
        D3DXVECTOR3 tangent;
        D3DXVECTOR4 bones;
        D3DXVECTOR4 weights;
    };

    // This represents a uniquie sub-section of the character
    struct SubMesh
    {
        ID3D10Buffer*pVB;
        ID3D10Buffer*pIB;
        int numVertices,numIndices;
        int meshSet;
        UINT nameHash;
        int texture;
    };

    typedef std::vector<CharacterAnimation *> AnimationList;
    typedef std::vector<SubMesh> SubmeshList;

    AnimatedCharacter();
    ~AnimatedCharacter();

    CharacterAnimation *GetAnimation(int index){return index < (int)m_animations.size()?m_animations[index]:NULL;};
    HRESULT Initialize(ID3D10Device* pd3dDevice);
    void Release();

    void addSingleDrawMesh(ID3D10Device* pd3dDevice,Vertex *vertices,UINT vtxCount, DWORD *indices, UINT idxCount,int meshSet,UINT nameHash, int texture);

protected:

    void addSingleDrawMesh(ID3D10Buffer*pVB,int numVerts,ID3D10Buffer*pIB,int numIndices,int meshSet,UINT namehash,int texture);
    void CreateDefaultBuffer(ID3D10Device* pd3dDevice,void *data, UINT size,UINT bindFlags, ID3D10Buffer **ppBuffer);
    void CreateEmptyDefaultBuffer(ID3D10Device* pd3dDevice, UINT size,UINT bindFlags, ID3D10Buffer **ppBuffer);

public:

    // DX10 vars
    ID3D10Texture2D *animationsTexture;
    SubmeshList m_characterMeshes;

    // our local mesh data
    AnimationList m_animations;

    std::string material;
    int numAttachments;
    int numBones;
    float boundingRadius;    // rough gauge of size

    // character logic consts
    float fWalkSpeed;

};