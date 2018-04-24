//----------------------------------------------------------------------------------
// File:   AnimatedCharacterInstance.h
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

class AnimatedCharacter;

/*
    A placed and animating character instance.  Simply wraps up the position, animation, and mesh set for this instance.
*/
class AnimatedCharacterInstance
{
public:
    AnimatedCharacterInstance();
    ~AnimatedCharacterInstance();

    void Initialize(D3DXVECTOR3 position,D3DXVECTOR3 rotation,D3DXVECTOR3 scale=D3DXVECTOR3(1,1,1));
    void SetLODParams(float lod1, float lod2, float lod3){this->lod1 = lod1;this->lod2 = lod2;this->lod3 = lod3;};
    
    void RandomMeshAndAnimation(AnimatedCharacter *m_ReferenceCharacter);
    void RandomAnimation(AnimatedCharacter *m_ReferenceCharacter,bool bRandomStart=true);
    void RandomMeshSet(AnimatedCharacter *m_ReferenceCharacter);
    void RandomScale();

    float GetRandomYaw();
    
    void UpdateWorldMatrix();
    void UpdateLOD(const D3DXVECTOR3 &cameraAt);
    void Update(float deltatime, const D3DXVECTOR3 &cameraAt);

    AnimatedCharacter *m_ReferenceCharacter;
    int animation;
    float animationTime;
    int meshSet;
    int LOD;    // 0 is full, goes down from there
    D3DXVECTOR3 color;
    D3DXVECTOR3 baseColor;
    D3DXMATRIX mWorld;

protected:

    D3DXVECTOR3 mPosition;
    D3DXVECTOR3 mRotation;
    D3DXVECTOR3 mScale;

    float lod1;
    float lod2;
    float lod3;

    // Character logic
    enum AIState
    {
        STANDING = 0,
        ENTER_FIX = 1,
        FIXING = 2,
        EXIT_FIX= 3,
    };

    int GetAnimationForState(AIState &state);

    static char **animationNames;

    AIState        iState;
    D3DXVECTOR3 targetPosition;
};

