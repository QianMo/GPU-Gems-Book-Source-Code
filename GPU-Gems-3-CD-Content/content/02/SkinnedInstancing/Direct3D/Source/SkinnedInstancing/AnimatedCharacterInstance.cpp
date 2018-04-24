//----------------------------------------------------------------------------------
// File:   AnimatedCharacterInstance.cpp
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
#include "sdkmesh_old.h"
#include "AnimatedCharacterInstance.h"
#include "AnimatedCharacter.h"
#include "CharacterAnimation.h"
#include "ArmyManager.h"

AnimatedCharacterInstance::AnimatedCharacterInstance()
{
    iState = STANDING;
    targetPosition = D3DXVECTOR3(0,0,0);
    m_ReferenceCharacter = NULL;
    color = D3DXVECTOR3(1,1,1);
    lod1 = 1.f;
    lod2 = 2.f;
    lod3 = 3.f;
}

AnimatedCharacterInstance::~AnimatedCharacterInstance()
{

}

void AnimatedCharacterInstance::Initialize(D3DXVECTOR3 position,D3DXVECTOR3 rotation,D3DXVECTOR3 scale)
{
    assert(m_ReferenceCharacter);
    mPosition = position;
    mRotation = rotation;
    mScale = scale;
    animation = iState = STANDING;
    this->animationTime = m_ReferenceCharacter->m_animations[animation]->duration * ((float)rand()/32768.0f);
}

void AnimatedCharacterInstance::RandomMeshAndAnimation(AnimatedCharacter *m_ReferenceCharacter)
{
    this->m_ReferenceCharacter = m_ReferenceCharacter;

    RandomMeshSet(m_ReferenceCharacter);
    RandomAnimation(m_ReferenceCharacter,true);
}

#define INTENSITY 0.9f
D3DXVECTOR3 colors[] = 
{
    D3DXVECTOR3(INTENSITY,0,0),
    D3DXVECTOR3(0,INTENSITY,0),
    D3DXVECTOR3(0,0,INTENSITY),
    D3DXVECTOR3(INTENSITY,0,INTENSITY),
    D3DXVECTOR3(0,INTENSITY,INTENSITY),
    D3DXVECTOR3(INTENSITY,INTENSITY,0),
    D3DXVECTOR3(INTENSITY,INTENSITY,INTENSITY),
    D3DXVECTOR3(INTENSITY/3.f,INTENSITY/3.f,INTENSITY/3.f)
};

void AnimatedCharacterInstance::RandomMeshSet(AnimatedCharacter *m_ReferenceCharacter)
{
    this->m_ReferenceCharacter = m_ReferenceCharacter;

    int chestMasks[] = { 1<<0,1<<3, };
    int headMasks[] = { 1<<1,1<<2 };

    int numChests = sizeof(chestMasks)/sizeof(int);
    int numHeads = sizeof(headMasks)/sizeof(int);

    int iChest = rand()*numChests / 32769;    // dividing by one more than max to ensure no array out of bounds
    int iHead = rand()*numHeads / 32769;    // dividing by one more than max to ensure no array out of bounds

    meshSet = chestMasks[iChest] | headMasks[iHead];

    float randomvalue = rand() / 32769.f;
    float index = randomvalue * (sizeof(colors)/sizeof(D3DXVECTOR3));

    color = colors[(int)index];
    baseColor = color;
}

void AnimatedCharacterInstance::RandomScale()
{
    mScale.x = 0.8f + 0.4f*(rand() / 32769.f);
    mScale.y = 0.8f + 0.4f*(rand() / 32769.f);
    mScale.z = 0.65f + 0.7f*(rand() / 32769.f);
}

void AnimatedCharacterInstance::RandomAnimation(AnimatedCharacter *m_ReferenceCharacter,bool bRandomStart)
{
    this->m_ReferenceCharacter = m_ReferenceCharacter;

    if(m_ReferenceCharacter->m_animations.size() > 0)
    {
        float randAnim = 0.99f * ((float)rand()/32768.0f);
        animation = (int)(randAnim * (float)m_ReferenceCharacter->m_animations.size());

        if(bRandomStart)
            animationTime = m_ReferenceCharacter->m_animations[animation]->duration * ((float)rand()/32768.0f);
        else
            animationTime = 0.f;
    }
    else
    {
        animation = 0;
        animationTime = 0.f;
    }
}

void AnimatedCharacterInstance::UpdateWorldMatrix()
{
    D3DXMATRIX translation;
    D3DXMATRIX rotation;
    D3DXMATRIX scale;

    D3DXMatrixTranslation(&translation,mPosition.x,mPosition.y,mPosition.z);
    D3DXMatrixRotationYawPitchRoll(&rotation,mRotation.x,mRotation.y,mRotation.z);
    D3DXMatrixScaling(&scale,mScale.x,mScale.y,mScale.z);
    mWorld = scale * rotation * translation;
}

/*
    UpdateLOD chooses the LOD level to draw this instance at.  It also optionally selects an LOd color based on UI selections.
*/
void AnimatedCharacterInstance::UpdateLOD(const D3DXVECTOR3 &cameraAt)
{
    D3DXVECTOR3 current = D3DXVECTOR3(mWorld._41,mWorld._42,mWorld._43);
    current = current - cameraAt;
    float distSq = D3DXVec3LengthSq(&current);
    if(distSq < lod1*lod1)
    {
        LOD = 0;
    }
    else if(distSq < lod2*lod2)
    {
        LOD = 1;
    }
    else if(distSq < lod3*lod3)
    {
        LOD = 2;
    }
    else
    {
        LOD = 3;
    }

    // $ For debug to show where the LOD line is.
    if(ArmyManager::singleton()->m_bVisualizeLOD)
        color = colors[LOD];
    else
        color = baseColor;
}

float AnimatedCharacterInstance::GetRandomYaw()
{
    return 360.f * ((float)rand()/32768.0f);
}

/*
    this is a hardcoded list of animations that we expect to be in the mesh file.  
    
    If it can't find the right animation for the state it is in, returns the first animation in the list.
*/
int AnimatedCharacterInstance::GetAnimationForState(AIState &state)
{
    static const char *animationNames[4] = 
    {
        "Stopped",
        "FixingEnter",
        "FixingLoop",
        "FixingExit"
    };

    for(int i=0;i<(int)m_ReferenceCharacter->m_animations.size();i++)
    {
        if(strcmp(animationNames[(int)state],m_ReferenceCharacter->m_animations[i]->name.c_str()) == 0)
        {
            return i;
        }
    }
    return 0;
}

/*
    this is a silly simple state machine for a robot who decides to go a'fixin something and then wait.
*/
void AnimatedCharacterInstance::Update(float deltatime, const D3DXVECTOR3 &cameraAt)
{
    animationTime += deltatime;

    if(!m_ReferenceCharacter) return;    // only do AI if we have a base character to refer to

    // Silly stupid AI state machine.  When done playing an anim, 50% chance to change state
    D3DXVECTOR3 currentPos = mPosition;
    switch(iState)
    {
    case STANDING:
        {
            if(animationTime > m_ReferenceCharacter->m_animations[animation]->duration)
            {
                if(rand() > 16384)
                    iState = ENTER_FIX;
                        
                animationTime = 0.f;
            }
        }
        break;
    case ENTER_FIX:
        {
            if(animationTime > m_ReferenceCharacter->m_animations[animation]->duration)
            {
                iState = FIXING;
                animationTime = 0.f;
            }
        }
        break;
    case EXIT_FIX:
        {
            if(animationTime > m_ReferenceCharacter->m_animations[animation]->duration)
            {
                iState = STANDING;
                animationTime = 0.f;
            }
        }
        break;
    case FIXING:
        {
            if(animationTime > m_ReferenceCharacter->m_animations[animation]->duration)
            {
                if(rand() > 16384)
                    iState = EXIT_FIX;
                animationTime = 0.f;
            }
        }
        break;
    }

    animation = GetAnimationForState(iState);
    
    UpdateWorldMatrix();
    UpdateLOD(cameraAt);
}
