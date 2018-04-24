//----------------------------------------------------------------------------------
// File:   ArmyManager.h
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
class AnimatedCharacterInstance;
class CBaseCamera;

#define INTERNAL_MAX_INSTANCES_PER_BUFFER 682
#define INTERNAL_MAX_BUFFERS 15

/*
    The Army Manager wraps up the full effect.  It handles loading the animation data, creation of all DX10 resources, and rendering.
    Important bits:
    - AnimatedCharacter contains the mesh data, broken into seperate meshes to allow instancing of each piece.
    - Characters are rendered with instancing, and the custom data per instance is stored in a buiffer in each instance draw call struct
    - All animation frames are stored in a texture resource, m_pAnimationTexture
    - Color maps for all polys are set for each draw, 
*/
class ArmyManager
{
public:
    static ArmyManager *singleton()
    {
        if(m_Singleton==NULL) m_Singleton = new ArmyManager();
        return m_Singleton;
    }

    static void destroy()
    {
        delete m_Singleton;
    }

    struct InstanceDataElement
    {
        D3DXVECTOR4 world1;            // the world transform for this matrix row 1
        D3DXVECTOR4 world2;            // the world transform for this matrix row 2
        D3DXVECTOR4 world3;            // the world transform for this matrix row 3 (row 4 is implicit)
        D3DXCOLOR color;

        // $ Technically this is bundled, but there is not class that makes a uint vector, so just keep flat
        UINT animationIndex;            // offset in vectors into the whole data stream for the start of the animation playing
        UINT frameOffset;            // offset in vectors into the animation stream for the start of the frame playing
        UINT attachmentSet;            // the id to determine which geo attachments get set
        UINT lerpValue;            // lerp between frames
    };

    struct InstancedDrawCallState
    {
        InstancedDrawCallState() 
        {
            for(int i=0;i<INTERNAL_MAX_BUFFERS;i++)
                pInstanceDataBuffer[i] = NULL;numToDraw=0;
        }
        ID3D10Buffer *pInstanceDataBuffer[INTERNAL_MAX_BUFFERS];
        std::vector<int> instancesUsingThisMesh;
        int numToDraw;  // number to draw form this batch
        AnimatedCharacter *referenceCharacter;
        int meshIndex;
    };

    ~ArmyManager();
    HRESULT Initialize(ID3D10Device* pd3dDevice, int maxNumCharacters);
    HRESULT Update(float deltaTime,const D3DXVECTOR3 &cameraAt);
    HRESULT Render(ID3D10Device* pd3dDevice,D3DXMATRIX &mView,D3DXMATRIX &mProj, float time);
    void Release();
    HRESULT UpdateInstancedDrawState(ID3D10Device *pd3dDevice);
    void FillDataElement(InstanceDataElement*pData,int character);

    HRESULT CreateInstanceBuffer( ID3D10Device* pd3dDevice, ID3D10Buffer **pBuffer, int maxCharacters );
    HRESULT CreateAnimationTexture(ID3D10Device* pd3dDevice);
    HRESULT CreateAlbedoTextureArray(ID3D10Device* pd3dDevice);

    UINT GetAnimationOffset(int animation);
    void SetNumCharacters(int num);
    int GetMaxCharacters() {return (INTERNAL_MAX_BUFFERS * INTERNAL_MAX_INSTANCES_PER_BUFFER)-1;}

    HRESULT Draw(ID3D10Device *pd3dDevice,D3DXMATRIX &mView,D3DXMATRIX &mProj, float time);
    HRESULT DrawHybridInstanced(ID3D10Device *pd3dDevice,D3DXMATRIX &mView,D3DXMATRIX &mProj, float time);

    // Game logic
    D3DXVECTOR3 GetRandomScenePosition();
    int GetPolysDrawn() {return m_polysDrawnLastFrame;}
    int GetDrawCalls() {return m_drawCallsLastFrame;}

    // D3D10 Rendering vars (Release needed)
    ID3D10Device*               m_pd3dDevice;
    ID3D10InputLayout*          m_pInputLayout;
    ID3D10InputLayout*          m_pInputLayoutTerrain;
    ID3D10Effect*               m_pEffect10;

    ID3D10Texture2D*            m_pAnimationTexture;
    ID3D10ShaderResourceView*   m_pAnimationResourceView;

    ID3D10Texture2D *           m_pAlbedoTextureArray;
    ID3D10ShaderResourceView *  m_pAlbedoTextureArraySRV;

    CDXUTMesh10                 m_GroundMesh;
    AnimatedCharacter *         m_ReferenceCharacter;   // no release needed since it is also in m_CharacterLODs[]
    AnimatedCharacter *         m_CharacterLODs[4];
    
    // No need to Release these vars
    ID3D10EffectTechnique*  m_pTechAnimationInstanced;
    ID3D10EffectTechnique*  m_pTechAnimation;
    ID3D10EffectTechnique*  m_pTechTerrain;

    ID3D10EffectMatrixVariable*             m_pWorldVar;
    ID3D10EffectMatrixVariable*             m_pWorldViewProjectionVar;
    ID3D10EffectShaderResourceVariable*     m_pAnimationsVar;
    ID3D10EffectShaderResourceVariable*     m_pDiffuseTexArrayVar;
    ID3D10EffectShaderResourceVariable*     m_pDiffuseTexVar;

    // Character data vars
    bool m_bVisualizeLOD;
    int m_iInstancingMode;
    int m_MaxCharacters;
    int m_NumToDraw;
    std::vector<AnimatedCharacterInstance*> m_Characters;
    std::vector<InstancedDrawCallState> m_InstancedDraws;

    int m_polysDrawnLastFrame;
    int m_drawCallsLastFrame;

protected:

    InstanceDataElement* m_pDataTemp;

private:
    ArmyManager();
    static ArmyManager*m_Singleton;

};