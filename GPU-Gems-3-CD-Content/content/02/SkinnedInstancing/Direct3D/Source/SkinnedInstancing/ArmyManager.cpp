//----------------------------------------------------------------------------------
// File:   ArmyManager.cpp
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
#include "SDKmisc.h"
#include "DXUTcamera.h"
#include "sdkmesh_old.h"

#include <vector>

#include "ArmyManager.h"
#include "TextureLibray.h"
#include "CharacterAnimation.h"
#include "AnimatedCharacterInstance.h"
#include "DX9AnimationLoader.h"

ArmyManager*ArmyManager::m_Singleton = NULL;

ArmyManager::ArmyManager()
{
    m_pEffect10 = NULL;
    m_pTechAnimation = NULL;
    m_pWorldViewProjectionVar = NULL;
    m_pAnimationResourceView = NULL;
    m_pAnimationTexture = NULL;
    m_pAlbedoTextureArray = NULL;
    m_pAlbedoTextureArraySRV = NULL;
    m_MaxCharacters = 0;
    m_NumToDraw = 0;
    m_iInstancingMode = 1;
    m_pd3dDevice = NULL;
    m_ReferenceCharacter = NULL;
    m_pInputLayout = NULL;
    m_pInputLayoutTerrain = NULL;
    m_pDataTemp = new InstanceDataElement[INTERNAL_MAX_INSTANCES_PER_BUFFER];
    m_bVisualizeLOD = false;
}

ArmyManager::~ArmyManager()
{
    delete [] m_pDataTemp;
}

HRESULT ArmyManager::Initialize(ID3D10Device* pd3dDevice, int maxNumCharacters)
{
    HRESULT hr;

    m_pd3dDevice = pd3dDevice;

    const D3D10_INPUT_ELEMENT_DESC meshonlylayout[] =
    {
        // Normal character mesh data, setup for GPU skinning
        { "POSITION",    0, DXGI_FORMAT_R32G32B32_FLOAT,        0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",      0, DXGI_FORMAT_R32G32B32_FLOAT,        0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,           0, 24, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TANGENT",     0, DXGI_FORMAT_R32G32B32_FLOAT,        0, 32, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "BONES",       0, DXGI_FORMAT_R32G32B32A32_FLOAT,     0, 44, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "WEIGHTS",     0, DXGI_FORMAT_R32G32B32A32_FLOAT,     0, 60, D3D10_INPUT_PER_VERTEX_DATA, 0 },    
    };

    // First load character meshes
    DX9AnimationLoader loader;
    V_RETURN(loader.load(pd3dDevice,L"robot-lod0-animations.x.gz",meshonlylayout,sizeof(meshonlylayout)/sizeof(D3D10_INPUT_ELEMENT_DESC)));
    m_ReferenceCharacter = loader.pAnimatedCharacter;
    V(m_ReferenceCharacter->Initialize(pd3dDevice));
    m_CharacterLODs[0] = m_ReferenceCharacter;

    V_RETURN(loader.loadAndBindLODMesh(pd3dDevice,L"robot-lod1.x",meshonlylayout,sizeof(meshonlylayout)/sizeof(D3D10_INPUT_ELEMENT_DESC),false));
    V(loader.pAnimatedCharacter->Initialize(pd3dDevice));
    m_CharacterLODs[1] = loader.pAnimatedCharacter;

    V_RETURN(loader.loadAndBindLODMesh(pd3dDevice,L"robot-lod2.x",meshonlylayout,sizeof(meshonlylayout)/sizeof(D3D10_INPUT_ELEMENT_DESC),false));
    V(loader.pAnimatedCharacter->Initialize(pd3dDevice));
    m_CharacterLODs[2] = loader.pAnimatedCharacter;

    V_RETURN(loader.loadAndBindLODMesh(pd3dDevice,L"robot-lod2.x",meshonlylayout,sizeof(meshonlylayout)/sizeof(D3D10_INPUT_ELEMENT_DESC),false));
    V(loader.pAnimatedCharacter->Initialize(pd3dDevice));
    m_CharacterLODs[3] = loader.pAnimatedCharacter;
           
    // the matrix palette is only needed for non instanced
    LPCSTR sNumBones = new CHAR[MAX_PATH];
    sprintf_s((char*)sNumBones,MAX_PATH,"%d",m_ReferenceCharacter->numBones);
    LPCSTR sMaxInstanceConsts = new CHAR[MAX_PATH];
    sprintf_s((char*)sMaxInstanceConsts,MAX_PATH,"%d",INTERNAL_MAX_INSTANCES_PER_BUFFER);

    D3D10_SHADER_MACRO mac[3] =
    {
        { "MATRIX_PALETTE_SIZE_DEFAULT", sNumBones },
        { "MAX_INSTANCE_CONSTANTS", sMaxInstanceConsts},
        { NULL,                          NULL }
    };

    // Read the D3DX effect file
    WCHAR str[MAX_PATH];
    V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, L"SkinnedInstancing.fx" ) );
    V_RETURN( D3DX10CreateEffectFromFile( str, mac, NULL, D3D10_SHADER_ENABLE_STRICTNESS, 0, pd3dDevice, NULL, NULL, &m_pEffect10, NULL ) );

    delete [] sNumBones;
    delete [] sMaxInstanceConsts;

    // Grab some pointers to techniques
    m_pTechAnimationInstanced = m_pEffect10->GetTechniqueByName( "RenderSceneWithAnimationInstanced" );
    m_pTechAnimation = m_pEffect10->GetTechniqueByName( "RenderSceneWithAnimation" );
    m_pTechTerrain = m_pEffect10->GetTechniqueByName( "RenderTerrain" );

    // Grab some pointers to effects variables
    m_pWorldViewProjectionVar = m_pEffect10->GetVariableBySemantic( "WorldViewProjection" )->AsMatrix();
    m_pWorldVar = m_pEffect10->GetVariableBySemantic( "World" )->AsMatrix();
    m_pAnimationsVar = m_pEffect10->GetVariableBySemantic( "ANIMATIONS" )->AsShaderResource();
    m_pDiffuseTexArrayVar  = m_pEffect10->GetVariableBySemantic( "DIFFUSEARRAY" )->AsShaderResource();
    m_pDiffuseTexVar  = m_pEffect10->GetVariableBySemantic( "DIFFUSE" )->AsShaderResource();

    // Create our vertex input layouts
    D3D10_PASS_DESC PassDesc;
    V_RETURN( m_pTechAnimation->GetPassByIndex( 0 )->GetDesc( &PassDesc ) );
    V_RETURN( pd3dDevice->CreateInputLayout( meshonlylayout, sizeof(meshonlylayout)/sizeof(D3D10_INPUT_ELEMENT_DESC), PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_pInputLayout ) );

    // Initialize our numbers with the passed in value
    SetNumCharacters(maxNumCharacters);

    // Create a texture with all m_animations in it
    V(CreateAnimationTexture(pd3dDevice));

    // Make our texture array for texturing of the m_Characters
    CreateAlbedoTextureArray(pd3dDevice);

    // Make a simple ground plane for m_Characters to walk on.
    const D3D10_INPUT_ELEMENT_DESC groundLayout[] =
    {
        { "POSITION",    0, DXGI_FORMAT_R32G32B32_FLOAT,        0, 0,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",        0, DXGI_FORMAT_R32G32B32_FLOAT,        0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,        0, 24, D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };

    V_RETURN( m_pTechTerrain->GetPassByIndex( 0 )->GetDesc( &PassDesc ) );
    V_RETURN( pd3dDevice->CreateInputLayout( groundLayout, sizeof(groundLayout)/sizeof(D3D10_INPUT_ELEMENT_DESC), PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &m_pInputLayoutTerrain ) );
    m_GroundMesh.Create(pd3dDevice,L"groundBlock.x",(D3D10_INPUT_ELEMENT_DESC*)groundLayout,sizeof(groundLayout)/sizeof(D3D10_INPUT_ELEMENT_DESC),false);

    // Assign it to our effect
    if(m_pAlbedoTextureArraySRV)
        m_pDiffuseTexArrayVar->SetResource(m_pAlbedoTextureArraySRV);

    return S_OK;
}

/*
    This sets the # of characters drawn, and algorithmically places them around the world.
*/
void ArmyManager::SetNumCharacters(int num)
{
    // If we need to increase the max size, then Release and instance buffer we had and Update max count
    if(num > m_MaxCharacters)
    {
        while(m_Characters.size()>0)
        {
            AnimatedCharacterInstance *c = m_Characters.back();
            delete c;
            m_Characters.pop_back();
        }

        while(m_InstancedDraws.size() > 0)
        {
            for(int i=0;i<INTERNAL_MAX_BUFFERS;i++)
                SAFE_RELEASE(m_InstancedDraws.back().pInstanceDataBuffer[i]);
            m_InstancedDraws.back().instancesUsingThisMesh.clear();
            m_InstancedDraws.pop_back();
        }

        m_MaxCharacters = num;
    }

    // assign the # we want to Draw
    m_NumToDraw = num;

    //  We need to make our characters again.
    if(m_Characters.size() == 0)
    {
        D3DXMATRIX mScale;
        D3DXMATRIX mTranslation;
        D3DXMATRIX mRotation;
        D3DXMATRIX mWorld;
        AnimatedCharacterInstance *character = NULL;

        // a bunch of character instances. arranged areana style
        int charactersLeft = m_MaxCharacters - ((int)m_Characters.size());
//        float currentRadius = 5*m_ReferenceCharacter->boundingRadius;
        float rowHeight = m_ReferenceCharacter->boundingRadius/2.f;
        int currentRow = 0;
        int charactersPerRow = 20;
        int startCharacter = 0;
        for(int iCharacter=0;iCharacter<charactersLeft;iCharacter++)
        {
            int iLocalCharacter = iCharacter/4;
            // detect row change and increment to allow widening of rows as we go.
            if(iLocalCharacter > 0 && iLocalCharacter-startCharacter == charactersPerRow) 
            {
                currentRow++;
                charactersPerRow += 2;  // add one on each end per row, since a row is one character thick.
                startCharacter = iLocalCharacter;
            }

            D3DXMATRIX mTranslation;
            D3DXMatrixTranslation(&mTranslation,
                                m_ReferenceCharacter->boundingRadius*(float)(iLocalCharacter-startCharacter),
                                rowHeight*(float)currentRow,m_ReferenceCharacter->boundingRadius*(float)currentRow);

            D3DXVECTOR3 position = D3DXVECTOR3(-(charactersPerRow/2.f)*m_ReferenceCharacter->boundingRadius,
                                                0,
                                                10*m_ReferenceCharacter->boundingRadius);
            D3DXVECTOR3 rotation = D3DXVECTOR3(0,0,0);

            D3DXMATRIX standRotation;
            switch(iCharacter %4)
            {
            case 0:
                rotation.x = 0.f;
                break;
            case 1:
                rotation.x = 3.141592654f/2.f;
                break;
            case 2:
                rotation.x = 3.141592654f;
                break;
            case 3:
                rotation.x = -3.141592654f/2.f;
                break;
            }

            D3DXMatrixRotationY(&standRotation,rotation.x);

            D3DXVECTOR4 temp;
            D3DXMATRIX realTranslation = mTranslation * standRotation;
            D3DXVec3Transform(&temp,&position,&realTranslation);
            memcpy(&position,&temp,sizeof(D3DXVECTOR3));

            character = new AnimatedCharacterInstance();
            character->RandomMeshSet(m_ReferenceCharacter);
            character->Initialize(position,rotation);
            character->RandomScale();
            character->SetLODParams(m_ReferenceCharacter->boundingRadius*10.f,
                                    m_ReferenceCharacter->boundingRadius*30.f,
                                    m_ReferenceCharacter->boundingRadius*50.f);
            m_Characters.push_back(character);
        }
    }
}

/*
    This calls to each character instance to Update it's animation data
*/
HRESULT ArmyManager::Update(float deltaTime,const D3DXVECTOR3 &cameraAt)
{
    //m_pEffect10->GetVariableByName("g_lightPos")->AsVector()->SetFloatVector((float*)&cameraAt);

    for(int i=0;i<(int)m_Characters.size();i++)
    {
        m_Characters[i]->Update(deltaTime,cameraAt);
    }
    return S_OK;
}

/*
    Update the instance data buffer with new animation data, then Draw
*/
HRESULT ArmyManager::Render(ID3D10Device* pd3dDevice,D3DXMATRIX &mView,D3DXMATRIX &mProj, float time)
{
    HRESULT hr;

    m_polysDrawnLastFrame = 0;
    m_drawCallsLastFrame = 0;

    if(!m_Characters.size()) return S_OK;

    pd3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    switch(m_iInstancingMode)
    {
        /// Single Draws
    case 0:
        V(Draw(pd3dDevice,mView,mProj,time));
        break;
        // Hybrid instancing
    case 1:
    default:
        V(UpdateInstancedDrawState(pd3dDevice));
        V(DrawHybridInstanced(pd3dDevice,mView,mProj,time));
        break;
    }

    // Now Draw our ground plane, this is simple DX10 std rendering
    D3DXMATRIX mWorld;
    D3DXMatrixIdentity(&mWorld);
    D3DXMatrixScaling(&mWorld,10,1,10);
    D3DXMATRIX mWorldViewProj = mWorld * mView * mProj;

    m_pWorldVar->SetMatrix((float*)&mWorld);
    m_pWorldViewProjectionVar->AsMatrix()->SetMatrix((float*)&mWorldViewProj);

    pd3dDevice->IASetInputLayout(m_pInputLayoutTerrain);
    pd3dDevice->IASetIndexBuffer(NULL,DXGI_FORMAT_R32_UINT,0);
    m_GroundMesh.Render(pd3dDevice,m_pTechTerrain,m_pDiffuseTexVar);
    if(m_GroundMesh.m_pMesh10)
    {
        m_polysDrawnLastFrame += m_GroundMesh.m_pMesh10->GetFaceCount();
        m_drawCallsLastFrame += 1;
    }

    return S_OK;    
}

/*
    Non Instanced Draw.  This just iterates over the instances, and updates VS constants for each animation, then draws each character given it's world transform
*/
HRESULT ArmyManager::Draw(ID3D10Device *pd3dDevice,D3DXMATRIX &mView,D3DXMATRIX &mProj, float time)
{
    pd3dDevice->IASetInputLayout(m_pInputLayout);

    // iterate over the # of charactes to draw
    int realNumToDraw = min((int)m_Characters.size(),m_NumToDraw);
    for(int currentCharacter = 0;currentCharacter<realNumToDraw;currentCharacter++)
    {
        AnimatedCharacterInstance *pCharacter = m_Characters[currentCharacter];
        D3DXMATRIX mWorldViewProj = pCharacter->mWorld * mView * mProj;
        m_pWorldViewProjectionVar->AsMatrix()->SetMatrix((float*)&mWorldViewProj);
        m_pWorldVar->SetMatrix((float*)&pCharacter->mWorld);

        // Grab the current frame of animation and set it to constants
        ID3D10EffectTechnique *pRenderTechnique = m_pTechAnimation;
        CharacterAnimation *pAnimation = m_ReferenceCharacter->m_animations[pCharacter->animation];
        D3DXMATRIX *frameMatrices = pAnimation->GetEncodedFrameAt(pCharacter->animationTime);
        m_pEffect10->GetVariableByName("g_matrices")->AsMatrix()->SetMatrixArray(*frameMatrices,0,m_ReferenceCharacter->numBones);
        m_pEffect10->GetVariableByName("g_instanceColor")->AsVector()->SetFloatVector((float*)&pCharacter->color);

        AnimatedCharacter *referenceCharacter = m_CharacterLODs[m_Characters[currentCharacter]->LOD];

        // draw each submesh on each character, optionally skipping if we dont want to draw.
        int previousTexture = -1;
        for(int iSubMesh=0;iSubMesh<(int)referenceCharacter->m_characterMeshes.size();iSubMesh++)
        {
            // m_characterMeshes are the list of submeshes for the character.
            AnimatedCharacter::SubMesh &subMesh = referenceCharacter->m_characterMeshes[iSubMesh];

            // skip sub meshes if they don't match our submesh set. $$ Note this logic should match GS logic.
            if(subMesh.meshSet != 0 && !(subMesh.meshSet & pCharacter->meshSet)) continue;

            // Set our texture if we need to.
            if(previousTexture != subMesh.texture)
            {
                m_pDiffuseTexVar->SetResource(TextureLibrary::singleton()->GetShaderResourceView(subMesh.texture));
                previousTexture = subMesh.texture;
            }

            // Ensure no second buffer left over from instancing...
            UINT uiVertexStride = sizeof(AnimatedCharacter::Vertex);
            UINT uOffset = 0;

            ID3D10Buffer *vertBuffer = subMesh.pVB;
            ID3D10Buffer *indexBuffer = subMesh.pIB;

            pd3dDevice->IASetVertexBuffers(0,1,&vertBuffer, &uiVertexStride, &uOffset );
            if(indexBuffer)
                pd3dDevice->IASetIndexBuffer(indexBuffer,DXGI_FORMAT_R32_UINT,0);

            // Apply the technique contained in the effect
            D3D10_TECHNIQUE_DESC techDesc;
            memset( &techDesc, 0, sizeof( D3D10_TECHNIQUE_DESC ) );
            pRenderTechnique->GetDesc( &techDesc );
            for (UINT iPass = 0; iPass < techDesc.Passes; iPass++)
            {
                ID3D10EffectPass *pCurrentPass = pRenderTechnique->GetPassByIndex( iPass );
                pCurrentPass->Apply( 0 );

                if(indexBuffer)
                    pd3dDevice->DrawIndexed(subMesh.numIndices,0,0);
                else
                    pd3dDevice->Draw(subMesh.numVertices,0);

                m_polysDrawnLastFrame += subMesh.numIndices/3;
                m_drawCallsLastFrame += 1;
            }
        }
    }
    return S_OK;
}

/*
    Instanced Draw.  Binds the instance data buffer, and mesh buffer.  Also binds animation texture and color texture arrays
*/
HRESULT ArmyManager::DrawHybridInstanced(ID3D10Device *pd3dDevice,D3DXMATRIX &mView,D3DXMATRIX &mProj, float time)
{
    HRESULT hr;

    // Common to all isntanced draws
    D3DXMATRIX mWorldViewProj = mView * mProj;
    m_pWorldViewProjectionVar->AsMatrix()->SetMatrix((float*)&mWorldViewProj);

    if(m_pAnimationTexture)
    {
        V(m_pAnimationsVar->SetResource(m_pAnimationResourceView));
    
        D3D10_TEXTURE2D_DESC desc;
        m_pAnimationTexture->GetDesc(&desc);
        m_pEffect10->GetVariableByName("g_InstanceMatricesHeight")->AsScalar()->SetInt(desc.Height);
        m_pEffect10->GetVariableByName("g_InstanceMatricesWidth")->AsScalar()->SetInt(desc.Width);
    }

    pd3dDevice->IASetInputLayout(m_pInputLayout);

    // One instanced draw per submesh
    int previousTexture = -1;
    for(int iDrawState=0;iDrawState<(int)m_InstancedDraws.size();iDrawState++)
    {
        InstancedDrawCallState &drawState = m_InstancedDraws[iDrawState];
        if(drawState.numToDraw)
        {
            int iMesh = m_InstancedDraws[iDrawState].meshIndex; // straight mapping
            AnimatedCharacter::SubMesh &subMesh = m_InstancedDraws[iDrawState].referenceCharacter->m_characterMeshes[iMesh];

            // Set our texture if we need to.
            if(previousTexture != subMesh.texture)
            {
                m_pDiffuseTexVar->SetResource(TextureLibrary::singleton()->GetShaderResourceView(subMesh.texture));
                previousTexture = subMesh.texture;
            }


            ID3D10Buffer *vertBuffer = subMesh.pVB;
            ID3D10Buffer *indexBuffer = subMesh.pIB;

            UINT uiVertexStride = sizeof(AnimatedCharacter::Vertex);
            UINT uOffset = 0;

            pd3dDevice->IASetVertexBuffers(0,1,&vertBuffer, &uiVertexStride, &uOffset );
            if(indexBuffer)
                pd3dDevice->IASetIndexBuffer(indexBuffer,DXGI_FORMAT_R32_UINT,0);

            // Now loop over internal instance data constant buffers, since we are limited in the # we can fit in a single
            //   constant buffer
            unsigned int buffersUsed = (drawState.numToDraw / INTERNAL_MAX_INSTANCES_PER_BUFFER)+1;
            for(unsigned int buffer = 0;buffer < buffersUsed ; buffer++)
            {
                int numInstances = INTERNAL_MAX_INSTANCES_PER_BUFFER;
                if(buffer == buffersUsed - 1)       // restrict drawing all instances in the final buffer to get an exact count
                    numInstances = drawState.numToDraw % INTERNAL_MAX_INSTANCES_PER_BUFFER;

                ID3D10Buffer *pEffectConstantBuffer = NULL;
                m_pEffect10->GetConstantBufferByName("cInstanceData")->AsConstantBuffer()->GetConstantBuffer(&pEffectConstantBuffer);

                // Copy ourselves into the subregion
                D3D10_BUFFER_DESC srcDesc;
                D3D10_BUFFER_DESC dstDesc;
                drawState.pInstanceDataBuffer[buffer]->GetDesc(&srcDesc);
                pEffectConstantBuffer->GetDesc(&dstDesc);

                D3D10_BOX updateBox;
                ZeroMemory(&updateBox,sizeof(D3D10_BOX));
                updateBox.left = 0;
                updateBox.right = updateBox.left + numInstances*sizeof(InstanceDataElement);
                updateBox.top = 0;
                updateBox.bottom = 1;
                updateBox.front = 0;
                updateBox.back = 1;

                pd3dDevice->CopySubresourceRegion(pEffectConstantBuffer,0,0,0,0,drawState.pInstanceDataBuffer[buffer],0,&updateBox);
                SAFE_RELEASE(pEffectConstantBuffer);

                // Apply the technique contained in the effect, and then draw!
                D3D10_TECHNIQUE_DESC techDesc;
                memset( &techDesc, 0, sizeof( D3D10_TECHNIQUE_DESC ) );
                m_pTechAnimationInstanced->GetDesc( &techDesc );
                for (UINT iPass = 0; iPass < techDesc.Passes; iPass++)
                {
                    ID3D10EffectPass *pCurrentPass = m_pTechAnimationInstanced->GetPassByIndex( iPass );
                    pCurrentPass->Apply( 0 );
                    if(indexBuffer)
                        pd3dDevice->DrawIndexedInstanced(subMesh.numIndices,numInstances,0,0,0);
                    else
                        pd3dDevice->DrawInstanced(subMesh.numVertices,numInstances,0,0);

                    m_polysDrawnLastFrame += numInstances * subMesh.numIndices/3;
                    m_drawCallsLastFrame += 1;
                }
            }   // buffers loop
        } // id drawState.numToDraw
    }   // instance draw state loop

    return S_OK;
}



/*
    This creates the actual DX10 buffer object that will hold our per instance data.
*/
HRESULT ArmyManager::CreateInstanceBuffer( ID3D10Device* pd3dDevice, ID3D10Buffer **pBuffer, int maxCharacters )
{
    HRESULT hr = S_OK;

    // Create a resource with the input matrices
    D3D10_BUFFER_DESC bufferDesc =
    {
        maxCharacters * sizeof( InstanceDataElement ),
        D3D10_USAGE_DEFAULT,
        D3D10_BIND_VERTEX_BUFFER|D3D10_BIND_SHADER_RESOURCE,
        0,
        0
    };

    V(pd3dDevice->CreateBuffer( &bufferDesc, NULL, pBuffer ));

    return hr;
}

void ArmyManager::FillDataElement(InstanceDataElement*pData,int character)
{
    AnimatedCharacterInstance* pCharacter = m_Characters[character];
    assert(pCharacter);

    D3DXMATRIX mWorld = pCharacter->mWorld;

    // kind of an encoding.  put in the 3x3 rotation scale, plus the translation at the end
    //  allows my 4x4 matrix to fit into a 3x4
    pData->world1 = D3DXVECTOR4(mWorld._11,mWorld._12,mWorld._13,mWorld._41);
    pData->world2 = D3DXVECTOR4(mWorld._21,mWorld._22,mWorld._23,mWorld._42);
    pData->world3 = D3DXVECTOR4(mWorld._31,mWorld._32,mWorld._33,mWorld._43);

    // this is the real meat, the animation offset and the character mesh set.
    pData->animationIndex = GetAnimationOffset(pCharacter->animation);
    pData->attachmentSet = pCharacter->meshSet;

    pData->color.r = pCharacter->color.x;
    pData->color.g = pCharacter->color.y;
    pData->color.b = pCharacter->color.z;

    // this is offset in elements, so # bones * frame index
    CharacterAnimation *pAnimation = m_ReferenceCharacter->GetAnimation(pCharacter->animation);
    if(pAnimation)
    {
        UINT texelsPerBone = 4;
        pData->frameOffset = texelsPerBone*m_ReferenceCharacter->numBones * pAnimation->GetFrameIndexAt(pCharacter->animationTime);    
        pData->lerpValue = 0;   // $ lerp Value is unimplemented right now
    }
}


/*
    Updates constant buffers for each instanced draw state element.
*/
HRESULT ArmyManager::UpdateInstancedDrawState(ID3D10Device *pd3dDevice)
{
    // Some calcs about size and lod level locations...
    int size = 0;
    int lodLevels[4];
    for(int i=0;i<4;i++)
    {
        size += (int)m_CharacterLODs[i]->m_characterMeshes.size();
        lodLevels[i] = size;
    }

    // initially make the buffers, or remake
    if(m_InstancedDraws.size() == 0)
    {
        // resize to # of meshes
        m_InstancedDraws.resize(size);

        // Now make the instance data buffers based on usage
        for(int iDraw=0;iDraw<(int)m_InstancedDraws.size();iDraw++)
        {
            for(int i=0;i<4;i++)
            {
                if(iDraw < lodLevels[i])
                {
                    m_InstancedDraws[iDraw].referenceCharacter = m_CharacterLODs[i];
                    m_InstancedDraws[iDraw].meshIndex = i==0?iDraw:iDraw-lodLevels[i-1];
                    break;
                }
            }

            InstancedDrawCallState &drawState = m_InstancedDraws[iDraw];
            for(int i=0;i<INTERNAL_MAX_BUFFERS;i++)
                CreateInstanceBuffer(pd3dDevice,&drawState.pInstanceDataBuffer[i],(int)INTERNAL_MAX_INSTANCES_PER_BUFFER);
        }
    }

    // first clear
    for(int iDraw=0;iDraw<(int)m_InstancedDraws.size();iDraw++)
        m_InstancedDraws[iDraw].instancesUsingThisMesh.clear();

    // allocate instances into each draw call
    for(int iInstance=0;iInstance<min(m_NumToDraw,(int)m_Characters.size());iInstance++)
    {
        AnimatedCharacter *referenceCharacter = m_CharacterLODs[m_Characters[iInstance]->LOD];
        int lodOffset = m_Characters[iInstance]->LOD==0?0:lodLevels[m_Characters[iInstance]->LOD-1];

        for(int iMesh=0;iMesh<(int)referenceCharacter->m_characterMeshes.size();iMesh++)
        {
            if(referenceCharacter->m_characterMeshes[iMesh].meshSet == 0 || m_Characters[iInstance]->meshSet & referenceCharacter->m_characterMeshes[iMesh].meshSet)
            {
                m_InstancedDraws[lodOffset + iMesh].instancesUsingThisMesh.push_back(iInstance);
            }
        }
    }

    // for each single mesh, make the state for the instanced rendering...
    HRESULT hr = S_OK;
    for(int iDraw=0;iDraw<(int)m_InstancedDraws.size();iDraw++)
    {
        InstancedDrawCallState &drawState = m_InstancedDraws[iDraw];

        // Re calc # to draw by traversing backwards
        drawState.numToDraw = (int)drawState.instancesUsingThisMesh.size();
/*        for(int iInstance=(int)drawState.instancesUsingThisMesh.size()-1;iInstance >= 0;iInstance--)
        {
            // Stop when we reach an element we will be drawing... assuming this list is sorted by index
            if(drawState.instancesUsingThisMesh[iInstance] < m_NumToDraw)
            {
                drawState.numToDraw = iInstance+1;
                break;
            }
        }*/

        // update our array of constant buffers...
        unsigned int buffersUsed = (drawState.numToDraw / INTERNAL_MAX_INSTANCES_PER_BUFFER)+1;
        for(unsigned int buffer = 0;buffer < buffersUsed ; buffer++)
        {
            int numInstances = INTERNAL_MAX_INSTANCES_PER_BUFFER;
            if(buffer == buffersUsed - 1)       // restrict drawing all instances in the final buffer to get an exact count
                numInstances = drawState.numToDraw % INTERNAL_MAX_INSTANCES_PER_BUFFER;
        
            if(!drawState.pInstanceDataBuffer) return E_INVALIDARG;

            D3D10_BUFFER_DESC bDesc;
            drawState.pInstanceDataBuffer[buffer]->GetDesc(&bDesc);

            memset((void*)m_pDataTemp,0,bDesc.ByteWidth); // clear from previous runs

            for(int i=0;i<numInstances;i++)
            {
                // use lookup list to find the index into the full character list
                FillDataElement(&(m_pDataTemp[i]),drawState.instancesUsingThisMesh[buffer*INTERNAL_MAX_INSTANCES_PER_BUFFER+i]);
            }

            pd3dDevice->UpdateSubresource(drawState.pInstanceDataBuffer[buffer],D3D10CalcSubresource(0,0,1),NULL,(void*)m_pDataTemp,bDesc.ByteWidth,1);
        }
    }
    return hr;
}

/*
    This creates and populates a texture that contains all the animation data for all frames of all m_animations.
*/
HRESULT ArmyManager::CreateAnimationTexture(ID3D10Device* pd3dDevice)
{
    // Loop through all m_animations and find the total size of all m_animations in bones
    int maxBones=0;
    for(int currentAnimation=0;currentAnimation<(int)m_ReferenceCharacter->m_animations.size();currentAnimation++)
    {
        CharacterAnimation *pAnimation = m_ReferenceCharacter->m_animations[currentAnimation];
        maxBones += (int)pAnimation->frames.size() * m_ReferenceCharacter->numBones;
    }

    UINT texelsPerBone = 4;

    UINT pixelCount = maxBones * texelsPerBone;    // rowsPerBone lines per matrix, since no projection
    UINT texWidth = 0;
    UINT texHeight = 0;

    // This basically fits the animation into a roughly square texture where the 
    //      width is a multiple of rowsPerBone(our size requirement for matrix storage)
    //      AND both dimensions are power of 2 since it seems to fail without this...
    texWidth = (int)sqrt((float)pixelCount);    // gives us a starting point
    texHeight = 1;
    while(texHeight < texWidth) 
        texHeight = texHeight<<1;
    texWidth = texHeight;

    HRESULT hr = S_OK;
    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory( &desc, sizeof(D3D10_TEXTURE2D_DESC) );

    // Create our texture
    desc.MipLevels = 1;
    desc.Usage = D3D10_USAGE_IMMUTABLE;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.ArraySize = 1;
    desc.Width = texWidth;
    desc.Height = texHeight;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;

    // Make a offline buffer that is the full size of the texture
    UINT bufferSize = texHeight*texWidth*sizeof(D3DXVECTOR4);
    D3DXVECTOR4 *pData = new D3DXVECTOR4[desc.Width*desc.Height];
    memset((void*)pData,0,bufferSize);    // clear it

    D3DXVECTOR4 * pCurrentDataPtr = pData;
    for(int currentAnimation=0;currentAnimation<(int)m_ReferenceCharacter->m_animations.size();currentAnimation++)
    {
        CharacterAnimation *pAnimation = m_ReferenceCharacter->m_animations[currentAnimation];

        // now copy in each frame of the animation by copying each bone matrix.
        for(int currentFrame=0;currentFrame<(int)pAnimation->frames.size();currentFrame++)
        {
            // Here we pass through the full matrix, but the 4th row is just padding, weve encoded
            //      all the data into the first 3 rows.
            D3DXVECTOR4 *pFrameMatrices = (D3DXVECTOR4 *)pAnimation->GetEncodedFrame(currentFrame);

            for(int iBone=0;iBone < m_ReferenceCharacter->numBones;iBone++)
            {
                // Full uncompressed float4x4 matrix
                memcpy((void*)pCurrentDataPtr++,(void*)(pFrameMatrices[iBone*4]),sizeof(D3DXVECTOR4));    // copy in our matrix data
                memcpy((void*)pCurrentDataPtr++,(void*)(pFrameMatrices[iBone*4 + 1]),sizeof(D3DXVECTOR4));    // copy in our matrix data
                memcpy((void*)pCurrentDataPtr++,(void*)(pFrameMatrices[iBone*4 + 2]),sizeof(D3DXVECTOR4));    // copy in our matrix data
                memcpy((void*)pCurrentDataPtr++,(void*)(pFrameMatrices[iBone*4 + 3]),sizeof(D3DXVECTOR4));    // copy in our matrix data
            }
        }
    }

    D3D10_SUBRESOURCE_DATA srd;
    srd.pSysMem = (void*)pData;
    srd.SysMemPitch = texWidth*(sizeof(D3DXVECTOR4));
    srd.SysMemSlicePitch = 1;
    V(pd3dDevice->CreateTexture2D( &desc,&srd, &m_pAnimationTexture));

    delete[] pData;

    // Make a resource view for it
    D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
    ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
    SRVDesc.Format = desc.Format;
    SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
    SRVDesc.Texture2D.MipLevels = desc.MipLevels;
    V_RETURN(pd3dDevice->CreateShaderResourceView( m_pAnimationTexture, &SRVDesc, &m_pAnimationResourceView ));

    return hr;
}

/*
    Calculates an offset in texels for the start of an animation
*/
UINT ArmyManager::GetAnimationOffset(int animation)
{
    if(!m_ReferenceCharacter->m_animations.size()) return 0;

    int bonesOffset = 0;
    assert(animation < (int)m_ReferenceCharacter->m_animations.size());
    for(int currentAnimation=0;currentAnimation<animation;currentAnimation++)
    {
        CharacterAnimation *pAnimation = m_ReferenceCharacter->m_animations[currentAnimation];
        bonesOffset += (int)pAnimation->frames.size() * m_ReferenceCharacter->numBones;
    }

    return bonesOffset * 4;    // 4 pixels per bone
}

void ArmyManager::Release()
{
    for(int i=0;i<4;i++)
    {
        m_CharacterLODs[i]->Release();
        delete m_CharacterLODs[i];
    }

    SAFE_RELEASE(m_pInputLayout);
    SAFE_RELEASE(m_pInputLayoutTerrain);
    SAFE_RELEASE(m_pEffect10);
    SAFE_RELEASE(m_pAnimationResourceView);
    SAFE_RELEASE(m_pAnimationTexture);
    SAFE_RELEASE(m_pAlbedoTextureArray);
    SAFE_RELEASE(m_pAlbedoTextureArraySRV);
    while(m_Characters.size() > 0)
    {
        AnimatedCharacterInstance *c = m_Characters.back();
        delete c;
        m_Characters.pop_back();
    }

    while(m_InstancedDraws.size() > 0)
    {
        for(int i=0;i<INTERNAL_MAX_BUFFERS;i++)
            SAFE_RELEASE(m_InstancedDraws.back().pInstanceDataBuffer[i]);
        m_InstancedDraws.back().instancesUsingThisMesh.clear();
        m_InstancedDraws.pop_back();
    }

    m_GroundMesh.Destroy();
}

/*
    Creates a texture array where each slice is a dfferent texture loaded from the textures that the mesh references.
*/
HRESULT ArmyManager::CreateAlbedoTextureArray(ID3D10Device* pd3dDevice)
{
    TextureLibrary *library = TextureLibrary::singleton();
    int arraySize = library->NumTextures();

    if(!arraySize) return S_OK;

    // $$ Assuming that all textures are same dimensions ($$$ explain more!!)
    D3D10_TEXTURE2D_DESC singledesc;
    library->GetTexture(0)->GetDesc(&singledesc);

    HRESULT hr = S_OK;
    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory( &desc, sizeof(D3D10_TEXTURE2D_DESC) );

    // Create our texture
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    desc.ArraySize = arraySize; // array dimensions

    desc.Width = singledesc.Width;
    desc.Height = singledesc.Height;
    desc.Format = singledesc.Format;
    desc.MipLevels = singledesc.MipLevels;

    V(pd3dDevice->CreateTexture2D( &desc,NULL, &m_pAlbedoTextureArray));

    // Now Update the array with each slice, have to manually copy each mip level as well...
    for(int iTexture = 0; iTexture < arraySize; iTexture++)
    {
        ID3D10Texture2D *pTexture = library->GetTexture(iTexture);
        pTexture->GetDesc(&singledesc);

//        int iMip = 0;
        D3D10_BOX box;
        box.back = 1;
        box.front = 0;
        box.left = 0;
        box.right = singledesc.Width;
        box.top = 0;
        box.bottom = singledesc.Height;
        for(int iMip=0;iMip<(int)desc.MipLevels;iMip++)
        {

//        pd3dDevice->CopyResource(m_pAlbedoTextureArray,pTexture);
            pd3dDevice->CopySubresourceRegion(    m_pAlbedoTextureArray,D3D10CalcSubresource(iMip,iTexture,desc.MipLevels),
                                                0,0,0,pTexture,D3D10CalcSubresource(iMip,0,singledesc.MipLevels),NULL);
        }
    }

    D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
    ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
    SRVDesc.Format = desc.Format;
    SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
    SRVDesc.Texture2DArray.ArraySize = arraySize;
    SRVDesc.Texture2DArray.FirstArraySlice = 0;
    SRVDesc.Texture2DArray.MipLevels = desc.MipLevels;
    SRVDesc.Texture2DArray.MostDetailedMip = 0;
    V(pd3dDevice->CreateShaderResourceView( m_pAlbedoTextureArray, &SRVDesc, &m_pAlbedoTextureArraySRV));

    return S_OK;
}

