//----------------------------------------------------------------------------------
// File:   DX9AnimationLoader.cpp
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
#include "DX9AnimationLoader.h"
#include "TextureLibray.h"


/*
    PLEASE read the header for information on this class.  It is not meant to demonstrate DX10 features.
*/
DX9AnimationLoader::DX9AnimationLoader()
{
    multiAnim = NULL;
    AH = NULL;
    pMats = NULL;
    pAnimatedCharacter = NULL;
    pD3D9 = NULL;
    pDev9 = NULL;
    pDev10 = NULL;
    pBoneMatrixPtrs = NULL;
    pBoneOffsetMatrices = NULL;
}

DX9AnimationLoader::~DX9AnimationLoader()
{
    frameNames.clear();
    boneNames.clear();
    attachmentNames.clear();
}

HRESULT DX9AnimationLoader::load(ID3D10Device *pDev10, LPWSTR filename,CONST D3D10_INPUT_ELEMENT_DESC*playout,int cElements,bool bLoadAnimations,float timeStep)
{
    HRESULT hr = S_OK;
    this->pDev10 = pDev10;
    this->pDev10->AddRef();

    prepareDevice();
    
    hr = loadDX9Data(filename);
    if(hr != S_OK) return hr;

    fillMeshList((D3DXFRAME*)multiAnim->m_pFrameRoot);    

    generateDX10Buffer(playout,cElements);

    if(bLoadAnimations && pSkinInfos.size() > 0)
    {
        extractSkeleton();

        processAnimations(timeStep);

    }
    cleanup();
    this->pDev10->Release();
    return hr;
}

HRESULT DX9AnimationLoader::loadAndBindLODMesh(ID3D10Device *pDev10, LPWSTR filename,CONST D3D10_INPUT_ELEMENT_DESC*playout,int cElements,bool bLoadAnimations,float timeStep)
{
    // must have already run once on the base LOD
    // $$ BED This is also ugly.. boneNames is initialized from first load and contains the skeleton bone name to index mapping...
    if(boneNames.size() == 0) return E_FAIL;

    HRESULT hr = S_OK;
    this->pDev10 = pDev10;
    this->pDev10->AddRef();

    prepareDevice();
    
    hr = loadDX9Data(filename);
    if(hr != S_OK) return hr;

    fillMeshList((D3DXFRAME*)multiAnim->m_pFrameRoot);    

    generateDX10Buffer(playout,cElements);

    // $ We skip animations, since they were loaded on previous load

    cleanup();
    this->pDev10->Release();
    return hr;
}

void DX9AnimationLoader::prepareDevice()
{
    HRESULT hr;

    pD3D9 = Direct3DCreate9( D3D_SDK_VERSION );
    assert( pD3D9 );

    D3DPRESENT_PARAMETERS pp;
    pp.BackBufferWidth = 320;
    pp.BackBufferHeight = 240;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8;
    pp.BackBufferCount = 1;
    pp.MultiSampleType = D3DMULTISAMPLE_NONE;
    pp.MultiSampleQuality = 0;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.hDeviceWindow = GetShellWindow();
    pp.Windowed = true;
    pp.Flags = 0;
    pp.FullScreen_RefreshRateInHz = 0;
    pp.PresentationInterval = D3DPRESENT_INTERVAL_DEFAULT;
    pp.EnableAutoDepthStencil = false;

    hr = pD3D9->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, NULL, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &pp, &pDev9 );
    if( FAILED( hr ) )
    {
        V(pD3D9->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_REF, NULL, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &pp, &pDev9 ));
    }

    pAnimatedCharacter = new AnimatedCharacter();
    AH = new CMultiAnimAllocateHierarchy();
    multiAnim = new CMultiAnim();
}

HRESULT DX9AnimationLoader::loadDX9Data(LPWSTR filename)
{
    HRESULT hr;
    WCHAR sXFile[MAX_PATH];

    V( DXUTFindDXSDKMediaFileCch( sXFile, MAX_PATH, filename ) );

    AH->SetMA( multiAnim );

    // This is taken from the MultiAnim DX9 sample
    V( multiAnim->Setup( pDev9, sXFile, AH ) );

    return hr;
}

// simple recursive to extract all mesh containers.
void DX9AnimationLoader::fillMeshList(D3DXFRAME *pFrame)
{
    // if we have a mesh container, add it to the list
    if(pFrame->pMeshContainer && pFrame->pMeshContainer->MeshData.pMesh) 
    {
        MultiAnimMC* pMCFrame = (MultiAnimMC*)pFrame->pMeshContainer;
        //if(!this->pSkinInfo && pMCFrame->pSkinInfo) this->pSkinInfo = pMCFrame->pSkinInfo;
        meshList.insert(meshList.begin(),pMCFrame);
        //meshList.push_back(pMCFrame);
    }
    else
    {
        frameNames.push_back(pFrame->Name);
    }

    // then process siblings and children
    if(pFrame->pFrameSibling) fillMeshList(pFrame->pFrameSibling);
    if(pFrame->pFrameFirstChild) fillMeshList(pFrame->pFrameFirstChild);
}

void DX9AnimationLoader::generateDX10Buffer(CONST D3D10_INPUT_ELEMENT_DESC*playout,int cElements)
{
    HRESULT hr;
    this->playout = playout;
    this->cElements = cElements;

    // First parse all the meshes and extract their textures
    for(int iCurrentMeshContainer=0;iCurrentMeshContainer<(int)meshList.size();iCurrentMeshContainer++)
    {
        MultiAnimMC *pMeshContainer = (MultiAnimMC *)meshList[iCurrentMeshContainer];

        // Process material...  should only be one per mesh
        ID3D10Texture2D *pTexture10 = NULL;
        ID3D10ShaderResourceView *pSRV10 = NULL;

        if(pMeshContainer->m_pTextureFilename && pMeshContainer->m_pTextureFilename[0] != 0)
        {
            std::string filename = pMeshContainer->m_pTextureFilename;
            std::string name = filename;
            std::string::size_type pos = name.find_last_of('\\') + 1;
            if(pos == 0) pos = name.find_last_of('/') + 1;
            name = name.substr(pos,name.length()-pos);

            // reassign to be the shortened version of the name
            strcpy_s(pMeshContainer->m_pTextureFilename,name.c_str());

            if(TextureLibrary::singleton()->GetTexture(name.c_str()) == NULL)
            {
                WCHAR strName[MAX_PATH];
                MultiByteToWideChar( CP_ACP, 0, pMeshContainer->m_pTextureFilename, -1, strName, MAX_PATH );

                WCHAR str[MAX_PATH];
                if( SUCCEEDED(DXUTFindDXSDKMediaFileCch( str, MAX_PATH, strName )) )
                {
                    ID3D10Resource *pRes = NULL;
                    hr = D3DX10CreateTextureFromFile( pDev10, str, NULL, NULL, &pRes );
                    if( SUCCEEDED(hr) && pRes )
                    {
                        pRes->QueryInterface( __uuidof( ID3D10Texture2D ), (LPVOID*)&pTexture10 );
                        D3D10_TEXTURE2D_DESC desc;
                        pTexture10->GetDesc( &desc );
                        D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
                        ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
                        SRVDesc.Format = desc.Format;
                        SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
                        SRVDesc.Texture2D.MipLevels = desc.MipLevels;
                        pDev10->CreateShaderResourceView( pTexture10, &SRVDesc, &pSRV10 );
                        SAFE_RELEASE( pRes );

                        TextureLibrary::singleton()->AddTexture(name.c_str(),pTexture10,pSRV10);
                        pAnimatedCharacter->material = name.c_str();

                        SAFE_RELEASE(pTexture10);
                        SAFE_RELEASE(pSRV10);

                    }
                }
            }
        }
    }

    pAnimatedCharacter->numAttachments = (int)attachmentNames.size();

    // Go thru all mesh containers and extract geometry into our buffer
    for(int iCurrentMeshContainer=0;iCurrentMeshContainer<(int)meshList.size();iCurrentMeshContainer++)
    {
        MultiAnimMC *pMeshContainer = (MultiAnimMC *)meshList[iCurrentMeshContainer];
        std::string name = pMeshContainer->m_pTextureFilename;

        if(!pMeshContainer->pSkinInfo) continue;    // ignore non skinned geo

        float textureIndex = (float)TextureLibrary::singleton()->GetTextureIndex(name);    // texture map index

        // and now the mesh
        LPD3DXMESH    pDX9Mesh = pMeshContainer->MeshData.pMesh;
        LPD3DXMESH pMesh = NULL;
        DWORD dwNumVerts = 0;
        DWORD dwNumIndices = 0;
        DWORD uStride  = 0;

        // Now extract data info DX10
        D3DVERTEXELEMENT9 pdecl9[] = 
        {
            { 0, 0,  D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
            { 0, 12, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0 },
            { 0, 24, D3DDECLTYPE_FLOAT2,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
            { 0, 32, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0 },
            { 0, 44, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BINORMAL, 0 },
            D3DDECL_END()
        };

        // Make a clone with the desired vertex format.
        if( SUCCEEDED( pDX9Mesh->CloneMesh( D3DXMESH_32BIT | D3DXMESH_DYNAMIC, pdecl9, pDev9, &pMesh ) ) )
        {
            DWORD dwFlags = 0;
            dwFlags |= D3DXTANGENT_GENERATE_IN_PLACE;
            dwFlags |= D3DXTANGENT_DONT_NORMALIZE_PARTIALS;
            hr = D3DXComputeTangentFrameEx(pMesh, 
                                    D3DDECLUSAGE_TEXCOORD, 
                                    0,   
                                    D3DDECLUSAGE_BINORMAL, 
                                    0, 
                                    D3DDECLUSAGE_TANGENT, 0, 
                                    D3DDECLUSAGE_NORMAL, 0, 
                                    dwFlags ,
                                    NULL, 0.01f, 0.25f, 0.01f, NULL, NULL);

            //DWORD *adjacency = new DWORD[3*pMesh->GetNumFaces()];
            //pMesh->GenerateAdjacency(0.01f,adjacency);
            //hr = D3DXComputeTangent(pMesh,D3DDECLUSAGE_TEXCOORD,D3DDECLUSAGE_TANGENT,0,0,adjacency);
            //delete [] adjacency;
            if( FAILED(hr) )
            {
                DXUTTRACE( L"Failed to compute tangents for the mesh.\n" );
            }

            // get some basic data about the mesh
            dwNumVerts = pMesh->GetNumVertices();
            uStride = pMesh->GetNumBytesPerVertex();
            dwNumIndices = pMesh->GetNumFaces()*3;

            //set the VB
            struct _LocalVertElement
            {
                D3DXVECTOR3 pos;
                D3DXVECTOR3 normal;
                D3DXVECTOR2 tex;
                D3DXVECTOR3 tangent;
                D3DXVECTOR3 binormal;
            };

            _LocalVertElement *sourceVerts = NULL;

            AnimatedCharacter::Vertex *vertices = new AnimatedCharacter::Vertex[dwNumVerts];
            pMesh->LockVertexBuffer( 0, (void**)&sourceVerts );

            // scan for meshes that have been flagged as "attachments", these can be selectively killed.
            float attachmentValue = 0.f;
            if(pMeshContainer->Name && strncmp("attachment_",pMeshContainer->Name,strlen("attachment_")) == 0)
            {
                // find attachment if it already exists...
                int attachIndex = -1;
                for(int i=0;i<(int)attachmentNames.size();i++)
                {
                    if(strcmp(attachmentNames[i].c_str(),pMeshContainer->Name) == 0)
                    {
                        attachIndex = i;
                        break;
                    }
                }
                if(attachIndex == -1)
                {
                    attachIndex = (int)attachmentNames.size();
                    attachmentNames.push_back(pMeshContainer->Name);
                    pAnimatedCharacter->numAttachments++;
                }

                // attachment flag(0 = base, >0 = bit flag for each attach)
                attachmentValue = (float)(1<<(attachIndex));    
            }

            for(int iCurrentVert = 0;iCurrentVert<(int)dwNumVerts;iCurrentVert++)
            {
                vertices[iCurrentVert].position = sourceVerts[iCurrentVert].pos;
                vertices[iCurrentVert].normal = sourceVerts[iCurrentVert].normal;
                vertices[iCurrentVert].uv.x = sourceVerts[iCurrentVert].tex.x;
                vertices[iCurrentVert].uv.y = -sourceVerts[iCurrentVert].tex.y;
                vertices[iCurrentVert].tangent = sourceVerts[iCurrentVert].tangent;
                vertices[iCurrentVert].bones = D3DXVECTOR4(0,0,0,0);
                vertices[iCurrentVert].weights = D3DXVECTOR4(0.f,0,0,0);
            }

            if(pMeshContainer->pSkinInfo)
            {
                // Update our list of aggregate bones
                pSkinInfos.push_back(pMeshContainer->pSkinInfo);
                DWORD dwNumBones = pMeshContainer->pSkinInfo->GetNumBones();

                // Now go through the skin and set the bone indices and weights, max 4 per vert
                for(int iCurrentBone=0;iCurrentBone<(int)dwNumBones;iCurrentBone++)
                {
                    LPCSTR boneName = pMeshContainer->pSkinInfo->GetBoneName(iCurrentBone);
                    int numInfl = pMeshContainer->pSkinInfo->GetNumBoneInfluences(iCurrentBone);
                    DWORD *verts = new DWORD[numInfl];
                    FLOAT *weights = new FLOAT[numInfl];
                    pMeshContainer->pSkinInfo->GetBoneInfluence(iCurrentBone,verts,weights);

                    // $ BED find bone index and update bones used array
                    int boneIndex = -1;
                    for(int i=0;i<(int)boneNames.size();i++)
                    {
                        if(strcmp(boneNames[i].c_str(),boneName) == 0)
                        {
                            boneIndex = i;
                            break;
                        }
                    }
                    if(boneIndex == -1)
                    {
                        boneIndex = (int)boneNames.size();
                        boneNames.push_back(boneName);
                    }

                    // find the top 4 influences

                    for(int iInfluence = 0;iInfluence<numInfl;iInfluence++)
                    {
                        if(weights[iInfluence] <= 0.05f) continue;
                        if(vertices[verts[iInfluence]].weights.x <= 0.f)
                        {
                            vertices[verts[iInfluence]].bones.x = (float)boneIndex;
                            vertices[verts[iInfluence]].weights.x = weights[iInfluence];
                        }
                        else if(vertices[verts[iInfluence]].weights.y <= 0.f)
                        {
                            vertices[verts[iInfluence]].bones.y = (float)boneIndex;
                            vertices[verts[iInfluence]].weights.y = weights[iInfluence];
                        }
                        else if(vertices[verts[iInfluence]].weights.z <= 0.f)
                        {
                            vertices[verts[iInfluence]].bones.z = (float)boneIndex;
                            vertices[verts[iInfluence]].weights.z = weights[iInfluence];
                        }
                        else if(vertices[verts[iInfluence]].weights.w <= 0.f)
                        {
                            vertices[verts[iInfluence]].bones.w = (float)boneIndex;
                            vertices[verts[iInfluence]].weights.w = weights[iInfluence];
                        }
                    }
                    delete [] verts;
                    delete [] weights;
                }
            }
            pMesh->UnlockVertexBuffer();

            //set the IB
            DWORD *pIndices = new DWORD[dwNumIndices];
            void*pData = NULL;
            pMesh->LockIndexBuffer( 0, (void**)&pData );
            if(pMesh->GetOptions() & D3DXMESH_32BIT)
            {
                CopyMemory((void*)pIndices,pData,dwNumIndices*sizeof(DWORD));
            }
            else
            {
                WORD* pSmallIndices = (WORD*)pData;
                DWORD* pBigIndices = new DWORD[ dwNumIndices ];
                for(DWORD i=0; i<dwNumIndices; i++)
                    pBigIndices[i] = pSmallIndices[i];
                CopyMemory((void*)pIndices,(void *)pBigIndices,dwNumIndices*sizeof(DWORD));
                SAFE_DELETE_ARRAY( pBigIndices );
            }
            pMesh->UnlockIndexBuffer();

            unsigned int hash = 1315423911;
            for(int i=0;i<(int)strlen(pMeshContainer->Name);i++)
            {
                hash ^= ((hash << 5) + pMeshContainer->Name[i] + (hash >> 2));
            }

            if(attachmentValue == 0) hash = 0;

            // Add in separate buffers for the single Draw case
            pAnimatedCharacter->addSingleDrawMesh(pDev10,vertices,dwNumVerts,pIndices,dwNumIndices,(int)attachmentValue,hash,(int)textureIndex);

            // clean up after ourselves.
            delete [] vertices;
            delete [] pIndices;

            // copy in our data pointers
            pAnimatedCharacter->boundingRadius = multiAnim->m_fBoundingRadius;
        }
        SAFE_RELEASE(pMesh);
    }
}

// walks tree and updates the transformations matrices to be combined
void DX9AnimationLoader::UpdateFrames( MultiAnimFrame * pFrame, D3DXMATRIX * pmxBase )
{
    assert( pFrame != NULL );
    assert( pmxBase != NULL );

    // transform the bone, aggregating in our parent matrix
    D3DXMatrixMultiply( &pFrame->CombinedTransformationMatrix,
        &pFrame->TransformationMatrix,
        pmxBase );

    // transform siblings by the same matrix
    if( pFrame->pFrameSibling )
        UpdateFrames( (MultiAnimFrame *) pFrame->pFrameSibling, pmxBase );

    // transform children by the transformed matrix - hierarchical transformation
    if( pFrame->pFrameFirstChild )
        UpdateFrames( (MultiAnimFrame *) pFrame->pFrameFirstChild,
        &pFrame->CombinedTransformationMatrix );
}

void DX9AnimationLoader::extractSkeleton()
{
    // Update the bone count
    pAnimatedCharacter->numBones = (int)boneNames.size();

    MultiAnimFrame *pFrame;
    pBoneMatrixPtrs = new D3DXMATRIX*[pAnimatedCharacter->numBones];
    pBoneOffsetMatrices = new D3DXMATRIX [ pAnimatedCharacter->numBones ];

    // pull out pointers to the bone matrices in the hierarchy, for each skin info, effectively merging the skeletons
    //
    // NOTE:  this loop will most certainly redundantly set the matrices and ptrs, but should set to same values
    //      and is only on load, so we conveniently ignore that fact... :/
    for(int iCurrentSI=0;iCurrentSI<(int)pSkinInfos.size();iCurrentSI++)
    {
        LPD3DXSKININFO pSkinInfo = pSkinInfos[iCurrentSI];
        DWORD dwNumBones = pSkinInfo->GetNumBones();

        // pull out pointers to the bone matrices in the hierarchy        
        // Insert them into our list of bones, offset by the subskeleton
        for (DWORD iBone = 0; iBone < dwNumBones; iBone++)
        {
            // find the actual index in our local list...  this coalesces the palette
            LPCSTR boneName = pSkinInfo->GetBoneName(iBone);
            int boneIndex = -1;
            for(int iLocalBone = 0;iLocalBone<(int)boneNames.size();iLocalBone++)
            {
                if(strcmp(boneNames[iLocalBone].c_str(),boneName) == 0)
                {
                    boneIndex = iLocalBone;
                    break;
                }
            }

            assert(boneIndex != -1);

            pFrame = (MultiAnimFrame*)D3DXFrameFind( multiAnim->m_pFrameRoot, boneName );
            assert(pFrame);
            pBoneMatrixPtrs[boneIndex] = &pFrame->CombinedTransformationMatrix;
            pBoneOffsetMatrices[boneIndex] = *pSkinInfo->GetBoneOffsetMatrix(iBone);
        }
    }
}

void DX9AnimationLoader::extractFrame(CharacterAnimation *pAnimation)
{
    // Push down the hierarchy
    D3DXMATRIX mIdentity;
    D3DXMatrixIdentity(&mIdentity);
    UpdateFrames(multiAnim->m_pFrameRoot,&mIdentity);
    
    D3DXMATRIX *pFrame = new D3DXMATRIX[pAnimatedCharacter->numBones];
    D3DXMATRIX *pEncodedFrame = new D3DXMATRIX[pAnimatedCharacter->numBones];
    for(int i=0;i<pAnimatedCharacter->numBones;i++)
    {
        D3DXMatrixMultiply(&(pFrame[i]),&(pBoneOffsetMatrices[i]),pBoneMatrixPtrs[i]);

        pEncodedFrame[i] = pFrame[i];
        pEncodedFrame[i]._14 = pEncodedFrame[i]._41;
        pEncodedFrame[i]._24 = pEncodedFrame[i]._42;
        pEncodedFrame[i]._34 = pEncodedFrame[i]._43;
    }
    pAnimation->frames.push_back(pFrame);
    pAnimation->encodedFrames.push_back(pEncodedFrame);
}

// Go through each animation, and run through it once, and generate key frames based on a constant subdivision.
void DX9AnimationLoader::processAnimations(float timeStep)
{
    LPD3DXANIMATIONCONTROLLER pAnimationController = multiAnim->m_pAC;
    
    if(pAnimationController == NULL) return;

    // Start with all tracks disabled
    UINT dwTracks = pAnimationController->GetMaxNumTracks();
    for( int i = 0; i < (int)dwTracks; ++ i )
        pAnimationController->SetTrackEnable( i, FALSE );

    pAnimationController->SetTrackEnable( 0, TRUE );    // only need one as we aren't blending m_animations here.

    // Go through all animation sets, set them to a track, and play them through capturing frames... ugh
    int numAnimationSets = pAnimationController->GetNumAnimationSets();
    for(int iCurrentAnimationSet=0;iCurrentAnimationSet<numAnimationSets;iCurrentAnimationSet++)
    {
        LPD3DXKEYFRAMEDANIMATIONSET pAnimationSet = NULL;
        CharacterAnimation *pCharacterAnimation = new CharacterAnimation();

        pAnimationController->GetAnimationSet(iCurrentAnimationSet,(LPD3DXANIMATIONSET *)&pAnimationSet);
        pAnimationController->SetTrackAnimationSet(0,pAnimationSet);
        pAnimationSet->Release();

        pAnimationController->ResetTime();
        pCharacterAnimation->duration = (float)pAnimationSet->GetPeriod();
        pCharacterAnimation->timeStep = timeStep;
        pCharacterAnimation->name = pAnimationSet->GetName();
        for(double dTime =0.0;dTime < (double)pCharacterAnimation->duration;dTime+=(double)pCharacterAnimation->timeStep)
        {
            pAnimationController->AdvanceTime((double)pCharacterAnimation->timeStep,NULL);
            extractFrame(pCharacterAnimation);
        }
        pAnimatedCharacter->m_animations.push_back(pCharacterAnimation);
    }
}

void DX9AnimationLoader::cleanup()
{
    meshList.clear();
    multiAnim->Cleanup(AH);
    delete multiAnim;
    delete AH;
    SAFE_DELETE_ARRAY(pMats);
    SAFE_DELETE_ARRAY(pBoneOffsetMatrices);
    SAFE_DELETE_ARRAY(pBoneMatrixPtrs);
    SAFE_RELEASE(pDev9);
    SAFE_RELEASE(pD3D9);
}