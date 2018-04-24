//-----------------------------------------------------------------------------
// File: MultiAnimationLib.cpp
//
// Desc: Implementation of the CMultiAnim class. This class manages the animation
//       data (frames and meshes) obtained from a single X file.
//
// Copyright (c) Microsoft Corporation. All rights reserved
//-----------------------------------------------------------------------------
#include "DXUT.h"
#include "SDKmisc.h"
#pragma warning(disable: 4995)
#include "MultiAnimation.h"
#pragma warning(default: 4995)

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "zlib/zlib.h"

using namespace std;

CMultiAnim::CMultiAnim() :
    m_pDevice( NULL ),
    m_pFrameRoot( NULL ),
    m_pAC( NULL )
{
}


//-----------------------------------------------------------------------------
// Name: CMultiAnim::~CMultiAnim()
// Desc: Destructor for CMultiAnim
//-----------------------------------------------------------------------------
CMultiAnim::~CMultiAnim()
{
    m_pFrameRoot = NULL;
    m_pAC = NULL;
}



//-----------------------------------------------------------------------------
// Name: CMultiAnim::Setup()
// Desc: The class is initialized with this method.
//       We create the effect from the fx file, and load the animation mesh
//       from the given X file.  We then call SetupBonePtrs() to Initialize
//       the mesh containers to enable bone matrix lookup by index.  The
//       Allocation Hierarchy is passed by pointer to allow an app to subclass
//       it for its own implementation.
//-----------------------------------------------------------------------------
HRESULT CMultiAnim::Setup( LPDIRECT3DDEVICE9 pDevice,
                           WCHAR sXFile[],
                           CMultiAnimAllocateHierarchy *pAH,
                           LPD3DXLOADUSERDATA pLUD )
{
    WCHAR wszPath[MAX_PATH];
    D3DXVECTOR3 vCenter = D3DXVECTOR3(0,0,0);
    // set the MA instance for CMultiAnimAllocateHierarchy
    pAH->SetMA( this );

    // set the device
    m_pDevice = pDevice;
    m_pDevice->AddRef();

    HRESULT hr;

    // create the mesh, frame hierarchy, and animation controller from the x file
    hr = DXUTFindDXSDKMediaFileCch( wszPath, MAX_PATH, sXFile );
    if( FAILED( hr ) )
        goto e_Exit;

    hr = D3DXLoadMeshHierarchyFromX( wszPath,
                                     0,
                                     m_pDevice,
                                     pAH,
                                     pLUD,
                                     (LPD3DXFRAME *) &m_pFrameRoot,
                                     &m_pAC );
    if( FAILED( hr ) )
    {
        // try to load using the zlib decompressor, maybe the input is compressed?
        // This opens the file using zlib utils, then allocates a HUGE heap mem pool for the decompressed output.
        // tries to load from that using D3DX utils, and then cleans up.
        CHAR localFilename[MAX_PATH];
        WideCharToMultiByte( CP_ACP, 0,wszPath,MAX_PATH,localFilename,MAX_PATH,"",FALSE);
        gzFile zcompFile = gzopen(localFilename,"rb");
        if(zcompFile == NULL) goto e_Exit;
        const unsigned int CHUNK_SIZE = 131072;
        const unsigned int MAX_INPUTFILE_SIZE = 50 * 1024 * 1024;
        unsigned char *pData = new unsigned char[MAX_INPUTFILE_SIZE];
        int numbytes = 0;
        unsigned int bytesRead = 0;
        do
        {
            // reads from file, decompress right into pur dest buffer
            numbytes = gzread(zcompFile,(void*)(pData+bytesRead),CHUNK_SIZE);
            if(numbytes == -1 || bytesRead + numbytes > MAX_INPUTFILE_SIZE) 
            {
                delete [] pData;
                goto e_Exit;
            }
            bytesRead += numbytes;
        }while(numbytes > 0);

        hr = D3DXLoadMeshHierarchyFromXInMemory(pData,bytesRead,
                                     0,
                                     m_pDevice,
                                     pAH,
                                     pLUD,
                                     (LPD3DXFRAME *) &m_pFrameRoot,
                                     &m_pAC );

        delete [] pData;

        if( FAILED(hr))
        {
            goto e_Exit;
        }

    }
        

    // get bounding radius
    hr = D3DXFrameCalculateBoundingSphere( m_pFrameRoot, & vCenter, & m_fBoundingRadius );
    if( FAILED( hr ) )
        goto e_Exit;

e_Exit:

    if( FAILED( hr ) )
    {

        if( m_pAC )
        {
            m_pAC->Release();
            m_pAC = NULL;
        }

        if( m_pFrameRoot )
        {
            D3DXFrameDestroy( m_pFrameRoot, pAH );
            m_pFrameRoot = NULL;
        }

        m_pDevice->Release();
        m_pDevice = NULL;
    }

    return hr;
}



//-----------------------------------------------------------------------------
// Name: CMultiAnim::Cleanup()
// Desc: Performs clean up work and free up memory.
//-----------------------------------------------------------------------------
HRESULT CMultiAnim::Cleanup( CMultiAnimAllocateHierarchy * pAH )
{
    if( m_pAC )
    {
        m_pAC->Release();
        m_pAC = NULL;
    }

    if( m_pFrameRoot )
    {
        D3DXFrameDestroy( m_pFrameRoot, pAH );
        m_pFrameRoot = NULL;
    }

    if( m_pDevice )
    {
        m_pDevice->Release();
        m_pDevice = NULL;
    }

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name: CMultiAnim::GetDevice()
// Desc: Returns the D3D device we work with.  The caller must call Release()
//       on the pointer when done with it.
//-----------------------------------------------------------------------------
LPDIRECT3DDEVICE9 CMultiAnim::GetDevice()
{
    m_pDevice->AddRef();
    return m_pDevice;
}



//-----------------------------------------------------------------------------
// Name: CMultiAnim::GetBoundingRadius()
// Desc: Returns the bounding radius for the mesh object.
//-----------------------------------------------------------------------------
float CMultiAnim::GetBoundingRadius()
{
    return m_fBoundingRadius;
}

