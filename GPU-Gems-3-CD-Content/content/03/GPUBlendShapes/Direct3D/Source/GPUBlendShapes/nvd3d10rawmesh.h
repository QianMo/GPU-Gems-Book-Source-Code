//----------------------------------------------------------------------------------
// File:   nvd3d10rawmesh.h
// Author: Tristan Lorach
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

//#ifndef __D3D10RawMesh_h__
//#define __D3D10RawMesh_h__

#include "NVRawMesh.h"
#ifndef LOGMSG
#   define LOG_MSG stdout
#   define LOG_WARN stdout
#   define LOG_ERR stderr
#define LOGMSG fprintf
#endif
/*----------------------------------------------------------------------------------*/ /**


**/ //----------------------------------------------------------------------------------
class NVD3D10RawMesh
{
protected:
public:
  NVRawMesh::FileHeader     *pMesh;
  D3D10_INPUT_ELEMENT_DESC  *pLayoutDesc;
  int                       numLayoutAttribs;
  int                       numLayoutSlots;
  int                       numVertexBuffers;
  int                       numStreamOutBuffers;
  int                       numIndexBuffers;
  //
  // Various ways of exposing the buffers :
  // as a typical buffer
  // as a resource + its resource view : to bind it to a Buffer<> templated variable
  //
  struct BlendShapesResource{
      ID3D10Buffer              *pVertexResource;   // big chunk of memory with all the vtx buffers as resources
      ID3D10ShaderResourceView  *pVertexView;       // corresponding views
      ID3D10Buffer              *pOffsetsResource;  // offsets of Blendshapes to use
      ID3D10ShaderResourceView  *pOffsetsView;
      ID3D10Buffer              *pWeightsResource;  // weights
      ID3D10ShaderResourceView  *pWeightsView;
      int                       numUsedBS;          // amount of used BS in Offsets/Weights
  };
  BlendShapesResource       bsResource;

  ID3D10Buffer              **pVertexBuffers;   // array of ptr of basic Vtx buffers
  ID3D10Buffer              **pIndexBuffers;    // array of ptr
  ID3D10Buffer              **pStreamOutBuffers;// array of ptr
  ID3D10InputLayout         *pLayout;
  UINT                      *pStrides;

/*----------------------------------------------------------------------------------*/ /**

**/ //----------------------------------------------------------------------------------
  NVD3D10RawMesh()
  {
    numLayoutSlots = 0;
    numLayoutAttribs = 0;
    numStreamOutBuffers = 0;
    pLayoutDesc = NULL;
    pMesh = NULL;
    pVertexBuffers = NULL;
    pIndexBuffers = NULL;
    pLayout = NULL;
    numIndexBuffers = 0;
    numVertexBuffers = 0;
    pStrides = 0;
    pStreamOutBuffers = NULL;
  }
/*----------------------------------------------------------------------------------*/ /**

**/ //----------------------------------------------------------------------------------
  bool checkDX10LayoutCompatibility(NVRawMesh::FileHeader *pFile, const char * layout)
  {
    return true;
  }
/*----------------------------------------------------------------------------------*/ /**

**/ //----------------------------------------------------------------------------------
  NVRawMesh::FileHeader* loadMesh(LPCSTR fname)
  {
    pMesh = NVRawMesh::loadMesh(fname);
    return pMesh;
  }
/*----------------------------------------------------------------------------------*/ /**

**/ //----------------------------------------------------------------------------------
  void destroy()
  {
    for(int i=0; i<numLayoutAttribs; i++)
      delete [] pLayoutDesc[i].SemanticName;
    SAFE_RELEASE(pLayout);
    pLayout = NULL;
    
    for(int i=0; i<numStreamOutBuffers; i++)
      if(pStreamOutBuffers[i]) SAFE_RELEASE(pStreamOutBuffers[i]);
    if(pStreamOutBuffers) delete [] pStreamOutBuffers;
    pStreamOutBuffers = NULL;
    numStreamOutBuffers = 0;

    if(bsResource.pVertexView)      SAFE_RELEASE(bsResource.pVertexView);
    if(bsResource.pVertexResource)  SAFE_RELEASE(bsResource.pVertexResource);
    if(bsResource.pOffsetsView)     SAFE_RELEASE(bsResource.pOffsetsView);
    if(bsResource.pOffsetsResource) SAFE_RELEASE(bsResource.pOffsetsResource);
    if(bsResource.pWeightsView)     SAFE_RELEASE(bsResource.pWeightsView);
    if(bsResource.pWeightsResource) SAFE_RELEASE(bsResource.pWeightsResource);
    bsResource.numUsedBS = 0;
    if(pVertexBuffers) 
    {
        for(int i=0; i<numVertexBuffers; i++)
          if(pVertexBuffers[i]) SAFE_RELEASE(pVertexBuffers[i]);
        delete [] pVertexBuffers;
    }
    pVertexBuffers = NULL;
    numVertexBuffers = 0;

    for(int i=0; i<numIndexBuffers; i++)
      if(pIndexBuffers[i]) SAFE_RELEASE(pIndexBuffers[i]);
    if(pIndexBuffers) delete [] pIndexBuffers;
    pIndexBuffers = NULL;
    numIndexBuffers = 0;

    if(pStrides) delete [] pStrides;
    pStrides = NULL;
    numLayoutAttribs = 0;
    numLayoutSlots = 0;

    if(pLayoutDesc) delete [] pLayoutDesc;
    pLayoutDesc = NULL;

    if(pMesh) free(pMesh);
    pMesh = NULL;
  }
/*----------------------------------------------------------------------------------*/ /**

**/ //----------------------------------------------------------------------------------
  void releaseBufferData()
  {
    if(pMesh) 
      pMesh = NVRawMesh::releaseBufferData(pMesh);
  }
/*----------------------------------------------------------------------------------*/ /**
Generate Layout description from what we have in the mesh
**/ //----------------------------------------------------------------------------------
  const D3D10_INPUT_ELEMENT_DESC * generateLayoutDesc(int appendBlendshapes=0, int slotstart=0)
  {
    int nbsattr = pMesh->bsLayout.num_attribs;
    pLayoutDesc = new D3D10_INPUT_ELEMENT_DESC[pMesh->layout.num_attribs + appendBlendshapes*nbsattr];
    for(int i=0; i<pMesh->layout.num_attribs; i++)
    {
        size_t l = strlen(pMesh->layout.attribs[i].name) + 1;
        char *name = new char[l];
        strcpy_s(name, l, pMesh->layout.attribs[i].name);
        pLayoutDesc[i].SemanticName = name;
        pLayoutDesc[i].SemanticIndex = pMesh->layout.attribs[i].semanticIdx;
        pLayoutDesc[i].Format = pMesh->layout.attribs[i].formatDX10;
        pLayoutDesc[i].InputSlot = pMesh->layout.attribs[i].slot;
        pLayoutDesc[i].AlignedByteOffset = pMesh->layout.attribs[i].AlignedByteOffset;
        pLayoutDesc[i].InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
        pLayoutDesc[i].InstanceDataStepRate = 0; // ??
    }
    numLayoutAttribs = pMesh->layout.num_attribs + (appendBlendshapes*nbsattr);
    int slotnum = slotstart == 0 ? pMesh->numSlots : slotstart;
    int layoutnum = pMesh->layout.num_attribs;
    for(int i=0; i<appendBlendshapes; i++)
    {
      for(int j=0; j<nbsattr; j++)
      {
        char *name = new char[30];
        sprintf_s(name, 30, "bs_%s", pMesh->layout.attribs[j].name);
        pLayoutDesc[layoutnum+i*nbsattr+j].SemanticName = name;
        pLayoutDesc[layoutnum+i*nbsattr+j].SemanticIndex = i;//pMesh->layout.attribs[j].semanticIdx;
        pLayoutDesc[layoutnum+i*nbsattr+j].Format = pMesh->bsLayout.attribs[j].formatDX10;
        pLayoutDesc[layoutnum+i*nbsattr+j].InputSlot = slotnum;
        pLayoutDesc[layoutnum+i*nbsattr+j].AlignedByteOffset = pMesh->bsLayout.attribs[j].AlignedByteOffset;
        pLayoutDesc[layoutnum+i*nbsattr+j].InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
        pLayoutDesc[layoutnum+i*nbsattr+j].InstanceDataStepRate = 0; // ??
      }
      slotnum++;
    }
    numLayoutSlots = slotnum;
    return pLayoutDesc;
  }
/*----------------------------------------------------------------------------------*/ /**
Create the Imput Layout interface, depending on the technique
// TODO : to give the Pass number...
**/ //----------------------------------------------------------------------------------
  HRESULT createInputLayout(ID3D10Device* pd3dDevice, ID3D10EffectTechnique *m_Tech)
  {
    HRESULT hr;
    D3D10_PASS_DESC passDesc;
    if(!m_Tech)
      return S_FALSE;
    if(pLayout) SAFE_RELEASE(pLayout);
    passDesc.Name = NULL; // Because of a DX10 Bug
      hr = m_Tech->GetPassByIndex(0)->GetDesc(&passDesc);
    if(!passDesc.Name) // DX10 Bug issue again
    {
      LOGMSG(LOG_ERR, L"failed to create Layout for %S", pMesh->meshName);
      return S_FALSE;
    }
    if( FAILED(hr = pd3dDevice->CreateInputLayout( pLayoutDesc, numLayoutAttribs, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &pLayout ) ) )
    {
      LOGMSG(LOG_ERR, L"CreateInputLayout() failed." );
    }
    return hr;
  }
/*----------------------------------------------------------------------------------*/ /**
Create Misc vertex buffers

The buffers can be either used as 
- typical vertex buffers
- resource view so we can bind them to a Buffer<> templated variable
This choice is available either for the main mesh and the blendshapes
Note: DX10 doesn't want to use the same buffer for the 2 usages :(... It complains about incompatible formats...
So we have to create separate buffers.
**/ //----------------------------------------------------------------------------------
  HRESULT createVertexBuffers(ID3D10Device* pd3dDevice, 
      bool bAsVtxBuffer=true, 
      bool bAsResource=false, 
      bool bBSAsVtxBuffer=true, 
      bool bBSAsResource=true, 
      bool bVerbose = false)
  {
    HRESULT hr = S_OK;
    //
    // Free data before reallocating them
    //
    if(bsResource.pVertexResource)  SAFE_RELEASE(bsResource.pVertexResource);
    if(bsResource.pVertexView)      SAFE_RELEASE(bsResource.pVertexView);
    if(pVertexBuffers) 
    {
        for(int i=0; i<numVertexBuffers; i++)
          if(pVertexBuffers[i]) SAFE_RELEASE(pVertexBuffers[i]);
        delete [] pVertexBuffers;
    }
    numVertexBuffers = pMesh->numSlots + pMesh->numBlendShapes;
    pVertexBuffers = new ID3D10Buffer*[numVertexBuffers];
    ZeroMemory(pVertexBuffers, sizeof(ID3D10Buffer*)*numVertexBuffers);
    if(pStrides) delete [] pStrides;
    pStrides = new UINT[numVertexBuffers];

    //#pragma message("TODO: buffers for the main shape - Not needed for this sample")
    if(bAsResource) LOGMSG(LOG_ERR, L"Main mesh 'as a resource' not working yet..." );
    //
    // basic Shapes
    //
    for(unsigned int i=0; i<pMesh->numSlots; i++)
    {
        D3D10_SUBRESOURCE_DATA data;
        data.SysMemPitch = 0;
        data.SysMemSlicePitch = 0;
        data.pSysMem = pMesh->slots[i].pVtxBufferData;
        //
        // Note : here we also use D3D10_BIND_SHADER_RESOURCE so we can
        // bind this buffer as a shader resource, too : To a Buffer<float3>, for example
        // This is used in the mode using buffers instead of Stream out.
        //
        D3D10_BUFFER_DESC bufferDescMesh =
        {
            pMesh->slots[i].vtxBufferSizeBytes,
            D3D10_USAGE_DYNAMIC,
            D3D10_BIND_VERTEX_BUFFER,
            D3D10_CPU_ACCESS_WRITE,
            0
        };
        pStrides[i] = pMesh->slots[i].vtxBufferStrideBytes;
        if(bAsVtxBuffer)
        {
            hr = pd3dDevice->CreateBuffer( &bufferDescMesh, &data, pVertexBuffers + i);
            if( FAILED( hr ) )
            {
                LOGMSG(LOG_ERR, L"Failed creating vertex buffer" );
                return hr;
            }
            if(bVerbose) LOGMSG(LOG_MSG, L"Created vertex buffer %d of %d bytes", i, pMesh->slots[i].vtxBufferSizeBytes );
        }
    }
    //
    // Blend-shapes
    //
    for(unsigned int i=0; i<pMesh->numBlendShapes; i++)
    {
        D3D10_SUBRESOURCE_DATA data;
        data.SysMemPitch = 0;
        data.SysMemSlicePitch = 0;
        data.pSysMem = pMesh->bsSlots[i].pVtxBufferData;
        D3D10_BUFFER_DESC bufferDescMesh =
        {
            pMesh->bsSlots[i].vtxBufferSizeBytes,
            D3D10_USAGE_DYNAMIC,
            D3D10_BIND_VERTEX_BUFFER,
            D3D10_CPU_ACCESS_WRITE,
            0
        };
        pStrides[i + pMesh->numSlots] = pMesh->bsSlots[i].vtxBufferStrideBytes;
        if(bBSAsVtxBuffer)
        {
            hr = pd3dDevice->CreateBuffer( &bufferDescMesh, &data, pVertexBuffers + i + pMesh->numSlots);
            if( FAILED( hr ) )
            {
                LOGMSG(LOG_ERR, L"Failed creating vertex buffer" );
                return hr;
            }
            if(bVerbose) LOGMSG(LOG_MSG, L"Created Blendshape vertex buffer %d of %d bytes", i, pMesh->bsSlots[i].vtxBufferSizeBytes );
        }
    }
    //
    // Create resource and view for the Blendshapes
    //
    if(bBSAsResource && (pMesh->numBlendShapes > 0))
    {
        // all BS slots are the same (using bsSlots[0] for all)
        unsigned int sizeBytes = pMesh->numBlendShapes * pMesh->bsSlots[0].vtxBufferSizeBytes;
        D3D10_BUFFER_DESC bufferDescMesh =
        {
            sizeBytes,
            D3D10_USAGE_IMMUTABLE,
            D3D10_BIND_SHADER_RESOURCE,
            0,
            0
        };
        D3D10_SUBRESOURCE_DATA data;
        data.SysMemPitch = 0;
        data.SysMemSlicePitch = 0;
        //
        // The mesh exposes the Blendshapes one after each other in memory.
        // So we know that all BS are available starting pMesh->bsSlots[0].pVtxBufferData
        //
        data.pSysMem = pMesh->bsSlots[0].pVtxBufferData;
        hr = pd3dDevice->CreateBuffer( &bufferDescMesh, &data, &(bsResource.pVertexResource) );
        if( FAILED( hr ) )
        {
            LOGMSG(LOG_ERR, L"Failed creating BS Shader resource" );
            return hr;
        }
        if(bVerbose) LOGMSG(LOG_MSG, L"Created BS shader resource of %d bytes", sizeBytes );
        //
        // WARNING: for now, attributes can only be float3
        // TODO: find a way to have a more complex description of the Buffer
        // in the shader, the buffer is Buffer<float3>... need Buffer<VSAttribsStruct>
        //
        D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
        SRVDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
        SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.ElementOffset = 0;
        SRVDesc.Buffer.ElementWidth = // all BS slots are the same (using bsSlots[0] for all)
            pMesh->numBlendShapes * pMesh->bsSlots[0].vertexCount * (pMesh->bsSlots[0].vtxBufferStrideBytes/(3*sizeof(float)));
        hr = pd3dDevice->CreateShaderResourceView( bsResource.pVertexResource, &SRVDesc, &(bsResource.pVertexView) );
        if( FAILED( hr ) )
        {
            LOGMSG(LOG_ERR, L"Failed creating resource view for BS" );
            return hr;
        }
        else if(bVerbose) LOGMSG(LOG_MSG, L"Created BS resource view");
    }
    return hr;
  }
/*----------------------------------------------------------------------------------*/ /**
    Create buffers for Weights and offsets of Blendshapes
**/ //----------------------------------------------------------------------------------
  HRESULT createBSOffsetsAndWeightBuffers(ID3D10Device* pd3dDevice, bool bVerbose = false)
  {
    HRESULT hr = S_OK;
    if(pMesh->numBlendShapes == 0)
        return S_OK;
    if(bsResource.pOffsetsResource) SAFE_RELEASE(bsResource.pOffsetsResource);
    if(bsResource.pOffsetsView)     SAFE_RELEASE(bsResource.pOffsetsView);
    if(bsResource.pWeightsResource) SAFE_RELEASE(bsResource.pWeightsResource);
    if(bsResource.pWeightsView)     SAFE_RELEASE(bsResource.pWeightsView);
    bsResource.numUsedBS = 0;
    //
    // Weights
    //
    D3D10_BUFFER_DESC bufferDescMesh =
    {
        pMesh->numBlendShapes * sizeof(float),
        D3D10_USAGE_DYNAMIC,
        D3D10_BIND_SHADER_RESOURCE,
        D3D10_CPU_ACCESS_WRITE,
        0
    };
    hr = pd3dDevice->CreateBuffer( &bufferDescMesh, NULL, &(bsResource.pWeightsResource));
    if( FAILED( hr ) )
    {
        LOGMSG(LOG_ERR, L"Failed creating BS Weight buffer resource" );
        return hr;
    }
    if(bVerbose) LOGMSG(LOG_MSG, L"Created Weight buffer of %d bytes", pMesh->numBlendShapes * sizeof(float) );
    D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
    ZeroMemory( &SRVDesc, sizeof(SRVDesc) );
    SRVDesc.Format = DXGI_FORMAT_R32_FLOAT;
    SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_BUFFER;
    SRVDesc.Buffer.ElementOffset = 0;
    SRVDesc.Buffer.ElementWidth = pMesh->numBlendShapes;
    hr = pd3dDevice->CreateShaderResourceView( bsResource.pWeightsResource, &SRVDesc, &(bsResource.pWeightsView) );
    if( FAILED( hr ) )
    {
        LOGMSG(LOG_ERR, L"Failed creating BS Weight resource view" );
        return hr;
    }
    else if(bVerbose) LOGMSG(LOG_MSG, L"Created Weight buffer resource view");
    //
    // Offsets
    //
    bufferDescMesh.ByteWidth = pMesh->numBlendShapes * sizeof(UINT);
    SRVDesc.Format = DXGI_FORMAT_R32_UINT;
    hr = pd3dDevice->CreateBuffer( &bufferDescMesh, NULL, &(bsResource.pOffsetsResource));
    if( FAILED( hr ) )
    {
        LOGMSG(LOG_ERR, L"Failed creating BS Weight buffer resource" );
        return hr;
    }
    if(bVerbose) LOGMSG(LOG_MSG, L"Created Weight buffer of %d bytes", pMesh->numBlendShapes * sizeof(UINT) );
    hr = pd3dDevice->CreateShaderResourceView( bsResource.pOffsetsResource, &SRVDesc, &(bsResource.pOffsetsView) );
    if( FAILED( hr ) )
    {
        LOGMSG(LOG_ERR, L"Failed creating BS Weight resource view" );
        return hr;
    }
    else if(bVerbose) LOGMSG(LOG_MSG, L"Created Weight buffer resource view");
    return hr;
  }
/*----------------------------------------------------------------------------------*/ /**
Create misc index buffers
**/ //----------------------------------------------------------------------------------
  HRESULT createIndexBuffers(ID3D10Device* pd3dDevice, bool bVerbose = false)
  {
    HRESULT hr = S_OK;
    for(int i=0; i<numIndexBuffers; i++)
    {
      if(bVerbose) LOGMSG(LOG_MSG, L"releasing index buffer %d...", i);
      if(pIndexBuffers[i]) SAFE_RELEASE(pIndexBuffers[i]);
    }
    if(pIndexBuffers) delete [] pIndexBuffers;
    numIndexBuffers = pMesh->numPrimGroups;
    pIndexBuffers = new ID3D10Buffer*[numIndexBuffers];

    for(int i=0; i<pMesh->numPrimGroups; i++)
    {
        D3D10_BUFFER_DESC bufferDescMesh =
        {
            pMesh->primGroup[i].indexArrayByteSize,
            D3D10_USAGE_DYNAMIC,
            D3D10_BIND_INDEX_BUFFER,
            D3D10_CPU_ACCESS_WRITE,
            0
        };
      D3D10_SUBRESOURCE_DATA data;
      data.SysMemPitch = 0;
      data.SysMemSlicePitch = 0;
      data.pSysMem = pMesh->primGroup[i].pIndexBufferData;
        if( FAILED( pd3dDevice->CreateBuffer( &bufferDescMesh, &data, pIndexBuffers + i ) ) )
        {
            LOGMSG(LOG_ERR, L"Failed creating index buffer" );
            return false;
        }
      if(bVerbose) LOGMSG(LOG_MSG, L"Created index buffer %S of %d bytes", pMesh->primGroup[i].name, pMesh->primGroup[i].indexArrayByteSize );
    }
    return hr;
  }
/*----------------------------------------------------------------------------------*/ /**
Create Vertex Stream out buffers. These are for Blendshape passes : same size
**/ //----------------------------------------------------------------------------------
  HRESULT createStreamOutBuffers(ID3D10Device* pd3dDevice, int numSOBuffers, int slot, bool bVerbose = false)
  {
    HRESULT hr = S_OK;
    for(int i=0; i<numStreamOutBuffers; i++)
    {
      if(bVerbose) LOGMSG(LOG_MSG, L"releasing StreamOut buffer %d...", i);
      if(pStreamOutBuffers[i]) SAFE_RELEASE(pStreamOutBuffers[i]);
    }
    if(pStreamOutBuffers) delete [] pStreamOutBuffers;
    numStreamOutBuffers = numSOBuffers;
    pStreamOutBuffers = new ID3D10Buffer*[numStreamOutBuffers];

    for(int i=0; i<numStreamOutBuffers; i++)
    {
      D3D10_BUFFER_DESC bufferDescMesh =
      {
          pMesh->slots[slot].vtxBufferSizeBytes,
          D3D10_USAGE_DEFAULT,
          D3D10_BIND_VERTEX_BUFFER|D3D10_BIND_STREAM_OUTPUT,
          0,
          0
      };
      hr = pd3dDevice->CreateBuffer( &bufferDescMesh, NULL, pStreamOutBuffers + i );
      if( FAILED( hr ) )
      {
          LOGMSG(LOG_ERR, L"Failed creating vertex buffer" );
        return hr;
      }
    }
    return hr;
  }
};
//#endif