//----------------------------------------------------------------------------------
// File:   NVRawMesh.h
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

#ifndef NOGZLIB
#include "zlib/zlib.h"
#define GFILE gzFile
#define GOPEN gzopen
#define GREAD(a,b,c) gzread(a, b, c)
#define GCLOSE gzclose
#else
#define GFILE FILE *
#define GOPEN fopen
#define GREAD(a,b,c) (int)fread(b, 1, c, a)
#define GCLOSE fclose
#endif

#pragma warning(disable: 4505)

  #ifndef LOGMSG
  #	define LOG_MSG stdout
  #	define LOG_WARN stdout
  #	define LOG_YES stdout
  #	define LOG_NO stdout
  #	define LOG_NOTE stdout
  #	define LOG_ERR stderr
  #define LOGMSG fprintf
  #endif
  #ifndef CONSOLEMSG
  # define CONSOLEMSG LOGMSG
  #endif
  #define RAWMESHVERSION 0x105

  //--------------------------------
  // 
  //--------------------------------
#ifndef _d3d9TYPES_H_
#pragma message("defining D3DPRIMITIVETYPE here...")
  enum D3DPRIMITIVETYPE
  {
    D3DPT_UNDEFINED             = 0,
    D3DPT_POINTLIST             = 1,
    D3DPT_LINELIST              = 2,
    D3DPT_LINESTRIP             = 3,
    D3DPT_TRIANGLELIST          = 4,
    D3DPT_TRIANGLESTRIP         = 5,
    D3DPT_TRIANGLEFAN           = 6,
};
#pragma message("defining D3DFORMAT here...")
  enum D3DFORMAT
  {
    D3DFMT_INDEX16              = 101,
    D3DFMT_INDEX32              = 102,
  };
#pragma message("defining D3DDECLTYPE here...")
  enum D3DDECLTYPE
  {
    D3DDECLTYPE_FLOAT1    =  0,  // 1D float expanded to (value, 0., 0., 1.)
    D3DDECLTYPE_FLOAT2    =  1,  // 2D float expanded to (value, value, 0., 1.)
    D3DDECLTYPE_FLOAT3    =  2,  // 3D float expanded to (value, value, value, 1.)
    D3DDECLTYPE_FLOAT4    =  3,  // 4D float
    D3DDECLTYPE_D3DCOLOR  =  4,  // 4D packed unsigned bytes mapped to 0. to 1. range
    D3DDECLTYPE_UNDEF     =  -1,
  };
#endif
#ifndef __d3d10_h__
#pragma message("defining D3D10_PRIMITIVE_TOPOLOGY enum...")
  enum D3D10_PRIMITIVE_TOPOLOGY
  {	
      D3D10_PRIMITIVE_TOPOLOGY_UNDEFINED	        = 0,
	  D3D10_PRIMITIVE_TOPOLOGY_POINTLIST	        = 1,
	  D3D10_PRIMITIVE_TOPOLOGY_LINELIST	            = 2,
	  D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP	        = 3,
	  D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST	        = 4,
	  D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP	    = 5,
	  D3D10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ	        = 10,
	  D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ	    = 11,
	  D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ	    = 12,
	  D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ	= 13,
	  //D3D10_PRIMITIVE_TOPOLOGY_FAN	            = 14                // Doesn't exist in DX10...
  };
#pragma message("defining DXGI_FORMAT enum...")
  enum DXGI_FORMAT // stick to DX10 values
  {
    FORMAT_UNDEF                = 0,
    FORMAT_R32_FLOAT	        = 41,
    FORMAT_R32G32_FLOAT	        = 16,
    FORMAT_R32G32B32_FLOAT	    = 6,
    FORMAT_R32G32B32A32_FLOAT	= 2,
    FORMAT_R16_UINT	            = 57,
    FORMAT_R32_UINT	            = 42
  };
#endif
//-------------------------------------------------
// OpenGL enums...
//-------------------------------------------------
#ifndef __gl_h_
  typedef unsigned int GLenum;
  typedef GLenum GLTopology;
  //enum GLTopology // turn GL enums in real enums ?
  //{
#define    GL_POINTS                         =0x0000,
#define    GL_LINES                          =0x0001,
#define    GL_LINE_LOOP                      =0x0002,
#define    GL_LINE_STRIP                     =0x0003,
#define    GL_TRIANGLES                      =0x0004,
#define    GL_TRIANGLE_STRIP                 =0x0005,
#define    GL_TRIANGLE_FAN                   =0x0006,
#define    GL_QUADS                          =0x0007,
#define    GL_QUAD_STRIP                     =0x0008
  //};
  typedef GLenum GLType;
  //enum GLType
  //{
    // enums from OpenGL so that we are directly ready
#define    GL_BYTE                           =0x1400,
#define    GL_UNSIGNED_BYTE                  =0x1401,
#define    GL_SHORT                          =0x1402,
#define    GL_UNSIGNED_SHORT                 =0x1403,
#define    GL_INT                            =0x1404,
#define    GL_UNSIGNED_INT                   =0x1405,
#define    GL_FLOAT                          =0x1406,
#define    GL_2_BYTES                        =0x1407,
#define    GL_3_BYTES                        =0x1408,
#define    GL_4_BYTES                        =0x1409,
#define    GL_DOUBLE                         =0x140A
  //};
#endif
//-------------------------------------------------
// 
//-------------------------------------------------

namespace NVRawMesh
{
  struct Attribute
  {
    D3DDECLTYPE         formatDX9;
    DXGI_FORMAT         formatDX10;
    GLType              formatGL;               // doesn't say how many components. See numComp.
    unsigned char       semanticIdx;
    unsigned char       numComp;
//TODO: no need for this slot info since Attribute will belong to the slot
    unsigned char       slot;
    unsigned int        strideBytes;            // strideBytes info for OpenGL
    unsigned int        AlignedByteOffset;      // DirectX10 needs it in the Layout definition : offset in the layout.
    unsigned int        dataOffsetBytes;        // for example, use this for VertexPointer() in VBO or what we do with offsetof(VertexStructure, p)...
    void                *pAttributeBufferData;  // this pointer will have to be resolved after load operation
    unsigned int        nameOffset;             // must be pointing to namebuffer
    char                name[16];               // could be somewhere else
  };

  //--------------------------------
  // 
  //--------------------------------
  // TODO : remove it since we may want to put the info in Slot
  struct Layout
  {
    int           num_attribs;
    Attribute     attribs[16]; // Max = 16 = D3D10_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT
  };

  //--------------------------------
  // 
  // a slot/stream is made of interleaved attribs. In the easiest case it has only one attr.
  // Note that this slotDataOffsets/pVtxBufferData may contain a subset of vertices instead of the whole : if the slot
  // is being used on a smaller part of the mesh, no need to provide all.
  // TODO : indices to shift.
  //--------------------------------
  struct Slot 
  {
    unsigned int        vtxBufferSizeBytes;     // size of the vertex buffer, in bytes
    unsigned int        vtxBufferStrideBytes;   // strideBytes of the vertex buffer
    unsigned int        vertexCount;            // [DX9 wants it] Num used vertices
    unsigned int        slotDataOffsets;
    void                *pVtxBufferData;     // this pointer will have to be resolved after load operation
    char                name[32];               // for example, use in blendshapes : a slot is a blendshape. Good to have its name
    // TODO : layout infos
    // unsigned int     numAttribs;
    // Attribute        attribs[16];
  };

/*--------------------------------
Primitive group
TO CHECK:
* can be created after stripifier process : 
one triangle list primitive group is split to many tristrip primitive group
NOT sure : seems it will always endup with 1 single tristrip (+degenerated tris)
QUESTION ? Do we want to add primitive restart
* one primitive group for one 'shader' : vertices associated with a shader will be referenced by a primitive group
QUESTION : one primgroup will use a specific set of streams. Give them.
simplest case : 1 stream for 1 primgroup. However we can have many streams for 1 primgroup
* a primitive group can work only with a smaller set of vertices.
--------------------------------*/
  struct PrimGroup
  {
    void init()
    {
      memset((void*)this, 0, sizeof(PrimGroup));
    }
    char name[64];
    //TODO : unsigned int       numSlots;
    // use an array of offsets + ptr to resolve, instead ?
    //TODO : unsigned int       slotReferences[16]; ///< references to used slots
    unsigned int                indexCount;             // total number of elements
    unsigned int                minIndex;               // min element index
    unsigned int                maxIndex;               // max element index
    unsigned int                indexArrayByteOffset;   // offset to reach the element array from &data[1]
    unsigned int                indexArrayByteSize;     // total size in bytes
    unsigned int                primitiveCount;         // DX9 wants it
    D3DFORMAT                   indexFormatDX9;
    DXGI_FORMAT                 indexFormatDX10;
    GLType                      indexFormatGL;
    D3DPRIMITIVETYPE            topologyDX9;
    D3D10_PRIMITIVE_TOPOLOGY    topologyDX10;
    GLTopology                  topologyGL;
    void                        *pIndexBufferData;       // this pointer will have to be resolved after load operation
  };
  //------------------------------------------------------------------------------------------
  // FileHeader FileHeader FileHeader FileHeader FileHeader FileHeader FileHeader
  //------------------------------------------------------------------------------------------
  struct FileHeader
  {
    unsigned int  magic;                // magic unsigned int  "MESH". Just to make sure this is a correct file. \todo version checking ?
    unsigned int  version;              // magic unsigned int  "MESH". Just to make sure this is a correct file. \todo version checking ?
    unsigned int  size;                 // size of the header + data following it.

    char          meshName[64];         // name of the mesh

    Layout        layout;               // layout (in DX10 sense), or Vertex Format + streams definition

    unsigned int  numSlots;
    Slot          slots[16];            // or called STREAM. Vertex Buffers for each slot/layer/stream
    Layout        bsLayout;             // layout (in DX10 sense) for the Blendshapes
    unsigned int  numBlendShapes;
    unsigned int  blendShapesOffset;    // offset where to find the numBlendShapes Slot structs, starting from rawdata
    Slot          *bsSlots;             // to be resolved with blendShapesOffset+rawdata

    int           numPrimGroups;
    PrimGroup     primGroup[1];         // primitive group structures is always at the begining of the data chunk.

    //------------>> Data starting here : Vertex Buffers and Index Buffers
    char          rawdata[1];           // Raw access to the data. Used to compute the offset

    //------------------------------------------------------------------------------------------
    // METHODS METHODS METHODS METHODS
    //------------------------------------------------------------------------------------------
    FileHeader() {init();}
    void init()
    {
      strncpy_s((char*)&magic, 4, "MESH",4);
      version = RAWMESHVERSION;
      size = sizeof(FileHeader);
      numPrimGroups = 0;
      numSlots = 0;
      numBlendShapes = 0;
      blendShapesOffset = 0;
      bsSlots = NULL;
      primGroup[0].init();
    }
    void resolvePointers()
    {
      int i;
      bsSlots = (Slot*)(rawdata + blendShapesOffset);
      for(i=0; i<numPrimGroups; i++)
        primGroup[i].pIndexBufferData = rawdata + primGroup[i].indexArrayByteOffset;
      for(i=0; i<(int)numSlots; i++)
        slots[i].pVtxBufferData = rawdata + slots[i].slotDataOffsets;
      for(i=0; i<(int)numBlendShapes; i++)
        bsSlots[i].pVtxBufferData = rawdata + bsSlots[i].slotDataOffsets;
      for(i=0; i<layout.num_attribs; i++)
      {
        layout.attribs[i].pAttributeBufferData =  rawdata + layout.attribs[i].dataOffsetBytes;
        //TODO : change name[16]
        //layout.attribs[i].name = rawdata + layout.attribs[i].nameOffset;
      }
    }
    void debugDumpLayout()
    {
      CONSOLEMSG(LOG_MSG, L"\nMesh Infos : %S\nAttributes:", meshName);
      for(int i=0; i<layout.num_attribs; i++)
      {
        CONSOLEMSG(LOG_MSG, L"%S : slot %d, strideBytes %d, format %d", layout.attribs[i].name, layout.attribs[i].slot, layout.attribs[i].strideBytes, layout.attribs[i].formatDX10);
      }
	  CONSOLEMSG(LOG_MSG, L"\n%d Primitive groups : ", numPrimGroups);
      for(int i=0; i<numPrimGroups; i++)
      {
        CONSOLEMSG(LOG_MSG, L"\nPrim group %d : %S", i, primGroup[i].name);
          CONSOLEMSG(LOG_MSG, L"\tnumelements = %d", primGroup[i].indexCount);
          CONSOLEMSG(LOG_MSG, L"\tminelement = %d", primGroup[i].minIndex);
          CONSOLEMSG(LOG_MSG, L"\tmaxelement = %d", primGroup[i].maxIndex);
          CONSOLEMSG(LOG_MSG, L"\tprimitiveCount = %d", primGroup[i].primitiveCount);
          CONSOLEMSG(LOG_MSG, L"\tPrimitive type = ");
          switch(primGroup[i].topologyDX10)
          {
          case D3D10_PRIMITIVE_TOPOLOGY_UNDEFINED: CONSOLEMSG(LOG_MSG, L"UNDEFINED"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_POINTLIST:CONSOLEMSG(LOG_MSG, L"POINT LIST"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_LINELIST:CONSOLEMSG(LOG_MSG, L"LINE LIST"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP: CONSOLEMSG(LOG_MSG, L"LINE STRIP"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST: CONSOLEMSG(LOG_MSG, L"TRI LIST"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: CONSOLEMSG(LOG_MSG, L"TRI STRIP"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ: CONSOLEMSG(LOG_MSG, L"ADJ LINE LIST"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ: CONSOLEMSG(LOG_MSG, L"ADJ LINE STRIP"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ: CONSOLEMSG(LOG_MSG, L"ADJ TRIANGLE LIST"); break;
          case D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ: CONSOLEMSG(LOG_MSG, L"ADJ TRI STRIP"); break;
          //case D3D10_PRIMITIVE_TOPOLOGY_FAN: CONSOLEMSG(LOG_MSG, L"FAN. WARNING Not available in DX10"); break;
          }
      }
      if(numBlendShapes > 0)
      {
        CONSOLEMSG(LOG_MSG, L"\n%d Blendshapes : ", numBlendShapes);
        CONSOLEMSG(LOG_MSG, L"Attributes in the Blendshapes : ", numBlendShapes);
        for(int i=0; i<bsLayout.num_attribs; i++)
        {
          CONSOLEMSG(LOG_MSG, L"%S : slot %d, strideBytes %d, format %d", bsLayout.attribs[i].name, bsLayout.attribs[i].slot, bsLayout.attribs[i].strideBytes, bsLayout.attribs[i].formatDX10);
        }
        CONSOLEMSG(LOG_MSG, L"\nBlendshape list:");
        for(int i=0; i<(int)numBlendShapes; i++)
        {
          CONSOLEMSG(LOG_MSG, L"Blendshape %d : %S", i, bsSlots[i].name);
        }
      }
      CONSOLEMSG(LOG_MSG, L"-------------------------------------------------------");
    }
  };

  //--------------------------------
  // HELPERS HELPERS HELPERS HELPERS
  //--------------------------------
  static FileHeader * loadMesh(LPCSTR fname)
  {
    GFILE fd = NULL;
    fd = GOPEN(fname, "rb");
    if(!fd)
    {
      LOGMSG(LOG_ERR, L"D3D10RawMesh: couldn't load %S", fname);
      CONSOLEMSG(LOG_MSG, L"Error : D3D10RawMesh: couldn't load %S", fname);
	    return NULL;
    }
#   define RAWMESHMINSZ (1024*1000)
    char * memory = (char*)malloc(RAWMESHMINSZ);
    int offs = 0;
    int n = 0;
    do {
      if(n > 0)
      {
	      offs += RAWMESHMINSZ;
	      memory = (char*)realloc(memory, RAWMESHMINSZ + offs);
      }
      if(fd)
        n= GREAD(fd, memory + offs, RAWMESHMINSZ);
    } while(n == RAWMESHMINSZ);
    if(n > 0)
    {
      offs -= RAWMESHMINSZ-n;
      memory = (char*)realloc(memory, RAWMESHMINSZ + offs);
    }
    if(fd)
        GCLOSE(fd);
    if(strncmp(memory, "MESH", 4))
    {
      LOGMSG(LOG_ERR, L"Not a mesh file");
      CONSOLEMSG(LOG_MSG, L"Error : Not a mesh file");
	    free(memory);
	    return false;
    }
    if(((FileHeader *)memory)->version != RAWMESHVERSION)
    {
      LOGMSG(LOG_ERR, L"Wrong version in Mesh description");
      CONSOLEMSG(LOG_MSG, L"Error : Wrong version in Mesh description");
      free(memory);
      return NULL;
    }
    ((FileHeader *)memory)->resolvePointers();
    return (FileHeader *)memory;
  }

  static FileHeader *releaseBufferData(FileHeader *p)
  {
    int i;
    for(i=0; i<p->numPrimGroups; i++)
      p->primGroup[i].pIndexBufferData = NULL;
    for(i=0; i<(int)p->numSlots; i++)
      p->slots[i].pVtxBufferData = NULL;
    for(i=0; i<p->layout.num_attribs; i++)
    {
      p->layout.attribs[i].pAttributeBufferData =  NULL;
    }
    //////////////
//#pragma message ("WRONG : keep primitive groups !!!")
    return (FileHeader *)realloc(p, sizeof(FileHeader));
  }

} // namespace NVRawMesh