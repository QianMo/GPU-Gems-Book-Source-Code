#include "LWOFile.h"

#include <sstream>
#define WINVER 0x0500
#include <windows.h> // for file-access
#include <float.h>
#include <math.h>

// convert basic types to a string
template<class Type>
inline std::string ConvertToString(const Type &t)
{
  std::ostringstream oss;
  oss<<t;
  return oss.str();
}

// creates a chunk id from a 4 byte string
inline unsigned int ID4(const char *pChunkID)
{
  unsigned int iChunk=*(unsigned int *)pChunkID;
  return iChunk;
}

// convert ID4 type to string
inline std::string ID4ToString(unsigned int iID)
{
  char pString[5];
  *(unsigned int*)pString=iID;
  pString[4]=0;
  return std::string(pString);
}

// change byte order on a 4 byte variable
template<class Type>
inline void Flip4Bytes(Type &t)
{
  char iTempByte=((char *)&t)[0];
  ((char *)&t)[0]=((char *)&t)[3];
  ((char *)&t)[3]=iTempByte;

  iTempByte=((char *)&t)[1];
  ((char *)&t)[1]=((char *)&t)[2];
  ((char *)&t)[2]=iTempByte;
}

// change byte order on a 2 byte variable
template<class Type>
inline void Flip2Bytes(Type &t)
{
  char iTempByte=((char *)&t)[0];
  ((char *)&t)[0]=((char *)&t)[1];
  ((char *)&t)[1]=iTempByte;
}

inline void LWOFile::AddError(const std::string &strError)
{
  m_Errors.push_back(strError);
}

// checks for buffer overflow
inline bool LWOFile::SafeToRead(unsigned int iBytes)
{
  return m_pData+iBytes<=m_pDataEnd;
}

inline bool LWOFile::ReadID4(unsigned int &iVar)
{
  if(!SafeToRead(4)) return false;
  iVar=*(unsigned int *)m_pData;
  m_pData+=4;
  return true;
}

inline bool LWOFile::ReadU4(unsigned int &iVar)
{
  if(!SafeToRead(4)) return false;
  iVar=*(unsigned int *)m_pData;
  Flip4Bytes(iVar);
  m_pData+=4;
  return true;
}

inline bool LWOFile::ReadU2(unsigned short &iVar)
{
  if(!SafeToRead(2)) return false;
  iVar=*(unsigned short *)m_pData;
  Flip2Bytes(iVar);
  m_pData+=2;
  return true;
}

inline bool LWOFile::ReadVX(unsigned int &iVar)
{
  if(!SafeToRead(2)) return false;

  // quote: When reading an index, if the first byte encountered
  //        is 255 (0xFF), then the four-byte form is being used
  //        and the first byte should be discarded or masked out.
  if(((unsigned char *)m_pData)[0]==0xFF)
  {
    if(!SafeToRead(4)) return false;

    ((char *)&iVar)[0]=m_pData[3];
    ((char *)&iVar)[1]=m_pData[2];
    ((char *)&iVar)[2]=m_pData[1];
    ((char *)&iVar)[3]=0;

    m_pData+=4;

  } else {

    ((char *)&iVar)[0]=m_pData[1];
    ((char *)&iVar)[1]=m_pData[0];
    ((char *)&iVar)[2]=0;
    ((char *)&iVar)[3]=0;

    m_pData+=2;
  }

  return true;
}

inline bool LWOFile::ReadF4(float &fVar)
{
  if(!SafeToRead(4)) return false;
  fVar=*(float *)m_pData;
  Flip4Bytes(fVar);
  m_pData+=4;
  return true;
}

inline bool LWOFile::ReadS0(std::string &strVar)
{
  char pChars[3]={0,0,0};

  // thanks to padding we can read two
  // chars at a time until we hit a zero
  while(true)
  {
    if(!SafeToRead(2)) return false;
    *(short*)pChars=*(short*)m_pData;
    m_pData+=2;

    strVar+=pChars;
    if(pChars[0]==0 || pChars[1]==0) break;
  }

  return true;
}

inline bool LWOFile::ReadS0(std::string &strVar, unsigned int &iBytesRead)
{
  char pChars[3]={0,0,0};
  iBytesRead=0;

  // thanks to padding we can read two
  // chars at a time until we hit a zero
  while(true)
  {
    if(!SafeToRead(2)) return false;
    *(short*)pChars=*(short*)m_pData;
    m_pData+=2;
    iBytesRead+=2;

    strVar+=pChars;
    if(pChars[0]==0 || pChars[1]==0) break;
  }

  return true;
}

inline bool LWOFile::ReadAndFlip4(void *pData, unsigned int iLength)
{
  if(!SafeToRead(iLength)) return false;

  // flip 4 byte slots
  for(unsigned int i=0;i<iLength;i+=4)
  {
    ((unsigned char *)pData)[i+0]=m_pData[i+3];
    ((unsigned char *)pData)[i+1]=m_pData[i+2];
    ((unsigned char *)pData)[i+2]=m_pData[i+1];
    ((unsigned char *)pData)[i+3]=m_pData[i+0];
  }
  m_pData+=iLength;

  return true;
}

inline bool LWOFile::Read(void *pData, unsigned int iLength)
{
  if(!SafeToRead(iLength)) return false;
  memcpy(pData, m_pData, iLength);
  m_pData+=iLength;
  return true;
}


LWOFile::LWOFile()
{
}

LWOFile::~LWOFile()
{
  {std::vector<Layer *>::iterator it;
  for(it=m_Layers.begin();it!=m_Layers.end();it=m_Layers.erase(it))
  {
    delete (*it);
  }}
}

bool LWOFile::LoadFromFile(const char *strFile)
{
  // try to open file
  HANDLE hFile=CreateFileA(strFile, GENERIC_READ, FILE_SHARE_READ, NULL,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if(hFile==INVALID_HANDLE_VALUE)
  {
    AddError("Could not open file: " + std::string(strFile));
    return false;
  }

  // get file size
  LARGE_INTEGER LI_Size;
  if(!GetFileSizeEx(hFile,&LI_Size))
  {
    AddError("Could not get file size!");
    return false;
  }
  unsigned int iFileSize=(unsigned int)LI_Size.QuadPart;

  // allocate memory to hold the file
  char *pData=new char[iFileSize];

  // read file to memory
  DWORD iBytesRead=0;
  if(!ReadFile(hFile,pData,iFileSize,&iBytesRead,NULL) || iBytesRead!=iFileSize)
  {
    AddError("Reading file failed!");
    delete[] pData;
    CloseHandle(hFile);
    return false;
  }

  // load
  if(!LoadFromMemory(pData, iFileSize)) return false;

  delete[] pData;
  CloseHandle(hFile);
  return true;
}

bool LWOFile::LoadFromMemory(const void *pData, unsigned int iFileSize)
{
  if(pData==NULL || iFileSize==0)
  {
    AddError("Loading from memory failed: no data");
    return false;
  }
  m_pData=(const char*)pData;
  m_pDataEnd=m_pData+iFileSize;

  // read main chunk id
  //
  unsigned int iID=0;
  if(!ReadID4(iID) || iID!=ID4("FORM"))
  {
    AddError("Main chunk FORM not found!");
    return false;
  }

  // read main chunk size
  //
  unsigned int iMainSize=0;
  if(!ReadU4(iMainSize)) return false;

  // read file format
  //
  if(!ReadID4(iID)) return false;
  if(iID!=ID4("LWO2"))
  {
    AddError("Format "+ID4ToString(iID)+" is not supported!");
    return false;
  }

  unsigned int iLayerID=ID4("LAYR");
  unsigned int iPointsID=ID4("PNTS");
  unsigned int iPolygonsID=ID4("POLS");
  unsigned int iVertexMapID=ID4("VMAP");
  unsigned int iVertexMapDID=ID4("VMAD");
  unsigned int iTagsID=ID4("TAGS");
  unsigned int iPTagsID=ID4("PTAG");
  unsigned int iSurfID=ID4("SURF");

  // while we have data left
  //
  while(m_pData<m_pDataEnd)
  {
    // get chunk ID
    //
    unsigned int iID=0;
    if(!ReadID4(iID)) return false;

    // get chunk size
    //
    unsigned int iChunkSize=0;
    if(!ReadU4(iChunkSize)) return false;


    // attempt to load chunk

    // LAYR
    ///////////////////
    if(iID==iLayerID)
    {
      if(!LoadLAYR(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // PNTS
    ///////////////////
    else if(iID==iPointsID)
    {
      if(!LoadPNTS(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // POLS
    ///////////////////
    else if(iID==iPolygonsID)
    {
      if(!LoadPOLS(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // VMAP
    ///////////////////
    else if(iID==iVertexMapID)
    {
      if(!LoadVMAP(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // VMAD
    ///////////////////
    else if(iID==iVertexMapDID)
    {
      if(!LoadVMAD(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // TAGS
    ///////////////////
    else if(iID==iTagsID)
    {
      if(!LoadTAGS(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // PTAG
    ///////////////////
    else if(iID==iPTagsID)
    {
      if(!LoadPTAG(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // SURF
    ///////////////////
    else if(iID==iSurfID)
    {
      if(!LoadSURF(iChunkSize))
      {
        AddError("Loading chunk "+ID4ToString(iID)+" failed!");
        return false;
      }
    }

    // unknown
    ///////////////////
    else
    {
      AddError("Warning: Unknown chunk ID "+ID4ToString(iID));
      m_pData+=iChunkSize;
    }


    // quote: If the chunk size is odd, the chunk is followed by a 0 pad byte,
    //        so that the next chunk begins on an even byte boundary.
    if(iChunkSize%2==1) m_pData+=1;
  }

  m_pData=NULL;
  m_pDataEnd=NULL;
  return true;
}


bool LWOFile::LoadLAYR(unsigned int iChunkSize)
{
  Layer *pLayer=new Layer();
  m_Layers.push_back(pLayer);

  // LAYR { number[U2], flags[U2], pivot[VEC12], name[S0], parent[U2] ? }
  const char *pChunkEndPos=m_pData+iChunkSize;
  if(!ReadU2(pLayer->m_iID)) return false;
  if(!ReadU2(pLayer->m_iFlags)) return false;
  if(!ReadF4(pLayer->m_vPivot[0])) return false;
  if(!ReadF4(pLayer->m_vPivot[1])) return false;
  if(!ReadF4(pLayer->m_vPivot[2])) return false;
  if(!ReadS0(pLayer->m_strName)) return false;

  // parent id is optional, so read it only if theres data left
  if(m_pData<pChunkEndPos)
  {
    if(!ReadU2(pLayer->m_iParentID)) return false;
  } else {
    pLayer->m_iParentID=0;
  }

  return true;
}

bool LWOFile::LoadPNTS(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();
  if(pLayer==NULL)
  {
    AddError("No LAYR before PNTS");
    return false;
  }

  if(pLayer->m_Points.size()>0)
  {
    AddError("PNTS already in layer");
    return false;
  }

  if(iChunkSize%12!=0)
  {
    AddError("PNTS size does not match to a VEC12 array");
    return false;
  }

  pLayer->m_iPoints=iChunkSize/12;
  pLayer->m_Points.resize(iChunkSize/4);

  // PNTS { point-location[VEC12] * }
  if(!ReadAndFlip4(&pLayer->m_Points[0],iChunkSize)) return false;
  return true;
}

bool LWOFile::LoadPOLS(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();
  if(pLayer==NULL)
  {
    AddError("No LAYR before POLS");
    return false;
  }

  const char *pChunkEndPos=m_pData+iChunkSize;

  // POLS { type[ID4], ( numvert+flags[U2], vert[VX] # numvert )* }
  unsigned int iType=0;
  if(!ReadID4(iType)) return false;

  // only faces supported (but patches control cage can be loaded as well)
  if(iType!=ID4("FACE") && iType!=ID4("PTCH"))
  {
    AddError("Warning: POLS has unsupported type "+ID4ToString(iType));
    // skip
    m_pData=pChunkEndPos;
    return true;
  }

  unsigned int iBytesLeft=iChunkSize-4;

  // allocate for worst case polygon count
  unsigned int iPolysNeeded=iBytesLeft/(2+2);
  unsigned int iExistingPolys=(unsigned int)pLayer->m_Polygons.size();
  pLayer->m_Polygons.resize(iExistingPolys+iPolysNeeded);

  unsigned int iPolygon=0;
  while(m_pData<pChunkEndPos)
  {
    // read vertex count
    unsigned short iVertexCount;
    if(!ReadU2(iVertexCount)) return false;

    // quote: The 6 high-order bits of the vertex count are flag
    //        bits with different meanings for each polygon type.
    //
    // let's just mask them out
    iVertexCount &= 0x03FF;

    pLayer->m_Polygons[iExistingPolys+iPolygon].m_iSurface=0xffff;
    pLayer->m_Polygons[iExistingPolys+iPolygon].m_Vertices.resize(iVertexCount);

    unsigned int *pVertices=&pLayer->m_Polygons[iExistingPolys+iPolygon].m_Vertices[0];
    for(unsigned short j=0;j<iVertexCount;j++)
    {
      if(!ReadVX(pVertices[j])) return false;
    }
    iPolygon++;
  }

  // make array size match the actual polygon count
  pLayer->m_Polygons.resize(iExistingPolys+iPolygon);

  return true;
}

bool LWOFile::LoadVMAD(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();
  if(pLayer==NULL)
  {
    AddError("No LAYR before VMAD");
    return false;
  }

  const char *pChunkEndPos=m_pData+iChunkSize;

  // VMAD { type[ID4], dimension[U2], name[S0],
  // ( vert[VX], poly[VX], value[F4] # dimension )* }
  unsigned int iType=0;
  if(!ReadID4(iType)) return false;

  // type must be uvmap
  if(iType!=ID4("TXUV"))
  {
    AddError("Warning: Unknown discontinuous vertex map type "+ID4ToString(iType));
    m_pData+=iChunkSize-4;
    return true;
  }

  // dimension
  unsigned short iDimension=0;
  if(!ReadU2(iDimension))
  {
    return false;
  }

  // not 2 floats per vertex
  if(iDimension!=2)
  {
    AddError("Warning: Discontinuous UVMap has "+ConvertToString(iDimension)+" floats per vertex (2 expected)");
    // skip
    m_pData+=iChunkSize-4-2;
    return true;
  }

  // name
  unsigned int iStrBytes=0;
  std::string strName;
  if(!ReadS0(strName,iStrBytes))
  {
    return false;
  }

  // for each uvmap
  UVMap *pUV=NULL;
  for(unsigned int iUV=0;iUV<pLayer->m_UVMaps.size();iUV++)
  {
    if(pLayer->m_UVMaps[iUV]->m_strName==strName)
    {
      pUV=pLayer->m_UVMaps[iUV];
    }
  }

  if(pUV==NULL)
  {
    AddError("No matching UVMap for discontinuous UVMap \""+strName+"\"");
    return false;
  }


  // read rest of chunk to memory
  //
  unsigned int iBytesLeft=iChunkSize-4-2-iStrBytes;

  UVMapD *pUVMap=new UVMapD();
  pUVMap->m_strName=strName;
  pUVMap->m_pUVMap=pUV;
  // worst case estimate
  pUVMap->m_Entries.reserve(iBytesLeft/(2+2+4+4));
  pLayer->m_UVMapDs.push_back(pUVMap);

  while(m_pData<pChunkEndPos)
  {
    unsigned int iVertexID=0;
    if(!ReadVX(iVertexID)) return false;

    unsigned int iPolyID=0;
    if(!ReadVX(iPolyID)) return false;

    UVMapD::Entry e;
    e.iVertex=iVertexID;
    e.iPolygon=iPolyID;
    if(!ReadF4(e.u)) return false;
    if(!ReadF4(e.v)) return false;
    // flip v coordinate
    e.v=1-e.v;
    pUVMap->m_Entries.push_back(e);
  }

  return true;
}

bool LWOFile::LoadVMAP(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();
  if(pLayer==NULL)
  {
    AddError("No LAYR before VMAP");
    return false;
  }

  // VMAP { type[ID4], dimension[U2], name[S0],
  //  ( vert[VX], value[F4] # dimension )* }
  unsigned int iType=0;
  if(!ReadID4(iType))
  {
    return false;
  }

  // uvmap
  if(iType==ID4("TXUV"))
  {
    return LoadUVMap(iChunkSize);
  }
  // weightmap
  else if(iType==ID4("WGHT"))
  {
    return LoadWeightMap(iChunkSize);
  }

  // unknown type - skip
  m_pData+=iChunkSize-4;
  AddError("Warning: Unknown vertex map type "+ID4ToString(iType));
  return true;
}

bool LWOFile::LoadUVMap(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();

  const char *pChunkEndPos=m_pData+iChunkSize-4;

  unsigned short iDimension=0;
  if(!ReadU2(iDimension))
  {
    return false;
  }

  // not 2 floats per vertex
  if(iDimension!=2)
  {
    // just skip
    m_pData+=iChunkSize-4-2;
    AddError("Warning: UVMap has "+ConvertToString(iDimension)+" floats per vertex (2 expected)");
    return true;
  }

  unsigned int iStrBytes=0;
  std::string strName;
  if(!ReadS0(strName,iStrBytes))
  {
    return false;
  }


  // VMAP { type[ID4], dimension[U2], name[S0],
  //  ( vert[VX], value[F4] # dimension )* }

  UVMap *pUVMap=new UVMap();
  pUVMap->m_strName=strName;
  pUVMap->m_Values.resize(2*pLayer->m_iPoints);
  pLayer->m_UVMaps.push_back(pUVMap);

  float *pValues=&(pUVMap->m_Values[0]);
  memset(pValues,0,sizeof(float)*2*pLayer->m_iPoints);

  while(m_pData<pChunkEndPos)
  {
    unsigned int iVertexID=0;
    if(!ReadVX(iVertexID)) return false;
    if(iVertexID>=pLayer->m_iPoints) return false;

    if(!ReadF4(pValues[iVertexID*2+0])) return false;
    if(!ReadF4(pValues[iVertexID*2+1])) return false;

    // flip v coordinate
    pValues[iVertexID*2+1]=1-pValues[iVertexID*2+1];
  }

  return true;
}

bool LWOFile::LoadWeightMap(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();

  const char *pChunkEndPos=m_pData+iChunkSize-4;

  unsigned short iDimension=0;
  if(!ReadU2(iDimension))
  {
    return false;
  }

  // is not 1 float per vertex
  if(iDimension!=1)
  {
    // just skip
    m_pData+=iChunkSize-4-2;
    AddError("Warning: WeightMap has "+ConvertToString(iDimension)+" floats per vertex (1 expected)");
    return true;
  }

  unsigned int iStrBytes=0;
  std::string strName;
  if(!ReadS0(strName,iStrBytes))
  {
    return false;
  }


  // VMAP { type[ID4], dimension[U2], name[S0],
  //  ( vert[VX], value[F4] # dimension )* }

  WeightMap *pWeightMap=new WeightMap();
  pWeightMap->m_strName=strName;
  pWeightMap->m_Values.resize(1*pLayer->m_iPoints);
  pLayer->m_WeightMaps.push_back(pWeightMap);

  float *pValues=&pWeightMap->m_Values[0];
  memset(pValues,0,sizeof(float)*1*pLayer->m_iPoints);

  while(m_pData<pChunkEndPos)
  {
    unsigned int iVertexID=0;
    if(!ReadVX(iVertexID)) return false;
    if(iVertexID>=pLayer->m_iPoints) return false;
    if(!ReadF4(pValues[iVertexID])) return false;
  }

  return true;
}

bool LWOFile::LoadTAGS(unsigned int iChunkSize)
{
  if(m_StringTable.size() != 0)
  {
    AddError("TAGS already defined!");
    return false;
  }

  const char *pChunkEndPos=m_pData+iChunkSize;
  while(m_pData<pChunkEndPos)
  {
    std::string strTemp;
    if(!ReadS0(strTemp)) return false;
    m_StringTable.push_back(strTemp);
  }
  return true;
}

bool LWOFile::LoadPTAG(unsigned int iChunkSize)
{
  Layer *pLayer=GetLastLayer();
  if(pLayer==NULL)
  {
    AddError("No LAYR before PTAG");
    return false;
  }

  if(pLayer->m_Polygons.size()==0)
  {
    AddError("No POLS before PTAG");
    return false;
  }

  const char *pChunkEndPos=m_pData+iChunkSize;

  // read type
  unsigned int iType=0;
  if(!ReadID4(iType))
  {
    return false;
  }

  if(iType!=ID4("SURF"))
  {
    // unknown type - skip
    m_pData+=iChunkSize-4;
    AddError("Warning: Unknown PTAG type "+ID4ToString(iType));
    return true;
  }

  while(m_pData<pChunkEndPos)
  {
    // read poly id
    unsigned int iPolyID;
    if(!ReadVX(iPolyID)) return false;

    // read tag
    unsigned short iTag;
    if(!ReadU2(iTag)) return false;

    if(iPolyID>=pLayer->m_Polygons.size()) return false;
    pLayer->m_Polygons[iPolyID].m_iSurface=iTag;
  }

  return true;
}

bool LWOFile::LoadSURF(unsigned int iChunkSize)
{
  Surface surf;
  surf.m_vBaseColor[0] = 1.0f;
  surf.m_vBaseColor[1] = 1.0f;
  surf.m_vBaseColor[2] = 1.0f;

  // SURF { name[S0], source[S0], attributes[SUB-CHUNK] * }
  const char *pChunkEndPos=m_pData+iChunkSize;

  // name
  unsigned int iStrBytes=0;
  if(!ReadS0(surf.m_strName,iStrBytes))
  {
    return false;
  }

  // source
  if(!ReadS0(surf.m_strSource,iStrBytes))
  {
    return false;
  }
  
  // while there's data left
  while(m_pData<pChunkEndPos)
  {
    // get chunk ID
    unsigned int iID=0;
    if(!ReadID4(iID)) return false;

    // get chunk size
    unsigned short iChunkSize=0;
    if(!ReadU2(iChunkSize)) return false;

    const char *pSubChunkEndPos=m_pData+iChunkSize;

    // color
    if(iID == ID4("COLR"))
    {
      // COLR { base-color[COL12], envelope[VX]  }

      // read values
      if(!ReadF4(surf.m_vBaseColor[0])) return false;
      if(!ReadF4(surf.m_vBaseColor[1])) return false;
      if(!ReadF4(surf.m_vBaseColor[2])) return false;
    }

    // skip to end of sub-chunk
    m_pData = pSubChunkEndPos;
    if(iChunkSize%2==1) m_pData+=1;
  }

  m_Surfaces.push_back(surf);

  return true;
}

LWOFile::Layer::Layer()
{
  m_iPoints=0;
}

LWOFile::Layer::~Layer()
{
  for(unsigned int i=0;i<m_UVMaps.size();i++)
  {
    delete m_UVMaps[i];
  }

  for(unsigned int i=0;i<m_UVMapDs.size();i++)
  {
    delete m_UVMapDs[i];
  }

  for(unsigned int i=0;i<m_WeightMaps.size();i++)
  {
    delete m_WeightMaps[i];
  }
}

struct SeperatedVertex
{
  unsigned int iVertex;
  unsigned int iNewIndex;
};

struct SeperatedPoly
{
  std::vector<SeperatedVertex> Vertices;
};

void LWOFile::FixDiscontinuousUVMaps(void)
{
  // for each layer
  for(unsigned int iLayer=0;iLayer<m_Layers.size();iLayer++)
  {
    Layer *pLayer=m_Layers[iLayer];

    // for each uvmapd
    unsigned int iAdditionalSize=0;
    for(unsigned int iUVD=0;iUVD<pLayer->m_UVMapDs.size();iUVD++)
    {
      iAdditionalSize+=(unsigned int)pLayer->m_UVMapDs[iUVD]->m_Entries.size();
    }
    // allocate extra space
    pLayer->m_Points.reserve(pLayer->m_Points.size()+3*iAdditionalSize);
    for(unsigned int iUV=0;iUV<pLayer->m_UVMaps.size();iUV++)
      pLayer->m_UVMaps[iUV]->m_Values.reserve(pLayer->m_UVMaps[iUV]->m_Values.size()+2*iAdditionalSize);
    for(unsigned int iWM=0;iWM<pLayer->m_WeightMaps.size();iWM++)
      pLayer->m_WeightMaps[iWM]->m_Values.reserve(pLayer->m_WeightMaps[iWM]->m_Values.size()+1*iAdditionalSize);

    std::vector<SeperatedPoly> SeperatedPolys;
    SeperatedPolys.resize(pLayer->m_Polygons.size());


    // for each uvmapd
    for(unsigned int iUVD=0;iUVD<pLayer->m_UVMapDs.size();iUVD++)
    {
      UVMapD *pUVD=pLayer->m_UVMapDs[iUVD];

      for(unsigned int iEntry=0;iEntry<pUVD->m_Entries.size();iEntry++)
      {
        const UVMapD::Entry &e=pUVD->m_Entries[iEntry];

        bool bFound=false;
        for(unsigned int iSV=0;iSV<SeperatedPolys[e.iPolygon].Vertices.size();iSV++)
        {
          if(SeperatedPolys[e.iPolygon].Vertices[iSV].iVertex==e.iVertex)
          {
            // set new uv
            pUVD->m_pUVMap->m_Values[SeperatedPolys[e.iPolygon].Vertices[iSV].iNewIndex*2+0]=e.u;
            pUVD->m_pUVMap->m_Values[SeperatedPolys[e.iPolygon].Vertices[iSV].iNewIndex*2+1]=e.v;
            bFound=true;
            break;
          }
        }

        // not seperated yet
        if(!bFound)
        {
          // add new vertex to polygon
          pLayer->m_iPoints++;
          unsigned int iNewIndex=(unsigned int)(pLayer->m_Points.size()/3);
          Polygon *pPoly=&(pLayer->m_Polygons[e.iPolygon]);
          for(unsigned int i=0;i<pPoly->m_Vertices.size();i++)
          {
            if(pPoly->m_Vertices[i]==e.iVertex) pPoly->m_Vertices[i]=iNewIndex;
          }

          SeperatedVertex sv;
          sv.iVertex=e.iVertex;
          sv.iNewIndex=iNewIndex;
          SeperatedPolys[e.iPolygon].Vertices.push_back(sv);

          // copy position
          pLayer->m_Points.resize(pLayer->m_Points.size()+3);
          pLayer->m_Points[iNewIndex*3+0]=pLayer->m_Points[e.iVertex*3+0];
          pLayer->m_Points[iNewIndex*3+1]=pLayer->m_Points[e.iVertex*3+1];
          pLayer->m_Points[iNewIndex*3+2]=pLayer->m_Points[e.iVertex*3+2];
          // copy uvmaps
          for(unsigned int iUV=0;iUV<pLayer->m_UVMaps.size();iUV++)
          {
            pLayer->m_UVMaps[iUV]->m_Values.resize(pLayer->m_UVMaps[iUV]->m_Values.size()+2);
            pLayer->m_UVMaps[iUV]->m_Values[iNewIndex*2+0]=pLayer->m_UVMaps[iUV]->m_Values[e.iVertex*2+0];
            pLayer->m_UVMaps[iUV]->m_Values[iNewIndex*2+1]=pLayer->m_UVMaps[iUV]->m_Values[e.iVertex*2+1];
          }
          // copy weightmaps
          for(unsigned int iWM=0;iWM<pLayer->m_WeightMaps.size();iWM++)
          {
            pLayer->m_WeightMaps[iWM]->m_Values.push_back(pLayer->m_WeightMaps[iWM]->m_Values[e.iVertex]);
          }
          // copy normals
          if(!pLayer->m_Normals.empty())
          {
            pLayer->m_Normals.resize(pLayer->m_Normals.size()+3);
            pLayer->m_Normals[iNewIndex*3+0]=pLayer->m_Normals[e.iVertex*3+0];
            pLayer->m_Normals[iNewIndex*3+1]=pLayer->m_Normals[e.iVertex*3+1];
            pLayer->m_Normals[iNewIndex*3+2]=pLayer->m_Normals[e.iVertex*3+2];
          }

          // set new uv
          pUVD->m_pUVMap->m_Values[iNewIndex*2+0]=e.u;
          pUVD->m_pUVMap->m_Values[iNewIndex*2+1]=e.v;
        }
      }
    }
  }
}

void LWOFile::CalculateNormals(void)
{
  // for each layer
  for(unsigned int iLayer=0;iLayer<m_Layers.size();iLayer++)
  {
    m_Layers[iLayer]->CalculateNormals();
  }
}

void LWOFile::Layer::CalculateNormals(void)
{
  // zero normals
  m_Normals.resize(m_iPoints*3);
  memset(&m_Normals[0],0,sizeof(float)*m_iPoints*3);

  // for each polygon
  for(unsigned int iPoly=0;iPoly<m_Polygons.size();iPoly++)
  {
    const Polygon &poly=m_Polygons[iPoly];

    // ignore non-polygons
    if(poly.m_Vertices.size()<3) continue;

    // vertex offsets in float array
    const unsigned int &iV1=3*poly.m_Vertices[0];
    const unsigned int &iV2=3*poly.m_Vertices[1];
    const unsigned int &iV3=3*poly.m_Vertices[poly.m_Vertices.size()-1];

    // calculate edge vectors
    float vFirstEdge[3]={m_Points[iV2+0]-m_Points[iV1+0],
                         m_Points[iV2+1]-m_Points[iV1+1],
                         m_Points[iV2+2]-m_Points[iV1+2]};

    float vLastEdge[3]={m_Points[iV3+0]-m_Points[iV1+0],
                        m_Points[iV3+1]-m_Points[iV1+1],
                        m_Points[iV3+2]-m_Points[iV1+2]};

    // calculate cross product
    float vCrossProduct[3]={vFirstEdge[1]*vLastEdge[2] - vFirstEdge[2]*vLastEdge[1],
                            vFirstEdge[2]*vLastEdge[0] - vFirstEdge[0]*vLastEdge[2],
                            vFirstEdge[0]*vLastEdge[1] - vFirstEdge[1]*vLastEdge[0]};


    // calculate length
    float fLength=sqrtf(vCrossProduct[0]*vCrossProduct[0] +
                        vCrossProduct[1]*vCrossProduct[1] +
                        vCrossProduct[2]*vCrossProduct[2]);


    if(fLength>=FLT_MIN)
    {
      // normalize
      vCrossProduct[0]/=fLength;
      vCrossProduct[1]/=fLength;
      vCrossProduct[2]/=fLength;
    }

    // add normals to each vertex of poly
    for(unsigned int i=0;i<poly.m_Vertices.size();i++)
    {
      m_Normals[poly.m_Vertices[i]*3+0]+=vCrossProduct[0];
      m_Normals[poly.m_Vertices[i]*3+1]+=vCrossProduct[1];
      m_Normals[poly.m_Vertices[i]*3+2]+=vCrossProduct[2];
    }
  }

  // renormalize the summed normals
  for(unsigned int i=0;i<m_Points.size();i+=3)
  {
    float fLength=sqrtf(m_Normals[i+0]*m_Normals[i+0]+
                        m_Normals[i+1]*m_Normals[i+1]+
                        m_Normals[i+2]*m_Normals[i+2]);
    if(fLength>=FLT_MIN)
    {
      m_Normals[i+0]/=fLength;
      m_Normals[i+1]/=fLength;
      m_Normals[i+2]/=fLength;
    }
  }
}
