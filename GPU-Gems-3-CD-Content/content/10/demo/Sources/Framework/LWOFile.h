#pragma once

#include <string>
#include <vector>
#include <list>

class LWOFile
{
public:
  LWOFile();
  ~LWOFile();

  bool LoadFromFile(const char *strFile);
  bool LoadFromMemory(const void *pData, unsigned int iFileSize);
  void CalculateNormals(void);
  void FixDiscontinuousUVMaps(void);

  // Surfaces
  //
  class Surface
  {
  public:
    float m_vBaseColor[3];
    std::string m_strName;
    std::string m_strSource;
  };

  // Weightmaps
  //
  class WeightMap
  {
  public:
    std::vector<float> m_Values;
    std::string m_strName;
  };

  // UVMaps
  //
  class UVMap
  {
  public:
   std::vector<float> m_Values;
   std::string m_strName;
  };

  // Discontinuous UVMaps
  //
  class UVMapD
  {
  public:
    typedef struct
    {
      unsigned int iPolygon;
      unsigned int iVertex;
      float u,v;
    } Entry;

    std::vector<Entry> m_Entries;
    UVMap *m_pUVMap;
    std::string m_strName;
  };

  // Polygons
  //
  class Polygon
  {
  public:
    std::vector<unsigned int> m_Vertices;
    unsigned short m_iSurface;
  };

  // Layers
  //
  class Layer
  {
  public:
    Layer();
    ~Layer();

    // vertex positions
    std::vector<float> m_Points; // num floats = m_iPoints*3
    unsigned int m_iPoints;
    // vertex normals
    std::vector<float> m_Normals; // num floats = m_iPoints*3

    // vertex maps
    //
    std::vector<UVMap *> m_UVMaps;
    std::vector<UVMapD *> m_UVMapDs;
    std::vector<WeightMap *> m_WeightMaps;


    // polygons
    std::vector<Polygon> m_Polygons;
    void CalculateNormals(void);

    unsigned short m_iID;
    unsigned short m_iFlags;
    float m_vPivot[3];
    std::string m_strName;
    unsigned short m_iParentID;
  };

  // pointers to layers
  std::vector<Layer *> m_Layers;

  // array of strings (eg. surface names)
  std::vector<std::string> m_StringTable;
  std::vector<Surface> m_Surfaces;

  // list of errors that occurred while reading
  std::list<std::string> m_Errors;

private:
  inline void AddError(const std::string &strError);
  inline Layer *GetLastLayer(void) { if(m_Layers.empty()) return NULL; return (*m_Layers.rbegin()); }

  // reading chunks
  bool LoadLAYR(unsigned int iChunkSize);
  bool LoadPNTS(unsigned int iChunkSize);
  bool LoadPOLS(unsigned int iChunkSize);
  bool LoadVMAD(unsigned int iChunkSize);
  bool LoadVMAP(unsigned int iChunkSize);
  bool LoadUVMap(unsigned int iChunkSize);
  bool LoadWeightMap(unsigned int iChunkSize);
  bool LoadTAGS(unsigned int iChunkSize);
  bool LoadPTAG(unsigned int iChunkSize);
  bool LoadSURF(unsigned int iChunkSize);

  // reading basic types
  //
  inline bool SafeToRead(unsigned int iBytes);
  inline bool ReadID4(unsigned int &iVar);
  inline bool ReadU4(unsigned int &iVar);
  inline bool ReadVX(unsigned int &iVar);
  inline bool ReadU2(unsigned short &iVar);
  inline bool ReadU1(unsigned char &iVar);
  inline bool ReadI4(int &iVar);
  inline bool ReadI2(short &iVar);
  inline bool ReadI1(char &iVar);
  inline bool ReadF4(float &fVar);
  inline bool ReadS0(std::string &strVar);
  inline bool ReadS0(std::string &strVar, unsigned int &iBytesRead);
  inline bool ReadAndFlip4(void *pData, unsigned int iLength);
  inline bool Read(void *pData, unsigned int iLength);

  // temporary
  const char *m_pData;
  const char *m_pDataEnd;
};
