#include "Common.h"
#include "LWOFile.h"
#include "Mesh.h"

Mesh::Mesh()
{
  m_iNumVertices = 0;
  m_iNumTris = 0;
  m_iVertexSize = 0;
  m_pIndices = NULL;
  m_pVertices = NULL;
}

Mesh::~Mesh()
{
  delete[] m_pVertices;
  delete[] m_pIndices;
}

bool Mesh::LoadFromLWO(const char *strFile)
{
  LWOFile lwo;

  // try to load
  if(!lwo.LoadFromFile(strFile))
  {
    // failed, print all errors
    std::string strErrors = "Loading LWO failed:\n";
    std::list<std::string>::iterator it;
    for(it = lwo.m_Errors.begin(); it != lwo.m_Errors.end(); ++it) strErrors += (*it)+"\n";
    strErrors += "\nIf you launched the demo from Visual Studio, set the 'Working Directory' in project settings correctly.";
    MessageBoxA(NULL,strErrors.c_str(),"Error!",MB_OK);
    return false;
  }

  // calculate normals
  lwo.CalculateNormals();

  // get first layer only
  LWOFile::Layer *pLayer = lwo.m_Layers[0];

  // make vertex buffer
  //
  // 9 floats per vertex (3 pos, 3 norm, 3 color)
  m_pVertices = new float[pLayer->m_iPoints*9];
  m_iVertexSize = sizeof(float)*9;
  m_iNumVertices = pLayer->m_iPoints;
  for(unsigned int i = 0; i < pLayer->m_iPoints; i++)
  {
    // copy pos
    memcpy(&m_pVertices[i*9+0], &pLayer->m_Points[i*3], sizeof(float)*3);

    // copy normal
    memcpy(&m_pVertices[i*9+3], &pLayer->m_Normals[i*3], sizeof(float)*3);

    // color
    m_pVertices[i*9+6] = 1.0f;
    m_pVertices[i*9+7] = 1.0f;
    m_pVertices[i*9+8] = 1.0f;
  }

  // make index buffer
  //
  m_pIndices = new unsigned short[pLayer->m_Polygons.size()*3];
  m_iNumTris = 0;
  for(unsigned int i = 0; i < pLayer->m_Polygons.size(); i++)
  {
    const LWOFile::Polygon &poly = pLayer->m_Polygons[i];

    // skip non-triangles
    if(poly.m_Vertices.size() != 3) continue;

    // find surface
    const LWOFile::Surface *pSurf = NULL;
    for(unsigned int j = 0; j < lwo.m_Surfaces.size(); j++)
    {
      const LWOFile::Surface *pSurf = &lwo.m_Surfaces[j];
      if(pSurf->m_strName != lwo.m_StringTable[poly.m_iSurface]) continue;

      // apply base color
      for(int j = 0; j < 3; j++)
      {
        m_pVertices[poly.m_Vertices[j]*9 + 6] = pSurf->m_vBaseColor[0];
        m_pVertices[poly.m_Vertices[j]*9 + 7] = pSurf->m_vBaseColor[1];
        m_pVertices[poly.m_Vertices[j]*9 + 8] = pSurf->m_vBaseColor[2];
      }
      break;
    }

    m_pIndices[m_iNumTris*3+0] = poly.m_Vertices[0];
    m_pIndices[m_iNumTris*3+1] = poly.m_Vertices[1];
    m_pIndices[m_iNumTris*3+2] = poly.m_Vertices[2];
    m_iNumTris++;
  }

  // apply ambient occlusion from texture coordinates, if available
  //
  if(pLayer->m_UVMaps.size() > 0)
  {
    LWOFile::UVMap *pUVMap = pLayer->m_UVMaps[0];
    for(unsigned int i = 0; i < m_iNumVertices; i++)
    {
      m_pVertices[i*9+6] *= pUVMap->m_Values[i*2];
      m_pVertices[i*9+7] *= pUVMap->m_Values[i*2];
      m_pVertices[i*9+8] *= pUVMap->m_Values[i*2];
    }
  }


  CalculateOOBB();

  if(!CreateBuffers()) return false;
  return true;
}

void Mesh::CalculateOOBB(void)
{
  m_OOBB.Set(m_pVertices, m_iNumVertices, m_iVertexSize);
}
