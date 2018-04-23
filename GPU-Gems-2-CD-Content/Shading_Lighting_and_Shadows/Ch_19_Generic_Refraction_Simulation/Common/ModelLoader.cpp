#include "stdafx.h"
#include "ModelLoader.h"
#include <stdio.h>
#include <memory.h>


CModelLoader::CModelLoader(void)
{
  m_pMaterials=0;
  m_pMeshes=0;
  m_pVertices=0;
}

CModelLoader::~CModelLoader(void)
{
  Release();
}

bool CModelLoader::LoadFromFile(const char *szFileName)
{
	FILE *hFile = fopen(szFileName, "rb");

	if (!hFile)
	{
		return false;
	}

	if (fread(&m_pHeader, sizeof(TSGHeader), 1, hFile) != 1)
	{
		fclose(hFile);

		return false;
	}
 
	if(m_pHeader.dwID!=TSG_SIGNATURE)
	{
		fclose(hFile);
		return false;
	}

	// read the materials
	if (m_pHeader.wMaterialCount)
	{
		m_pMaterials = new TSGMaterial[m_pHeader.wMaterialCount];

		memset(m_pMaterials, 0, sizeof(TSGMaterial) * m_pHeader.wMaterialCount);

		for (int i = 0; i < m_pHeader.wMaterialCount; i++)
		{
			if (fread(&m_pMaterials[i], sizeof(TSGMaterial), 1, hFile) != 1)
			{
				fclose(hFile);

				return false;
			}
		}
	}

	// read the meshes
	if (m_pHeader.wMeshCount)
	{
		m_pMeshes = new TSGMesh[m_pHeader.wMeshCount];

		memset(m_pMeshes, 0, sizeof(TSGMesh) * m_pHeader.wMeshCount);

		for (int i = 0; i < m_pHeader.wMeshCount; i++)
		{
			if (fread(&m_pMeshes[i], sizeof(TSGMesh) - sizeof(unsigned short *), 1, hFile) != 1)
			{
				fclose(hFile);

				return false;
			}
		}

		// read indices
		for (int j = 0; j < m_pHeader.wMeshCount; j++)
		{
			m_pMeshes[j].pIndices = new unsigned short[m_pMeshes[j].dwFaceCount * 3];

			if (fread(m_pMeshes[j].pIndices, sizeof(unsigned short), m_pMeshes[j].dwFaceCount * 3, hFile) != m_pMeshes[j].dwFaceCount * 3)
			{
				fclose(hFile);

				return false;
			}
		}
	}

	// read vertices
	if (m_pHeader.wVertexCount)
	{
		m_pVertices = new TSGVertex[m_pHeader.wVertexCount];

		memset(m_pVertices, 0, sizeof(TSGVertex) * m_pHeader.wVertexCount);

		if (fread(m_pVertices, sizeof(TSGVertex), m_pHeader.wVertexCount, hFile) != m_pHeader.wVertexCount)
		{
			fclose(hFile);

			return false;
		}
	}

	fclose(hFile);

	return true;
}

void CModelLoader::Release()
{
  SAFE_DELETE_ARRAY(m_pMaterials)
  SAFE_DELETE_ARRAY(m_pVertices)

	if (m_pMeshes)
	{
		for (int i = 0; i < m_pHeader.wMeshCount; i++)
		{
      SAFE_DELETE_ARRAY(m_pMeshes[i].pIndices)			
		}
	}

  SAFE_DELETE_ARRAY(m_pMeshes)

	memset(&m_pHeader, 0, sizeof(TSGHeader));
}