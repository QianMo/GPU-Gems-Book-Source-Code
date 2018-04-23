#pragma once

#pragma pack(push)
#pragma pack(1)

#define TSG_MAXNAME		64
#define TSG_VERSION		0x3000
#define TSG_SIGNATURE	'FGST'



// header
typedef struct
{
	unsigned int		dwID;				// always TDA_SIGNATURE
	unsigned short	wVersion;			// format version

	unsigned short	wMaterialCount;		// number of materials
	unsigned short	wVertexCount;		// number of vertices
	unsigned short	wMeshCount;			// number of meshes
	unsigned short	wBoneCount;			// number of joints

} TSGHeader;


// material
typedef struct
{
	char		pName[TSG_MAXNAME];				// material name
	char		pTextureMap[TSG_MAXNAME];		// texture map
	char		pReflectionMap[TSG_MAXNAME];	// environment map texture

	float		cAmbient[3];			// ambient component rgb
	float		cDiffuse[3];			// diffuse component rgb
	float		cSpecular[3];			// specular component rgb
	float		cEmissive[3];			// emissive component rgb
	float		fShininess;				// shininess
	float		fOpacity;				// opacity

} TSGMaterial;


// vertex
typedef struct
{
	float		vPosition[3];						// xyz
	float		vNormal[3];							// normal
	float		vTexCoord[2];						// texcoord
	char		bBoneCount;							// number of assigned bones
	short		wBoneID[4];							// 4 bone indices
	float		fBoneWeight[4];						// 4 bone weights

} TSGVertex;

// mesh
typedef struct
{
	char					pName[TSG_MAXNAME];		// the material name
	short					wMaterialID;			// the material index, -1 if no material
	unsigned int	dwFaceCount;			// number of triangles

	unsigned short	*pIndices;				// vertex indices	(used internaly only)

} TSGMesh;

#pragma pack(pop)


class CModelLoader
{
public:
	CModelLoader();
	virtual ~CModelLoader();

	bool LoadFromFile(const char *szFileName);
	void Release();

	int GetMaterialCount() { return m_pHeader.wMaterialCount; };
	int GetVertexCount() { return m_pHeader.wVertexCount; };
	int GetMeshCount() { return m_pHeader.wMeshCount; };

	TSGVertex		*GetVertices() { return m_pVertices; };
	TSGMesh			*GetMeshes() { return m_pMeshes; };
	TSGMaterial *GetMaterials() { return m_pMaterials; };

private:

	TSGHeader		m_pHeader;
	TSGMaterial	*m_pMaterials;
	TSGMesh			*m_pMeshes;
	TSGVertex		*m_pVertices;
};
