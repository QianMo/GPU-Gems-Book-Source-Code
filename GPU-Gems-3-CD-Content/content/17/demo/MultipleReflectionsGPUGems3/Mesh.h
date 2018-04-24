#pragma once
/**
	\brief Encapsulates a mesh of .X format. 
	\brief Adds helper functions for easier scaling, moving, texturing
	\see OptimizedMesh DirectX SDK Sample
*/
enum RenderMode
{
	RENDERMODE_COLOR,
	RENDERMODE_NORMAL,
	RENDERMODE_FINAL
};

class Material
{
public:
	char colorRenderTechnique[100];
	char normalRenderTechnique[100];
	char finalRenderTechnique[100];
	char colorTextureName[256];
	IDirect3DTexture9* colorTexture;
	D3DXVECTOR4 F0;
	float N0;
	bool basicFinalRender;
	ID3DXEffect* currentEffect;

	Material(){colorTexture = NULL;}
	~Material(){destroyTexture();}
	void beginMaterial(RenderMode mode);
	void endMaterial();	
	void destroyTexture()
	{
		SAFE_RELEASE(colorTexture);
	}
	void loadTexture();
};

class Mesh
{
	ID3DXMesh* pMesh;					///< THE .X MESH OBJECT
	
	DWORD numMaterials;
	Material* materials;
	D3DXVECTOR3 originalSize; 
	float originalDiameter;
	float preferredDiameter;
	D3DXVECTOR3 containerSize;

	HRESULT hr;
	D3DXVECTOR3 position;
	D3DXMATRIXA16 rotation;

public:

	/// Loads the specified .X file
	/// \param fileName name of the mesh file
	/// \param texFileName name of the texture file
	/// \param preferredDiameter final size of the mesh
	/// \param position initial position of the mesh.
	/// During rendering, use GetMeshScale() to get the appropriate scaling factor 
	/// to achieve the preferred mesh size.
	Mesh(LPCWSTR fileName, float preferredDiameter, D3DXVECTOR3 position);
	/// Destructor. Deletes dynamically created resources.
	~Mesh();

	/// \brief LOADS THE SPECIFIED .X FILE
	/// Eliminates mesh offset (centers the mesh to the origin) 
	/// and determines mesh size (#originalSize, #originalDiameter).
	void Load(LPCWSTR fileName); 
	/// Adds the specified offset to the mesh position.
	/// \param bContainerOnly indicates whether the object can leave the room
	void Move(D3DXVECTOR3 offset, bool bContainerOnly = false);
	/// Draws the mesh. Before drawing, use GetMeshScale() to get the appropriate scaling factor.
	HRESULT Draw(enum RenderMode mode);
	/// current mesh size	
	D3DXVECTOR3 GetMeshSize()   { return GetMeshScale() * originalSize; }
	/// current mesh scale
	float GetMeshScale()		{ return preferredDiameter / originalDiameter; }
	/// current mesh position
	D3DXVECTOR3 GetMeshPosition()	{ return position; }
	
	/// Sets the size of the encapsulating room.
	void SetContainerSize(D3DXVECTOR3 size) { containerSize = size; }
	/// Sets mesh position.
	void SetMeshPosition(D3DXVECTOR3 pos)	{ position = pos; }
	/// Sets preferred mesh size (diameter).
	void SetPreferredDiameter(float d) { 
		preferredDiameter = d;
		Move(D3DXVECTOR3(0,0,0),true);		// to stay inside the room
	}

	void setRotation(D3DXMATRIXA16& rotmatrix){rotation = rotmatrix;}
	D3DXMATRIXA16& getRotation(){return rotation;}

	Material& getMaterial(int subMesh){return materials[subMesh];}

protected:
	/// Calculates mesh size and updates #originalSize and #originalDiameter.
	HRESULT CalculateMeshSize( );
};