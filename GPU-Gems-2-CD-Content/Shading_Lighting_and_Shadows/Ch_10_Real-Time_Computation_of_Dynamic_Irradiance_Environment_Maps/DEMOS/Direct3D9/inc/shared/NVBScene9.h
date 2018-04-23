/*********************************************************************NVMH4****
Path:  SDK\DEMOS\Direct3D9\inc\shared
File:  NVBScene9.h

Copyright NVIDIA Corporation 2002
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.



Comments:


******************************************************************************/


#ifndef NVBSCENE_H
#define NVBSCENE_H

#pragma warning(disable : 4786)
#include <string>
#include <vector>
#include <d3dx9.h>
#include <nv_nvb/nv_nvb.h>
#include <TCHAR.H>
typedef std::basic_string<TCHAR> tstring; 



class NVBScene {
public:
	//function that finds the full path of a given filename
	typedef tstring (*GetFilePathCallback)(const tstring&);

	//a scene is made of a list of meshes, a list of cameras and a list of lights

	//a base mesh is essentially a vertex buffer
	class BaseMesh {
	public:
		class Vertex {
		public:
			D3DXVECTOR3 m_Position; //position
			unsigned int m_Color; //color
			D3DXVECTOR2 m_Texture; //texture coordinates
		};
		BaseMesh();
		~BaseMesh();
		virtual HRESULT Draw() const;

		LPDIRECT3DDEVICE9 m_Device;
		std::string m_Name; //name of the geometry part

		//the vertex buffer
		unsigned int m_NumVertices;
		LPDIRECT3DVERTEXBUFFER9 m_VertexBuffer;
	protected:
		virtual HRESULT Free();
	};

	//an indexed base mesh also has an index buffer
	class BaseMeshIndexed : public BaseMesh {
	public:
		BaseMeshIndexed();
		~BaseMeshIndexed();
		virtual HRESULT Draw() const;

		//the faces
		unsigned int m_NumTriangles;
		LPDIRECT3DINDEXBUFFER9 m_IndexBuffer;

	protected:
		virtual HRESULT Free();
	};

	//a mesh is a base mesh with textures and tangent space
	class Mesh : public BaseMeshIndexed {
	public:
		class Vertex {
		public:
			D3DXVECTOR3 m_Position; //position
			D3DXVECTOR3 m_Normal; //normal
			unsigned int m_Color; //color
			D3DXVECTOR2 m_Texture; //texture coordinates
			D3DXVECTOR3 m_S; //normalized tangent along the first texture coordinate s
			D3DXVECTOR3 m_T; //normalized tangent along the second texture coordinate t
			D3DXVECTOR3 m_SxT; //normalized cross product of the two previous tangents
		};
		Mesh();
		~Mesh();
		HRESULT Draw() const;
		HRESULT DrawNormals(const D3DXMATRIX&, const D3DXMATRIX&, const D3DXMATRIX&) const;
		HRESULT DrawTangentBasis(const D3DXMATRIX&, const D3DXMATRIX&, const D3DXMATRIX&) const;

		//vertex data
		Vertex* m_Vertices;

		//index data
		unsigned short* m_Indices;

		//matrix to transform the mesh vertices from object space to scene space
		D3DXMATRIX m_Transform;

		//textures
		LPDIRECT3DTEXTURE9 m_DiffuseMap; //diffuse texture
		LPDIRECT3DTEXTURE9 m_HeightMap; //height fields
		LPDIRECT3DTEXTURE9 m_NormalMap; //normal map

		//name of the material assigned to the geometry part
		std::string m_Material;

		//boolean values that describe the information available for every vertex of the mesh
		bool m_VertexHasNormal;
		bool m_VertexHasColor;
		bool m_VertexHasTexture;
		bool m_VertexHasS;
		bool m_VertexHasT;
		bool m_VertexHasSxT;
		bool m_VertexHasTangentBasis;

		//vertex buffers used to display the normals and tangent basis
		LPDIRECT3DVERTEXBUFFER9 m_NormalBuffer;
		LPDIRECT3DVERTEXBUFFER9 m_SBuffer;
		LPDIRECT3DVERTEXBUFFER9 m_TBuffer;
		LPDIRECT3DVERTEXBUFFER9 m_SxTBuffer;

	protected:
		virtual HRESULT Free();
	};

	//camera
	class Camera {
	public:
		D3DXMATRIX m_Projection;
		D3DXMATRIX m_View;
	};

	//light
	class Light : public D3DLIGHT9 {
	public:
		Light();
		std::string m_Name;
		unsigned int m_NumKeys;
	};

	//allocation/deallocation
	NVBScene();
	~NVBScene();
	void Free();

	//loading
	HRESULT Load(const tstring&, LPDIRECT3DDEVICE9, GetFilePathCallback getFilePath = 0);

	//updating
	//those constants are used to selectively update part of the scene
	static const unsigned int MESH;
	static const unsigned int CAMERA;
	static const unsigned int LIGHT;
	void Update(float key = -1, const D3DXMATRIX* transform = 0, unsigned int what = MESH | CAMERA | LIGHT);

	//rendering
	HRESULT Draw();
	HRESULT DrawNormals(const D3DXMATRIX* projection = 0, const D3DXMATRIX* view = 0, const D3DXMATRIX* world = 0);
	HRESULT DrawTangentBasis(const D3DXMATRIX* projection = 0, const D3DXMATRIX* view = 0, const D3DXMATRIX* world = 0);
	HRESULT DrawCoordinateAxis(const D3DXMATRIX* projection = 0, const D3DXMATRIX* view = 0, const D3DXMATRIX* world = 0);
	HRESULT DrawCube();

	//the buffers
	LPDIRECT3DDEVICE9 m_Device;
	unsigned int m_NumMeshes;
	Mesh* m_Meshes;
	unsigned int m_NumMeshKeys;
	unsigned int m_NumCameras;
	Camera* m_Cameras;
	unsigned int m_NumCameraKeys;
	unsigned int m_NumLights;
	Light* m_Lights;
	unsigned int m_NumLightKeys;
	D3DXVECTOR3 m_Center; //center of the scene axis-aligned bounding box
	float m_Radius; //radius of the sphere circumscribing the scene
	float m_NormalLength; //length of a segment representing a normal

	//string holding the last error message
	tstring m_ErrorMessage;

	//boolean values that describe the information available for every vertex of every mesh
	bool m_VertexHasNormal;
	bool m_VertexHasColor;
	bool m_VertexHasTexture;
	bool m_VertexHasS;
	bool m_VertexHasT;
	bool m_VertexHasSxT;
	bool m_VertexHasTangentBasis;

private:
	//smart pointer to simplify freeing of temporary heap allocation
	template<class T> class Ptr {
	public:
		Ptr(unsigned int n) { m_Ptr = (n ? new T[n] : 0); }
		~Ptr() { if (m_Ptr) delete [] m_Ptr; }
		T& operator[](unsigned int i) { return m_Ptr[i]; }
	private:
		T* m_Ptr;
	};
	void Reset();
	void Analyze();
	HRESULT LoadTextures(GetFilePathCallback);
	HRESULT LoadTexture(const tstring&, LPDIRECT3DTEXTURE9&, GetFilePathCallback, D3DFORMAT format = D3DFMT_A8R8G8B8, D3DPOOL pool = D3DPOOL_DEFAULT);
	void ReleaseTextures();
	HRESULT LoadMeshes(GetFilePathCallback);
	HRESULT LoadCameras();
	HRESULT LoadLights();
	static void MatToD3DXMATRIX(const mat4&, D3DXMATRIX&);
	nv_scene m_Scene;
	std::vector<LPDIRECT3DTEXTURE9> m_Textures;

	//each scene component has its own key cursor to allow them to be animated independently from each other
	float m_TimeMesh;
	float m_TimeCamera;
	float m_TimeLight;

	//the coordinate axis mesh
	HRESULT CreateCoordinateAxis();
	BaseMesh m_CoordinateAxis;

	//the environment cube mesh
	class CubeMesh : public BaseMeshIndexed {//a mesh is a base mesh with textures and tangent space
	public:
		class Vertex {
		public:
			Vertex(const D3DXVECTOR3& position) : m_Position(position), m_Texture(position) { };
			D3DXVECTOR3 m_Position; //position
			D3DXVECTOR3 m_Texture; //texture coordinates
		};
	};
	HRESULT CreateCube();
	CubeMesh m_Cube;
};

#endif
