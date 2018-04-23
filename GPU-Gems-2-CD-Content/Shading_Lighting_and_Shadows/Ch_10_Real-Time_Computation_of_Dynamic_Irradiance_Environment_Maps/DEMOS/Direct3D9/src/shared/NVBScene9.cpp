/*********************************************************************NVMH4****
Path:  SDK\DEMOS\Direct3D9\src\shared
File:  NVBScene9.cpp

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

#include <shared/NVBScene9.h>
//#include <shared/nvtexture.h>
#include <TCHAR.H>
typedef std::basic_string<TCHAR> tstring; 

const unsigned int NVBScene::MESH = 1 << 0;
const unsigned int NVBScene::CAMERA = 1 << 1;
const unsigned int NVBScene::LIGHT = 1 << 2;

/*******************************************************************************

	Constructors / destructors

*******************************************************************************/

NVBScene::NVBScene()
{
	Reset();
}

NVBScene::~NVBScene()
{
	Free();
}

void NVBScene::Reset()
{
	m_Device = 0;
	m_NumMeshes = 0;
	m_Meshes = 0;
	m_Radius = 0;
	m_NormalLength = 0;
	m_NumMeshKeys = 0;
	m_NumCameraKeys = 0;
	m_NumLightKeys = 0;
	m_TimeMesh = -1;
	m_TimeCamera = -1;
	m_TimeLight = -1;
	m_VertexHasNormal = false;
	m_VertexHasColor = false;
	m_VertexHasTexture = false;
	m_VertexHasS = false;
	m_VertexHasT = false;
	m_VertexHasSxT = false;
	m_VertexHasTangentBasis = false;
	m_NumCameras = 0;
	m_NumLights = 0;
	m_Cameras = 0;
	m_Lights = 0;
}

void NVBScene::Free()
{
	ReleaseTextures();
	if (m_Meshes) {
		delete [] m_Meshes;
		m_Meshes = 0;
	}
	if (m_Cameras) {
		delete [] m_Cameras;
		m_Cameras = 0;
	}
	if (m_Lights) {
		delete [] m_Lights;
		m_Lights = 0;
	}
	Reset();
}

NVBScene::BaseMesh::BaseMesh() :
	m_Device(0),
	m_NumVertices(0), m_VertexBuffer(0)
{
}

NVBScene::BaseMesh::~BaseMesh()
{
	Free();
}

HRESULT NVBScene::BaseMesh::Free()
{
	if (m_VertexBuffer) {
		m_VertexBuffer->Release();
		m_VertexBuffer = 0;
	}
	return S_OK;
}

NVBScene::BaseMeshIndexed::BaseMeshIndexed() :
	m_NumTriangles(0), m_IndexBuffer(0)
{
}

NVBScene::BaseMeshIndexed::~BaseMeshIndexed()
{
	Free();
}

HRESULT NVBScene::BaseMeshIndexed::Free()
{
	if (m_IndexBuffer) {
		m_IndexBuffer->Release();
		m_IndexBuffer = 0;
	}
	return S_OK;
}

NVBScene::Mesh::Mesh() :
	m_Vertices(0), m_Indices(0),
	m_DiffuseMap(0), m_HeightMap(0), m_NormalMap(0),
	m_NormalBuffer(0), m_SBuffer(0), m_TBuffer(0), m_SxTBuffer(0),
	m_VertexHasNormal(false), m_VertexHasColor(false), m_VertexHasTexture(false), m_VertexHasS(false), m_VertexHasT(false), m_VertexHasSxT(false), m_VertexHasTangentBasis(false)
{
}

NVBScene::Mesh::~Mesh()
{
	Free();
}

HRESULT NVBScene::Mesh::Free()
{
	if (m_Vertices) {
		delete [] m_Vertices;
		m_Vertices = 0;
	}
	if (m_Indices) {
		delete [] m_Indices;
		m_Indices = 0;
	}
	if (m_NormalBuffer) {
		m_NormalBuffer->Release();
		m_NormalBuffer = 0;
	}
	if (m_SBuffer) {
		m_SBuffer->Release();
		m_SBuffer = 0;
	}
	if (m_TBuffer) {
		m_TBuffer->Release();
		m_TBuffer = 0;
	}
	if (m_SxTBuffer) {
		m_SxTBuffer->Release();
		m_SxTBuffer = 0;
	}
	return S_OK;
}

NVBScene::Light::Light() : m_NumKeys(0)
{
}

/*******************************************************************************

	Loading

*******************************************************************************/

HRESULT NVBScene::Load(const tstring& filename, LPDIRECT3DDEVICE9 device, GetFilePathCallback getFilePath)
{
	Free();
	m_Device = device;
	tstring fullname = (getFilePath ? (*getFilePath)(filename) : filename);
#ifdef UNICODE
	int nLen = WideCharToMultiByte(CP_ACP, 0, fullname.c_str(), -1, NULL, NULL, NULL, NULL);
	char *lpszW = new char[nLen];
	WideCharToMultiByte(CP_ACP, 0,fullname.c_str(), -1, lpszW, nLen, NULL, NULL);
	
	if (!NVBLoad(lpszW, &m_Scene, NVB_LHS)) {
		m_ErrorMessage = _T("Could not load NVB scene ") + fullname;
		return E_FAIL;
	}
#else
	if (!NVBLoad(fullname.c_str(), &m_Scene, NVB_LHS)) {
		m_ErrorMessage = _T("Could not load NVB scene ") + fullname;
		return E_FAIL;
	}
#endif UNICODE
	m_Center.x = (m_Scene.models_aabb_min.x + m_Scene.models_aabb_max.x) / 2;
	m_Center.y = (m_Scene.models_aabb_min.y + m_Scene.models_aabb_max.y) / 2;
	m_Center.z = (m_Scene.models_aabb_min.z + m_Scene.models_aabb_max.z) / 2;
	D3DXVECTOR3 diagonal(m_Scene.models_aabb_max.x - m_Scene.models_aabb_min.x, m_Scene.models_aabb_max.y - m_Scene.models_aabb_min.y, m_Scene.models_aabb_max.z - m_Scene.models_aabb_min.z);
	m_Radius = D3DXVec3Length(&diagonal) / 2;
	m_NormalLength = 0.1f * m_Radius;
	Analyze();
	HRESULT hr;
	if (FAILED(hr = LoadTextures(getFilePath)))
		return hr;
	if (FAILED(hr = LoadMeshes(getFilePath)))
		return hr;
	if (FAILED(hr = LoadCameras()))
		return hr;
	if (FAILED(hr = LoadLights()))
		return hr;
	m_NumMeshKeys = m_Scene.num_keys;
	Update(-1);
	if (FAILED(hr = CreateCoordinateAxis()))
		return hr;
	if (FAILED(hr = CreateCube()))
		return hr;
	return S_OK;
}

void NVBScene::ReleaseTextures()
{
	for (unsigned int i = 0; i < m_Textures.size(); ++i)
		if (m_Textures[i])
			m_Textures[i]->Release();
	m_Textures.clear();
}

HRESULT NVBScene::LoadTexture(const tstring& name, LPDIRECT3DTEXTURE9& texture, GetFilePathCallback getFilePath, D3DFORMAT format, D3DPOOL pool)
{
	HRESULT hr;
	tstring fullname = (getFilePath ? (*getFilePath)(name) : name);
	if (FAILED(hr = D3DXCreateTextureFromFileEx(m_Device, fullname.c_str(),
												D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT,
												0, format, pool,
												D3DX_FILTER_LINEAR, D3DX_FILTER_LINEAR,
												0, 0, 0, &texture))) {
		texture = 0;
		m_ErrorMessage = _T("Could not load texture ") + fullname;
		return hr;
	}
	return S_OK;
}

HRESULT NVBScene::LoadTextures(GetFilePathCallback getFilePath)
{
	HRESULT hr;
	ReleaseTextures();
	for (unsigned int i = 0; i < m_Scene.num_textures; ++i) {
		LPDIRECT3DTEXTURE9 tex;
#ifdef UNICODE
		int nLen = MultiByteToWideChar(CP_ACP, 0, m_Scene.textures[i].name, -1, NULL, NULL);
		TCHAR *lpszW = new TCHAR[nLen];
		MultiByteToWideChar(CP_ACP, 0, m_Scene.textures[i].name, -1, lpszW, nLen);
		
		if (FAILED(hr = LoadTexture(lpszW, tex, getFilePath, D3DFMT_DXT3)))
			return hr;
		else
			m_Textures.push_back(tex);
#else
		if (FAILED(hr = LoadTexture(m_Scene.textures[i].name, tex, getFilePath, D3DFMT_DXT3)))
			return hr;
		else
			m_Textures.push_back(tex);
#endif UNICODE
	}
	return S_OK;
}

void NVBScene::Analyze()
{
	for (unsigned int i = 0; i < m_Scene.num_nodes; ++i) {
		const nv_node* node = m_Scene.nodes[i];
		if (node->get_type() == nv_node::GEOMETRY) {
			const nv_model* model = reinterpret_cast<const nv_model*>(node);
			for (unsigned int j = 0; j < model->num_meshes; ++j) {
				if (model->meshes[j].vertices == 0)
					break;
				++m_NumMeshes;
			}
		}
		else if (node->get_type() == nv_node::CAMERA)
			++m_NumCameras;
		else if (node->get_type() == nv_node::LIGHT)
			++m_NumLights;
	} 
}

HRESULT NVBScene::LoadMeshes(GetFilePathCallback getFilePath)
{
	HRESULT hr;

	//allocate meshes
	m_Meshes = new Mesh [m_NumMeshes];
	m_VertexHasNormal = true;
	m_VertexHasColor = true;
	m_VertexHasTexture = true;
	m_VertexHasS = true;
	m_VertexHasT = true;
	m_VertexHasSxT = true;
 
	//fill in the meshes
	unsigned int currentMesh = 0;
	for (unsigned int i = 0; i < m_Scene.num_nodes; ++i) {
		const nv_node* node = m_Scene.nodes[i];
		if (node->get_type() == nv_node::GEOMETRY) {
			const nv_model* model = reinterpret_cast<const nv_model*>(node);
			for (unsigned int j = 0; j < model->num_meshes; ++j) {
				const nv_mesh& mesh = model->meshes[j];
				if (mesh.vertices == 0)
					break;
				assert((mesh.num_texcoord_sets == 0) || (mesh.texcoord_sets[0].dim == 2)); //assume texture coordinates of dimension 2
				Mesh& d3dMesh = m_Meshes[currentMesh];
				d3dMesh.m_Device = m_Device;
				d3dMesh.m_Name = node->name;
				d3dMesh.m_NumVertices = mesh.num_vertices;
				d3dMesh.m_NumTriangles = mesh.num_faces;
				d3dMesh.m_VertexHasNormal = (mesh.normals != 0);
				d3dMesh.m_VertexHasColor = (mesh.colors != 0);
				d3dMesh.m_VertexHasTexture = (mesh.num_texcoord_sets > 0) && (mesh.texcoord_sets[0].texcoords != 0);
				d3dMesh.m_VertexHasS = (mesh.num_texcoord_sets > 0) && (mesh.texcoord_sets[0].tangents != 0);
				d3dMesh.m_VertexHasT = (mesh.num_texcoord_sets > 0) && (mesh.texcoord_sets[0].binormals != 0);
				d3dMesh.m_VertexHasSxT = (mesh.normals != 0);
				d3dMesh.m_VertexHasTangentBasis = (d3dMesh.m_VertexHasS && d3dMesh.m_VertexHasT && d3dMesh.m_VertexHasSxT);

				//vertex buffer
				d3dMesh.m_Vertices = new Mesh::Vertex[d3dMesh.m_NumVertices];
				Ptr<D3DXVECTOR3[2]> normals(mesh.normals ? d3dMesh.m_NumVertices : 0);
				Ptr<D3DXVECTOR3[2]> S(d3dMesh.m_VertexHasS ? d3dMesh.m_NumVertices : 0);
				Ptr<D3DXVECTOR3[2]> T(d3dMesh.m_VertexHasT ? d3dMesh.m_NumVertices : 0);
				Ptr<D3DXVECTOR3[2]> SxT(d3dMesh.m_VertexHasSxT ? d3dMesh.m_NumVertices : 0);
				for (unsigned int k = 0; k < d3dMesh.m_NumVertices; ++k) {
					Mesh::Vertex& vert = d3dMesh.m_Vertices[k];
					vert.m_Position.x = mesh.vertices[k].x;
					vert.m_Position.y = mesh.vertices[k].y;
					vert.m_Position.z = mesh.vertices[k].z;
					if (mesh.normals) {
						vert.m_Normal.x = mesh.normals[k].x;
						vert.m_Normal.y = mesh.normals[k].y;
						vert.m_Normal.z = mesh.normals[k].z;
						normals[k][0] = vert.m_Position;
						normals[k][1] = vert.m_Position + m_NormalLength * vert.m_Normal;
					}
					if (mesh.colors)
						vert.m_Color = D3DXCOLOR(mesh.colors[k].x, mesh.colors[k].y, mesh.colors[k].z, mesh.colors[k].w);
					if (d3dMesh.m_VertexHasTexture) {
						vert.m_Texture.x = mesh.texcoord_sets[0].texcoords[2 * k];
						vert.m_Texture.y = 1 - mesh.texcoord_sets[0].texcoords[2 * k + 1];
					}
					if (d3dMesh.m_VertexHasS) {
						vert.m_S.x = mesh.texcoord_sets[0].tangents[k].x;
						vert.m_S.y = mesh.texcoord_sets[0].tangents[k].y;
						vert.m_S.z = mesh.texcoord_sets[0].tangents[k].z;
						S[k][0] = vert.m_Position;
						S[k][1] = vert.m_Position + m_NormalLength * vert.m_S;
					}
					if (d3dMesh.m_VertexHasT) {
						vert.m_T.x = - mesh.texcoord_sets[0].binormals[k].x;
						vert.m_T.y = - mesh.texcoord_sets[0].binormals[k].y;
						vert.m_T.z = - mesh.texcoord_sets[0].binormals[k].z;
						T[k][0] = vert.m_Position;
						T[k][1] = vert.m_Position + m_NormalLength * vert.m_T;
					}
					if (mesh.normals) {
						vert.m_SxT.x = mesh.normals[k].x;
						vert.m_SxT.y = mesh.normals[k].y;
						vert.m_SxT.z = mesh.normals[k].z;
						SxT[k][0] = vert.m_Position;
						SxT[k][1] = vert.m_Position + m_NormalLength * vert.m_SxT;
					}
				}
				unsigned int size = d3dMesh.m_NumVertices * sizeof(Mesh::Vertex);
				if (FAILED(hr = m_Device->CreateVertexBuffer(size, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &d3dMesh.m_VertexBuffer, NULL))) {
					m_ErrorMessage = _T("Could not create vertex buffer");
					return hr;
				}
				unsigned char* buffer;
				if (FAILED(hr = d3dMesh.m_VertexBuffer->Lock(0, size, (void**)&buffer, 0))) {
					m_ErrorMessage = _T("Could not lock vertex buffer");
					return hr;
				}
				memcpy(buffer, &d3dMesh.m_Vertices[0], size);
				if (FAILED(hr = d3dMesh.m_VertexBuffer->Unlock())) {
					m_ErrorMessage = _T("Could not unlock vertex buffer");
					return hr;
				}

				//index buffer
				unsigned int numIndices = 3 * d3dMesh.m_NumTriangles;
				d3dMesh.m_Indices = new unsigned short[numIndices];
				for (unsigned int n = 0; n < numIndices; ++n)
					d3dMesh.m_Indices[n] = mesh.faces_idx[n];
				size = numIndices * sizeof(unsigned short);
				if (FAILED(hr = m_Device->CreateIndexBuffer(size, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &d3dMesh.m_IndexBuffer, NULL))) {
					m_ErrorMessage = _T("Could not create index buffer");
					return hr;
				}
				unsigned short* indexBuffer;
				if (FAILED(hr = d3dMesh.m_IndexBuffer->Lock(0, size, reinterpret_cast<void**>(&indexBuffer), 0))) {
					m_ErrorMessage = _T("Could not lock index buffer");
					return hr;
				}
				memcpy(indexBuffer, &d3dMesh.m_Indices[0], size);
				if (FAILED(hr = d3dMesh.m_IndexBuffer->Unlock())) {
					m_ErrorMessage = _T("Could not unlock index buffer");
					return hr;
				}

				size = d3dMesh.m_NumVertices * sizeof(D3DXVECTOR3[2]);
				//normal buffer
				if (mesh.normals) {
					if (FAILED(hr = m_Device->CreateVertexBuffer(size, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &d3dMesh.m_NormalBuffer, NULL))) {
						m_ErrorMessage = _T("Could not create normal buffer");
						return hr;
					}
					if (FAILED(hr = d3dMesh.m_NormalBuffer->Lock(0, size, (void**)&buffer, 0))) {
						m_ErrorMessage = _T("Could not lock normal buffer");
						return hr;
					}
					memcpy(buffer, &normals[0][0], size);
					if (FAILED(hr = d3dMesh.m_NormalBuffer->Unlock())) {
						m_ErrorMessage = _T("Could not unlock normal buffer");
						return hr;
					}
				}

				//S buffer
				if (d3dMesh.m_VertexHasS) {
					if (FAILED(hr = m_Device->CreateVertexBuffer(size, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &d3dMesh.m_SBuffer, NULL))) {
						m_ErrorMessage = _T("Could not create S buffer");
						return hr;
					}
					if (FAILED(hr = d3dMesh.m_SBuffer->Lock(0, size, (void**)&buffer, 0))) {
						m_ErrorMessage = _T("Could not lock S buffer");
						return hr;
					}
					memcpy(buffer, &S[0][0], size);
					if (FAILED(hr = d3dMesh.m_SBuffer->Unlock())) {
						m_ErrorMessage = _T("Could not unlock S buffer");
						return hr;
					}
				}

				//T buffer
				if (d3dMesh.m_VertexHasT) {
					if (FAILED(hr = m_Device->CreateVertexBuffer(size, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &d3dMesh.m_TBuffer, NULL))) {
						m_ErrorMessage = _T("Could not create S buffer");
						return hr;
					}
					if (FAILED(hr = d3dMesh.m_TBuffer->Lock(0, size, (void**)&buffer, 0))) {
						m_ErrorMessage = _T("Could not lock T buffer");
						return hr;
					}
					memcpy(buffer, &T[0][0], size);
					if (FAILED(hr = d3dMesh.m_TBuffer->Unlock())) {
						m_ErrorMessage = _T("Could not unlock T buffer");
						return hr;
					}
				}

				//SxT buffer
				if (mesh.normals) {
					if (FAILED(hr = m_Device->CreateVertexBuffer(size, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &d3dMesh.m_SxTBuffer, NULL))) {
						m_ErrorMessage = _T("Could not create SxT buffer");
						return hr;
					}
					if (FAILED(hr = d3dMesh.m_SxTBuffer->Lock(0, size, (void**)&buffer, 0))) {
						m_ErrorMessage = _T("Could not lock SxT buffer");
						return hr;
					}
					memcpy(buffer, &SxT[0][0], size);
					if (FAILED(hr = d3dMesh.m_SxTBuffer->Unlock())) {
						m_ErrorMessage = _T("Could not unlock SxT buffer");
						return hr;
					}
				}

				if (mesh.material_id != -1) {
					//textures (assume at most one texture of each sort)
					const nv_material& material = m_Scene.materials[mesh.material_id];
					d3dMesh.m_Material = material.name;
					unsigned int diffuse = m_Scene.num_textures;
					unsigned int bump = m_Scene.num_textures;
					unsigned int normal = m_Scene.num_textures;
					unsigned int gloss = m_Scene.num_textures;
					for (unsigned int t = 0; t < material.num_textures; ++t)
						switch (m_Scene.textures[material.textures[t]].type) {
						case nv_texture::DIFFUSE:
							diffuse = material.textures[t];
							break;
						case nv_texture::BUMP:
							bump = material.textures[t];
							break;
						case nv_texture::NORMAL:
							normal = material.textures[t];
							break;
						case nv_texture::GLOSS:
							gloss = material.textures[t];
							break;
						default:
							break;
						}
					if (diffuse < m_Scene.num_textures) 
						d3dMesh.m_DiffuseMap = m_Textures[diffuse];
					if (bump < m_Scene.num_textures)
						d3dMesh.m_HeightMap = m_Textures[bump];
					if (normal < m_Scene.num_textures)
						d3dMesh.m_NormalMap = m_Textures[normal];
				}

				if (!d3dMesh.m_VertexHasNormal)
					m_VertexHasNormal = false;
				if (!d3dMesh.m_VertexHasColor)
					m_VertexHasColor = false;
				if (!d3dMesh.m_VertexHasTexture)
					m_VertexHasTexture = false;
				if (!d3dMesh.m_VertexHasS)
					m_VertexHasS = false;
				if (!d3dMesh.m_VertexHasT)
					m_VertexHasT = false;
				if (!d3dMesh.m_VertexHasSxT)
					m_VertexHasSxT = false;
				++currentMesh;
			}
		}
	}
	m_VertexHasTangentBasis = (m_VertexHasS && m_VertexHasT && m_VertexHasSxT);
	return S_OK;
}

HRESULT NVBScene::LoadCameras()
{
	m_Cameras = new Camera [m_NumCameras];
	return S_OK;
}

HRESULT NVBScene::LoadLights()
{
	m_Lights = new Light [m_NumLights];
	return S_OK;
}

/*******************************************************************************

	Updating

*******************************************************************************/

void NVBScene::Update(float time, const D3DXMATRIX* transform, unsigned int what)
{
	if (m_Scene.nodes == 0)
		return;
	if ((time >= 0) && (m_NumMeshKeys == 0))
		time = -1;
	float timeMesh;
	if (what & MESH)
		timeMesh = m_TimeMesh = time;
	else
		timeMesh = m_TimeMesh;
	unsigned int keyMesh = 0;
	float subKeyMesh = 0;
	if (timeMesh >= 0) {
		keyMesh = (unsigned int)timeMesh;
		subKeyMesh = timeMesh - keyMesh;
		keyMesh = keyMesh % (m_NumMeshKeys - 1);
	}
	float timeCamera;
	if (what & CAMERA)
		timeCamera = m_TimeCamera = time;
	else
		timeCamera = m_TimeCamera;
	unsigned int keyCamera = 0;
	float subKeyCamera = 0;
	if (timeCamera >= 0) {
		keyCamera = (unsigned int)timeCamera;
		subKeyCamera = timeCamera - keyCamera;
		keyCamera = keyCamera % (m_NumCameraKeys - 1);
	}
	float timeLight;
	if (what & LIGHT)
		timeLight = m_TimeLight = time;
	else
		timeLight = m_TimeLight;
	unsigned int keyLight = 0;
	float subKeyLight = 0;
	if (timeLight >= 0) {
		keyLight = (unsigned int)timeLight;
		subKeyLight = timeLight - keyLight;
		keyLight = keyLight % (m_NumLightKeys - 1);
	}
	unsigned int currentMesh = 0;
	unsigned int currentCamera = 0;
	unsigned int currentLight = 0;
	for (unsigned int i = 0; i < m_Scene.num_nodes; ++i) {
		const nv_node* node = m_Scene.nodes[i];
		assert(node);
		if (node->get_type() == nv_node::GEOMETRY) {
			const nv_model& model = *reinterpret_cast<const nv_model*>(node);
			D3DXMATRIX mat;
			if (0 <= timeMesh) {
				mat4 xform(model.xform);
				if (model.anim.rot) {
					quat q;
					q = model.anim.rot[keyMesh + 1];
					if (dot(model.anim.rot[keyMesh], q) < 0)
						q = - q;
					slerp_quats(q, subKeyMesh, model.anim.rot[keyMesh], q);
					q.Normalize();
					xform.set_rot(q);
				}
				if (model.anim.pos)
					xform.set_translation((1 - subKeyMesh) * model.anim.pos[keyMesh] + subKeyMesh * model.anim.pos[keyMesh + 1]);
				MatToD3DXMATRIX(xform, mat);
			}
			else
				MatToD3DXMATRIX(model.xform, mat);
			for (unsigned int j = 0; j < model.num_meshes; ++j) {
				if (model.meshes[j].vertices == 0)
					break;
				if (transform)
					m_Meshes[currentMesh].m_Transform = mat * *transform;
				else
					m_Meshes[currentMesh].m_Transform = mat;
				++currentMesh;
			}
		}
		else if (node->get_type() == nv_node::CAMERA) {
			const nv_camera& camera = *reinterpret_cast<const nv_camera*>(node);
			Camera& d3dCamera = m_Cameras[currentCamera];
			float zFar = 15 * m_Radius;
			D3DXMatrixPerspectiveFovLH(&d3dCamera.m_Projection, D3DXToRadian(camera.fov), 1.0f, 0.5f, zFar);
			vec3 lookAtPoint;
			if (camera.target == NV_BAD_IDX) {
				add(lookAtPoint, m_Scene.models_aabb_max, m_Scene.models_aabb_min);
				scale(lookAtPoint, nv_zero_5);
			}
			else {
				const nv_node& target = *m_Scene.nodes[camera.target];
				if (target.anim.num_keys > m_NumCameraKeys)
					m_NumCameraKeys = target.anim.num_keys;
				if ((0 <= timeCamera) && target.anim.pos)
					lookAtPoint = (1 - subKeyMesh) * target.anim.pos[keyCamera] + subKeyMesh * target.anim.pos[keyCamera + 1];
				else
					target.xform.get_translation(lookAtPoint);
			}
			vec3 cameraPosition;
			if (camera.anim.num_keys > m_NumCameraKeys)
				m_NumCameraKeys = camera.anim.num_keys;
			if ((0 <= timeCamera) && camera.anim.pos)
				cameraPosition = (1 - subKeyMesh) * camera.anim.pos[keyCamera] + subKeyMesh * camera.anim.pos[keyCamera + 1];
			else
				camera.xform.get_translation(cameraPosition);
			D3DXVECTOR3 eyePt(cameraPosition.x, cameraPosition.y, cameraPosition.z);
			D3DXVECTOR3 lookatPt(lookAtPoint.x, lookAtPoint.y, lookAtPoint.z);
			D3DXVECTOR3 up(0.0f, 1.0f, 0.0f);
			D3DXMatrixLookAtLH(&d3dCamera.m_View, &eyePt, &lookatPt, &up);
			if (transform) {
				D3DXMATRIX transformInverse;
				D3DXMatrixInverse(&transformInverse, 0, transform);
				d3dCamera.m_View = transformInverse * d3dCamera.m_View;
			}
		}
		else if (node->get_type() == nv_node::LIGHT) {
			const nv_light& light = *reinterpret_cast<const nv_light*>(node);
			D3DXMATRIX mat;
			if (light.anim.num_keys > m_NumLightKeys)
				m_NumLightKeys = light.anim.num_keys;
			if (0 <= timeLight) {
				mat4 xform(mat4_id);
				unsigned int key = static_cast<unsigned int>(timeLight);
				if (light.anim.rot) {
					quat q;
					q = light.anim.rot[keyLight + 1];
					if (dot(light.anim.rot[keyLight], q) < 0)
						q = - q;
					slerp_quats(q, subKeyLight, light.anim.rot[keyLight], q);
					q.Normalize();
					xform.set_rot(q);
				}
				if (light.anim.pos)
					xform.set_translation((1 - subKeyLight) * light.anim.pos[keyLight] + subKeyLight * light.anim.pos[keyLight + 1]);
				MatToD3DXMATRIX(xform, mat);
			}
			else
				MatToD3DXMATRIX(light.xform, mat);
			if (transform) {
				D3DXMATRIX transformInverse;
				D3DXMatrixInverse(&transformInverse, 0, transform);
				mat = transformInverse * mat;
			}
			Light& d3dLight = m_Lights[currentLight];
			d3dLight.m_NumKeys = light.anim.num_keys;
			d3dLight.Position.x = mat._41;
			d3dLight.Position.y = mat._42;
			d3dLight.Position.z = mat._43;
			d3dLight.Direction.x = - mat._21;
			d3dLight.Direction.y = - mat._22;
			d3dLight.Direction.z = - mat._23;
			switch (light.light) {
			default:
			case nv_light::POINT:
				d3dLight.Type = D3DLIGHT_POINT;
				break;
			case nv_light::DIRECTIONAL:
				d3dLight.Type = D3DLIGHT_DIRECTIONAL;
				break;
			case nv_light::SPOT:
				d3dLight.Type = D3DLIGHT_SPOT;
				break;
			}
			m_Lights[currentLight].m_Name = light.name;
			++currentLight;
		}
	}
}

/*******************************************************************************

	Rendering

*******************************************************************************/

HRESULT NVBScene::Draw()
{
	HRESULT hr;
	for (unsigned int i = 0; i < m_NumMeshes; ++i) {
		const Mesh& mesh = m_Meshes[i];
		m_Device->SetTexture(0, mesh.m_DiffuseMap);
		m_Device->SetTransform(D3DTS_WORLD, &mesh.m_Transform);
		if (FAILED(hr = mesh.Draw())) {
			m_ErrorMessage = _T("Could not draw mesh");
			return hr;
		}
	}
	return S_OK;
}

HRESULT NVBScene::DrawNormals(const D3DXMATRIX* projection, const D3DXMATRIX* view, const D3DXMATRIX* world)
{
	HRESULT hr;
	if (projection == 0)
		if (m_NumCameras > 0)
			projection = &m_Cameras[0].m_Projection;
		else
			return E_FAIL;
	if (view == 0)
		if (m_NumCameras > 0)
			view = &m_Cameras[0].m_View;
		else
			return E_FAIL;
	for (unsigned int i = 0; i < m_NumMeshes; ++i) {
		D3DXMATRIX meshWorld = m_Meshes[i].m_Transform;
		if (world)
			meshWorld = m_Meshes[i].m_Transform * *world;
		if (FAILED(hr = m_Meshes[i].DrawNormals(*projection, *view, meshWorld))) {
			m_ErrorMessage = _T("Could not draw mesh normals");
			return hr;
		}
	}
	return S_OK;
}

HRESULT NVBScene::DrawTangentBasis(const D3DXMATRIX* projection, const D3DXMATRIX* view, const D3DXMATRIX* world)
{
	HRESULT hr;
	if (projection == 0)
		if (m_NumCameras > 0)
			projection = &m_Cameras[0].m_Projection;
		else
			return E_FAIL;
	if (view == 0)
		if (m_NumCameras > 0)
			view = &m_Cameras[0].m_View;
		else
			return E_FAIL;
	for (unsigned int i = 0; i < m_NumMeshes; ++i) {
		D3DXMATRIX meshWorld = m_Meshes[i].m_Transform;
		if (world)
			meshWorld = m_Meshes[i].m_Transform * *world;
		if (FAILED(hr = m_Meshes[i].DrawTangentBasis(*projection, *view, meshWorld))) {
			m_ErrorMessage = _T("Could not draw mesh tangent basis");
			return hr;
		}
	}
	return S_OK;
}

HRESULT NVBScene::BaseMesh::Draw() const
{
	HRESULT hr;
	if (FAILED(hr = m_Device->SetStreamSource(0, m_VertexBuffer, 0, sizeof(Vertex))))
		return hr;
	if (FAILED(hr = m_Device->DrawPrimitive(D3DPT_TRIANGLELIST, 0, 3 * m_NumVertices)))
		return hr;
	return S_OK;
}

HRESULT NVBScene::BaseMeshIndexed::Draw() const
{
	HRESULT hr;
	if (FAILED(hr = m_Device->SetStreamSource(0, m_VertexBuffer, 0, sizeof(Vertex))))
		return hr;
	if (FAILED(hr = m_Device->SetIndices(m_IndexBuffer)))
		return hr;
	if (FAILED(hr = m_Device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, m_NumVertices, 0, m_NumTriangles)))
		return hr;
	return S_OK;
}

HRESULT NVBScene::Mesh::Draw() const
{
	HRESULT hr;
	unsigned int u = sizeof(Vertex);
	if (FAILED(hr = m_Device->SetStreamSource(0, m_VertexBuffer, 0, sizeof(Vertex))))
		return hr;
	if (FAILED(hr = m_Device->SetIndices(m_IndexBuffer)))
		return hr;
	if (FAILED(hr = m_Device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, m_NumVertices, 0, m_NumTriangles)))
		return hr;
	return S_OK;
}

HRESULT NVBScene::Mesh::DrawNormals(const D3DXMATRIX& projection, const D3DXMATRIX& view, const D3DXMATRIX& world) const
{
	if (m_VertexHasNormal) {
		m_Device->SetTransform(D3DTS_PROJECTION, &projection);
		m_Device->SetTransform(D3DTS_VIEW, &view);
		m_Device->SetTransform(D3DTS_WORLD, &world);
		m_Device->SetTexture(0, 0);
		m_Device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
		m_Device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
		m_Device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TFACTOR);
		m_Device->SetVertexShader(NULL);
		m_Device->SetFVF(D3DFVF_XYZ);
		m_Device->SetPixelShader(0);
		HRESULT hr;
		m_Device->SetRenderState(D3DRS_TEXTUREFACTOR, 0xffffff00);
		if (FAILED(hr = m_Device->SetStreamSource(0, m_NormalBuffer, 0, sizeof(D3DXVECTOR3))))
			return hr;
		if (FAILED(hr = m_Device->DrawPrimitive(D3DPT_LINELIST, 0, m_NumVertices)))
			return hr;
	}
	return S_OK;
}

HRESULT NVBScene::Mesh::DrawTangentBasis(const D3DXMATRIX& projection, const D3DXMATRIX& view, const D3DXMATRIX& world) const
{
	if (m_VertexHasS || m_VertexHasT || m_VertexHasSxT) {
		m_Device->SetTransform(D3DTS_PROJECTION, &projection);
		m_Device->SetTransform(D3DTS_VIEW, &view);
		m_Device->SetTransform(D3DTS_WORLD, &world);
		m_Device->SetTexture(0, 0);
		m_Device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
		m_Device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
		m_Device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TFACTOR);
		m_Device->SetVertexShader(NULL);
		m_Device->SetFVF(D3DFVF_XYZ);
		m_Device->SetPixelShader(0);
		HRESULT hr;
		if (m_VertexHasS) {
			m_Device->SetRenderState(D3DRS_TEXTUREFACTOR, 0xffff0000);
			if (FAILED(hr = m_Device->SetStreamSource(0, m_SBuffer, 0, sizeof(D3DXVECTOR3))))
				return hr;
			if (FAILED(hr = m_Device->DrawPrimitive(D3DPT_LINELIST, 0, m_NumVertices)))
				return hr;
		}
		if (m_VertexHasT) {
			m_Device->SetRenderState(D3DRS_TEXTUREFACTOR, 0xff00ff00);
			if (FAILED(hr = m_Device->SetStreamSource(0, m_TBuffer, 0, sizeof(D3DXVECTOR3))))
				return hr;
			if (FAILED(hr = m_Device->DrawPrimitive(D3DPT_LINELIST, 0, m_NumVertices)))
				return hr;
		}
		if (m_VertexHasSxT) {
			m_Device->SetRenderState(D3DRS_TEXTUREFACTOR, 0xff0000ff);
			if (FAILED(hr = m_Device->SetStreamSource(0, m_SxTBuffer, 0, sizeof(D3DXVECTOR3))))
				return hr;
			if (FAILED(hr = m_Device->DrawPrimitive(D3DPT_LINELIST, 0, m_NumVertices)))
				return hr;
		}
	}
	return S_OK;
}

/*******************************************************************************

	Coordinate axis

*******************************************************************************/

HRESULT NVBScene::CreateCoordinateAxis()
{
	HRESULT hr;
	m_CoordinateAxis.m_NumVertices = 3 * 2;
	if (FAILED(hr = m_Device->CreateVertexBuffer(m_CoordinateAxis.m_NumVertices * sizeof(BaseMesh::Vertex), D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &m_CoordinateAxis.m_VertexBuffer, NULL))) {
		m_ErrorMessage = _T("Could not create vertex buffer!");
		return hr;
	}
	BaseMesh::Vertex* vertices = 0;
	unsigned short* indices = 0;
	if (FAILED(hr = m_CoordinateAxis.m_VertexBuffer->Lock(0, sizeof(BaseMesh::Vertex) * m_CoordinateAxis.m_NumVertices, reinterpret_cast<void**>(&vertices), 0))) {
		m_ErrorMessage = _T("Could not lock vertex buffer!");
		return hr;
	}
	float length = 10 * m_NormalLength;
	//x-axis
	vertices->m_Position = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	vertices->m_Color = D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f);
	++vertices;
	vertices->m_Position = D3DXVECTOR3(length, 0.0f, 0.0f);
	vertices->m_Color = D3DXCOLOR(1.0f, 0.0f, 0.0f, 1.0f);
	++vertices;
	//y-axis
	vertices->m_Position = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	vertices->m_Color = D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f);
	++vertices;
	vertices->m_Position = D3DXVECTOR3(0.0f, length, 0.0f);
	vertices->m_Color = D3DXCOLOR(0.0f, 1.0f, 0.0f, 1.0f);
	++vertices;
	//z-axis
	vertices->m_Position = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	vertices->m_Color = D3DXCOLOR(0.0f, 0.0f, 1.0f, 1.0f);
	++vertices;
	vertices->m_Position = D3DXVECTOR3(0.0f, 0.0f, length);
	vertices->m_Color = D3DXCOLOR(0.0f, 0.0f, 1.0f, 1.0f);
	++vertices;
	if (FAILED(hr = m_CoordinateAxis.m_VertexBuffer->Unlock())) {
		m_ErrorMessage = _T("Could not unlock vertex buffer!");
		return hr;
	}
	return S_OK;
}

HRESULT NVBScene::DrawCoordinateAxis(const D3DXMATRIX* projection, const D3DXMATRIX* view, const D3DXMATRIX* world)
{
	HRESULT hr;
	if (projection == 0)
		if (m_NumCameras > 0)
			projection = &m_Cameras[0].m_Projection;
		else
			return E_FAIL;
	if (view == 0)
		if (m_NumCameras > 0)
			view = &m_Cameras[0].m_View;
		else
			return E_FAIL;
	m_Device->SetTransform(D3DTS_PROJECTION, projection);
	m_Device->SetTransform(D3DTS_VIEW, view);
	if (world)
		m_Device->SetTransform(D3DTS_WORLD, world);
	else {
		D3DXMATRIX id;
		D3DXMatrixIdentity(&id);
		m_Device->SetTransform(D3DTS_WORLD, &id);
	}
	m_Device->SetTexture(0, 0);
	m_Device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
	m_Device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
	m_Device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_DIFFUSE);
	m_Device->SetVertexShader(NULL);
	m_Device->SetFVF(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE2(0));
	m_Device->SetPixelShader(0);
	if (FAILED(hr = m_Device->SetStreamSource(0, m_CoordinateAxis.m_VertexBuffer, 0, sizeof(BaseMesh::Vertex))))
		return hr;
	if (FAILED(hr = m_Device->DrawPrimitive(D3DPT_LINELIST, 0, m_CoordinateAxis.m_NumVertices / 2)))
		return hr;
	return S_OK;
}

/*******************************************************************************

	Cube

*******************************************************************************/

HRESULT NVBScene::CreateCube()
{
	HRESULT hr;
	m_Cube.m_NumVertices = 4 * 6;
	m_Cube.m_NumTriangles = 2 * 6;
	if (FAILED(hr = m_Device->CreateVertexBuffer(m_Cube.m_NumVertices * sizeof(CubeMesh::Vertex), D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &m_Cube.m_VertexBuffer, NULL))) {
		m_ErrorMessage = _T("Could not create cube vertex buffer!");
		return hr;
	}
	if (FAILED(hr = m_Device->CreateIndexBuffer(3 * m_Cube.m_NumTriangles * sizeof(unsigned short), D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &m_Cube.m_IndexBuffer, NULL))) {
		m_ErrorMessage = _T("Could not create cube index buffer!");
		return hr;
	}
	CubeMesh::Vertex* vertices = 0;
	unsigned short* indices = 0;
	if (FAILED(hr = m_Cube.m_VertexBuffer->Lock(0, sizeof(CubeMesh::Vertex) * m_Cube.m_NumVertices, reinterpret_cast<void**>(&vertices), 0))) {
		m_ErrorMessage = _T("Could not lock cube vertex buffer!");
		return hr;
	}
	// -Z face
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f, 1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f, 1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f,-1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f,-1.0f,-1.0f));
	// +Z face
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f, 1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f,-1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f,-1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	// -Y face
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f, 1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f, 1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f, 1.0f,-1.0f));
	// +Y face
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f,-1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f,-1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f,-1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f,-1.0f, 1.0f));
	// -X face
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f, 1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f,-1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(1.0f,-1.0f,-1.0f));
	// +X face
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f, 1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f,-1.0f,-1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f,-1.0f, 1.0f));
	*vertices++ = CubeMesh::Vertex(D3DXVECTOR3(-1.0f, 1.0f, 1.0f));
	if (FAILED(hr = m_Cube.m_VertexBuffer->Unlock())) {
		m_ErrorMessage = _T("Could not unlock cube vertex buffer!");
		return hr;
	}
	if (FAILED(hr = m_Cube.m_IndexBuffer->Lock(0, sizeof(unsigned short) * 3 * m_Cube.m_NumTriangles, reinterpret_cast<void**>(&indices), 0))) {
		m_ErrorMessage = _T("Could not lock cube index buffer!");
		return hr;
	}
	*indices++ =  0+0;   *indices++ =  0+1;   *indices++ =  0+2;
	*indices++ =  0+2;   *indices++ =  0+3;   *indices++ =  0+0;
	*indices++ =  4+0;   *indices++ =  4+1;   *indices++ =  4+2;
	*indices++ =  4+2;   *indices++ =  4+3;   *indices++ =  4+0;
	*indices++ =  8+0;   *indices++ =  8+1;   *indices++ =  8+2;
	*indices++ =  8+2;   *indices++ =  8+3;   *indices++ =  8+0;
	*indices++ = 12+0;   *indices++ = 12+1;   *indices++ = 12+2;
	*indices++ = 12+2;   *indices++ = 12+3;   *indices++ = 12+0;
	*indices++ = 16+0;   *indices++ = 16+1;   *indices++ = 16+2;
	*indices++ = 16+2;   *indices++ = 16+3;   *indices++ = 16+0;
	*indices++ = 20+0;   *indices++ = 20+1;   *indices++ = 20+2;
	*indices++ = 20+2;   *indices++ = 20+3;   *indices++ = 20+0;
	if (FAILED(hr = m_Cube.m_IndexBuffer->Unlock())) {
		m_ErrorMessage = _T("Could not unlock cube index buffer!");
		return hr;
	}
	return S_OK;
}

HRESULT NVBScene::DrawCube()
{
	HRESULT hr;
	m_Device->SetVertexShader(NULL);
	m_Device->SetFVF(D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE3(0));
	m_Device->SetPixelShader(0);
	if (FAILED(hr = m_Device->SetStreamSource(0, m_Cube.m_VertexBuffer, 0, sizeof(CubeMesh::Vertex))))
		return hr;
	if (FAILED(hr = m_Device->SetIndices(m_Cube.m_IndexBuffer)))
		return hr;
	if (FAILED(hr = m_Device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, m_Cube.m_NumVertices, 0, m_Cube.m_NumTriangles)))
		return hr;
	return S_OK;
}

/*******************************************************************************

	Math conversion

*******************************************************************************/

void NVBScene::MatToD3DXMATRIX(const mat4& mat, D3DXMATRIX& D3DMat)
{
	D3DMat._11 = mat._11; D3DMat._12 = mat._12; D3DMat._13 = mat._13; D3DMat._14 = mat._14;
    D3DMat._21 = mat._21; D3DMat._22 = mat._22; D3DMat._23 = mat._23; D3DMat._24 = mat._24;
    D3DMat._31 = mat._31; D3DMat._32 = mat._32; D3DMat._33 = mat._33; D3DMat._34 = mat._34;
    D3DMat._41 = mat._41; D3DMat._42 = mat._42; D3DMat._43 = mat._43; D3DMat._44 = mat._44;
}
