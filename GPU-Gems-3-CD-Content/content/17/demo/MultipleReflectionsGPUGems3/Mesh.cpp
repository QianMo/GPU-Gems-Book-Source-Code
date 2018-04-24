#include "dxstdafx.h"
#include "Mesh.h"
#include "L.h"

extern IDirect3DCubeTexture9*	g_CubeTexture1;
extern IDirect3DCubeTexture9*	g_CubeTexture2;
extern IDirect3DCubeTexture9*	g_CubeTexture3;
extern IDirect3DCubeTexture9*	g_CubeTexture4;
extern IDirect3DCubeTexture9*	g_CubeTexture5;
extern IDirect3DCubeTexture9*	g_CubeTexture6;
extern IDirect3DDevice9*		g_pd3dDevice;
extern D3DXVECTOR4 g_eyePos;
extern D3DXVECTOR4 g_refPos;
extern D3DXMATRIXA16 g_viewMatrix;
extern D3DXMATRIXA16 g_projMatrix;
extern D3DXMATRIXA16 g_worldMatrix;
extern ID3DXEffect*	g_basicShaders;
extern ID3DXEffect*	g_reflectionShaders;
extern int g_linearMinIter;
extern int g_linearMaxIter;
extern int g_secantIter;
extern int g_rayDepth;
extern void setWorldViewProj(D3DXMATRIXA16& mWorld, D3DXMATRIXA16& mView, D3DXMATRIXA16& mProj, ID3DXEffect* effect );

void Material::beginMaterial(enum RenderMode mode)
{
	HRESULT hr;

	if(mode == RENDERMODE_FINAL && !basicFinalRender)
		currentEffect = g_reflectionShaders;
	else
		currentEffect = g_basicShaders;

	switch(mode)
	{
	case RENDERMODE_COLOR:
		V( currentEffect->SetTechnique(colorRenderTechnique));break;
	case RENDERMODE_NORMAL:
		V( currentEffect->SetTechnique(normalRenderTechnique));break;
	case RENDERMODE_FINAL:
		V( currentEffect->SetTechnique(finalRenderTechnique));break;
	}

	UINT uPasses;
	V( currentEffect->Begin( &uPasses, 0 ) );
	V( currentEffect->BeginPass( 0 ) );	// only one pass exists
	
	V( currentEffect->SetVector( "eyePos", &g_eyePos ) );
	V( currentEffect->SetVector( "referencePos", &g_refPos ) );
	V( currentEffect->SetVector( "F0", &F0));
	V( currentEffect->SetFloat( "N0", N0));

	if(mode == RENDERMODE_FINAL && !basicFinalRender)
	{
		V( currentEffect->SetInt("MIN_LIN_ITERATIONCOUNT", g_linearMinIter));
		V( currentEffect->SetInt("MAX_LIN_ITERATIONCOUNT", g_linearMaxIter));
		V( currentEffect->SetInt("SECANT_ITERATIONCOUNT", g_secantIter));
		V( currentEffect->SetInt("MAX_RAY_DEPTH", g_rayDepth));
		V( currentEffect->SetTexture("envCube1", g_CubeTexture1));
		V( currentEffect->SetTexture("envCube2", g_CubeTexture2));
		V( currentEffect->SetTexture("envCube3", g_CubeTexture3));
		V( currentEffect->SetTexture("envCube4", g_CubeTexture4));
		V( currentEffect->SetTexture("envCube5", g_CubeTexture5));
		V( currentEffect->SetTexture("envCube6", g_CubeTexture6));
	}
	else
	{
		if(colorTexture != NULL)
			V( g_basicShaders->SetTexture("colorMap", colorTexture));

		V( currentEffect->SetTexture("envCube", g_CubeTexture3));
	}

	setWorldViewProj(g_worldMatrix, g_viewMatrix, g_projMatrix, currentEffect);
}

void Material::endMaterial()
{
	HRESULT hr;
	V( currentEffect->EndPass() );
	V( currentEffect->End() );
}

void Material::loadTexture()
{
	if(colorTextureName != "")
	{
		char filename[256];
		strcpy(filename, "Media//Textures//");
		strcat(filename, colorTextureName);
		HRESULT hr;
		V( D3DXCreateTextureFromFile(g_pd3dDevice, L::l+filename, &colorTexture) );
	}
}

Mesh::Mesh(LPCWSTR fileName, float preferredDiameter, D3DXVECTOR3 offset)
{
	originalDiameter = 1;
	this->preferredDiameter = preferredDiameter;
	originalSize = D3DXVECTOR3(1,1,1); 
	position = D3DXVECTOR3(0,0,0); 
	containerSize = D3DXVECTOR3(1,1,1); 
	numMaterials = 0;
	pMesh = NULL;

	D3DXMatrixIdentity(&rotation);

	Load(fileName);
	Move(offset);
	
	//V( D3DXCreateTextureFromFile(g_pd3dDevice, texFileName, &pMeshTexture) );
}

Mesh::~Mesh()
{
	delete[] materials;
    SAFE_RELEASE( pMesh );
}

void Mesh::Move(D3DXVECTOR3 offset, bool bContainerOnly /*= false*/) 
{
	position += offset;

	if ( bContainerOnly )	// keep some distance from the walls
	{
		D3DXVECTOR3 maxOffset = containerSize * 0.99f - GetMeshSize();
		D3DXVECTOR3 minOffset = -maxOffset;

		if (position.x > maxOffset.x) position.x = maxOffset.x;
		if (position.y > maxOffset.y) position.y = maxOffset.y;
		if (position.z > maxOffset.z) position.z = maxOffset.z;

		if (position.x < minOffset.x) position.x = minOffset.x;
		if (position.y < minOffset.y) position.y = minOffset.y;
		if (position.z < minOffset.z) position.z = minOffset.z;
	}
}

HRESULT Mesh::CalculateMeshSize( )
{
	// default size
	originalSize = D3DXVECTOR3(1,1,1); 
	originalDiameter = 1;

	IDirect3DVertexBuffer9* pMeshVB = NULL;
	D3DXVECTOR3 minPos, maxPos;
	BYTE* pVertices;

	// Lock the vertex buffer to generate a simple bounding box

	hr = pMesh->GetVertexBuffer( &pMeshVB );
	if( SUCCEEDED( hr ) )
	{
        hr = pMeshVB->Lock( 0, 0, (void**)&pVertices, D3DLOCK_NOSYSLOCK );
		if( SUCCEEDED(hr) )
		{
			DWORD numVerts = pMesh->GetNumVertices();
			D3DVERTEXELEMENT9 decl[10];
			pMesh->GetDeclaration(decl);
			DWORD vertSize = D3DXGetDeclVertexSize(decl, 0);		// calculate vertex size

			/*D3DXVECTOR3 *vertices = (D3DXVECTOR3*)pVertices;
			for (DWORD i = 0; i < numVerts; i++)						// loop through the vertices
			{
				D3DXVECTOR3 *vPtr = vertices;
				vertices = (D3DXVECTOR3*) ((BYTE*)vertices + vertSize);
			}*/

			V( D3DXComputeBoundingBox( ( D3DXVECTOR3*)pVertices,
										numVerts,
										vertSize,
										&minPos,
										&maxPos ) );

			D3DXVECTOR3 vCenter = ( minPos + maxPos ) / 2.0f;

			// eliminating offset from the mesh

			
			for (DWORD i = 0; i < numVerts; i++)						// loop through the vertices
			{
				D3DXVECTOR3 *vPtr=(D3DXVECTOR3*)pVertices;				
				*vPtr -= vCenter;										// eliminating offset
				pVertices += vertSize;									// set pointer to next vertex
			}

			pMeshVB->Unlock();
			pMeshVB->Release();

			// size of the object
			originalSize = ( maxPos - minPos ) / 2.0f;
			// object "diameter" is calculated from the size of the bounding box only
			originalDiameter = sqrt( originalSize.x * originalSize.x + 
									 originalSize.y * originalSize.y +
									 originalSize.z * originalSize.z) / 1.732f;
		}
	}
	
	return hr;
}

void Mesh::Load( LPCWSTR fileName )
{  
	SAFE_RELEASE( pMesh );
    pMesh = NULL;

    HRESULT hr;

	LPD3DXBUFFER EffectInstances;

    if (FAILED( D3DXLoadMeshFromX(fileName, D3DXMESH_MANAGED, g_pd3dDevice, NULL, NULL, &EffectInstances, &numMaterials, &pMesh) ))
	{
		MessageBox(NULL, L"Media not found!", fileName, MB_ICONEXCLAMATION);
		exit(-1);
	}

	D3DXEFFECTINSTANCE* shaderAttr = (D3DXEFFECTINSTANCE*)EffectInstances->GetBufferPointer();
	materials = new Material[numMaterials];
	for(int i = 0; i< numMaterials; i++)
	{
		D3DXEFFECTDEFAULT* attrArray = shaderAttr[i].pDefaults;
		int nParams = shaderAttr[i].NumDefaults;
		for(int iParams=0; iParams < nParams; iParams++)
		{
			if(strcmp(attrArray[iParams].pParamName, "colorRenderTechnique") == 0)
			{
				strcpy(materials[i].colorRenderTechnique, (const char*)attrArray[iParams].pValue);                
			}
			if(strcmp(attrArray[iParams].pParamName, "normalRenderTechnique") == 0)
			{
				strcpy(materials[i].normalRenderTechnique, (const char*)attrArray[iParams].pValue);                
			}
			if(strcmp(attrArray[iParams].pParamName, "finalRenderTechnique") == 0)
			{
				strcpy(materials[i].finalRenderTechnique, (const char*)attrArray[iParams].pValue);                
			}
			if(strcmp(attrArray[iParams].pParamName, "colorTextureName") == 0)
			{
				strcpy(materials[i].colorTextureName, (const char*)attrArray[iParams].pValue);
				materials[i].loadTexture();
			}
			if(strcmp(attrArray[iParams].pParamName, "F0") == 0)
			{
				float* f0 = ((float*)attrArray[iParams].pValue);
				materials[i].F0 = D3DXVECTOR4(f0[0], f0[1], f0[2], 1);                
			}
			if(strcmp(attrArray[iParams].pParamName, "N0") == 0)
			{
				materials[i].N0 =  *((float*)attrArray[iParams].pValue);                
			}
			if(strcmp(attrArray[iParams].pParamName, "basicMaterial") == 0)
			{
				float bFn = *((float*)attrArray[iParams].pValue);
				if(bFn == 0)
					materials[i].basicFinalRender = false;                
				if(bFn == 1)
					materials[i].basicFinalRender = true;  
			}
		}
	}
		
	if ( FAILED( CalculateMeshSize() ))
	{
		MessageBox(NULL, L"Could not calculate bounding box!\nUsing original mesh size...",
			fileName, MB_ICONEXCLAMATION);
	}   
/*
    // Make sure there are normals which are required for lighting
    if( !(pMesh->GetFVF() & D3DFVF_NORMAL) )
    {
        ID3DXMesh* pTempMesh;
        V( pMesh->CloneMeshFVF( pMesh->GetOptions(), 
                                  pMesh->GetFVF() | D3DFVF_NORMAL, 
                                  g_pd3dDevice, &pTempMesh ) );
        V( D3DXComputeNormals( pTempMesh, NULL ) );

        SAFE_RELEASE( pMesh );
        pMesh = pTempMesh;
    } */   
}

HRESULT Mesh::Draw(enum RenderMode mode)
{
	HRESULT hr;
    // Set and draw each of the materials in the mesh
    for( DWORD iMaterial = 0; iMaterial < numMaterials; iMaterial++ )
    {
		materials[iMaterial].beginMaterial(mode);
        V( pMesh->DrawSubset( iMaterial ) );
		materials[iMaterial].endMaterial();
    }

    return S_OK;
}