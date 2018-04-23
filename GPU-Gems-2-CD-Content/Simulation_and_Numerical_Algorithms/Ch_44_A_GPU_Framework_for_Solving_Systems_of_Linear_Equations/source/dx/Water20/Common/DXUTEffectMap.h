//-----------------------------------------------------------------------------
// File: DXUTEffectMap.h
//
// Desc: Maps the set of standard semantics and annotations to a collection
//       of ID3DXEffect objects. 
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once
#ifndef DXUT_EFFECTMAP_H
#define DXUT_EFFECTMAP_H

const int MAX_INDEX = 5;

enum DXUT_SEMANTIC
{
    DXUT_UNKNOWN_SEMANTIC,

    DXUT_Diffuse,
    DXUT_Specular,
    DXUT_Ambient,
    DXUT_SpecularPower,
    DXUT_Emissive,
    DXUT_Normal,
    DXUT_Height,
    DXUT_Refraction,
    DXUT_Opacity,
    DXUT_Environment,
    DXUT_EnvironmentNormal,
    DXUT_Fresnel,

    DXUT_World,
    DXUT_WorldInverse,
    DXUT_WorldInverseTranspose,
    DXUT_WorldView,
    DXUT_WorldViewInverse,
    DXUT_WorldViewInverseTranspose,
    DXUT_WorldViewProjection,
    DXUT_WorldViewProjectionInverse,
    DXUT_WorldViewProjectionInverseTranspose,
    DXUT_View,
    DXUT_ViewInverse,
    DXUT_ViewInverseTranspose,
    DXUT_ViewProjection,
    DXUT_ViewProjectionInverse,
    DXUT_ViewProjectionInverseTranspose,
    DXUT_Projection,
    DXUT_ProjectionInverse,
    DXUT_ProjectionInverseTranspose,

    DXUT_RenderTargetDimensions,
    DXUT_RenderTargetClipping,
    DXUT_Time,
    DXUT_LastTime,
    DXUT_ElapsedTime,
    DXUT_Position,
    DXUT_Direction,
    DXUT_BoundingCenter,
    DXUT_BoundingSphereSize,
    DXUT_BoundingSphereMin,
    DXUT_BoundingSphereMax,
    DXUT_BoundingBoxSize,
    DXUT_BoundingBoxMin,
    DXUT_BoundingBoxMax,
    DXUT_Attenuation,
    DXUT_RenderColorTarget,
    DXUT_RenderDepthStencilTarget,
    DXUT_UnitsScale,
    DXUT_StandardsGlobal,

    NUM_DXUT_SEMANTICS
};

enum DXUT_OBJECT
{
    DXUT_UNKNOWN_OBJECT,

    DXUT_Geometry,
    DXUT_Light,
    DXUT_Camera,
    DXUT_Frame,

    NUM_DXUT_OBJECTS
};


struct ParamList
{
	ID3DXEffect* pEffect;
    CGrowableArray< D3DXHANDLE > ahParameters;

	void Reset();
};


class CDXUTEffectMap
{
public:
	CDXUTEffectMap() { Reset(); }
	~CDXUTEffectMap() { Reset(); }
	
	VOID    Reset();

	HRESULT AddEffect( ID3DXEffect* pEffect );
	HRESULT RemoveEffect( ID3DXEffect* pEffect );

	HRESULT SetStandardParameter( const WCHAR* strSemantic, const WCHAR* strObject, DWORD dwObjectIndex, float* pData, DWORD dwDataLen, const WCHAR* strType = NULL, const WCHAR* strUnits = NULL, const WCHAR* strSpace = NULL ); 
    HRESULT SetStandardParameter( DXUT_SEMANTIC eSemantic, DXUT_OBJECT eObject, DWORD dwObjectIndex, float* pData, DWORD dwDataLen, const WCHAR* strType = NULL, const WCHAR* strUnits = NULL, const WCHAR* strSpace = NULL ); 

	HRESULT SetWorldMatrix( D3DXMATRIXA16* pWorldMatrix, const WCHAR* strUnits = NULL );
    HRESULT SetViewMatrix( D3DXMATRIXA16* pViewMatrix, const WCHAR* strUnits = NULL );
    HRESULT SetProjectionMatrix( D3DXMATRIXA16* pProjectionMatrix );
    
	static DXUT_SEMANTIC StringToSemantic( const char* cstrSemantic );
	static DXUT_SEMANTIC StringToSemantic( const WCHAR* strSemantic );
	static DXUT_OBJECT StringToObject( const char* cstrObject );
	static DXUT_OBJECT StringToObject( const WCHAR* strObject );

private:
	HRESULT UpdateTransforms( DXUT_SEMANTIC eSemantic, const WCHAR* strUnits = L"m" );

	D3DXMATRIXA16 m_matWorld;
    D3DXMATRIXA16 m_matView;
    D3DXMATRIXA16 m_matProjection;

	// Database of DXUTEffect object parameter handles which are indexed accoring to 
    // Semantic, Object annotation, index, and containing mesh pointer
    CGrowableArray<ParamList> m_Bindings[ NUM_DXUT_SEMANTICS ][ NUM_DXUT_OBJECTS ][ MAX_INDEX ];
};


//-------------------------------------------------------------------------------------
inline bool StringBegins( const char* strTest, const char* strStartsWith ) 
{ 
    return ( 0 == _strnicmp( strTest, strStartsWith, strlen(strStartsWith) ) ); 
}


//-------------------------------------------------------------------------------------
inline bool StringEquals( const char* strTest, const char* strEquals ) 
{ 
    return( 0 == _stricmp( strTest, strEquals ) ); 
}


//-------------------------------------------------------------------------------------
inline void RemoveTrailingNumber( char* strWithoutNumber, DWORD dwBufLen, const char* str )
{
    strncpy( strWithoutNumber, str, dwBufLen );
    
    char* strIndex = strWithoutNumber + strlen(strWithoutNumber);

    // If there is a digit at the end of the semantic discard it
    while( isdigit( *(strIndex-1) ) )
    {
        --strIndex;
    }

    *strIndex = 0;
}



#endif //DXUT_EFFECTMAP_H