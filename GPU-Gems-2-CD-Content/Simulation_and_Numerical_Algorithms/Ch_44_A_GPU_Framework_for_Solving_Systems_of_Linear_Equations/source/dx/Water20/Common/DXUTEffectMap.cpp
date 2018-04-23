//-----------------------------------------------------------------------------
// File: DXUTEffectMap.cpp
//
// Desc: Maps the set of standard semantics and annotations to a collection
//       of ID3DXEffect objects. 
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-----------------------------------------------------------------------------
#include "dxstdafx.h"



//-------------------------------------------------------------------------------------
DXUT_SEMANTIC CDXUTEffectMap::StringToSemantic( const char* cstrSemantic )
{
    char strSemantic[MAX_PATH+1] = {0};
    RemoveTrailingNumber( strSemantic, MAX_PATH, cstrSemantic );

    if( StringEquals( strSemantic, "Diffuse" ) )                               return DXUT_Diffuse;
    if( StringEquals( strSemantic, "Specular" ) )                              return DXUT_Specular;
    if( StringEquals( strSemantic, "Ambient" ) )                               return DXUT_Ambient;
    if( StringEquals( strSemantic, "SpecularPower" ) )                         return DXUT_SpecularPower;
    if( StringEquals( strSemantic, "Emissive" ) )                              return DXUT_Emissive;
    if( StringEquals( strSemantic, "Normal" ) )                                return DXUT_Normal;
    if( StringEquals( strSemantic, "Height" ) )                                return DXUT_Height;
    if( StringEquals( strSemantic, "Refraction" ) )                            return DXUT_Refraction;
    if( StringEquals( strSemantic, "Opacity" ) )                               return DXUT_Opacity;
    if( StringEquals( strSemantic, "Environment" ) )                           return DXUT_Environment;
    if( StringEquals( strSemantic, "EnvironmentNormal" ) )                     return DXUT_EnvironmentNormal;
    if( StringEquals( strSemantic, "Fresnel" ) )                               return DXUT_Fresnel;
    
    if( StringEquals( strSemantic, "World" ) )                                 return DXUT_World;
    if( StringEquals( strSemantic, "WorldInverse" ) )                          return DXUT_WorldInverse;
    if( StringEquals( strSemantic, "WorldInverseTranspose" ) )                 return DXUT_WorldInverseTranspose;
    if( StringEquals( strSemantic, "WorldView" ) )                             return DXUT_WorldView;
    if( StringEquals( strSemantic, "WorldViewInverse" ) )                      return DXUT_WorldViewInverse;
    if( StringEquals( strSemantic, "WorldViewInverseTranspose" ) )             return DXUT_WorldViewInverseTranspose;
    if( StringEquals( strSemantic, "WorldViewProjection" ) )                   return DXUT_WorldViewProjection;
    if( StringEquals( strSemantic, "WorldViewProjectionInverse" ) )            return DXUT_WorldViewProjectionInverse;
    if( StringEquals( strSemantic, "WorldViewProjectionInverseTranspose" ) )   return DXUT_WorldViewProjectionInverseTranspose;
    if( StringEquals( strSemantic, "View" ) )                                  return DXUT_View;
    if( StringEquals( strSemantic, "ViewInverse" ) )                           return DXUT_ViewInverse;
    if( StringEquals( strSemantic, "ViewInverseTranspose" ) )                  return DXUT_ViewInverseTranspose;
    if( StringEquals( strSemantic, "ViewProjection" ) )                        return DXUT_ViewProjection;
    if( StringEquals( strSemantic, "ViewProjectionInverse" ) )                 return DXUT_ViewProjectionInverse;
    if( StringEquals( strSemantic, "ViewProjectionInverseTranspose" ) )        return DXUT_ViewProjectionInverseTranspose;
    if( StringEquals( strSemantic, "Projection" ) )                            return DXUT_Projection;
    if( StringEquals( strSemantic, "ProjectionInverse" ) )                     return DXUT_ProjectionInverse;
    if( StringEquals( strSemantic, "ProjectionInverseTranspose" ) )            return DXUT_ProjectionInverseTranspose;

    if( StringEquals( strSemantic, "RenderTargetDimensions" ) )                return DXUT_RenderTargetDimensions;
    if( StringEquals( strSemantic, "RenderTargetClipping" ) )                  return DXUT_RenderTargetClipping;
    if( StringEquals( strSemantic, "Time" ) )                                  return DXUT_Time;
    if( StringEquals( strSemantic, "LastTime" ) )                              return DXUT_LastTime;
    if( StringEquals( strSemantic, "ElapsedTime" ) )                           return DXUT_ElapsedTime;
    if( StringEquals( strSemantic, "Position" ) )                              return DXUT_Position;
    if( StringEquals( strSemantic, "Direction" ) )                             return DXUT_Direction;
    if( StringEquals( strSemantic, "BoundingCenter" ) )                        return DXUT_BoundingCenter;
    if( StringEquals( strSemantic, "BoundingSphereSize" ) )                    return DXUT_BoundingSphereSize;
    if( StringEquals( strSemantic, "BoundingSphereMin" ) )                     return DXUT_BoundingSphereMin;
    if( StringEquals( strSemantic, "BoundingSphereMax" ) )                     return DXUT_BoundingSphereMax;
    if( StringEquals( strSemantic, "BoundingBoxSize" ) )                       return DXUT_BoundingBoxSize;
    if( StringEquals( strSemantic, "BoundingBoxMin" ) )                        return DXUT_BoundingBoxMin;
    if( StringEquals( strSemantic, "BoundingBoxMax" ) )                        return DXUT_BoundingBoxMax;
    if( StringEquals( strSemantic, "Attenuation" ) )                           return DXUT_Attenuation;
    if( StringEquals( strSemantic, "RenderColorTarget" ) )                     return DXUT_RenderColorTarget;
    if( StringEquals( strSemantic, "RenderDepthStencilTarget" ) )              return DXUT_RenderDepthStencilTarget;
    if( StringEquals( strSemantic, "UnitsScale" ) )                            return DXUT_UnitsScale;
    if( StringEquals( strSemantic, "StandardsGlobal" ) )                       return DXUT_StandardsGlobal;
        
    return DXUT_UNKNOWN_SEMANTIC;
}


//-------------------------------------------------------------------------------------
DXUT_SEMANTIC CDXUTEffectMap::StringToSemantic( const WCHAR* strSemantic )
{
    char cstr[MAX_PATH+1] = {0};
    WideCharToMultiByte( CP_ACP, 0, strSemantic, -1, cstr, MAX_PATH, NULL, NULL );
    return StringToSemantic( cstr );
}


//-------------------------------------------------------------------------------------
DXUT_OBJECT CDXUTEffectMap::StringToObject( const char* cstrObject )
{
    char strObject[MAX_PATH+1] = {0};
    RemoveTrailingNumber( strObject, MAX_PATH, cstrObject );

    if( StringEquals( strObject, "Geometry" ) )  return DXUT_Geometry;
    if( StringEquals( strObject, "Light" ) )     return DXUT_Light;
    if( StringEquals( strObject, "Camera" ) )    return DXUT_Camera;
    if( StringEquals( strObject, "Frame" ) )     return DXUT_Frame;
    
    return DXUT_UNKNOWN_OBJECT;
}


//-------------------------------------------------------------------------------------
DXUT_OBJECT CDXUTEffectMap::StringToObject( const WCHAR* strObject )
{
    char cstr[MAX_PATH+1] = {0};
    WideCharToMultiByte( CP_ACP, 0, strObject, -1, cstr, MAX_PATH, NULL, NULL );
    return StringToObject( cstr );
}


//-------------------------------------------------------------------------------------
VOID CDXUTEffectMap::Reset()
{
	D3DXMatrixIdentity( &m_matWorld );
    D3DXMatrixIdentity( &m_matView );
    D3DXMatrixIdentity( &m_matProjection );

	// Reset all the stored parameter lists
	for( UINT iSemantic = 0; iSemantic < NUM_DXUT_SEMANTICS; iSemantic++ )
	{
		for( UINT iObject = 0; iObject < NUM_DXUT_OBJECTS; iObject++ )
		{
			for( UINT iIndex = 0; iIndex < MAX_INDEX; iIndex++ )
			{
				CGrowableArray<ParamList>* pBinding = &m_Bindings[ iSemantic ][ iObject ][ iIndex ];

				// Clear nested arrays first
				for( int iParamList = 0; iParamList < pBinding->GetSize(); iParamList++ )
				{
					pBinding->GetAt( iParamList ).Reset();
				}
				
				// Remove all the bound parameter lists
				pBinding->RemoveAll();
			}
		}
	}
}


//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::SetStandardParameter( const WCHAR* strSemantic, 
                                         const WCHAR* strObject, 
                                         DWORD dwObjectIndex, 
                                         float* pData, 
                                         DWORD dwDataLen, 
                                         const WCHAR* strType, 
                                         const WCHAR* strUnits, 
                                         const WCHAR* strSpace )
{
    // Map the semantic to the standard set
    DXUT_SEMANTIC eSemantic = StringToSemantic( strSemantic );
    if( eSemantic == DXUT_UNKNOWN_SEMANTIC )
        return E_FAIL;

    // Map the object to the standard set
    DXUT_OBJECT eObject = StringToObject( strObject );
    if( eObject == DXUT_UNKNOWN_OBJECT )
        return E_FAIL;  

    return SetStandardParameter( eSemantic, eObject, dwObjectIndex, pData, dwDataLen, strType, strUnits, strSpace );
}


//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::SetStandardParameter( DXUT_SEMANTIC eSemantic, 
                                         DXUT_OBJECT eObject, 
                                         DWORD dwObjectIndex, 
                                         float* pData, 
                                         DWORD dwDataLen, 
                                         const WCHAR* strType, 
                                         const WCHAR* strUnits, 
                                         const WCHAR* strSpace )
{
    HRESULT hr;

    // TODO: remove index limits
    if( dwObjectIndex >= MAX_INDEX )
        return E_INVALIDARG;

    // TODO: handle unit and space conversions

    // Retrieve the interested handles
    CGrowableArray<ParamList>* pBindings = &m_Bindings[ eSemantic ][ eObject ][ dwObjectIndex ];
        
    for( int iList=0; iList < pBindings->GetSize(); iList++ )
    {
        ParamList& paramList = pBindings->GetAt(iList);

        for( int iParam=0; iParam < paramList.ahParameters.GetSize(); iParam++ )
        {
            V_RETURN( paramList.pEffect->SetFloatArray( paramList.ahParameters[iParam], pData, dwDataLen ) );
        }
    }

    return S_OK;
}

void ParamList::Reset() 
{ 
	SAFE_RELEASE(pEffect); 
	ahParameters.RemoveAll(); 
}


//-------------------------------------------------------------------------------------
// Investigates all the parameters, looking at semantics and annotations and placing 
// handles to these parameters within the internal database.
//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::AddEffect( ID3DXEffect* pEffect )
{
    HRESULT hr;

	if( pEffect == NULL )
		return E_INVALIDARG;

    // Get the number of parameters
    D3DXEFFECT_DESC descEffect;
    V_RETURN( pEffect->GetDesc( &descEffect ) );
    
    // Enumerate the parameters
    for( UINT iParam=0; iParam < descEffect.Parameters; iParam++ )
    {
        // Retrieve param
        D3DXHANDLE hParameter = pEffect->GetParameter( NULL, iParam );
        if( NULL == hParameter )
            return E_FAIL;

        // Grab description
        D3DXPARAMETER_DESC desc;
        V_RETURN( pEffect->GetParameterDesc( hParameter, &desc ) );

        // If this parameter doesn't have a semantic, skip to the next parameter
        if( desc.Semantic == NULL )
            continue;

        // Map the semantic to the standard set
        DXUT_SEMANTIC eSemantic = StringToSemantic( desc.Semantic );
        if( eSemantic == DXUT_UNKNOWN_SEMANTIC )
            continue;

        // Get the object annotation
        const char* cstrObject = "Geometry";
        D3DXHANDLE hAnnotation = pEffect->GetAnnotationByName( hParameter, "Object" );
        if( hAnnotation )
        {
            V_RETURN( pEffect->GetString( hAnnotation, &cstrObject ) );
        }

        // Map the object to the standard set
        DXUT_OBJECT eObject = StringToObject( cstrObject );
        if( eObject == DXUT_UNKNOWN_OBJECT )
            continue;

        // Extract the index from the semantic
        int index = 0;
        const char* strIndex = desc.Semantic + strlen(desc.Semantic)-1;

        // If there is a digit at the end of the semantic, locate the beginning of the index
        // and convert to an integer
        if( isdigit( *strIndex ) )
        {
            while( isdigit( *(strIndex-1) ) )
            {
                --strIndex;
            }

            index = atoi( strIndex );
        }

        // Check whether index is out of bounds
        if( index < 0 || index >= MAX_INDEX )
            continue;

        // Store the handle
        CGrowableArray<ParamList>* pBindings = &m_Bindings[ eSemantic ][ eObject ][ index ];
        
        bool bBound = false;
        for( int i=0; i < pBindings->GetSize(); i++ )
        {
            if( pBindings->GetAt(i).pEffect == pEffect )
            {
                // Found the containing effect for this parameter in the list, add the new handle
                pBindings->GetAt(i).ahParameters.Add( hParameter );
                bBound = true;
                break;
            }
        }

        if( !bBound )
        {
            // This parameter belongs to a new effect
            ParamList newParamList;
            newParamList.pEffect = pEffect;
			pEffect->AddRef();
            newParamList.ahParameters.Add( hParameter );
            pBindings->Add( newParamList );
        }
       
    }
    
    return S_OK;
}


//-------------------------------------------------------------------------------------
// Removes all instances of this effect from the binding list
//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::RemoveEffect( ID3DXEffect* pEffect )
{
	if( pEffect == NULL )
		return E_INVALIDARG;

	// Search through the list of registered semantics and remove all items
	// assigned to the given effect
	for( UINT iSemantic = 0; iSemantic < NUM_DXUT_SEMANTICS; iSemantic++ )
	{
		for( UINT iObject = 0; iObject < NUM_DXUT_OBJECTS; iObject++ )
		{
			for( UINT iIndex = 0; iIndex < MAX_INDEX; iIndex++ )
			{
				CGrowableArray<ParamList>* pBinding = &m_Bindings[ iSemantic ][ iObject ][ iIndex ];

				// Clear nested arrays first
				for( int iParamList = 0; iParamList < pBinding->GetSize(); iParamList++ )
				{
					ParamList& rParamList = pBinding->GetAt( iParamList );

					if( rParamList.pEffect == pEffect )
						rParamList.Reset();
				}
			}
		}
	}

	return S_OK;
}


//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::SetWorldMatrix( D3DXMATRIXA16* pWorldMatrix, const WCHAR* strUnits )
{
    m_matWorld = *pWorldMatrix;
    return UpdateTransforms( DXUT_World );
}


//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::SetViewMatrix( D3DXMATRIXA16* pViewMatrix, const WCHAR* strUnits )
{
    m_matView = *pViewMatrix;
    return UpdateTransforms( DXUT_View );
}


//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::SetProjectionMatrix( D3DXMATRIXA16* pProjectionMatrix )
{
    m_matProjection = *pProjectionMatrix;
    return UpdateTransforms( DXUT_Projection );
}


//-------------------------------------------------------------------------------------
HRESULT CDXUTEffectMap::UpdateTransforms( DXUT_SEMANTIC eSemantic, const WCHAR* strUnits )
{
    HRESULT hr;

    // Determine which transforms are required by the effect
    bool bEffectContains[ NUM_DXUT_SEMANTICS ] = {0};
    for( int iTransform = DXUT_World; iTransform <= DXUT_ProjectionInverseTranspose; iTransform++ )
    {
        bEffectContains[iTransform] = ( m_Bindings[ iTransform ][ DXUT_Geometry ][ 0 ].GetSize() > 0 );
    }

    if( eSemantic == DXUT_World )
    {
        // World matrix permutations
        if( bEffectContains[ DXUT_World ] ||
            bEffectContains[ DXUT_WorldInverse ] ||
            bEffectContains[ DXUT_WorldInverseTranspose ] )
        {
            V_RETURN( SetStandardParameter( DXUT_World, DXUT_Geometry, 0, (float*)&m_matWorld, 16, L"", strUnits, L"World" ) );

            if( bEffectContains[ DXUT_WorldInverse ] ||
                bEffectContains[ DXUT_WorldInverseTranspose ] )
            {
                D3DXMATRIXA16 matWorldInverse;
                D3DXMatrixInverse( &matWorldInverse, NULL, &m_matWorld );
                V_RETURN( SetStandardParameter( DXUT_WorldInverse, DXUT_Geometry, 0, (float*)&matWorldInverse, 16, L"", strUnits, L"World" ) );
            
                if( bEffectContains[ DXUT_WorldInverseTranspose ] )
                {
                    D3DXMATRIXA16 matWorldInverseTranspose;
                    D3DXMatrixTranspose( &matWorldInverseTranspose, &matWorldInverse );
                    V_RETURN( SetStandardParameter( DXUT_WorldInverseTranspose, DXUT_Geometry, 0, (float*)&matWorldInverseTranspose, 16, L"", strUnits, L"World" ) );
                }
            }
        }
    }

    if( eSemantic == DXUT_World ||
        eSemantic == DXUT_View )
    {
        // WorldView matrix permutations
        if( bEffectContains[ DXUT_WorldView ] ||
            bEffectContains[ DXUT_WorldViewInverse ] ||
            bEffectContains[ DXUT_WorldViewInverseTranspose ] )
        {
            D3DXMATRIXA16 matWorldView;
            D3DXMatrixMultiply( &matWorldView, &m_matWorld, &m_matView );
            V_RETURN( SetStandardParameter( DXUT_WorldView, DXUT_Geometry, 0, (float*)&matWorldView, 16, L"", strUnits, L"World" ) );
        
            if( bEffectContains[ DXUT_WorldViewInverse ] ||
                bEffectContains[ DXUT_WorldViewInverseTranspose ] )
            {
                D3DXMATRIXA16 matWorldViewInverse;
                D3DXMatrixInverse( &matWorldViewInverse, NULL, &matWorldView );    
                V_RETURN( SetStandardParameter( DXUT_WorldViewInverse, DXUT_Geometry, 0, (float*)&matWorldViewInverse, 16, L"", strUnits, L"World" ) );
            
                if( bEffectContains[ DXUT_WorldViewInverseTranspose ] )
                {
                    D3DXMATRIXA16 matWorldViewInverseTranspose;
                    D3DXMatrixTranspose( &matWorldViewInverseTranspose, &matWorldViewInverse );
                    V_RETURN( SetStandardParameter( DXUT_WorldViewInverseTranspose, DXUT_Geometry, 0, (float*)&matWorldViewInverseTranspose, 16, L"", strUnits, L"World" ) );
                }
            }
        }
    }

    if( eSemantic == DXUT_World ||
        eSemantic == DXUT_View ||
        eSemantic == DXUT_Projection )
    {
        // WorldViewProjection matrix permutations
        if( bEffectContains[ DXUT_WorldViewProjection ] ||
            bEffectContains[ DXUT_WorldViewProjectionInverse ] ||
            bEffectContains[ DXUT_WorldViewProjectionInverseTranspose ] )
        {
            D3DXMATRIXA16 matWorldViewProjection;
            D3DXMatrixMultiply( &matWorldViewProjection, &m_matWorld, &m_matView );
            D3DXMatrixMultiply( &matWorldViewProjection, &matWorldViewProjection, &m_matProjection );

            V_RETURN( SetStandardParameter( DXUT_WorldViewProjection, DXUT_Geometry, 0, (float*)&matWorldViewProjection, 16, L"", strUnits, L"World" ) );
        
            if( bEffectContains[ DXUT_WorldViewProjectionInverse ] ||
                bEffectContains[ DXUT_WorldViewProjectionInverseTranspose ] )
            {
                D3DXMATRIXA16 matWorldViewProjectionInverse;
                D3DXMatrixInverse( &matWorldViewProjectionInverse, NULL, &matWorldViewProjection );    
                V_RETURN( SetStandardParameter( DXUT_WorldViewProjectionInverse, DXUT_Geometry, 0, (float*)&matWorldViewProjectionInverse, 16, L"", strUnits, L"World" ) );
            
                if( bEffectContains[ DXUT_WorldViewProjectionInverseTranspose ] )
                {
                    D3DXMATRIXA16 matWorldViewProjectionInverseTranspose;
                    D3DXMatrixTranspose( &matWorldViewProjectionInverseTranspose, &matWorldViewProjectionInverse );
                    V_RETURN( SetStandardParameter( DXUT_WorldViewProjectionInverseTranspose, DXUT_Geometry, 0, (float*)&matWorldViewProjectionInverseTranspose, 16, L"", strUnits, L"World" ) );
                }
            }
        }
    }

    if( eSemantic == DXUT_View )
    {
        // View matrix permutations
        if( bEffectContains[ DXUT_View ] ||
            bEffectContains[ DXUT_ViewInverse ] ||
            bEffectContains[ DXUT_ViewInverseTranspose ] )
        {
            V_RETURN( SetStandardParameter( DXUT_View, DXUT_Geometry, 0, (float*)&m_matView, 16, L"", strUnits, L"World" ) );

            if( bEffectContains[ DXUT_ViewInverse ] ||
                bEffectContains[ DXUT_ViewInverseTranspose ] )
            {
                D3DXMATRIXA16 matViewInverse;
                D3DXMatrixInverse( &matViewInverse, NULL, &m_matView );
                V_RETURN( SetStandardParameter( DXUT_ViewInverse, DXUT_Geometry, 0, (float*)&matViewInverse, 16, L"", strUnits, L"World" ) );
            
                if( bEffectContains[ DXUT_ViewInverseTranspose ] )
                {
                    D3DXMATRIXA16 matViewInverseTranspose;
                    D3DXMatrixTranspose( &matViewInverseTranspose, &matViewInverse );
                    V_RETURN( SetStandardParameter( DXUT_ViewInverseTranspose, DXUT_Geometry, 0, (float*)&matViewInverseTranspose, 16, L"", strUnits, L"World" ) );
                }
            }
        }
    }

    if( eSemantic == DXUT_View ||
        eSemantic == DXUT_Projection )
    {
        // ViewProjection matrix permutations
        if( bEffectContains[ DXUT_ViewProjection ] ||
            bEffectContains[ DXUT_ViewProjectionInverse ] ||
            bEffectContains[ DXUT_ViewProjectionInverseTranspose ] )
        {
            D3DXMATRIXA16 matViewProjection;
            D3DXMatrixMultiply( &matViewProjection, &m_matView, &m_matProjection );
            V_RETURN( SetStandardParameter( DXUT_ViewProjection, DXUT_Geometry, 0, (float*)&matViewProjection, 16, L"", strUnits, L"World" ) );
        
            if( bEffectContains[ DXUT_ViewProjectionInverse ] ||
                bEffectContains[ DXUT_ViewProjectionInverseTranspose ] )
            {
                D3DXMATRIXA16 matViewProjectionInverse;
                D3DXMatrixInverse( &matViewProjectionInverse, NULL, &matViewProjection );    
                V_RETURN( SetStandardParameter( DXUT_ViewProjectionInverse, DXUT_Geometry, 0, (float*)&matViewProjectionInverse, 16, L"", strUnits, L"World" ) );
            
                if( bEffectContains[ DXUT_ViewProjectionInverseTranspose ] )
                {
                    D3DXMATRIXA16 matViewProjectionInverseTranspose;
                    D3DXMatrixTranspose( &matViewProjectionInverseTranspose, &matViewProjectionInverse );
                    V_RETURN( SetStandardParameter( DXUT_ViewProjectionInverseTranspose, DXUT_Geometry, 0, (float*)&matViewProjectionInverseTranspose, 16, L"", strUnits, L"World" ) );
                }
            }
        }
    }

    if( eSemantic == DXUT_Projection )
    {
        // Projection matrix permutations
        if( bEffectContains[ DXUT_Projection ] ||
            bEffectContains[ DXUT_ProjectionInverse ] ||
            bEffectContains[ DXUT_ProjectionInverseTranspose ] )
        {
            V_RETURN( SetStandardParameter( DXUT_Projection, DXUT_Geometry, 0, (float*)&m_matProjection, 16, L"", strUnits, L"World" ) );

            if( bEffectContains[ DXUT_ProjectionInverse ] ||
                bEffectContains[ DXUT_ProjectionInverseTranspose ] )
            {
                D3DXMATRIXA16 matProjectionInverse;
                D3DXMatrixInverse( &matProjectionInverse, NULL, &m_matProjection );
                V_RETURN( SetStandardParameter( DXUT_ProjectionInverse, DXUT_Geometry, 0, (float*)&matProjectionInverse, 16, L"", strUnits, L"World" ) );
            
                if( bEffectContains[ DXUT_ProjectionInverseTranspose ] )
                {
                    D3DXMATRIXA16 matProjectionInverseTranspose;
                    D3DXMatrixTranspose( &matProjectionInverseTranspose, &matProjectionInverse );
                    V_RETURN( SetStandardParameter( DXUT_ProjectionInverseTranspose, DXUT_Geometry, 0, (float*)&matProjectionInverseTranspose, 16, L"", strUnits, L"World" ) );
                }
            }
        }
    }

    return S_OK;
}

