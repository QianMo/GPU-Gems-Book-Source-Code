//----------------------------------------------------------------------------------
// File:   TextureLibrary.cpp
// Author: Bryan Dudash
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

#include "DXUT.h"
#include "TextureLibray.h"

TextureLibrary *TextureLibrary::m_Singleton = NULL;

TextureLibrary::TextureLibrary()
{
}

void TextureLibrary::AddTexture(std::string name,ID3D10Texture2D*tex,ID3D10ShaderResourceView *srv)
{
    if(GetTexture(name)) return;

    tex->AddRef();
    srv->AddRef();

    // all is good?  Then add to our list
    m_Textures.insert(TextureMap::value_type(name,tex));
    m_ShaderResourceViews.insert(SRVMap::value_type(name,srv));
}

ID3D10Texture2D *TextureLibrary::GetTexture(std::string id)
{
    for(TextureMap::iterator it = m_Textures.begin();it!=m_Textures.end();it++)
    {
        if(strcmp(it->first.c_str(),id.c_str()) == 0)
        {
            return it->second;
        }
    }
    return NULL;
}

int TextureLibrary::GetTextureIndex(std::string id)
{
    int i=0;
    for(TextureMap::iterator it = m_Textures.begin();it!=m_Textures.end();i++,it++)
    {
        if(strcmp(it->first.c_str(),id.c_str()) == 0)
        {
            return i;
        }
    }
    return 0;
}

ID3D10Texture2D *TextureLibrary::GetTexture(int id)
{
    int i=0;
    for(TextureMap::iterator it = m_Textures.begin();i!= id || it!=m_Textures.end();i++,it++)
    {
        if(i==id)
        {
            return it->second;
        }
    }
    return NULL;
}

ID3D10ShaderResourceView *TextureLibrary::GetShaderResourceView(std::string id)
{
    for(SRVMap::iterator it = m_ShaderResourceViews.begin();it!=m_ShaderResourceViews.end();it++)
    {
        if(strcmp(it->first.c_str(),id.c_str()) == 0)
        {
            return it->second;
        }
    }
    return NULL;
}

ID3D10ShaderResourceView *TextureLibrary::GetShaderResourceView(int id)
{
    int i=0;
    for(SRVMap::iterator it = m_ShaderResourceViews.begin();i!= id || it!=m_ShaderResourceViews.end();i++,it++)
    {
        if(i==id)
        {
            return it->second;
        }
    }
    return NULL;
}

void TextureLibrary::Release()
{
    for(TextureMap::iterator it = m_Textures.begin();it!= m_Textures.end();it++)
    {
        if(it->second)    SAFE_RELEASE(it->second);
    }
    m_Textures.clear();

    for(SRVMap::iterator it = m_ShaderResourceViews.begin();it!= m_ShaderResourceViews.end();it++)
    {
        if(it->second)    SAFE_RELEASE(it->second);
    }
    m_ShaderResourceViews.clear();
}
