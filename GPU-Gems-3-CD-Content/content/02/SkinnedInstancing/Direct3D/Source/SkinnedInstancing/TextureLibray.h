//----------------------------------------------------------------------------------
// File:   TextureLibrary.h
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

#pragma once 

#include <string>
#include <map>

typedef std::map<std::string,ID3D10Texture2D*> TextureMap;
typedef std::map<std::string,ID3D10ShaderResourceView*> SRVMap;

/*
    This is just a cache from the scene containing all textures, and resource views for them.  
*/
class TextureLibrary
{
public:

    static TextureLibrary *singleton()
    {
        if(m_Singleton==NULL) m_Singleton = new TextureLibrary();
        return m_Singleton;
    }

    static void destroy()
    {
        delete m_Singleton;
    }


    ID3D10Texture2D *GetTexture(std::string id);
    ID3D10Texture2D *GetTexture(int id);

    int GetTextureIndex(std::string id);

    ID3D10ShaderResourceView *GetShaderResourceView(std::string id);
    ID3D10ShaderResourceView *GetShaderResourceView(int id);

    void AddTexture(std::string name,ID3D10Texture2D*tex,ID3D10ShaderResourceView *srv);
    int NumTextures(){return (int)m_Textures.size();}
    void Release();

    TextureMap m_Textures;
    SRVMap m_ShaderResourceViews;

private:
    TextureLibrary();
    static TextureLibrary *m_Singleton;
};