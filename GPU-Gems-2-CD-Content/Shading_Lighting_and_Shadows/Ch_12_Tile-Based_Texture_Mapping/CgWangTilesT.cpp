/*
 * CgWangTilesT.cpp
 *
 * Li-Yi Wei
 * 8/9/2003
 *
 */

#include "CgWangTilesT.hpp"

#include <iostream>
#include <strstream>

using namespace std;

#include <glh/glh_extensions.h>

#define REQUIRED_EXTENSIONS "WGL_ARB_pbuffer " \
                            "WGL_ARB_pixel_format " \
                            "WGL_ARB_render_texture " \
                            "GL_NV_float_buffer " \
                            "GL_NV_texture_rectangle " \
                            "GL_ARB_multitexture "

CgWangTilesT::CgWangTilesT(const GLuint tilesTextureID,
                           const GLuint cornersTextureID,
                           const int tileHeight,
                           const int tileWidth,
                           const GLuint tileMappingTextureID,
                           const GLuint cornerMappingTextureID,
                           const float cornerSharpness) throw(Exception) : _context(0), _program(0), _tilesTexture(0), _cornersTexture(0), _tileMappingTexture(0), _cornerMappingTexture(0)
{
    if( ((tilesTextureID <= 0) && (tileMappingTextureID > 0)) ||
        ((tilesTextureID > 0) && (tileMappingTextureID <= 0)) ||
        ((cornersTextureID <= 0) && (cornerMappingTextureID > 0)) ||
        ((cornersTextureID > 0) && (cornerMappingTextureID <= 0)) )
    {
        throw Exception("CgWangTilesT: illegal tile or corner selection!");
    }
    
    cgSetErrorCallback(CheckError);
    
    _context = cgCreateContext();
    
    if(! _context)
    {
        throw Exception("null context");
    }

    _fragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    cgGLSetOptimalOptions(_fragmentProfile);

    // tile texture size
    int tileTextureHeight = 0;
    int tileTextureWidth = 0;

    if(tilesTextureID > 0)
    {
        glActiveTextureARB(GL_TEXTURE0_ARB);
        glBindTexture(GL_TEXTURE_2D, tilesTextureID);
    
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                                 GL_TEXTURE_HEIGHT, &tileTextureHeight);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                                 GL_TEXTURE_WIDTH, &tileTextureWidth);

        if( (tileTextureHeight <= 0) || (tileTextureWidth <= 0) )
        {
            throw Exception("CgWangTilesT: illegal tile texture size");
        }

        tileTextureHeight /= tileHeight;
        tileTextureWidth /= tileWidth;
    }
    
    // corner texture size
    int cornerTextureHeight = 0;
    int cornerTextureWidth = 0;

    if(cornersTextureID > 0)
    {
        glActiveTextureARB(GL_TEXTURE1_ARB);
        glBindTexture(GL_TEXTURE_2D, cornersTextureID);
    
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                                 GL_TEXTURE_HEIGHT, &cornerTextureHeight);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                                 GL_TEXTURE_WIDTH, &cornerTextureWidth);

        if( (cornerTextureHeight <= 0) || (cornerTextureWidth <= 0) )
        {
            throw Exception("CgWangTilesT: illegal corner texture size");
        }

        cornerTextureHeight /= tileHeight;
        cornerTextureWidth /= tileWidth;
    }
    
    // tile mapping texture size
    int tileMappingTextureHeight = 0;
    int tileMappingTextureWidth = 0;

    if(tileMappingTextureID > 0)
    {
        glActiveTextureARB(GL_TEXTURE0_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, tileMappingTextureID);
    
        glGetTexLevelParameteriv(GL_TEXTURE_RECTANGLE_NV, 0,
                                 GL_TEXTURE_HEIGHT, &tileMappingTextureHeight);
        glGetTexLevelParameteriv(GL_TEXTURE_RECTANGLE_NV, 0,
                                 GL_TEXTURE_WIDTH, &tileMappingTextureWidth);

        if( (tileMappingTextureHeight <= 0) || (tileMappingTextureWidth <= 0) )
        {
            throw Exception("CgWangTilesT: illegal tile mapping texture size");
        }
    }
    
    // corner mapping texture size
    int cornerMappingTextureHeight = 0;
    int cornerMappingTextureWidth = 0;

    if(cornerMappingTextureID > 0)
    {
        glActiveTextureARB(GL_TEXTURE1_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, cornerMappingTextureID);
    
        glGetTexLevelParameteriv(GL_TEXTURE_RECTANGLE_NV, 0,
                                 GL_TEXTURE_HEIGHT, &cornerMappingTextureHeight);
        glGetTexLevelParameteriv(GL_TEXTURE_RECTANGLE_NV, 0,
                                 GL_TEXTURE_WIDTH, &cornerMappingTextureWidth);

        if( (cornerMappingTextureHeight <= 0) || (cornerMappingTextureWidth <= 0) )
        {
            throw Exception("CgWangTilesT: illegal corner mapping texture size");
        }
    }

    if((tileMappingTextureHeight &
        cornerMappingTextureHeight &
        (tileMappingTextureHeight != cornerMappingTextureHeight)) ||
       (tileMappingTextureWidth &
        cornerMappingTextureWidth &
        (tileMappingTextureWidth != cornerMappingTextureWidth))
       )
    {
        throw Exception("CgWangTilesT: incompatible tile and corner mapping texture sizes");
    }
       
    string program = \
        CreateProgram(tileHeight, tileWidth,
                      tileTextureHeight, tileTextureWidth,
                      cornerTextureHeight, cornerTextureWidth,
                      tileMappingTextureHeight, tileMappingTextureWidth,
                      cornerSharpness);
    
    _program = cgCreateProgram(_context, CG_SOURCE, program.c_str(), _fragmentProfile, "fragment", 0);

#if 0
    cout << program << endl;
    cout<<"---- PROGRAM BEGIN ----"<<endl;
    cout<<cgGetProgramString(_program, CG_COMPILED_PROGRAM);
    cout<<"---- PROGRAM END ----"<<endl;
#endif
    
    if(_program)
    {
        cgGLLoadProgram(_program);

        CheckError();
    }

    // tile texture
    if(tilesTextureID > 0)
    {
        glActiveTextureARB(GL_TEXTURE0_ARB);
        _tilesTexture = cgGetNamedParameter(_program, "tilesTexture");
        cgGLSetTextureParameter(_tilesTexture, tilesTextureID);
        cgGLEnableTextureParameter(_tilesTexture);
    }
    
    // corner texture
    if(cornersTextureID > 0)
    {
        glActiveTextureARB(GL_TEXTURE1_ARB);
        _cornersTexture = cgGetNamedParameter(_program, "cornersTexture");
        cgGLSetTextureParameter(_cornersTexture, cornersTextureID);
        cgGLEnableTextureParameter(_cornersTexture);
    }
    
    // tile mapping texture
    if(tileMappingTextureID > 0)
    {
        _tileMappingTexture = cgGetNamedParameter(_program, "tileMappingTexture");
        cgGLSetTextureParameter(_tileMappingTexture, tileMappingTextureID);
  
        cgGLEnableTextureParameter(_tileMappingTexture);
    }
    
    // corner mapping texture
    if(cornerMappingTextureID > 0)
    {
        _cornerMappingTexture = cgGetNamedParameter(_program, "cornerMappingTexture");
        cgGLSetTextureParameter(_cornerMappingTexture, cornerMappingTextureID);
  
        cgGLEnableTextureParameter(_cornerMappingTexture);
    }
    
    // enable stuff
    cgGLEnableProfile(_fragmentProfile);
    
    cgGLBindProgram(_program);
}

CgWangTilesT::~CgWangTilesT(void)
{
    Disable();
    cgDestroyContext(_context);
}

void CgWangTilesT::CheckError(void) 
{
    CGerror error = cgGetError();

    if(error != CG_NO_ERROR)
    {
        throw Exception(cgGetErrorString(error));
    }
}
    
void CgWangTilesT::Enable(void) const
{
    if(_tileMappingTexture) cgGLEnableTextureParameter(_tileMappingTexture);
    if(_cornerMappingTexture) cgGLEnableTextureParameter(_cornerMappingTexture);
    if(_tilesTexture) cgGLEnableTextureParameter(_tilesTexture);
    if(_cornersTexture) cgGLEnableTextureParameter(_cornersTexture);
    
    cgGLEnableProfile(_fragmentProfile);
    
    cgGLBindProgram(_program);
}

void CgWangTilesT::Disable(void) const
{
    if(_tileMappingTexture) cgGLDisableTextureParameter(_tileMappingTexture);
    if(_cornerMappingTexture) cgGLDisableTextureParameter(_cornerMappingTexture);
    if(_tilesTexture) cgGLDisableTextureParameter(_tilesTexture);
    if(_cornersTexture) cgGLDisableTextureParameter(_cornersTexture);
    
    cgGLDisableProfile(_fragmentProfile);
}

string CgWangTilesT::CreateProgram(const int tileHeight,
                                   const int tileWidth,
                                   const int tileTextureHeight,
                                   const int tileTextureWidth,
                                   const int cornerTextureHeight,
                                   const int cornerTextureWidth,
                                   const int tileMappingTextureHeight,
                                   const int tileMappingTextureWidth,
                                   const float cornerSharpness)
{
    strstream strResult;

    strResult << "struct FragmentInput" << endl;
    strResult << "{" << endl;
    strResult << "    float4 tex : TEX0;" << endl;
    strResult << "    float4 col : COL0;" << endl;
    strResult << "};" << endl;
    strResult << "struct FragmentOutput" << endl;
    strResult << "{" << endl;
    strResult << "    float4 col : COL;" << endl;
    strResult << "};" << endl;
    strResult << "" << endl;
    strResult << "float2 mod(const float2 a, const float2 b)" << endl;
    strResult << "{" << endl;
    strResult << "    return floor(frac(a/b)*b);" << endl;
    strResult << "}" << endl;
    strResult << "" << endl;
    strResult << "FragmentOutput fragment(FragmentInput input," << endl;
    strResult << "               uniform sampler2D tilesTexture," << endl;
    strResult << "               uniform sampler2D cornersTexture," << endl;
    strResult << "               uniform samplerRECT tileMappingTexture," << endl;
    strResult << "               uniform samplerRECT cornerMappingTexture)" << endl;
    strResult << "{" << endl;
    strResult << "    FragmentOutput output;" << endl;

    strResult << "    float2 mappingScale = float2(" << tileMappingTextureWidth << ", " << tileMappingTextureHeight << ");" << endl;
    strResult << "    float2 mappingAddress = input.tex.xy * mappingScale;" <<endl;

    if( (tileTextureHeight > 0) && (tileTextureWidth > 0))
    {
        strResult << "    float2 tileScale = float2(" << tileTextureWidth << ", " << tileTextureHeight << ");" << endl; 
        strResult << "    float2 tileScaledTex = input.tex.xy * float2(" << tileMappingTextureWidth*1.0/tileTextureWidth << ", " << tileMappingTextureHeight*1.0/tileTextureHeight << ");" << endl;
        strResult << "    float4 whichTile = texRECT(tileMappingTexture, mod(mappingAddress, mappingScale));" << endl;
        strResult << "    float4 result1 = tex2D(tilesTexture, (whichTile.xy + frac(mappingAddress))/tileScale, ddx(tileScaledTex), ddy(tileScaledTex));" << endl;
        strResult << "    output.col = result1;" << endl;
    }
    
    if( (cornerTextureHeight > 0) && (cornerTextureWidth > 0))
    {
        strResult << "    float2 cornerScale = float2(" << cornerTextureWidth << ", " << cornerTextureHeight << ");" << endl;
        strResult << "    float2 cornerScaledTex = input.tex.xy * float2(" << tileMappingTextureWidth*1.0/cornerTextureWidth << ", " << tileMappingTextureHeight*1.0/cornerTextureHeight << ");" << endl;
        strResult << "    mappingAddress.xy = mappingAddress.xy - float2(0.5, 0.5);" << endl;
        strResult << "    float4 whichCornerTile = texRECT(cornerMappingTexture, mod(mappingAddress, mappingScale));" << endl;
        strResult << "    float4 result2 = tex2D(cornersTexture, (whichCornerTile.xy + frac(mappingAddress) + float2(0.5, 0.5))/cornerScale, ddx(cornerScaledTex), ddy(cornerScaledTex));" << endl;
        strResult << "    output.col = result2;" << endl;
    }
    
    if( (tileTextureHeight > 0) && (tileTextureWidth > 0) &&
        (cornerTextureHeight > 0) && (cornerTextureWidth > 0) )
    {
        strResult << "    float2 cornerDistance = frac(mappingAddress) - float2(0.5, 0.5);" << endl;
        strResult << "    float bweight = exp(-dot(cornerDistance, cornerDistance)/" << cornerSharpness << ");" << endl;
        strResult << "    output.col = result1*(1 - bweight) + result2*bweight;" << endl;
        //strResult << "    output.col.xy = bweight; output.col.z = 0;" << endl;
    }
    
    strResult << "    return output;" << endl;
    strResult << "}" << endl;

    // done
    string result(strResult.str(), strResult.pcount());
    strResult.rdbuf()->freeze(0);
    return result;
}
