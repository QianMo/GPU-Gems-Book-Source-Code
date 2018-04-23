/*
 * CgWangTilesC.cpp
 *
 * Li-Yi Wei
 * 8/14/2003
 *
 */

#include "CgWangTilesC.hpp"
#include "WangTiles.hpp"

#include <math.h>
#include <iostream>
#include <strstream>

using namespace std;

#include <glh/glh_extensions.h>

#define REQUIRED_EXTENSIONS "WGL_ARB_pbuffer " \
                            "WGL_ARB_pixel_format " \
                            "WGL_ARB_render_texture " \
                            "GL_NV_float_buffer " \
                            "GL_NV_texture_rectangle "

CgWangTilesC::CgWangTilesC(const GLuint tilesTextureID,
                           const GLuint cornersTextureID,
                           const int tileHeight,
                           const int tileWidth,
                           const int tileBlockHeight,
                           const int tileBlockWidth,
                           const int cornerBlockHeight,
                           const int cornerBlockWidth,
                           const int mappingTextureHeight,
                           const int mappingTextureWidth,
                           const GLuint hashTextureID,
                           const float cornerSharpness) throw(Exception): _context(0), _program(0), _tilesTexture(0), _cornersTexture(0), _hashTexture(0)
{
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
            throw Exception("CgWangTilesC: illegal tile texture size");
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
            throw Exception("CgWangTilesC: illegal corner texture size");
        }

        cornerTextureHeight /= tileHeight;
        cornerTextureWidth /= tileWidth;
    }
    
    // hash texture size
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, hashTextureID);
    
    int hashTextureHeight, hashTextureWidth;
    
    glGetTexLevelParameteriv(GL_TEXTURE_RECTANGLE_NV, 0,
                             GL_TEXTURE_HEIGHT, &hashTextureHeight);
    glGetTexLevelParameteriv(GL_TEXTURE_RECTANGLE_NV, 0,
                             GL_TEXTURE_WIDTH, &hashTextureWidth);

    if( (hashTextureHeight != 1) || (hashTextureWidth <= 0) )
    {
        throw Exception("CgWangTilesT: illegal hash texture size");
    }
    
    string program = CreateProgram(tileHeight, tileWidth,
                                   tileTextureHeight, tileTextureWidth,
                                   tileBlockHeight, tileBlockWidth,
                                   cornerTextureHeight, cornerTextureWidth,
                                   cornerBlockHeight, cornerBlockWidth,
                                   mappingTextureHeight, mappingTextureWidth,
                                   hashTextureWidth,
                                   cornerSharpness);
    
#if 0
    cout << program << endl;
#endif
    
    _program = cgCreateProgram(_context, CG_SOURCE, program.c_str(), _fragmentProfile, "fragment", 0);

#if 0
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
    
    // hash texture
    _hashTexture = cgGetNamedParameter(_program, "hashTexture");
    cgGLSetTextureParameter(_hashTexture, hashTextureID);
  
    cgGLEnableTextureParameter(_hashTexture);
    
    // enable stuff
    cgGLEnableProfile(_fragmentProfile);
    
    cgGLBindProgram(_program);
}

CgWangTilesC::~CgWangTilesC(void)
{
    Disable();
    cgDestroyContext(_context);
}

void CgWangTilesC::CheckError(void) 
{
    CGerror error = cgGetError();

    if(error != CG_NO_ERROR)
    {
        throw Exception(cgGetErrorString(error));
    }
}
    
void CgWangTilesC::Enable(void) const
{
    cgGLEnableTextureParameter(_hashTexture);
    if(_tilesTexture) cgGLEnableTextureParameter(_tilesTexture);
    if(_cornersTexture) cgGLEnableTextureParameter(_cornersTexture);
    
    cgGLEnableProfile(_fragmentProfile);
    
    cgGLBindProgram(_program);
}

void CgWangTilesC::Disable(void) const
{
    cgGLDisableTextureParameter(_hashTexture);
    if(_tilesTexture) cgGLDisableTextureParameter(_tilesTexture);
    if(_cornersTexture) cgGLDisableTextureParameter(_cornersTexture);
    
    cgGLDisableProfile(_fragmentProfile);
}

string CgWangTilesC::CreateProgram(const int tileHeight,
                                   const int tileWidth,
                                   const int tileTextureHeight,
                                   const int tileTextureWidth,
                                   const int tileBlockHeight,
                                   const int tileBlockWidth,
                                   const int cornerTextureHeight,
                                   const int cornerTextureWidth,
                                   const int cornerBlockHeight,
                                   const int cornerBlockWidth,
                                   const int mappingTextureHeight,
                                   const int mappingTextureWidth,
                                   const int hashTextureSize,
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
    strResult << WangTiles::EdgeOrderingCgProgram() << endl;

    const int singleBlock = \
        ! (((tileTextureHeight > 0) &&
            (tileTextureWidth > 0) &&
            ((tileTextureHeight != tileBlockHeight) ||
             (tileTextureWidth != tileBlockWidth)))
           ||
           ((cornerTextureHeight > 0) &&
            (cornerTextureWidth > 0) &&
            ((cornerTextureHeight != cornerBlockHeight) ||
             (cornerTextureWidth != cornerBlockWidth)))
            );

    if(singleBlock)
    {
        strResult << "float2 TileLocation(const float4 e)" << endl;
        strResult << "{" << endl;
        strResult << "   float2 result;" << endl;
        strResult << "   result.x = EdgeOrdering(e.w, e.y);" << endl;
        strResult << "   result.y = EdgeOrdering(e.x, e.z);" << endl;
        strResult << "   return result;" << endl;
        strResult << "}" << endl;
    }
    else
    {
        // tile location
        {
            const int numVBlocks = tileTextureHeight/tileBlockHeight;
            const int numHBlocks = tileTextureWidth/tileBlockWidth;
    
            strResult << "float2 TileLocation(const float4 e, const float c)" << endl;
            strResult << "{" << endl;
            strResult << "   float2 result;" << endl;
            strResult << "   result.x = EdgeOrdering(e.w, e.y) + frac(c/" << numHBlocks << ") * " << numHBlocks*tileBlockWidth << ";" << endl;
            strResult << "   result.y = EdgeOrdering(e.x, e.z) + floor(c/" << numHBlocks << ") * " << tileBlockHeight << ";" << endl;
            strResult << "   return result;" << endl;
            strResult << "}" << endl;
        }
    
        // corner location
        {
            const int numVBlocks = cornerTextureHeight/cornerBlockHeight;
            const int numHBlocks = cornerTextureWidth/cornerBlockWidth;

            strResult << "float2 CornerLocation(const float4 e, const float c)" << endl;
            strResult << "{" << endl;
            strResult << "   float2 result;" << endl;
            strResult << "   result.x = EdgeOrdering(e.w, e.y) + frac(c/" << numHBlocks << ") * " << numHBlocks*cornerBlockWidth << ";" << endl;
            strResult << "   result.y = EdgeOrdering(e.x, e.z) + floor(c/" << numHBlocks << ") * " << cornerBlockHeight << ";" << endl;
            strResult << "   return result;" << endl;
            strResult << "}" << endl;
        }
    }
    
    strResult << "float4 Hash(uniform samplerRECT hashTexture," << endl;
    strResult << "            const float4 input)" << endl;
    strResult << "{" << endl;
    strResult << "   return texRECT(hashTexture, frac(input.xy/" << hashTextureSize << ") * " << hashTextureSize << ");" << endl;
    strResult << "}" << endl;
    
    strResult << "FragmentOutput fragment(const FragmentInput input," << endl;
    strResult << "               uniform sampler2D tilesTexture," << endl;
    strResult << "               uniform sampler2D cornersTexture," << endl;
    strResult << "               uniform samplerRECT hashTexture)" << endl;
    strResult << "{" << endl;
    strResult << "    FragmentOutput output;" << endl;

    strResult << "    float2 mappingScale = float2(" << mappingTextureWidth << ", " << mappingTextureHeight << ");" << endl;
    strResult << "    float2 mappingAddress = input.tex.xy * mappingScale;" << endl;

    strResult << "    float2 whichVirtualTile;" << endl;
    strResult << "    float2 nextVirtualTile;" << endl;
    strResult << "    float4 edgeColors;" << endl;
    if(!singleBlock) strResult << "    float centerColor;" << endl;
    
    const int numHColors = sqrt(tileBlockHeight);
    const int numVColors = sqrt(tileBlockWidth);
    
    strResult << "    float4 numColors = float4(" << numHColors << ", " << numVColors << ", " << numHColors << ", " << numVColors << ");" << endl;
    
    const int numTilesPerColor = tileTextureHeight*tileTextureWidth/(tileBlockHeight*tileBlockWidth);

    if( (tileTextureHeight > 0) && (tileTextureWidth > 0))
    {
        // use hash to generate random edgeColors and centerColor
        strResult << "    whichVirtualTile = mod(mappingAddress, mappingScale);" << endl;
        strResult << "    nextVirtualTile = whichVirtualTile.xy + float2(1, 1);" << endl;
        strResult << "    nextVirtualTile = frac(nextVirtualTile/mappingScale)*mappingScale;" << endl;
    
        strResult << "    edgeColors.x = Hash(hashTexture, Hash(hashTexture, whichVirtualTile.x) + whichVirtualTile.y);" << endl;
        strResult << "    edgeColors.y = Hash(hashTexture, nextVirtualTile.x + Hash(hashTexture, 2*whichVirtualTile.y));" << endl;
        strResult << "    edgeColors.z = Hash(hashTexture, Hash(hashTexture, whichVirtualTile.x) + nextVirtualTile.y);" << endl;
        strResult << "    edgeColors.w = Hash(hashTexture, whichVirtualTile.x + Hash(hashTexture, 2*whichVirtualTile.y));" << endl;
         if(!singleBlock)
         {
             strResult << "    centerColor = Hash(hashTexture, Hash(hashTexture, 2*whichVirtualTile.x) + Hash(hashTexture, whichVirtualTile.y));" << endl;
         }
         
        // modulus
        strResult << "    edgeColors = frac(edgeColors/numColors)*numColors;" << endl;
        if(!singleBlock)
        {
            strResult << "    centerColor = frac(centerColor/" << numTilesPerColor << ")*" << numTilesPerColor << ";" << endl;
        }
        
        // determine which tile to use from edge and center colors
        if(singleBlock)
        {
            strResult << "    float2 whichTile = TileLocation(edgeColors);" << endl;
        }
        else
        {
            strResult << "    float2 whichTile = TileLocation(edgeColors, centerColor);" << endl;
        }
        
        strResult << "    float2 tileScale = float2(" << tileTextureWidth << ", " << tileTextureHeight << ");" << endl;
        strResult << "    float2 tileScaledTex = input.tex.xy * float2(" << mappingTextureWidth*1.0/tileTextureWidth << ", " << mappingTextureHeight*1.0/tileTextureHeight << ");" << endl;
        strResult << "    float4 result1 = tex2D(tilesTexture, (whichTile.xy + frac(mappingAddress))/tileScale, ddx(tileScaledTex), ddy(tileScaledTex));" << endl;
        
        strResult << "    output.col = result1;" << endl;
    }
    
    if( ((cornerTextureHeight > 0) && (cornerTextureWidth > 0)) ||
        (cornerSharpness < 0) )
    {   
        strResult << "    mappingAddress.xy = mappingAddress.xy - float2(0.5, 0.5);" << endl;
    }

    if( (cornerTextureHeight > 0) && (cornerTextureWidth > 0) )
    {
        // corner code here
        // use hash to generate random edgeColors and centerColor
        strResult << "    whichVirtualTile = mod(mappingAddress, mappingScale);" << endl;
        strResult << "    nextVirtualTile = whichVirtualTile.xy + float2(1, 1);" << endl;
        strResult << "    nextVirtualTile = frac(nextVirtualTile/mappingScale)*mappingScale;" << endl;
  
        strResult << "    edgeColors.x = Hash(hashTexture, nextVirtualTile.x + Hash(hashTexture, 2*whichVirtualTile.y));" << endl;
        strResult << "    edgeColors.y = Hash(hashTexture, Hash(hashTexture, nextVirtualTile.x) + nextVirtualTile.y);" << endl;
        strResult << "    edgeColors.z = Hash(hashTexture, nextVirtualTile.x + Hash(hashTexture, 2*nextVirtualTile.y));" << endl;
        strResult << "    edgeColors.w = Hash(hashTexture, Hash(hashTexture, whichVirtualTile.x) + nextVirtualTile.y);" << endl;
        if(!singleBlock)
        {
            strResult << "    centerColor = Hash(hashTexture, Hash(hashTexture, 2*whichVirtualTile.x) + Hash(hashTexture, whichVirtualTile.y));" << endl;
        }
        
        // modulus
        strResult << "    edgeColors = frac(edgeColors/numColors.yxwz)*numColors.yxwz;" << endl;
        if(!singleBlock)
        {
            strResult << "    centerColor = frac(centerColor/" << numTilesPerColor << ")*" << numTilesPerColor << ";" << endl;
        }
        
        // determine which tile to use from edge and center colors
        if(singleBlock)
        {
            strResult << "    float2 whichCorner = TileLocation(edgeColors);" << endl;
        }
        else
        {
            strResult << "    float2 whichCorner = CornerLocation(edgeColors, centerColor);" << endl;
        }
        
        strResult << "    float2 cornerScale = float2(" << cornerTextureWidth << ", " << cornerTextureHeight << ");" << endl;
        strResult << "    float2 cornerScaledTex = input.tex.xy * float2(" << mappingTextureWidth*1.0/cornerTextureWidth << ", " << mappingTextureHeight*1.0/cornerTextureHeight << ");" << endl;
        strResult << "    float4 result2 = tex2D(cornersTexture, (whichCorner.xy + frac(mappingAddress) + float2(0.5, 0.5))/cornerScale, ddx(cornerScaledTex), ddy(cornerScaledTex));" << endl;

        strResult << "    output.col = result2;" << endl;

    }
    
    if( ((tileTextureHeight > 0) && (tileTextureWidth > 0) &&
         (cornerTextureHeight > 0) && (cornerTextureWidth > 0)) ||
        (cornerSharpness < 0))

    {
        strResult << "    float2 cornerDistance = frac(mappingAddress) - float2(0.5, 0.5);" << endl;
        strResult << "    float bweight = exp(-dot(cornerDistance, cornerDistance)/" << fabs(cornerSharpness) << ");" << endl;
    
        if(cornerSharpness > 0)
        {
            strResult << "    output.col = result1*(1 - bweight) + result2*bweight;" << endl;
        }
        else
        {
            // demonstrate the corner blending weight
            strResult << "    output.col.xy = bweight; output.col.z = 0;" << endl;
        }
    }

#if 0
    // demonstrate the hash noise
    {
        strResult << "    output.col = (edgeColors.x * numColors.y * numColors.z * numColors.w + edgeColors.y * numColors.z * numColors.w + edgeColors.z * numColors.w + edgeColors.w)/(numColors.x * numColors.y * numColors.z * numColors.w);" << endl;
    }
#endif
    
    strResult << "    return output;" << endl;
    strResult << "}" << endl;

    // done
    string result(strResult.str(), strResult.pcount());
    strResult.rdbuf()->freeze(0);
    return result;
}
