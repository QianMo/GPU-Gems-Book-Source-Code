/*
 * CgWangTilesT.hpp
 *
 * The Cg class for Wang Tiles, storing tile encoding in a texture
 *
 * Li-Yi Wei
 * 8/9/2003
 *
 */

#ifndef _CG_WANG_TILES_T_HPP
#define _CG_WANG_TILES_T_HPP

#include <GL/glut.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "Exception.hpp"

class CgWangTilesT
{
public:
    // tilesTextureID : the texture containing all tiles
    // cornersTextureID : the texture containing all tile corners
    // set any of these texture ID to <= 0 if you do not want to use them
    // tileHeight, tileWidth : the size of each sample tile
    // tileMappingTextureID : texture ID for the wang tile mapping texture
    // cornerMappingTextureID : texture ID for the corner mapping texture
    // they must have the same size of both > 0
    // the size of mapping-texture specifies the number of tiles
    // in the output virtual texture
    // each entry in the mapping-texture specifies which tile to use
    // in integer (x, y) format
    // the tilesTexture must be a 2D texture (we need mipmapping)
    // while the mappingTexture must be a RECT texture
    // cornerSharpness : the variance of the exponent corner function
    CgWangTilesT(const GLuint tilesTextureID,
                 const GLuint cornersTextureID,
                 const int tileHeight, const int tileWidth,
                 const GLuint tileMappingTextureID,
                 const GLuint cornerMappingTextureID,
                 const float cornerSharpness) throw(Exception);
    ~CgWangTilesT(void);

    void Enable(void) const;
    void Disable(void) const;
    
protected:
    static void CheckError(void);
    static string CreateProgram(const int tileHeight,
                                const int tileWidth,
                                const int tileTextureHeight,
                                const int tileTextureWidth,
                                const int cornerTextureHeight,
                                const int cornerTextureWidth,
                                const int mappingTextureHeight,
                                const int mappingTextureWidth,
                                const float cornerSharpness);
    
protected:
    CGcontext _context;
    CGprogram _program;
    CGparameter _tilesTexture;
    CGparameter _cornersTexture;
    CGparameter _tileMappingTexture;
    CGparameter _cornerMappingTexture;
    CGprofile _fragmentProfile;
};

#endif
