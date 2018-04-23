/*
 * CgWangTilesC.hpp
 *
 * The Cg class for Wang Tiles, computing tile encoding on the fly
 *
 * Li-Yi Wei
 * 8/14/2003
 *
 */

#ifndef _CG_WANG_TILES_C_HPP
#define _CG_WANG_TILES_C_HPP

#include <GL/glut.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "Exception.hpp"

class CgWangTilesC
{
public:
    // tilesTextureID : the texture containing all tiles
    // cornersTextureID : the texture containing all tile corners
    // set any of these texture ID to <= 0 if you do not want to use them
    // tileHeight, tileWidth : the size of each sample/corner tile
    // tileBlockHeight, tileBlockWidth : the size of each tile block
    // cornerBlockHeight, cornerBlockWidth : the size of each corner block
    // mappingTextureHeight/Width: the number of tiles in the virtual texture
    // the tilesTexture must be a 2D texture (we need mipmapping)
    // while the hashTexture must be a RECT texture
    // cornerSharpness : the variance of the exponent corner function
    CgWangTilesC(const GLuint tilesTextureID,
                 const GLuint cornersTextureID,
                 const int tileHeight, const int tileWidth,
                 const int tileBlockHeight, const int tileBlockWidth,
                 const int cornerBlockHeight, const int cornerBlockWidth,
                 const int mappingTextureHeight, const int mappingTextureWidth,
                 const GLuint hashTextureID,
                 const float cornerSharpness) throw(Exception);
    ~CgWangTilesC(void);

    void Enable(void) const;
    void Disable(void) const;
    
protected:
    static void CheckError(void);
    static string CreateProgram(const int tileHeight,
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
                                const float cornerSharpness);
    
protected:
    CGcontext _context;
    CGprogram _program;
    CGparameter _tilesTexture;
    CGparameter _cornersTexture;
    CGparameter _hashTexture;
    CGprofile _fragmentProfile;
};

#endif
