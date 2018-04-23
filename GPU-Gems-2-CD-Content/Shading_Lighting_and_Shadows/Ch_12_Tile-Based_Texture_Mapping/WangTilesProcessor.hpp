/*
 * WangTilesProcessor.hpp
 *
 * a collection of routines for processing Wang Tiles
 *
 * Li-Yi Wei
 * 8/8/2004
 *
 */

#ifndef _WANG_TILES_PROCESSOR_HPP
#define _WANG_TILES_PROCESSOR_HPP

#include "WangTiles.hpp"
#include "ImagePyramid.hpp"

class WangTilesProcessor
{
public:
    typedef float Pixel;
    typedef ImagePyramid<Pixel> Pyramid;
    typedef vector< vector<WangTiles::Tile> > TilePack;
    
    // sanitize the content of the input pyramid
    // will perform internal box filtering to build lower mipmap levels
    // return 1 if successful, 0 else
    static int BoxSanitize(const TilePack & tiles,
                           const Pyramid & input,
                           Pyramid & output,
                           const int do_corner);

    // check the pyramid to see if all tiles have matching boundaries
    // return the highest resolution level which has mismatch
    // or -1 if all match well
    static int TileMismatch(const TilePack & tiles,
                            const Pyramid & input);
    
protected:
    // compute the image indices of the edge within the tile
    static void EdgeIndices(const int edge_orientation, 
                            const int tile_height, const int tile_width,
                            int & row_start, int & row_end,
                            int & col_start, int & col_end);
    // compute the corner indices of the corner within the tile
    static void CornerIndices(const int edge_orientation, 
                              const int tile_height, const int tile_width,
                              int & row, int & col);

    // return 1 if successful, 0 else
    static int CountEdgeColors(const TilePack & tiles,
                               int & e0_colors,
                               int & e1_colors,
                               int & e2_colors,
                               int & e3_colors);

    // sanitize each level of the pyramid
    // it is o.k. for input and output to be the same object
    static int SanitizeSmall(const TilePack & tiles,
                             const Array3D<Pixel> & input,
                             Array3D<Pixel> & output);
    static int SanitizeEdges(const TilePack & tiles,
                             const Array3D<Pixel> & input,
                             Array3D<Pixel> & output);
    static int SanitizeCorners(const TilePack & tiles,
                               const Array3D<Pixel> & input,
                               Array3D<Pixel> & output);

    // check to see if a particular image has tile edge mismatch
    // return 1 if mismatch, 0 else
    static int TileMismatch(const TilePack & tiles,
                            const Array3D<Pixel> & input);
    
    struct MeanTile
    {
        Array3D<Pixel> image;
        int count;
    };
};

#endif

