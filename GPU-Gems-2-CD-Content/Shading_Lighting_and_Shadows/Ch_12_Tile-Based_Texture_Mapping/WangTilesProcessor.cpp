/*
 * WangTilesProcessor.cpp
 *
 * Li-Yi Wei
 * 8/8/2004
 *
 */

#include <iostream>
using namespace std;

#include "WangTilesProcessor.hpp"

int WangTilesProcessor::BoxSanitize(const TilePack & tiles,
                                    const Pyramid & input,
                                    Pyramid & output,
                                    const int do_corner)
{
    // initialize output size
    output = input;

    // sanitize each level, from high resolution to low resolution
    for(int level = 0; level < input.NumLevels(); level++)
    {
        Array3D<Pixel> & level_now = output[level];
            
        if(level > 0)
        {
            const Array3D<Pixel> & level_up = output[level-1];
        
            // perform initial box filtering + down-sampling from previous level
            for(int row = 0; row < level_now.Size(0); row++)
                for(int col = 0; col < level_now.Size(1); col++)
                    for(int cha = 0; cha < level_now.Size(2); cha++)
                    {
                        Pixel temp = 0;
                    
                        for(int i = 2*row; i <= 2*row+1; i++)
                            for(int j = 2*col; j <= 2*col+1; j++)
                            {
                                const int row_up = i%level_up.Size(0);
                                const int col_up = j%level_up.Size(1);
                                        
                                temp += level_up[row_up][col_up][cha];
                            }

                        level_now[row][col][cha] = temp/4.0;
                    }
        }

        if(!SanitizeEdges(tiles, level_now, level_now)) return 0;
        if(do_corner && !SanitizeCorners(tiles, level_now, level_now)) return 0;
    }
    
    // done
    return 1;
}

void WangTilesProcessor::EdgeIndices(const int edge_orientation, 
                                     const int tile_height,
                                     const int tile_width,
                                     int & row_start, int & row_end,
                                     int & col_start, int & col_end)
{
    switch(edge_orientation%4)
    {
    case 2: // bottom
        row_start = row_end = tile_height-1;
        col_start = 0; col_end = tile_width-1;
        break;

    case 1: // right
        row_start = 0; row_end = tile_height-1;
        col_start = col_end = tile_width-1;
        break;

    case 0: // top
        row_start = row_end = 0;
        col_start = 0; col_end = tile_width-1;
        break;

    case 3: // left
        row_start = 0; row_end = tile_height-1;
        col_start = col_end = 0;
        break;

    default: // impossible
        row_start = col_start = 1;
        row_end = col_end = 0;
        break;
    }
}

void WangTilesProcessor::CornerIndices(const int edge_orientation, 
                                       const int tile_height,
                                       const int tile_width,
                                       int & row, int & col)
{
    switch(edge_orientation%4)
    {
    case 2: // bottom
        row = tile_height-1; col = 0;
        break;

    case 1: // right
        row = tile_height-1; col = tile_width-1;
        break;

    case 0: // top
        row = 0; col = tile_width-1;
        break;

    case 3: // left
        row = 0; col = 0;
        break;

    default: // impossible
        row = 0; col = 0;
        break;
    }
}

int WangTilesProcessor::CountEdgeColors(const TilePack & tiles,
                                        int & e0_colors,
                                        int & e1_colors,
                                        int & e2_colors,
                                        int & e3_colors)
{
    e0_colors = 0; e1_colors = 0; e2_colors = 0; e3_colors = 0;
    
    for(unsigned int i = 0; i < tiles.size(); i++)
        for(unsigned int j = 0; j < tiles[i].size(); j++)
        {
            const WangTiles::Tile & tile = tiles[i][j];
                
            if((tile.e0() < 0) || (tile.e1() < 0) || (tile.e2() < 0) || (tile.e3() < 0))
            {
                // no negative tile edges
                return 0;
            }
            else
            {
                if(e0_colors < (tile.e0()+1)) e0_colors = tile.e0()+1;
                if(e1_colors < (tile.e1()+1)) e1_colors = tile.e1()+1;
                if(e2_colors < (tile.e2()+1)) e2_colors = tile.e2()+1;
                if(e3_colors < (tile.e3()+1)) e3_colors = tile.e3()+1;
            }
        }

    return 1;
}

int WangTilesProcessor::SanitizeSmall(const TilePack & tiles,
                                      const Array3D<Pixel> & input,
                                      Array3D<Pixel> & output)
{
    if((tiles.size() <= 0) ||
       (input.Size(0)%tiles.size()) || (input.Size(1)%tiles[0].size()))
    {
        // small image, just make it a constant average of input
        output = input;

        for(int cha = 0; cha < input.Size(2); cha++)
        {
            Pixel sum = 0;

            {
                for(int row = 0; row < input.Size(0); row++)
                    for(int col = 0; col < input.Size(1); col++)
                    {
                        sum += input[row][col][cha];
                    }

                sum /= (input.Size(0)*input.Size(1));
            }

            {
                for(int row = 0; row < output.Size(0); row++)
                    for(int col = 0; col < output.Size(1); col++)
                    {
                        output[row][col][cha] = sum;
                    }
            }
        }

        return 1;
    }
    else
    {
        return 0;
    }
}

int WangTilesProcessor::SanitizeEdges(const TilePack & tiles,
                                      const Array3D<Pixel> & input,
                                      Array3D<Pixel> & output)
{
    // error checking
    if(SanitizeSmall(tiles, input, output))
    {
        return 1;
    }
    
    const int tile_height = input.Size(0)/tiles.size();
    const int tile_width = input.Size(1)/tiles[0].size();
    const int tile_depth = input.Size(2);
    
    // count the number of edge colors
    int e_colors[4] = {0, 0, 0, 0};
    { 
        if(!CountEdgeColors(tiles,
                            e_colors[0],
                            e_colors[1],
                            e_colors[2],
                            e_colors[3]))
        {
            return 0;
        }
        
        // check consistency
        if((e_colors[0] != e_colors[2]) || (e_colors[1] != e_colors[3]))
        {
            return 0;
        }
    }

    // initialize
    vector< vector<MeanTile> > mean_tiles(4);
    {
        Array3D<Pixel> image(tile_height, tile_width, tile_depth);
        for(int row = 0; row < image.Size(0); row++)
            for(int col = 0; col < image.Size(1); col++)
                for(int cha = 0; cha < image.Size(2); cha++)
                {
                    image[row][col][cha] = 0;
                }
        
        for(unsigned int i = 0; i < mean_tiles.size(); i++)
        {
            mean_tiles[i] = vector< MeanTile >(e_colors[i]);

            for(unsigned int j = 0; j < mean_tiles[i].size(); j++)
            {          
                mean_tiles[i][j].image = image;
                mean_tiles[i][j].count = 0;
            }
        }
    }

    // compute mean/average tiles per edge color
    {
        for(unsigned int i = 0; i < tiles.size(); i++)
            for(unsigned int j = 0; j < tiles[i].size(); j++)
            {
                const WangTiles::Tile & tile = tiles[i][j];

                const int input_row_start = i*tile_height;
                const int input_col_start = j*tile_width;

                const int tile_edges[4] = {tile.e0(), tile.e1(), tile.e2(), tile.e3()};
                
                for(int k = 0; k < 4; k++)
                {
                    MeanTile & output_tile = mean_tiles[k][tile_edges[k]];

                    for(int row = 0; row < tile_height; row++)
                        for(int col = 0; col < tile_width; col++)
                            for(int cha = 0; cha < tile_depth; cha++)
                            {
                                output_tile.image[row][col][cha] += input[row+input_row_start][col+input_col_start][cha];
                            }

                    output_tile.count++;
                }
            }
    }
    
    {
        for(unsigned int i = 0; i < mean_tiles.size(); i++)
            for(unsigned int j = 0; j < mean_tiles[i].size(); j++)
            {
                MeanTile & tile = mean_tiles[i][j];

                if(tile.count > 0)
                {
                    Array3D<Pixel> & image = tile.image;
                    
                    for(int row = 0; row < image.Size(0); row++)
                        for(int col = 0; col < image.Size(1); col++)
                            for(int cha = 0; cha < image.Size(2); cha++)
                            {
                                image[row][col][cha] /= tile.count;
                            }

                    tile.count = 1;
                }
            }
    }
    
    // fix the output tile boundaries
    output = input;
    Array3D<Pixel> boundary(output);
    Array3D<Pixel> boundary_count(output);

    {
        // initialization
        for(int i = 0; i < boundary.Size(0); i++)
            for(int j = 0; j < boundary.Size(1); j++)
                for(int k = 0; k < boundary.Size(2); k++)
                {
                    boundary[i][j][k] = 0;
                    boundary_count[i][j][k] = 0;
                }
    }
    
    {
        // compute the correct tile boundary
        for(unsigned int i = 0; i < tiles.size(); i++)
            for(unsigned int j = 0; j < tiles[i].size(); j++)
            {
                const WangTiles::Tile & tile = tiles[i][j];

                const int output_row_offset = i*tile_height;
                const int output_col_offset = j*tile_width;

                const int tile_edges[4] = {tile.e0(), tile.e1(), tile.e2(), tile.e3()};
                
                for(int k = 0; k < 4; k++)
                {
                    const Array3D<Pixel> & mean_tile = mean_tiles[k][tile_edges[k]].image;

                    int row_start, row_end, col_start, col_end;
                    EdgeIndices(k, tile_height, tile_width,
                                row_start, row_end, col_start, col_end);

                    for(int row = row_start; row <= row_end; row++)
                        for(int col = col_start; col <= col_end; col++)
                            for(int cha = 0; cha < tile_depth; cha++)
                            {
                                boundary[row+output_row_offset][col+output_col_offset][cha] += mean_tile[row][col][cha];
                                boundary_count[row+output_row_offset][col+output_col_offset][cha] += 1.0;
                            }
                }
            }
    }

    {
        // normalization and assign to the output
        for(int i = 0; i < boundary.Size(0); i++)
            for(int j = 0; j < boundary.Size(1); j++)
                for(int k = 0; k < boundary.Size(2); k++)
                {
                    if(boundary_count[i][j][k] > 0)
                    {
                        output[i][j][k] = boundary[i][j][k]/boundary_count[i][j][k];
                    }
                }
    }
    
    // done
    return 1;
}

int WangTilesProcessor::SanitizeCorners(const TilePack & tiles,
                                        const Array3D<Pixel> & input,
                                        Array3D<Pixel> & output)
{
    if(SanitizeSmall(tiles, input, output))
    {
        return 1;
    }
    
    const int tile_height = input.Size(0)/tiles.size();
    const int tile_width = input.Size(1)/tiles[0].size();
    const int tile_depth = input.Size(2);
    
    // initialize
    // note that for each row we have all possible corner combinations
    MeanTile mean_corner_tile;
    {
        Array3D<Pixel> image(tile_height, tile_width, tile_depth);
    
        for(int row = 0; row < image.Size(0); row++)
            for(int col = 0; col < image.Size(1); col++)
                for(int cha = 0; cha < image.Size(2); cha++)
                {
                    image[row][col][cha] = 0;
                }

        mean_corner_tile.image = image;
        mean_corner_tile.count = 0;
    }

    // compute mean/average tiles for corner
    // note that we need to consider all tiles to get correct results!
    {
        for(unsigned int i = 0; i < tiles.size(); i++)
            for(unsigned int j = 0; j < tiles[i].size(); j++)
            {
                const WangTiles::Tile & tile = tiles[i][j];

                const int input_row_start = i*tile_height;
                const int input_col_start = j*tile_width;

                for(int row = 0; row < tile_height; row++)
                    for(int col = 0; col < tile_width; col++)
                        for(int cha = 0; cha < tile_depth; cha++)
                        {
                            mean_corner_tile.image[row][col][cha] += input[row+input_row_start][col+input_col_start][cha];
                        }

                    mean_corner_tile.count++;
            }
    }

    // normalization
    {
        if(mean_corner_tile.count > 0)
        {
            Array3D<Pixel> & image = mean_corner_tile.image;
            
            for(int row = 0; row < image.Size(0); row++)
                for(int col = 0; col < image.Size(1); col++)
                    for(int cha = 0; cha < image.Size(2); cha++)
                    {
                        image[row][col][cha] /= mean_corner_tile.count;
                    }

            mean_corner_tile.count = 1;
        }
    }

    // assign to the output
    {
        for(unsigned int i = 0; i < tiles.size(); i++)
            for(unsigned int j = 0; j < tiles[i].size(); j++)
            {
                const WangTiles::Tile & tile = tiles[i][j];

                const int tile_edges[4] = {tile.e0(), tile.e1(), tile.e2(), tile.e3()};
                
                const int output_row_start = i*tile_height;
                const int output_col_start = j*tile_width;
                
                for(int k = 0; k < 4; k++)
                {
                    int row, col;
                    CornerIndices(k, tile_height, tile_width, row, col);
                        
                    // only touch the corners
                    for(int cha = 0; cha < tile_depth; cha++)
                    {
                        output[row+output_row_start][col+output_col_start][cha] = mean_corner_tile.image[row][col][cha];
                    } 
                }
            }
    }
    
    // done
    return 1;
}

int WangTilesProcessor::TileMismatch(const TilePack & tiles,
                                     const Pyramid & input)
{
    int result = -1;

    for(int lev = 0; lev < input.NumLevels(); lev++)
    {
        if(TileMismatch(tiles, input[lev]))
        {
            result = lev;
            break;
        }
    }

    return result;
}
  
int WangTilesProcessor::TileMismatch(const TilePack & tiles,
                                     const Array3D<Pixel> & input)
{
    int mismatch = 0;
    
    // count the number of edge colors
    int e_colors[4] = {0, 0, 0, 0};
    if(!CountEdgeColors(tiles,
                        e_colors[0],
                        e_colors[1],
                        e_colors[2],
                        e_colors[3]))
    {
        return 0;
    }
    
    // compute tile size
    const int tile_height = (tiles.size() > 0) ? input.Size(0)/tiles.size() : 0;
    const int tile_width = (tiles.size() > 0 && tiles[0].size() > 0) ? input.Size(1)/tiles[0].size() : 0;
    const int tile_depth = input.Size(2);

    // make sure all edges with
    // (1) the same S/E/N/W orientations and
    // (2) same edge id
    // have identical images
    for(int e_index = 0; e_index < 4; e_index++)
        for(int e_color = 0; e_color < e_colors[e_index]; e_color++)
        {
            // find a tile with e_color in e_index
            int t_row = -1; int t_col = -1;

            {
                for(unsigned int i = 0; i < tiles.size(); i++)
                    for(unsigned int j = 0; j < tiles[i].size(); j++)
                    {  
                        const WangTiles::Tile & tile = tiles[i][j];

                        const int my_e_colors[4] = {tile.e0(), tile.e1(), tile.e2(), tile.e3()};
                        if(my_e_colors[e_index] == e_color)
                        {
                            // found the tile
                            t_row = i; t_col = j;
                            break;
                        }
                    }
            }
            
            if((t_row >= 0) && (t_col >= 0))
            {
                const WangTiles::Tile & sample_tile = tiles[t_row][t_col];

                const int row_offset1 = t_row*tile_height;
                const int col_offset1 = t_col*tile_width;
                            
                for(unsigned int i = 0; i < tiles.size(); i++)
                    for(unsigned int j = 0; j < tiles[i].size(); j++)
                    {  
                        const WangTiles::Tile & tile = tiles[i][j];

                        const int my_e_colors[4] = {tile.e0(), tile.e1(), tile.e2(), tile.e3()};
                        if(my_e_colors[e_index] == e_color)
                        {
                            // try to see if they match
                            const int row_offset2 = i*tile_height;
                            const int col_offset2 = j*tile_width;
                
                            int row_start, row_end, col_start, col_end;

                            EdgeIndices(e_index, tile_height, tile_width,
                                        row_start, row_end,
                                        col_start, col_end);

                            for(int row = row_start; row <= row_end; row++)
                                for(int col = col_start; col <= col_end; col++)
                                    for(int cha = 0; cha < tile_depth; cha++)
                                    {
                                        if(input[row+row_offset1][col+col_offset1][cha] != input[row+row_offset2][col+col_offset2][cha])
                                        {
                                            mismatch = 1;
                                        }
                                    }
                        }
                    }
            }
        }
    
    // done
    return mismatch;
}
