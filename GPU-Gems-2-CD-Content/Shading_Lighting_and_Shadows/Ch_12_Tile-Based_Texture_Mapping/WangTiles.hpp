/*
 * WangTiles.hpp
 *
 * the utilities for generic Wang Tiles
 *
 * Li-Yi Wei
 * 8/10/2003
 *
 */

#ifndef _WANG_TILES_HPP
#define _WANG_TILES_HPP

#include <vector>
#include <string>
using namespace std;

#include "Exception.hpp"

class WangTiles
{
public: // some class definitions
    class Tile
    {
    public:
        // the colors in bottom, right, top, and left tile edges
        Tile(void);
        Tile(const int id,
             const int e0, const int e1, const int e2, const int e3);
        ~Tile(void);

        int ID(void) const; int & ID(void);
        
        int e0(void) const; int & e0(void);
        int e1(void) const; int & e1(void);
        int e2(void) const; int & e2(void);
        int e3(void) const; int & e3(void);
        
    protected:
        int _id;
        int _e0, _e1, _e2, _e3; // the 4 edge colors
    };

    class TileSet
    {
    public:
        TileSet(const int numHColors, const int numVColors,
                const int numTilesPerColor) throw(Exception);
        ~TileSet(void);

        const vector<Tile> & Tiles(const int e0,
                                   const int e1,
                                   const int e2,
                                   const int e3) const throw(Exception);
        vector<Tile> & Tiles(const int e0,
                             const int e1,
                             const int e2,
                             const int e3) throw(Exception);

        int NumHColors(void) const;
        int NumVColors(void) const;
        int NumTilesPerColor(void) const;
        int NumTiles(void) const;

    protected:
        int SetIndex(const int e0,
                     const int e1,
                     const int e2,
                     const int e3) const;
        
    protected:
        const int _numHColors, _numVColors, _numTilesPerColor;
        vector< vector<Tile> > _tiles;
    };
    
public:
    // random compaction
    static int RandomCompaction(const TileSet & tiles,
                                vector< vector<Tile> > & result);
    
    // compaction methods
    static int SimpleCompaction(const TileSet & tiles,
                                vector< vector<Tile> > & result);

    // even compaction
    // requires numHColors*numVColors*numTilesPerColor be multiple of 4
    // the name even comes from two fact:
    // 1. the compaction has even size in both height and width
    // 2. the compaction has a nice aspect ratio
    static int EvenCompaction(const TileSet & tiles,
                              vector< vector<Tile> > & result);

    // given an edge with (startNode, endNode)
    // figure out its relative location in the final Euler circuit
    // (assuming N nodes are numbered from 0 to N-1
    static int EdgeOrdering(const int startNode, const int endNode);

    static string EdgeOrderingCgProgram(void);
    
    // given a complete graph containing nodes [startNode to endNode]
    // with directed edges connecting any 2 nodes (including self cycles)
    // return a travel schedule for visiting all edges once
    // starting from startNode and end in also startNode
    static int TravelEdges(const int startNode, const int endNode,
                           vector<int> & result);

    // orthogonal compaction
    static int OrthogonalCompaction(const TileSet & tiles,
                                    vector< vector<Tile> > & result);

    // orthogonal corner compaction
    static int OrthogonalCornerCompaction(const TileSet & tiles,
                                          vector< vector<Tile> > & result);
    
    // tiling methods
    static int SequentialTiling(const TileSet & tiles,
                                const int numRowTiles, const int numColTiles,
                                vector< vector<Tile> > & result);

    // compute the shifted corner tiling
    // this is not for computing dual corner packing!
    static int ShiftedCornerTiling(const vector< vector<Tile> > & cornerPack,
                                   const vector< vector<Tile> > & input,
                                   vector< vector<Tile> > & result);

    // find a tile within a compaction or tiling
    // the answer will be in (row, col) if any tile with id is found
    // return the # of occurances of tiles with this id
    static int TileLocation(const vector< vector<Tile> > & tiling,
                            const int id,
                            int & row, int & col);

    // find a corner with colors (e0, e1, e2, e3) within a tiling
    // return a tile with NE corner matching this corner in (row, col)
    // return the # of occurances of this corner
    static int CornerLocation(const vector< vector<Tile> > & tiling,
                              const int e0,
                              const int e1,
                              const int e2,
                              const int e3,
                              int & row, int & col);
protected:
};

#endif
