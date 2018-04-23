/*
 * WangTiles.cpp
 *
 * Li-Yi Wei
 * 8/10/2003
 *
 */

#pragma warning (disable: 4786)

#include <math.h>

#include "WangTiles.hpp"

#include <iostream>
#include <strstream>
#include <deque>
using namespace std;

WangTiles::Tile::Tile(void) : _id(-1), _e0(-1), _e1(-1), _e2(-1), _e3(-1)
{
    // nothing to do
}

WangTiles::Tile::Tile(const int id,
                      const int e0, const int e1, const int e2, const int e3) : _id(id), _e0(e0), _e1(e1), _e2(e2), _e3(e3)
{
    // nothing to do
}

WangTiles::Tile::~Tile(void)
{
    // nothing to do
}

int WangTiles::Tile::ID(void) const
{
    return _id;
}

int & WangTiles::Tile::ID(void)
{
    return _id;
}

int WangTiles::Tile::e0(void) const {return _e0;}
int WangTiles::Tile::e1(void) const {return _e1;}
int WangTiles::Tile::e2(void) const {return _e2;}
int WangTiles::Tile::e3(void) const {return _e3;}

int & WangTiles::Tile::e0(void) {return _e0;}
int & WangTiles::Tile::e1(void) {return _e1;}
int & WangTiles::Tile::e2(void) {return _e2;}
int & WangTiles::Tile::e3(void) {return _e3;}

WangTiles::TileSet::TileSet(const int numHColors,
                            const int numVColors,
                            const int numTilesPerColor) throw(Exception) : _numHColors(numHColors), _numVColors(numVColors), _numTilesPerColor(numTilesPerColor)
{
    // error checking
    if((numHColors <= 0) || (numVColors <= 0) || (numTilesPerColor <= 0))
    {
        throw Exception("WangTiles::TileSet::TileSet() : illegal parameters");
    }
    
    {
        // memory allocation
        _tiles = vector< vector<Tile> >(numHColors*numHColors*numVColors*numVColors);

        for(int i = 0; i < _tiles.size(); i++)
        {
            _tiles[i] = vector<Tile>(numTilesPerColor);
        }
    }

    {
        // build tile set
        int id = 0;
        for(int e0 = 0; e0 < numHColors; e0++)
            for(int e1 = 0; e1 < numVColors; e1++)
                for(int e2 = 0; e2 < numHColors; e2++)
                    for(int e3 = 0; e3 < numVColors; e3++)
                    {
                        vector<Tile> & tiles = Tiles(e0, e1, e2, e3);
                        
                        for(int i = 0; i < tiles.size(); i++)
                        {
                            tiles[i] = Tile(id++, e0, e1, e2, e3);
                        }
                    }
    }
}

WangTiles::TileSet::~TileSet(void)
{
    // nothing to do
}

const vector<WangTiles::Tile> & WangTiles::TileSet::Tiles(const int e0,
                                                          const int e1,
                                                          const int e2,
                                                          const int e3) const throw(Exception)
{
    int index = SetIndex(e0, e1, e2, e3);

    if((index >= 0) && (index < _tiles.size()))
    {
        return _tiles[index];
    }
    else
    {
        throw Exception("WangTiles::TileSet::Tiles() : illegal tile edge colors");
    }
}

vector<WangTiles::Tile> & WangTiles::TileSet::Tiles(const int e0,
                                                    const int e1,
                                                    const int e2,
                                                    const int e3) throw(Exception)
{
    int index = SetIndex(e0, e1, e2, e3);

    if((index >= 0) && (index < _tiles.size()))
    {
        return _tiles[index];
    }
    else
    {
        throw Exception("WangTiles::TileSet::Tiles() : illegal tile edge colors");
    }
}

int WangTiles::TileSet::NumHColors(void) const
{
    return _numHColors;
}

int WangTiles::TileSet::NumVColors(void) const
{
    return _numVColors;
}

int WangTiles::TileSet::NumTilesPerColor(void) const
{
    return _numTilesPerColor;
}

int WangTiles::TileSet::NumTiles(void) const
{
    return _tiles.size() * _tiles[0].size();
}

int WangTiles::TileSet::SetIndex(const int e0,
                                 const int e1,
                                 const int e2,
                                 const int e3) const
{
    return
        (e0*(_numVColors * _numHColors * _numVColors) +
         e1*(_numHColors * _numVColors) +
         e2*(_numVColors) +
         e3);
}

int WangTiles::RandomCompaction(const TileSet & tileSet,
                                vector< vector<Tile> > & result)
{
    const int numHColors = tileSet.NumHColors();
    const int numVColors = tileSet.NumVColors();
    const int numTilesPerColor = tileSet.NumTilesPerColor();
   
    // find the best aspect ratio
    int numTilesPerColorH = numTilesPerColor;
    int numTilesPerColorV = 1;
    while((numVColors*numVColors*numTilesPerColorH >
           numHColors*numHColors*numTilesPerColorV) &&
          (numTilesPerColorH%2 == 0))
    {
        numTilesPerColorH /= 2;
        numTilesPerColorV *= 2;
    }

    const int height = numHColors*numHColors*numTilesPerColorV;
    const int width = numVColors*numVColors*numTilesPerColorH;

    {
        // space allocation for the result
        result = vector< vector<Tile> > (height);

        for(int i = 0; i < result.size(); i++)
        {
            result[i] = vector<Tile>(width);
        }
    }

    // do the random assignment
    {
        // build a random permutation
        deque<int> permutation(height*width);

        {
            for(int i = 0; i < permutation.size(); i++)
            {
                permutation[i] = i;
            }
        }

        deque<int> permutationNew;

        while(permutation.size() > 0)
        {
            const int select = rand()%permutation.size();
            permutationNew.push_back(permutation[select]);
            permutation[select] = permutation[permutation.size()-1];
            permutation.pop_back();
        }

        permutation = permutationNew;

        
        for(int e1 = 0; e1 < numVColors; e1++)
            for(int e3 = 0; e3 < numVColors; e3++)
                for(int e0 = 0; e0 < numHColors; e0++)
                    for(int e2 = 0; e2 < numHColors; e2++)
                    {
                        const vector<Tile> & tiles = tileSet.Tiles(e0, e1, e2, e3);
                        for(int k = 0; k < tiles.size(); k++)
                        {
                            const int whereToGo = permutation[0];
                            permutation.pop_front();

                            const int row = whereToGo/width;
                            const int col = whereToGo%width;

                            result[row][col] = tiles[k];
                        }
                    }
    }

    // done
    return 1;
}

int WangTiles::SimpleCompaction(const TileSet & tileSet,
                                vector< vector<Tile> > & result)
{
    const int numTiles = tileSet.NumTiles();

    const int numHColors = tileSet.NumHColors();
    const int numVColors = tileSet.NumVColors();
    const int numTilesPerColor = tileSet.NumTilesPerColor();
    
    // find the best aspect ratio
    const int maxFactor = floor(sqrt(numTiles));

    const int height = numVColors*numVColors;
    const int width = numHColors*numHColors*numTilesPerColor;

    {
        result = vector< vector<Tile> > (height);

        for(int i = 0; i < result.size(); i++)
        {
            result[i] = vector<Tile>(width);
        }
    }

    {
        int i = 0; int j = 0;
        for(int e1 = 0; e1 < numVColors; e1++)
            for(int e3 = 0; e3 < numVColors; e3++)
                for(int e0 = 0; e0 < numHColors; e0++)
                    for(int e2 = 0; e2 < numHColors; e2++)
                    {
                        const vector<Tile> & tiles = tileSet.Tiles(e0, e1, e2, e3);

                        for(int k = 0; k < tiles.size(); k++)
                        {
                            result[i][j] = tiles[k];

                            j++;

                            if(j >= result[i].size())
                            {
                                i++; j = 0;
                            }
                        }
                    }
    }
    
    // done 
    return 1;
}

// algorithm:
// find the tiling for each center color (i.e. assuming numTilesPerColor is 1)
// and repeat the tiling for multiple numTilesPerColor
int WangTiles::EvenCompaction(const TileSet & tileSet,
                              vector< vector<Tile> > & result)
{
    //const int numTiles = tileSet.NumTiles();

    const int numHColors = tileSet.NumHColors();
    const int numVColors = tileSet.NumVColors();
    const int numTilesPerColor = tileSet.NumTilesPerColor();

    // error checking
    if((numHColors*numVColors*numTilesPerColor)%4)
    {
        // illegal input sizes
        return 0;
    }

    // find the best aspect ratio
    int numTilesPerColorH = numTilesPerColor/2;
    int numTilesPerColorV = 2;
    {
        int hFactor = 1;
        int vFactor = 1;
        int numTilesPerColorhv = numTilesPerColor;

        if(numHColors%2)
        {
            hFactor *= 2; numTilesPerColorhv /= 2;
        }

        if(numVColors%2)
        {
            vFactor *= 2; numTilesPerColorhv /= 2;
        }

        const int maxFactor = ceil(sqrt(numTilesPerColorhv));
        int factor = 1;

        for(int i = 1; i <= maxFactor; i++)
        {
            if((numTilesPerColorhv%i) == 0) factor = i;
        }

        if( (numVColors*numVColors*vFactor) <
            (numHColors*numHColors*hFactor) )
        {
            factor = numTilesPerColorhv/factor;
        }

        numTilesPerColorH = factor * hFactor;
        numTilesPerColorV = numTilesPerColorhv/factor * vFactor;
    }

    const int height = numVColors*numVColors*numTilesPerColorV;
    const int width = numHColors*numHColors*numTilesPerColorH;

    {
        // space allocation for the result
        result = vector< vector<Tile> > (height);

        for(int i = 0; i < result.size(); i++)
        {
            result[i] = vector<Tile>(width);
        }
    }

    {
        // put the tiles
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
            {
                int ec = i/(numVColors*numVColors)*numTilesPerColorH +
                    j/(numHColors*numHColors);

                int e0 = (j/numHColors)%numHColors;
                int e1 = (i/numVColors)%numVColors;
                int e2 = (j%numHColors);
                int e3 = (i%numVColors);

                if(j%2)
                {
                    // swap e1 and e3
                    int tmp = e1; e1 = e3; e3 = tmp;
                }

                if(i%2)
                {
                    // swap e0 and e2
                    int tmp = e0; e0 = e2; e2 = tmp;
                }
                
                // assignment
                const vector<Tile> & tiles = tileSet.Tiles(e0, e1, e2, e3);

                if(tiles.size() <= ec)
                {
                    return 0;
                }
                
                result[i][j] = tiles[ec];
            }
    }
    
    // done
    return 1;
}

int WangTiles::EdgeOrdering(const int startNode, const int endNode)
{
    const int x = startNode; const int y = endNode;

    int result = -1;
    
    if(x < y)
    {
        result = (2*x + y*y);
    }
    else if(x == y)
    {
        if(x > 0)
        {
            result = ((x+1)*(x+1) - 2);
        }
        else
        {
            result = 0;
        }
    }
    else // x > y
    {
        if(y > 0)
        {
            result = (x*x + 2*y - 1);
        }
        else
        {
            result = ((x+1)*(x+1) - 1);
        }
    }

    return result;
}

string WangTiles::EdgeOrderingCgProgram(void)
{
    strstream strResult;

    strResult << "float EdgeOrdering(const float x, const float y)" << endl;
    strResult << "{" << endl;
#if 1
    strResult << "  float result;" << endl;
    strResult << "  if(x < y)" << endl;
    strResult << "  {" << endl;
    strResult << "     result = (2*x + y*y);" << endl;
    strResult << "  }" << endl;
    strResult << "  else if(x == y)" << endl;
    strResult << "  {" << endl;
    strResult << "      if(x > 0)" << endl;
    strResult << "      {" << endl;
    strResult << "          result = ((x+1)*(x+1) - 2);" << endl;
    strResult << "      }" << endl;
    strResult << "      else" << endl;
    strResult << "      {" << endl;
    strResult << "          result = 0;" << endl;
    strResult << "      }" << endl;
    strResult << "  }" << endl;
    strResult << "  else" << endl;
    strResult << "  {" << endl;
    strResult << "      if(y > 0)" << endl;
    strResult << "      {" << endl;
    strResult << "          result = (x*x + 2*y - 1);" << endl;
    strResult << "      }" << endl;
    strResult << "      else" << endl;
    strResult << "      {" << endl;
    strResult << "          result = ((x+1)*(x+1) - 1);" << endl;
    strResult << "      }" << endl;
    strResult << "  }" << endl;

    strResult << "  return result;" << endl;
#else
    // Cg cannot mix boolean and float operands
    strResult << "  return (x > y)*((x + (y == 0))*(x + (y == 0)) + 2*y - 1)";
    strResult << "  + (x == y)*(x > 0)*((x+1)*(x+1)-2)";
    strResult << "  + (x < y)*(2*x + y*y);" << endl;
#endif
    strResult << "}" << endl;

    // done
    string result(strResult.str(), strResult.pcount());
    strResult.rdbuf()->freeze(0);
    return result;
}

int WangTiles::TravelEdges(const int startNode, const int endNode,
                           vector<int> & result)
{
    if(startNode > endNode)
    {
        return 0;
    }

    result.clear();

    // non-recursive algorithm
    
    if(startNode != 0)
    {
        return 0;
    }

    const int numNodes = (endNode - startNode + 1);
    {
        // initialization
        result = vector<int>(numNodes*numNodes + 1);

        for(unsigned int i = 0; i < result.size(); i++)
        {
            result[i] = -1;
        }
    }
    
    for(int i = startNode; i <= endNode; i++)
        for(int j = startNode; j <= endNode; j++)
        {
            int index = EdgeOrdering(i - startNode, j - startNode);

            if((index < 0) || (index >= (numNodes*numNodes)))
            {
                return 0;
            }

            result[index] = i - startNode;
            result[index + 1] = j - startNode;
        }
    
    return 1;
}

int WangTiles::OrthogonalCompaction(const TileSet & tileSet,
                                    vector< vector<Tile> > & result)
{
    const int numHColors = tileSet.NumHColors();
    const int numVColors = tileSet.NumVColors();
    const int numTilesPerColor = tileSet.NumTilesPerColor();
   
    // find the best aspect ratio
    int numTilesPerColorH = numTilesPerColor;
    int numTilesPerColorV = 1;
    while((numVColors*numVColors*numTilesPerColorH >
           numHColors*numHColors*numTilesPerColorV) &&
          (numTilesPerColorH%2 == 0))
    {
        numTilesPerColorH /= 2;
        numTilesPerColorV *= 2;
    }

    const int height = numHColors*numHColors*numTilesPerColorV;
    const int width = numVColors*numVColors*numTilesPerColorH;

    {
        // space allocation for the result
        result = vector< vector<Tile> > (height);

        for(int i = 0; i < result.size(); i++)
        {
            result[i] = vector<Tile>(width);
        }
    }

    {
        vector<int> travelHEdges, travelVEdges;
        if(! TravelEdges(0, numHColors-1, travelHEdges)) return 0;
        if(! TravelEdges(0, numVColors-1, travelVEdges)) return 0;

        // put the tiles
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
            {
                int whichVBlock = i/(numHColors*numHColors);
                int whichHBlock = j/(numVColors*numVColors);
                
                int ec = whichVBlock*numTilesPerColorH + whichHBlock;

                int hIndex0 = i%(numHColors*numHColors);
                int hIndex2 = hIndex0 + 1;
                int vIndex1 = j%(numVColors*numVColors);
                int vIndex3 = vIndex1 + 1;
                
                int e0 = travelHEdges[hIndex0];
                int e3 = travelVEdges[vIndex1];
                int e2 = travelHEdges[hIndex2];
                int e1 = travelVEdges[vIndex3];

                // assignment
                const vector<Tile> & tiles = tileSet.Tiles(e0, e1, e2, e3);

                if(tiles.size() <= ec)
                {
                    return 0;
                }
                
                result[i][j] = tiles[ec];
            }
    }
 
    // done
    return 1;
}

int WangTiles::OrthogonalCornerCompaction(const TileSet & tileSet,
                                          vector< vector<Tile> > & result)
{
    const int numHColors = tileSet.NumHColors();
    const int numVColors = tileSet.NumVColors();
    const int numTilesPerColor = tileSet.NumTilesPerColor();
   
    // find the best aspect ratio
    int numTilesPerColorH = numTilesPerColor;
    int numTilesPerColorV = 1;
    while((numHColors*numHColors*numTilesPerColorH >
           numVColors*numVColors*numTilesPerColorV) &&
          (numTilesPerColorH%2 == 0))
    {
        numTilesPerColorH /= 2;
        numTilesPerColorV *= 2;
    }

    const int height = numVColors*numVColors*numTilesPerColorV;
    const int width = numHColors*numHColors*numTilesPerColorH;

    {
        // space allocation for the result
        result = vector< vector<Tile> > (height);

        for(int i = 0; i < result.size(); i++)
        {
            result[i] = vector<Tile>(width);
        }
    }

    {
        vector<int> travelHEdges, travelVEdges;
        if(! TravelEdges(0, numHColors-1, travelHEdges)) return 0;
        if(! TravelEdges(0, numVColors-1, travelVEdges)) return 0;

        // put the tiles
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
            {
                int whichVBlock = i/(numVColors*numVColors);
                int whichHBlock = j/(numHColors*numHColors);
                
                int ec = whichVBlock*numTilesPerColorH + whichHBlock;

                int hIndex0 = j%(numHColors*numHColors);
                int hIndex2 = hIndex0 + 1;
                int vIndex1 = i%(numVColors*numVColors);
                int vIndex3 = vIndex1 + 1;
                
                int e0 = travelHEdges[hIndex0];
                int e3 = travelVEdges[vIndex1];
                int e2 = travelHEdges[hIndex0];
                int e1 = travelVEdges[vIndex1];

                // assignment
                const vector<Tile> & tiles = tileSet.Tiles(e0, e1, e2, e3);

                if(tiles.size() <= ec)
                {
                    return 0;
                }
                
                result[i][j] = tiles[ec];
            }
    }
 
    // done
    return 1;
}

int WangTiles::SequentialTiling(const TileSet & tileSet,
                                const int numRowTiles,
                                const int numColTiles,
                                vector< vector<Tile> > & result)
{
    {
        // null initialization
        if(result.size() != numRowTiles)
        {
            result = vector< vector<Tile> >(numRowTiles);
        }

        for(int i = 0; i < result.size(); i++)
        {
            if(result[i].size() != numColTiles)
            {
                result[i] = vector<Tile>(numColTiles);
            }
        }

        {
            Tile empty;
            for(int i = 0; i < numRowTiles; i++)
                for(int j = 0; j < numColTiles; j++)
                {
                    result[i][j] = empty;
                }
        }
    }

    {
        const int numHColors = tileSet.NumHColors();
        const int numVColors = tileSet.NumVColors();
        
        // add tiles
        for(int i = 0; i < numRowTiles; i++)
            for(int j = 0; j < numColTiles; j++)
            {
                // find out the color selection from neighbors
                int e0, e1, e2, e3;

                e0 = result[(i+numRowTiles-1)%numRowTiles][j].e2();
                e1 = result[i][(j+1)%numColTiles].e3();
                e2 = result[(i+1)%numRowTiles][j].e0();
                e3 = result[i][(j+numColTiles-1)%numColTiles].e1();
                
                if(e0 < 0) e0 = rand()%numHColors;
                if(e1 < 0) e1 = rand()%numVColors;
                if(e2 < 0) e2 = rand()%numHColors;
                if(e3 < 0) e3 = rand()%numVColors;

                const vector<Tile> & selections = tileSet.Tiles(e0, e1, e2, e3);

                if(selections.size() <= 0)
                {
                    return 0;
                }
                
                result[i][j] = selections[rand()%selections.size()];
            }
    }

    // done
    return 1;
}

int WangTiles::ShiftedCornerTiling(const vector< vector<Tile> > & cornerPack,
                                   const vector< vector<Tile> > & input,
                                   vector< vector<Tile> > & result)
{
    const int numRowTiles = input.size();
    const int numColTiles = input[0].size();

    {
        // null initialization
        result = input;
    }

    {
        // add tiles
        for(int i = 0; i < numRowTiles; i++)
            for(int j = 0; j < numColTiles; j++)
            {
                // find out the color selection from neighbors
                int e0, e1, e2, e3;

                e0 = input[i][j].e1();
                e1 = input[i][(j+1)%numColTiles].e2();
                e2 = input[(i+1)%numRowTiles][(j+1)%numColTiles].e3();
                e3 = input[(i+1)%numRowTiles][j].e0();

                // find in
                int row, col;
                
                if(CornerLocation(cornerPack, e0, e1, e2, e3, row, col))
                {
                    result[i][j] = cornerPack[row][col];
                }
                else
                {
                    return 0;
                }
            }
    }
     
    // done
    return 1;
}

int WangTiles::TileLocation(const vector< vector<Tile> > & tiling,
                            const int id,
                            int & row, int & col)
{
    int result = 0;

    for(int i = 0; i < tiling.size(); i++)
        for(int j = 0; j < tiling[i].size(); j++)
        {
            if(tiling[i][j].ID() == id)
            {
                result++;
                row = i; col = j;
            }
        }
    
    return result;
}

int WangTiles::CornerLocation(const vector< vector<Tile> > & tiling,
                              const int e0,
                              const int e1,
                              const int e2,
                              const int e3,
                              int & row, int & col)
{
    int result = 0;

    const int height = tiling.size();
    const int width = tiling[0].size();
    
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            if( (e0 == tiling[i][j].e1()) &&
                (e1 == tiling[i][(j+1)%width].e2()) &&
                (e2 == tiling[(i+1)%height][(j+1)%width].e3()) &&
                (e3 == tiling[(i+1)%height][j].e0()) )
            {
                result++;
                row = i; col = j;
            }
        }
    
    return result;
}
