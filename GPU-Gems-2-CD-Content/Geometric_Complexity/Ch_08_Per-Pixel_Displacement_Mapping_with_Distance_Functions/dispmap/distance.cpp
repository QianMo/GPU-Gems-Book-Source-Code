/* 
  Copyright (C) 2005, William Donnelly (wdonnelly@uwaterloo.ca)
                  and Stefanus Du Toit (sjdutoit@cgl.uwaterloo.ca)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "distance.hpp"
#include <cmath>
#include <limits>

// A type to hold the distance map while it's being constructed
struct DistanceMap {
  typedef unsigned short Distance;

  DistanceMap(int width, int height, int depth)
    : m_data(new Distance[width*height*depth*3]),
      m_width(width), m_height(height), m_depth(depth)
  {
  }

  Distance& operator()(int x, int y, int z, int i)
  {
    return m_data[ 3*(m_depth*(m_height*x + y) + z) + i];
  }

  Distance operator()(int x, int y, int z, int i) const
  {
    return m_data[ 3*(m_depth*(m_height*x + y) + z) + i];
  }

  bool inside(int x, int y, int z) const
  {
    return (0 <= x && x < m_width &&
            0 <= y && y < m_height &&
            0 <= z && z < m_depth); 
  }

  // Do a single pass over the data.
  // Start at (x,y,z) and walk in the direction (cx,cy,cz)
  // Combine each pixel (x,y,z) with the value at (x+dx,y+dy,z+dz)
  void combine(int dx, int dy, int dz,
               int cx, int cy, int cz,
               int x, int y, int z)
  {
    while (inside(x, y, z) && inside(x + dx, y + dy, z + dz)) {

      Distance d[3]; d[0] = abs(dx); d[1] = abs(dy); d[2] = abs(dz);
      
      unsigned long v1[3], v2[3];
      for (int i = 0; i < 3; i++) v1[i] = (*this)(x, y, z, i);
      for (int i = 0; i < 3; i++) v2[i] = (*this)(x+dx, y+dy, z+dz, i) + d[i];

      if (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2] >
          v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]) {
        for (int i = 0; i < 3; i++) (*this)(x,y,z, i) = v2[i];
      }

      x += cx; y += cy; z += cz;
    }
  }

  Distance* m_data;
  int m_width, m_height, m_depth;
};

using namespace SH;

ShImage3D init_distance_map(const ShImage& heightmap, int depth)
{

  // Initialize the distance map to zero below the surface,
  // and +infinity above
  DistanceMap dmap(heightmap.width(), heightmap.height(), depth);
  for (int y = 0; y < heightmap.height(); y++) {
    for (int x = 0; x < heightmap.width(); x++) {
      for (int z = 0; z < depth; z++) {
        for (int i = 0; i < 3; i++) {
        if (z < heightmap(x, y, 0) * depth || z <= 1) {
            dmap(x, y, z, i) = 0;
          } else {
            dmap(x, y, z, i) = std::numeric_limits<DistanceMap::Distance>::max();
          }
        }
      }
    }
  }


  // Compute the rest of dmap by sequential sweeps over the data
  // using a 3d variant of Danielsson's algorithm
  std::cerr << "Computing Distance Map" << std::endl;

  for (int z = 1; z < depth; z++) {

    std::cerr << ".";

    // combine with everything with dz = -1
    for (int y = 0; y < heightmap.height(); y++) {
      dmap.combine( 0, 0, -1,
                    1, 0, 0,
                    0, y, z);
    }
    
    for (int y = 1; y < heightmap.height(); y++) {
      dmap.combine( 0, -1, 0,
                    1, 0, 0,
                    0, y, z);
      dmap.combine(-1, 0, 0,
                    1, 0, 0,
                    1, y, z);
      dmap.combine(+1, 0, 0,
                   -1, 0, 0,
                   heightmap.width() - 2, y, z);
    }

    for (int y = heightmap.height() - 2; y >= 0; y--) {
      dmap.combine( 0, +1, 0,
                    1, 0, 0,
                    0, y, z);
      dmap.combine(-1, 0, 0,
                    1, 0, 0,
                    1, y, z);
      dmap.combine(+1, 0, 0,
                   -1, 0, 0,
                   heightmap.width() - 2, y, z);
    }
  }
  std::cerr << " done first pass" << std::endl;

  for (int z = depth - 2; z >= 0; z--) {
    std::cerr << ".";

    // combine with everything with dz = +1
    for (int y = 0; y < heightmap.height(); y++) {
      dmap.combine( 0, 0, +1,
                    1, 0, 0,
                    0, y, z);
    }
    
    for (int y = 1; y < heightmap.height(); y++) {
      dmap.combine( 0, -1, 0,
                    1, 0, 0,
                    0, y, z);
      dmap.combine(-1, 0, 0,
                    1, 0, 0,
                    1, y, z);
      dmap.combine(+1, 0, 0,
                   -1, 0, 0,
                   heightmap.width() - 2, y, z);
    }
    for (int y = heightmap.height() - 2; y >= 0; y--) {
      dmap.combine( 0, +1, 0,
                    1, 0, 0,
                    0, y, z);
      dmap.combine(-1, 0, 0,
                    1, 0, 0,
                    1, y, z);
      dmap.combine(+1, 0, 0,
                   -1, 0, 0,
                   heightmap.width() - 2, y, z);
    }
  }
  
  std::cerr << " done second pass" << std::endl;

  // Construct a 3d texture img and fill it with the magnitudes of the 
  // displacements in dmap, scaled appropriately
  ShImage3D img(heightmap.width(), heightmap.height(), depth, 1);
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < heightmap.height(); y++) {
      for (int x = 0; x < heightmap.width(); x++) {
        double value = 0;
        for (int i = 0; i < 3; i++) {
          value += dmap(x, y, z, i)*dmap(x, y, z, i);
        }
        value = sqrt(value)/depth;
        if (value > 1.0) value = 1.0;
        img(x, y, z, 0) = value;
      }
    }
  }
  
  return img;
}

