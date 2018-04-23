This directory contains sample code for the article "Terrain Rendering Using
GPU-Based Geometry Clipmaps.

Included files:
Render.fx 
  This file contains the HLSL (both vertex and pixel shader) code for clipmap
  rendering.
  The vertex shader reads the z value from an elevation map stored as a vertex
  texture. It then computes a blending parameter and blends the geometry at the
  outer boundary of each level to ensure smooth transitions.
  The pixel shader accesses the normal map and shades the surface using a
  normal obtained by blending the fine and coarse level clipmap normals.
  For more details refer to Section 2.3.5 and 2.3.6 in the book.


Upsample.fx
  This file contains HLSL code for creating/updating a clipmap elevation map.
  A tensor-product version of the well-known four-point subdivision curve 
  interpolant, which has mask weights (-1/16, 9/16, 9/16, -1/16) is used to 
  predict the finer level geometry from the coarser one. Next decompression
  residual/synthesized fractal noise is added to this predicted value to get 
  the actual elevation which is written to the elevation map.
  For more details refer to Section 2.4 in the book.


ComputeNormals.fx
  This file contains HLSL code for computing the normal map given the elevation
  map.
  The normal is computed as the cross product of two grid-aligned tangent
  vectors. A texture lookup is then done to gather the normal from the coarser
  level and pack both the computed normal for the current level and the normal
  from the coarser level into a four channel texture.
  For more details refer to Section 2.4.3 in the book.