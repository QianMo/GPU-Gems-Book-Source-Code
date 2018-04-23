/* ----------------------------------------------------------

Octree Textures on the GPU - source code - GPU Gems 2 release
                                                   2004-11-21

Updates on http://www.aracknea.net/octreetex
--
(c) 2004 Sylvain Lefebvre - all rights reserved
--
The source code is provided 'as it is', without any warranties. 
Use at your own risk. The use of any part of the source code in a
commercial or non commercial product without explicit authorisation
from the author is forbidden. Use for research and educational
purposes is allowed and encouraged, provided that a short notice
acknowledges the author's work.
---------------------------------------------------------- */
#ifndef __CONFIG__
#define __CONFIG__

#define SIMULTREE_MAX_DEPTH        7

#define SIMULTREE_NODE_POOL_SIZE_U 256
#define SIMULTREE_NODE_POOL_SIZE_V 512

// #define SIMULTREE_8BIT

#define SCREEN_SIZE        512
#define SCREEN_BUFFER_SIZE SCREEN_SIZE*2

#define FOV         45.0

// -----------------------------------------------
// ----------  END OF MANUAL CONFIG --------------
// -----------------------------------------------

#ifdef SIMULTREE_8BIT
#  define SAMPLER           sampler2D
#  define TEX0(S,uv)        tex2D(S,uv) 
#  define TEX1(S,uv)        tex2D(S,SIMULTREE_NODE_POOL_SIZE*uv) 
#  define DECODE_DENSITY(c) float4((c.x)+(c.w*256.0), c.y,c.z, 0.0)
#  define ENCODE_DENSITY(c) float4(frac(c.x)        , c.y,c.z, (floor(c.x)/256.0)
#else
#  define SAMPLER           samplerRECT
#  define TEX0(S,uv)        texRECT(S,float2(SIMULTREE_NODE_POOL_SIZE_U,SIMULTREE_NODE_POOL_SIZE_V)*uv)
#  define TEX1(S,uv)        texRECT(S,uv)
#  define DECODE_DENSITY(c) (c)
#  define ENCODE_DENSITY(c) (c)
#endif

#define CPU_ENCODE_INDEX8_R(i,j) (i % 256)
#define CPU_ENCODE_INDEX8_G(i,j) (j % 256)
#define CPU_ENCODE_INDEX8_B(i,j) (j >> 8)

#define GPU_DECODE_INDEX8(c)       ( floor(0.5+float4(255.0*c.x,255.0*c.y + 256.0*255.0*c.z,0.0,0.0) ) \
  / float4(SIMULTREE_NODE_POOL_SIZE_U,SIMULTREE_NODE_POOL_SIZE_V,1.0,1.0) )

//#define GPU_DECODE_INDEX8(c) (c)

#define CPU_DECODE_INDEX8_U(r,g,b) (r)
#define CPU_DECODE_INDEX8_V(r,g,b) (g + (b << 8))

// -----------------------------------------------
// -----------------------------------------------
// -----------------------------------------------

#endif
