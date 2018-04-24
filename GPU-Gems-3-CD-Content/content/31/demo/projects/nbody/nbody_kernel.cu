/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _NBODY_KERNEL_H_
#define _NBODY_KERNEL_H_

#include <math.h>

#define BLOCKDIM 256
#define LOOP_UNROLL 4

__constant__ float softeningSquared;

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]


__device__ float3 
bodyBodyInteraction(float3 ai, float4 bi, float4 bj) {
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bi.x - bj.x;
    r.y = bi.y - bj.y;
    r.z = bi.z - bj.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}


// This is the "tile_calculation" function from the GPUG3 article.
__device__ float3 gravitation(float4 myPos, float3 accel)
{
    extern __shared__ float4 sharedPos[];
    int i;

    // Here we unroll the loop
    for (i = 0; i < blockDim.x; ) 
    {
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
#if LOOP_UNROLL > 1
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
#endif
#if LOOP_UNROLL > 2
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
#endif
#if LOOP_UNROLL > 4
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
        accel = bodyBodyInteraction(accel, SX(i), myPos); i += 1;
#endif
    }

    return accel;
}

// WRAP is used to force each block to start working on a different 
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at 
// once.
#define WRAP(x,m) (((x)<m)?(x):(x-m))  // Mod without divide, works on values from 0 up to 2m

template <bool multithreadBodies>
__device__ float3
computeBodyForce(float4 bodyPos, float4* positions, int numBodies)
{
    extern __shared__ float4 sharedPos[];

    float3 acc = {0.0f, 0.0f, 0.0f};
    
    int p = blockDim.x;
    int q = blockDim.y;
    int n = numBodies;

    int start = n/q * threadIdx.y;
    int tile0 = start/(n/q);
    int tile = tile0;
    int finish = start + n/q;

    for (int i = start; i < finish; i += p, tile++) 
    {
        sharedPos[threadIdx.x+blockDim.x*threadIdx.y] = 
            multithreadBodies ? 
            positions[(WRAP(blockIdx.x+tile, gridDim.x)*blockDim.y + threadIdx.y )* blockDim.x + threadIdx.x] :
            positions[WRAP(blockIdx.x+tile, gridDim.x) * blockDim.x + threadIdx.x];
       
        __syncthreads();
        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(bodyPos, acc);
        __syncthreads();
    }

    // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is underutilized
    // For example, with a 256 threads per block and 1024 bodies, there will only be 4 thread blocks, so the 
    // GPU will only be 25% utilized.  To improve this, we use multiple threads per body.  We still can use 
    // blocks of 256 threads, but they are arranged in q rows of p threads each.  Each thread processes 1/q
    // of the forces that affect each body, and then 1/q of the threads (those with threadIdx.y==0) add up
    // the partial sums from the other threads for that body.  To enable this, use the "--p=" and "--q=" 
    // command line options to this example.  e.g.:
    // "nbody.exe --n=1024 --p=64 --q=4" will use 4 threads per body and 256 threads per block. There will be
    // n/p = 16 blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads per body is greater than one, 
    // so that when it is not we don't have to execute the more complex code required!
    if (multithreadBodies)
    {
        SX_SUM(threadIdx.x, threadIdx.y).x = acc.x;
        SX_SUM(threadIdx.x, threadIdx.y).y = acc.y;
        SX_SUM(threadIdx.x, threadIdx.y).z = acc.z;

        __syncthreads();

        // Save the result in global memory for the integration step
        if (threadIdx.y == 0) {
            for (int i = 1; i < blockDim.y; i++) {
                acc.x += SX_SUM(threadIdx.x,i).x;
                acc.y += SX_SUM(threadIdx.x,i).y;
                acc.z += SX_SUM(threadIdx.x,i).z;
            }

        }
    }

    return acc;
}

template<bool multithreadBodies>
__global__ void
integrateBodies(float4* newPos, float4* newVel, 
                float4* oldPos, float4* oldVel,
                float deltaTime, float damping,
                int numBodies)
{
    int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	float4 pos = oldPos[index];   

    float3 force = computeBodyForce<multithreadBodies>(pos, oldPos, numBodies);

    // acceleration = force \ mass; 
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here
    // and in bodyBodyInteraction (because they cancel out).  Thus here
    // force == acceleration
    float4 vel = oldVel[index];
       
    vel.x += force.x * deltaTime;
    vel.y += force.y * deltaTime;
    vel.z += force.z * deltaTime;  

    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;
        
    // new position = old position + velocity * deltaTime
    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

    // store new position and velocity
    newPos[index] = pos;
    newVel[index] = vel;
}


#endif // #ifndef _NBODY_KERNEL_H_
