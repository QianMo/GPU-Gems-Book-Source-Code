// ************************************************
// wallace_kernel.cu
// authors: Lee Howes and David B. Thomas
//
// Wallace random number generator demonstration code.
//
// Contains code both for simply generating the
// next number, such that can be called by
// the options simulations and also for outputting
// directly into a memory buffer.
//
// Note that in this code, unlike in the descriptions in the chapter, we have given
// each thread a pair of 4 element transforms to perform, each using
// a slightly different hadamard matrix. The reason behind this is that
// the complexity of computation caused a register shortage when 
// 512 threads were needed (2048 pool/4) which is solved by doubling
// the number of values computed per thread and halving the number
// of threads.
// ************************************************

#ifndef _WALLACE_KERNEL_CU_
#define _WALLACE_KERNEL_CU_

#include <stdio.h>

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)


extern __shared__ float pool[];
extern __shared__ unsigned sMod[];

__device__ void Hadamard4x4a(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = p - t;
	q = q - t;
	r = t - r;
	s = t - s;
}

__device__ void Hadamard4x4b(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = t - p;
	q = t - q;
	r = r - t;
	s = s - t;
}


const unsigned lcg_a = 241;
const unsigned lcg_c = 59;
const unsigned lcg_m = 256;
const unsigned mod_mask = lcg_m - 1;

__device__ void initialise_wallace(unsigned &seed, float *globalPool)
{

	// Load global pool into shared memory
	// Load global pool into shared memory
	unsigned offset = __mul24(WALLACE_POOL_SIZE, blockIdx.x);
	pool[threadIdx.x] = globalPool[offset + threadIdx.x];
	pool[threadIdx.x + WALLACE_NUM_THREADS] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS];
	pool[threadIdx.x + WALLACE_NUM_THREADS * 2] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 2];
	pool[threadIdx.x + WALLACE_NUM_THREADS * 3] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 3];
	__syncthreads();
	pool[threadIdx.x + WALLACE_NUM_THREADS * 4] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 4];
	pool[threadIdx.x + WALLACE_NUM_THREADS * 5] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 5];
	pool[threadIdx.x + WALLACE_NUM_THREADS * 6] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 6];
	pool[threadIdx.x + WALLACE_NUM_THREADS * 7] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 7];

	__syncthreads();

}

__device__ void store_wallace(float *globalPool)
{

	// Load global pool into shared memory
	// Load global pool into shared memory
	unsigned offset = __mul24(WALLACE_POOL_SIZE, blockIdx.x);
	__syncthreads();
	globalPool[offset + threadIdx.x] = pool[threadIdx.x];
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS] = pool[threadIdx.x + WALLACE_NUM_THREADS];
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 2] = pool[threadIdx.x + WALLACE_NUM_THREADS * 2];
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 3] = pool[threadIdx.x + WALLACE_NUM_THREADS * 3];
	__syncthreads();
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 4] = pool[threadIdx.x + WALLACE_NUM_THREADS * 4];
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 5] = pool[threadIdx.x + WALLACE_NUM_THREADS * 5];
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 6] = pool[threadIdx.x + WALLACE_NUM_THREADS * 6];
	globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * 7] = pool[threadIdx.x + WALLACE_NUM_THREADS * 7];

}

__device__ void transform_pool(unsigned &m_seed)
{
	float rin0_0, rin1_0, rin2_0, rin3_0, rin0_1, rin1_1, rin2_1, rin3_1;
	for (int i = 0; i < WALLACE_NUM_POOL_PASSES; i++)
	{
		unsigned seed = (m_seed + threadIdx.x) & mod_mask;
		__syncthreads();
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin0_0 = pool[((seed << 3))];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin1_0 = pool[((seed << 3) + 1)];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin2_0 = pool[((seed << 3) + 2)];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin3_0 = pool[((seed << 3) + 3)];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin0_1 = pool[((seed << 3) + 4)];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin1_1 = pool[((seed << 3) + 5)];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin2_1 = pool[((seed << 3) + 6)];
		seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
		rin3_1 = pool[((seed << 3) + 7)];


		__syncthreads();
		Hadamard4x4a(rin0_0, rin1_0, rin2_0, rin3_0);
		pool[0 * WALLACE_NUM_THREADS + threadIdx.x] = rin0_0;
		pool[1 * WALLACE_NUM_THREADS + threadIdx.x] = rin1_0;
		pool[2 * WALLACE_NUM_THREADS + threadIdx.x] = rin2_0;
		pool[3 * WALLACE_NUM_THREADS + threadIdx.x] = rin3_0;
		__syncthreads();
		Hadamard4x4b(rin0_1, rin1_1, rin2_1, rin3_1);
		pool[4 * WALLACE_NUM_THREADS + threadIdx.x] = rin0_1;
		pool[5 * WALLACE_NUM_THREADS + threadIdx.x] = rin1_1;
		pool[6 * WALLACE_NUM_THREADS + threadIdx.x] = rin2_1;
		pool[7 * WALLACE_NUM_THREADS + threadIdx.x] = rin3_1;
		__syncthreads();
	}
}


__device__ float getRandomValue(unsigned n, unsigned loop_counter, float chi2CorrAndScale)
{
	return pool[__mul24(n, WALLACE_NUM_THREADS) + threadIdx.x] * chi2CorrAndScale;
}


__device__ void generateRandomNumbers_wallace(unsigned m_seed, float *chi2Corrections,	// move this into constants
											  float *globalPool, float *output)
{
	initialise_wallace(m_seed, globalPool);

	// Loop generating outputs repeatedly
	for (int loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++)
	{

		m_seed = (1664525U * m_seed + 1013904223U) & 0xFFFFFFFF;

		unsigned intermediate_address = __mul24(loop,
												8 * WALLACE_TOTAL_NUM_THREADS) + __mul24(8 * WALLACE_NUM_THREADS, blockIdx.x) + threadIdx.x;

		if (threadIdx.x == 0)
			pool[WALLACE_CHI2_OFFSET] = chi2Corrections[__mul24(blockIdx.x, WALLACE_NUM_OUTPUTS_PER_RUN) + loop];
		__syncthreads();
		float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
		for (int i = 0; i < 8; i++)
		{
			output[intermediate_address + i * WALLACE_NUM_THREADS] = getRandomValue(i, loop, chi2CorrAndScale);
		}

		transform_pool(m_seed);

	}

	store_wallace(globalPool);
}





__global__ void rng_wallace(unsigned seed, float *globalPool, float *generatedRandomNumberPool, float *chi2Corrections)
{

	generateRandomNumbers_wallace(seed, chi2Corrections, globalPool, generatedRandomNumberPool);
}

#endif
