// ************************************************
// cuda_rng_options.cu
// authors: Lee Howes and David B. Thomas
//
// Main source file for cuda versions of code.
// Built to an object, hence the extern "C"
// exports for calling from the surrounding
// framework code.
// Contains a function to call the Wallace
// random number generator outputting to 
// a poor, one to call the Tausworthe 
// generator (these are used by the 
// statistical tests) and a speed test
// function which calls versions which output
// timing results.
// ************************************************

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

#include "rand_helpers.h"

#include "constants.h"

unsigned int timer_pre_computation_init_wallace = 0;
unsigned int timer_pre_computation_init_tw = 0;
unsigned int *tauswortheSeeds = 0;
unsigned int *deviceTauswortheSeeds = 0;
float *hostPool = 0;
float *devPool = 0;

// includes, kernels
#include <wallace_kernel.cu>
#include <tausworthe_kernel.cu>
#include <rng_kernel.cu>
#include <lookback_option_kernel.cu>
#include <asian_option_kernel.cu>

void montecarloHost();

////////////////////////////////////////////////////////////////////////////////
//! Wallace generator for random number quality tests
////////////////////////////////////////////////////////////////////////////////
extern "C" void cuda_wallace_two_block(unsigned int seed, float *chi2Corrections, float *hostPool, float *poolOut, float *output)
{
	CUT_CHECK_DEVICE();

	initRand();

	// Host mallocs
	// allocate device memory for pool and wallace output
	// allocate a pool and fill with normal random numbers
	float *devPool, *deviceGeneratedRandomNumberPool, *deviceChi2Corrections;
	float *refOutput = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

	// just to keep call happy
	float *devicePrices, *deviceStrike, *deviceYears;
	CUDA_SAFE_CALL(cudaMalloc((void **) &devicePrices, 4));
	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceStrike, 4));
	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceYears, 4));


	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceChi2Corrections, 4 * WALLACE_NUM_BLOCKS * WALLACE_NUM_OUTPUTS_PER_RUN));
	CUDA_SAFE_CALL(cudaMalloc((void **) &devPool, 4 * WALLACE_TOTAL_POOL_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceGeneratedRandomNumberPool, 4 * WALLACE_OUTPUT_SIZE));

	// Perform copies to GPU
	// copy the start pool in
	CUDA_SAFE_CALL(cudaMemcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(deviceChi2Corrections, chi2Corrections, 4 * WALLACE_NUM_BLOCKS * WALLACE_NUM_OUTPUTS_PER_RUN, cudaMemcpyHostToDevice));

	// setup execution parameters and execute
	dim3 wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
	dim3 wallace_threads(WALLACE_NUM_THREADS, 1, 1);
	rng_wallace <<< wallace_grid, wallace_threads, WALLACE_POOL_SIZE * 4 >>> (seed, devPool, deviceGeneratedRandomNumberPool, deviceChi2Corrections);

	CUT_CHECK_ERROR("Kernel execution failed: Wallace");

	// get the transformed pool and the output back
	CUDA_SAFE_CALL(cudaMemcpy(poolOut, devPool, 4 * WALLACE_TOTAL_POOL_SIZE, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(output, deviceGeneratedRandomNumberPool, 4 * WALLACE_OUTPUT_SIZE, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(deviceChi2Corrections));
	CUDA_SAFE_CALL(cudaFree(devPool));
	CUDA_SAFE_CALL(cudaFree(deviceGeneratedRandomNumberPool));

}


////////////////////////////////////////////////////////////////////////////////
//! Tausworthe generator for random number quality tests
////////////////////////////////////////////////////////////////////////////////
extern "C" void cuda_tausworthe(unsigned *pool, float *output)
{
	CUT_CHECK_DEVICE();

	initRand();

	// Host mallocs
	// allocate device memory for pool and wallace output
	// allocate a pool and fill with normal random numbers
	unsigned *devPool;
	float *deviceGeneratedRandomNumberPool;

	CUDA_SAFE_CALL(cudaMalloc((void **) &devPool, 4 * TAUSWORTHE_TOTAL_SEED_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceGeneratedRandomNumberPool, 4 * TAUSWORTHE_OUTPUT_SIZE));

	// Perform copies to GPU
	// copy the start pool in
	CUDA_SAFE_CALL(cudaMemcpy(devPool, pool, 4 * TAUSWORTHE_TOTAL_SEED_SIZE, cudaMemcpyHostToDevice));
	// setup execution parameters and execute
	dim3 tausworthe_grid(TAUSWORTHE_NUM_BLOCKS, 1, 1);
	dim3 tausworthe_threads(TAUSWORTHE_NUM_THREADS, 1, 1);
	rng_tausworthe <<< tausworthe_grid, tausworthe_threads, 0 >>> (devPool, deviceGeneratedRandomNumberPool);

	CUT_CHECK_ERROR("Kernel execution failed: Wallace or montecarlo");

	// get the transformed pool and the output back
	CUDA_SAFE_CALL(cudaMemcpy(pool, devPool, 4 * TAUSWORTHE_TOTAL_SEED_SIZE, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(output, deviceGeneratedRandomNumberPool, 4 * TAUSWORTHE_OUTPUT_SIZE, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(devPool));
	CUDA_SAFE_CALL(cudaFree(deviceGeneratedRandomNumberPool));

}


////////////////////////////////////////////////////////////////////////////////
//! Perform montecarlo speed tests
////////////////////////////////////////////////////////////////////////////////
void speedTests()
{
	CUT_CHECK_DEVICE();

	CUT_SAFE_CALL(cutCreateTimer(&timer_pre_computation_init_wallace));
	CUT_SAFE_CALL(cutCreateTimer(&timer_pre_computation_init_tw));

	// allocate device memory for pool and wallace output
	// allocate a pool and fill with normal random numbers
	hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
	tauswortheSeeds = (unsigned int *) malloc(4 * TAUSWORTHE_NUM_SEEDS);

	CUDA_SAFE_CALL(cudaMalloc((void **) &devPool, 4 * WALLACE_TOTAL_POOL_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceTauswortheSeeds, 4 * TAUSWORTHE_NUM_SEEDS));


	CUT_SAFE_CALL(cutStartTimer(timer_pre_computation_init_wallace));
	// Fill wallace initialisation pool with random numbers
	double sumSquares = 0;
	for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
	{
		float x = RandN();
		sumSquares += x * x;
		hostPool[i] = x;
	}

	double scale = sqrt(WALLACE_TOTAL_POOL_SIZE / sumSquares);
	for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
	{
		float x = hostPool[i];
		hostPool[i] = x * scale;
	}
	CUT_SAFE_CALL(cutStopTimer(timer_pre_computation_init_wallace));

	CUT_SAFE_CALL(cutStartTimer(timer_pre_computation_init_tw));
	// Prepare Tausworthe seeds
	for (unsigned i = 0; i < TAUSWORTHE_NUM_SEEDS; i++)
	{
		tauswortheSeeds[i] = (unsigned int) Rand();
	}
	CUT_SAFE_CALL(cutStopTimer(timer_pre_computation_init_tw));

	// Upload tausworthe seeds
	CUDA_SAFE_CALL(cudaMemcpy(deviceTauswortheSeeds, tauswortheSeeds, 4 * TAUSWORTHE_NUM_SEEDS, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE, cudaMemcpyHostToDevice));

	// Execute the speed tests
// RNG only
	computeRNG();

	// Asian options
	computeAsianOptions();

	// Lookback options
	computeLookbackOptions();

	free(hostPool);
	free(tauswortheSeeds);

	printf("\n\n");
	printf("Initialisation time for options: %f (ms)\n", cutGetTimerValue(timer_pre_computation_init_tw));
	printf("Initialisation time for wallace: %f (ms)\n", cutGetTimerValue(timer_pre_computation_init_wallace));


	CUT_SAFE_CALL(cutDeleteTimer(timer_pre_computation_init_tw));
	CUT_SAFE_CALL(cutDeleteTimer(timer_pre_computation_init_wallace));

	CUDA_SAFE_CALL(cudaFree(devPool));
	CUDA_SAFE_CALL(cudaFree(deviceTauswortheSeeds));
}
