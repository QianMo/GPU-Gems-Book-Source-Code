// ************************************************
// rng_kernel.cu
// authors: Lee Howes and David B. Thomas
//
// Initialises, executes, and cleans up a
// random number generation only performance
// test timing and outputting timing results for
// various aspects of the execution including
// memory allocation, initialisation and generation.
// ************************************************

#ifndef _RNG_KERNEL_CU_
#define _RNG_KERNEL_CU_


float *randomNumbers;
float *device_randomNumbers;

float *rngChi2Corrections = 0;
float *devicerngChi2Corrections = 0;

unsigned int timer_rng_tw = 0;
unsigned int timer_rng_wallace = 0;
unsigned int timer_rng_init = 0;
unsigned int timer_rng_upload = 0;
unsigned int timer_rng_download = 0;
unsigned int timer_rng_malloc = 0;
unsigned int timer_rng_cuda_malloc = 0;
unsigned int timer_rng_free = 0;
unsigned int timer_rng_cuda_free = 0;


void init_rng_tests()
{
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_tw));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_wallace));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_init));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_upload));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_download));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_malloc));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_cuda_malloc));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_free));
	CUT_SAFE_CALL(cutCreateTimer(&timer_rng_cuda_free));


	CUT_SAFE_CALL(cutStartTimer(timer_rng_malloc));


	rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);

	randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

	CUT_SAFE_CALL(cutStopTimer(timer_rng_malloc));


	CUT_SAFE_CALL(cutStartTimer(timer_rng_cuda_malloc));
	// Asian option memory allocations

	CUDA_SAFE_CALL(cudaMalloc((void **) &devicerngChi2Corrections, 4 * WALLACE_CHI2_COUNT));

	CUDA_SAFE_CALL(cudaMalloc((void **) &device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE));

	CUT_SAFE_CALL(cutStopTimer(timer_rng_cuda_malloc));

	CUT_SAFE_CALL(cutStartTimer(timer_rng_init));
	// Initialise asian option parameters, random guesses at this point...

	for (int i = 0; i < WALLACE_CHI2_COUNT; i++)
	{
		rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
	}
	CUT_SAFE_CALL(cutStopTimer(timer_rng_init));
	CUT_SAFE_CALL(cutStartTimer(timer_rng_upload));
	CUDA_SAFE_CALL(cudaMemcpy(devicerngChi2Corrections, rngChi2Corrections, 4 * WALLACE_CHI2_COUNT, cudaMemcpyHostToDevice));
	CUT_SAFE_CALL(cutStopTimer(timer_rng_upload));

}

void cleanup_rng_options()
{
	CUT_SAFE_CALL(cutStartTimer(timer_rng_free));
	// Asian option memory allocations

	free(rngChi2Corrections);

	free(randomNumbers);

	CUT_SAFE_CALL(cutStopTimer(timer_rng_free));


	CUT_SAFE_CALL(cutStartTimer(timer_rng_cuda_free));
	// Asian option memory allocations
	CUDA_SAFE_CALL(cudaFree(devicerngChi2Corrections));

	CUDA_SAFE_CALL(cudaFree(device_randomNumbers));

	CUT_SAFE_CALL(cutStopTimer(timer_rng_cuda_free));


	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_tw));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_wallace));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_init));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_upload));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_download));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_malloc));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_cuda_malloc));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_free));
	CUT_SAFE_CALL(cutDeleteTimer(timer_rng_cuda_free));



}


void computeRNG()
{

	init_rng_tests();

	// setup execution parameters and execute
	dim3 rng_tausworth_grid(TAUSWORTHE_NUM_BLOCKS, 1, 1);
	dim3 rng_tausworth_threads(TAUSWORTHE_NUM_THREADS, 1, 1);
	dim3 rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
	dim3 rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);

	// Execute the Tausworthe RNG, outputting into memory, and timing as we go.
	CUT_SAFE_CALL(cutStartTimer(timer_rng_tw));
	rng_tausworthe <<< rng_tausworth_grid, rng_tausworth_threads, 0 >>> (deviceTauswortheSeeds, device_randomNumbers);
	CUT_SAFE_CALL(cutStopTimer(timer_rng_tw));
	CUT_CHECK_ERROR("Kernel execution failed: rng tausworthe");

	unsigned seed = 1;

	// Execute the Wallace RNG, outputting into memory, and timing as we go.
	CUT_SAFE_CALL(cutStartTimer(timer_rng_wallace));
	rng_wallace <<< rng_wallace_grid, rng_wallace_threads,
		4 * (WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE) >>> (seed, devicerngChi2Corrections, devPool, device_randomNumbers);
	CUT_SAFE_CALL(cutStopTimer(timer_rng_wallace));

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed: rng wallace");

	CUT_SAFE_CALL(cutStartTimer(timer_rng_download));
	CUDA_SAFE_CALL(cudaMemcpy(randomNumbers, device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE, cudaMemcpyDeviceToHost));
	CUT_SAFE_CALL(cutStopTimer(timer_rng_download));

	printf("\n\nRng results:\n");
	printf
		("Processing time for rng initialisation code: %f (ms) for %d values, %f MValues/sec\n",
		 cutGetTimerValue(timer_rng_init), WALLACE_OUTPUT_SIZE, WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_init) / 1000.0) / 1000000.0);
	printf
		("Processing time for rng tausworthe: %f (ms) for %d values, %f MValues/sec\n",
		 cutGetTimerValue(timer_rng_tw), WALLACE_OUTPUT_SIZE, WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_tw) / 1000.0) / 1000000.0);
	printf
		("Processing time for rng wallace: %f (ms) for %d values, %f MValues/sec\n",
		 cutGetTimerValue(timer_rng_wallace), WALLACE_OUTPUT_SIZE, WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_wallace) / 1000.0) / 1000000.0);
	printf("Upload time for rng options: %f (ms) for %d bytes, %f MB/sec\n",
		   cutGetTimerValue(timer_rng_upload), 4 * WALLACE_CHI2_COUNT, (4 * WALLACE_CHI2_COUNT / (cutGetTimerValue(timer_rng_upload))) / 1000.0);
	printf("Download time for rng options: %f (ms) for %d bytes, %f MB/sec\n",
		   cutGetTimerValue(timer_rng_download), 4 * WALLACE_OUTPUT_SIZE, (4 * WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_download))) / 1000.0);
	printf("Malloc time for rng options: %f (ms) for %d bytes, %f MB/sec\n",
		   cutGetTimerValue(timer_rng_malloc),
		   4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE,
		   (4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_malloc))) / 1000.0);
	printf
		("cudaMalloc time for rng options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_rng_cuda_malloc),
		 4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE,
		 (4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_cuda_malloc))) / 1000.0);
	printf("free time for rng options: %f (ms) for %d bytes, %f MB/sec\n",
		   cutGetTimerValue(timer_rng_free),
		   4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE, (4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_free))) / 1000.0);
	printf("cudaFree time for rng options: %f (ms) for %d bytes, %f MB/sec\n",
		   cutGetTimerValue(timer_rng_cuda_free),
		   4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE,
		   (4 * WALLACE_CHI2_COUNT + 4 * WALLACE_OUTPUT_SIZE / (cutGetTimerValue(timer_rng_cuda_free))) / 1000.0);

	cleanup_rng_options();
}


#endif
