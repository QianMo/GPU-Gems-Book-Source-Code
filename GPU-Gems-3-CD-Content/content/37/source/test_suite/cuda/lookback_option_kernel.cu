// ************************************************
// lookback_option_kernel.cu
// authors: Lee Howes and David B. Thomas
//
// Initialises, executes, and cleans up a
// lookback option performance test timing
// and outputting timing results for various 
// aspects of the execution including memory
// allocation, initialisation and generation.
// ************************************************

#ifndef _LOOKBACK_OPTIONS_CU_
#define _LOOKBACK_OPTIONS_CU_


float *lookback_VOL_0, *lookback_A_0, *lookback_A_1, *lookback_A_2, *lookback_S_0, *lookback_EPS_0, *lookback_MU;
float *device_lookback_VOL_0, *device_lookback_A_0, *device_lookback_A_1,
	*device_lookback_A_2, *device_lookback_S_0, *device_lookback_EPS_0, *device_lookback_MU;
float *lookbackSimulationResultsMean, *lookbackSimulationResultsVariance;
float *device_lookbackSimulationResultsMean, *device_lookbackSimulationResultsVariance;

unsigned int timer_lookback_tw = 0;
unsigned int timer_lookback_init = 0;
unsigned int timer_lookback_upload = 0;
unsigned int timer_lookback_download = 0;
unsigned int timer_lookback_malloc = 0;
unsigned int timer_lookback_cuda_malloc = 0;
unsigned int timer_lookback_free = 0;
unsigned int timer_lookback_cuda_free = 0;


void init_lookback_options()
{
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_tw));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_init));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_upload));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_download));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_malloc));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_cuda_malloc));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_free));
	CUT_SAFE_CALL(cutCreateTimer(&timer_lookback_cuda_free));


	CUT_SAFE_CALL(cutStartTimer(timer_lookback_malloc));

	// Lookback option memory allocations
	lookback_VOL_0 = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookback_A_0 = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookback_A_1 = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookback_A_2 = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookback_S_0 = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookback_EPS_0 = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookback_MU = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);

	lookbackSimulationResultsMean = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);
	lookbackSimulationResultsVariance = (float *) malloc(4 * LOOKBACK_NUM_PARAMETER_VALUES);

	CUT_SAFE_CALL(cutStopTimer(timer_lookback_malloc));


	CUT_SAFE_CALL(cutStartTimer(timer_lookback_cuda_malloc));
	// Lookback option memory allocations
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_VOL_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_A_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_A_1, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_A_2, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_S_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_EPS_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookback_MU, 4 * LOOKBACK_NUM_PARAMETER_VALUES));

	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookbackSimulationResultsMean, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_lookbackSimulationResultsVariance, 4 * LOOKBACK_NUM_PARAMETER_VALUES));
	CUT_SAFE_CALL(cutStopTimer(timer_lookback_cuda_malloc));

	CUT_SAFE_CALL(cutStartTimer(timer_lookback_init));
	// Initialise lookback option parameters, random guesses at this point...
	for (unsigned i = 0; i < LOOKBACK_NUM_PARAMETER_VALUES; i++)
	{
		lookback_VOL_0[i] = Rand();
		lookback_A_0[i] = Rand();
		lookback_A_1[i] = Rand();
		lookback_A_2[i] = Rand();
		lookback_S_0[i] = Rand();
		lookback_EPS_0[i] = Rand();
		lookback_MU[i] = Rand();

	}
	CUT_SAFE_CALL(cutStopTimer(timer_lookback_init));

	CUT_SAFE_CALL(cutStartTimer(timer_lookback_upload));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_VOL_0, lookback_VOL_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_A_0, lookback_A_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_A_1, lookback_A_1, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_A_2, lookback_A_2, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_S_0, lookback_S_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_EPS_0, lookback_EPS_0, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_lookback_MU, lookback_MU, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUT_SAFE_CALL(cutStopTimer(timer_lookback_upload));
}

void cleanup_lookback_options()
{
	CUT_SAFE_CALL(cutStartTimer(timer_lookback_free));
	// Lookback option memory allocations
	free(lookback_VOL_0);
	free(lookback_A_0);
	free(lookback_A_1);
	free(lookback_A_2);
	free(lookback_S_0);
	free(lookback_EPS_0);
	free(lookback_MU);

	free(lookbackSimulationResultsMean);
	free(lookbackSimulationResultsVariance);

	CUT_SAFE_CALL(cutStopTimer(timer_lookback_free));


	CUT_SAFE_CALL(cutStartTimer(timer_lookback_cuda_free));
	// Lookback option memory allocations
	CUDA_SAFE_CALL(cudaFree(device_lookback_VOL_0));
	CUDA_SAFE_CALL(cudaFree(device_lookback_A_0));
	CUDA_SAFE_CALL(cudaFree(device_lookback_A_1));
	CUDA_SAFE_CALL(cudaFree(device_lookback_A_2));
	CUDA_SAFE_CALL(cudaFree(device_lookback_S_0));
	CUDA_SAFE_CALL(cudaFree(device_lookback_EPS_0));
	CUDA_SAFE_CALL(cudaFree(device_lookback_MU));

	CUDA_SAFE_CALL(cudaFree(device_lookbackSimulationResultsMean));
	CUDA_SAFE_CALL(cudaFree(device_lookbackSimulationResultsVariance));
	CUT_SAFE_CALL(cutStopTimer(timer_lookback_cuda_free));


	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_tw));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_init));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_upload));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_download));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_malloc));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_cuda_malloc));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_free));
	CUT_SAFE_CALL(cutDeleteTimer(timer_lookback_cuda_free));


}


extern __shared__ float path[];	// LOOKBACK_TAUSWORTHE_NUM_THREADS*LOOKBACK_MAX_T
// CUDA Tausworthe sim. z1 to z4 are random number generator state variables
__device__ float tausworthe_lookback_sim(unsigned T, float VOL_0, float EPS_0, float A_0, float A_1, float A_2, float S_0, float MU, unsigned &z1, unsigned &z2,
										 unsigned &z3, unsigned &z4)
{

	float temp_random_value;

	float vol = VOL_0, eps = EPS_0;
	float s = S_0;
	int base = __mul24(threadIdx.x,
					   LOOKBACK_MAX_T) - LOOKBACK_TAUSWORTHE_NUM_THREADS;

	// Choose random asset path
	for (unsigned t = 0; t < T; t++)
	{
		// store the current asset price
		base = base + LOOKBACK_TAUSWORTHE_NUM_THREADS;
		path[base] = s;

		// Calculate the next asset price
		vol = sqrt(A_0 + A_1 * vol * vol + A_2 * eps * eps);

		eps = getRandomValueTausworthe(z1, z2, z3, z4, temp_random_value, t) * vol;

		s = s * exp(MU + eps);
	}

	// Look back at path to find payoff
	float sum = 0;
	for (unsigned t = 0; t < T; t++)
	{
		base = base - LOOKBACK_TAUSWORTHE_NUM_THREADS;
		sum += max(path[base] - s, (float) 0.0);
	}

	return sum;
}

__global__ void tausworthe_lookback(unsigned num_cycles,
									unsigned int *seedValues,
									float *simulationResultsMean,
									float *simulationResultsVariance,
									float *g_VOL_0, float *g_EPS_0, float *g_A_0, float *g_A_1, float *g_A_2, float *g_S_0, float *g_MU)
{
	// RNG state
	unsigned z1, z2, z3, z4;

	// Initialise tausworth with seeds
	z1 = seedValues[threadIdx.x];
	z2 = seedValues[TAUSWORTHE_TOTAL_NUM_THREADS + threadIdx.x];
	z3 = seedValues[2 * TAUSWORTHE_TOTAL_NUM_THREADS + threadIdx.x];
	z4 = seedValues[3 * TAUSWORTHE_TOTAL_NUM_THREADS + threadIdx.x];

	unsigned address = blockIdx.x * WALLACE_NUM_THREADS + threadIdx.x;

	float VOL_0, EPS_0, A_0, A_1, A_2, S_0, MU;
	VOL_0 = g_VOL_0[address];
	A_0 = g_A_0[address];
	A_1 = g_A_1[address];
	A_2 = g_A_2[address];
	S_0 = g_S_0[address];
	MU = g_MU[address];

	float mean = 0, varAcc = 0;
	for (float i = 1; i <= LOOKBACK_PATHS_PER_SIM; i++)
	{
		float res = tausworthe_lookback_sim(num_cycles, VOL_0, EPS_0,
											A_0, A_1, A_2, S_0,
											MU,
											z1, z2, z3, z4	// rng state variables
			);

		// update mean and variance in a numerically stable way
		float delta = res - mean;
		mean += delta / i;
		varAcc += delta * (res - mean);
	}

	simulationResultsMean[address] = mean;

	float variance = varAcc / (LOOKBACK_PATHS_PER_SIM - 1);
	simulationResultsVariance[address] = variance;
}




void computeLookbackOptions()
{


	init_lookback_options();

	// setup execution parameters and execute
	dim3 lookback_tausworth_grid(LOOKBACK_TAUSWORTHE_NUM_BLOCKS, 1, 1);
	dim3 lookback_tausworth_threads(LOOKBACK_TAUSWORTHE_NUM_THREADS, 1, 1);

	const unsigned num_cycles = LOOKBACK_MAX_T;

	// Execute the Tausworthe version of the lookback option, timing as we go
	CUT_SAFE_CALL(cutStartTimer(timer_lookback_tw));
	tausworthe_lookback <<< lookback_tausworth_grid,
		lookback_tausworth_threads,
		num_cycles * LOOKBACK_TAUSWORTHE_NUM_THREADS >>> (num_cycles,
														  deviceTauswortheSeeds,
														  device_lookbackSimulationResultsMean,
														  device_lookbackSimulationResultsVariance,
														  device_lookback_VOL_0,
														  device_lookback_EPS_0,
														  device_lookback_A_0,
														  device_lookback_A_1, device_lookback_A_2, device_lookback_S_0, device_lookback_MU);
	CUT_SAFE_CALL(cutStopTimer(timer_lookback_tw));
	CUT_CHECK_ERROR("Kernel execution failed: lookback tausworthe");

	// There is no Wallace version of the lookback option because the shared memory requirements clash. 
	// Wallace is not the appropriate generator for all cases as this demonstrates.

	CUT_SAFE_CALL(cutStartTimer(timer_lookback_download));
	CUDA_SAFE_CALL(cudaMemcpy(lookbackSimulationResultsMean, device_lookbackSimulationResultsMean, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy
				   (lookbackSimulationResultsVariance, device_lookbackSimulationResultsVariance, 4 * LOOKBACK_NUM_PARAMETER_VALUES, cudaMemcpyDeviceToHost));
	CUT_SAFE_CALL(cutStopTimer(timer_lookback_download));

	printf("\n\nLookback option results:\n");
	printf
		("Processing time for lookback initialisation code: %f (ms) for %d Simulations, %f MSimulations/sec\n",
		 cutGetTimerValue(timer_lookback_init), LOOKBACK_NUM_PARAMETER_VALUES * LOOKBACK_PATHS_PER_SIM,
		 LOOKBACK_NUM_PARAMETER_VALUES * LOOKBACK_PATHS_PER_SIM / (cutGetTimerValue(timer_lookback_init) / 1000.0) / 1000000.0);
	printf
		("Processing time for lookback tausworthe: %f (ms) for %d Steps, %f MSteps/sec\n",
		 cutGetTimerValue(timer_lookback_tw), LOOKBACK_NUM_PARAMETER_VALUES * LOOKBACK_PATHS_PER_SIM * LOOKBACK_MAX_T,
		 LOOKBACK_NUM_PARAMETER_VALUES * LOOKBACK_PATHS_PER_SIM * LOOKBACK_MAX_T / (cutGetTimerValue(timer_lookback_tw) / 1000.0) / 1000000.0);
	printf
		("Upload time for lookback options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_lookback_upload),
		 7 * 4 * LOOKBACK_NUM_PARAMETER_VALUES, (7 * 4 * LOOKBACK_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_lookback_upload))) / 1000.0);
	printf
		("Download time for lookback options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_lookback_download),
		 2 * 4 * LOOKBACK_NUM_PARAMETER_VALUES, (2 * 4 * LOOKBACK_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_lookback_download))) / 1000.0);
	printf
		("Malloc time for lookback options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_lookback_malloc),
		 9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES, (9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_lookback_malloc))) / 1000.0);
	printf
		("cudaMalloc time for lookback options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_lookback_cuda_malloc),
		 9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES, (9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_lookback_cuda_malloc))) / 1000.0);
	printf
		("free time for lookback options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_lookback_free),
		 9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES, (9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_lookback_free))) / 1000.0);
	printf
		("cudaFree time for lookback options: %f (ms) for %d bytes, %f MB/sec\n",
		 cutGetTimerValue(timer_lookback_cuda_free),
		 9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES, (9 * 4 * LOOKBACK_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_lookback_cuda_free))) / 1000.0);



	cleanup_lookback_options();
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
