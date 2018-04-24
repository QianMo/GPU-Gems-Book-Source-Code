// ************************************************
// asian_option_kernel.cu
// authors: Lee Howes and David B. Thomas
//
// Initialises, executes, and cleans up an
// Asian option performance test timing
// and outputting timing results for various 
// aspects of the execution including memory
// allocation, initialisation and generation.
// ************************************************

#ifndef _ASIAN_OPTIONS_CU_
#define _ASIAN_OPTIONS_CU_


float *asian_A_0, *asian_B_0, *asian_MU_A, *asian_SIG_AA, *asian_MU_B, *asian_SIG_AB, *asian_SIG_BB;
float *device_asian_A_0, *device_asian_B_0, *device_asian_MU_A, *device_asian_SIG_AA, *device_asian_MU_B, *device_asian_SIG_AB, *device_asian_SIG_BB;
float *asianSimulationResultsMean, *asianSimulationResultsVariance;
float *device_asianSimulationResultsMean, *device_asianSimulationResultsVariance;

float *asianChi2Corrections = 0;
float *deviceAsianChi2Corrections = 0;

unsigned int timer_asian_tw = 0;
unsigned int timer_asian_wallace = 0;
unsigned int timer_asian_init = 0;
unsigned int timer_asian_upload = 0;
unsigned int timer_asian_download = 0;
unsigned int timer_asian_malloc = 0;
unsigned int timer_asian_cuda_malloc = 0;
unsigned int timer_asian_free = 0;
unsigned int timer_asian_cuda_free = 0;


void init_asian_options()
{
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_tw));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_wallace));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_init));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_upload));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_download));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_malloc));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_cuda_malloc));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_free));
	CUT_SAFE_CALL(cutCreateTimer(&timer_asian_cuda_free));


	CUT_SAFE_CALL(cutStartTimer(timer_asian_malloc));

	// Asian option memory allocations
	asian_A_0 = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asian_B_0 = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asian_MU_A = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asian_SIG_AA = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asian_MU_B = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asian_SIG_AB = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asian_SIG_BB = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asianChi2Corrections = (float *) malloc(4 * ASIAN_WALLACE_CHI2_COUNT);

	asianSimulationResultsMean = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);
	asianSimulationResultsVariance = (float *) malloc(4 * ASIAN_NUM_PARAMETER_VALUES);

	CUT_SAFE_CALL(cutStopTimer(timer_asian_malloc));


	CUT_SAFE_CALL(cutStartTimer(timer_asian_cuda_malloc));
	// Asian option memory allocations
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_A_0, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_B_0, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_MU_A, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_SIG_AA, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_MU_B, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_SIG_AB, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asian_SIG_BB, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &deviceAsianChi2Corrections, 4 * ASIAN_WALLACE_CHI2_COUNT));

	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asianSimulationResultsMean, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUDA_SAFE_CALL(cudaMalloc((void **) &device_asianSimulationResultsVariance, 4 * ASIAN_NUM_PARAMETER_VALUES));
	CUT_SAFE_CALL(cutStopTimer(timer_asian_cuda_malloc));

	CUT_SAFE_CALL(cutStartTimer(timer_asian_init));
	// Initialise asian option parameters, random guesses at this point...
	for (unsigned i = 0; i < ASIAN_NUM_PARAMETER_VALUES; i++)
	{
		asian_A_0[i] = Rand();
		asian_B_0[i] = Rand();
		asian_MU_A[i] = Rand();
		asian_MU_B[i] = Rand();
		asian_SIG_AA[i] = Rand();
		asian_SIG_AB[i] = Rand();
		asian_SIG_BB[i] = Rand();
	}

	for (int i = 0; i < ASIAN_WALLACE_CHI2_COUNT; i++)
	{
		asianChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
	}
	CUT_SAFE_CALL(cutStopTimer(timer_asian_init));

	CUT_SAFE_CALL(cutStartTimer(timer_asian_upload));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_A_0, asian_A_0, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_B_0, asian_B_0, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_MU_A, asian_MU_A, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_MU_B, asian_MU_B, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_SIG_AA, asian_SIG_AA, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_SIG_AB, asian_SIG_AB, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_asian_SIG_BB, asian_SIG_BB, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(deviceAsianChi2Corrections, asianChi2Corrections, 4 * ASIAN_WALLACE_CHI2_COUNT, cudaMemcpyHostToDevice));
	CUT_SAFE_CALL(cutStopTimer(timer_asian_upload));
}

void cleanup_asian_options()
{
	CUT_SAFE_CALL(cutStartTimer(timer_asian_free));
	// Asian option memory allocations
	free(asian_A_0);
	free(asian_B_0);
	free(asian_MU_A);
	free(asian_SIG_AA);
	free(asian_MU_B);
	free(asian_SIG_AB);
	free(asian_SIG_BB);
	free(asianChi2Corrections);

	free(asianSimulationResultsMean);
	free(asianSimulationResultsVariance);

	CUT_SAFE_CALL(cutStopTimer(timer_asian_free));


	CUT_SAFE_CALL(cutStartTimer(timer_asian_cuda_free));
	// Asian option memory allocations
	CUDA_SAFE_CALL(cudaFree(device_asian_A_0));
	CUDA_SAFE_CALL(cudaFree(device_asian_B_0));
	CUDA_SAFE_CALL(cudaFree(device_asian_MU_A));
	CUDA_SAFE_CALL(cudaFree(device_asian_SIG_AA));
	CUDA_SAFE_CALL(cudaFree(device_asian_MU_B));
	CUDA_SAFE_CALL(cudaFree(device_asian_SIG_AB));
	CUDA_SAFE_CALL(cudaFree(device_asian_SIG_BB));
	CUDA_SAFE_CALL(cudaFree(deviceAsianChi2Corrections));

	CUDA_SAFE_CALL(cudaFree(device_asianSimulationResultsMean));
	CUDA_SAFE_CALL(cudaFree(device_asianSimulationResultsVariance));
	CUT_SAFE_CALL(cutStopTimer(timer_asian_cuda_free));


	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_tw));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_wallace));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_init));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_upload));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_download));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_malloc));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_cuda_malloc));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_free));
	CUT_SAFE_CALL(cutDeleteTimer(timer_asian_cuda_free));

}


__device__ float
wallace_asian_basket_sim(unsigned seed,
						 unsigned &loop, float *chi2Corrections, float A_0, float B_0, float MU_A, float SIG_AA, float MU_B, float SIG_AB, float SIG_BB)
{
	float a = A_0, b = B_0, s = 0, sum = 0;

	// Timesteps for a single simulation
	for (unsigned t = 0; t < (ASIAN_TIME_STEPS / 4); t++)
	{
		float ra, rb;
		// Read in the chi2Correction value only in a single thread and use shared memory to broadcast to other threads
		if (threadIdx.x == 0)
			pool[WALLACE_CHI2_OFFSET_ASIAN] = chi2Corrections[__mul24(blockIdx.x, ASIAN_WALLACE_CHI2_VALUES_PER_BLOCK) + loop];
		__syncthreads();
		float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET_ASIAN];
		for (int i = 0; i < 8; i += 2)
		{
			seed = (1664525U * seed + 1013904223U) & 0xFFFFFFFF;

			ra = getRandomValue(i, loop, chi2CorrAndScale);
			rb = getRandomValue(i + 1, loop, chi2CorrAndScale);

			a *= exp(MU_A + ra * SIG_AA);
			b *= exp(MU_B + ra * SIG_AB + rb * SIG_BB);

			s = max(a, b);
			sum += s;
		}

		// Count up and temporarily store loop value to ease register on transform
		loop++;

		// Transform the pool
		transform_pool(seed);

	}

	return max((sum / ASIAN_TIME_STEPS) - s, (float) 0.0);
}

__global__ void wallace_asian_basket(unsigned seed,
									 float *chi2Corrections,
									 float *globalPool,
									 float *simulationResultsMean,
									 float *simulationResultsVariance,
									 float *g_A_0, float *g_B_0, float *g_MU_A, float *g_SIG_AA, float *g_MU_B, float *g_SIG_AB, float *g_SIG_BB)
{
	// Initialise loop and temporary loop storage
	unsigned loop = 0;

	// Define and load parameters for simulation
	float A_0, B_0, MU_A, SIG_AA, MU_B, SIG_AB, SIG_BB;
	unsigned address = (blockIdx.x * WALLACE_NUM_THREADS + threadIdx.x);
	A_0 = g_A_0[address];
	B_0 = g_B_0[address];
	MU_A = g_MU_A[address];
	MU_B = g_MU_B[address];
	SIG_AA = g_SIG_AA[address];
	SIG_AB = g_SIG_AB[address];
	SIG_BB = g_SIG_BB[address];

	// Initialise generator
	initialise_wallace(seed, globalPool);

	__syncthreads();

	float mean = 0, varAcc = 0;
	for (float i = 1; i <= ASIAN_PATHS_PER_SIM; i++)
	{
		float res = wallace_asian_basket_sim(seed, loop, chi2Corrections, A_0, B_0,
											 MU_A, MU_B, SIG_AA, SIG_AB, SIG_BB);

		// update mean and variance in a numerically stable way
		float delta = res - mean;
		mean += delta / i;
		varAcc += delta * (res - mean);
	}
	simulationResultsMean[address] = mean;

	float variance = varAcc / (ASIAN_PATHS_PER_SIM - 1);
	simulationResultsVariance[address] = variance;
}

__device__ float tausworthe_asian_basket_sim(float A_0, float B_0, float MU_A, float SIG_AA, float MU_B, float SIG_AB, float SIG_BB, unsigned &z1, unsigned &z2,
											 unsigned &z3, unsigned &z4)
{
	float a = A_0, b = B_0, s = 0, sum = 0;
	float temp_random_value;

	// Timesteps for a single simulation
	// Divide by 4 because we then do an internal loop 4 times
	for (unsigned t = 0; t < (ASIAN_TIME_STEPS / 4); t++)
	{
		float ra, rb;
		for (int i = 0; i < 8; i += 2)
		{
			ra = getRandomValueTausworthe(z1, z2, z3, z4, temp_random_value, 0);
			rb = getRandomValueTausworthe(z1, z2, z3, z4, temp_random_value, 1);

			a *= exp(MU_A + ra * SIG_AA);
			b *= exp(MU_B + ra * SIG_AB + rb * SIG_BB);

			s = max(a, b);
			sum += s;
		}
	}

	return max((sum / ASIAN_TIME_STEPS) - s, (float) 0.0);
}

__global__ void tausworthe_asian_basket(unsigned int *seedValues,
										float *simulationResultsMean,
										float *simulationResultsVariance,
										float *g_A_0, float *g_B_0, float *g_MU_A, float *g_SIG_AA, float *g_MU_B, float *g_SIG_AB, float *g_SIG_BB)
{
	// RNG state
	unsigned z1, z2, z3, z4;

	// Initialise tausworth with seeds
	z1 = seedValues[threadIdx.x];
	z2 = seedValues[TAUSWORTHE_TOTAL_NUM_THREADS + threadIdx.x];
	z3 = seedValues[2 * TAUSWORTHE_TOTAL_NUM_THREADS + threadIdx.x];
	z4 = seedValues[3 * TAUSWORTHE_TOTAL_NUM_THREADS + threadIdx.x];

	// Define and load parameters for simulation
	float A_0, B_0, MU_A, SIG_AA, MU_B, SIG_AB, SIG_BB;
	unsigned address = (blockIdx.x * TAUSWORTHE_NUM_THREADS + threadIdx.x);
	A_0 = g_A_0[address];
	B_0 = g_B_0[address];
	MU_A = g_MU_A[address];
	MU_B = g_MU_B[address];
	SIG_AA = g_SIG_AA[address];
	SIG_AB = g_SIG_AB[address];
	SIG_BB = g_SIG_BB[address];

	float mean = 0, varAcc = 0;
	for (float i = 1; i <= ASIAN_PATHS_PER_SIM; i++)
	{
		float res = tausworthe_asian_basket_sim(A_0, B_0, MU_A, MU_B, SIG_AA, SIG_AB,
												SIG_BB, z1, z2, z3, z4);

		// update mean and variance in a numerically stable way
		float delta = res - mean;
		mean += delta / i;
		varAcc += delta * (res - mean);
	}

	simulationResultsMean[address] = mean;

	float variance = varAcc / (ASIAN_PATHS_PER_SIM - 1);
	simulationResultsVariance[address] = variance;
}


void computeAsianOptions()
{
	init_asian_options();

	// setup execution parameters and execute
	dim3 asian_tausworth_grid(TAUSWORTHE_NUM_BLOCKS, 1, 1);
	dim3 asian_tausworth_threads(TAUSWORTHE_NUM_THREADS, 1, 1);
	dim3 asian_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
	dim3 asian_wallace_threads(WALLACE_NUM_THREADS, 1, 1);

	// Execute the Tausworthe version of the code, timing as we go
	CUT_SAFE_CALL(cutStartTimer(timer_asian_tw));
	tausworthe_asian_basket <<< asian_tausworth_grid, asian_tausworth_threads,
		0 >>> (deviceTauswortheSeeds, device_asianSimulationResultsMean,
			   device_asianSimulationResultsVariance, device_asian_A_0,
			   device_asian_B_0, device_asian_MU_A, device_asian_SIG_AA, device_asian_MU_B, device_asian_SIG_AB, device_asian_SIG_BB);
	CUT_SAFE_CALL(cutStopTimer(timer_asian_tw));
	CUT_CHECK_ERROR("Kernel execution failed: asian tausworthe");

	unsigned seed = 1;

	CUT_SAFE_CALL(cutStartTimer(timer_asian_wallace));

	// Execute the Wallace version of the code, timing as we go
	// Extra shared memory space to store loop counter temporarily to ease register pressure
	wallace_asian_basket <<< asian_wallace_grid, asian_wallace_threads,
		WALLACE_POOL_SIZE * 4 + WALLACE_NUM_THREADS * 4 +
		WALLACE_CHI2_SHARED_SIZE * 4 >>> (seed, deviceAsianChi2Corrections,
										  devPool,
										  device_asianSimulationResultsMean,
										  device_asianSimulationResultsVariance,
										  device_asian_A_0, device_asian_B_0,
										  device_asian_MU_A, device_asian_SIG_AA, device_asian_MU_B, device_asian_SIG_AB, device_asian_SIG_BB);
	CUT_SAFE_CALL(cutStopTimer(timer_asian_wallace));

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed: asian wallace");

	CUT_SAFE_CALL(cutStartTimer(timer_asian_download));
	CUDA_SAFE_CALL(cudaMemcpy(asianSimulationResultsMean, device_asianSimulationResultsMean, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(asianSimulationResultsVariance, device_asianSimulationResultsVariance, 4 * ASIAN_NUM_PARAMETER_VALUES, cudaMemcpyDeviceToHost));
	CUT_SAFE_CALL(cutStopTimer(timer_asian_download));

	printf("\n\nAsian option results:\n");
	printf
		("Processing time for asian initialisation code: %f (ms) for %d Simulations, %f MSimulations/sec\n",
		 cutGetTimerValue(timer_asian_init), ASIAN_NUM_PARAMETER_VALUES * ASIAN_PATHS_PER_SIM,
		 ASIAN_NUM_PARAMETER_VALUES * ASIAN_PATHS_PER_SIM / (cutGetTimerValue(timer_asian_init) / 1000.0) / 1000000.0);
	printf
		("Processing time for asian tausworthe: %f (ms) for %d Steps, %f MSteps/sec\n",
		 cutGetTimerValue(timer_asian_tw), ASIAN_NUM_PARAMETER_VALUES * ASIAN_PATHS_PER_SIM * ASIAN_TIME_STEPS,
		 ASIAN_NUM_PARAMETER_VALUES * ASIAN_PATHS_PER_SIM * ASIAN_TIME_STEPS / (cutGetTimerValue(timer_asian_tw) / 1000.0) / 1000000.0);
	printf("Processing time for asian wallace: %f (ms) for %d Simulations, %f Simulations/sec\n", cutGetTimerValue(timer_asian_wallace),
		   ASIAN_NUM_PARAMETER_VALUES, ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_wallace) / 1000.0) / 1000000.0);
	printf("Upload time for asian options: %f (ms) for %d bytes, %f MB/sec\n", cutGetTimerValue(timer_asian_upload),
		   4 * ASIAN_WALLACE_CHI2_COUNT + 7 * 4 * ASIAN_NUM_PARAMETER_VALUES,
		   (4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_upload))) / 1000.0);
	printf("Download time for asian options: %f (ms) for %d bytes, %f MB/sec\n", cutGetTimerValue(timer_asian_download),
		   4 * ASIAN_WALLACE_CHI2_COUNT + 4 * ASIAN_NUM_PARAMETER_VALUES,
		   (4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_download))) / 1000.0);
	printf("Malloc time for asian options: %f (ms) for %d bytes, %f MB/sec\n", cutGetTimerValue(timer_asian_malloc),
		   4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES,
		   (4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_malloc))) / 1000.0);
	printf("cudaMalloc time for asian options: %f (ms) for %d bytes, %f MB/sec\n", cutGetTimerValue(timer_asian_cuda_malloc),
		   4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES,
		   (4 * ASIAN_WALLACE_CHI2_COUNT + 7 * 4 * ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_cuda_malloc))) / 1000.0);
	printf("free time for asian options: %f (ms) for %d bytes, %f MB/sec\n", cutGetTimerValue(timer_asian_free),
		   4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES,
		   (4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_free))) / 1000.0);
	printf("cudaFree time for asian options: %f (ms) for %d bytes, %f MB/sec\n", cutGetTimerValue(timer_asian_cuda_free),
		   4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES,
		   (4 * ASIAN_WALLACE_CHI2_COUNT + 9 * 4 * ASIAN_NUM_PARAMETER_VALUES / (cutGetTimerValue(timer_asian_cuda_free))) / 1000.0);

	cleanup_asian_options();
}



#endif // #ifndef _TEMPLATE_KERNEL_H_
