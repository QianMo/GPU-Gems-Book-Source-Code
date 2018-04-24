// ************************************************
// constants.h
// authors: Lee Howes and David B. Thomas
//
// Exactly what it says on the tin. Constant 
// definitions to support the cuda code.
// Define block sizes, number of random
// numbers to generate in a sequence,
// memory offsets etc etc.
// ************************************************

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// Use fast trig functions, drastically changes execution
// time for the Box-Muller, but fast trig appears to give
// adequate random number quality as per the tests
// executed.
#define USE_FAST_TRIG 1

const float PI = 3.14159265f;


// ****************************
// Wallace parameters

const unsigned WALLACE_POOL_SIZE=2048;	// number of random numbers per wallace pool. Must be multiple of 16 for alignment
const unsigned WALLACE_POOL_SIZE_MASK=WALLACE_POOL_SIZE-1;
const unsigned WALLACE_RUNS_PER_THREAD=2;
const unsigned WALLACE_NUM_THREADS=WALLACE_POOL_SIZE/(4*WALLACE_RUNS_PER_THREAD);		// each thread transforms four values from pool
const unsigned WALLACE_MAX_OUTPUTS_PER_ITERATION=4; // maximum number of outputs per iteration if combine count is 1
#define WALLACE_OUTPUT_COMBINE_COUNT_DEF 1
const unsigned WALLACE_OUTPUT_COMBINE_COUNT=WALLACE_OUTPUT_COMBINE_COUNT_DEF;		// combine this many pool samples for each aggregate (can be 1,2 or 4)
const unsigned WALLACE_NUM_BLOCKS = 16;
const unsigned WALLACE_NUM_POOL_PASSES=1;

const unsigned WALLACE_NUM_OUTPUTS_PER_RUN=2048;

const unsigned WALLACE_NUM_RANDOM_NUMBERS_PER_THREAD=WALLACE_NUM_OUTPUTS_PER_RUN*WALLACE_RUNS_PER_THREAD*WALLACE_MAX_OUTPUTS_PER_ITERATION;
const unsigned WALLACE_TOTAL_NUM_THREADS = WALLACE_NUM_BLOCKS * WALLACE_NUM_THREADS;
const unsigned WALLACE_TOTAL_POOL_SIZE=WALLACE_POOL_SIZE*WALLACE_NUM_BLOCKS;
const unsigned WALLACE_NUM_RANDOM_NUMBERS_PER_BLOCK = WALLACE_NUM_RANDOM_NUMBERS_PER_THREAD*WALLACE_NUM_THREADS;
const unsigned WALLACE_OUTPUT_SIZE=WALLACE_NUM_RANDOM_NUMBERS_PER_BLOCK * WALLACE_NUM_BLOCKS;


// EAch block (threads in a block share) needs a chi2 value for each pool transform
const unsigned WALLACE_CHI2_VALUES_PER_BLOCK = WALLACE_NUM_OUTPUTS_PER_RUN;
// Chi2 is total number of values needed divided by number of values produced per iteration per block.
const unsigned WALLACE_CHI2_COUNT = WALLACE_CHI2_VALUES_PER_BLOCK*WALLACE_NUM_BLOCKS;

const unsigned WALLACE_CHI2_OFFSET=WALLACE_POOL_SIZE;
const unsigned WALLACE_CHI2_SHARED_SIZE = 1;


// ****************************
// Tausworth parameters
const unsigned TAUSWORTHE_NUM_THREADS = 256;
const unsigned TAUSWORTHE_NUM_BLOCKS = 16;
const unsigned TAUSWORTHE_TOTAL_NUM_THREADS = TAUSWORTHE_NUM_THREADS * TAUSWORTHE_NUM_BLOCKS;
const unsigned TAUSWORTHE_NUM_SEEDS_PER_GENERATOR = 4;
const unsigned TAUSWORTHE_NUM_SEEDS = TAUSWORTHE_NUM_BLOCKS * TAUSWORTHE_NUM_THREADS * TAUSWORTHE_NUM_SEEDS_PER_GENERATOR ;
const unsigned TAUSWORTHE_NUM_OUTPUTS_PER_RUN = WALLACE_NUM_OUTPUTS_PER_RUN;
const unsigned TAUSWORTHE_SEED_SIZE = TAUSWORTHE_NUM_THREADS*4;
const unsigned TAUSWORTHE_TOTAL_SEED_SIZE = TAUSWORTHE_SEED_SIZE * TAUSWORTHE_NUM_BLOCKS;
const unsigned TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_THREAD = TAUSWORTHE_NUM_OUTPUTS_PER_RUN * 8;
const unsigned TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_BLOCK = TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_THREAD*TAUSWORTHE_NUM_THREADS;
const unsigned TAUSWORTHE_OUTPUT_SIZE=TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_BLOCK * TAUSWORTHE_NUM_BLOCKS;


// ****************************
// Asian parameters
const unsigned ASIAN_TIME_STEPS=256;
const unsigned ASIAN_PATHS_PER_SIM=256;
// This will be the larger of the thread counts
const unsigned ASIAN_NUM_PARAMETER_VALUES = TAUSWORTHE_TOTAL_NUM_THREADS;

// EAch block (threads in a block share) needs a chi2 value for each pool transform
const unsigned ASIAN_WALLACE_CHI2_VALUES_PER_BLOCK = (ASIAN_PATHS_PER_SIM * ASIAN_TIME_STEPS/4) ;
// Chi2 is total number of values needed divided by number of values produced per iteration per block.
const unsigned ASIAN_WALLACE_CHI2_COUNT = WALLACE_NUM_BLOCKS*ASIAN_WALLACE_CHI2_VALUES_PER_BLOCK;

const unsigned WALLACE_CHI2_OFFSET_ASIAN=WALLACE_POOL_SIZE;


// ****************************
// Lookback parameters
const unsigned LOOKBACK_MAX_T = (4096-256)/TAUSWORTHE_NUM_THREADS;	
const unsigned LOOKBACK_NUM_PARAMETER_VALUES = TAUSWORTHE_TOTAL_NUM_THREADS;
const unsigned LOOKBACK_PATHS_PER_SIM=512;
const unsigned LOOKBACK_TAUSWORTHE_NUM_BLOCKS=16;
const unsigned LOOKBACK_TAUSWORTHE_NUM_THREADS=256;


// ****************************
// Trig functions

#if USE_FAST_TRIG
#define mc_sinf __sinf
#define mc_cosf __cosf
#define mc_logf __logf
#else
#define mc_sinf sinf
#define mc_cosf cosf
#define mc_logf logf
#endif


#endif
