// ************************************************
// nrng_cuda_tausworthe.hpp
// authors: Lee Howes and David B. Thomas
//
// Wrapper for allowing the calling of our 
// CUDA based Tausworthe generator
// from the test framework.
// ************************************************

#ifndef nrng_cuda_tausworthe_hpp
#define nrng_cuda_tausworthe_hpp

#include "nrng_cuda.hpp"

#include <cmath>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"


#include "../cuda/constants.h"


extern "C" void cuda_tausworthe(unsigned *pool, float *output);



extern "C" void NrngCUDA_Tausworthe(unsigned STATE_SIZE,	// size of each thread's state size
									unsigned RNG_COUNT,	// number of rngs (i.e. total threads across all grids)
									unsigned PER_RNG_OUTPUT_COUNT,	// number of outputs for each RNG
									unsigned *state,	// [in,out] STATE_SIZE*RNG_COUNT  On output is assumed to contain updated state.
									float *output	// [out] RNG_COUNT*PER_RNG_OUTPUT_SIZE
	)
{
	assert(Tausworthe::state_dword_count == STATE_SIZE);
	assert(STATE_SIZE == 4);
	assert((PER_RNG_OUTPUT_COUNT % 2) == 0);

	cuda_tausworthe(state, output);
}


boost::shared_ptr < RNG > MakeNrngCUDA_Tausworthe()
{
	return boost::shared_ptr < RNG > (new NrngCUDA < NrngCUDA_Tausworthe, 4,	//TAUSWORTHE_TOTAL_SEED_SIZE,
									  TAUSWORTHE_NUM_THREADS *
									  TAUSWORTHE_NUM_BLOCKS,
									  TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_THREAD
									  > ("NrngCUDA_Tausworthe", "Tausworthe implemented in CUDA via NrngCUDA, using box-muller."));
}

#endif
