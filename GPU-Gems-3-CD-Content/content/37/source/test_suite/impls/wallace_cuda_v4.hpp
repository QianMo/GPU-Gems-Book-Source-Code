// ************************************************
// wallace_cuda_v4.hpp
// authors: Lee Howes and David B. Thomas
//
// Wrapper for allowing the calling of a specific
// Wallace implementation in CUDA from the
// test framework.
// ************************************************


#ifndef wallace_cuda_v4_hpp
#define wallace_cuda_v4_hpp

#include "wallace_cuda.hpp"

#include <cmath>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

#include "../cuda/constants.h"

extern "C" void cuda_wallace_two_block(unsigned int seed, float *chi2Corrections, float *pool, float *poolOut, float *output);

extern "C" void CUDATransformFunc_V4(unsigned POOL_SIZE, unsigned BATCH_SIZE, unsigned BATCH_COUNT, unsigned seed, float *chi2CorrIn,	// [in] BATCH_SIZE
									 float *poolIn,	// [in] POOL_SIZE
									 float *poolOut,	// [out] POOL_SIZE. poolIn!=poolOut
									 float *output	// [out] POOL_SIZE*BATCH_SIZE
	)
{
	cuda_wallace_two_block(seed, chi2CorrIn, poolIn, poolOut, output);
}

boost::shared_ptr < RNG > MakeWallaceCUDA_V4()
{
	return boost::shared_ptr < RNG > (new WallaceCUDA < CUDATransformFunc_V4,
									  WALLACE_POOL_SIZE,
									  WALLACE_NUM_OUTPUTS_PER_RUN,
									  WALLACE_NUM_BLOCKS > ("WallaceCUDA_V4", "Implements a v4 variant using CUDA, via the WallaceCUDA class."));
}

#endif
