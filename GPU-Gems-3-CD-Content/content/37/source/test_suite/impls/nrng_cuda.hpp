// ************************************************
// nrng_cuda.hpp
// authors: Lee Howes and David B. Thomas
//
// Wrapper for allowing the calling of CUDA based
// normal random number generators from the
// test framework.
// ************************************************

#ifndef nrng_cuda_hpp
#define nrng_cuda_hpp

#include "batched_rng_base.hpp"

#include <cmath>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

typedef void (*cuda_nrng_func_t) (unsigned STATE_SIZE,	// size of each thread's state size
								  unsigned RNG_COUNT,	// number of rngs (i.e. total threads across all grids)
								  unsigned PER_RNG_OUTPUT_COUNT,	// number of outputs for each RNG
								  unsigned *state,	// [in,out] STATE_SIZE*RNG_COUNT  On output is assumed to contain updated state.
								  float *output	// [out] RNG_COUNT*PER_RNG_OUTPUT_SIZE
	);

template < cuda_nrng_func_t RNG_FUNC, unsigned STATE_SIZE, unsigned RNG_COUNT,
	unsigned PER_RNG_OUTPUT_COUNT > class NrngCUDA:public BatchedRngBase < RNG_COUNT * PER_RNG_OUTPUT_COUNT >
{
  private:
	boost::mt19937 m_mt;

	BOOST_STATIC_ASSERT(sizeof(unsigned) == 4);
	unsigned m_states[STATE_SIZE * RNG_COUNT];

	const char *m_name, *m_desc;

	void InitSeeds()
	{
		for (unsigned i = 0; i < STATE_SIZE * RNG_COUNT; i++)
		{
			do
			{
				m_states[i] = m_mt();
			}
			while (m_states[i] < 128);
		}
	}
  protected:
	virtual void GenerateImpl(float *dest)
	{
		RNG_FUNC(STATE_SIZE, RNG_COUNT, PER_RNG_OUTPUT_COUNT, m_states, dest);
	}
  public:
	NrngCUDA(const char *name, const char *desc):m_mt(lrand48()), m_name(name), m_desc(desc)
	{
		InitSeeds();
	}

	virtual const char *Name()
	{
		return m_name;
	}
	virtual const char *Description()
	{
		return m_desc;
	}
};

#endif
