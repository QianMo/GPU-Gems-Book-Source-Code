// ************************************************
// wallace_v4.hpp
// authors: Lee Howes and David B. Thomas
//
// Wrapper for calling a specific CPU based
// Wallace implementation from the test 
// framework.
// ************************************************


#ifndef wallace_v4_hpp
#define wallace_v4_hpp

#include "../wallace_base.hpp"

#include <cmath>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

#ifdef NDEBUG
template < unsigned POOL_SIZE = 2048, bool VERIFY = false >
#else
template < unsigned POOL_SIZE = 2048, bool VERIFY = true >
#endif
class Wallace_V4:public WallaceBase < POOL_SIZE, POOL_SIZE / 8, 24, POOL_SIZE, VERIFY >
{
  protected:
	typedef WallaceBase < POOL_SIZE, POOL_SIZE / 8, 24, POOL_SIZE, VERIFY > base_t;

	enum
	{ NUM_THREADS = base_t::NUM_THREADS };
	enum
	{ POOL_SIZE_MASK = POOL_SIZE - 1 };

	float m_chi2Corr;
	unsigned m_seed;

	virtual void PreTransformSetup()
	{
		m_chi2Corr = MakeChi2Scale(POOL_SIZE);
		m_seed = (1664525U * m_seed + 1013904223U) & 0xFFFFFFFF;
	}

	// Runs a kernel thread. Per thread arguments are managed by the inheritor.
	virtual void TransformKernel(unsigned tid, unsigned pass)
	{
		assert(pass == 0);

		const unsigned lcg_a = 241;
		const unsigned lcg_c = 59;
		const unsigned lcg_m = 256;
		assert(lcg_m == POOL_SIZE / 8);

		unsigned addrIn[8];
		float tmp[8];

		unsigned seed = (m_seed + tid) % lcg_m;

		for (unsigned i = 0; i < 8; i++)
		{
			seed = (seed * lcg_a + lcg_c) % lcg_m;
			addrIn[i] = (seed << 3) + i;
			tmp[i] = base_t::ReadPool(tid, i, addrIn[i]);
		}

		Hadamard4x4a(tmp[0], tmp[1], tmp[2], tmp[3]);
		Hadamard4x4b(tmp[4], tmp[5], tmp[6], tmp[7]);

		// these seem like a good idea, as it swaps values between the
		// different bands of low order bits. In hardware it should be free.
		std::swap(tmp[0], tmp[7]);
		std::swap(addrIn[0], addrIn[7]);
		std::swap(tmp[3], tmp[4]);
		std::swap(addrIn[3], addrIn[4]);

		for (unsigned i = 0; i < 8; i++)
		{
			base_t::WritePool(tid, i + 8, i * NUM_THREADS + tid, tmp[i], 4, i < 4 ? addrIn : (addrIn + 4));
			base_t::WriteOutput(tid, i + 16, i * NUM_THREADS + tid, tmp[i] * m_chi2Corr);
		}
	}

  public:
  Wallace_V4():m_seed(lrand48())
	{
	}

	const char *Name()
	{
		return "Wallace_V4";
	}

	const char *Description()
	{
		return
			"Wallace implementation using an LCG to mix the pool resulting in a fair number of "
			"shared memory access collisions but good statistical quality.";
	}
};

#endif
