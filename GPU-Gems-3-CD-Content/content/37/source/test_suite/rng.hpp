// ************************************************
// rng.hpp
// authors: Lee Howes and David B. Thomas
//
// Base class for random number generator
// implementations to ease use of mappers
// and so on.
// ************************************************

#ifndef rng_hpp
#define rng_hpp

#include <cmath>
#include <numeric>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

class RNG
{
  protected:
	float *m_rngOutputCurr, *m_rngOutputEnd;


	// must leave at least one value in curr..end that can be returned
	virtual void UpdateRngOutput() = 0;
  public:
	  RNG():m_rngOutputCurr(0), m_rngOutputEnd(0)
	{
	}

	virtual ~ RNG()
	{
	}

	virtual const char *Name() = 0;
	virtual const char *Description() = 0;

	virtual void Generate(unsigned count, float *values) = 0;

	// these have been pulled back from being virtual functions for
	// performance reasons. Now they can be inlined into the simulations.
	float Generate()
	{
		if (m_rngOutputCurr >= m_rngOutputEnd)
		{
			UpdateRngOutput();
			assert(m_rngOutputCurr < m_rngOutputEnd);
		}
		return *m_rngOutputCurr++;
	}
	float operator() ()
	{
		return Generate();
	}
};


#endif // wallace_base_hpp
