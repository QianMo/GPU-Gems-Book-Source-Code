// ************************************************
// gem_lookback_option.hpp
// authors: Lee Howes and David B. Thomas
//
// Implementation of a lookback option simulation.
// ************************************************

#ifndef __gem_lookback_option_hpp
#define __gem_lookback_option_hpp

#include "simulation_impl.hpp"

template < unsigned MAX_T > class GemLookbackOption
{
  private:
	float m_A0, m_A1, m_A2, m_mu;
	float m_sigma0, m_eps0, m_s0;
	unsigned m_T;
  public:
	  GemLookbackOption()
	{
		m_A0 = drand48() * 0.1;
		m_A1 = drand48() * 0.2;
		m_A2 = drand48() * 0.2;
		m_sigma0 = drand48() * 0.5;
		m_eps0 = drand48() * 0.5;
		m_s0 = 1 + drand48();
		m_T = 15;

		assert(m_T < MAX_T);
	}

	template < class TSrc > float operator() (TSrc & src) const
	{
		float history[MAX_T];
		float s = m_s0, sigma = m_sigma0, eps = m_eps0;
		for (unsigned t = 0; t < m_T; t++)
		{
			history[t] = s;
			sigma = sqrtf(m_A0 + m_A1 * sigma * sigma + m_A2 * eps * eps);
			eps = src() * sigma;
			s = s * expf(m_mu + eps);
		}

		float sum = 0;
		for (unsigned t = 0; t < m_T; t++)
		{
			sum += std::max(history[t] - s, 0.0f);
		}
		return sum;
	}

	unsigned StepsPerSim() const
	{
		return m_T;
	}

	unsigned RngsPerSim() const
	{
		return m_T;
	}

	const char *Name() const
	{
		return "GemLookbackOption";
	}

	const char *Description() const
	{
		return "Pays the sum of positive difference between end price and historical prices.";
	}
};

boost::shared_ptr < Simulation > MakeGemLookbackOption()
{
	return boost::shared_ptr < Simulation > (new SimulationImpl < GemLookbackOption < 16 > >());
}

#endif
