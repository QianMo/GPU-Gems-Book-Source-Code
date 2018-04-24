// ************************************************
// gem_asian_option.hpp
// authors: Lee Howes and David B. Thomas
//
// Implementation of a Asian option simulation.
// ***********************************************

#ifndef __gem_asian_option_hpp
#define __gem_asian_option_hpp

#include "simulation_impl.hpp"

template < unsigned MAX_T > class GemAsianOption
{
  private:
	float m_A0, m_B0;
	float m_muA, m_muB;
	float m_sig_aa, m_sig_bb, m_sig_ab;
	unsigned m_T;
  public:
	  GemAsianOption()
	{
		m_A0 = 1 + drand48();
		m_B0 = 1 + drand48();
		m_muA = drand48() * 0.01;
		m_muB = drand48() * 0.01;
		m_sig_aa = drand48() * 0.5;
		m_sig_bb = drand48() * 0.25;
		m_sig_ab = drand48() * 0.25;

		m_T = 256;
	}

	template < class TSrc > float operator() (TSrc & src) const
	{
		float a = m_A0, b = m_B0, sum = 0;
		for (unsigned i = 0; i < m_T; i++)
		{
			float ra = src(), rb = src();
			  a *= exp(m_muA + ra * m_sig_aa);
			  b *= exp(m_muB + ra * m_sig_ab + rb * m_sig_bb);
			  sum += std::max(a, b);
		}
		return std::max(sum / m_T - std::max(a, b), 0.0f);
	}

	unsigned StepsPerSim() const
	{
		return m_T;
	}

	unsigned RngsPerSim() const
	{
		return 2 * m_T;
	}

	const char *Name() const
	{
		return "GemAsianOption";
	}

	const char *Description() const
	{
		return "Pays difference between average of max of two options and final price.";
	}
};

boost::shared_ptr < Simulation > MakeGemAsianOption()
{
	return boost::shared_ptr < Simulation > (new SimulationImpl < GemAsianOption < 16 > >());
}

#endif
