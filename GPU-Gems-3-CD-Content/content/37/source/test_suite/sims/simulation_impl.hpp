// ************************************************
// simulation_impl.hpp
// authors: Lee Howes and David B. Thomas
//
// Implementation of the Simulation abstract class.
// Takes a specific simulation class as a template 
// parameter to execute accordingly.
// ************************************************


#ifndef __simulation_impl_hpp
#define __simulation_impl_hpp

#include "simulation.hpp"

template < class TKernel > class SimulationImpl:public Simulation
{
  private:
	TKernel m_kernel;
  public:
	SimulationImpl()
	{
	}

	SimulationImpl(const TKernel & kernel):m_kernel(kernel)
	{
	}

	virtual const char *Name() const
	{
		return m_kernel.Name();
	}

	virtual const char *Description() const
	{
		return m_kernel.Description();
	}

	virtual unsigned StepsPerSim() const
	{
		return m_kernel.StepsPerSim();
	}

	virtual unsigned RngsPerSim() const
	{
		return m_kernel.RngsPerSim();
	}

	virtual std::pair < float, float >Execute(RNG * pSrc, double count) const
	{
		double mean = 0, S = 0;
		for (double i = 1; i <= count; i++)
		{
			float x = m_kernel(*pSrc);

			double delta = x - mean;
			  mean = mean + delta / i;
			  S = S + delta * (x - mean);
		}
		double variance = S / (count - 1);
		return std::make_pair((float) mean, (float) sqrt(variance));
	}
};

#endif
