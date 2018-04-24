// ************************************************
// simulation.hpp
// authors: Lee Howes and David B. Thomas
//
// Abstract simulation superclass forming part
// of options simulation code.
// ***********************************************

#ifndef __simulation_hpp
#define __simulation_hpp

#include "../wallace_base.hpp"

class Simulation
{
  public:
	virtual const char *Name() const = 0;
	virtual const char *Description() const = 0;

	virtual unsigned StepsPerSim() const = 0;
	virtual unsigned RngsPerSim() const = 0;

	virtual std::pair < float, float >Execute(RNG * pSrc, double count) const = 0;

	  virtual ~ Simulation()
	{
	}
};

#endif
