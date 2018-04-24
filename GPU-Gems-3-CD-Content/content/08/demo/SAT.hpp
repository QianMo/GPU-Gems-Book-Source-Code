#pragma once

#include "VSM.hpp"
#include "PostProcess.hpp"

//--------------------------------------------------------------------------------------
namespace {
  // Misc UI
  const unsigned int c_SATGenerateTimeUpdatePeriod = 50;
}

//--------------------------------------------------------------------------------------
// Base class for all summed-area-tables implementations.
// Implements the filtering interface by generating summed area tables.
class SAT : public VSM
{
public:
  struct Stats
  {
    double GenerateTime;
    double GenerateTimeSum;
    unsigned int SinceUpdate;

    Stats()
      : GenerateTime(0.0), GenerateTimeSum(0.0), SinceUpdate(0)
    {}
  };

  virtual ~SAT();

  // Return statistics
  const Stats & GetStats() const
  {
    return m_Stats;
  }

protected:
  SAT(ID3D10Device *d3dDevice,
      ID3D10Effect* Effect,
      int Width, int Height,
      PostProcess* PostProcess,
      bool FPDistribute);

  PostProcess*                        m_PostProcess;         // Used for generating SATs
  bool                                m_FPDistribute;        // Distribute precision (2 textures)
  Stats                               m_Stats;               // Statistics about one SAT

private:
};