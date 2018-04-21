/**
  @file Quake3Bsp.h

  @maintainer Kevin Egan (ktegan@cs.brown.edu)

  This class is actually not used because Quake3 files tend to have a lot
  of t-junctions in them making it impossible to get valid edges using
  our current algorithm.  And since the shadow algorithm depends on searching
  edges for silhouettes the level geometry is then not able to cast shadows
  (it can still have shadows cast on it).

*/

#ifndef _QUAKE_3_BSP_
#define _QUAKE_3_BSP_

#include <string>

class BasicModel;

BasicModel* loadQuake3Bsp(
        const std::string&          filename,
        int                         tesselationLevel,
        float                       scale);

#endif

