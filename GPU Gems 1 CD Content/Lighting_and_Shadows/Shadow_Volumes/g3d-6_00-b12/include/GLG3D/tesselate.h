/** 
  @file tesselate.h
 
  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2003-05-01
  @edited  2003-05-01

  Copyright 2000-2003, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_TESSELATE_H
#define G3D_TESSELATE_H

#include "G3D/Vector3.h"
#include "G3D/Triangle.h"
#include "G3D/Array.h"

namespace G3D {

/**
 Tesselates a complex polygon into a triangle set which is appended
 to the output.  The input is a series of counter-clockwise winding
 vertices, where the last is implicitly connected to the first.
 Self-intersections are allowed; "inside" is determined by an "odd"
 winding rule.  You may need to introduce a sliver polygon to cut
 holes out of the center.
 */
void tesselateComplexPolygon(Array<Vector3>& input, Array<Triangle>& output);

}

#endif
