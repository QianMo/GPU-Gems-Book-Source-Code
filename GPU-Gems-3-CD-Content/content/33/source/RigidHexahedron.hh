/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/

#ifndef  RIGIDHEXAHEDRON_HH
#define  RIGIDHEXAHEDRON_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "RigidShape.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

class RigidHexahedron
    : public RigidShape
{
public:
    RigidHexahedron(float fSize, float fMass, const GbVec3<float>& rkPos,
		    const GbVec3<float>& rkLinMom, const GbVec3<float>& rkAngMom);
};

// #ifndef OUTLINE
// #include "RigidHexahedron.in"
// #endif  // OUTLINE

#endif  // RIGIDHEXAHEDRON_HH
/*----------------------------------------------------------------------
|
| $Log$
|
+---------------------------------------------------------------------*/
