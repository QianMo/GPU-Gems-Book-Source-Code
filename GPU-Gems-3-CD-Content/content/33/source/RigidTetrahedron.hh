/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/

#ifndef  RIGIDTETRAHEDRON_HH
#define  RIGIDTETRAHEDRON_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "RigidShape.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

class RigidTetrahedron
    : public RigidShape
{
public:
    RigidTetrahedron(float fSize, float fMass, const GbVec3<float>& rkPos,
		     const GbVec3<float>& rkLinMom, const GbVec3<float>& rkAngMom);
};

// #ifndef OUTLINE
// #include "RigidTetrahedron.in"
// #endif  // OUTLINE

#endif  // RIGIDTETRAHEDRON_HH
/*----------------------------------------------------------------------
|
| $Log$
|
+---------------------------------------------------------------------*/
