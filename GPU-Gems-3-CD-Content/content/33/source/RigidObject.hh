/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/

#ifndef  RIGIDOBJECT_HH
#define  RIGIDOBJECT_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "RigidShape.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

class RigidObject
    : public RigidShape
{
public:
    RigidObject(float fScale, float fMass, const GbVec3<float>& rkPos,
		const GbVec3<float>& rkLinMom, const GbVec3<float>& rkAngMom);
};

// #ifndef OUTLINE
// #include "RigidObject.in"
// #endif  // OUTLINE

#endif  // RIGIDOBJECT_HH
/*----------------------------------------------------------------------
|
| $Log$
|
+---------------------------------------------------------------------*/
