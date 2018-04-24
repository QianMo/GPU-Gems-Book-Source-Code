#ifdef USE_RCSID
static const char RCSid_GoRigidBody[] = "$Id$";
#endif

#ifdef OUTLINE

#include "GoRigidBody.hh"
#include "GoRigidBody.in"
#include "GoRigidBody.T"



// instantiate templates
template class GRIDLIB_API GoRigidBody<float>;
template class GRIDLIB_API GoRigidBody<double>; 
// instantiate friends
// initialize static consts
#ifdef WIN32
#endif

#endif 
