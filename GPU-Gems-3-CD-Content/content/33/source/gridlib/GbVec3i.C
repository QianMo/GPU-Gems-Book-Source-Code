#ifdef USE_RCSID
static const char RCSid_GbVec3i[] = "$Id: GbVec3i.C,v 1.3 2003/03/06 17:01:52 prkipfer Exp $";
#endif

#ifdef OUTLINE

#include "GbVec3i.hh"
#include "GbVec3i.in"
#include "GbVec3i.T"


// instantiate templates
template class GRIDLIB_API GbVec3i<int>;
// instantiate friends
template GRIDLIB_API GbVec3i<int> operator*(int s, const GbVec3i<int>& a);
template GRIDLIB_API std::ostream& operator << (std::ostream&, const GbVec3i<int>&);
template GRIDLIB_API std::istream& operator >> (std::istream&, GbVec3i<int>&);
// initialize static consts
#if 0
def WIN32
const GbVec3i<int> GbVec3i<int>::ZERO   = GbVec3i<int>(0,0,0);
const GbVec3i<int> GbVec3i<int>::UNIT_X = GbVec3i<int>(1,0,0);
const GbVec3i<int> GbVec3i<int>::UNIT_Y = GbVec3i<int>(0,1,0);
const GbVec3i<int> GbVec3i<int>::UNIT_Z = GbVec3i<int>(0,0,1);
#endif

#endif
