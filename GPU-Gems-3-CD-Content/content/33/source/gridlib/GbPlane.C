#ifdef USE_RCSID
static const char RCSid_GbPlane[] = "$Id: GbPlane.C,v 1.3 2003/03/06 17:01:52 prkipfer Exp $";
#endif

#ifdef OUTLINE

#include "GbPlane.hh"
#include "GbPlane.in"
#include "GbPlane.T"

// instantiate templates
template class GRIDLIB_API GbPlane<float>;
template class GRIDLIB_API GbPlane<double>;
// instantiate friends
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbPlane<float>&);
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbPlane<double>&);
// initialize static consts


#endif
