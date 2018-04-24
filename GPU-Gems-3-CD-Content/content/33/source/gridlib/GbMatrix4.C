#ifdef USE_RCSID
static const char RCSid_GbMatrix4[] = "$Id: GbMatrix4.C,v 1.3 2003/03/06 17:01:52 prkipfer Exp $";
#endif

#ifdef OUTLINE

#include "GbMatrix4.hh"
#include "GbMatrix4.in"
#include "GbMatrix4.T"

// instantiate templates
template class GRIDLIB_API GbMatrix4<float>;
template class GRIDLIB_API GbMatrix4<double>;
// instantiate friends
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbMatrix4<float>&);
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbMatrix4<double>&);
template GRIDLIB_API std::istream& operator>>(std::istream&, GbMatrix4<float>&);
template GRIDLIB_API std::istream& operator>>(std::istream&, GbMatrix4<double>&);
// initialize static consts
#if 0
def WIN32
const GbMatrix4<float> GbMatrix4<float>::ZERO      = GbMatrix4<float>(0.f,0.f,0.f,0.f, 0.f,0.f,0.f,0.f, 0.f,0.f,0.f,0.f, 0.f,0.f,0.f,0.f);
const GbMatrix4<float> GbMatrix4<float>::IDENTITY  = GbMatrix4<float>(1.f,0.f,0.f,0.f, 0.f,1.f,0.f,0.f, 0.f,0.f,1.f,0.f, 0.f,0.f,0.f,1.f);
const GbMatrix4<double> GbMatrix4<double>::ZERO    = GbMatrix4<double>(0.,0.,0.,0., 0.,0.,0.,0., 0.,0.,0.,0., 0.,0.,0.,0.);
const GbMatrix4<double> GbMatrix4<double>::IDENTITY= GbMatrix4<double>(1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.);
#endif

#endif 
