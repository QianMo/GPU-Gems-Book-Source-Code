#ifdef USE_RCSID
static const char RCSid_GbVec4[] = "$Id: GbVec4.C,v 1.4 2003/03/06 17:01:52 prkipfer Exp $";
#endif

#ifdef OUTLINE

#include "GbVec4.hh"
#include "GbVec4.in"
#include "GbVec4.T"

// instantiate templates
template class GRIDLIB_API GbVec4<float>;
template class GRIDLIB_API GbVec4<double>; 
// instantiate friends
template GRIDLIB_API GbVec4<float> operator*(float s, const GbVec4<float>& a) ;
template GRIDLIB_API GbVec4<double> operator*(double s, const GbVec4<double>& a) ;
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbVec4<float>&) ; 
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbVec4<double>&) ;
template GRIDLIB_API std::istream& operator>>(std::istream&, GbVec4<float>&) ;
template GRIDLIB_API std::istream& operator>>(std::istream&, GbVec4<double>&) ;
// initialize static consts
#if 0
def WIN32
const GbVec4<float> GbVec4<float>::ZERO   = GbVec4<float>(0.f,0.f,0.f,0.f);
const GbVec4<float> GbVec4<float>::UNIT_A = GbVec4<float>(1.f,0.f,0.f,0.f);
const GbVec4<float> GbVec4<float>::UNIT_B = GbVec4<float>(0.f,1.f,0.f,0.f);
const GbVec4<float> GbVec4<float>::UNIT_C = GbVec4<float>(0.f,0.f,1.f,0.f);
const GbVec4<float> GbVec4<float>::UNIT_D = GbVec4<float>(0.f,0.f,0.f,1.f);
const GbVec4<double> GbVec4<double>::ZERO   = GbVec4<double>(0.,0.,0.,0.);
const GbVec4<double> GbVec4<double>::UNIT_A = GbVec4<double>(1.,0.,0.,0.);
const GbVec4<double> GbVec4<double>::UNIT_B = GbVec4<double>(0.,1.,0.,0.);
const GbVec4<double> GbVec4<double>::UNIT_C = GbVec4<double>(0.,0.,1.,0.);
const GbVec4<double> GbVec4<double>::UNIT_D = GbVec4<double>(0.,0.,0.,1.);
#endif

#endif
