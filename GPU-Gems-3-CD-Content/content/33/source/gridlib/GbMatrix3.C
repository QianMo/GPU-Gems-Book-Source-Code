#ifdef USE_RCSID
static const char RCSid_GbMatrix3[] = "$Id: GbMatrix3.C,v 1.3 2003/03/06 17:01:52 prkipfer Exp $";
#endif

#ifdef OUTLINE

#include "GbMatrix3.hh"
#include "GbMatrix3.in"
#include "GbMatrix3.T"

// instantiate templates
template class GRIDLIB_API GbMatrix3<float>;
template class GRIDLIB_API GbMatrix3<double>;
// instantiate friends
template GRIDLIB_API GbVec3<float>  operator* (const GbVec3<float>& v, const GbMatrix3<float>& m);
template GRIDLIB_API GbVec3<double> operator* (const GbVec3<double>& v, const GbMatrix3<double>& m);
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbMatrix3<float>&);
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbMatrix3<double>&);
template GRIDLIB_API std::istream& operator>>(std::istream&, GbMatrix3<float>&) ;
template GRIDLIB_API std::istream& operator>>(std::istream&, GbMatrix3<double>&); 

// initialize static consts
#if 0
def WIN32
const float GbMatrix3<float>::EPSILON = std::numeric_limits<float>::epsilon();
const GbMatrix3<float> GbMatrix3<float>::ZERO = GbMatrix3<float>(0.f,0.f,0.f, 0.f,0.f,0.f, 0.f,0.f,0.f);
const GbMatrix3<float> GbMatrix3<float>::IDENTITY = GbMatrix3<float>(1.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f,1.f);
const double GbMatrix3<double>::EPSILON = std::numeric_limits<double>::epsilon();
const GbMatrix3<double> GbMatrix3<double>::ZERO = GbMatrix3<double>(0.,0.,0., 0.,0.,0., 0.,0.,0.);
const GbMatrix3<double> GbMatrix3<double>::IDENTITY= GbMatrix3<double>(1.,0.,0., 0.,1.,0., 0.,0.,1.);
#endif

#endif

