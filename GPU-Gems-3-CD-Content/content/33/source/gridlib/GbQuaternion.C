#ifdef USE_RCSID
static const char RCSid_GbQuaternion[] = "$Id: GbQuaternion.C,v 1.3 2003/03/06 17:01:52 prkipfer Exp $";
#endif

#ifdef OUTLINE

#include "GbQuaternion.hh"
#include "GbQuaternion.in"
#include "GbQuaternion.T"

// instantiate templates
template class GRIDLIB_API GbQuaternion<float>;
template class GRIDLIB_API GbQuaternion<double>;
// instantiate friends
template GRIDLIB_API GbQuaternion<float>  operator* (float fScalar, const GbQuaternion<float>& rkQ);
template GRIDLIB_API GbQuaternion<double> operator* (double fScalar, const GbQuaternion<double>& rkQ);
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbQuaternion<float>&);
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbQuaternion<double>&);
template GRIDLIB_API std::istream& operator>>(std::istream&, GbQuaternion<float>&);
template GRIDLIB_API std::istream& operator>>(std::istream&, GbQuaternion<double>&);
// initialize static consts
#if 0
def WIN32
const GbQuaternion<float> GbQuaternion<float>::ZERO = GbQuaternion<float>(0.f,0.f,0.f,0.f);
const GbQuaternion<float> GbQuaternion<float>::IDENTITY = GbQuaternion<float>(1.f,0.f,0.f,0.f);
const float GbQuaternion<float>::EPSILON = std::numeric_limits<float>::epsilon();
const GbQuaternion<double> GbQuaternion<double>::ZERO = GbQuaternion<double>(0.,0.,0.,0.);
const GbQuaternion<double> GbQuaternion<double>::IDENTITY = GbQuaternion<double>(1.,1.,1.,1.);
const double GbQuaternion<double>::EPSILON = std::numeric_limits<double>::epsilon();
#endif

#endif
