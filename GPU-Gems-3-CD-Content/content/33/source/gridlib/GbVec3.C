#ifdef USE_RCSID
static const char RCSid_GbVec3[] = "$Id: GbVec3.C,v 1.4 2003/03/06 17:01:52 prkipfer Exp $";
#endif

/*----------------------------------------------------------------------
|
|
| $Log: GbVec3.C,v $
| Revision 1.4  2003/03/06 17:01:52  prkipfer
| changed default template instantiation target to be GNU flavour
|
| Revision 1.3  2002/12/13 09:32:22  prkipfer
| changed memory pool scoping to work around Visual C++ bug
|
| Revision 1.2  2000/06/14 15:39:13  prkipfer
| improved base classes and added funcstruct processing for mesh
|
| Revision 1.1.1.1  2000/06/08 16:24:44  prkipfer
| Imported source tree to start the project
|
|
|
+---------------------------------------------------------------------*/

#ifdef OUTLINE

#include "GbVec3.hh"
#include "GbVec3.in"
#include "GbVec3.T"



// instantiate templates
template class GRIDLIB_API GbVec3<float>;
template class GRIDLIB_API GbVec3<double>; 
// instantiate friends
template GRIDLIB_API GbVec3<float> operator*(float s, const GbVec3<float>& a) ;
template GRIDLIB_API GbVec3<double> operator*(double s, const GbVec3<double>& a) ;
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbVec3<float>&) ; 
template GRIDLIB_API std::ostream& operator<<(std::ostream&, const GbVec3<double>&) ;
template GRIDLIB_API std::istream& operator>>(std::istream&, GbVec3<float>&) ;
template GRIDLIB_API std::istream& operator>>(std::istream&, GbVec3<double>&) ;
// initialize static consts
#if 0
def WIN32
const GbVec3<float> GbVec3<float>::ZERO   = GbVec3<float>(0.f,0.f,0.f);
const GbVec3<float> GbVec3<float>::UNIT_X = GbVec3<float>(1.f,0.f,0.f);
const GbVec3<float> GbVec3<float>::UNIT_Y = GbVec3<float>(0.f,1.f,0.f);
const GbVec3<float> GbVec3<float>::UNIT_Z = GbVec3<float>(0.f,0.f,1.f);
const GbVec3<double> GbVec3<double>::ZERO   = GbVec3<double>(0.,0.,0.);
const GbVec3<double> GbVec3<double>::UNIT_X = GbVec3<double>(1.,0.,0.);
const GbVec3<double> GbVec3<double>::UNIT_Y = GbVec3<double>(0.,1.,0.);
const GbVec3<double> GbVec3<double>::UNIT_Z = GbVec3<double>(0.,0.,1.);
#endif

#endif 
