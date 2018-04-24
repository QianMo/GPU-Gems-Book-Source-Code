/*----------------------------------------------------------------------
|
| $Id: GbPlane.hh,v 1.5 2003/03/06 17:01:52 prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBPLANE_HH
#define  GBPLANE_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbVec3.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief A simple plane class with pseudodistance query
  \author Peter Kipfer
  $Revision: 1.5 $
  $Date: 2003/03/06 17:01:52 $

  \note This is a version modified for GPUGems from the original gridlib file

  This class implements a simple plane. The plane is defined by a reference point
  and a normal vector. The reference point is expressed by a constant given the
  distance from the origin in direction of the normal. So for any point \p P on the
  plane with normal \p N
  
  \f[  \langle N ; P \rangle = const \f]

 */
template <class T>
class GRIDLIB_API GbPlane
{
public:
  //! Construct an empty (invalid) plane object: normal is zero
  INLINE GbPlane ();
  //! Construct a plane with given normal and constant
  INLINE GbPlane (const GbVec3<T>& rkNormal, T fConstant);
  //! Construct a plane with given normal and reference point
  INLINE GbPlane (const GbVec3<T>& rkNormal, const GbVec3<T>& rkPoint);
  //! Construct a plane through the given three points
  GbPlane (const GbVec3<T>& rkPoint0, const GbVec3<T>& rkPoint1, const GbVec3<T>& rkPoint2);

  //! Set the plane normal
  INLINE void setNormal(GbVec3<T>& n);
  //! Get the plane normal
  INLINE const GbVec3<T> getNormal () const;

  //! Set the plane constant
  INLINE void setConstant(T c);
  //! Get the plane constant
  INLINE const T getConstant () const;

  //! Access the plane \p P as \p P[0]=N.x, \p P[1]=N.y, \p P[2]=N.z, \p P[3]=c
  INLINE T operator[] (int i) const;

  /*! 
    \brief Symbols for the two half spaces

    The "positive side" of the plane is the half space to which the plane
    normal points.  The "negative side" is the other half space.  The symbol
    "no side" indicates the plane itself.
  */
  typedef enum {
    NO_SIDE,         //< The plane itself
    POSITIVE_SIDE,   //< The half space to which the normal points
    NEGATIVE_SIDE    //< The other half space
  } Side;

  //! Check on which side of the plane the given point is
  Side whichSide (const GbVec3<T>& rkPoint) const;

  //! Pseudodistance to the plane
  INLINE T distanceTo (const GbVec3<T>& rkPoint) const;

  //! Output the plane in specific ASCII format
  //friend GRIDLIB_API std::ostream& operator<< GB_TEXPORT (std::ostream&, const GbPlane<T>&);

protected:
  //! Storage for the plane normal
  GbVec3<T> normal_;
  //! Storage for the plane constant
  T constant_;
};

/*!
  \param s The stream to write to
  \param v The plane to be written
  \return The filled stream
*/
template<class T>
std::ostream&
operator<<(std::ostream& s, const GbPlane<T>& v)
{
  s<<typeid(v).name()<<": normal("<<v.getNormal()[0]<<","<<v.getNormal()[1]<<","<<v.getNormal()[2]<<") ";
  s<<"const("<<v.getConstant()<<")"<<std::endl;
  return s;
}

#ifndef OUTLINE

#include "GbPlane.in"
#include "GbPlane.T"

#else

INSTANTIATE( GbPlane<float> );
INSTANTIATE( GbPlane<double> );
INSTANTIATEF( std::ostream& operator<<(std::ostream& s, const GbPlane<float>& v) );
INSTANTIATEF( std::ostream& operator<<(std::ostream& s, const GbPlane<double>& v) );

#endif 

#endif  // GBPLANE_HH
