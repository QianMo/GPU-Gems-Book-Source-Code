// -*- C++ -*- automatisch in C++mode wechseln (emacs)

/*!
  \param rkPoint0 The first point
  \param rkPoint1 The second point
  \param rkPoint2 The third point

  Construct a plane through the three given points.
*/
template <class T>
GbPlane<T>::GbPlane (const GbVec3<T>& rkPoint0, 
		     const GbVec3<T>& rkPoint1,
		     const GbVec3<T>& rkPoint2)
{
  GbVec3<T> kEdge1 = rkPoint1 - rkPoint0;
  GbVec3<T> kEdge2 = rkPoint2 - rkPoint0;
  normal_ = kEdge1.unitCross(kEdge2);
  constant_ = normal_ | rkPoint0;
}

/*!
  \param rkPoint The point in question
  \return Symbol for the plane's side

  The "positive side" of the plane is the half space to which the plane
  normal points.  The "negative side" is the other half space.  The symbol
  "no side" indicates the plane itself.
*/
template <class T>
typename GbPlane<T>::Side 
GbPlane<T>::whichSide (const GbVec3<T>& rkPoint) const
{
  T fDistance = distanceTo(rkPoint);

  if ( fDistance < T(0.0) )
    return GbPlane::NEGATIVE_SIDE;

  if ( fDistance > T(0.0) )
    return GbPlane::POSITIVE_SIDE;

  return GbPlane::NO_SIDE;
}
