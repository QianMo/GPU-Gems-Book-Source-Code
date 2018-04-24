// -*- C++ -*- automatisch in C++mode wechseln (emacs)

/*!
  Predefined zero vector
*/
template <class T>
const GbVec3i<T>
GbVec3i<T>::ZERO = GbVec3i<T>(0,0,0);

/*!
  Predefined unit vector X direction
*/
template <class T>
const GbVec3i<T>
GbVec3i<T>::UNIT_X = GbVec3i<T>(1,0,0);

/*!
  Predefined unit vector Y direction
*/
template <class T>
const GbVec3i<T>
GbVec3i<T>::UNIT_Y = GbVec3i<T>(0,1,0);

/*!
  Predefined unit vector Z direction
*/
template <class T>
const GbVec3i<T>
GbVec3i<T>::UNIT_Z = GbVec3i<T>(0,0,1);

/*!
  Construct a 3D vector for integral numbers. The data storage is not initialized. This enables 
  fast construction if the compiler is able to optimize it. Use one of
  the other constructors if you want an initialized vector, or assign one
  of the predefined unit vectors.
*/
template <class T>
GbVec3i<T>::GbVec3i() 
{
}

/*!
  \param s A scalar value

  Construct a 3D vector for integral numbers. All coordinates are initialized with the scalar
  \a s.
*/
template <class T>
GbVec3i<T>::GbVec3i(T s)
{ 
  xyz[0]=xyz[1]=xyz[2]=s; 
}

/*!
  \param x X coordinate
  \param y Y coordinate
  \param z Z coordinate

  Construct a 3D vector for integral numbers. The coordinates are explicitly initialized with
  the given values.
*/
template <class T>
GbVec3i<T>::GbVec3i(T x, T y, T z)
{ 
  xyz[0]=x; xyz[1]=y; xyz[2]=z; 
}

/*!
  \param p Array of three values for the coordinates

  Construct a 3D vector for integral numbers. The coordinates are explicitly initialized with
  the given values.
*/
template <class T>
GbVec3i<T>::GbVec3i(const T p[3])
{ 
  xyz[0]=p[0]; xyz[1]=p[1]; xyz[2]=p[2]; 
}
  
/*!
  \param p The vector to copy

  Construct a 3D vector for integral numbers. The coordinates are copied from the \a p vector.
*/
template <class T>
GbVec3i<T>::GbVec3i(const GbVec3i<T> &p)
{ 
  xyz[0]=p.xyz[0]; xyz[1]=p.xyz[1]; xyz[2]=p.xyz[2]; 
}

/*!
  Destruct a 3D vector for integral numbers. Nothing unusual to pay attention to here. Has
  no side effects.
*/
template <class T>
GbVec3i<T>::~GbVec3i()
{
}
