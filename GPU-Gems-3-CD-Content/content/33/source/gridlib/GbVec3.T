// -*- C++ -*- automatisch in C++mode wechseln (emacs)
/*!
  Predefined zero vector
*/
template <class T>
const GbVec3<T>
GbVec3<T>::ZERO = GbVec3<T>(T(0.0),T(0.0),T(0.0));

/*!
  Predefined unit vector X direction
*/
template <class T>
const GbVec3<T>
GbVec3<T>::UNIT_X = GbVec3<T>(T(1.0),T(0.0),T(0.0));

/*!
  Predefined unit vector Y direction
*/
template <class T>
const GbVec3<T>
GbVec3<T>::UNIT_Y = GbVec3<T>(T(0.0),T(1.0),T(0.0));

/*!
  Predefined unit vector Z direction
*/
template <class T>
const GbVec3<T>
GbVec3<T>::UNIT_Z = GbVec3<T>(T(0.0),T(0.0),T(1.0));

/*!
  Threshold for comparison operators. Set this to > 0 to enable
  fuzzy comparison logic.

  \warning As this is a static member, modifying it applys to all
           successive vector operations that support fuzzy logic.
	   Don't forget to reset it when done.
*/
template <class T>
T
GbVec3<T>::EPSILON = T(0.0);

/*!
  Construct a 3D vector. The data storage is not initialized. This enables 
  fast construction if the compiler is able to optimize it. Use one of
  the other constructors if you want an initialized vector, or assign one
  of the predefined unit vectors.
*/
template <class T>
GbVec3<T>::GbVec3() 
{
}

/*!
  \param s A scalar value

  Construct a 3D vector. All coordinates are initialized with the scalar
  \a s.
*/
template <class T>
GbVec3<T>::GbVec3(T s)
{ 
  xyz[0]=xyz[1]=xyz[2]=s; 
}

/*!
  \param x X coordinate
  \param y Y coordinate
  \param z Z coordinate

  Construct a 3D vector. The coordinates are explicitly initialized with
  the given values.
*/
template <class T>
GbVec3<T>::GbVec3(T x, T y, T z)
{ 
  xyz[0]=x; xyz[1]=y; xyz[2]=z; 
}

/*!
  \param p Array of three values for the coordinates

  Construct a 3D vector. The coordinates are explicitly initialized with
  the given values.
*/
template <class T>
GbVec3<T>::GbVec3(const T p[3])
{ 
  xyz[0]=p[0]; xyz[1]=p[1]; xyz[2]=p[2]; 
}

/*!
  \param p The vector to copy

  Construct a 3D vector. The coordinates are copied from the \a p vector.
*/
template <class T>
GbVec3<T>::GbVec3(const GbVec3<T> &p)
{ 
  xyz[0]=p.xyz[0]; xyz[1]=p.xyz[1]; xyz[2]=p.xyz[2]; 
}

/*!
  Destruct a 3D vector. Nothing unusual to pay attention to here. Has
  no side effects.
*/
template <class T>
GbVec3<T>::~GbVec3()
{
}
