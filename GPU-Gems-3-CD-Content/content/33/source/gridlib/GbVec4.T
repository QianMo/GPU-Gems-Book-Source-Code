// -*- C++ -*- automatisch in C++mode wechseln (emacs)

/*!
  Predefined zero vector
*/
template <class T>
const GbVec4<T>
GbVec4<T>::ZERO = GbVec4<T>(T(0.0),T(0.0),T(0.0),T(0.0));

/*!
  Predefined unit vector A dimension
*/
template <class T>
const GbVec4<T>
GbVec4<T>::UNIT_A = GbVec4<T>(T(1.0),T(0.0),T(0.0),T(0.0));

/*!
  Predefined unit vector B dimension
*/
template <class T>
const GbVec4<T>
GbVec4<T>::UNIT_B = GbVec4<T>(T(0.0),T(1.0),T(0.0),T(0.0));

/*!
  Predefined unit vector C dimension
*/
template <class T>
const GbVec4<T>
GbVec4<T>::UNIT_C = GbVec4<T>(T(0.0),T(0.0),T(1.0),T(0.0));

/*!
  Predefined unit vector D dimension
*/
template <class T>
const GbVec4<T>
GbVec4<T>::UNIT_D = GbVec4<T>(T(0.0),T(0.0),T(0.0),T(1.0));

/*!
  Threshold for comparison operators. Set this to > 0 to enable
  fuzzy comparison logic.

  \warning As this is a static member, modifying it applys to all
           successive vector operations that support fuzzy logic.
	   Don't forget to reset it when done.
*/
template <class T>
T
GbVec4<T>::EPSILON = T(0.0);

/*!
  Construct a 4D vector. The data storage is not initialized. This enables 
  fast construction if the compiler is able to optimize it. Use one of
  the other constructors if you want an initialized vector, or assign one
  of the predefined unit vectors.
*/
template <class T>
GbVec4<T>::GbVec4() 
{
}

/*!
  \param s A scalar value

  Construct a 4D vector. All coordinates are initialized with the scalar
  \a s.
*/
template <class T>
GbVec4<T>::GbVec4(T s)
{ 
  xyz[0]=xyz[1]=xyz[2]=xyz[3]=s; 
}

/*!
  \param a A dimension value
  \param b B dimension value
  \param c C dimension value
  \param d D dimension value

  Construct a 4D vector. The coordinates are explicitly initialized with
  the given values.
*/
template <class T>
GbVec4<T>::GbVec4(T a, T b, T c, T d)
{ 
  xyz[0]=a; xyz[1]=b; xyz[2]=c; xyz[3]=d; 
}

/*!
  \param p Array of four values for the coordinates

  Construct a 4D vector. The coordinates are explicitly initialized with
  the given values.
*/
template <class T>
GbVec4<T>::GbVec4(const T p[4])
{ 
  xyz[0]=p[0]; xyz[1]=p[1]; xyz[2]=p[2]; xyz[3]=p[3]; 
}
  
/*!
  \param p The vector to copy

  Construct a 4D vector. The coordinates are copied from the \a p vector.
*/
template <class T>
GbVec4<T>::GbVec4(const GbVec4<T> &p)
{ 
  xyz[0]=p.xyz[0]; xyz[1]=p.xyz[1]; xyz[2]=p.xyz[2]; xyz[3]=p.xyz[3]; 
}

/*!
  Destruct a 4D vector. Nothing unusual to pay attention to here. Has
  no side effects.
*/
template <class T>
GbVec4<T>::~GbVec4()
{
}
