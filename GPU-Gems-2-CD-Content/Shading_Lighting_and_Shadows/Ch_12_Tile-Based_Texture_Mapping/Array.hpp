/*
 * Array.hpp
 *
 * Li-Yi Wei
 * 6/22/2002
 *
 */

#ifndef _ARRAY_HPP
#define _ARRAY_HPP

#include <vector>
#include "Exception.hpp"

template<class T>
class Array1D
{
public:
    Array1D(void);
    Array1D(const int size);
    ~Array1D(void);

    int Size(const int dimension) const;
    const T & operator[](const int index) const;
    T & operator[](const int index);
    
protected:
    vector<T> _data;
};

template<class T>
class Array2D
{
public:
    Array2D(void);
    Array2D(const int height, const int width);
    ~Array2D(void);

    int Size(const int dimension) const;
    const Array1D<T> & operator[](const int index) const;
    Array1D<T> & operator[](const int index);
    
protected:
    vector< Array1D<T> > _data;
};

template<class T>
class Array3D
{
public:
    Array3D(void);
    Array3D(const int height, const int width, const int depth);
    ~Array3D(void);

    int Size(const int dimension) const;
    const Array2D<T> & operator[](const int index) const;
    Array2D<T> & operator[](const int index);
    
protected:
    vector< Array2D<T> > _data;
};

/*
 * implementations
 *
 */

// Array1D

template<class T>
Array1D<T>::Array1D(void)
{
    // nothing to do
}

template<class T>
Array1D<T>::Array1D(const int size) : _data(size)
{
    // nothing to do
}

template<class T>
Array1D<T>::~Array1D(void)
{
    // nothing to do
}

template<class T>
int Array1D<T>::Size(const int dimension) const
{
    if(dimension == 0)
    {
        return _data.size();
    }
    else
    {
        return -1;
    }
}

template<class T>
const T & Array1D<T>::operator[](const int index) const
{
    return _data[index];
}

template<class T>
T & Array1D<T>::operator[](const int index)
{
    return _data[index];
}

// Array2D
template<class T>
Array2D<T>::Array2D(void)
{
    // nothing to do
}

template<class T>
Array2D<T>::Array2D(const int height, const int width) : _data(height)
{
    for(int i = 0; i < height; i++)
    {
        _data[i] = Array1D<T>(width);
    }
}

template<class T>
Array2D<T>::~Array2D(void)
{
    // nothing to do
}

template<class T>
int Array2D<T>::Size(const int dimension) const
{
    if(dimension == 0)
    {
        return _data.size();
    }
    else if(_data.size() > 0)
    {
        return _data[0].Size(dimension - 1);
    }
    else
    {
        return -1;
    }
}

template<class T>
const Array1D<T> & Array2D<T>::operator[](const int index) const
{
    return _data[index];
}

template<class T>
Array1D<T> & Array2D<T>::operator[](const int index)
{
    return _data[index];
}

// Array3D
template<class T>
Array3D<T>::Array3D(void)
{
    // nothing to do
}

template<class T>
Array3D<T>::Array3D(const int height, const int width, const int depth) : _data(height)
{
    for(int i = 0; i < height; i++)
    {
        _data[i] = Array2D<T>(width, depth);
    }
}

template<class T>
Array3D<T>::~Array3D(void)
{
    // nothing to do
}

template<class T>
int Array3D<T>::Size(const int dimension) const
{
    if(dimension == 0)
    {
        return _data.size();
    }
    else if(_data.size() > 0)
    {
        return _data[0].Size(dimension - 1);
    }
    else
    {
        return -1;
    }
}

template<class T>
const Array2D<T> & Array3D<T>::operator[](const int index) const
{
    return _data[index];
}

template<class T>
Array2D<T> & Array3D<T>::operator[](const int index)
{
    return _data[index];
}
    
#endif

