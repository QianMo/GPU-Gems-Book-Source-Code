/*
 * Pyramid.hpp
 *
 * Li-Yi Wei
 * 6/21/2002
 *
 */

#ifndef _PYRAMID_HPP
#define _PYRAMID_HPP

#include <string>
#include <fstream>

#include "Exception.hpp"
#include "Array.hpp"

template<class T>
class Pyramid
{
public:
    Pyramid(void);
    Pyramid(const int numLevels) throw(Exception);
    virtual ~Pyramid(void);

    T & operator[](const int level);
    const T & operator[](const int level) const;
    
    int NumLevels(void) const;

    int Read(const char * fileName);
    int Write(const char * fileName) const;
    
protected:
    virtual int Read_a(istream & input);
    virtual int Write_a(ostream & output) const;
     
protected:
    Array1D<T> _data;
};

template<class T>
Pyramid<T>::Pyramid(void)
{
    // nothing to do
}

template<class T>
Pyramid<T>::Pyramid(const int numLevels) throw(Exception) : _data(numLevels)
{
    // nothing to do
}

template<class T>
Pyramid<T>::~Pyramid(void)
{
    // nothing to do
}

template<class T>
T & Pyramid<T>::operator[](const int level)
{
    return _data[level];
}

template<class T>
const T & Pyramid<T>::operator[](const int level) const
{
    return _data[level];
}

template<class T>
int Pyramid<T>::NumLevels(void) const
{
    return _data.Size(0);
}

template<class T>
int Pyramid<T>::Read(const char * fileName)
{
    int result;
    
    ifstream input(fileName);
    
    if(! input )
    {
        return 0;  
    }
    
    result = Read_a(input);
    
    input.close();

    return result;
}

template<class T>
int Pyramid<T>::Write(const char * fileName) const
{
    int result;
    
    ofstream output(fileName);

    if(! output )
    {
        return 0;
    }
    
    result = Write_a(output);

    output.close();

    return result;
}

template<class T>
int Pyramid<T>::Read_a(istream & input)
{
    // shouldn't be called
    return 0;
}

template<class T>
int Pyramid<T>::Write_a(ostream & output) const
{
    // shouldn't be called
    return 0;
}

#endif
