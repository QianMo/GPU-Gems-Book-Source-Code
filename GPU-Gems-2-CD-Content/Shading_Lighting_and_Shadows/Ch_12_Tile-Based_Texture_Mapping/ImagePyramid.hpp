/*
 * ImagePyramid.hpp
 *
 * pyramid for 2D images (of certain components)
 *
 * Li-Yi Wei
 * 6/28/2002
 *
 */

#ifndef _IMAGE_PYRAMID_HPP
#define _IMAGE_PYRAMID_HPP

#include "Pyramid.hpp"

// each pyramid level contains a 3D array (height, width, numComponents)
// and the sizes of the first two dimensions must diminish in power 2
// as in ordinary image pyramids
template<class T>
class ImagePyramid : public Pyramid< Array3D<T> >
{
public:
    ImagePyramid(void);
    ImagePyramid(const char * fileName) throw(Exception);
    ImagePyramid(const Array2D<int> & specification) throw(Exception);
    virtual ~ImagePyramid(void);

protected:
    int CheckStructure(const Array2D<int> & spec) const;
    virtual int Read_a(istream & input);
    virtual int Write_a(ostream & output) const;
    void Error(const string message);

    int NewMemory(const Array2D<int> & spec);
    
protected:
    string _message;
};

template<class T>
ImagePyramid<T>::ImagePyramid(void)
{
    // nothing to do
}

template<class T>
ImagePyramid<T>::ImagePyramid(const char * fileName) throw(Exception) 
{
    if(! Read(fileName))
    {
        throw Exception("cannot construct pyramid from file : " + _message);
    }
}

template<class T>
ImagePyramid<T>::ImagePyramid(const Array2D<int> & spec) throw(Exception) : Pyramid< Array3D<T> >(spec.Size(0))
{
    if(! CheckStructure(spec))
    {
        throw Exception("illegal pyramid specification");
    }
    
    if(! NewMemory(spec))
    {
        throw Exception("cannot initialize memory");
    }       
}

template<class T>
ImagePyramid<T>::~ImagePyramid(void)
{
    // nothing to do
}

template<class T>
int ImagePyramid<T>::CheckStructure(const Array2D<int> & spec) const
{
    for(int i = 1; i < spec.Size(0); i++)
        for(int j = 0; j < 2; j++)
        {
            if(spec[i][j] != ((spec[i-1][j] + 1)/2))
            {
                return 0;
            }
        }

    return 1;
}

template<class T>
int ImagePyramid<T>::Read_a(istream & input)
{
    // number of levels
    int num_levels = 0;

    input >> num_levels;

    if(input.rdstate())
    {
        // error
        Error("error in reading number of levels");
        return 0;
    }

    if(num_levels <= 0)
    {
        Error("number of levels <= 0");
        return 0;
    }

    // spec
    Array2D<int> spec(num_levels, 3);

    for(int row = 0; row < spec.Size(0); row++)
        for(int col = 0; col < spec.Size(1); col++)
        {
            input >> spec[row][col];
        }

    if(input.rdstate())
    {
        Error("error in reading pyramid structure");
        return 0;
    }

    if(! NewMemory(spec) )
    {
        Error("error in memory allocation");
        return 0;
    }

    // read in the data
    // note the col major ordering
    // (for compatibility with Matlab)
    for(int level = 0; level < _data.Size(0); level++)
    {
        for(int col = 0; col < spec[level][1]; col++)
            for(int row = 0; row < spec[level][0]; row++)
                for(int cha = 0; cha < spec[level][2]; cha++)
                {
                    input >> _data[level][row][col][cha];
                }
    }

    if(input.rdstate())
    {
        Error("error in reading in pyramid data");
        return 0;
    }

    // done
    return 1;
}

template<class T>
int ImagePyramid<T>::Write_a(ostream & output) const
{
    // number of levels
    output << _data.Size(0)<<endl;
    
    // spec
    {
        for(int level = 0; level < _data.Size(0); level++)
        {
            for(int dim = 0; dim < 3; dim++)
            {
                output << _data[level].Size(dim)<<" ";
            }

            output<<endl;
        }
    }
    
    // data
    {
        for(int level = 0; level < _data.Size(0); level++)
        {
            for(int col = 0; col < _data[level].Size(1); col++)
                for(int row = 0; row < _data[level].Size(0); row++)
                {
                    for(int cha = 0; cha < _data[level].Size(2); cha++)
                    {
                        output << _data[level][row][col][cha] << " ";
                    }

                    output << endl;
                }
        }
    }
    
    if(output.rdstate())
    {
        return 0;
    }
    
    // done
    return 1;
}

template<class T> 
void ImagePyramid<T>::Error(const string message)
{
    _message = message;
}

template<class T> 
int ImagePyramid<T>::NewMemory(const Array2D<int> & spec)
{
    _data = Array1D< Array3D<T> >(spec.Size(0));

    if(spec.Size(1) != 3)
    {
        return 0;
    }

    for(int i = 0; i < spec.Size(0); i++)
    {
        _data[i] = Array3D<T>(spec[i][0], spec[i][1], spec[i][2]);
    }

    return 1;
}
    
#endif
