/*
 * FrameBuffer.cpp
 *
 * Li-Yi Wei
 *
 */

#pragma warning (disable: 4786)

#include "FrameBuffer.hpp"

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int FrameBuffer::ReadPPM(const char * file_name,
                         vector< vector<P3> > & pixels,
                         int & maximum_value)
{
    ifstream input(file_name);

    if(!input)
    {
        return 0;
    }
    else
    {
        // check magic number
        string magic;
        
        input >> magic;
        if(magic != "P3") return 0;
        
        // height, width, maximum
        int height, width;
        input >> width >> height >> maximum_value;
        if(input.bad()) return 0;

        // pixels
        pixels = vector< vector<P3> >(height);

        for(int row = height-1; row >= 0; row--)
        {
            pixels[row] = vector<P3>(width);

            for(int col = 0; col < width; col++)
            {
                input >> pixels[row][col].r;
                input >> pixels[row][col].g;
                input >> pixels[row][col].b;
                if(input.bad()) return 0;
            }
        }
    }

    return 1;
}

int FrameBuffer::WritePPM(const vector< vector<P3> > & pixels,
                          const int maximum_value,
                          const char * file_name)
{
    ofstream output(file_name);

    if(!output)
    {
        return 0;
    }
    else
    {
        // magic number
        output << "P3" << endl;

        const int height = pixels.size();
        if(height <= 0)
        {
            return 0;
        }

        const int width = pixels[0].size();
        
        // header
        output << width << " " << height << " " << maximum_value << endl;

        // content
        for(int i = height-1; i >= 0; i--)
            for(int j = 0; j < width; j++)
            {
                output << pixels[i][j].r << " " << pixels[i][j].g << " " <<  pixels[i][j].b << endl;
            }
        
        return 1;
    }
}

int FrameBuffer::WriteColor(const GLsizei width,
                            const GLsizei height,
                            const char * file_name)
{
    // read back
    const GLenum format = GL_RGBA;
    const GLenum type = GL_UNSIGNED_BYTE;
    
    unsigned char *data = new unsigned char[height*width*4];

    glReadPixels(0, 0, width, height, format, type, data);

    // write out
    vector< vector<P3> > pixels(height);
    {
        for(int i = 0; i < height; i++)
        {
            pixels[i] = vector<P3>(width);
        }
    }

    {
        int index = 0;
        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                pixels[i][j].r = data[index++];
                pixels[i][j].g = data[index++];
                pixels[i][j].b = data[index++];
                index++;
            }
        }
    }

    delete[] data;

    // done
    return WritePPM(pixels, 255, file_name);
}

