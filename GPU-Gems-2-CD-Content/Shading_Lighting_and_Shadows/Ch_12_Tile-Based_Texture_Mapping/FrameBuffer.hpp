/*
 * FrameBuffer.hpp
 *
 * some commom frame buffer operations
 *
 * Li-Yi Wei
 *
 */

#ifndef _FRAME_BUFFER_HPP
#define _FRAME_BUFFER_HPP

#include <vector>
using namespace std;

class FrameBuffer
{
public:
    struct P3
    {
        int r, g, b;
    };
    
    static int ReadPPM(const char * file_name,
                       vector< vector<P3> > & pixels,
                       int & maximum_value);
    
    static int WritePPM(const vector< vector<P3> > & pixels,
                        const int maximum_value,
                        const char * file_name);

    // write out color buffer to a ppm file
    static int WriteColor(const int width,
                          const int height,
                          const char * file_name);
};

#endif
