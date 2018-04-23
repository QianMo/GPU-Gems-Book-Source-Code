/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)

#include "ppm.h"

bool
Ppm::open (const char *filename, int nchannels, int width, int height)
{
    fd = fopen (filename, "wb");
    if (fd == NULL) {
        return false;
    }

    type;
    if (nchannels == 3 || nchannels == 4)
        type = '6';
    else if (nchannels == 1)
        type = '5';
    else
        return false;

    fprintf(fd, "P%c %d %d 255 ", type, width, height);

    this->nchannels = nchannels;
    this->width = width;
    this->height = height;

    current_scanline = -1;
    
    return true;
}



bool
Ppm::write_scanline (int y, float *data)
{
    if (fd == NULL)
        return false;
    if (width < 1 || height < 1 || nchannels < 1 || nchannels > 4)
        return false;
    if (y != current_scanline + 1)
        return false;
    if (y >= height)
        return false;
    
    current_scanline = y;
    
    switch (type) {
    case '5':
        for (int x = 0; x < width; x++, data += nchannels) 
            fputc((int)(*data++ * 255), fd);
        break;

    case '6':
        for (int x = 0; x < width; x++) {
            float r, g, b;
            r = *data++;
            g = *data++;
            b = *data++;
            if (nchannels == 4)     // skip alpha
                data++;
            if (r < 0) r = 0;
            if (r > 1) r = 1;
            if (g < 0) g = 0;
            if (g > 1) g = 1;
            if (b < 0) b = 0;
            if (b > 1) b = 1;
            fputc((int)(r*255), fd);
            fputc((int)(g*255), fd);
            fputc((int)(b*255), fd);
        }
    }

    return true;
}



bool
Ppm::close ()
{
    if (fd == NULL) {
        std::cerr << "Ppm::close cannot close invalid file\n";
        return false;
    }
    if (current_scanline != height - 1) {
        std::cerr << "Ppm::close current scanline is " << current_scanline;
        std::cerr << " and full height is " << height << "\n";
        return false;
    }
    fclose (fd);
    fd = NULL;
    return true;
}
