/*
 * PBuffer.hpp
 *
 * p-buffer for render to texture
 *
 * Li-Yi Wei
 * 3/15/2003
 *
 */

#ifndef _P_BUFFER_HPP    
#define _P_BUFFER_HPP

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>

#include <GL/glext.h>
#include <GL/wglext.h>

#include "Exception.hpp"

class PBuffer
{
public:
    enum Options {NONE=0, RENDER_TO_TEXTURE=1, FLOAT_COMPONENTS=2};

    PBuffer(const int height, const int width,
            const Options options,
            const unsigned long mode) throw(Exception);
    ~PBuffer(void);

    // put rendering to p-buffer stuff between Begin() and End() 
    string Begin(void) const;
    string End(void) const;

protected:
    string Init(const int height, const int width);
    string LastError(void) const;
    
protected:
    HPBUFFERARB  _hpbuffer;      // Handle to the pbuffer
    HDC          _hdc;           // Handle to the device context
    HGLRC        _hglrc;         // Handle to the GL rendering context
    int          _height;        // Height of the pbuffer
    int          _width;         // Width of the pbuffer
    const unsigned long _mode;
    
    const int _r2t; // enable render to texture
    const int _float; // float compnents 
    
    mutable HDC    _hdc0;    // Handle to the original device context
    mutable HGLRC  _hglrc0;  // Handle to the original GL rendering context
    mutable int _shared;
    mutable int _binded;
};

#endif
