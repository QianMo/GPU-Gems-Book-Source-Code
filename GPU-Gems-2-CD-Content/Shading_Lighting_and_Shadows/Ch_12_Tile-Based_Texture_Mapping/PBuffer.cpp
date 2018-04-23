/*
 * PBuffer.cpp
 *
 * Li-Yi Wei
 * 3/15/2003
 *
 */

#include <GL/glut.h>
#include "PBuffer.hpp"

// Convenience stuff for pbuffer object.
#define GLH_EXT_SINGLE_FILE
#define REQUIRED_EXTENSIONS "WGL_ARB_pbuffer " \
                            "WGL_ARB_pixel_format " \
                            "WGL_ARB_render_texture " \
                            "GL_NV_float_buffer " 

#include <glh/glh_extensions.h>
#include <iostream>

PBuffer::PBuffer(const int height, const int width,
                 const Options options,
                 const unsigned long mode) throw(Exception) : _mode(mode), _r2t(options & PBuffer::RENDER_TO_TEXTURE), _float(options & PBuffer::FLOAT_COMPONENTS)
{
    string result = Init(height, width);
    
    if(result != "")
    {
        throw Exception(result);
    }
}

PBuffer::~PBuffer(void)
{
    if(_hpbuffer)
    {
        // Check if we are currently rendering in the pbuffer
        if(wglGetCurrentContext() == _hglrc)
            wglMakeCurrent(0, 0);
        
        // delete the pbuffer context
        wglDeleteContext(_hglrc);
        wglReleasePbufferDCARB(_hpbuffer, _hdc);
        wglDestroyPbufferARB(_hpbuffer);
        _hpbuffer = 0;
    }
}

string PBuffer::Init(const int height, const int width)
{
    string result = "";
    
    // set the entry points for the extension.
    if(!glh_init_extensions(REQUIRED_EXTENSIONS))
    {
        result = "pbuffer creation error: Necessary extensions were not supported";
        return result;
    }

    _hpbuffer = 0;
    _hdc = 0;
    _hglrc = 0;
    _height = 0;
    _width = 0;
    _hdc0 = 0;
    _hglrc0 = 0;
    
    _shared = 1;
    _binded = 0;
    
    _hdc0 = wglGetCurrentDC();
    _hglrc0 = wglGetCurrentContext();

    result = LastError();
    if(result != "") return result;

    // Query for a suitable pixel format based on the specified mode.
    const int MAX_ATTRIBS = 256;
    const int MAX_PFORMATS = 256;
    int     format;
    int     pformat[MAX_PFORMATS];
    unsigned int nformats;
    int     iattributes[2*MAX_ATTRIBS];
    float   fattributes[2*MAX_ATTRIBS];
    int     nfattribs = 0;
    int     niattribs = 0;

    // Attribute arrays must be "0" terminated - for simplicity, first
    // just zero-out the array entire, then fill from left to right.
    memset(iattributes, 0, sizeof(int)*2*MAX_ATTRIBS);
    memset(fattributes, 0, sizeof(float)*2*MAX_ATTRIBS);
    
    // Since we are trying to create a pbuffer, the pixel format we
    // request (and subsequently use) must be "p-buffer capable".
    iattributes[niattribs++] = WGL_DRAW_TO_PBUFFER_ARB;
    iattributes[niattribs++] = GL_TRUE;

    if(_r2t)
    {
        // we are asking for a pbuffer that is meant to be bound
        // as an RGBA texture - therefore we need a color plane
        iattributes[niattribs++] = WGL_BIND_TO_TEXTURE_RGBA_ARB;
        iattributes[niattribs++] = GL_TRUE;
    }

    if(_float)
    {
        iattributes[niattribs++] = WGL_RED_BITS_ARB;
        iattributes[niattribs++] = 32;
        iattributes[niattribs++] = WGL_GREEN_BITS_ARB;
        iattributes[niattribs++] = 32;
        iattributes[niattribs++] = WGL_BLUE_BITS_ARB;
        iattributes[niattribs++] = 32;
        iattributes[niattribs++] = WGL_ALPHA_BITS_ARB;
        iattributes[niattribs++] = 32;
        iattributes[niattribs++] = WGL_FLOAT_COMPONENTS_NV;
        iattributes[niattribs++] = GL_TRUE;
    }
    
    {
        if(_mode & GLUT_INDEX)
        {
            iattributes[niattribs++] = WGL_PIXEL_TYPE_ARB;
            iattributes[niattribs++] = WGL_TYPE_COLORINDEX_ARB;  // Yikes!
        }
        else
        {
            iattributes[niattribs++] = WGL_PIXEL_TYPE_ARB;
            iattributes[niattribs++] = WGL_TYPE_RGBA_ARB;
        }

        if(_mode & GLUT_DOUBLE)
        {
            iattributes[niattribs++] = WGL_DOUBLE_BUFFER_ARB;
            iattributes[niattribs++] = GL_TRUE;
        }

        if(_mode & GLUT_DEPTH)
        {
            iattributes[niattribs++] = WGL_DEPTH_BITS_ARB;
            iattributes[niattribs++] = 24;
        }
        if(_mode & GLUT_STENCIL)
        {
            iattributes[niattribs++] = WGL_STENCIL_BITS_ARB;
            iattributes[niattribs++] = 8;
        }

        if(_mode & GLUT_ACCUM)
        {
            iattributes[niattribs++] = WGL_ACCUM_BITS_ARB;
            iattributes[niattribs++] = 1;
        }
    }
    
    if ( !wglChoosePixelFormatARB(_hdc0,
                                  iattributes,
                                  fattributes,
                                  MAX_PFORMATS,
                                  pformat,
                                  &nformats ))
    {
        return "pbuffer creation error: wglChoosePixelFormatARB() failed";
    }
    
    result = LastError();
    
    if(result != "") return result;
    
    if(nformats <= 0)
    {
        return "pbuffer creation error:  Couldn't find a suitable pixel format";
    }
    
    format = pformat[0];

    // Set up the pbuffer attributes
    memset(iattributes, 0, sizeof(int)*2*MAX_ATTRIBS);
    niattribs = 0;
    
    if(_r2t)
    {
        // the render texture format is RGBA
        iattributes[niattribs++] = WGL_TEXTURE_FORMAT_ARB;
        iattributes[niattribs++] = WGL_TEXTURE_RGBA_ARB;

        // the render texture target is GL_TEXTURE_2D
        iattributes[niattribs++] = WGL_TEXTURE_TARGET_ARB;
        iattributes[niattribs++] = WGL_TEXTURE_2D_ARB;

        // ask to allocate room for the mipmaps
        iattributes[niattribs++] = WGL_MIPMAP_TEXTURE_ARB;
        iattributes[niattribs++] = TRUE;

        // ask to allocate the largest pbuffer it can, if it is
        // unable to allocate for the width and height
        iattributes[niattribs++] = WGL_PBUFFER_LARGEST_ARB;
        iattributes[niattribs++] = FALSE;
    }
    
    // Create the p-buffer.
    _hpbuffer = wglCreatePbufferARB(_hdc0, format, width, height, iattributes);
      
    if(_hpbuffer == 0)
    {
        return "pbuffer creation error:  wglCreatePbufferARB() failed";
    }
    
    result = LastError();
    if(result != "") return result;

    // Get the device context.
    _hdc = wglGetPbufferDCARB(_hpbuffer);
    
    if(_hdc == 0)
    {
        return "pbuffer creation error:  wglGetPbufferDCARB() failed";
    }
    
    result = LastError();
    if(result != "") return result;

    // Create a gl context for the p-buffer.
    _hglrc = wglCreateContext(_hdc);
    if(_hglrc == 0)
    {
        return "pbuffer creation error:  wglCreateContext() failed";
    }
    
    result = LastError();
    if(result != "") return result;

    // Determine the actual width and height we were able to create.
    wglQueryPbufferARB(_hpbuffer, WGL_PBUFFER_WIDTH_ARB, &_width);
    wglQueryPbufferARB(_hpbuffer, WGL_PBUFFER_HEIGHT_ARB, &_height);

    // share
    if(_shared)
    {
        if(!wglShareLists(_hglrc0, _hglrc))
        {
            return "pbuffer creation error:  wglShareLists() failed";
        }
    }
    
    // done
    return result;
}

string PBuffer::LastError(void) const
{
    DWORD err = GetLastError();
    SetLastError(0);
    string result = "";
    
    switch(err)
    {
    case ERROR_INVALID_PIXEL_FORMAT:
        result = "Win32 Error:  ERROR_INVALID_PIXEL_FORMAT";
        break;
    case ERROR_NO_SYSTEM_RESOURCES:
        result = "Win32 Error:  ERROR_NO_SYSTEM_RESOURCES";
        break;
    case ERROR_INVALID_DATA:
        result = "Win32 Error:  ERROR_INVALID_DATA";
        break;
    case ERROR_INVALID_WINDOW_HANDLE:
        result = "Win32 Error:  ERROR_INVALID_WINDOW_HANDLE";
        break;
    case ERROR_RESOURCE_TYPE_NOT_FOUND:
        result = "Win32 Error:  ERROR_RESOURCE_TYPE_NOT_FOUND";
        break;
    case ERROR_SUCCESS:
        // no error
        break;
    default:
        result = "Win32 Error:  unknown error";
        break;
    }

    return result;
}
    
string PBuffer::Begin(void) const
{
    string result = "";
    
    // release the pbuffer from the render texture object
    if(_binded)
    {
        if(wglReleaseTexImageARB(_hpbuffer, WGL_FRONT_LEFT_ARB) == FALSE)
        {
            result = LastError();
        }
        else
        {
            _binded = 0;
        }
    }
    
    // get the GLUT window HDC and HGLRC
    _hdc0 = wglGetCurrentDC();
    _hglrc0 = wglGetCurrentContext();

    if(wglMakeCurrent(_hdc, _hglrc) == FALSE)
    {
        result = LastError();
    }

    return result;
}

string PBuffer::End(void) const
{
    string result = "";
    
    if(wglMakeCurrent(_hdc0, _hglrc0) == FALSE)
    {
        result = LastError();
    }

    if(_r2t && !_binded)
    {
        if(wglBindTexImageARB(_hpbuffer, WGL_FRONT_LEFT_ARB) == FALSE)
        {
            result = LastError();
        }
        else
        {
            _binded = 1;
        }
    }
    
    return result;
}
    
