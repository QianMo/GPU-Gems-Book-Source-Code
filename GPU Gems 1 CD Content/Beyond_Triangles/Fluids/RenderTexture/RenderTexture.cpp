//------------------------------------------------------------------------------
// File : RenderTexture.cpp
//------------------------------------------------------------------------------
// Copyright 2002 Mark J. Harris and
// The University of North Carolina at Chapel Hill
//------------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software and its 
// documentation for any purpose is hereby granted without fee, provided that 
// the above copyright notice appear in all copies and that both that copyright 
// notice and this permission notice appear in supporting documentation. 
// Binaries may be compiled with this software without any royalties or 
// restrictions. 
//
// The author(s) and The University of North Carolina at Chapel Hill make no 
// representations about the suitability of this software for any purpose. 
// It is provided "as is" without express or implied warranty.
//
// -----------------------------------------------------------------------------
// Credits:
// Original RenderTexture class: Mark J. Harris
// Original Render-to-depth-texture support: Thorsten Scheuermann
// Linux Copy-to-texture: Eric Werness
// Various Bug Fixes: Daniel (Redge) Sperl 
//                    Bill Baxter
//
// -----------------------------------------------------------------------------
/**
 * @file RenderTexture.cpp
 * 
 * Implementation of class RenderTexture.  A multi-format render to 
 * texture wrapper.
 */
#include "RenderTexture.h"
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

//------------------------------------------------------------------------------
// Function     	  : IsPowerOfTwo
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn IsPowerOfTwo(int n)
 * @brief Returns true if /param n is an integer power of 2.
 * 
 * Taken from Steve Baker's Cute Code Collection.
 */ 
bool IsPowerOfTwo(int n)
{
  return ((n&(n-1))==0);
}

//------------------------------------------------------------------------------
// Function      : RenderTexture::RenderTexture
// Description	 : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::RenderTexture()
 * @brief Constructor.
 */ 
RenderTexture::RenderTexture(int iWidth, int iHeight, bool bIsTexture /* = true */,
							 bool bIsDepthTexture /* = false */)
: _iWidth(iWidth), 
  _iHeight(iHeight), 
  _bIsTexture(bIsTexture),
  _bIsDepthTexture(bIsDepthTexture),
  _bHasArbDepthTexture(true),            // [Redge]
  _eUpdateMode(RT_RENDER_TO_TEXTURE),
  _bInitialized(false),
  _bFloat(false),
  _bRectangle(false),
  _bHasDepth(false),
  _bHasStencil(false),
  _bMipmap(false),
  _bAnisoFilter(false),
#ifdef _WIN32
  _hDC(NULL), 
  _hGLContext(NULL), 
  _hPBuffer(NULL),
  _hPreviousDC(0),
  _hPreviousContext(0),
#else
  _pDpy(NULL),
  _hGLContext(NULL),
  _hPBuffer(0),
  _hPreviousContext(0),
  _hPreviousDrawable(0),
#endif
  _iTextureTarget(GL_NONE),
  _iTextureID(0),
  _pPoorDepthTexture(0) // [Redge]
{
  assert(iWidth > 0 && iHeight > 0);
  _iBits[0] = _iBits[1] = _iBits[2] = _iBits[3] = 0;
  _bRectangle = !(IsPowerOfTwo(iWidth) && IsPowerOfTwo(iHeight));
}


//------------------------------------------------------------------------------
// Function     	  : RenderTexture::~RenderTexture
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::~RenderTexture()
 * @brief Destructor.
 */ 
RenderTexture::~RenderTexture()
{
  _Invalidate();
}


//------------------------------------------------------------------------------
// Function     	  : wglGetLastError
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn wglGetLastError()
 * @brief Returns the last windows error generated.
 */ 
#ifdef _WIN32
void RenderTexture::_wglGetLastError()
{
#ifdef _DEBUG

  DWORD err = GetLastError();
  switch(err)
  {
  case ERROR_INVALID_PIXEL_FORMAT:
    fprintf(stderr, "Win32 Error:  ERROR_INVALID_PIXEL_FORMAT\n");
    break;
  case ERROR_NO_SYSTEM_RESOURCES:
    fprintf(stderr, "Win32 Error:  ERROR_NO_SYSTEM_RESOURCES\n");
    break;
  case ERROR_INVALID_DATA:
    fprintf(stderr, "Win32 Error:  ERROR_INVALID_DATA\n");
    break;
  case ERROR_INVALID_WINDOW_HANDLE:
    fprintf(stderr, "Win32 Error:  ERROR_INVALID_WINDOW_HANDLE\n");
    break;
  case ERROR_RESOURCE_TYPE_NOT_FOUND:
    fprintf(stderr, "Win32 Error:  ERROR_RESOURCE_TYPE_NOT_FOUND\n");
    break;
  case ERROR_SUCCESS:
    // no error
    break;
  default:
    LPVOID lpMsgBuf;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | 
                  FORMAT_MESSAGE_FROM_SYSTEM | 
                  FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL,
                  err,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                  (LPTSTR) &lpMsgBuf,
                  0,
                  NULL);
    
    fprintf(stderr, "Win32 Error %d: %s\n", err, lpMsgBuf);
    LocalFree( lpMsgBuf );
    break;
  }
  SetLastError(0);

#endif // _DEBUG
}
#endif

//------------------------------------------------------------------------------
// Function     	  : PrintExtensionError
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn PrintExtensionError( char* strMsg, ... )
 * @brief Prints an error about missing OpenGL extensions.
 */ 
void PrintExtensionError( char* strMsg, ... )
{
  fprintf(stderr, "Error: RenderTexture requires the following unsupported "
                  "OpenGL extensions: \n");
  char strBuffer[512];
  va_list args;
  va_start(args, strMsg);
#ifdef _WIN32
  _vsnprintf( strBuffer, 512, strMsg, args );
#else
  vsnprintf( strBuffer, 512, strMsg, args );
#endif
  va_end(args);
  
  fprintf(stderr, strMsg);

  exit(1);
}


//------------------------------------------------------------------------------
// Function     	  : RenderTexture::Initialize
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::Initialize(bool bShare, bool bDepth, bool bStencil, bool bMipmap, bool bAnisoFilter, unsigned int iRBits, unsigned int iGBits, unsigned int iBBits, unsigned int iABits);
 * @brief Initializes the RenderTexture, sharing display lists and textures if specified.
 * 
 * This function actually does the creation of the p-buffer.  It can only be called 
 * once a GL context has already been created.  Note that if the texture is not
 * power of two dimensioned, or has more than 8 bits per channel, enabling mipmapping
 * or aniso filtering will cause an error.
 */ 
bool RenderTexture::Initialize(bool         bShare       /* = true */, 
                               bool         bDepth       /* = false */, 
                               bool         bStencil     /* = false */, 
                               bool         bMipmap      /* = false */, 
                               bool         bAnisoFilter /* = false */,
                               unsigned int iRBits       /* = 8 */, 
                               unsigned int iGBits       /* = 8 */, 
                               unsigned int iBBits       /* = 8 */, 
                               unsigned int iABits       /* = 8 */,
                               UpdateMode   updateMode   /* = RT_RENDER_TO_TEXTURE */)
{
#ifdef _WIN32
  if (!WGLEW_ARB_pbuffer)
  {
    PrintExtensionError("WGL_ARB_pbuffer");
    return false;
  }
  if (!WGLEW_ARB_pixel_format)
  {
    PrintExtensionError("WGL_ARB_pixel_format");
    return false;
  }
  if (_bIsTexture && !WGLEW_ARB_render_texture)
  {
    PrintExtensionError("WGL_ARB_render_texture");
    return false;
  }
  if (_bIsDepthTexture && !GLEW_ARB_depth_texture)
  {
    // [Redge]
#if defined(_DEBUG) | defined(DEBUG)
    fprintf(stderr, "Warning: OpenGL extension GL_ARB_depth_texture not available.\n"
                    "         Using glReadPixels() to emulate behavior.\n");
#endif   
    _bHasArbDepthTexture = false;
    //PrintExtensionError("GL_ARB_depth_texture");
    //return false;
    // [/Redge]
  }
  SetLastError(0);
#else
  if (!GLXEW_SGIX_pbuffer)
  {
    PrintExtensionError("GLX_SGIX_pbuffer");
    return false;
  }
  if (!GLXEW_SGIX_fbconfig)
  {
    PrintExtensionError("GLX_SGIX_fbconfig");
    return false;
  }
  if (_bIsDepthTexture)
  {
    PrintExtensionError("I don't know");
    return false;
  }
  if (updateMode == RT_RENDER_TO_TEXTURE)
  {
    PrintExtensionError("Some GLX render texture extension");
  }
#endif

  if (_bInitialized)
    _Invalidate();
 
  _bFloat = (iRBits > 8 || iGBits > 8 || iBBits > 8 || iABits > 8);
  
  bool bNVIDIA = true;
#ifdef _WIN32
  if (_bFloat && !GLEW_NV_float_buffer)
  {
    bNVIDIA = false;
    if (!WGLEW_ATI_pixel_format_float)
    {
      PrintExtensionError("GL_NV_float_buffer or GL_ATI_pixel_format_float");
      return false;
    }
  }
  if (_bFloat && _bIsTexture && !bNVIDIA && !GLEW_ATI_texture_float)
  {
	  PrintExtensionError("NV_float_buffer or ATI_texture_float");
  }
#else
  if (_bFloat && _bIsTexture && !GLXEW_NV_float_buffer)
  {
    PrintExtensionError("GLX_NV_float_buffer");
    return false;
  }
#endif  
  if (!_bFloat && !GLEW_NV_texture_rectangle)
  {
    bNVIDIA = false;
  }
  
  _bRectangle = _bRectangle || (_bFloat && bNVIDIA);

  if(_bIsDepthTexture)
    _bHasDepth  = true;    // we need depth for a depth texture...
  else
    _bHasDepth  = bDepth;

  _bHasStencil  = bStencil;
  _bMipmap      = false;   // until it is enabled.
  _bAnisoFilter = false;   // until it is enabled.
  _eUpdateMode  = updateMode;

  GLuint iWGLTextureTarget = 0;
  GLuint iBindTarget = 0;
  GLuint iTextureFormat = 0;
  GLuint iDepthBindTarget = 0;
  GLuint iDepthTextureFormat = 0;

  if (_bIsTexture)
    glGenTextures(1, &_iTextureID);
  if (_bIsDepthTexture)
    glGenTextures(1, &_iDepthTextureID);
  
  // Determine the appropriate texture formats and filtering modes.
  if (_bIsTexture)
  {
    if (_bFloat)
    {      
      if (!bNVIDIA && _bRectangle)
      {
        fprintf(stderr, 
                "RenderTexture Error: ATI textures must be power-of-two-dimensioned.\n");
        return false;
      }

      if (bNVIDIA)
        _iTextureTarget = GL_TEXTURE_RECTANGLE_NV;
      else
        _iTextureTarget = GL_TEXTURE_2D;
    
      glBindTexture(_iTextureTarget, _iTextureID);  
    
      // We'll use clamp to edge as the default texture wrap mode for all tex types
      glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
      glTexParameteri( _iTextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri( _iTextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
      if (bMipmap)
      {
        fprintf(stderr, 
                "RenderTexture Error: float textures do not support mipmaps\n");
        return false;
      }

      if (RT_COPY_TO_TEXTURE == _eUpdateMode)
      {
        GLuint iInternalFormat;
        GLuint iFormat;
        if (iABits > 0)
        {
          if (bNVIDIA)
            iInternalFormat = (iABits > 16) ? GL_FLOAT_RGBA32_NV : GL_FLOAT_RGBA16_NV;
          else
            iInternalFormat = (iABits > 16) ? GL_RGBA_FLOAT32_ATI : GL_RGBA_FLOAT16_ATI;
          iFormat = GL_RGBA;
        }
        else if (iBBits > 0)
        {
          if (bNVIDIA)
            iInternalFormat = (iBBits > 16) ? GL_FLOAT_RGB32_NV : GL_FLOAT_RGB16_NV;
          else
            iInternalFormat = (iBBits > 16) ? GL_RGB_FLOAT32_ATI : GL_RGB_FLOAT16_ATI;
          iFormat = GL_RGB;
        }
        else if (iGBits > 0)
        {
          if (bNVIDIA)
            iInternalFormat = (iGBits > 16) ? GL_FLOAT_RG32_NV : GL_FLOAT_RG16_NV;
          else
            iInternalFormat = (iGBits > 16) ? GL_LUMINANCE_ALPHA_FLOAT32_ATI : 
                                              GL_LUMINANCE_ALPHA_FLOAT16_ATI;
          iFormat = GL_LUMINANCE_ALPHA;
        }
        else 
        {
          if (bNVIDIA)
            iInternalFormat = (iRBits > 16) ? GL_FLOAT_R32_NV : GL_FLOAT_R16_NV;
          else
            iInternalFormat = (iRBits > 16) ? GL_LUMINANCE_FLOAT32_ATI : 
                                              GL_LUMINANCE_FLOAT16_ATI;
          iFormat = GL_LUMINANCE;
        }
        // Allocate the texture image (but pass it no data for now).
        glTexImage2D(_iTextureTarget, 0, iInternalFormat, _iWidth, _iHeight, 
                     0, iFormat, GL_FLOAT, NULL);
      }
      else
      {
#ifdef _WIN32 
        if (bNVIDIA)
          iWGLTextureTarget = WGL_TEXTURE_RECTANGLE_NV;
        else
          iWGLTextureTarget = WGL_TEXTURE_2D_ARB;

        if (iABits > 0) 
        { 
          if (bNVIDIA)
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV;  
            iTextureFormat = WGL_TEXTURE_FLOAT_RGBA_NV; 
          }
          else
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RGBA_ARB;  
            iTextureFormat = WGL_TEXTURE_RGBA_ARB; 
          }
        } 
        else if (iBBits > 0)
        { 
          if (bNVIDIA)
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV;  
            iTextureFormat = WGL_TEXTURE_FLOAT_RGB_NV; 
          }
          else
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RGB_ARB;  
            iTextureFormat = WGL_TEXTURE_RGB_ARB; 
          }
        } 
        else if (iGBits > 0)
        { 
          if (bNVIDIA)
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV;  
            iTextureFormat = WGL_TEXTURE_FLOAT_RG_NV; 
          }
          else
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RGB_ARB;  
            iTextureFormat = WGL_TEXTURE_RGB_ARB; 
          }
        } 
        else 
        { 
          if (bNVIDIA)
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV;  
            iTextureFormat = WGL_TEXTURE_FLOAT_R_NV; 
          }
          else
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RGB_ARB;  
            iTextureFormat = WGL_TEXTURE_RGB_ARB; 
          }
        }
#else if defined(DEBUG) || defined(_DEBUG)
        printf("RenderTexture Error: Render-to-Texture not supported in Linux\n");
#endif    
      }
    }
    else
    {
      if (!_bRectangle)
      {
        _iTextureTarget = GL_TEXTURE_2D;
        glBindTexture(_iTextureTarget, _iTextureID);
            
        // We'll use clamp to edge as the default texture wrap mode for all tex types
        glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
        glTexParameteri( _iTextureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
        if (bMipmap)
        {
          _bMipmap = true;
          glTexParameteri( _iTextureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        
          // Generate mipmap automatically if supported
          if (GLEW_SGIS_generate_mipmap)
          {
            glTexParameteri( _iTextureTarget, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
          }
          else
          {
            PrintExtensionError("GL_SGIS_generate_mipmap");
          }
        }
        else
        {
          glTexParameteri( _iTextureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
      
        // Set anisotropic filter to the max ratio
        if (bAnisoFilter)
        {
          if (GLEW_EXT_texture_filter_anisotropic)
          {
            _bAnisoFilter = true;
            float rMaxAniso;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &rMaxAniso);
            glTexParameterf( _iTextureTarget, GL_TEXTURE_MAX_ANISOTROPY_EXT, rMaxAniso);
          }
        }

        if (RT_COPY_TO_TEXTURE == _eUpdateMode)
        {
          GLuint iInternalFormat;
          GLuint iFormat;
          if (iABits > 0)
          {
            iInternalFormat = GL_RGBA8;
            iFormat = GL_RGBA;
          }
          else 
          {
            iInternalFormat = GL_RGB8;
            iFormat = GL_RGB;
          }
          // Allocate the texture image (but pass it no data for now).
          glTexImage2D(_iTextureTarget, 0, iInternalFormat, _iWidth, _iHeight, 
                       0, iFormat, GL_FLOAT, NULL);
        }
        else
        {
#ifdef _WIN32    
          iWGLTextureTarget = WGL_TEXTURE_2D_ARB; 
          if (iABits > 0) 
          { 
            iBindTarget    = WGL_BIND_TO_TEXTURE_RGBA_ARB;
            iTextureFormat = WGL_TEXTURE_RGBA_ARB;
          } 
          else 
          {
            iBindTarget    = WGL_BIND_TO_TEXTURE_RGB_ARB;  
            iTextureFormat = WGL_TEXTURE_RGB_ARB; 
          } 
#endif
        }
      } 
      else
      {
        if (!bNVIDIA)
        {
          fprintf(stderr, 
                  "RenderTexture Error: ATI textures must be power-of-two-dimensioned.\n");
          return false;
        }
        
        _iTextureTarget = GL_TEXTURE_RECTANGLE_NV;
        
        glBindTexture(_iTextureTarget, _iTextureID);
      
        glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
        glTexParameteri( _iTextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri( _iTextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        
        if (bMipmap) 
        { 

          fprintf(stderr, 
                  "RenderTexture Error: rectangle textures do not support mipmaps\n"); 
          return false; 
        } 

        if (RT_COPY_TO_TEXTURE == _eUpdateMode)
        {
          GLuint iInternalFormat;
          GLuint iFormat;
          if (iABits > 0)
          {
            iInternalFormat = GL_RGBA8;
            iFormat = GL_RGBA;
          }
          else 
          {
            iInternalFormat = GL_RGB8;
            iFormat = GL_RGB;
          }
          // Allocate the texture image (but pass it no data for now).
          glTexImage2D(_iTextureTarget, 0, iInternalFormat, _iWidth, _iHeight, 
                       0, iFormat, GL_FLOAT, NULL);
        }
        else
        {
#ifdef _WIN32
          iWGLTextureTarget = WGL_TEXTURE_RECTANGLE_NV; 
          if (iABits > 0) 
          { 
            iBindTarget    = WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV; 
            iTextureFormat = WGL_TEXTURE_RGBA_ARB; 
          } 
          else 
          { 
            iBindTarget   = WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV; 
            iTextureFormat= WGL_TEXTURE_RGB_ARB; 
          }
#endif 
        }
      } 
    }
  }

  if (_bIsDepthTexture)
  {
    if (!bNVIDIA && _bRectangle)
    {
      fprintf(stderr, 
        "RenderTexture Error: ATI textures must be power-of-two-dimensioned.\n");
      return false;
    }

    if (!_iTextureTarget)
      _iTextureTarget = _bRectangle ? GL_TEXTURE_RECTANGLE_NV : GL_TEXTURE_2D;
    
    glBindTexture(_iTextureTarget, _iDepthTextureID);  
    
    // We'll use clamp to edge as the default texture wrap mode for all tex types
    glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( _iTextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
    glTexParameteri( _iTextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( _iTextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // user will need to set up hardware shadow mapping himself (using ARB_shadow)
        
    if (RT_COPY_TO_TEXTURE == _eUpdateMode)
    {
      // [Redge]
      if (_bHasArbDepthTexture) 
      {
        // Allocate the texture image (but pass it no data for now).
        glTexImage2D(_iTextureTarget, 0, GL_DEPTH_COMPONENT, _iWidth, _iHeight, 
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
      } 
      else 
      {
        // allocate memory for depth texture
        // Since this is slow, we warn the user in debug mode. (above)
        _pPoorDepthTexture = new unsigned short[_iWidth * _iHeight];
        glTexImage2D(_iTextureTarget, 0, GL_LUMINANCE16, _iWidth, _iHeight, 
                   0, GL_LUMINANCE, GL_UNSIGNED_SHORT, _pPoorDepthTexture);
      }
      // [/Redge]
    }
    else  // RENDER_TO_TEXTURE
    {
#ifdef _WIN32
      if(_bRectangle)
      {
        if (!iWGLTextureTarget)  iWGLTextureTarget = WGL_TEXTURE_RECTANGLE_NV; 
        iDepthBindTarget    = WGL_BIND_TO_TEXTURE_RECTANGLE_DEPTH_NV;
        iDepthTextureFormat = WGL_TEXTURE_DEPTH_COMPONENT_NV;
      }
      else
      {
        if (!iWGLTextureTarget)  iWGLTextureTarget = WGL_TEXTURE_2D_ARB; 
        iDepthBindTarget    = WGL_BIND_TO_TEXTURE_DEPTH_NV;
        iDepthTextureFormat = WGL_TEXTURE_DEPTH_COMPONENT_NV;
      }
#endif
    }
  }

#if _WIN32
  // Get the current context.
  HDC hdc = wglGetCurrentDC();
  if (NULL == hdc)
    _wglGetLastError();
  HGLRC hglrc = wglGetCurrentContext();
  if (NULL == hglrc)
    _wglGetLastError();
  
  int iFormat = 0;
  unsigned int iNumFormats;
  int attribChooseList[50]; 
  int attribCreateList[50];
  int attribChoose = 0;
  int attribCreate = 0;
  
  // Setup the attrib list for wglChoosePixelFormat()
  
  attribChooseList[attribChoose++] = WGL_RED_BITS_ARB;
  attribChooseList[attribChoose++] = iRBits;
  attribChooseList[attribChoose++] = WGL_GREEN_BITS_ARB;
  attribChooseList[attribChoose++] = iGBits;
  attribChooseList[attribChoose++] = WGL_BLUE_BITS_ARB;
  attribChooseList[attribChoose++] = iBBits;
  attribChooseList[attribChoose++] = WGL_ALPHA_BITS_ARB;
  attribChooseList[attribChoose++] = iABits;
  if (_bFloat)
  {
    if (bNVIDIA)
    {
      attribChooseList[attribChoose++] = WGL_FLOAT_COMPONENTS_NV;
      attribChooseList[attribChoose++] = GL_TRUE;
    }
    else
    {
      attribChooseList[attribChoose++] = WGL_PIXEL_TYPE_ARB;
      attribChooseList[attribChoose++] = WGL_TYPE_RGBA_FLOAT_ATI;
    }
  }
    
  attribChooseList[attribChoose++] = WGL_STENCIL_BITS_ARB;
  attribChooseList[attribChoose++] = (bStencil) ? 8 : 0;
  attribChooseList[attribChoose++] = WGL_DEPTH_BITS_ARB;
  attribChooseList[attribChoose++] = (bDepth) ? 24 : 0;
  attribChooseList[attribChoose++] = WGL_DRAW_TO_PBUFFER_ARB;
  attribChooseList[attribChoose++] = GL_TRUE;

  if (_bIsTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    attribChooseList[attribChoose++] = iBindTarget;
    attribChooseList[attribChoose++] = GL_TRUE;
  }
  if (_bIsDepthTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    attribChooseList[attribChoose++] = iDepthBindTarget;
    attribChooseList[attribChoose++] = GL_TRUE;
  }

  attribChooseList[attribChoose++] = 0;

  // Setup the attrib list for wglCreatePbuffer()
  if ((_bIsTexture || _bIsDepthTexture) && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    attribCreateList[attribCreate++] = WGL_TEXTURE_TARGET_ARB;
    attribCreateList[attribCreate++] = iWGLTextureTarget;
  }
  if (_bIsTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    attribCreateList[attribCreate++] = WGL_TEXTURE_FORMAT_ARB;
    attribCreateList[attribCreate++] = iTextureFormat;
    /*attribCreateList[attribCreate++] = WGL_TEXTURE_TARGET_ARB;
    attribCreateList[attribCreate++] = iWGLTextureTarget;*/
    attribCreateList[attribCreate++] = WGL_MIPMAP_TEXTURE_ARB;
    attribCreateList[attribCreate++] = (bMipmap) ? GL_TRUE : GL_FALSE;  
  }
  if (_bIsDepthTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    attribCreateList[attribCreate++] = WGL_DEPTH_TEXTURE_FORMAT_NV;
    attribCreateList[attribCreate++] = iDepthTextureFormat;
  }
  attribCreateList[attribCreate++] = WGL_PBUFFER_LARGEST_ARB;
  attribCreateList[attribCreate++] = GL_FALSE;
  attribCreateList[attribCreate++] = 0;

  if (!wglChoosePixelFormatARB( hdc, attribChooseList, NULL, 1, &iFormat, &iNumFormats))
  {
    fprintf(stderr, 
            "RenderTexture::Initialize() creation error: wglChoosePixelFormatARB() failed.\n");
    _wglGetLastError();
    return false;
  }
  if ( iNumFormats <= 0 )
  {
    fprintf(stderr, 
            "RenderTexture::Initialize() creation error: Couldn't find a suitable pixel format.\n");
    _wglGetLastError();
    return false;
  }

  // Create the p-buffer.    
  _hPBuffer = wglCreatePbufferARB( hdc, iFormat, _iWidth, _iHeight, attribCreateList );
  if (!_hPBuffer)
  {
    fprintf(stderr, "RenderTexture::Initialize() pbuffer creation error: wglCreatePbufferARB() failed\n");
    _wglGetLastError();
    return false;
  }
  
   // Get the device context.
  _hDC = wglGetPbufferDCARB( _hPBuffer);
  if ( !_hDC )
  {
    fprintf(stderr, 
            "RenderTexture::Initialize() creation error: wglGetGetPbufferDCARB() failed\n");
    _wglGetLastError();
    return false;
  }
  
  // Create a gl context for the p-buffer.

  
  _hGLContext = wglCreateContext( _hDC );
  if ( !_hGLContext )
  {
    fprintf(stderr, "RenderTexture::Initialize() creation error:  wglCreateContext() failed\n");
    _wglGetLastError();
    return false;
  }
  
  // Share lists, texture objects, and program objects.
  if( bShare )
  {
    if( !wglShareLists(hglrc, _hGLContext) )
    {
      fprintf(stderr, "RenderTexture::Initialize() creation error: wglShareLists() failed\n" );
      _wglGetLastError();
      return false;
    }
  }

  // bind the pbuffer to the render texture object
  if (_bIsTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    glBindTexture(_iTextureTarget, _iTextureID);
    if (wglBindTexImageARB(_hPBuffer, WGL_FRONT_LEFT_ARB) == FALSE)
    {
      _wglGetLastError();
      return false;
    }
  }
  if (_bIsDepthTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    glBindTexture(_iTextureTarget, _iDepthTextureID);
    if (wglBindTexImageARB(_hPBuffer, WGL_DEPTH_COMPONENT_NV) == FALSE)
    {
      _wglGetLastError();
      return false;
    }
  }

   // Determine the actual width and height we were able to create.
  wglQueryPbufferARB( _hPBuffer, WGL_PBUFFER_WIDTH_ARB, &_iWidth );
  wglQueryPbufferARB( _hPBuffer, WGL_PBUFFER_HEIGHT_ARB, &_iHeight );

  _bInitialized = true;
  
  // get the actual number of bits allocated:
  int attrib = WGL_RED_BITS_ARB;
  int value;
  _iBits[0] = (wglGetPixelFormatAttribivARB(_hDC, iFormat, 0, 1, &attrib, &value)) ? value : iRBits;
  attrib = WGL_GREEN_BITS_ARB;
  _iBits[1] = (wglGetPixelFormatAttribivARB(_hDC, iFormat, 0, 1, &attrib, &value)) ? value : iGBits;
  attrib = WGL_BLUE_BITS_ARB;
  _iBits[2] = (wglGetPixelFormatAttribivARB(_hDC, iFormat, 0, 1, &attrib, &value)) ? value : iBBits;
  attrib = WGL_ALPHA_BITS_ARB;
  _iBits[3] = (wglGetPixelFormatAttribivARB(_hDC, iFormat, 0, 1, &attrib, &value)) ? value : iABits; 
  attrib = WGL_DEPTH_BITS_ARB;
  _iBits[4] = (wglGetPixelFormatAttribivARB(_hDC, iFormat, 0, 1, &attrib, &value)) ? value : 0; 
  attrib = WGL_STENCIL_BITS_ARB;
  _iBits[5] = (wglGetPixelFormatAttribivARB(_hDC, iFormat, 0, 1, &attrib, &value)) ? value : 0; 
 
#if defined(_DEBUG) | defined(DEBUG)
  fprintf(stderr, "Created a %dx%d RenderTexture with BPP(%d, %d, %d, %d)",
          _iWidth, _iHeight, _iBits[0], _iBits[1], _iBits[2], _iBits[3]);
  if (_iBits[4]) fprintf(stderr, " depth=%d", _iBits[4]);
  if (_iBits[5]) fprintf(stderr, " stencil=%d", _iBits[5]);
  fprintf(stderr, "\n");
#endif
#else
  _pDpy = glXGetCurrentDisplay();
  GLXContext context = glXGetCurrentContext();
  int screen = DefaultScreen(_pDpy);
  XVisualInfo *visInfo;

  int iFormat = 0;
  int iNumFormats;
  int fbAttribs[50];
  int attrib = 0;

  fbAttribs[attrib++] = GLX_RENDER_TYPE_SGIX;
  fbAttribs[attrib++] = GLX_RGBA_BIT_SGIX;
  fbAttribs[attrib++] = GLX_DRAWABLE_TYPE_SGIX;
  fbAttribs[attrib++] = GLX_PBUFFER_BIT_SGIX;
  fbAttribs[attrib++] = GLX_STENCIL_SIZE;
  fbAttribs[attrib++] = (bStencil) ? 8 : 0;
  fbAttribs[attrib++] = GLX_DEPTH_SIZE;
  fbAttribs[attrib++] = (bDepth) ? 24 : 0;
  if (_bFloat)
  {
    fbAttribs[attrib++] = GLX_RED_SIZE;
    fbAttribs[attrib++] = iRBits;
    fbAttribs[attrib++] = GLX_GREEN_SIZE;
    fbAttribs[attrib++] = iGBits;
    fbAttribs[attrib++] = GLX_BLUE_SIZE;
    fbAttribs[attrib++] = iBBits;
    fbAttribs[attrib++] = GLX_ALPHA_SIZE;
    fbAttribs[attrib++] = iABits;
    fbAttribs[attrib++] = GLX_FLOAT_COMPONENTS_NV;
    fbAttribs[attrib++] = 1;    
  }
  fbAttribs[attrib++] = None;

  GLXFBConfigSGIX *fbConfigs;
  int nConfigs;

  fbConfigs = glXChooseFBConfigSGIX(_pDpy, screen, fbAttribs, &nConfigs);

  if (nConfigs == 0 || !fbConfigs) {
    fprintf(stderr,
	    "RenderTexture::Initialize() creation error: Couldn't find a suitable pixel format\n");
    return false;
  }

  // Pick the first returned format that will return a pbuffer
  for (int i=0;i<nConfigs;i++)
  {
    _hPBuffer = glXCreateGLXPbufferSGIX(_pDpy, fbConfigs[i], _iWidth, _iHeight, NULL);
    if (_hPBuffer) {
      _hGLContext = glXCreateContextWithConfigSGIX(_pDpy, fbConfigs[i], GLX_RGBA_TYPE, bShare?context:NULL, True);
      break;
    }
  }

  if (!_hPBuffer)
  {
    fprintf(stderr, "RenderTexture::Initialize() pbuffer creation error: glXCreateGLXPbufferSGIX() failed\n");
    return false;
  }

  if(!_hGLContext)
  {
      // Try indirect
      _hGLContext = glXCreateContext(_pDpy, visInfo, bShare?context:NULL, False);
      if ( !_hGLContext )
      {
	fprintf(stderr, "RenderTexture::Initialize() creation error:  glXCreateContext() failed\n");
	return false;
      }
  }

  glXQueryGLXPbufferSGIX(_pDpy, _hPBuffer, GLX_WIDTH_SGIX, (GLuint*)&_iWidth);
  glXQueryGLXPbufferSGIX(_pDpy, _hPBuffer, GLX_HEIGHT_SGIX, (GLuint*)&_iHeight);

  _bInitialized = true;

  // XXX Query the color format

#endif
  
  return true;
}

//------------------------------------------------------------------------------
// Function     	  : RenderTexture::_Invalidate
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::_Invalidate()
 * @brief Returns the pbuffer memory to the graphics device.
 * 
 */ 
bool RenderTexture::_Invalidate()
{
  _bFloat       = false;
  _bRectangle   = false;
  _bInitialized = false;
  _bHasDepth    = false;
  _bHasStencil  = false;
  _bMipmap      = false;
  _bAnisoFilter = false;
  _iBits[0] = _iBits[1] = _iBits[2] = _iBits[3] = 0;

  if (_bIsTexture)
    glDeleteTextures(1, &_iTextureID);
  if (_bIsDepthTexture) {
      // [Redge]
      if (!_bHasArbDepthTexture) delete[] _pPoorDepthTexture;
      // [/Redge]
      glDeleteTextures(1, &_iDepthTextureID);
  }

#if _WIN32
  if ( _hPBuffer )
  {
    // Check if we are currently rendering in the pbuffer
    if (wglGetCurrentContext() == _hGLContext)
      wglMakeCurrent(0,0);
    wglDeleteContext( _hGLContext);
    wglReleasePbufferDCARB( _hPBuffer, _hDC);
    wglDestroyPbufferARB( _hPBuffer );
    _hPBuffer = 0;
    return true;
  }
#else
  if ( _hPBuffer )
  {
    if(glXGetCurrentContext() == _hGLContext)
      // XXX I don't know if this is right at all
      glXMakeCurrent(_pDpy, _hPBuffer, 0);
    glXDestroyGLXPbufferSGIX(_pDpy, _hPBuffer);
    _hPBuffer = 0;
    return true;
  }
#endif
  return false;
}


//------------------------------------------------------------------------------
// Function     	  : RenderTexture::Reset
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::Reset(int iWidth, int iHeight, unsigned int iMode, bool bIsTexture, bool bIsDepthTexture)
 * @brief Resets the resolution of the offscreen buffer.
 * 
 * Causes the buffer to delete itself.  User must call Initialize() again
 * before use.
 */ 
bool RenderTexture::Reset(int iWidth, int iHeight, bool bIsTexture /* = true */,
                           bool bIsDepthTexture /* = false */)
{
  if (!_Invalidate())
  {
    fprintf(stderr, "RenderTexture::Reset(): failed to invalidate.\n");
    return false;
  }
  _iWidth     = iWidth;
  _iHeight    = iHeight;
  _bIsTexture = bIsTexture;
  _bIsDepthTexture = bIsDepthTexture;
  
  return true;
}


//------------------------------------------------------------------------------
// Function     	  : RenderTexture::BeginCapture
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::BeginCapture()
 * @brief Activates rendering to the RenderTexture.
 */ 
bool RenderTexture::BeginCapture()
{
  if (!_bInitialized)
  {
    fprintf(stderr, "RenderTexture::BeginCapture(): Texture is not initialized!\n");
    exit(1);
    return false;
  }
#ifdef _WIN32
  // cache the current context so we can reset it when EndCapture() is called.
  _hPreviousDC      = wglGetCurrentDC();
  if (NULL == _hPreviousDC)
    _wglGetLastError();
  _hPreviousContext = wglGetCurrentContext();
  if (NULL == _hPreviousContext)
    _wglGetLastError();

  if (_bIsTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    glBindTexture(_iTextureTarget, _iTextureID);

	  // release the pbuffer from the render texture object
    if (FALSE == wglReleaseTexImageARB(_hPBuffer, WGL_FRONT_LEFT_ARB))
    {
      _wglGetLastError();
      return false;
    }
  }

  if (_bIsDepthTexture && RT_RENDER_TO_TEXTURE == _eUpdateMode)
  {
    glBindTexture(_iTextureTarget, _iDepthTextureID);

	  // release the pbuffer from the render texture object
    if (FALSE == wglReleaseTexImageARB(_hPBuffer, WGL_DEPTH_COMPONENT_NV))
    {
      _wglGetLastError();
      return false;
    }
  }

  // make the pbuffer's rendering context current.
  if (FALSE == wglMakeCurrent( _hDC, _hGLContext))
  {
    _wglGetLastError();
    return false;
  }
#else
  _hPreviousContext = glXGetCurrentContext();
  _hPreviousDrawable = glXGetCurrentDrawable();

  if (False == glXMakeCurrent(_pDpy, _hPBuffer, _hGLContext)) {
    return false;
  }
#endif
  
  return true;
}


//------------------------------------------------------------------------------
// Function     	  : RenderTexture::EndCapture
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::EndCapture()
 * @brief Ends rendering to the RenderTexture.
 */ 
bool RenderTexture::EndCapture()
{
  bool bContextReset = false;

  if (!_bInitialized)
  {
    fprintf(stderr, "RenderTexture::EndCapture(): Texture is not initialized!\n");
    return false;
  }
#ifdef _WIN32  
  if (_bIsTexture)
  {
    if (RT_RENDER_TO_TEXTURE == _eUpdateMode)
    {
      // make the previous rendering context current 
      if (FALSE == wglMakeCurrent( _hPreviousDC, _hPreviousContext))
      {
        _wglGetLastError();
        return false;
      }
      bContextReset = true;
    
      // [Redge] moved binding completely to Bind()
      // [Mark] Can't do that, because it makes things difficult for Cg users.
      // bind the pbuffer to the render texture object      
      glBindTexture(_iTextureTarget, _iTextureID);
      if (FALSE == wglBindTexImageARB(_hPBuffer, WGL_FRONT_LEFT_ARB))
      {
        _wglGetLastError();
        return false;
      }
    }
    else
    {
      glBindTexture(_iTextureTarget, _iTextureID);
      glCopyTexSubImage2D(_iTextureTarget, 0, 0, 0, 0, 0, _iWidth, _iHeight);
    }
  }
  if (_bIsDepthTexture)
  {
    if (RT_RENDER_TO_TEXTURE == _eUpdateMode)
    {
      // make the previous rendering context current 
      if(!bContextReset)
      {
        if (FALSE == wglMakeCurrent( _hPreviousDC, _hPreviousContext))
        {
          _wglGetLastError();
          return false;
        }
        bContextReset = true;
      }

      // [Redge] moved binding completely to Bind()
      // [Mark] Can't do that, because it makes things difficult for Cg users.
      // bind the pbuffer to the render texture object
      glBindTexture(_iTextureTarget, _iDepthTextureID);
      if (FALSE == wglBindTexImageARB(_hPBuffer, WGL_DEPTH_COMPONENT_NV))
      {
        _wglGetLastError();
        return false;
      }
    }
    else
    {
      glBindTexture(_iTextureTarget, _iDepthTextureID);
      // HOW TO COPY DEPTH TEXTURE??? Supposedly this just magically works...
      // [Redge]
      if (_bHasArbDepthTexture) 
      {
        glCopyTexSubImage2D(_iTextureTarget, 0, 0, 0, 0, 0, _iWidth, _iHeight);
      } 
      else 
      {
        // no 'real' depth texture available, so behavior has to be emulated
        // using glReadPixels (beware, this is (naturally) slow ...)
        glReadPixels(0, 0, _iWidth, _iHeight,
                     GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, _pPoorDepthTexture);
        glTexImage2D(_iTextureTarget, 0, GL_LUMINANCE16,
                     _iWidth, _iHeight,
                     0, GL_LUMINANCE, GL_UNSIGNED_SHORT, _pPoorDepthTexture);
      }
      // [/Redge]
    }
  }
  
  if(!bContextReset)
  {
    // make the previous rendering context current 
    if (FALSE == wglMakeCurrent( _hPreviousDC, _hPreviousContext))
    {
      _wglGetLastError();
      return false;
    }
  }
#else
  assert(_bIsTexture);
  glBindTexture(_iTextureTarget, _iTextureID);
  glCopyTexSubImage2D(_iTextureTarget, 0, 0, 0, 0, 0, _iWidth, _iHeight);

  if(!bContextReset)
  {
    if (False == glXMakeCurrent(_pDpy, _hPreviousDrawable, _hPreviousContext))
    {
      return false;
    }
  }
#endif
  return true;
}



//------------------------------------------------------------------------------
// Function     	  : RenderTexture::Bind
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::Bind()
 * @brief Binds RGB texture.
 */ 
void RenderTexture::Bind() const 
{ 
  if (_bInitialized) {
    glBindTexture(_iTextureTarget, _iTextureID);   
  }    
}


//------------------------------------------------------------------------------
// Function     	  : RenderTexture::BindDepth
// Description	    : 
//------------------------------------------------------------------------------
/**
 * @fn RenderTexture::BindDepth()
 * @brief Binds depth texture.
 */ 
void RenderTexture::BindDepth() const 
{ 
  if (_bInitialized && _bIsDepthTexture) {
    glBindTexture(_iTextureTarget, _iDepthTextureID); 
  }
}
