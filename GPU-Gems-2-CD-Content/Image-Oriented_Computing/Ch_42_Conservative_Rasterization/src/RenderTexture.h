//------------------------------------------------------------------------------
// File : rendertexture.h
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
// -----------------------------------------------------------------------------
// Credits:
// Original RenderTexture code: Mark J. Harris
// Original Render-to-depth-texture support: Thorsten Scheuermann
// Linux Copy-to-texture: Eric Werness
// Various Bug Fixes: Daniel (Redge) Sperl 
//                    Bill Baxter
//
// -----------------------------------------------------------------------------
/**
 * @file RenderTexture.h
 * 
 * Interface definition for class RenderTexture.  A multi-format render to 
 * texture wrapper.
 */
#ifndef __RENDERTEXTURE_HPP__
#define __RENDERTEXTURE_HPP__

#include "glew/glew.h"
#ifdef _WIN32
#include "glew/wglew.h"
#else
#include "glew/glxew.h"
#endif

class RenderTexture
{
public: // enums
  enum UpdateMode
  {
    RT_RENDER_TO_TEXTURE,
    RT_COPY_TO_TEXTURE
  };
  
public: // interface
  RenderTexture(int iWidth, int iHeight, 
                bool bIsTexture = true,
                bool bIsDepthTexture = false);
  ~RenderTexture();

  //! Call this once before use.  Set bShare to true to share lists, textures, 
  //! and program objects between the render texture context and the 
  //! current active GL context.
  bool Initialize(bool bShare             = true, 
                  bool bDepth             = false, 
                  bool bStencil           = false,
                  bool bMipmap            = false, 
                  bool bAnisoFilter       = false,
                  unsigned int iRBits     = 8,
                  unsigned int iGBits     = 8,
                  unsigned int iBBits     = 8,
                  unsigned int iABits     = 8,
		  // Only Win32 has RT now, so only make it default there
#ifdef _WIN32
                  UpdateMode   updateMode = RT_RENDER_TO_TEXTURE
#else
		              UpdateMode   updateMode = RT_COPY_TO_TEXTURE
#endif
);

  // !Change the render texture resolution.
  bool Reset(int iWidth, int iHeight, bool bIsTexture = true,
             bool bIsDepthTexture = false);

  // !Begin drawing to the texture. (i.e. use as "output" texture)
  bool BeginCapture();
  // !End drawing to the texture.
  bool EndCapture();

  // [Redge] 'const's added
  // !Bind the texture to the active texture unit for use as an "input" texture
  void Bind() const;   // [Redge] moved definition to cpp-file (since it's longer now)
  // !Bind the depth texture to the active texture unit for use as an "input" texture
  void BindDepth() const;  // [Redge] moved definition to cpp-file

  //! Enables the texture target appropriate for this render texture.
  void EnableTextureTarget() const { if (_bInitialized) glEnable(_iTextureTarget); }
  //! Disables the texture target appropriate for this render texture.
  void DisableTextureTarget() const { if (_bInitialized) glDisable(_iTextureTarget); }
  // [/Redge]

  //! Returns the texture ID.  Useful in Cg applications.
  unsigned int GetTextureID() const { return _iTextureID; }
  //! Returns the depth texture ID.  Useful in Cg applications.
  unsigned int GetDepthTextureID() const { return _iDepthTextureID; }
  //! Returns the texture target this texture is bound to.
  unsigned int GetTextureTarget() const { return _iTextureTarget; }

  //! Returns the width of the offscreen buffer.
  int GetWidth() const   { return _iWidth;  } 
  //! Returns the width of the offscreen buffer.
  int GetHeight() const  { return _iHeight; }     

  //! Returns the number of red bits allocated.
  int GetRedBits() const   { return _iBits[0]; }
  //! Returns the number of green bits allocated.
  int GetGreenBits() const { return _iBits[1]; }
  //! Returns the number of blue bits allocated.
  int GetBlueBits() const  { return _iBits[2]; }
  //! Returns the number of alpha bits allocated.
  int GetAlphaBits() const { return _iBits[3]; }
  //! Returns the number of depth bits allocated.
  int GetDepthBits() const { return _iBits[4]; }
  //! Returns the number of stencil bits allocated.
  int GetStencilBits() const { return _iBits[5]; }

  //! True if this RenderTexture has been properly initialized.
  bool IsInitialized() const      { return _bInitialized; }
  //! True if this is a texture and not just an offscreen buffer.
  bool IsTexture() const          { return _bIsTexture; }
  //! True if this is a depth texture and not just an offscreen buffer.
  bool IsDepthTexture() const     { return _bIsDepthTexture; }
  //! True if this is a floating point buffer / texture.
  bool IsFloatTexture() const     { return _bFloat; }
  //! True if this texture has non-power-of-two dimensions.
  bool IsRectangleTexture() const { return _bRectangle; }
  //! True if this pbuffer has a depth buffer.
  bool HasDepth() const           { return _bHasDepth; }
  //! True if this pbuffer has a stencil buffer.
  bool HasStencil() const         { return _bHasStencil; }
  //! True if this texture has mipmaps.
  bool IsMipmapped() const        { return _bMipmap; }
  //! True if this texture is anisotropically filtered.
  bool HasAnisoFilter() const     { return _bAnisoFilter; }

protected: // methods
  bool         _Invalidate();
#ifdef _WIN32
  void         _wglGetLastError();
#endif

protected: // data
  int          _iWidth;     // width of the pbuffer
  int          _iHeight;    // height of the pbuffer

  bool         _bIsTexture;
  bool         _bIsDepthTexture;
  bool         _bHasArbDepthTexture; // [Redge]

  UpdateMode   _eUpdateMode;
  
  bool         _bInitialized;

  unsigned int _iBits[6];
  bool         _bFloat;
  bool         _bRectangle;
  bool         _bHasDepth;
  bool         _bHasStencil;
  bool         _bMipmap;
  bool         _bAnisoFilter;
  
#ifdef _WIN32
  HDC          _hDC;        // Handle to a device context.
  HGLRC        _hGLContext; // Handle to a GL context.
  HPBUFFERARB  _hPBuffer;   // Handle to a pbuffer.

  HDC          _hPreviousDC;
  HGLRC        _hPreviousContext;
#else
  Display      *_pDpy;
  GLXContext   _hGLContext;
  GLXPbuffer   _hPBuffer;

  GLXDrawable  _hPreviousDrawable;
  GLXContext   _hPreviousContext;
#endif
  
  GLenum       _iTextureTarget;
  unsigned int _iTextureID;
  unsigned int _iDepthTextureID;

  unsigned short* _pPoorDepthTexture; // [Redge]
};

#endif //__RENDERTEXTURE_HPP__
