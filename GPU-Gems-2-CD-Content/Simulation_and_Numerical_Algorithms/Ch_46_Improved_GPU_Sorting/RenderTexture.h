//------------------------------------------------------------------------------
// File : RenderTexture.h
//------------------------------------------------------------------------------
// Copyright (c) 2002-2004 Mark J. Harris
//---------------------------------------------------------------------------
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any
// damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any
// purpose, including commercial applications, and to alter it and
// redistribute it freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you
//    must not claim that you wrote the original software. If you use
//    this software in a product, an acknowledgment in the product
//    documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and
//    must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
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
#ifndef __RENDERTEXTURE2_HPP__
#define __RENDERTEXTURE2_HPP__

#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <GL/glxew.h>
#endif

#include <string>
#include <vector>

/* The pixel format for the pbuffer is controlled by the mode string passed
* into the PBuffer constructor. This string can have the following attributes:
*
* To specify the pixel format, use the following syntax.
*   <channels>=<bits>
* <channels> must match one of the following.
*
* r			   - r pixel format (for float buffer).
* rg		   - rg pixel format (for float buffer).
* rgb          - rgb pixel format. 8 bit or 16/32 bit in float buffer mode
* rgba         - same as "rgb alpha" string
*
* <bits> can either be a scalar--meaning the same bit depth for each 
* channel-- or a 2-, 3-, 4-component vector matching the specified number of 
* channels. Vector components should be comma separated. An optional 'f' 
* suffix on the bit depth specifies float components.  In this case <bits> 
* must be either "32f" or "16f".  If <bits> is empty, the default 8 bits per
* channel will be used.
*   r=32f
*   rg=16f
*   rgb=8
*   rgb=5,6,5
*
* The following other attributes are supported.
*
* depth=n      - must have n-bit depth buffer, omit n for default (24 bits)
* stencil=n    - must have n-bit stencil buffer, omit n for default (8 bits)
* samples=n    - must support n-sample antialiasing (n can be 2 or 4)
* aux=n        - must have n AUX buffers
* doublebuffer - must support double buffered rendering
* 
* tex2D
* texRECT
* texCUBE  - must support binding pbuffer as texture to specified target
*          - binding the depth buffer is also supported by specifying
* depthTex2D
* depthTexRECT
* depthTexCUBE
*          - Both depth and color texture binding, may be specified, but
*            the targets must match!
*            For example: "tex2D depthTex2D" or "texRECT depthTexRECT"
*
* rtt
* ctt      - These mutually exclusive options specify the update method used
*            for render textures that support texture binding. "rtt"
*            indicates that render to texture will be used to update the 
*            texture. "ctt" indicates that copy to texture will be used 
*            (i.e. glCopyTexSubImage2D()). "rtt" is the default if neither is 
*            specified, and one of the "tex*" options above is. 
* 
*
*---------------------------------------------------------------------------
*
* USAGE NOTES:
*
* * Texture Parameters:
*   The default texture wrap mode is GL_CLAMP_TO_EDGE for all textures, and
*   the default texture filtering modes (min and mag) are GL_NEAREST. 
*   To change these parameters, simply bind the RenderTexture (using the
*   Bind() method), and set them the way you would for any GL texture object.
*   The same goes for depth textures.
*
* * Enabling Mipmapping:
*   This is similar to the texture parameters above.  When "rtt" is specified
*   in the mode string, "mipmap" must also be specified in order to enable
*   a mipmapped pbuffer.  Then, the mipmaps must be created by enabling the
*   GL_SGIS_GENERATE_MIPMAP texture parameter in the same way as above, and
*   the min filter mode must be set to a mipmap filter mode, as with any
*   mipmapped texture object.
*
* * Enabling Anisotropic Filtering  
*   As with the texture parameters above, except as in the following code:
*   glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, max);
*   glTexParameterf(target, GL_TEXTURE_MAX_ANISOTROPY_EXT, <value < max>);
*/
class RenderTexture
{
public: // enums
    enum UpdateMode
    {
        RT_RENDER_TO_TEXTURE,
        RT_COPY_TO_TEXTURE
    };
    
public: // interface
    // ctor / dtor
    RenderTexture(const char *strMode="rgb tex2D");
    ~RenderTexture();
    
    //! Call this once before use.  Set bShare to true to share lists, textures, 
    //! and program objects between the render texture context and the 
    //! current active GL context.
    bool Initialize(int width, int height, 
                    bool shareObjects=true, 
                    bool copyContext=false);

    // !Change the render texture format.
    bool Reset(const char* strMode,...);
    // !Change the size of the render texture.
    bool Resize(int width, int height);
    
    // !Begin drawing to the texture. (i.e. use as "output" texture)
    bool BeginCapture(bool releaseBoundBuffers = true);
    // !Ends drawing to 'current', begins drawing to this RenderTexture
    bool BeginCapture(RenderTexture* current, bool releaseBoundBuffers = true);
    // !End drawing to the texture.
    bool EndCapture();
    
    // !Bind the texture to the active texture unit for use as an "input" texture
    void Bind() const;

    // !Bind the depth texture to the active texture unit for use as an "input" texture
    void BindDepth() const; 

    // !Associate the RTT texture with 'iBuffer' (default is WGL_FRONT_LEFT_ARB) 
    bool BindBuffer( int iBuffer );

    //! Enables the texture target appropriate for this render texture.
    void EnableTextureTarget() const 
    { if (_bInitialized) glEnable(_iTextureTarget); }
    //! Disables the texture target appropriate for this render texture.
    void DisableTextureTarget() const 
    { if (_bInitialized) glDisable(_iTextureTarget); }
    
    //! Returns the texture ID.  Useful in Cg applications.
    unsigned int GetTextureID() const  { return _iTextureID; }
    //! Returns the depth texture ID.  Useful in Cg applications.
    unsigned int GetDepthTextureID() const { return _iDepthTextureID; }
    //! Returns the texture target this texture is bound to.
    unsigned int GetTextureTarget() const { return _iTextureTarget; }
    //! Conversion operator allows RenderTexture to be passed to GL calls
    operator unsigned int()const{return _iTextureID;}     
    
    //! Returns the width of the offscreen buffer.
    int GetWidth() const            { return _iWidth;  } 
    //! Returns the width of the offscreen buffer.
    int GetHeight() const           { return _iHeight; }
    //! Returns the maximum S texture coordinate.
    int GetMaxS() const      { return IsRectangleTexture() ? _iWidth : 1; }                  
    //! Returns the maximum T texture coordinate.
    int GetMaxT() const      { return IsRectangleTexture() ? _iHeight : 1; }                  
    
    //! Returns the number of red bits allocated.
    int GetRedBits() const          { return _iNumColorBits[0]; }
    //! Returns the number of green bits allocated.
    int GetGreenBits() const        { return _iNumColorBits[1]; }
    //! Returns the number of blue bits allocated.
    int GetBlueBits() const         { return _iNumColorBits[2]; }
    //! Returns the number of alpha bits allocated.
    int GetAlphaBits() const        { return _iNumColorBits[3]; }

    //! Returns the number of depth bits allocated.
    int GetDepthBits() const        { return _iNumDepthBits; }
    //! Returns the number of stencil bits allocated.
    int GetStencilBits() const      { return _iNumStencilBits; }
    
    //! True if this RenderTexture has been properly initialized.
    bool IsInitialized() const      { return _bInitialized; }
    //! True if this is a texture and not just an offscreen buffer.
    bool IsTexture() const          { return _bIsTexture; }
    //! True if this is a depth texture and not just an offscreen buffer.
    bool IsDepthTexture() const     { return _bIsDepthTexture; }
    //! True if this is a floating point buffer / texture.
    bool IsFloatTexture() const     { return _bFloat; }
    //! True if this is a double-buffered pbuffer
    bool IsDoubleBuffered() const   { return _bDoubleBuffered; }
    //! True if this texture has non-power-of-two dimensions.
    bool IsRectangleTexture() const { return _bRectangle; }
    //! True if this texture has non-power-of-two dimensions.
    //! True if this pbuffer has a depth buffer.
    bool HasDepth() const           { return (_iNumDepthBits > 0); }
    //! True if this pbuffer has a stencil buffer.
    bool HasStencil() const         { return (_iNumStencilBits > 0); }
    //! True if this texture has mipmaps.
    bool IsMipmapped() const        { return _bMipmap; }

    /**
    * @fn IsPowerOfTwo(int n)
    * @brief Returns true if /param n is an integer power of 2.
    * 
    * Taken from Steve Baker's Cute Code Collection. 
    * http://www.sjbaker.org/steve/software/cute_code.html
    */ 
    static bool IsPowerOfTwo(int n) { return ((n&(n-1))==0); }


    /////////////////////////////////////////////////////////////////////////
    // This is the deprecated (old) interface.  It will likely be removed
    // in a future version, so it is recommended that you transition to the 
    // new mode-string-based interface.
    RenderTexture(int width, int height,
                   bool bIsTexture = true,
                   bool bIsDepthTexture = false);
    //
    // Call this once before use.  Set bShare to true to share lists, 
    // textures, and program objects between the render texture context 
    // and the current active GL context. [deprecated]
    bool Initialize(bool bShare              = true, 
                    bool bDepth              = false, 
                    bool bStencil            = false,
                    bool bMipmap             = false, 
                    bool bAnisoFilter        = false,
                    unsigned int iRBits      = 8,
                    unsigned int iGBits      = 8,
                    unsigned int iBBits      = 8,
                    unsigned int iABits      = 8,
// Only Win32 has RT now, so only make it default there
#ifdef _WIN32
                    UpdateMode   updateMode = RT_RENDER_TO_TEXTURE
#else
                    UpdateMode   updateMode = RT_COPY_TO_TEXTURE
#endif
                    );
    // !Change the render texture resolution. [deprecated]
    bool Reset(int iWidth, int iHeight);
    //
    /////////////////////////////////////////////////////////////////////////

    void _MaybeCopyBuffer();

protected: // methods
    bool         _Invalidate();

    typedef std::pair<std::string, std::string> KeyVal;

    void _ParseModeString(const char *modeString, 
                          std::vector<int> &pixelFormatAttribs, 
                          std::vector<int> &pbufferAttribs);

    std::vector<int> _ParseBitVector(std::string bitVector);
    KeyVal _GetKeyValuePair(std::string token);


    bool _VerifyExtensions();
    bool _InitializeTextures();
    
    bool _ReleaseBoundBuffers();
    bool _MakeCurrent();
    bool _BindDepthBuffer( ) const;

protected: // data
    int          _iWidth;     // width of the pbuffer
    int          _iHeight;    // height of the pbuffer
    
    bool         _bIsTexture;
    bool         _bIsDepthTexture;
    bool         _bHasARBDepthTexture; // [Redge]
    
    UpdateMode   _eUpdateMode;
        
    bool         _bInitialized;
    
    unsigned int _iNumAuxBuffers;
    bool         _bIsBufferBound;
    int          _iCurrentBoundBuffer;
    
    unsigned int _iNumComponents;
    unsigned int _iNumColorBits[4];
    unsigned int _iNumDepthBits;
    unsigned int _iNumStencilBits;

    
    bool         _bFloat;
    bool         _bDoubleBuffered;
    bool         _bPowerOf2;
    bool         _bRectangle;
    bool         _bMipmap;
    
    bool         _bShareObjects;
    bool         _bCopyContext;
    
#ifdef _WIN32
    HDC          _hDC;        // Handle to a device context.
    HGLRC        _hGLContext; // Handle to a GL context.
    HPBUFFERARB  _hPBuffer;   // Handle to a pbuffer.
    
    HDC          _hPreviousDC;
    HGLRC        _hPreviousContext;
#else
    Display     *_pDisplay;
    GLXContext   _hGLContext;
    GLXPbuffer   _hPBuffer;
    
    GLXDrawable  _hPreviousDrawable;
    GLXContext   _hPreviousContext;
#endif
    
    // Texture stuff
    GLenum       _iTextureTarget;
    unsigned int _iTextureID;
    unsigned int _iDepthTextureID;
    
    unsigned short* _pPoorDepthTexture; // [Redge]

    std::vector<int> _pixelFormatAttribs;
    std::vector<int> _pbufferAttribs;

private:
    // Using these could lead to some odd behavior
    RenderTexture(const RenderTexture&);
    RenderTexture& operator=(const RenderTexture&);
};

#endif //__RENDERTEXTURE2_HPP__
