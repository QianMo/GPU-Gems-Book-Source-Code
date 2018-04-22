// ------------------------------------------------------------------
//
// Description:
//      Buffer represents the image data.
//          It is implemented as a proxy for an OpenGL pixel 
//      buffer but specialized to be used as an image:
//          - the pixel type is RGBA
//          - each color channel is represented by a 16bit
//            float (datatype half).
//          - the buffer may have non-power-of-two size and 
//            thus can only be bound as an NV_rectangle texture.
//
//      It is possible to render to the buffer using any OpenGL
//      mechanism. To do so Buffer provides two methods: 
//          - renderBegin()  and 
//          - renderEnd(). 
//      The buffer itself can be used as a texture if
//      not currently being rendered to. The buffers texture id 
//      is returned by textureID().
//          Buffer and Image for an envelope/letter (proxy) pattern 
//      where Buffer is the letter. Image's primare responsibility 
//      is to assure correct reference counting for Buffers so that
//      the scarce pBuffer memory on the graphics card is freed
//      and reused as often as possible. So pBuffers in the framework
//      are managed through two levels of indirection. Image is a proxy
//      for Buffer and Buffer is a proxy for the actual OpenGL pbuffer.
//
// Author: 
//      Frank Jargstorff (2003)
//
// ------------------------------------------------------------------


//
// Includes
//

#include "Buffer.h"

#include "wglPixelFormatARB.h"
#include "wglRenderTextureARB.h"

#include <iostream>
#include <assert.h>

#include "AssertGL.h"


// -----------------------------------------------------------------------------
// Buffer class
//


//
// Construction and destruction
//

        // Default constructor
        //
Buffer::Buffer(int nWidth, int nHeight): _nReferenceCount(0)
                                       , _nWidth(nWidth)
                                       , _nHeight(nHeight)
{
    GL_ASSERT_NO_ERROR;
    glGenTextures(1, &_hTexture);
    GL_ASSERT_NO_ERROR;
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hTexture);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GL_ASSERT_NO_ERROR;

    _hOldDC = wglGetCurrentDC();
    _hOldRenderContext = wglGetCurrentContext();
    GL_ASSERT_NO_ERROR;

    int aIntegerAttributes[15] = {  
            WGL_DRAW_TO_PBUFFER_ARB,                            GL_TRUE,
            WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV,        GL_TRUE,
            WGL_FLOAT_COMPONENTS_NV,                            GL_TRUE,
			WGL_RED_BITS_ARB,									16,
			WGL_GREEN_BITS_ARB,									16,
			WGL_BLUE_BITS_ARB,									16,
			WGL_ALPHA_BITS_ARB,									16,
                                0};

    float aFloatAttributes[2] = { 0.0f, 0.0f };

    int          nPixelFormat;
    unsigned int nPixelFormats;

    if (0 == wglChoosePixelFormatARB(_hOldDC, aIntegerAttributes, aFloatAttributes,
                                     1, &nPixelFormat, &nPixelFormats) )
    {
        std::cerr << "Error: Couldn't find a suitable pixel format." << std::endl;

        exit(1);
    }
    

    int aPBufferAttributes[7] = {   
            WGL_TEXTURE_TARGET_ARB,         WGL_TEXTURE_RECTANGLE_NV,
            WGL_TEXTURE_FORMAT_ARB,         WGL_TEXTURE_FLOAT_RGBA_NV,
            WGL_PBUFFER_LARGEST_ARB,        0,
                                0};

    _hPBuffer = wglCreatePbufferARB(_hOldDC, nPixelFormat, 
                                    _nWidth, 
                                    _nHeight, 
                                    aPBufferAttributes);
    if (!_hPBuffer)
    {
        DWORD err = GetLastError();
        std::cerr << "Error: Couldn't allocate p-buffer: ";
        if ( err == ERROR_INVALID_PIXEL_FORMAT )
        {
            std::cerr << "ERROR_INVALID_PIXEL_FORMAT";
        }
        else if ( err == ERROR_NO_SYSTEM_RESOURCES )
        {
            std::cerr << "ERROR_NO_SYSTEM_RESOURCES";
        }
        else if ( err == ERROR_INVALID_DATA )
        {
            std::cerr << "ERROR_INVALID_DATA";
        }
        else
        {
            std::cerr << "unknow cause";
        }
        std::cerr << std::endl;

        exit(-1);
    }

    _hDC = wglGetPbufferDCARB(_hPBuffer);
    _hRenderContext = wglCreateContext(_hDC);
    wglShareLists(_hOldRenderContext, _hRenderContext);
    GL_ASSERT_NO_ERROR;

    wglMakeCurrent(_hDC, _hRenderContext);
    GL_ASSERT_NO_ERROR;
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hTexture);
    if (!wglBindTexImageARB(_hPBuffer, WGL_FRONT_LEFT_ARB))
    {
        std::cerr << "Error: Couldn't bind the p-buffer to texture." << std::endl;

        exit(-1);
    }
    wglMakeCurrent(_hOldDC, _hOldRenderContext);
    GL_ASSERT_NO_ERROR;
}

        // Destructor
        //
Buffer::~Buffer()
{
    wglReleasePbufferDCARB(_hPBuffer, _hDC);
    GL_ASSERT_NO_ERROR;
 
    wglDestroyPbufferARB(_hPBuffer);
    GL_ASSERT_NO_ERROR;
 
    wglDeleteContext(_hRenderContext);
    GL_ASSERT_NO_ERROR;
}


// 
// Public methods
//

        // width
        //
        // Description:
        //      Get the buffer's width.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      Buffer width.
        //
        int
Buffer::width()
        const
{
    return _nWidth;
}

        // height
        //
        // Description:
        //      Get the buffer's height.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      Buffer height.
        //
        int
Buffer::height()
        const
{
    return _nHeight;
}

        // texture
        //
        // Description:
        //      Get the GL texture handle to the buffer's
        //      texture.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      A GLuint with the texture id.
        //
        GLuint
Buffer::textureID()
        const
{
    return _hTexture;
}


        // renderBegin
        //
        // Description:
        //      Makes this buffer the target for rendering.
        //          Instead of rendering to the frame buffer
        //      OpenGL commands render to this buffer if
        //      issued after the renderBegin() command.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        void
Buffer::renderBegin()
{
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hTexture);
    GL_ASSERT_NO_ERROR;

    if (!wglReleaseTexImageARB(_hPBuffer, WGL_FRONT_LEFT_ARB))
    {
        std::cerr << "Error: Couldn't unbind the p-buffer from texture." << std::endl;

        exit(-1);
    }
    wglMakeCurrent(_hDC, _hRenderContext);
    GL_ASSERT_NO_ERROR;
}

        // renderEnd
        //
        // Description:
        //      Resets rendering to the previous state.
        //          After this command was issued this buffer may
        //      be used as a texture (textureID command).
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        void
Buffer::renderEnd()
{
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _hTexture);
    GL_ASSERT_NO_ERROR;

    if (!wglBindTexImageARB(_hPBuffer, WGL_FRONT_LEFT_ARB))
    {
        std::cerr << "Error: Couldn't bind the p-buffer to texture." << std::endl;

        exit(-1);
    }
    wglMakeCurrent(_hOldDC, _hOldRenderContext);
    GL_ASSERT_NO_ERROR;
}


//
// Protected methods
//

        // referenceCount
        //
        // Description:
        //      The current reference count.
        //
        // Parameters:
        //      None
        //
        // REturns:
        //      An int with the current reference count.
        //
        int
Buffer::referenceCount()
        const
{
    return _nReferenceCount;
}

        // incrementReferenceCount
        //
        // Description:
        //      Increment the current reference count by one.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        void
Buffer::incrementReferenceCount()
{
    _nReferenceCount++;
}

        // decrementReferenceCount
        //
        // Description:
        //      Decrement the current reference count by one.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        // Note:
        //      In debug mode an assertion makes sure that a
        //      reference count of zero cannot be decremented
        //      again.
        //
        void
Buffer::decrementReferenceCount()
{
    assert(_nReferenceCount > 0);
    _nReferenceCount--;
}

        // isReferenceCountZero
        //
        // Description:
        //      Is the current reference count zero?
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      A boolean indicating if the current reference
        //      count is zero.
        //
        bool
Buffer::isReferenceCountZero()
        const
{
    return _nReferenceCount == 0;
}
