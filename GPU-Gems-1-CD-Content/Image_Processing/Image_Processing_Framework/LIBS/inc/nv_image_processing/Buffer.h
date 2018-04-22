#ifndef BUFFER_H
#define BUFFER_H
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

#include "wglPBufferARB.h"


// -----------------------------------------------------------------------------
// Buffer class
//
class Buffer
{
public:
    //
    // Construction and destruction
    //

            // Default constructor
            //
    Buffer(int nWidth, int nHeight);

            // Destructor
            //
   ~Buffer();

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
    width()
            const;

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
    height()
            const;

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
    textureID()
            const;


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
    renderBegin();

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
    renderEnd();

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
    referenceCount()
            const;

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
    incrementReferenceCount();

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
    decrementReferenceCount();

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
    isReferenceCountZero()
            const;


private:
    //
    // Private methods
    //

            // Default constructor (private, not implemented)
    Buffer();

            // Copy constructor (private, not implemented)
    Buffer(const Buffer &);

            // Assignment operator (private, not implemented)
    operator=(const Buffer &);
    
    //
    // Private data
    //

    int         _nReferenceCount;

    int         _nWidth;
    int         _nHeight;

    GLuint      _hTexture;

    HDC         _hOldDC;
    HDC         _hDC;
    HGLRC       _hOldRenderContext;
    HGLRC       _hRenderContext;
    HPBUFFERARB _hPBuffer;

};

#endif // BUFFER_H
