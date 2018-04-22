// ------------------------------------------------------------------
//
// Description:
//      Image is a proxy class for Buffers. 
//          Its sole purpose is to reference count Buffers and 
//      provide infrastructure for efficient resource handling (i.e.
//      freelist).
//
// Author:
//      Frank Jargstorff (2003)
//
// ------------------------------------------------------------------


//
// Includes
//

#include "Image.h"

#include "Buffer.h"

#include <assert.h>


// -----------------------------------------------------------------------------
// Image class
//

//
// Public static data
//

int Image::gnNumberOfBuffers = 0;

//
// Construction and destruction
//

        // Default constructor
        //
Image::Image(): _pBuffer(0)
{
    ; // empty
}

        // Constructor
        //
Image::Image(int nWidth, int nHeight)
{
    _pBuffer = GetBuffer(nWidth, nHeight);
    assert(_pBuffer);
    gnNumberOfBuffers++;
    _pBuffer->incrementReferenceCount();
}

        // Copy constructor
        //
Image::Image(const Image & rImage): _pBuffer(rImage._pBuffer)
{
    if (_pBuffer)
        _pBuffer->incrementReferenceCount();
}

        // Destructor
        //
Image::~Image()
{
    if (_pBuffer)
    {
        _pBuffer->decrementReferenceCount();
        if (_pBuffer->isReferenceCountZero())
        {
            ReturnBuffer(_pBuffer);
            gnNumberOfBuffers--;
        }
    }
}


// 
// Public methods
//

        // width
        //
        // Description:
        //      Get the image's width.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      Image width.
        //
        int
Image::width()
        const
{
    if (_pBuffer)
        return _pBuffer->width();

    return 0;
}

        // height
        //
        // Description:
        //      Get the image's height.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      Image height.
        //
        int
Image::height()
        const
{
    if (_pBuffer)
        return _pBuffer->height();

    return 0;
}

        // setSize
        //
        // Description:
        //      Set a new image size.
        //          The method resizes the internal buffer in
        //      which the image is stored by deleting the current
        //      buffer and creating a new one in the desired size.
        //      All image data is lost.
        //
        // Parameters:
        //      nWidth  - new image width.
        //      nHeight - new image height.
        //
        // Returns:
        //      None
        //
        // Note:
        //      If either width or height is zero the size is
        //      set to width = height = 0.
        //
        void
Image::setSize(int nWidth, int nHeight)
{
    if (_pBuffer)
    {
        _pBuffer->decrementReferenceCount();
        if (_pBuffer->isReferenceCountZero())
        {
            ReturnBuffer(_pBuffer);
            gnNumberOfBuffers--;
        }
    }

    _pBuffer = GetBuffer(nWidth, nHeight);
    gnNumberOfBuffers++;
    _pBuffer->incrementReferenceCount();
}

        // texture
        //
        // Description:
        //      Get the GL texture handle to the image's
        //      texture.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      A GLuint with the texture id.
        //
        GLuint
Image::textureID()
        const
{
    assert(_pBuffer);

    return _pBuffer->textureID();
}

        // Assignment operator
        //
        // Description:
        //      Copy one image to an other.
        //
        // Parameters
        //      rImage - the image to be copied.
        //
        // Returns:
        //      A reference to it self.
        //
        Image &
Image::operator=(const Image & rImage)
{
    if (&rImage == this)
        return *this;

    if (_pBuffer)
    {
        _pBuffer->decrementReferenceCount();
        if (_pBuffer->isReferenceCountZero())
        {
            ReturnBuffer(_pBuffer);
            gnNumberOfBuffers--;
        }
    }

    _pBuffer = rImage._pBuffer;
    if (_pBuffer)
        _pBuffer->incrementReferenceCount();

    return *this;
}


//
// Protected methods
//

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
Image::renderBegin()
{
    assert(_pBuffer);
    _pBuffer->renderBegin();
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
Image::renderEnd()
{
    assert(_pBuffer);
    _pBuffer->renderEnd();
}


//
// Protected static data
//

Image::tBufferMap Image::_goBuffers;


//
// Protected static methods
//

        
        Buffer * 
Image::GetBuffer(int nWidth, int nHeight)
{
    tBufferMap::iterator iEntry;

    iEntry = _goBuffers.find(tSize(nWidth, nHeight));
    if (iEntry == _goBuffers.end())
        return new Buffer(nWidth, nHeight);
    else
    {
        Buffer * pBuffer = (*iEntry).second;
        _goBuffers.erase(iEntry);

        return pBuffer;
    }
}

        
        void
Image::ReturnBuffer(Buffer * pBuffer)
{
    _goBuffers.insert(tBufferMap::value_type(tSize(pBuffer->width(), pBuffer->height()), pBuffer));
}
