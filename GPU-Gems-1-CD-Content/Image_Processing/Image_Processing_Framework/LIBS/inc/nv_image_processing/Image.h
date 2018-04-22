#ifndef IMAGE_H
#define IMAGE_H
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
// Pragmas
//

#pragma warning(disable:4786) // stupid STL truncation warning


//
// Includes
//

#ifdef _WIN32
    #include <windows.h>
#endif

#include <gl/gl.h>

#include <map>


//
// Forward declarations
//

class Buffer;


// -----------------------------------------------------------------------------
// Image class
//
class Image
{
public:
    //
    // Public static data
    //

    static int gnNumberOfBuffers;

public:
    //
    // Construction and destruction
    //

            // Default constructor
            //
    Image();

            // Constructor
            //
    Image(int nWidth, int nHeight);

            // Copy constructor
            //
    Image(const Image & rImage);

            // Destructor
            //
   ~Image();


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
    width()
            const;

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
    height()
            const;

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
    setSize(int nWidth, int nHeight);

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
    textureID()
            const;

            // Assitnment operator
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
    operator=(const Image & rImage);

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


protected:
    //
    // Protected typedefs
    //

    struct tSize
    {
        tSize(int nWidth, int nHeight): _nWidth(nWidth), _nHeight(nHeight)
        { /* empty */  };
        
        int _nWidth;
        int _nHeight;

        bool operator<(const tSize & rSize) const
        {
            if (_nWidth == rSize._nWidth)
            {
                if (_nHeight < rSize._nHeight)
                    return true;
                else
                    return false;
            }
            if (_nWidth < rSize._nWidth)
                return true;
            else 
                return false;
        };
    };
    
    typedef std::multimap<tSize, Buffer *> tBufferMap;

   
    //
    // Protected static methods
    //

            static 
            Buffer * 
    GetBuffer(int nWidth, int nHeight);

            static
            void
    ReturnBuffer(Buffer * pBuffer);

    // 
    // Protected static data
    //

    static tBufferMap _goBuffers;


private:
    //
    // Private data
    //

    Buffer * _pBuffer;

};

#endif // IMAGE_H
