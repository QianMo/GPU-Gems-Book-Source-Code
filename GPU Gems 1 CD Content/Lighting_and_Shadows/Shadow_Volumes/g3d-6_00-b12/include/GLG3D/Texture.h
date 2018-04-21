/**
  @file Texture.h

  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2001-02-28
  @edited  2003-11-24
*/

#ifndef GLG3D_TEXTURE_H
#define GLG3D_TEXTURE_H

#include "graphics3D.h"
#include "GLG3D/glheaders.h"
#include "GLG3D/TextureFormat.h"

namespace G3D {


typedef ReferenceCountedPointer<class Texture> TextureRef;

/**

 Abstraction of OpenGL textures.  This class can be used with raw OpenGL, 
 without RenderDevice or SDL.

 If you use TextureRef instead of Texture*, the texture memory will be
 garbage collected.

 If you enable texture compression, textures will be compressed on the fly.
 This can be slow (up to a second).

 Unless DIM_2D_RECT is used, texture automatically scales non-power of 2
 size textures up to the next power of 2 (hardware requirement).

 Textures are loaded so that (0, 0) is the upper-left corner of the image.
 If you set the invertY flag, RenderDevice will automatically turn them upside
 down when rendering to allow a (0, 0) <B>lower</B>-left corner.  If you
 aren't using RenderDevice, you must change the texture matrix to have
 a -1 in the Y column yourself.

 DIM_2D_RECT requires the GL_EXT_texture_rectangle extension.
 Texture compression requires the EXT_texture_compression_s3tc extions.
 You can either query OpenGL for whether these are supported or
 use the RenderDevice facility for doing so.

 To use Texture with straight OpenGL:

 <PRE>
  TextureRef texture = new Texture("logo.jpg");

  ...
    
  GLint u = texture->getOpenGLTextureTarget();
  glEnable(u);
  glBindTexture(u, texture->getOpenGLID());
 </PRE>

 To use Texture with RenderDevice:

  <PRE>
  TextureRef texture = new Texture("logo.jpg");
  ...
  renderDevice->setTexture(0, texture);
  // (to disable: renderDevice->setTexture(0, NULL);)
  </PRE>


  3D MIP Maps are not supported because gluBuild3DMipMaps is not in all GLU implementations.
 */
class Texture : public ReferenceCountedObject {
public:

    enum Dimension       {DIM_2D = 2, DIM_2D_RECT = 4, DIM_CUBE_MAP = 5};

    /** TRANSPARENT_BORDER provides a border of Color4(0,0,0,0) and clamps to it. */
    enum WrapMode        {TILE = 1, CLAMP = 0, TRANSPARENT_BORDER = 2};

    enum InterpolateMode {TRILINEAR_MIPMAP = 3, BILINEAR_NO_MIPMAP = 2, NO_INTERPOLATION = 1};

    /**
     Splits a filename around the '*' character-- used by cube maps to generate all filenames.
     */
    static void splitFilenameAtWildCard(
        const std::string&  filename,
        std::string&        filenameBeforeWildCard,
        std::string&        filenameAfterWildCard);

private:

    /** OpenGL texture ID */
	GLuint                          textureID;

    std::string                     name;
    InterpolateMode                 interpolate;
    WrapMode                        wrap;
    Dimension                       dimension;
    const class TextureFormat*      format;
    int                             width;
    int                             height;
    int                             depth;
    bool                            _opaque;

    Texture(
        const std::string&          _name,
        GLuint                      _textureID,
        Dimension                   _dimension,
        const class TextureFormat*  _format,
        InterpolateMode             _interpolate,
        WrapMode                    _wrap,
        bool                        __opaque);

public:

    /**
     Returns a new OpenGL texture ID.
     */
    static unsigned int newGLTextureID();

    /**
     Creates an empty texture (useful for later reading from the screen).
     */
    static TextureRef createEmpty(
        int                             width,
        int                             height,
        const std::string&              name           = "Texture",
        const class TextureFormat*      desiredFormat  = TextureFormat::RGBA8,
        WrapMode                        wrap           = TILE,
        InterpolateMode                 interpolate    = TRILINEAR_MIPMAP,
        Dimension                       dimension      = DIM_2D);

    /**
     Wrap and interpolate will override the existing parameters on the
     GL texture.

     @param name Arbitrary name for this texture to identify it
     @param textureID Set to newGLTextureID() to create an empty texture.
     */
    static TextureRef fromGLTexture(
        const std::string&              name,
        GLuint                          textureID,
        const class TextureFormat*      textureFormat,
        WrapMode                        wrap           = TILE,
        InterpolateMode                 interpolate    = TRILINEAR_MIPMAP,
        Dimension                       dimension      = DIM_2D);


    /**
     Creates a texture from a single image.  The image must have a format understood
     by G3D::GImage.  If dimension is DIM_CUBE_MAP, this loads the 6 files with names
     _ft, _bk, ... following the G3D::Sky documentation.
     @param brighten A value to multiply all color channels by; useful for loading
            dark Quake textures.
     */
    static TextureRef fromFile(
        const std::string&              filename,
        const class TextureFormat*      desiredFormat  = TextureFormat::AUTO,
        WrapMode                        wrap           = TILE,
        InterpolateMode                 interpolate    = TRILINEAR_MIPMAP,
        Dimension                       dimension      = DIM_2D,
        double                          brighten       = 1.0);


    /**
     Creates a texture from the colors of filename and takes the alpha values
     from the red channel of alpha filename.
     */
    static TextureRef fromTwoFiles(
        const std::string&              filename,
        const std::string&              alphaFilename,
        const class TextureFormat*      desiredFormat  = TextureFormat::RGBA8,
        WrapMode                        wrap           = TILE,
        InterpolateMode                 interpolate    = TRILINEAR_MIPMAP,
        Dimension                       dimension      = DIM_2D);

    /**
     The bytes are described by byteFormat, which may differ from the
     format you want the graphics card to use (desiredFormat).
     If dimenion is DIM_CUBE_MAP bytes is an array of six images (for the faces)
     in the order: {FT, BK, UP, DN, RT, LF}.  Otherwise bytes is a pointer to
     an array of data.  Note that all faces must have the same dimensions and
     format for cube maps.
     */
    static TextureRef fromMemory(
        const std::string&              name,
        const uint8**                   bytes,
        const class TextureFormat*      bytesFormat,
        int                             width,
        int                             height,
        int                             depth,
        const class TextureFormat*      desiredFormat  = TextureFormat::AUTO,
        WrapMode                        wrap           = TILE,
        InterpolateMode                 interpolate    = TRILINEAR_MIPMAP,
        Dimension                       dimension      = DIM_2D);

    static TextureRef fromGImage(
        const std::string&              name,
        const GImage&                   image,
        const class TextureFormat*      desiredFormat  = TextureFormat::AUTO,
        WrapMode                        wrap           = TILE,
        InterpolateMode                 interpolate    = TRILINEAR_MIPMAP,
        Dimension                       dimension      = DIM_2D);

    /**
     Copies data from screen into an existing texture (replacing whatever was
     previously there).  The dimensions must be powers of two or a texture 
     rectangle will be created (not supported on some cards).

     The (x, y) coordinates are in real screen pixels.  (0, 0) is the top left
     of the screen.

     The texture dimensions will be updated but all other properties will be preserved:
     The previous wrap mode will be preserved.
     The interpolation mode will be preserved (unless it required a mipmap,
     in which case it will be set to BILINEAR_NO_MIPMAP).  The previous color depth
     and alpha depth will be preserved.  Texture compression is not supported for
     textures copied from the screen.

     To copy a depth texture, first create an empty depth texture then copy into it.

     If you invoke this method on a texture that is currently set on RenderDevice,
     the texture will immediately be updated (there is no need to rebind).

     @param useBackBuffer If true, the texture is created from the back buffer.
     If false, the texture is created from the front buffer.

     @param rect The rectangle to copy (relative to the viewport)
     */
    void copyFromScreen(const Rect2D& rect, bool useBackBuffer = true);

    /**
     Argument for copyFromScreen
     */
    enum CubeFace {
        CUBE_POS_X = 0,
        CUBE_NEG_X = 1,
        CUBE_POS_Y = 2,
        CUBE_NEG_Y = 3,
        CUBE_POS_Z = 4,
        CUBE_NEG_Z = 5};

    /**
     Copies into the specified face of a cube map.  Because cube maps can't have
     the Y direction inverted (and still do anything useful), you should render
     the cube map faces <B>upside-down</B> before copying them into the map.  This
     is an unfortunate side-effect of OpenGL's cube map convention.  
     
     Use G3D::Texture::getCameraRotation to generate the (upside-down) camera
     orientations.
     */
    void copyFromScreen(const Rect2D& rect, CubeFace face, bool useBackBuffer = true);

    /**
     Returns the rotation matrix that should be used for rendering the
     given cube map face.
     */
    static void getCameraRotation(CubeFace face, Matrix3& outMatrix);

    /**
     When true, rendering code that uses this texture is respondible for
     flipping texture coordinates applied to this texture vertically (initially,
     this is false).
     
     RenderDevice watches this flag and performs the appropriate transformation.
     If you are not using RenderDevice, you must do it yourself.
     */
    bool invertY;

    /**
     How much (texture) memory this texture occupies.
     */
    int sizeInMemory() const;

    /**
     True if this texture was created with an alpha channel.  Note that
     a texture may have a format that is not opaque (e.g. RGBA8) yet still
     have a completely opaque alpha channel, causing texture->opaque to
     be true.  This is just a flag set for the user's convenience-- it does
     not affect rendering in any way.
     */
    inline bool opaque() const {
        return _opaque;
    }


    inline unsigned int getOpenGLID() const {
        return textureID;
    }

    inline const int getTexelWidth() const {
        return width;
    }

    inline const int getTexelHeight() const {
        return height;
    }

    /**
     For 3D textures.
     */
    inline const int getTexelDepth() const {
        return depth;
    }

    inline const std::string& getName() const {
        return name;
    }

    inline InterpolateMode getInterpolateMode() const {
        return interpolate;
    }

    inline WrapMode getWrapMode() const {
        return wrap;
    }

    inline const TextureFormat* getFormat() const {
        return format;
    }
    
    inline Dimension getDimension() const {
        return dimension;
    }

    /**
     Deallocates the OpenGL texture.
     */
    virtual ~Texture();

    /**
     The OpenGL texture target this binds (e.g. GL_TEXTURE_2D)
     */
    unsigned int getOpenGLTextureTarget() const;

};

inline bool operator==(const TextureRef& a, const void* b) {
    return (b == NULL) && (a == (TextureRef)NULL);
}

inline bool operator==(const void* a, const TextureRef& b) {
    return b == a;
}

inline bool operator!=(const TextureRef& a, const void* b) {
    return !(a == b);
}

inline bool operator!=(const void* b, const TextureRef& a) {
    return !(a == b);
}


} // namespace

#endif
