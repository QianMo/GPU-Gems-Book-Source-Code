/**
  @file TextureFormat.h

  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2003-05-23
  @edited  2003-11-18
*/

#ifndef GLG3D_TEXTUREFORMAT_H
#define GLG3D_TEXTUREFORMAT_H

#include "graphics3D.h"
#include "GLG3D/glheaders.h"

namespace G3D {

/**
 Used to describe texture formats to the G3D::Texture class.
 Don't construct these; use the static constants provided.
 */
class TextureFormat {
public:

    /**
     Number of channels (1 for a depth texture).
     */
    int                 numComponents;
    bool                compressed;
    
    /**
     The GL format equivalent to this one.
     */
    GLenum              OpenGLFormat;
    GLenum              OpenGLBaseFormat;

    int                 luminanceBits;

    /**
     Number of bits per texel storage for alpha values; Zero for compressed textures.
     */
    int                 alphaBits;
    
    /**
     Number of bits per texel storage for red values; Zero for compressed textures.
     */
    int                 redBits;

    /**
     Number of bits per texel storage for green values; Zero for compressed textures.
     */
    int                 greenBits;
    
    /**
     Number of bits per texel storage for blue values; Zero for compressed textures.
     */
    int                 blueBits;

    /**
     Number of depth bits (for depth textures; e.g. shadow maps)
     */
    int                 depthBits;

    /**
     Sum of the per-channel bits, plus any additional bits required
     for byte alignment.
     */
    int                 packedBitsPerTexel;
    
    /**
     This may be greater than the sum of the per-channel bits
     because graphics cards need to pad to the nearest 1, 2, or
     4 bytes.
     */
    int                 hardwareBitsPerTexel;


    /**
     True if there is no alpha channel for this texture.
     */
    bool                opaque;

private:

    TextureFormat(
        int             _numComponents,
        bool            _compressed,
        GLenum          _glFormat,
        GLenum          _glBaseFormat,
        int             _luminanceBits,
        int             _alphaBits,
        int             _redBits,
        int             _greenBits,
        int             _blueBits,
        int             _depthBits,
        int             _hardwareBitsPerTexel,
        int             _packedBitsPerTexel,
        bool            _opaque) : 
        numComponents(_numComponents),
        compressed(_compressed),
        OpenGLFormat(_glFormat),
        OpenGLBaseFormat(_glBaseFormat),
        luminanceBits(_luminanceBits),
        alphaBits(_alphaBits),
        redBits(_redBits),
        greenBits(_greenBits),
        blueBits(_blueBits),
        depthBits(_depthBits),
        packedBitsPerTexel(_packedBitsPerTexel),
        hardwareBitsPerTexel(_hardwareBitsPerTexel),
        opaque(_opaque) {
    }

public:

    /**
     Default argument for TextureFormat::depth.
     */
    enum {SAME_AS_SCREEN = 0};

    static const TextureFormat* L8;

    static const TextureFormat* A8;

    static const TextureFormat* LA8;

    static const TextureFormat* RGB5;

    static const TextureFormat* RGB5A1;

    static const TextureFormat* RGB8;

    static const TextureFormat* RGBA8;
    
    static const TextureFormat* RGB_DXT1;

    static const TextureFormat* RGBA_DXT1;

    static const TextureFormat* RGBA_DXT3;

    static const TextureFormat* RGBA_DXT5;

    static const TextureFormat* DEPTH16;

    static const TextureFormat* DEPTH24;

    static const TextureFormat* DEPTH32;

    /**
     NULL pointer; indicates that the texture class should choose
     either RGBA8 or RGB8 depending on the presence of an alpha channel
     in the input.
     */
    static const TextureFormat* AUTO;

    /**
     Returns DEPTH16, DEPTH24, or DEPTH32 according to the bits
     specified.
     */
    static const TextureFormat* depth(int depthBits = SAME_AS_SCREEN);
};

}

#endif
