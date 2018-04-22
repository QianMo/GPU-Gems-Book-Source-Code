/**
  @file TextureManager.h

  @maintainer Morgan McGuire, matrix@graphics3d.com
  @cite by Morgan McGuire & Peter Sibley

  @created 2003-11-25
  @edited  2003-12-01
*/

#ifndef G3D_TEXTUREMANAGER_H
#define G3D_TEXTUREMANAGER_H

#include "graphics3D.h"
#include "GLG3D/Texture.h"

namespace G3D {

/**
 Caches previously loaded textures.

 You can load textures manually using the G3D::Texture static methods.
 This class allows you to manage a set of textures with cached loading.
 Textures with no external references are cleared when the cache exceeds
 a pre-set size hint.
 */
class TextureManager {
private:

    /**
     Wrapper for the arguments to a texture constructor
     */
    class TextureArgs : public Hashable {
    public:

        std::string                     filename;
        const TextureFormat*            format;  
        Texture::WrapMode               wrap;  
        Texture::InterpolateMode        interpolate;
        Texture::Dimension              dimension; 
        double                          brighten;
        
        TextureArgs() : format(NULL) {}
        
        TextureArgs(const TextureFormat* _format) : format(_format) {}
        virtual ~TextureArgs() {}

        unsigned int hashCode() const;

        bool operator==(const TextureArgs&) const;
    };


    Table<TextureArgs, TextureRef>      cache;
    
    size_t                              size;
    size_t                              sizeHint;

    /**
     Checks to see if we have exceeded sizeHint, if so we will
     remove as many old entries as needed.
    */
    void checkCacheSize();
    
    /**
      A stale entry is a texture with no external references to it. 
      This returns all of the stale entries in the cache.
     */
    void getStaleEntries(Array<TextureArgs>& staleEntry);

public:

    TextureManager(size_t _sizeHint = 10*1024*1024);


    /**
     If the texture has recently been loaded with the same options, a
     reference to the shared texture is returned.  Otherwise the
     texture is loaded from disk.
    */
    TextureRef loadTexture(
        const std::string&          filename, 
        const TextureFormat*        desiredFormat = TextureFormat::AUTO,  
        Texture::WrapMode           wrap = Texture::TILE,  
        Texture::InterpolateMode    interpolate = Texture::TRILINEAR_MIPMAP,  
        Texture::Dimension          dimension = Texture::DIM_2D,  
        double                      brighten = 1.0);

    /** 
     Sets the maxiumum amount of memory that the cache will use before it starts to purge old entries.
     */
    void setMemorySizeHint(size_t _size);
    
    size_t memorySizeHint() const;

    /**
     Returns the sum of the sizes of the textures in the cache.
     */
    size_t sizeInMemory() const;

    /**
     Completely empties the contents of the cache.
     If a client program has a textureRef, that texture will remain in memory, but be reloaded from disk when load is called.
     */
    void emptyCache();

    /**
     Removes any currently unreferenced textures from the cache.
     */
    void trimCache();
};


}

#endif
