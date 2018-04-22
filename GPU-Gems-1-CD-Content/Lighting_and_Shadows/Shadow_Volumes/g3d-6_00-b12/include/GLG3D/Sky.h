/**
  @file Sky.h

  @maintainer Morgan McGuire, morgan@cs.brown.edu

  @created 2002-10-04
  @edited  2004-01-06
*/

#ifndef G3D_SKY_H
#define G3D_SKY_H

#include "GLG3D/Texture.h"
#include "GLG3D/RenderDevice.h"
#include "graphics3D.h"

namespace G3D {

typedef ReferenceCountedPointer<class Sky> SkyRef;

/**
 A background cube with an appropriately warped texture to make it 
 look like an infinite environment.  The sky also manages drawing
 the sun and moon, with lens flare effects for the sun.  
 
 Copy the images from data/sky to your project directory to use this
 class or provide your own.
   
 Example:

  <PRE>
    Sky sky(renderDevice);
    LightingParameters lighting(toSeconds(9, 00, 00, AM));

    ...

    // Rendering loop
        sky.render(lighting);
        
        // Draw the rest of the scene
        ...

        sky.renderLensFlare(lighting);
        ...
  </PRE>

 */
class Sky : public ReferenceCountedObject {
private:
    /** Indices into texture array */
    enum Direction {UP = 0, LT = 1, RT = 2, BK = 3, FT = 4, DN = 5};
	
    /**
     Only used if the render device does not support cube maps.
     */
    TextureRef                                  texture[6];

    /**
     Used unless the render device does not support cube maps.
     */
    TextureRef                                  cubeMap;

    TextureRef                                  sun;
    TextureRef                                  moon;
    TextureRef                                  disk;
    TextureRef                                  sunRays;

    Array<Vector4>                              star;
    Array<float>                                starIntensity;

    bool                                        drawCelestialBodies;

    class RenderDevice*                         renderDevice;

    /**
     Renders the sky box.
     */
    void renderBox() const;

    /** Draw the sun, called by render() */
    void drawSun(const class LightingParameters&);

    /** Draw the night sky, called by render() */
    void drawMoonAndStars(const class LightingParameters&);

    Sky(
        class RenderDevice*                     renderDevice,
        const std::string&                      directory,
        const std::string&                      filename,
        bool                                    drawCelestialBodies,
        double                                  quality);

public:
    /**
     @param directory If directory is not "" it should
      end in a trailing slash.  This is the location of the real.str file
      and the sun, moon, etc. files.

     @param filename A filename with "*" in place of {up, lt, rt, bk, ft, dn}.
      The filename can be either fully qualified, relative to the current
      directory, or relative to <I>directory</I>.

     @param drawCelestialBodies If true, draw the sun, moon, and stars. Requires
      moon.jpg, moon-alpha.jpg, sun.jpg, lensflare.jpg, sun-rays.jpg be present
      in given directory. Defaults to true.

     @param quality Trade image quality for texture memory: 
       .5 -> 1/8  the texture memory of 1.0, 
        0 -> 1/16 the texture memory of 1.0.
        Color banding will occur at low quality settings.
     */
    static SkyRef create(
        class RenderDevice*                     renderDevice,
        const std::string&                      directory,
        const std::string&                      filename = "plainsky/null_plainsky512_*.jpg",
        bool                                    drawCelestialBodies = true,
        double                                  quality = 1.0);

    virtual ~Sky();

    /**
     Call at the very beginning of your rendering routine.
     Will restore all state it changes.
     */
	void render(
        const class LightingParameters&         lighting);

    /**
     Call at the very end of your rendering routine.
     Will restore all state it changes.
     */
    void renderLensFlare(
        const class LightingParameters&         lighting);

    /**
     Returns an environment map (NULL if cube maps are not supported
     on this machine).
     */
    inline TextureRef getEnvironmentMap() const {
        return cubeMap;
    }

};

inline bool operator==(const SkyRef& a, const void* b) {
    return (b == NULL) && (a == (SkyRef)NULL);
}

inline bool operator==(const void* a, const SkyRef& b) {
    return b == a;
}

inline bool operator!=(const SkyRef& a, const void* b) {
    return !(a == b);
}

inline bool operator!=(const void* b, const SkyRef& a) {
    return !(a == b);
}

}

#endif
