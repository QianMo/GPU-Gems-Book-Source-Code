/**
  @file Effects.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#ifndef _EFFECTS_H_
#define _EFFECTS_H_

#include <graphics3D.h>

class BasicCamera;
class Viewport;
class BasicModel;
struct DemoSettings;


/**
 * This loads a font image file and sets transparency
 * and foreground color.  This assumes the font is black on white.
 */
TextureRef loadFontTexture(
    const std::string &                     filename);


/**
 * This draws a string of characters to the string using the
 * font texture.
 */
void drawFontString(
    const std::string &                     str,
    const TextureRef                        tex,
    int                                     charWidth,
    int                                     charHeight,
    int                                     kerning,
    float                                   x,
    float                                   y,
    int                                     winWidth,
    int                                     winHeight);


/**
 * This is one of the first rendering steps in each frame
 */
void drawSkyBox(
    const BasicCamera&                           camera,
    const Array<TextureRef>&                skybox);


/**
 * This draws a lot of small special effects as a final
 * rendering pass.  Examples include visually showing shadow
 * volumes and printing the number of polygons in the scene.
 * This is the rendering step for each frame.
 */
void finalPass(
    Array<BasicModel*>&                     modelArray,
    const BasicCamera&                           camera,
    const Viewport&                         view,
    TextureRef                              tex,
    DemoSettings&                           vars);


#endif


