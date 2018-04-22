/**
 @file GFont.h
 
 @maintainer Morgan McGuire, morgan@graphics3d.com

 @created 2002-11-02
 @edited  2003-12-16
 */

#ifndef G3D_GFONT_H
#define G3D_GFONT_H

#include "GLG3D/Texture.h"

namespace G3D {

typedef ReferenceCountedPointer<class GFont> GFontRef;

/**
 Font class for use with RenderDevice.
 <P>
 The following fonts are provided with G3D in the <CODE>data/font</CODE> directory.  See
 the <CODE>copyright.txt</CODE> file in that directory for information about the
 source of these files and rules for distribution.
 <P>
 <IMG SRC="font.png">
 */
class GFont : public ReferenceCountedObject {
public:
    /** Constant for draw2DString.  Specifies the horizontal alignment of an entire string relative to the supplied x,y position */
    enum XAlign {XALIGN_RIGHT, XALIGN_LEFT, XALIGN_CENTER};

    /** Constant for draw2DString.  Specifies the vertical alignment of the characters relative to the supplied x,y position.
      */
    enum YAlign {YALIGN_TOP, YALIGN_BASELINE, YALIGN_CENTER, YALIGN_BOTTOM};

    /** Constant for draw2DString.  Proportional width (default) spaces characters based on their size.
        Fixed spacing gives uniform spacing regardless of character width. */
    enum Spacing {PROPORTIONAL_SPACING, FIXED_SPACING};

private:
    /** The actual width of the character. */ 
    int subWidth[128];

    /** The width of the box, in texels, around the character. */
    int charWidth;
    int charHeight;

    /** Y distance from top of the bounding box to the font baseline. */
    int baseline;

    TextureRef texture;

    /** Assumes you are already inside of beginPrimitive(QUADS) */
    Vector2 drawString(
        const std::string&      s,
        double                  x,
        double                  y,
        double                  w,
        double                  h,
        Spacing                 spacing) const;

    class RenderDevice*             renderDevice;

    GFont(class RenderDevice* renderDevice, const std::string& filename);

public:

    /** The filename must be a FNT (proportional width font) file.
        <P>   
        Several fonts in this format at varying resolutions are available in the data/font directory.
        The large fonts require 500k of memory when loaded and look good when rendering characters up to
        about 64 pixels high. The small fonts require 130k and look good up to about 32 pixels.

        See Font::convertTGAtoPWF for creating new fonts in the FNT format:
    
      <P>
       This file is compressed by BinaryOutput::compress().  The contents after decompression 
       have the following format (little endian): 
          <pre>
           int32                       Version number (must be 1)
           128 * int16                 Character widths, in texels
           uint16                      Baseline from top of box, in texels
           uint16                      Texture width (texture height is always 1/2 texture width)
           (pow(width, 2) / 2) * int8  Texture data
          </pre>
        The width of a character's bounding box is always width / 16.  The height is always width / 8.
    */
    static GFontRef fromFile(class RenderDevice* renderDevice, const std::string& filename);

    /**
     Converts an 8-bit RAW font texture and INI file as produced by the Bitmap Font Builder program
     to a graphics3d PWF font.  inFile should have no extension-- .tga and .ini will be appended to
     it.  outfile should end with ".FNT" or be "" for the default.
     <P>
      The Bitmap Font Builder program can be downloaded from http://www.lmnopc.com/bitmapfontbuilder/
      Write out RAW files with characters CENTER aligned and right side up using this program.  Use the
      full ASCII character set; the conversion will strip infrequently used characters automatically.
      
      Example:
      <PRE>
          Font::convertRAWINItoPWF("c:/tmp/g3dfont/news", "d:/graphics3d/book/cpp/data/font/news.fnt");
      </PRE>
	  @param infileBase The name of the tga/ini files
      @param outfile Defaults to infileBase + ".fnt"
     */
    static void convertRAWINItoPWF(const std::string& infileBase, std::string outfile = "");


    /** Returns the natural character width and height of this font. */
    Vector2 texelSize() const;

    /**
     Draws a proportional width font string.  Assumes device->push2D()
     has been called.  Leaves all rendering state as it was.

     @param size The distance between successive lines of text.  Specify
     texelSize().y / 1.5 to get 1:1 texel to pixel

     @param outline If this color has a non-zero alpha, a 1 pixel border of
     this color is drawn about the text.

     @param spacing Fixed width fonts are spaced based on the width of the 'M' character.

     @return Returns the x and y bounds (ala get2DStringBounds) of the printed string.
     */
    Vector2 draw2D(
        const std::string&  s,
        const Vector2&      pos2D,
        double              size    = 12,
        const Color4&       color   = Color3::BLACK,
        const Color4&       outline = Color4::CLEAR,
        XAlign              xalign  = XALIGN_LEFT,
        YAlign              yalign  = YALIGN_TOP,
        Spacing             spacing = PROPORTIONAL_SPACING) const;

    /**
     Useful for drawing centered text and boxes around text.
     */
    Vector2 get2DStringBounds(
        const std::string&  s,
        double              size = 12,
        Spacing             spacing = PROPORTIONAL_SPACING) const;
};

/**
 @deprecated
 */
typedef GFont CFont;

/**
 @deprecated
 */
typedef GFontRef CFontRef;

}
#endif

