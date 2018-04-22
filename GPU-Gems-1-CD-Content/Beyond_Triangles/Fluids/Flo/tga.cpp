//------------------------------------------------------------------------------
// File : tga.cpp
//------------------------------------------------------------------------------
// GLVU : Copyright 1997 - 2002 
//        The University of North Carolina at Chapel Hill
//------------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software and its 
// documentation for any purpose is hereby granted without fee, provided that 
// the above copyright notice appear in all copies and that both that copyright 
// notice and this permission notice appear in supporting documentation. 
// Binaries may be compiled with this software without any royalties or 
// restrictions. 
//
// The University of North Carolina at Chapel Hill makes no representations 
// about the suitability of this software for any purpose. It is provided 
// "as is" without express or implied warranty.

//============================================================================
// tga.hpp : Targa image format module
//============================================================================
// Modified by Vincent Scheib, Aug 2000 
//   from a modified version from David K. McAllister, Aug. 2000.

#include <stdio.h>
#include <assert.h>
#include <memory.h>
#include <fstream>
#include <iostream>

#ifdef _WIN32
// MSVC doesn't define 'ios::nocreate' in <fstream>, only in <fstream.h>.
// but SGI doesn't define 'ios::binary' in <fstream.h> only <fstream>!
// DOH!
// Follow up: 10/5/01  Apparently ios::nocreate is dead.  The latest standard
// doesn't include it because opening a file that doesn't exist never really
// should have been implemented to create the file.
#define NOCREATE 0x20
#else
#define NOCREATE ((ios::openmode)0x0)
#endif

#define TGA_VERBOSE 0

using namespace std;

//============================================================================
//============================================================================
// Definitions 
//============================================================================

// Header definition.
typedef struct TGA_Header_
{
  unsigned char ImageIDLength; // length of Identifier String.
  unsigned char CoMapType; // 0 = no map
  unsigned char ImgType; // image type (see below for values)
  unsigned char Index_lo, Index_hi; // index of first color map entry
  unsigned char Length_lo, Length_hi; // number of entries in color map
  unsigned char CoSize; // size of color map entry (15,16,24,32)
  unsigned char X_org_lo, X_org_hi; // x origin of image
  unsigned char Y_org_lo, Y_org_hi; // y origin of image
  unsigned char Width_lo, Width_hi; // width of image
  unsigned char Height_lo, Height_hi; // height of image
  unsigned char PixelSize; // pixel size (8,16,24,32)
  unsigned char Desc; // 4 bits are number of attribute bits per pixel
} TGA_Header;

// Definitions for image types.
#define TGA_NULL 0
#define TGA_MAP 1
#define TGA_RGB 2
#define TGA_MONO 3
#define TGA_RLEMAP 9
#define TGA_RLERGB 10
#define TGA_RLEMONO 11
#define TGA_DESC_ALPHA_MASK ((unsigned char)0xF) // number of alpha channel bits
#define TGA_DESC_ORG_MASK ((unsigned char)0x30) // origin mask
#define TGA_ORG_BOTTOM_LEFT 0x00 // origin mask
#define TGA_ORG_BOTTOM_RIGHT 0x10
#define TGA_ORG_TOP_LEFT 0x20
#define TGA_ORG_TOP_RIGHT 0x30

//============================================================================
//============================================================================
// Helper functions for LOAD
//============================================================================

static int decode_map8(const unsigned char *src0, unsigned char *dest0, const int size,
					   const int chan, const unsigned char *color_map)
{
	const unsigned char *src = src0;
	unsigned char *dest = dest0;
	
	for(int i=0; i<size; i++)
	{
		for(int c=0; c<chan; c++)
			dest[chan - c - 1] = color_map[src[0] * chan + c];
		src++;
		dest += chan;
	}
	
	return(0);
}

static int decode_map_rle8(
  const unsigned char *src0, unsigned char *dest0, int width, int height,
  const int chan, const unsigned char *color_map)
{
  const unsigned char *src = src0;
  unsigned char *dest = dest0;
	
  while(dest < dest0 + width*height*chan) {
    // Pixels encoded in "packets"

    // First byte is raw/rle flag(upper bit) and count(1-128 as 0-127
    // in lower 7 bits) If raw, the next count chan-byte color values
    // in the file are taken verbatim If rle, the next single
    // chan-byte color value speaks for the next count pixels
		
    int raw = (*src & 0x80) == 0; // Is this packet raw pixels or a repeating color
    int count = (*src & 0x7f) + 1; // How many raw pixels or color repeats
    src++; // Advance src beyond first byte to map value
		
    if(dest + count*chan > dest0 + width*height*chan)
      count = (dest0 + width*height*chan - dest) / chan;
		
    for(int j=0; j<count; j++)
    {

      for(int c=0; c<chan; c++)
        dest[chan - c - 1] = color_map[src[0] * chan + c];
			
      if(raw) // In raw mode, keep advancing "src" to subsequent values
        src++; // In RLE mode, just repeat the packet[1] RGB color
      dest += chan;
    }
    if(!raw) // After outputting count RGBA values, advance "src" if rle
      src ++;
  }
	
  return(0);
}


static int decode_rgb16(const unsigned char *src0, unsigned char *dest0,
						const int width, const int height)
{
	const unsigned char *src = src0;
	unsigned char *dest = dest0;
	
	for(int i=0; i<width*height; i++)
	{
		dest[0] = (src[1] << 1) & 0xf8;
		dest[1] = ((src[1] << 6) | (src[0] >> 2)) & 0xf8;
		dest[2] = (src[0] << 3) & 0xf8;
		src += 2;
		dest += 3;
	}
	
	return(0);
}

static int decode_rgb24(const unsigned char *src0, unsigned char *dest0,
						const int width, const int height)
{
	const unsigned char *src = src0;
	unsigned char *dest = dest0;
	
	for(int i=0; i<width*height; i++)
	{
		dest[0] = src[2]; // Red
		dest[1] = src[1]; // Green 
		dest[2] = src[0]; // Blue
		src += 3;
		dest += 3;
	}
	
	return(0);
}

static int decode_rgb32(const unsigned char *src0, unsigned char *dest0,
						const int width, const int height)
{
	const unsigned char *src = src0;
	unsigned char *dest = dest0;
	
	for(int i=0; i<width*height; i++)
	{
		dest[0] = src[2]; // Red
		dest[1] = src[1]; // Green 
		dest[2] = src[0]; // Blue
		dest[3] = src[3]; // Alpha
		src += 4;
		dest += 4;
	}
	
	return(0);
}

// Decode run-length encoded Targa into 32-bit pixels
// Stores 24-bit source data into a 32-bit dest.
// Sets alpha to 0.
static int decode_rgb_rle16(const unsigned char *src0, unsigned char *dest0,
							const int width, const int height)
{
	int chan = 3;
	const unsigned char *src = src0;
	unsigned char *dest = dest0;

	while(dest < dest0 + width*height*chan) {
		// Pixels encoded in "packets"
		// First byte is raw/rle flag(upper bit) and count(1-128 as 0-127 in lower 7 bits)
		// If raw, the next count chan-byte color values in the file are taken verbatim
		// If rle, the next single chan-byte color value speaks for the next count pixels
		
		int raw = (*src & 0x80) == 0; // Is this packet raw pixels or a repeating color
		int count = (*src & 0x7f) + 1; // How many raw pixels or color repeats
		src++; // Advance src beyond first byte to next color

		if(dest + count*chan > dest0 + width*height*chan) // prevent from writing out of dest range
			count = (dest0 + width*height*chan - dest) / chan;
		
		for(int j=0; j<count; j++)
		{
			dest[0] = (src[1] << 1) & 0xf8;
			dest[1] = ((src[1] << 6) | (src[0] >> 2)) & 0xf8;
			dest[2] = (src[0] << 3) & 0xf8;

			if(raw) // In raw mode, keep advancing "src" to subsequent values
				src += 2; // In RLE mode, just repeat the packet[1] RGB color
			dest += chan;
		}
		if(!raw) // After outputting count RGBA values, advance "src" beyond color if rle
			src += 2;
	}

	assert(dest <= dest0 + width*height*chan);

	return(0);
}

// Decode run-length encoded Targa into 32-bit pixels
// Stores 24-bit source data into a 32-bit dest.
// Sets alpha to 0.
static int decode_rgb_rle24(const unsigned char *src0, unsigned char *dest0,
							const int width, const int height)
{
	int chan = 3;
	const unsigned char *src = src0;
	unsigned char *dest = dest0;
	
	while(dest < dest0 + width*height*chan) {
		// Pixels encoded in "packets"
		// First byte is raw/rle flag(upper bit) and count(1-128 as 0-127 in lower 7 bits)
		// If raw, the next count chan-byte color values in the file are taken verbatim
		// If rle, the next single chan-byte color value speaks for the next count pixels
		
		int raw = (*src & 0x80) == 0; // Is this packet raw pixels or a repeating color
		int count = (*src & 0x7f) + 1; // How many raw pixels or color repeats
		src++; // Advance src beyond first byte to 32-bit color
		
		if(dest + count*chan > dest0 + width*height*chan) // prevent from writing out of dest range
			count = (dest0 + width*height*chan - dest) / chan;
		
		for(int j=0; j<count; j++)
		{
			dest[0] = src[2]; // Red
			dest[1] = src[1]; // Green 
			dest[2] = src[0]; // Blue
			
			if(raw) // In raw mode, keep advancing "src" to subsequent values
				src += chan; // In RLE mode, just repeat the packet[1] RGB color
			dest += chan;
		}
		if(!raw) // After outputting count RGBA values, advance "src" beyond color if rle
			src += chan;
	}
	
	assert(dest <= dest0 + width*height*chan);

	return(0);
}

// Decode run-length encoded Targa into 32-bit pixels
// This used to assume that RLE runs don't cross scanline boundaries, but our
// targa encoder in commonGrLib violates this.
// To make origin bottom left images work I put an image flip at the end.
static int decode_rgb_rle32(const unsigned char *src0, unsigned char *dest0,
							const int width, const int height)
{
	int chan = 4;
	const unsigned char *src = src0;
	unsigned char *dest = dest0;
	
	while(dest < dest0 + width*height*chan) {
		// Pixels encoded in "packets"
		// First byte is raw/rle flag(upper bit) and count(1-128 as 0-127 in lower 7 bits)
		// If raw, the next count chan-byte color values in the file are taken verbatim
		// If rle, the next single chan-byte color value speaks for the next count pixels
		
		int raw = (*src & 0x80) == 0; // Is this packet raw pixels or a repeating color
		int count = (*src & 0x7f) + 1; // How many raw pixels or color repeats
		src++; // Advance src beyond first byte to 32-bit color
		
		if(dest + count*chan > dest0 + width*height*chan) // prevent from writing out of dest range
			count = (dest0 + width*height*chan - dest) / chan;
		
		for(int j=0; j<count; j++)
		{
			dest[0] = src[2]; // Red
			dest[1] = src[1]; // Green 
			dest[2] = src[0]; // Blue
			dest[3] = src[3]; // Alpha
			
			if(raw) // In raw mode, keep advancing "src" to subsequent values
				src += chan; // In RLE mode, just repeat the packet[1] RGB color
			dest += chan;
		}
		if(!raw) // After outputting count RGBA values, advance "src" beyond color if rle
			src += chan;
	}
	
	return(0);
}

void VFlip(unsigned char * Pix, int width, int height, int chan)
{
	int lsize = width * chan;
	unsigned char *tbuf = new unsigned char[lsize];
	assert(tbuf);

  if (height == 0) 
    return;

	for(int y=0; y<height/2; y++)
	{
		memcpy(tbuf, &Pix[y*lsize], lsize);
		memcpy(&Pix[y*lsize], &Pix[(height-y-1)*lsize], lsize);
		memcpy(&Pix[(height-y-1)*lsize], tbuf, lsize);
	}
	delete [] tbuf;
}

//----------------------------------------------------------------------------
// READS AN IMAGE IN FROM A TGA FILE. RETURNS THE COLOR RGB ARRAY AND DIMENSIONS
// PERFORMS AUTO-ALLOCATION OF Color ARRAY IF SET TO NULL BEFORE CALLING; OTHERWISE
// ASSUMES THAT COLOR HAS BEEN PRE-ALLOCED.
//----------------------------------------------------------------------------
void LoadTGA(const char *FileName, unsigned char* &Color, int &Width, int &Height, int &Channels)
{
  // Read whole file "FileName" into array.
  std::ifstream InFile(FileName, ios::in | ios::binary | NOCREATE );
  if(!InFile.is_open())
  {
    fprintf(stderr, "TGA: Failed to open file `%s'.\n", FileName );
    Width = Height = 0;
    return;
  }
  
  // Get file size
  InFile.seekg(0, ios::end);
  int fsize = InFile.tellg();
  InFile.seekg(0, ios::beg);
  
  // Load file from disk 
  unsigned char *fdata = new unsigned char[fsize];
  if(fdata == NULL)
  {
    fprintf(stderr, "TGA: Failed to allocate temp memory to load file `%s'.\n", FileName );
    Width = Height = 0;
    return;
  }
  InFile.read((char *)fdata, fsize);
  
  // Check that expected data length was loaded
  if(InFile.gcount() != fsize)
  {
    fprintf(stderr, "TGA: Didn't get the right amount of data.\n");
    InFile.close();
    delete [] fdata;
    Width = Height = 0;
    return;
  }
  
  InFile.close();
  
  // Decode the contents of fdata.
  TGA_Header *header = (TGA_Header *)fdata; // Header starts at first byte of file
  unsigned char *color_map = fdata + sizeof(TGA_Header) + header->ImageIDLength;
  int cmapsize = header->CoMapType ? (header->CoSize / 8) *
    ((header->Length_hi << 8) | header->Length_lo) : 0;	
  unsigned char *encoded_pixels = color_map + cmapsize;
  
  char itype_names[16][16] = {"NULL", "MAP", "RGB", "MONO", "4", "5", "6", "7", "8",
    "RLE-MAP", "RLE-RGB", "RLE-MONO", "12", "13", "14", "15"};

#if TGA_VERBOSE
  fprintf(stderr, 
          "TGA Image type %c  bpp = %d\n", 
          itype_names[0xf & header->ImgType], int(header->PixelSize));
#endif
  
  if((header->Desc & TGA_DESC_ORG_MASK) != TGA_ORG_TOP_LEFT &&
    (header->Desc & TGA_DESC_ORG_MASK) != TGA_ORG_BOTTOM_LEFT)
  {
    fprintf(stderr, "TGA: Not top/bottom left origin: image desc %c\n", header->Desc );
    delete [] fdata;
    Width = Height = 0;
    return;
  }
  
  Width = ((header->Width_hi) << 8) | header->Width_lo;
  Height = ((header->Height_hi) << 8) | header->Height_lo;
  int size = Width * Height;
  int chan = header->PixelSize / 8; // 1, 3 or 4
  if(chan == 2) chan = 3; // 16-bit means R5G6B5.
  if(header->ImgType == TGA_MAP || header->ImgType == TGA_RLEMAP)
    chan = header->CoSize / 8;
  int dsize = size * chan;
  
  if (Color==NULL) Color = new unsigned char[dsize];
  if (Color==NULL)
  {
    fprintf(stderr, "TGA: Could not allocate internal memory for TGA file read %s\n", FileName);
    delete [] fdata;
    Width = Height = 0;
    return;
  }
  
  
  switch(header->ImgType)
  {
  case TGA_MAP:
    if(header->PixelSize == 8)
      decode_map8(encoded_pixels, Color, size, chan, color_map);
    else
    {
      fprintf(stderr, "TGA: Bad color mapped index size: %d bits/pixel.\n", (int)(header->PixelSize));
      delete [] fdata;
      Width = Height = 0;
      return;
    }
    break;
  case TGA_RLEMAP:
    if(header->PixelSize == 8)
      decode_map_rle8(encoded_pixels, Color, Width, Height, chan, color_map);
    else
    {
      fprintf(stderr, "TGA: Bad color mapped rle index size: %d bits/pixel.\n", (int)(header->PixelSize));
      delete [] fdata;
      Width = Height = 0;
      return;
    }
    break;
  case TGA_MONO:
    if(header->PixelSize == 8)
      memcpy(Color, encoded_pixels, dsize);
    else
    {
      fprintf(stderr, "TGA: Bad pixel size: %d bits/pixel\n", (int)(header->PixelSize));
      delete [] fdata;
      Width = Height = 0;
      return;
    }
    break;
  case TGA_RGB:
    switch(header->PixelSize)
    {
    case 16:
      decode_rgb16(encoded_pixels, Color, Width, Height);
      break;
    case 24:
      decode_rgb24(encoded_pixels, Color, Width, Height);
      break;
    case 32:
      decode_rgb32(encoded_pixels, Color, Width, Height);
      break;
    default:
      fprintf(stderr, "TGA: Bad pixel size: %d bits/pixel\n", (int)(header->PixelSize));
      delete [] fdata;
      Width = Height = 0;
      return;
    }
    break;
    case TGA_RLERGB:
      switch(header->PixelSize)
      {
      case 16:
        decode_rgb_rle16(encoded_pixels, Color, Width, Height);
        break;
      case 24:
        decode_rgb_rle24(encoded_pixels, Color, Width, Height);
        break;
      case 32:
        decode_rgb_rle32(encoded_pixels, Color, Width, Height);
        break;
      default:
        fprintf(stderr, "TGA: Bad pixel size: %d bits/pixel\n", (int)(header->PixelSize));
        delete [] fdata;
        Width = Height = 0;
        return;
      }
      break;
    default:
        fprintf(stderr, 
                "TGA: Image type %c bpp = %d.\n", 
                itype_names[0xf & header->ImgType], 
                int(header->PixelSize));
        delete [] fdata;
        Width = Height = 0;
        return;
  }
  
  // Flip it so that origin is always top left.
  // With Open GL this is backwards logic for some reason.
  if((header->Desc & TGA_DESC_ORG_MASK) == TGA_ORG_BOTTOM_LEFT)
    VFlip(Color, Width, Height, chan);
  
  delete [] fdata;
  
  Channels = chan;

  return;
}



//============================================================================
//============================================================================
// Helper functions for WRITE
//============================================================================

inline bool colors_equal(const unsigned char *a, const unsigned char *b, const int chan)
{
	bool same = true;
	for(int i=0; i<chan; i++)
		same = same && (*a++ == *b++);

	return same;
}

// Look-ahead for run-length encoding....
// Scan through *src characters
// If src[0] and src[1] are the same, return 0 and set count:
// how many match src[0] (including src[0] itself)
// If src[0] and src[1] differ, return 1 and set count: how many total
// non-consecutive values are there(not counting value which repeats)
// Thus: AAAAAAAB returns 0 with count = 7
// ABCDEFGG returns 1 with count = 6
// Never scan more than 128 ahead.
static bool match(const unsigned char *src, int &count, const int chan)
{
	const unsigned char *prev_color;
	count = 0;
	
	if(colors_equal(src, src+chan, chan))
	{
		// RLE
		prev_color = src;
		while(colors_equal(src, prev_color, chan) && count < 128)
		{
			src += chan;
			count++;
		}
		return false;
	}
	else
	{
		// Raw
		prev_color = src;
		src += chan;
		while(!colors_equal(src, prev_color, chan) && count < 128)
		{
			count++;
			prev_color = src;
			src += chan;
		}
		return true;
	}
}

static int encode_rle(const unsigned char *src0, unsigned char *dest,
					  const int size, const int chan)
{
	const unsigned char *src = src0;
	int dp = 0;
	
	do
	{
		int count;
		bool raw = match(src, count, chan);
		if(count > 128) count = 128;
		
		if(raw)
		{
			dest[dp++] = count - 1;
			if(chan == 4) {
				for(int i=0; i<count; i++)
				{
					dest[dp++] = src[2]; // Blue
					dest[dp++] = src[1]; // Green
					dest[dp++] = src[0]; // Red
					dest[dp++] = src[3]; // Alpha
					src += chan;
				}
			} else if(chan == 3) {
				for(int i=0; i<count; i++)
				{
					dest[dp++] = src[2]; // Blue
					dest[dp++] = src[1]; // Green
					dest[dp++] = src[0]; // Red
					src += chan;
				}
			} else if(chan == 1) {
				for(int i=0; i<count; i++)
				{
					dest[dp++] = src[0]; // Intensity
					src += chan;
				}
			}
		}
		else
		{
			dest[dp++] = (count - 1) | 0x80; // The RLE flag.
			if(chan == 1)
			{
				dest[dp++] = src[0]; // Intensity
			}
			else 
			{
				dest[dp++] = src[2]; // Blue
				dest[dp++] = src[1]; // Green
				dest[dp++] = src[0]; // Red
				if(chan == 4)
					dest[dp++] = src[3]; // Alpha
			}
			src += chan * count;
		}
	}
	while(src <= src0 + size * chan);
	
	return dp;
}

//----------------------------------------------------------------------------
// Writes an unsigned byte RGB color array out to a PPM file.
//----------------------------------------------------------------------------
void WriteTGA(const char *FileName, unsigned char* Color, int Width, int Height, int channels)
{
	if(Width > 65535 || Width <= 0 || Height > 65535 || Height <= 0)
	{
		cerr << "TGA: Can not write TGA file of dimentions" << Width << "x" << Height << " too big or empty.\n";
		return;
	}
	
  int size = Width*Height;
  int chan = channels;
  int dsize = size*chan;

	FILE *ft = fopen(FileName, "wb");
	if(ft == NULL)
	{
		cerr << "TGA: Failed to open file `" << FileName << "' for writing.\n";
		return;
	}
	
	TGA_Header header;	
	header.ImageIDLength = 0;
	header.CoMapType = 0; // no colormap
	header.Index_lo = header.Index_hi = 0; // no colormap
	header.Length_lo = header.Length_hi = header.CoSize = 0; // no colormap
	header.X_org_lo = header.X_org_hi = header.Y_org_lo = header.Y_org_hi = 0; // 0,0 origin
	header.Width_lo = Width;
	header.Width_hi = Width >> 8;
	header.Height_lo = Height;
	header.Height_hi = Height >> 8;
	header.Desc = TGA_ORG_TOP_LEFT;

	switch(chan)
	{
	case 1:
		header.ImgType = TGA_MONO;
		header.PixelSize = 8;
		break;
	case 3:
		header.ImgType = TGA_RLERGB;
		header.PixelSize = 24;
		break;
	case 4:
		header.ImgType = TGA_RLERGB;
		header.PixelSize = 32;
		header.Desc |= 8; // This many alpha bits.
		break;
	default:
		cerr << "TGA: Cannot save file of " << chan << " channels.\n";
		return;
	}
	
	fwrite(&header, sizeof(header), 1, ft);
	
	unsigned char *out_data = new unsigned char[dsize + size / 128 + 1];
	assert(out_data);

	int rle_size = encode_rle(Color, out_data, size, chan);
	
	fwrite(out_data, rle_size, 1, ft);

	fclose(ft);

  return;
}
