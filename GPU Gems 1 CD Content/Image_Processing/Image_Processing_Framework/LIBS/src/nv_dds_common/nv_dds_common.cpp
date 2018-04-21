/*********************************************************************NVMH4****
Path:  NVSDK\Common\src\nv_dds_common
File:  nv_dds_common.cpp

Copyright NVIDIA Corporation 2002
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.



Comments:

dds loader.



******************************************************************************/
///////////////////////////////////////////////////////////////////////////////
//
// Description:
// 
// Loads DDS images (DXTC1, DXTC3, DXTC5, RGB (888, 888X), and RGBA (8888) are
// supported) for use in OpenGL. Image is flipped when its loaded as DX images
// are stored with different coordinate system. If file has mipmaps and/or 
// cubemaps then these are loaded as well. Volume textures can be loaded as 
// well but they must be uncompressed.
//
// When multiple textures are loaded (i.e a volume or cubemap texture), 
// additional faces can be accessed using the array operator. 
//
// The mipmaps for each face are also stored in a list and can be accessed like 
// so: image.get_mipmap() (which accesses the first mipmap of the first 
// image). To get the number of mipmaps call the get_num_mipmaps function for
// a given texture.
//
// Call the is_volume() or is_cubemap() function to check that a loaded image
// is a volume or cubemap texture respectively. If a volume texture is loaded
// then the get_depth() function should return a number greater than 1. 
// Mipmapped volume textures and DXTC compressed volume textures are supported.
//
///////////////////////////////////////////////////////////////////////////////
//
// Update: 23/10/2002
//
// CMAUGHAN
// A branch from nv_dds to nv_dds_common to make a common file that doens't have the
// API specific stuff in, so that DX8/9 can use this file's useful capability of
// loading dds files into memory.

#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include "nv_dds_common\nv_dds_common.h"

using namespace std;
using namespace nv_dds_common;

///////////////////////////////////////////////////////////////////////////////
// CDDSImage public functions

///////////////////////////////////////////////////////////////////////////////
// default constructor
CDDSImage::CDDSImage()
  : format(0),
    components(0),
    compressed(false),
    cubemap(false),
    volume(false),
    valid(false)
{
}

CDDSImage::~CDDSImage()
{
}

///////////////////////////////////////////////////////////////////////////////
// loads DDS image
//
// filename - fully qualified name of DDS image
// flipImage - specifies whether image is flipped on load, default is true
bool CDDSImage::load(std::string filename, bool flipImage)
{
	DDS_HEADER ddsh;
	char filecode[4];
	FILE *fp;
    int width, height, depth;
    int (CDDSImage::*sizefunc)(int, int);

    // clear any previously loaded images
    clear();
    
    // open file
	fp = fopen(filename.data(), "rb");
	if (fp == NULL)
		return false;

    // read in file marker, make sure its a DDS file
	fread(filecode, 1, 4, fp);
	if (strncmp(filecode, "DDS ", 4) != 0)
	{
		fclose(fp);
		return false;
	}

    // read in DDS header
	fread(&ddsh, sizeof(ddsh), 1, fp);

    // check if image is a cubempa
    if (ddsh.dwCaps2 & DDS_CUBEMAP)
        cubemap = true;

    // check if image is a volume texture
    if ((ddsh.dwCaps2 & DDS_VOLUME) && (ddsh.dwDepth > 0))
        volume = true;

    // figure out what the image format is
    if (ddsh.ddspf.dwFlags & DDS_FOURCC) 
    {
		d3dformat = ddsh.ddspf.dwFourCC;
	    switch(ddsh.ddspf.dwFourCC)
	    {
            case FOURCC_DXT1:
                format = FORMAT_DXT1;
				components = 3;
                compressed = true;
                break;
            case FOURCC_DXT3:
                format = FORMAT_DXT3;
                components = 4;
                compressed = true;
                break;
            case FOURCC_DXT5:
                format = FORMAT_DXT5;
                components = 4;
                compressed = true;
                break;
            default:
                fclose(fp);
                return false;
	    }
    }
    else if (ddsh.ddspf.dwFlags == DDS_RGBA && ddsh.ddspf.dwRGBBitCount == 32)
    {
        format = FORMAT_RGBA; 
		d3dformat = NV_D3DFMT_A8R8G8B8;
        compressed = false;
        components = 4;
    }
    else if (ddsh.ddspf.dwFlags == DDS_RGB  && ddsh.ddspf.dwRGBBitCount == 32)
    {
        format = FORMAT_RGBA; 
		d3dformat = NV_D3DFMT_X8R8G8B8;
        compressed = false;
        components = 4;
    }
    else if (ddsh.ddspf.dwFlags == DDS_RGB  && ddsh.ddspf.dwRGBBitCount == 24)
    {
        format = FORMAT_RGB; 
		d3dformat = NV_D3DFMT_R8G8B8;
        compressed = false;
        components = 3;
    }
	else if (ddsh.ddspf.dwRGBBitCount == 8)
	{
		format = FORMAT_L8; 
		d3dformat = NV_D3DFMT_L8;
		compressed = false;
		components = 1;
	}
    else 
    {
        fclose(fp);
        return false;
    }
    
    // store primary surface width/height/depth
    width = ddsh.dwWidth;
    height = ddsh.dwHeight;
    depth = clamp_size(ddsh.dwDepth);   // set to 1 if 0
    
    // use correct size calculation function depending on whether image is 
    // compressed
    sizefunc = (compressed ? &CDDSImage::size_dxtc : &CDDSImage::size_rgb);

    // load all surfaces for the image (6 surfaces for cubemaps)
    for (int n = 0; n < (cubemap ? 6 : 1); n++)
    {
        int size; 

        // calculate surface size
        size = (this->*sizefunc)(width, height)*depth;

        // load surface
        CTexture img(width, height, depth, size);
        fread(img, 1, img.size, fp);

        align_memory(&img);
        
        if (flipImage)
            flip(img, img.width, img.height, img.depth, img.size);
        
        int w = clamp_size(width >> 1);
        int h = clamp_size(height >> 1);
        int d = clamp_size(depth >> 1); 

        // store number of mipmaps
        int numMipmaps = ddsh.dwMipMapCount;

        // number of mipmaps in file includes main surface so decrease count 
        // by one
        if (numMipmaps != 0)
            numMipmaps--;

        // load all mipmaps for current surface
        for (int i = 0; i < numMipmaps && (w || h); i++)
        {
            // calculate mipmap size
            size = (this->*sizefunc)(w, h)*d;

            CSurface mipmap(w, h, d, size);
            fread(mipmap, 1, mipmap.size, fp);
            
            if (flipImage)
            {
                flip(mipmap, mipmap.width, mipmap.height, mipmap.depth, 
                    mipmap.size);
            }

            img.mipmaps.push_back(mipmap);

            // shrink to next power of 2
            w = clamp_size(w >> 1);
            h = clamp_size(h >> 1);
            d = clamp_size(d >> 1); 
        }

        images.push_back(img);
    }

    // swap cubemaps on y axis (since image is flipped in OGL)
    if (cubemap && flipImage)
    {
        CTexture tmp;
        tmp = images[3];
        images[3] = images[2];
        images[2] = tmp;
    }
    
    fclose(fp);

    valid = true;

	return true;
}

///////////////////////////////////////////////////////////////////////////////
// free image memory
void CDDSImage::clear()
{
    components = 0;
    format = 0;
    compressed = false;
    cubemap = false;
    volume = false;
    valid = false;

    images.clear();
}

///////////////////////////////////////////////////////////////////////////////
// returns individual texture when multiple textures are loaded (as is the case
// with volume textures and cubemaps)
CTexture &CDDSImage::operator[](int index)
{ 
    // make sure an image has been loaded
    assert(valid);
    assert(index < (int)images.size());

    return images[index]; 
}

///////////////////////////////////////////////////////////////////////////////
// returns pointer to main image
CDDSImage::operator char*()
{ 
    assert(valid);

    return images[0]; 
}

///////////////////////////////////////////////////////////////////////////////
// clamps input size to [1-size]
inline int CDDSImage::clamp_size(int size)
{
    if (size <= 0)
        size = 1;

    return size;
}

///////////////////////////////////////////////////////////////////////////////
// CDDSImage private functions
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// calculates 4-byte aligned width of image
inline int CDDSImage::get_line_width(int width, int bpp)
{
    return ((width * bpp + 31) & -32) >> 3;
}

///////////////////////////////////////////////////////////////////////////////
// calculates size of DXTC texture in bytes
inline int CDDSImage::size_dxtc(int width, int height)
{
    return ((width+3)/4)*((height+3)/4)*
        (format == FORMAT_DXT1 ? 8 : 16);   
}

///////////////////////////////////////////////////////////////////////////////
// calculates size of uncompressed RGB texture in bytes
inline int CDDSImage::size_rgb(int width, int height)
{
    return width*height*components;
}

///////////////////////////////////////////////////////////////////////////////
// align to 4 byte boundary (add pad bytes to end of each line in the image)
void CDDSImage::align_memory(CTexture *surface)
{
    // don't bother with compressed images, volume textures, or cubemaps
    if (compressed || volume || cubemap)
        return;

    // calculate new image size
    int linesize = get_line_width(surface->width, components*8);
    int imagesize = linesize*surface->height;

    // exit if already aligned
    if (surface->size == imagesize)
        return;

    // create new image of new size
    CTexture newSurface(surface->width, surface->height, surface->depth, 
        imagesize);

    // add pad bytes to end of each line
    char *srcimage = (char*)*surface;
    char *dstimage = (char*)newSurface;
    for (int n = 0; n < surface->depth; n++)
    {
        char *curline = srcimage;
        char *newline = dstimage;

        int imsize = surface->size / surface->depth;
        int lnsize = imsize / surface->height;
        
        for (int i = 0; i < surface->height; i++)
        {
            memcpy(newline, curline, lnsize);
            newline += linesize;
            curline += lnsize;
        }
    }

    // save padded image
    *surface = newSurface;
}

///////////////////////////////////////////////////////////////////////////////
// flip image around X axis
void CDDSImage::flip(char *image, int width, int height, int depth, int size)
{
    int linesize;
    int offset;

    if (!compressed)
    {
        assert(depth > 0);

        int imagesize = size/depth;
        linesize = imagesize / height;

        for (int n = 0; n < depth; n++)
        {
            offset = imagesize*n;
            char *top = image + offset;
            char *bottom = top + (imagesize-linesize);
    
            for (int i = 0; i < (height >> 1); i++)
            {
                swap(bottom, top, linesize);

                top += linesize;
                bottom -= linesize;
            }
        }
    }
    else
    {
        void (CDDSImage::*flipblocks)(DXTColBlock*, int);
    	int xblocks = width / 4;
    	int yblocks = height / 4;
        int blocksize;

        switch (format)
        {
            case FORMAT_DXT1: 
                blocksize = 8;
                flipblocks = &CDDSImage::flip_blocks_dxtc1; 
                break;
            case FORMAT_DXT3: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc3; 
                break;
            case FORMAT_DXT5: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc5; 
                break;
            default:
                return;
        }

        linesize = xblocks * blocksize;

    	DXTColBlock *top;
    	DXTColBlock *bottom;
    
	    for (int j = 0; j < (yblocks >> 1); j++)
	    {
    		top = (DXTColBlock*)(image + j * linesize);
    		bottom = (DXTColBlock*)(image + (((yblocks-j)-1) * linesize));

            (this->*flipblocks)(top, xblocks);
            (this->*flipblocks)(bottom, xblocks);

            swap(bottom, top, linesize);
        }
    }
}    

///////////////////////////////////////////////////////////////////////////////
// swap to sections of memory
void CDDSImage::swap(void *byte1, void *byte2, int size)
{
    unsigned char *tmp = new unsigned char[size];

    memcpy(tmp, byte1, size);
    memcpy(byte1, byte2, size);
    memcpy(byte2, tmp, size);

    delete [] tmp;
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT1 color block
void CDDSImage::flip_blocks_dxtc1(DXTColBlock *line, int numBlocks)
{
    DXTColBlock *curblock = line;

    for (int i = 0; i < numBlocks; i++)
    {
        swap(&curblock->row[0], &curblock->row[3], sizeof(unsigned char));
        swap(&curblock->row[1], &curblock->row[2], sizeof(unsigned char));

        curblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT3 color block
void CDDSImage::flip_blocks_dxtc3(DXTColBlock *line, int numBlocks)
{
    DXTColBlock *curblock = line;
    DXT3AlphaBlock *alphablock;

    for (int i = 0; i < numBlocks; i++)
    {
        alphablock = (DXT3AlphaBlock*)curblock;

        swap(&alphablock->row[0], &alphablock->row[3], sizeof(unsigned short));
        swap(&alphablock->row[1], &alphablock->row[2], sizeof(unsigned short));

        curblock++;

        swap(&curblock->row[0], &curblock->row[3], sizeof(unsigned char));
        swap(&curblock->row[1], &curblock->row[2], sizeof(unsigned char));

        curblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT5 alpha block
void CDDSImage::flip_dxt5_alpha(DXT5AlphaBlock *block)
{
    unsigned char gBits[4][4];
    
	const unsigned long mask = 0x00000007;		    // bits = 00 00 01 11
	unsigned long bits = 0;
    memcpy(&bits, &block->row[0], sizeof(unsigned char) * 3);
    
	gBits[0][0] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[0][1] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[0][2] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[0][3] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[1][0] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[1][1] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[1][2] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[1][3] = (unsigned char)(bits & mask);

    bits = 0;
    memcpy(&bits, &block->row[3], sizeof(unsigned char) * 3);

	gBits[2][0] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[2][1] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[2][2] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[2][3] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[3][0] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[3][1] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[3][2] = (unsigned char)(bits & mask);
	bits >>= 3;
	gBits[3][3] = (unsigned char)(bits & mask);

    unsigned long *pBits = ((unsigned long*) &(block->row[0]));

    *pBits &= 0xff000000;

    *pBits = *pBits | (gBits[3][0] << 0);
    *pBits = *pBits | (gBits[3][1] << 3);
    *pBits = *pBits | (gBits[3][2] << 6);
    *pBits = *pBits | (gBits[3][3] << 9);

    *pBits = *pBits | (gBits[2][0] << 12);
    *pBits = *pBits | (gBits[2][1] << 15);
    *pBits = *pBits | (gBits[2][2] << 18);
    *pBits = *pBits | (gBits[2][3] << 21);

	pBits = ((unsigned long*) &(block->row[3]));

    *pBits &= 0xff000000;

    *pBits = *pBits | (gBits[1][0] << 0);
    *pBits = *pBits | (gBits[1][1] << 3);
    *pBits = *pBits | (gBits[1][2] << 6);
    *pBits = *pBits | (gBits[1][3] << 9);

    *pBits = *pBits | (gBits[0][0] << 12);
    *pBits = *pBits | (gBits[0][1] << 15);
    *pBits = *pBits | (gBits[0][2] << 18);
    *pBits = *pBits | (gBits[0][3] << 21);
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT5 color block
void CDDSImage::flip_blocks_dxtc5(DXTColBlock *line, int numBlocks)
{
    DXTColBlock *curblock = line;
    DXT5AlphaBlock *alphablock;
    
    for (int i = 0; i < numBlocks; i++)
    {
        alphablock = (DXT5AlphaBlock*)curblock;
        
        flip_dxt5_alpha(alphablock);

        curblock++;

        swap(&curblock->row[0], &curblock->row[3], sizeof(unsigned char));
        swap(&curblock->row[1], &curblock->row[2], sizeof(unsigned char));

        curblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// CTexture implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// default constructor
CTexture::CTexture()
  : CSurface()  // initialize base class part
{
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty texture
CTexture::CTexture(int w, int h, int d, int imgSize)
  : CSurface(w, h, d, imgSize)  // initialize base class part
{
}

///////////////////////////////////////////////////////////////////////////////
// copy constructor
CTexture::CTexture(const CTexture &copy)
  : CSurface(copy)
{
    for (unsigned int i = 0; i < copy.mipmaps.size(); i++)
        mipmaps.push_back(copy.mipmaps[i]);
}

///////////////////////////////////////////////////////////////////////////////
// assignment operator
CTexture &CTexture::operator= (const CTexture &rhs)
{
    if (this != &rhs)
    {
        CSurface::operator = (rhs);

        mipmaps.clear();
        for (unsigned int i = 0; i < rhs.mipmaps.size(); i++)
        {
            mipmaps.push_back(rhs.mipmaps[i]);
        }
    }

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// clean up texture memory
CTexture::~CTexture()
{
    mipmaps.clear();
}

///////////////////////////////////////////////////////////////////////////////
// CSurface implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// default constructor
CSurface::CSurface()
  : width(0),
    height(0),
    depth(0),
    size(0),
    pixels(NULL)
{
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty image
CSurface::CSurface(int w, int h, int d, int imgsize)
{
    pixels = NULL;
    create(w, h, d, imgsize);
}

///////////////////////////////////////////////////////////////////////////////
// copy constructor
CSurface::CSurface(const CSurface &copy)
  : width(0),
    height(0),
    depth(0),
    size(0),
    pixels(NULL)
{

    if (copy.pixels)
    {
        size = copy.size;
        width = copy.width;
        height = copy.height;
        depth = copy.depth;
        pixels = new char[size];
        memcpy(pixels, copy.pixels, copy.size);
    }
}

///////////////////////////////////////////////////////////////////////////////
// assignment operator
CSurface &CSurface::operator= (const CSurface &rhs)
{
    if (this != &rhs)
    {
        clear();

        if (rhs.pixels)
        {
            size = rhs.size;
            width = rhs.width;
            height = rhs.height;
            depth = rhs.depth;

            pixels = new char[size];
            memcpy(pixels, rhs.pixels, size);
        }
    }

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// clean up image memory
CSurface::~CSurface()
{
    clear();
}

///////////////////////////////////////////////////////////////////////////////
// returns a pointer to image
CSurface::operator char*()
{ 
    return pixels; 
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty image
void CSurface::create(int w, int h, int d, int imgsize)
{
    clear();

    width = w;
    height = h;
    depth = d;
    size = imgsize;
    pixels = new char[imgsize];
}

///////////////////////////////////////////////////////////////////////////////
// free surface memory
void CSurface::clear()
{
    delete [] pixels;
    pixels = NULL;
}
