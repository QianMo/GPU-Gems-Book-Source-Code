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
// Update: 6/11/2002
//
// Added some convenience functions to handle uploading textures to OpenGL. The
// following functions have been added:
//
//     bool upload_texture1D();
//     bool upload_texture2D(int imageIndex = 0, GLenum target = GL_TEXTURE_2D);
//     bool upload_textureRectangle();
//     bool upload_texture3D();
//     bool upload_textureCubemap();
//
// See function implementation below for instructions/comments on using each
// function.
//
// The open function has also been updated to take an optional second parameter
// specifying whether the image should be flipped on load. This defaults to 
// true.
//
///////////////////////////////////////////////////////////////////////////////
// Sample usage
///////////////////////////////////////////////////////////////////////////////
//
// Loading a compressed texture:
//
// CDDSImage image;
// GLuint texobj;
//
// image.load("compressed.dds");
// 
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_2D);
// glBindTexture(GL_TEXTURE_2D, texobj);
//
// glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, image.get_format(), 
//     image.get_width(), image.get_height(), 0, image.get_size(), 
//     image);
//
// for (int i = 0; i < image.get_num_mipmaps(); i++)
// {
//     glCompressedTexImage2DARB(GL_TEXTURE_2D, i+1, image.get_format(), 
//         image.get_mipmap(i).get_width(), image.get_mipmap(i).get_height(), 0, 
//         image.get_mipmap(i).get_size(), image.get_mipmap(i));
// } 
///////////////////////////////////////////////////////////////////////////////
// 
// Loading an uncompressed texture:
//
// CDDSImage image;
// GLuint texobj;
//
// image.load("uncompressed.dds");
//
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_2D);
// glBindTexture(GL_TEXTURE_2D, texobj);
//
// glTexImage2D(GL_TEXTURE_2D, 0, image.get_components(), image.get_width(), 
//     image.get_height(), 0, image.get_format(), GL_UNSIGNED_BYTE, image);
//
// for (int i = 0; i < image.get_num_mipmaps(); i++)
// {
//     glTexImage2D(GL_TEXTURE_2D, i+1, image.get_components(), 
//         image.get_mipmap(i).get_width(), image.get_mipmap(i).get_height(), 
//         0, image.get_format(), GL_UNSIGNED_BYTE, image.get_mipmap(i));
// }
//
///////////////////////////////////////////////////////////////////////////////
// 
// Loading an uncompressed cubemap texture:
//
// CDDSImage image;
// GLuint texobj;
// GLenum target;
// 
// image.load("cubemap.dds");
// 
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_CUBE_MAP_ARB);
// glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, texobj);
// 
// for (int n = 0; n < 6; n++)
// {
//     target = GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB+n;
// 
//     glTexImage2D(target, 0, image.get_components(), image[n].get_width(), 
//         image[n].get_height(), 0, image.get_format(), GL_UNSIGNED_BYTE, 
//         image[n]);
// 
//     for (int i = 0; i < image[n].get_num_mipmaps(); i++)
//     {
//         glTexImage2D(target, i+1, image.get_components(), 
//             image[n].get_mipmap(i).get_width(), 
//             image[n].get_mipmap(i).get_height(), 0,
//             image.get_format(), GL_UNSIGNED_BYTE, image[n].get_mipmap(i));
//     }
// }
//
///////////////////////////////////////////////////////////////////////////////
// 
// Loading a volume texture:
//
// CDDSImage image;
// GLuint texobj;
// 
// image.load("volume.dds");
// 
// glGenTextures(1, &texobj);
// glEnable(GL_TEXTURE_3D);
// glBindTexture(GL_TEXTURE_3D, texobj);
// 
// PFNGLTEXIMAGE3DPROC glTexImage3D;
// glTexImage3D(GL_TEXTURE_3D, 0, image.get_components(), image.get_width(), 
//     image.get_height(), image.get_depth(), 0, image.get_format(), 
//     GL_UNSIGNED_BYTE, image);
// 
// for (int i = 0; i < image.get_num_mipmaps(); i++)
// {
//     glTexImage3D(GL_TEXTURE_3D, i+1, image.get_components(), 
//         image[0].get_mipmap(i).get_width(), 
//         image[0].get_mipmap(i).get_height(), 
//         image[0].get_mipmap(i).get_depth(), 0, image.get_format(), 
//         GL_UNSIGNED_BYTE, image[0].get_mipmap(i));
// }

#if defined(WIN32)
#  include <windows.h>
#  define GET_EXT_POINTER(name, type) \
      name = (type)wglGetProcAddress(#name)
#elif defined(UNIX)
#  include <GL/glx.h>
#  define GET_EXT_POINTER(name, type) \
      name = (type)glXGetProcAddressARB((const GLubyte*)#name)
#else
#  define GET_EXT_POINTER(name, type)
#endif

#include <GL/gl.h>
#include <GL/glext.h>
#include <stdio.h>
#include <assert.h>
#include "nv_dds.h"

using namespace std;
using namespace nv_dds;

///////////////////////////////////////////////////////////////////////////////
// static function pointers for uploading 3D textures and compressed 1D, 2D
// and 3D textures.
#ifndef MACOS
PFNGLTEXIMAGE3DEXTPROC CDDSImage::glTexImage3D = NULL;
PFNGLCOMPRESSEDTEXIMAGE1DARBPROC CDDSImage::glCompressedTexImage1DARB = NULL;
PFNGLCOMPRESSEDTEXIMAGE2DARBPROC CDDSImage::glCompressedTexImage2DARB = NULL;
PFNGLCOMPRESSEDTEXIMAGE3DARBPROC CDDSImage::glCompressedTexImage3DARB = NULL;
#endif

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
bool CDDSImage::load(string filename, bool flipImage)
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
	    switch(ddsh.ddspf.dwFourCC)
	    {
            case FOURCC_DXT1:
                format = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
                components = 3;
                compressed = true;
                break;
            case FOURCC_DXT3:
                format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
                components = 4;
                compressed = true;
                break;
            case FOURCC_DXT5:
                format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
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
        format = GL_BGRA_EXT; 
        compressed = false;
        components = 4;
    }
    else if (ddsh.ddspf.dwFlags == DDS_RGB  && ddsh.ddspf.dwRGBBitCount == 32)
    {
        format = GL_BGRA_EXT; 
        compressed = false;
        components = 4;
    }
    else if (ddsh.ddspf.dwFlags == DDS_RGB  && ddsh.ddspf.dwRGBBitCount == 24)
    {
        format = GL_BGR_EXT; 
        compressed = false;
        components = 3;
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
        
        if (!cubemap && flipImage)
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
            
            if (!cubemap && flipImage)
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
// uploads a compressed/uncompressed 1D texture
bool CDDSImage::upload_texture1D()
{
    assert(valid);
    assert(images[0].height == 1);
    assert(images[0].width > 0);
    assert(images[0]);

    if (compressed)
    {
        // get function pointer if needed
        if (glCompressedTexImage1DARB == NULL)
        {
            GET_EXT_POINTER(glCompressedTexImage1DARB, 
                            PFNGLCOMPRESSEDTEXIMAGE1DARBPROC);
        }
        
        if (glCompressedTexImage1DARB == NULL)
            return false;
        
        glCompressedTexImage1DARB(GL_TEXTURE_1D, 0, format, 
            images[0].width, 0, images[0].size, 
            images[0]);
        
        // load all mipmaps
        for (unsigned int i = 0; i < images[0].mipmaps.size(); i++)
        {
            glCompressedTexImage1DARB(GL_TEXTURE_1D, i+1, format, 
                images[0].mipmaps[i].width, 0, 
                images[0].mipmaps[i].size, 
                images[0].mipmaps[i]);
        }
    }
    else
    {
        glTexImage1D(GL_TEXTURE_1D, 0, format, images[0].width, 0,
            format, GL_UNSIGNED_BYTE, images[0]);

        // load all mipmaps
        for (unsigned int i = 0; i < images[0].mipmaps.size(); i++)
        {
            glTexImage1D(GL_TEXTURE_1D, i+1, components, 
                images[0].mipmaps[i].width, 0, format, 
                GL_UNSIGNED_BYTE, images[0].mipmaps[i]);
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// uploads a compressed/uncompressed 2D texture
//
// imageIndex - allows you to optionally specify other loaded surfaces for 2D
//              textures such as a face in a cubemap or a slice in a volume
//
//              default: 0
//
// target     - allows you to optionally specify a different texture target for
//              the 2D texture such as a specific face of a cubemap
//
//              default: GL_TEXTURE_2D
bool CDDSImage::upload_texture2D(int imageIndex, GLenum target)
{
    assert(valid);
    assert(imageIndex >= 0);
    assert(imageIndex < (int)images.size());
    assert(images[imageIndex].height > 0);
    assert(images[imageIndex].width > 0);
    assert(images[imageIndex]);
    assert(target == GL_TEXTURE_2D || 
        (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB && 
         target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB));
    
    if (compressed)
    {
        // load function pointer if needed
        if (glCompressedTexImage2DARB == NULL)
        {
            GET_EXT_POINTER(glCompressedTexImage2DARB, 
                            PFNGLCOMPRESSEDTEXIMAGE2DARBPROC);
        }
        
        if (glCompressedTexImage2DARB == NULL)
            return false;
        
        glCompressedTexImage2DARB(target, 0, format, 
            images[imageIndex].width, images[imageIndex].height, 0, 
            images[imageIndex].size, images[imageIndex]);
        
        // load all mipmaps
        for (unsigned int i = 0; i < images[imageIndex].mipmaps.size(); i++)
        {
            glCompressedTexImage2DARB(target, i+1, format, 
                images[imageIndex].mipmaps[i].width, 
                images[imageIndex].mipmaps[i].height, 0, 
                images[imageIndex].mipmaps[i].size, 
                images[imageIndex].mipmaps[i]);
        }
    }
    else
    {
        glTexImage2D(target, 0, components, images[imageIndex].width, 
            images[imageIndex].height, 0, format, GL_UNSIGNED_BYTE, 
            images[imageIndex]);

        // load all mipmaps
        for (unsigned int i = 0; i < images[imageIndex].mipmaps.size(); i++)
        {
            glTexImage2D(target, i+1, components, 
                images[imageIndex].mipmaps[i].width, 
                images[imageIndex].mipmaps[i].height, 0, format, 
                GL_UNSIGNED_BYTE, images[imageIndex].mipmaps[i]); 
        }
    }
    
    return true;
}

#ifdef GL_NV_texture_rectangle
bool CDDSImage::upload_textureRectangle()
{
    assert(valid);
    assert(images.size() >= 1);

    if (!upload_texture2D(0, GL_TEXTURE_RECTANGLE_NV))
        return false;

    return true;
}
#endif

///////////////////////////////////////////////////////////////////////////////
// uploads a compressed/uncompressed cubemap texture
bool CDDSImage::upload_textureCubemap()
{
    assert(valid);
    assert(cubemap);
    assert(images.size() == 6);

    GLenum target;

    // loop through cubemap faces and load them as 2D textures 
    for (int n = 0; n < 6; n++)
    {
        // specify cubemap face
        target = GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB+n;
        if (!upload_texture2D(n, target))
            return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// uploads a compressed/uncompressed 3D texture
bool CDDSImage::upload_texture3D()
{
    assert(valid);
    assert(volume);
    assert(images[0].depth >= 1);

    if (compressed)
    {
        // retrieve function pointer if needed
        if (glCompressedTexImage3DARB == NULL)
        {
            GET_EXT_POINTER(glCompressedTexImage3DARB, 
                            PFNGLCOMPRESSEDTEXIMAGE3DARBPROC);
        }

        if (glCompressedTexImage3DARB == NULL)
            return false;

        glCompressedTexImage3DARB(GL_TEXTURE_3D, 0, format,  images[0].width, 
            images[0].height, images[0].depth, 0, images[0].size, images[0]);
        
        // load all mipmap volumes
        for (unsigned int i = 0; i < images[0].mipmaps.size(); i++)
        {
            glCompressedTexImage3DARB(GL_TEXTURE_3D, i+1, format, 
                images[0].mipmaps[i].width, images[0].mipmaps[i].height, 
                images[0].depth, 0, images[0].mipmaps[i].size, 
                images[0].mipmaps[i]);
        }
    }
    else
    {
        // retrieve function pointer if needed
        if (glTexImage3D == NULL)
        {
            GET_EXT_POINTER(glTexImage3D, PFNGLTEXIMAGE3DEXTPROC);
        }
    
        if (glTexImage3D == NULL)
            return false;
    
        glTexImage3D(GL_TEXTURE_3D, 0, components, images[0].width, 
            images[0].height, images[0].depth, 0, format, GL_UNSIGNED_BYTE, 
            images[0]);
        
        // load all mipmap volumes
        for (unsigned int i = 0; i < images[0].mipmaps.size(); i++)
        {
            glTexImage3D(GL_TEXTURE_3D, i+1, components, 
                images[0].mipmaps[i].width, images[0].mipmaps[i].height, 
                images[0].mipmaps[i].depth, 0, format, GL_UNSIGNED_BYTE, 
                images[0].mipmaps[i]);
        }
    }
    return true;
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
        (format == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT ? 8 : 16);   
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
            case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: 
                blocksize = 8;
                flipblocks = &CDDSImage::flip_blocks_dxtc1; 
                break;
            case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc3; 
                break;
            case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: 
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
