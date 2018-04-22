#ifndef __NV_DDS_H__
#define __NV_DDS_H__

#ifdef WIN32
#  include <windows.h>
#endif

#include <string>
#include <vector>
#include <assert.h>

#include <GL/gl.h>
#include <GL/glext.h>

namespace nv_dds
{
    const unsigned long DDS_FOURCC = 0x00000004;
    const unsigned long DDS_RGB    = 0x00000040;
    const unsigned long DDS_RGBA   = 0x00000041;
    const unsigned long DDS_DEPTH  = 0x00800000;

    const unsigned long DDS_COMPLEX = 0x00000008;
    const unsigned long DDS_CUBEMAP = 0x00000200;
    const unsigned long DDS_VOLUME  = 0x00200000;

    const unsigned long FOURCC_DXT1 = 0x31545844; //(MAKEFOURCC('D','X','T','1'))
    const unsigned long FOURCC_DXT3 = 0x33545844; //(MAKEFOURCC('D','X','T','3'))
    const unsigned long FOURCC_DXT5 = 0x35545844; //(MAKEFOURCC('D','X','T','5'))

    struct DDS_PIXELFORMAT
    {
        unsigned long dwSize;
        unsigned long dwFlags;
        unsigned long dwFourCC;
        unsigned long dwRGBBitCount;
        unsigned long dwRBitMask;
        unsigned long dwGBitMask;
        unsigned long dwBBitMask;
        unsigned long dwABitMask;
    };

    struct DXTColBlock
    {
	    unsigned short col0;
	    unsigned short col1;

	    unsigned char row[4];
    };

    struct DXT3AlphaBlock
    {
	    unsigned short row[4];
    };

    struct DXT5AlphaBlock
    {
	    unsigned char alpha0;
	    unsigned char alpha1;
        
        unsigned char row[6];
    };

    struct DDS_HEADER
    {
        unsigned long dwSize;
        unsigned long dwFlags;
        unsigned long dwHeight;
        unsigned long dwWidth;
        unsigned long dwPitchOrLinearSize;
        unsigned long dwDepth;
        unsigned long dwMipMapCount;
        unsigned long dwReserved1[11];
        DDS_PIXELFORMAT ddspf;
        unsigned long dwCaps1;
        unsigned long dwCaps2;
        unsigned long dwReserved2[3];
    };

    class CSurface
    {
        friend class CTexture;
        friend class CDDSImage;    

        public:
            CSurface();
            CSurface(int w, int h, int d, int imgsize);
            CSurface(const CSurface &copy);
            CSurface &operator= (const CSurface &rhs);
            virtual ~CSurface();

            operator char*();

            void create(int w, int h, int d, int imgsize);
            void clear();

            inline int get_width() { return width; }
            inline int get_height() { return height; }
            inline int get_depth() { return depth; }
            inline int get_size() { return size; }

        protected:
            int width;
            int height;
            int depth;
            int size;

            char *pixels;       
    };

    class CTexture : public CSurface
    {
        friend class CDDSImage;

        public:
            CTexture();
            CTexture(int w, int h, int d, int imgSize);
            CTexture(const CTexture &copy);
            CTexture &operator= (const CTexture &rhs);
            ~CTexture();

            inline CSurface &get_mipmap(int index) 
            { 
                assert(index < (int)mipmaps.size());
                return mipmaps[index]; 
            }

            inline int get_num_mipmaps() { return (int)mipmaps.size(); }
        protected:
			std::vector<CSurface> mipmaps;
    };

    class CDDSImage
    {
	    public:
		    CDDSImage();
		    ~CDDSImage();

            bool load(std::string filename, bool flipImage = true);
            void clear();
            
            operator char*();
            CTexture &operator[](int index);

            bool upload_texture1D();
            bool upload_texture2D(int imageIndex = 0, GLenum target = GL_TEXTURE_2D);
#ifdef GL_NV_texture_rectangle           
            bool upload_textureRectangle();
#endif
            bool upload_texture3D();
            bool upload_textureCubemap();

            inline int get_width() 
            {
                assert(valid);
                assert(images.size() > 0);
                
                return images[0].get_width(); 
            }

            inline int get_height()
            {
                assert(valid);
                assert(images.size() > 0);
                
                return images[0].get_height(); 
            }

            inline int get_depth()
            {
                assert(valid);
                assert(images.size() > 0);
                
                return images[0].get_depth(); 
            }

            inline int get_size()
            {
                assert(valid);
                assert(images.size() > 0);

                return images[0].get_size();
            }

            inline int get_num_mipmaps() 
            { 
                assert(valid);
                assert(images.size() > 0);

                return images[0].get_num_mipmaps(); 
            }

            inline CSurface &get_mipmap(int index) 
            { 
                assert(valid);
                assert(images.size() > 0);
                assert(index < images[0].get_num_mipmaps());

                return images[0].get_mipmap(index); 
            }

            inline int get_components() { return components; }
            inline int get_format() { return format; }

            inline bool is_compressed() { return compressed; }
            inline bool is_cubemap() { return cubemap; }
            inline bool is_volume() { return volume; }
            inline bool is_valid() { return valid; }
            
        private:
            int clamp_size(int size);
            int get_line_width(int width, int bpp);
            int size_dxtc(int width, int height);
            int size_rgb(int width, int height);
            void align_memory(CTexture *surface);

            void flip(char *image, int width, int height, int depth, int size);

            void swap(void *byte1, void *byte2, int size);
            void flip_blocks_dxtc1(DXTColBlock *line, int numBlocks);
            void flip_blocks_dxtc3(DXTColBlock *line, int numBlocks);
            void flip_blocks_dxtc5(DXTColBlock *line, int numBlocks);
            void flip_dxt5_alpha(DXT5AlphaBlock *block);

		    int format;
            int components;
            bool compressed;
            bool cubemap;
            bool volume;
            bool valid;

			std::vector<CTexture> images;

#ifndef MACOS
            static PFNGLTEXIMAGE3DEXTPROC glTexImage3D;
            static PFNGLCOMPRESSEDTEXIMAGE1DARBPROC glCompressedTexImage1DARB;
            static PFNGLCOMPRESSEDTEXIMAGE2DARBPROC glCompressedTexImage2DARB;
            static PFNGLCOMPRESSEDTEXIMAGE3DARBPROC glCompressedTexImage3DARB;
#endif
    };
}
#endif
