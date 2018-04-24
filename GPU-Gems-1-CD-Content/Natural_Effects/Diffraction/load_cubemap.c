/*
  =============================================================================
   load_cubemap.c --- code to load a cube map from 6 BMP files
  -----------------------------------------------------------------------------
   Author : Jos Stam (jstam@alias.com)

   note : this code only works under windows.

  =============================================================================
*/

#include <malloc.h>
#include <stdio.h>
#include <limits.h>

#include <windows.h>

#include "load_cubemap.h"


/*
  -----------------------------------------------------------------------------
   constants
  -----------------------------------------------------------------------------
*/

#define E_OK						1
#define E_FILE_NOT_OPENED			2
#define E_OVERFLOW					3
#define E_NOT_ENOUGH_MEMORY			4
#define E_FILE_FORMAT_NOT_SUPPORTED	5
#define E_FILE_BAD_DATA				6


/*
  -----------------------------------------------------------------------------
   load data from a BMP file that is either in RGB or RGBA format
  -----------------------------------------------------------------------------
*/

static int LoadBMP ( FILE * l_fFile, BITMAPINFOHEADER l_bmi, BITMAPFILEHEADER l_bmf, int nbytes,
unsigned char ** in_pucBits,
int * in_piWidth, int * in_piHeight, int * in_piDataLength, int * in_piLineLength )
{
	int error;
	long i, j;
	unsigned long l_ulMemLineLength;
	unsigned long l_ulDataLength;
	unsigned char * l_pBytes;
	unsigned long l_ulFileLineLength;
	unsigned long l_ulFileDataLength;
	unsigned char * p, * buf;

	/* Initial calculations for DWORD alignment. */
	l_ulMemLineLength = l_bmi.biWidth * 3;
	/* Set the file to the start of the data */
	fseek ( l_fFile, l_bmf.bfOffBits, SEEK_SET );
	
	if( l_ulMemLineLength > ULONG_MAX / l_bmi.biHeight ) {
		error = E_OVERFLOW;
		return ( error );
	}
	/* Attempt to allocate memory for image data. */
	l_ulDataLength = l_ulMemLineLength * l_bmi.biHeight;
	*in_pucBits = malloc ( l_ulDataLength );

	if( !(*in_pucBits) ) {
		error = E_NOT_ENOUGH_MEMORY;
		return ( error );
	}
	/* Image constructed properly so continue loading. */
	l_pBytes = *in_pucBits;
	l_ulFileLineLength = l_bmi.biWidth * nbytes;
	l_ulFileDataLength = l_ulFileLineLength * l_bmi.biHeight;
	p = NULL;
	buf = malloc ( l_ulFileDataLength );

	*in_piLineLength = l_ulMemLineLength;
	*in_piDataLength = l_ulDataLength;

	*in_piWidth  = l_bmi.biWidth;
	*in_piHeight = l_bmi.biHeight;

	if( !buf ) {
		error = E_NOT_ENOUGH_MEMORY;
		return ( error );
	}

	// Read entire file.
	fread ( buf, sizeof(unsigned char), l_ulFileDataLength, l_fFile );

	for ( i=0 ; i<l_bmi.biHeight ; i++ ) {
		p = buf + l_ulFileLineLength * i;

		for ( j=0; j<l_bmi.biWidth ; j++ ) {
			l_pBytes[0] = p[2];
			l_pBytes[1] = p[1];
			l_pBytes[2] = p[0];

			l_pBytes += 3;
			p += nbytes;
		}
	}
	free ( buf );

	return ( E_OK );
}


/*
  -----------------------------------------------------------------------------
   load texture data from a BMP file
  -----------------------------------------------------------------------------
*/

static int LoadBMP3 ( const char * in_cFileName, unsigned char ** in_pucBits,
int * in_piWidth,int * in_piHeight,int * in_piDataLength,int * in_piLineLength )
{
	int error = E_OK;
	BITMAPINFOHEADER l_bmi;
	BITMAPFILEHEADER l_bmf;
	unsigned long l_ulBytes;
	FILE * l_fFile;
	unsigned long l_ulMemLineLength;
	unsigned long l_ulDataLength;
	unsigned char * l_pBytes;
	unsigned long l_ulFileLineLength;
	unsigned long l_ulFileDataLength;
	unsigned char * p, * buf;

	if( in_cFileName[0] == '\0' ) {
		return ( E_FILE_BAD_DATA );
	}

	if( *in_pucBits ) {
		free ( *in_pucBits );
		(*in_pucBits) = 0;
	}

	*in_piDataLength	= 0;
	*in_piLineLength	= 0;
	*in_piWidth			= 0;
	*in_piHeight		= 0;

	/* Need image size from file for image init. */
	l_fFile = fopen(in_cFileName,"r");
	if( l_fFile == 0 ) {
		error = E_FILE_NOT_OPENED;
		return ( error );
	}

	fseek ( l_fFile, 0, SEEK_SET );
	l_ulBytes = fread ( &l_bmf, sizeof(char), sizeof(BITMAPFILEHEADER), l_fFile);

	/* check whether this is a bitmap file */
	if( !(l_ulBytes == sizeof(BITMAPFILEHEADER) && 
		  l_bmf.bfType == 0x4d42 /* 'BM' */) ) {
		error = E_FILE_FORMAT_NOT_SUPPORTED;
		goto exit_func;
	}

	/* Now, we  load a BITMAPINFO or BITMAPCOREINFO structure. */
	l_ulBytes = fread(&l_bmi, sizeof(char), 
		sizeof(BITMAPINFOHEADER), l_fFile);

	if( !(l_ulBytes    == sizeof(BITMAPINFOHEADER) &&
		  l_bmi.biSize == sizeof(BITMAPINFOHEADER) &&
		  l_bmi.biWidth > 0 && l_bmi.biHeight > 0) ) {
		error = E_FILE_BAD_DATA;
		goto exit_func;
	}

	if( l_bmi.biBitCount == 32 ) {
		/* file is in RGBA format */
		error = LoadBMP ( l_fFile, l_bmi, l_bmf, 4, in_pucBits, in_piWidth,
			              in_piHeight, in_piDataLength, in_piLineLength );
		if ( error != E_OK ) {
			goto exit_func;
		}
	} else if( l_bmi.biBitCount == 24 ) {
		/* file is in RGB format */
		error = LoadBMP ( l_fFile, l_bmi, l_bmf, 3, in_pucBits, in_piWidth,
			              in_piHeight, in_piDataLength, in_piLineLength );
		if ( error != E_OK ) {
			goto exit_func;
		}
	} else if ( l_bmi.biBitCount == 8 || l_bmi.biBitCount == 4 || l_bmi.biBitCount == 2 ) {
		/* file has a color table */ 
		int ncols, ncmem, nread;
		BYTE * colTable;
		/* Initial calculations for DWORD alignment. */
		l_ulMemLineLength = l_bmi.biWidth * 3;

		if( l_ulMemLineLength > ULONG_MAX / l_bmi.biHeight ) {
			error = E_OVERFLOW;
			goto exit_func;
		}
		ncols = l_bmi.biClrUsed == 0 ? 256 : l_bmi.biClrUsed;
		ncmem = 4*ncols;
		colTable = (BYTE *) malloc ( ncmem * sizeof(BYTE) );
		nread = fread ( colTable, sizeof(BYTE), ncmem, l_fFile );

		/* Attempt to allocate memory for image data. */
		l_ulDataLength = l_ulMemLineLength * l_bmi.biHeight;
		*in_pucBits = malloc ( l_ulDataLength );

		if( !(*in_pucBits && colTable) ) {
			error = E_NOT_ENOUGH_MEMORY;
			if ( colTable ) free ( colTable );
			goto exit_func;
		}

		/* Image constructed properly so continue loading. */
		l_pBytes = *in_pucBits;
		l_ulFileLineLength = l_bmi.biWidth;
		l_ulFileDataLength = l_ulFileLineLength * l_bmi.biHeight;

		p = NULL;
		buf = malloc ( l_ulFileDataLength );

		/* Set the file to the start of the data */
		fseek ( l_fFile, l_bmf.bfOffBits, SEEK_SET );

		*in_piLineLength = l_ulMemLineLength;
		*in_piDataLength = l_ulDataLength;

		*in_piWidth  = l_bmi.biWidth;
		*in_piHeight = l_bmi.biHeight;

		if( buf ) {
			long i, j;
			// Read entire file.
			fread ( buf, sizeof(unsigned char), l_ulFileDataLength, l_fFile );

			for ( i=0; i<l_bmi.biHeight ; i++ ) {
				p = buf + l_ulFileLineLength * i;

				for ( j =0 ; j<l_bmi.biWidth ; j++ ) {
					unsigned char b;
					memcpy( &b, p, 1 );	/* Copy RGB values. */
					l_pBytes[0] = colTable[4*b+2];
					l_pBytes[1] = colTable[4*b+1];
					l_pBytes[2] = colTable[4*b+0];

					l_pBytes += 3;
					p += 1;
				}
			}
			free( buf );
		}
		free ( colTable );
	}

exit_func:

	fclose( l_fFile );

	return ( error );
}


/*
  -----------------------------------------------------------------------------
   call texture loader with the texture data
  -----------------------------------------------------------------------------
*/

static void make_rgb_texture ( GLenum target, unsigned char * data, int width, int height, int mipmapped )
{
	if ( mipmapped ) {
		gluBuild2DMipmaps ( target, GL_RGB8, width, height, GL_RGB, GL_UNSIGNED_BYTE, data );
	} else {
		glTexImage2D ( target, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
	}
}


/*
  -----------------------------------------------------------------------------
   load six texture maps of the cube map
  -----------------------------------------------------------------------------
*/

void load_bmp_cubemap ( const char * string, int mipmap )
{
	char buff[1024];
	unsigned char * data=NULL;
	int w, h, l, ll;
	
	sprintf ( buff, string, "posx" );
	LoadBMP3 ( buff, &data, &w, &h, &l, &ll );
	make_rgb_texture ( GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB, data, w, h, mipmap );

	sprintf ( buff, string, "negx" );
	LoadBMP3 ( buff, &data, &w, &h, &l, &ll );
	make_rgb_texture ( GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB, data, w, h, mipmap );


	sprintf ( buff, string, "posy" );
	LoadBMP3 ( buff, &data, &w, &h, &l, &ll );
	make_rgb_texture ( GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB, data, w, h, mipmap );

	sprintf ( buff, string, "negy" );
	LoadBMP3 ( buff, &data, &w, &h, &l, &ll );
	make_rgb_texture ( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB, data, w, h, mipmap );


	sprintf ( buff, string, "posz" );
	LoadBMP3 ( buff, &data, &w, &h, &l, &ll );
	make_rgb_texture ( GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB, data, w, h, mipmap );

	sprintf ( buff, string, "negz" );
	LoadBMP3 ( buff, &data, &w, &h, &l, &ll );
	make_rgb_texture ( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB, data, w, h, mipmap );


	free ( data );


	glTexParameteri ( GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
}