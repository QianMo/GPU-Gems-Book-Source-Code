// Disable warning for loss of data
#pragma warning( disable : 4244 )  

#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <sys/stat.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//#include "bitmap.h"
#include "texture.h"



texture::texture()
{
}




int texture::loadtex(char *filename,int pident)
{
ident=pident;

name=strdup(filename);
glBindTexture(GL_TEXTURE_2D,ident);
glPixelStorei(GL_UNPACK_ALIGNMENT,1);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);

// This is faster on nvidia GPU's... 40.800 vs. 43.166 fps
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST);

/*glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);*/

FILE *f;
struct stat stbuffer;

stat(filename,&stbuffer);
int size=sqrt((stbuffer.st_size-4)/4);


f=fopen(filename,"rb");
if (f==NULL) return -1;
int wide=size;
int tall=size;

char *rgba=new char[wide*tall*4];
fread(rgba,4,1,f);
fread(rgba,4*wide*tall,1,f);
//glTexImage2D(GL_TEXTURE_2D,0,4,wide,tall,0,GL_RGBA,GL_UNSIGNED_BYTE,rgba);
gluBuild2DMipmaps(GL_TEXTURE_2D,GL_RGBA,wide,tall,GL_RGBA,GL_UNSIGNED_BYTE,rgba);
delete rgba;
// i return the allocated memory for the texture
return (wide*tall*4);
}



int texture::loadtga(char *filename,int pident)
{
ident=pident;

name=strdup(filename);
glBindTexture(GL_TEXTURE_2D,ident);
glPixelStorei(GL_UNPACK_ALIGNMENT,1);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);

// This is faster on nvidia GPU's... 40.800 vs. 43.166 fps
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST);

FILE *f;

f=fopen(filename,"rb");
if (f==NULL) 
	{
	// filename is wrong
	return -1;
	}

unsigned char IDLength,ColorMapType, ImageType;
unsigned short ColorMapStart,ColorMapLength;
unsigned char ColorMapDepth;
unsigned short XOffset, YOffset;
unsigned short Width,Height;
unsigned char Depth,ImageDescriptor;


fread(&IDLength,sizeof(unsigned char),1,f);
fread(&ColorMapType,sizeof(unsigned char),1,f);
fread(&ImageType,sizeof(unsigned char),1,f);
fread(&ColorMapStart,sizeof(unsigned short),1,f);
fread(&ColorMapLength,sizeof(unsigned short),1,f);
fread(&ColorMapDepth,sizeof(unsigned char),1,f);
fread(&XOffset,sizeof(unsigned short),1,f);
fread(&YOffset,sizeof(unsigned short),1,f);
fread(&Width,sizeof(unsigned short),1,f);
fread(&Height,sizeof(unsigned short),1,f);
fread(&Depth,sizeof(unsigned char),1,f);
fread(&ImageDescriptor,sizeof(unsigned char),1,f);
sizex=Width;
sizey=Height;
Depth=Depth/8;	// bits to bytes
depth=Depth;
char *rgba=new char[Width*Height*Depth];
fread(rgba,Width*Height*Depth,1,f);
//glTexImage2D(GL_TEXTURE_2D,0,4,wide,tall,0,GL_RGBA,GL_UNSIGNED_BYTE,rgba);
if (Depth==4)
	gluBuild2DMipmaps(GL_TEXTURE_2D,GL_RGBA,Width,Height,GL_BGRA_EXT,GL_UNSIGNED_BYTE,rgba);
if (Depth==3)
	gluBuild2DMipmaps(GL_TEXTURE_2D,GL_RGB,Width,Height,GL_BGR_EXT,GL_UNSIGNED_BYTE,rgba);
if (Depth==1)
	gluBuild2DMipmaps(GL_TEXTURE_2D,GL_LUMINANCE,Width,Height,GL_LUMINANCE,GL_UNSIGNED_BYTE,rgba);

delete rgba;
// i return the allocated memory for the texture
return (Width*Height*Depth);
}


void texture::use()
{
glBindTexture(GL_TEXTURE_2D,ident);
}


texture::~texture()
{

}


