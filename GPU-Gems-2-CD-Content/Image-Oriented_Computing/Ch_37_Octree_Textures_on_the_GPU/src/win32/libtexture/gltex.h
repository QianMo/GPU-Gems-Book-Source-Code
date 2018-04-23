#ifndef __GLTEX__
#define __GLTEX__

#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include "CTexture.h"

namespace gltex
{

	/**
	Creates an OpenGL texture id from a given texture object.
	*/
	inline GLuint gltexIdFromTexture(CTexture *tex)
	{
		unsigned int id;

		glGenTextures(1,&id);
		glBindTexture(GL_TEXTURE_2D,id);
		if (tex->isAlpha())
			gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, tex->getWidth(), tex->getHeight(), 
			GL_RGBA, GL_UNSIGNED_BYTE, tex->getData());
		else
			gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, tex->getWidth(), tex->getHeight(), 
			GL_RGB, GL_UNSIGNED_BYTE, tex->getData());
		return (id);
	}

	/** 
	Load a texture and return an OpenGL texture id (using gluBuild2DMipmaps).
	*/
	inline GLuint gltexLoadTexture(const char *n)
	{
		unsigned int  id;
		CTexture     *tex=CTexture::loadTexture(n);
		id=gltexIdFromTexture(tex);
		delete (tex);
		return (id);
	}

	/**
	Update the content of an OpenGL texture with a given texture object.
	*/
	inline void gltexUpdateTextureId(GLuint id,CTexture *tex)
	{
		glBindTexture(GL_TEXTURE_2D,id);
		if (tex->isAlpha())
			gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, tex->getWidth(), tex->getHeight(), 
			GL_RGBA, GL_UNSIGNED_BYTE, tex->getData());
		else
			gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, tex->getWidth(), tex->getHeight(), 
			GL_RGB, GL_UNSIGNED_BYTE, tex->getData());
	}

  /**
	Retrieve an OpenGL texture as a texture object. Usefull for saving
	dynamic textures.
	*/
	inline CTexture *gltexTextureFromId(GLuint id,GLenum format=GL_RGB)
	{
		int            w,h;
		unsigned char *pixels;
		CTexture      *tex;

		glBindTexture(GL_TEXTURE_2D,id);
		glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&w);
		glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&h);
		pixels = new unsigned char [w*h*(format==GL_RGBA?4:3)];
		glGetTexImage(GL_TEXTURE_2D,0,format,GL_UNSIGNED_BYTE,pixels);
		tex=new CTexture("gltexTextureFromId",w,h,(format==GL_RGBA),pixels);
		tex->setDataOwner(true);
		return (tex);
	}

	/**
	Retrieve an OpenGL color buffer as a texture object. Usefull for screen capture.
	*/
	inline CTexture *gltexTextureFromScreen(GLenum format=GL_RGB)
	{
		GLint          v[4];
		unsigned char *pixels;
		CTexture      *tex;

		glGetIntegerv(GL_VIEWPORT,v);
		pixels = new unsigned char [v[2]*v[3]*(format==GL_RGBA?4:3)];
		glReadPixels(v[0],v[1],v[2],v[3],format,GL_UNSIGNED_BYTE,pixels);
		tex=new CTexture("gltexTextureFromScreen",v[2],v[3],(format==GL_RGBA),pixels);
		tex->setDataOwner(true);
		return (tex);    
	}

	/**
	Retrieve an OpenGL depth buffer as a texture object.
	*/
	inline CTexture *gltexTextureFromDepth()
	{
		GLint          v[4];
		int            i,j,w,h;
		unsigned char *depth,*rgb;
		CTexture      *tex;

		glGetIntegerv(GL_VIEWPORT,v);
		depth = new unsigned char [v[2]*v[3]];
		rgb=new unsigned char[v[2]*v[3]*3];
		glReadPixels(v[0],v[1],v[2],v[3],
			GL_DEPTH_COMPONENT,GL_UNSIGNED_BYTE,depth);
		w=v[2];
		h=v[3];
		for (i=0;i<w;i++)
		{
			for (j=0;j<h;j++)
			{
				rgb[(i+(h-j-1)*w)*3  ]=depth[i+(h-j-1)*w];
				rgb[(i+(h-j-1)*w)*3+1]=depth[i+(h-j-1)*w];
				rgb[(i+(h-j-1)*w)*3+2]=depth[i+(h-j-1)*w];
			}
		}
		tex=new CTexture("gltexTextureFromDepth",w,h,false,rgb);
		delete (depth);
		tex->setDataOwner(true);
		return (tex);    
	}

	/**
	Retrieve an OpenGL stencil buffer as a texture object.
	*/
	inline CTexture *gltexTextureFromStencil()
	{
		GLint          v[4];
		int            i,j,w,h;
		unsigned char *stencil,*rgb;
		CTexture      *tex;

		glGetIntegerv(GL_VIEWPORT,v);
		stencil = new unsigned char [v[2]*v[3]];
		rgb=new unsigned char[v[2]*v[3]*3];
		glReadPixels(v[0],v[1],v[2],v[3],
			GL_STENCIL_INDEX,GL_UNSIGNED_BYTE,stencil);
		w=v[2];
		h=v[3];
		for (i=0;i<w;i++)
		{
			for (j=0;j<h;j++)
			{
				rgb[(i+(h-j-1)*w)*3  ]=stencil[i+(h-j-1)*w];
				rgb[(i+(h-j-1)*w)*3+1]=stencil[i+(h-j-1)*w];
				rgb[(i+(h-j-1)*w)*3+2]=stencil[i+(h-j-1)*w];
			}
		}
		tex=new CTexture("gltexTextureFromStencil",w,h,false,rgb);
		delete (stencil);
		tex->setDataOwner(true);
		return (tex);    
	}


	/**
	Convert an OpenGL texture to a normal texture.
	*/
	inline GLuint gltexConvertToNormal(GLuint id,double scale,double norme=1.0)
	{
		CTexture *tex=gltexTextureFromId(id,GL_RGB);
		tex->convertToNormal(scale,norme);
		GLuint glid=gltex::gltexIdFromTexture(tex);
		delete (tex);
		return (glid);
	}
};

#endif
