#pragma once

#include <shared/RenderTexture.h>

/**
 * This class is a fix of the RenderTexture class 
 * to also support WGL_DEPTH_COMPONENT_NV buffer.
 **/
class RenderTextureFix : public RenderTexture
{
public:
	RenderTextureFix(const char *strMode, int iWidth, int iHeight, GLenum target);

	void Bind(int iBuffer=WGL_FRONT_LEFT_ARB);
	void Release(int iBuffer=WGL_FRONT_LEFT_ARB);

private:
    void CreateTexture(GLuint &tex);

	PBuffer *m_pbuffer;
	GLuint m_depth_tex;
	GLenum m_target;
};
