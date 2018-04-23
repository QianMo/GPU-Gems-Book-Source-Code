#include "RenderTextureFix.h"

RenderTextureFix::RenderTextureFix(const char *strMode, int iWidth, int iHeight, GLenum target)
: RenderTexture(const_cast<char*>(strMode), iWidth, iHeight, target),
  m_depth_tex(0), m_target(target)
{
	// don't look!
	m_pbuffer = reinterpret_cast<PBuffer*&>(*this);
}


void RenderTextureFix::Bind(int iBuffer)
{
	if (iBuffer == WGL_DEPTH_COMPONENT_NV) {
		if (m_depth_tex == 0) {
            CreateTexture(m_depth_tex);
        }
		glBindTexture(m_target, m_depth_tex);
		m_pbuffer->Bind(iBuffer);
	} else {
		RenderTexture::Bind(iBuffer);
	}
}

void RenderTextureFix::Release(int iBuffer) 
{
	if (iBuffer == WGL_DEPTH_COMPONENT_NV) {
		glBindTexture(m_target, m_depth_tex);
		m_pbuffer->Release(iBuffer);
	} else {
		RenderTexture::Release(iBuffer);
	}
}

void RenderTextureFix::CreateTexture(GLuint &tex)
{
	glGenTextures(1, &tex);
	glBindTexture(m_target, tex);
	glTexParameteri(m_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(m_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}
