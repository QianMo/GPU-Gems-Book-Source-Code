#include "shared/pbuffer.h"

// simple wrapper to encapsulate a pbuffer using render-to-texture and its associated texture object
// v1.1 - updated to work with multiple draw buffers

class RenderTexture {
public:
	RenderTexture(char *strMode, int iWidth, int iHeight, GLenum target) : m_target(target)
	{
		m_pbuffer = new PBuffer(strMode);
		m_pbuffer->Initialize(iWidth, iHeight, false, true);

        for(int i=0; i<max_buffers; i++) {
            m_tex[i] = 0;
        }
	};

	~RenderTexture()
	{
		delete m_pbuffer;
        for(int i=0; i<max_buffers; i++) {
            if (m_tex[i]) {
		        glDeleteTextures(1, &m_tex[i]);
            }
        }
	}

	void Activate() { m_pbuffer->Activate(); }
	void Deactivate() { m_pbuffer->Deactivate(); }

	void Bind(int iBuffer=WGL_FRONT_LEFT_ARB)
	{
        int tex_no = iBuffer - WGL_FRONT_LEFT_ARB;
        if (m_tex[tex_no] == 0) {
            // lazily allocate texture objects on demand
            CreateTexture(m_tex[tex_no]);
        }
		glBindTexture(m_target, m_tex[tex_no]);
		m_pbuffer->Bind(iBuffer);
	}

	void Release(int iBuffer=WGL_FRONT_LEFT_ARB) {
		glBindTexture(m_target, m_tex[iBuffer - WGL_FRONT_LEFT_ARB]);
		m_pbuffer->Release(iBuffer);
	}

    inline int GetWidth() { return m_pbuffer->GetWidth(); }
    inline int GetHeight() { return m_pbuffer->GetHeight(); }

private:
    void CreateTexture(GLuint &tex)
    {
		glGenTextures(1, &tex);
		glBindTexture(m_target, tex);
		glTexParameteri(m_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(m_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

    const static int max_buffers = 14;  // WGL_FRONT_LEFT_ARB - WGL_AUX9_ARB
	PBuffer *m_pbuffer;
	GLuint m_tex[max_buffers];          // texture for each buffer
	GLenum m_target;
};
