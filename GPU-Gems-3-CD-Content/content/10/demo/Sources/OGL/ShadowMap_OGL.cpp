#include "../Framework/Common.h"
#include "Application_OGL.h"
#include "../Framework/DemoSetup.h"
#include "../Framework/Camera.h"
#include "ShadowMap_OGL.h"

extern float g_fSplitPos[10];
extern Camera g_Camera;

ShadowMap_OGL::ShadowMap_OGL()
{
  m_iTexture = 0;
  m_iFrameBuffer = 0;
  m_strInfo[0] = 0;
  m_iArraySize = 0;
}

ShadowMap_OGL::~ShadowMap_OGL()
{
  Destroy();
}

bool ShadowMap_OGL::Create(int &iSize)
{
  // create the texture
  glGenTextures (1, &m_iTexture);
  if(m_iTexture == 0)
  {
    MessageBox(NULL, TEXT("Creating shadow map texture failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  glBindTexture (GL_TEXTURE_2D, m_iTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, iSize, iSize, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE_ARB);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  float pBorderColor[]={1.0f, 1.0f, 1.0f, 1.0f};
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, pBorderColor);

  // create frame buffer and attach texture to it
  glGenFramebuffersEXT (1, &m_iFrameBuffer);
  if(m_iFrameBuffer == 0)
  {
    MessageBox(NULL, TEXT("Creating frame buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, m_iFrameBuffer);
  glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, m_iTexture, 0);
  glDrawBuffer(GL_FALSE);
  glReadBuffer(GL_FALSE);

  // verify
  GLenum status = glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT);
  if(status == GL_FRAMEBUFFER_UNSUPPORTED_EXT)
  {
    MessageBox(NULL, TEXT("Frame buffer configuration not supported!"), TEXT("Error!"), MB_OK);
    return false;
  } else if(status != GL_FRAMEBUFFER_COMPLETE_EXT) {
    MessageBox(NULL, TEXT("Frame buffer not successfully created!"), TEXT("Error!"), MB_OK);
    return false;
  }

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  m_iSize = iSize;
  m_iArraySize = 1;

  _snprintf(m_strInfo, 1024, "Depth %i²", m_iSize);
  return true;
}

bool ShadowMap_OGL::CreateAsTextureArray(int iSize, int iArraySize)
{
  // create the texture
  glGenTextures (1, &m_iTexture);
  if(m_iTexture == 0)
  {
    MessageBox(NULL, TEXT("Creating shadow map texture failed!"), TEXT("Error!"), MB_OK);
    return false;
  }
  glBindTexture (GL_TEXTURE_2D_ARRAY_EXT, m_iTexture);
  glTexImage3D (GL_TEXTURE_2D_ARRAY_EXT, 0, GL_DEPTH_COMPONENT24, iSize, iSize, iArraySize, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
  glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_DEPTH_TO_TEXTURE_EXT);
  glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  float pBorderColor[]={1.0f, 1.0f, 1.0f, 1.0f};
  glTexParameterfv(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_BORDER_COLOR, pBorderColor);

  // create fbo and attach texture to it
  glGenFramebuffersEXT (1, &m_iFrameBuffer);
  if(m_iFrameBuffer == 0)
  {
    MessageBox(NULL, TEXT("Creating frame buffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, m_iFrameBuffer);
  glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, m_iTexture, 0);
  glDrawBuffer(GL_FALSE);
  glReadBuffer(GL_FALSE);

  // verify
  GLenum status = glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT);
  if(status == GL_FRAMEBUFFER_UNSUPPORTED_EXT)
  {
    MessageBox(NULL, TEXT("Frame buffer configuration not supported!"), TEXT("Error!"), MB_OK);
    return false;
  } else if(status != GL_FRAMEBUFFER_COMPLETE_EXT) {
    MessageBox(NULL, TEXT("Frame buffer not successfully created!"), TEXT("Error!"), MB_OK);
    return false;
  }

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  m_iSize = iSize;
  m_iArraySize = iArraySize;

  char *strType = "Depth24/TextureArray";
  _snprintf(m_strInfo, 1024, "%s, %i²x%i", strType, m_iSize, m_iArraySize);
  return true;
}

void ShadowMap_OGL::Destroy(void)
{
  if(m_iFrameBuffer != 0)
  {
    glDeleteFramebuffersEXT(1, &m_iFrameBuffer);
    m_iFrameBuffer = 0;
  }

  if(m_iTexture != 0)
  {
    glDeleteTextures(1, &m_iTexture);
    m_iTexture = 0;
  }

  m_iSize = 0;
  m_iArraySize = 0;
}

void ShadowMap_OGL::EnableRendering(void)
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_iFrameBuffer);

  glPushAttrib(GL_VIEWPORT_BIT);
  glViewport(0, 0, m_iSize, m_iSize);
	glColorMask(0, 0, 0, 0);
}

void ShadowMap_OGL::DisableRendering(void)
{
  glColorMask(1, 1, 1, 1);
  glPopAttrib();

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

int ShadowMap_OGL::GetMemoryInMB(void)
{
  return m_iArraySize * m_iSize * m_iSize * 4 / 1048576;
}

void ShadowMap_OGL::Bind(void)
{
  if(m_iArraySize == 1) glBindTexture(GL_TEXTURE_2D, m_iTexture);
  else glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, m_iTexture);
}

void ShadowMap_OGL::Unbind(void)
{
  if(m_iArraySize == 1) glBindTexture(GL_TEXTURE_2D, 0);
  else glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, 0);
}