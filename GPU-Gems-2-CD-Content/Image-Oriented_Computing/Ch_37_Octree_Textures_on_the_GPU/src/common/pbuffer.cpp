/* ----------------------------------------------------------

Octree Textures on the GPU - source code - GPU Gems 2 release
                                                   2004-11-21

Updates on http://www.aracknea.net/octreetex
--
(c) 2004 Sylvain Lefebvre - all rights reserved
--
The source code is provided 'as it is', without any warranties. 
Use at your own risk. The use of any part of the source code in a
commercial or non commercial product without explicit authorisation
from the author is forbidden. Use for research and educational
purposes is allowed and encouraged, provided that a short notice
acknowledges the author's work.
---------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "pbuffer.h"

#ifdef WIN32

#include <glux.h>
  
GLUX_LOAD(GL_NV_float_buffer);
#if defined(WIN32)
GLUX_LOAD(WGL_ARB_pbuffer);
GLUX_LOAD(WGL_ARB_pixel_format);
GLUX_LOAD(WGL_NV_float_buffer);
#elif defined(UNIX)
GLUX_LOAD(GLX_SGIX_pbuffer);
GLUX_LOAD(GLX_SGIX_fbconfig);
// must be done by glux, but there is 
// no GLX_NV_float_buffer extension!
#define GLX_FLOAT_COMPONENTS_NV 0x20B0  
#endif

#endif

#include <string>
#include <vector>

using namespace std;

#if defined(WIN32)

PBuffer::PBuffer(const char *strMode) 
  : m_hDC(0), m_hGLRC(0), m_hPBuffer(0), m_hOldGLRC(0), m_hOldDC(0), 
    m_bIsTexture(false), m_iWidth(0), m_iHeight(0), m_strMode(strMode), 
    m_bSharedContext(false), m_bShareObjects(false), m_bIsBound(false),
    m_bIsActive(false)
{
    m_pfAttribList.push_back(WGL_DRAW_TO_PBUFFER_ARB);
    m_pfAttribList.push_back(true);
    m_pfAttribList.push_back(WGL_SUPPORT_OPENGL_ARB);
    m_pfAttribList.push_back(true);
    m_pfAttribList.push_back(WGL_PIXEL_TYPE_ARB);
    m_pfAttribList.push_back(WGL_TYPE_RGBA_ARB);

    m_pbAttribList.push_back(WGL_PBUFFER_LARGEST_ARB);
    m_pbAttribList.push_back(true);

    m_strMode = strMode;
    parseModeString(m_strMode, &m_pfAttribList, &m_pbAttribList);

    m_pfAttribList.push_back(0);
    m_pbAttribList.push_back(0);
}

PBuffer::~PBuffer()
{
    if (m_hPBuffer)
    {
        if (!m_bSharedContext) wglDeleteContext(m_hGLRC);
        wglReleasePbufferDCARB(m_hPBuffer, m_hDC);
        wglDestroyPbufferARB(m_hPBuffer);
    }
}

// This function actually does the creation of the p-buffer.
// It can only be called once a window has already been created.
bool PBuffer::Initialize(int iWidth, int iHeight, bool bShareContexts, bool bShareObjects)
{
    HDC hdc = wglGetCurrentDC();
    HGLRC hglrc = wglGetCurrentContext();
    int format = 0;
    int nfattribs = 0;
    int niattribs = 0;

    m_iWidth = iWidth;
    m_iHeight = iHeight;
    
    m_bSharedContext = bShareContexts;
    m_bShareObjects = bShareObjects;
    
    if (m_bSharedContext)
    {
        // Get the pixel format for the on-screen window.
        format = GetPixelFormat(hdc);
        if (format == 0)
        {
            fprintf(stderr, "pbuffer creation error:  GetPixelFormat() failed\n");
            return false;
        }
    }
    else
    {
        unsigned int nformats;
        wglChoosePixelFormatARB(hdc, &m_pfAttribList[0], NULL, 1, &format, &nformats);
        if (nformats == 0)
        {
            fprintf(stderr, "pbuffer creation error:  Couldn't find a suitable pixel format.\n");
            return false;
        }
    }
    
    m_hPBuffer = wglCreatePbufferARB(hdc, format, m_iWidth, m_iHeight, &m_pbAttribList[0]);
    if (!m_hPBuffer)
    {
        DWORD err = GetLastError();
        fprintf(stderr, "pbuffer creation error:  wglCreatePbufferARB() failed\n");
        if (err == ERROR_INVALID_PIXEL_FORMAT)
        {
            fprintf(stderr, "error:  ERROR_INVALID_PIXEL_FORMAT\n");
        }
        else if (err == ERROR_NO_SYSTEM_RESOURCES)
        {
            fprintf(stderr, "error:  ERROR_NO_SYSTEM_RESOURCES\n");
        }
        else if (err == ERROR_INVALID_DATA)
        {
            fprintf(stderr, "error:  ERROR_INVALID_DATA\n");
        }
        
        return false;
    }
    
    // Get the device context.
    m_hDC = wglGetPbufferDCARB(m_hPBuffer);
    if (!m_hDC)
    {
        fprintf(stderr, "pbuffer creation error:  wglGetPbufferDCARB() failed\n");
        return false;
    }
    
    if (m_bSharedContext)
    {
        // Let's use the same gl context..
        // Since the device contexts are compatible (i.e. same pixelformat),
        // we should be able to use the same gl rendering context.
        m_hGLRC = hglrc;
    }
    else
    {
        // Create a new gl context for the p-buffer.
        m_hGLRC = wglCreateContext(m_hDC);
        if (!m_hGLRC)
        {
            fprintf(stderr, "pbuffer creation error:  wglCreateContext() failed\n");
            return false;
        }
        
        if(m_bShareObjects)
        {
            if(!wglShareLists(hglrc, m_hGLRC))
            {
                fprintf(stderr, "pbuffer: wglShareLists() failed\n");
                return false;
            }
        }
    }
    
    GLint texFormat = WGL_NO_TEXTURE_ARB;
    wglQueryPbufferARB(m_hPBuffer, WGL_TEXTURE_FORMAT_ARB, &texFormat);

    if (texFormat != WGL_NO_TEXTURE_ARB)
        m_bIsTexture = true;

    // Determine the actual width and height we were able to create.
    wglQueryPbufferARB(m_hPBuffer, WGL_PBUFFER_WIDTH_ARB, &m_iWidth);
    wglQueryPbufferARB(m_hPBuffer, WGL_PBUFFER_HEIGHT_ARB, &m_iHeight);
    
    fprintf(stderr, "Created a %d x %d pbuffer\n", m_iWidth, m_iHeight);

    return true;
}

void PBuffer::parseModeString(const char *modeString, vector<int> *pfAttribList, vector<int> *pbAttribList)
{
    if (!modeString || strcmp(modeString, "") == 0)
        return;
    
    bool bIsFloatBuffer = false; 
    bool bIsTexture = false;
    bool bNeedAlpha = false;

    char *mode = strdup(modeString);

    vector<string> tokens;
    char *buf = strtok(mode, " ");
    while (buf != NULL)
    {
        if (strstr(buf, "float") != NULL)
            bIsFloatBuffer = true;

        if (strstr(buf, "texture") != NULL)
            bIsTexture = true;

        if (strstr(buf, "alpha") != NULL)
            bNeedAlpha = true;

        tokens.push_back(buf);
        buf = strtok(NULL, " ");
    }

    for (unsigned int i = 0; i < tokens.size(); i++)
    {
        string token = tokens[i];

        if (token == "rgb" && !bIsFloatBuffer)
        {
            pfAttribList->push_back(WGL_RED_BITS_ARB);
            pfAttribList->push_back(8);
            pfAttribList->push_back(WGL_GREEN_BITS_ARB);
            pfAttribList->push_back(8);
            pfAttribList->push_back(WGL_BLUE_BITS_ARB);
            pfAttribList->push_back(8);

            continue;
        }
        
        if (token == "alpha" && !bIsFloatBuffer)
        {
            pfAttribList->push_back(WGL_ALPHA_BITS_ARB);
            pfAttribList->push_back(8);

            continue;
        }

        if (token.find("depth") == 0)
        {
            pfAttribList->push_back(WGL_DEPTH_BITS_ARB);
            pfAttribList->push_back(getIntegerValue(token));
            
            continue;
        }

        if (token.find("stencil") == 0)
        {
            pfAttribList->push_back(WGL_STENCIL_BITS_ARB);
            pfAttribList->push_back(8);
            
            continue;
        }

        if (token.find("samples") == 0)
        {
            pfAttribList->push_back(WGL_SAMPLE_BUFFERS_ARB);
            pfAttribList->push_back(1);
            pfAttribList->push_back(WGL_SAMPLES_ARB);
            pfAttribList->push_back(getIntegerValue(token));
            
            continue;
        }

        if (token == "double")
        {
            pfAttribList->push_back(WGL_DOUBLE_BUFFER_ARB);
            pfAttribList->push_back(true);

            continue;
        }        

        if (token.find("float") == 0)
        {
            int precision = getIntegerValue(token);
            pfAttribList->push_back(WGL_RED_BITS_ARB);
            pfAttribList->push_back(precision);
            pfAttribList->push_back(WGL_GREEN_BITS_ARB);
            pfAttribList->push_back(precision);
            pfAttribList->push_back(WGL_BLUE_BITS_ARB);
            pfAttribList->push_back(precision);

            if (bNeedAlpha)
            {
                pfAttribList->push_back(WGL_ALPHA_BITS_ARB);
                pfAttribList->push_back(precision);
            }
            
            pfAttribList->push_back(WGL_FLOAT_COMPONENTS_NV);
            pfAttribList->push_back(true);

            continue;
        }
        
        if (token.find("texture") == 0)
        {
            if (token.find("textureRECT") == 0 || bIsFloatBuffer)
            {
                pbAttribList->push_back(WGL_TEXTURE_TARGET_ARB);
                pbAttribList->push_back(WGL_TEXTURE_RECTANGLE_NV);
            }
            else if (token.find("textureCUBE") == 0)
            {
                pbAttribList->push_back(WGL_TEXTURE_TARGET_ARB);
                pbAttribList->push_back(WGL_TEXTURE_CUBE_MAP_ARB);
            }
            else
            {
                pbAttribList->push_back(WGL_TEXTURE_TARGET_ARB);
                pbAttribList->push_back(WGL_TEXTURE_2D_ARB);
            }

            if (bIsFloatBuffer)
            {
                pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV);
                pfAttribList->push_back(true);
    
                pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
                pbAttribList->push_back(WGL_TEXTURE_FLOAT_RGBA_NV);
            } 
            else
            {
                pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RGBA_ARB);
                pfAttribList->push_back(true);
                
                pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
                pbAttribList->push_back(WGL_TEXTURE_RGBA_ARB);
            }
                
            string option = getStringValue(token);
            if (option == "depth")
            {
                pfAttribList->push_back(WGL_BIND_TO_TEXTURE_DEPTH_NV);
                pfAttribList->push_back(true);
                
                pbAttribList->push_back(WGL_DEPTH_TEXTURE_FORMAT_NV);
                pbAttribList->push_back(WGL_TEXTURE_DEPTH_COMPONENT_NV);
            }

            continue;
        }

        if (token.find("mipmap") == 0 && bIsTexture)
        {
            pfAttribList->push_back(WGL_MIPMAP_TEXTURE_ARB);
            pfAttribList->push_back(true);
            
            continue;
        }

        fprintf(stderr, "unknown pbuffer attribute: %s\n", token.c_str());
    }
}

// Check to see if the pbuffer was lost.
// If it was lost, destroy it and then recreate it.
void PBuffer::HandleModeSwitch()
{
    int lost = 0;
    
    wglQueryPbufferARB(m_hPBuffer, WGL_PBUFFER_LOST_ARB, &lost);
    
    if (lost)
    {
        this->~PBuffer();
        Initialize(m_iWidth, m_iHeight, m_bSharedContext, m_bShareObjects);
    }
}

int PBuffer::Bind(int iBuffer)
{
    if (!m_bIsTexture)
    {
        fprintf(stderr, "PBuffer::Bind() failed - pbuffer format does not support render to texture!\n");
        return 0;
    }

    if (m_bIsBound)
    {
        fprintf(stderr, "PBuffer::Bind() failed - pbuffer is already bound.\n");
        return 0;
    }

    int ret = wglBindTexImageARB(m_hPBuffer, iBuffer);
    if (!ret)
        fprintf(stderr, "PBuffer::Bind() failed.\n");
    
    m_bIsBound = true;

    return ret;
}

int PBuffer::Release(int iBuffer)
{
    if (!m_bIsTexture)
    {
        fprintf(stderr, "PBuffer::Release() failed - pbuffer format does not support render to texture!\n");
        return 0;
    }

    if (!m_bIsBound)
    {
        fprintf(stderr, "PBuffer::Release() failed - pbuffer is not bound.\n");
        return 0;
    }

    int ret = wglReleaseTexImageARB(m_hPBuffer, iBuffer);
    if (!ret)
        fprintf(stderr, "PBuffer::Release() failed.\n");
    
    m_bIsBound = false;
    
    return ret;
}

void PBuffer::Activate()
{
    if (m_bIsActive)
        return;

    m_hOldGLRC = wglGetCurrentContext();
    m_hOldDC = wglGetCurrentDC();

    if (!wglMakeCurrent(m_hDC, m_hGLRC))
        fprintf(stderr, "PBuffer::Activate() failed.\n");

    m_bIsActive = true;
}

void PBuffer::Deactivate()
{
    if (!m_bIsActive)
        return;
    
    if (!wglMakeCurrent(m_hOldDC, m_hOldGLRC))
        fprintf(stderr, "PBuffer::Deactivate() failed.\n");

    m_hOldGLRC = 0;
    m_hOldDC = 0;
    m_bIsActive = false;
}

#elif defined(UNIX)

PBuffer::PBuffer(const char *strMode) 
  : m_pDisplay(0), m_glxPbuffer(0), m_glxContext(0), m_pOldDisplay(0), m_glxOldDrawable(0), 
    m_glxOldContext(0), m_iWidth(0), m_iHeight(0), m_strMode(strMode), 
    m_bSharedContext(false), m_bShareObjects(false)
{
    m_pfAttribList.push_back(GLX_DRAWABLE_TYPE);
    m_pfAttribList.push_back(GLX_PBUFFER_BIT);
    m_pfAttribList.push_back(GLX_RENDER_TYPE);
    m_pfAttribList.push_back(GLX_RGBA_BIT);

    m_pbAttribList.push_back(GLX_LARGEST_PBUFFER);
    m_pbAttribList.push_back(true);
    m_pbAttribList.push_back(GLX_PRESERVED_CONTENTS);
    m_pbAttribList.push_back(true);

    m_strMode = strMode;
    parseModeString(m_strMode, &m_pfAttribList, &m_pbAttribList);

    m_pfAttribList.push_back(0);
    m_pbAttribList.push_back(0);
}    

PBuffer::~PBuffer()
{
    if (m_glxContext)
        if (!m_bSharedContext) glXDestroyContext(m_pDisplay, m_glxContext);

    if (m_glxPbuffer)
        glXDestroyGLXPbufferSGIX(m_pDisplay, m_glxPbuffer);        
}

bool PBuffer::Initialize(int iWidth, int iHeight, bool bShareContexts, bool bShareObjects)
{
    Display *pDisplay = glXGetCurrentDisplay();
    int iScreen = DefaultScreen(pDisplay);
    GLXContext glxContext = glXGetCurrentContext();
    
    GLXFBConfig *glxConfig;
    int iConfigCount;   
    
    m_bSharedContext = bShareContexts;
    m_bShareObjects = bShareObjects;
    
    m_iWidth = iWidth;
    m_iHeight = iHeight;
    
    if (m_bSharedContext)
    {
        glxConfig = glXGetFBConfigs(pDisplay, iScreen, &iConfigCount);
        if (!glxConfig)
        {
            fprintf(stderr, "pbuffer creation error:  glXGetFBConfigs() failed\n");
            return false;
        }
    }  
    else
    {
        glxConfig = glXChooseFBConfigSGIX(pDisplay, iScreen, &m_pfAttribList[0], &iConfigCount);
        if (!glxConfig)
        {
            fprintf(stderr, "pbuffer creation error:  glXChooseFBConfig() failed\n");
            return false;
        }
    }
    
    m_glxPbuffer = glXCreateGLXPbufferSGIX(pDisplay, glxConfig[0], m_iWidth, m_iHeight, &m_pbAttribList[0]);
    
    if (!m_glxPbuffer)
    {
        fprintf(stderr, "pbuffer creation error:  glXCreatePbuffer() failed\n");
        return false;
    }
    
    if (m_bSharedContext)
    {
        m_glxContext = glxContext;
    }
    else
    {
        if (m_bShareObjects)
            m_glxContext = glXCreateContextWithConfigSGIX(pDisplay, glxConfig[0], GLX_RGBA_TYPE, glxContext, true);
        else
            m_glxContext = glXCreateContextWithConfigSGIX(pDisplay, glxConfig[0], GLX_RGBA_TYPE, NULL, true);
        
        if (!glxConfig)
        {
            fprintf(stderr, "pbuffer creation error:  glXCreateNewContext() failed\n");
            return false;
        }
    }
    
    m_pDisplay = pDisplay;
    
    unsigned int w, h;
    w = h = 0;
    
    glXQueryGLXPbufferSGIX(m_pDisplay, m_glxPbuffer, GLX_WIDTH, &w);
    glXQueryGLXPbufferSGIX(m_pDisplay, m_glxPbuffer, GLX_HEIGHT, &h);
    m_iWidth = w;
    m_iHeight = h;
    
    fprintf(stderr, "Created a %d x %d pbuffer\n", m_iWidth, m_iHeight);
    
    return true;
}

void PBuffer::parseModeString(const char *modeString, vector<int> *pfAttribList, vector<int> *pbAttribList)
{
    pbAttribList = pbAttribList;
  
    if (!modeString || strcmp(modeString, "") == 0)
        return;

    bool bIsFloatBuffer = false; 
    bool bNeedAlpha = false;
    
    char *mode = strdup(modeString);

    vector<string> tokens;
    char *buf = strtok(mode, " ");
    while (buf != NULL)
    {
        if (strstr(buf, "float") != NULL)
            bIsFloatBuffer = true;

        if (strstr(buf, "alpha") != NULL)
            bNeedAlpha = true;
        
        tokens.push_back(buf);
        buf = strtok(NULL, " ");
    }
    
    for (unsigned int i = 0; i < tokens.size(); i++)
    {
        string token = tokens[i];

        if (token == "rgb" && !bIsFloatBuffer)
        {
            pfAttribList->push_back(GLX_RED_SIZE);
            pfAttribList->push_back(8);
            pfAttribList->push_back(GLX_GREEN_SIZE);
            pfAttribList->push_back(8);
            pfAttribList->push_back(GLX_BLUE_SIZE);
            pfAttribList->push_back(8);

            continue;
        }

        if (token.find("alpha") != token.npos)
        {
            pfAttribList->push_back(GLX_ALPHA_SIZE);
            pfAttribList->push_back(getIntegerValue(token));
            
            continue;
        }

        if (token.find("depth") != token.npos)
        {
            pfAttribList->push_back(GLX_DEPTH_SIZE);
            pfAttribList->push_back(getIntegerValue(token));
            
            continue;
        }

        if (token.find("stencil") != token.npos)
        {
            pfAttribList->push_back(GLX_STENCIL_SIZE);
            pfAttribList->push_back(getIntegerValue(token));
            
            continue;
        }
      
        if (token.find("samples") != token.npos)
        {
            pfAttribList->push_back(GLX_SAMPLE_BUFFERS_ARB);
            pfAttribList->push_back(1);
            pfAttribList->push_back(GLX_SAMPLES_ARB);
            pfAttribList->push_back(getIntegerValue(token));
            
            continue;
        }

        if (token == "double")
        {
            pfAttribList->push_back(GLX_DOUBLEBUFFER);
            pfAttribList->push_back(true);

            continue;
        }        
        
        if (token.find("float") == 0)
        {
            int precision = getIntegerValue(token);
            pfAttribList->push_back(GLX_RED_SIZE);
            pfAttribList->push_back(precision);
            pfAttribList->push_back(GLX_GREEN_SIZE);
            pfAttribList->push_back(precision);
            pfAttribList->push_back(GLX_BLUE_SIZE);
            pfAttribList->push_back(precision);

            if (bNeedAlpha)
            {
                pfAttribList->push_back(GLX_ALPHA_SIZE);
                pfAttribList->push_back(precision);
            }
            
            pfAttribList->push_back(GLX_FLOAT_COMPONENTS_NV);
            pfAttribList->push_back(true);

            continue;
        }        

        fprintf(stderr, "unknown pbuffer attribute: %s\n", token.c_str());
    }
}

void PBuffer::Activate()
{
    m_pOldDisplay = glXGetCurrentDisplay();
    m_glxOldDrawable = glXGetCurrentDrawable();
    m_glxOldContext = glXGetCurrentContext();

    if (!glXMakeCurrent(m_pDisplay, m_glxPbuffer, m_glxContext))
        fprintf(stderr, "PBuffer::Activate() failed.\n");
}

void PBuffer::Deactivate()
{
    if (!glXMakeCurrent(m_pOldDisplay, m_glxOldDrawable, m_glxOldContext))
        fprintf(stderr, "PBuffer::Deactivate() failed.\n");

    m_pOldDisplay = 0;
    m_glxOldDrawable = 0;
    m_glxOldContext = 0;
}

#endif

string PBuffer::getStringValue(string token)
{
    unsigned int pos;
    if ((pos = token.find("=")) != token.npos)
    {
        string value = token.substr(pos+1, token.length()-pos+1);
        return value;
    }
    else
        return "";
}

int PBuffer::getIntegerValue(string token)
{
    unsigned int pos;
    if ((pos = token.find("=")) != token.npos)
    {
        string value = token.substr(pos+1, token.length()-pos+1);
        if (value.empty())
            return 1;
        return atoi(value.c_str());
    }
    else
        return 1;
}
