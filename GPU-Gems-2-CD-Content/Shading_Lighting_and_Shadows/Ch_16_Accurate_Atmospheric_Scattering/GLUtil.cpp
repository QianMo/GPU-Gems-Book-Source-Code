// GLUtil.cpp
//

#include "Master.h"
#include "GLUtil.h"

CGLUtil g_glUtil;
CGLUtil *CGLUtil::m_pMain = &g_glUtil;

#ifdef USE_CG
CGcontext CShaderObject::m_cgContext;
CGprofile CShaderObject::m_cgVertexProfile;
CGprofile CShaderObject::m_cgFragmentProfile;
#endif


CGLUtil::CGLUtil()
{
	// Start by clearing out all the member variables
}

CGLUtil::~CGLUtil()
{
}

void CGLUtil::Init()
{
	// Start by storing the current HDC and HGLRC
	m_hDC = wglGetCurrentDC();
	m_hGLRC = wglGetCurrentContext();

	m_bATI = strstr((const char *)glGetString(GL_VENDOR), "ATI") != NULL;
	LogInfo((const char *)glGetString(GL_VENDOR));
	LogInfo((const char *)glGetString(GL_RENDERER));
	LogInfo((const char *)glGetString(GL_VERSION));
	LogInfo((const char *)glGetString(GL_EXTENSIONS));

	// Finally, initialize the default rendering context
	InitRenderContext(m_hDC, m_hGLRC);
#ifdef USE_CG
	CShaderObject::InitContext();
#endif
}

void CGLUtil::Cleanup()
{
#ifdef USE_CG
	CShaderObject::ReleaseContext();
#endif
}

void CGLUtil::InitRenderContext(HDC hDC, HGLRC hGLRC)
{
	wglMakeCurrent(hDC, hGLRC);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, CVector4(0.0f));

	wglMakeCurrent(m_hDC, m_hGLRC);
}
