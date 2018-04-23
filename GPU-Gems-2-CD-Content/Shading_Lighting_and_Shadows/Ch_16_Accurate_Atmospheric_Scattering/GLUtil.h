// GLUtil.h
//

#ifndef __GLUtil_h__
#define __GLUtil_h__

#define BUFFER_OFFSET(i) ((char *) NULL + (i))

#include "Texture.h"


class CGLUtil
{
// Attributes
protected:
	// Standard OpenGL members
	HDC m_hDC;
	HGLRC m_hGLRC;
	bool m_bATI;

	// Members for GL_ARB_vertex_buffer_object
	unsigned int m_nVertexBuffer;

public:
	static CGLUtil *m_pMain;

// Operations
public:
	CGLUtil();
	~CGLUtil();
	void Init();
	void Cleanup();
	void InitRenderContext(HDC hDC=NULL, HGLRC hGLRC=NULL);

	HDC GetHDC()					{ return m_hDC; }
	HGLRC GetHGLRC()				{ return m_hGLRC; }
	void MakeCurrent()				{ wglMakeCurrent(m_hDC, m_hGLRC); }
	bool IsATI()					{ return m_bATI; }

	void BeginOrtho2D(int nWidth=640, int nHeight=480)
	{
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		gluOrtho2D(0, nWidth, 0, nHeight);
	}
	void EndOrtho2D()
	{
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		glEnable(GL_LIGHTING);
		glEnable(GL_DEPTH_TEST);
	}
};

inline CGLUtil *GLUtil()			{ return CGLUtil::m_pMain; }

#endif // __GLUtil_h__


#ifndef __ShaderObject_h__
#define __ShaderObject_h__

#define USE_CG
#ifdef USE_CG
#include <Cg\cg.h>
#include <Cg\cgGL.h>

class CShaderObject
{
protected:
	static CGcontext m_cgContext;
	static CGprofile m_cgVertexProfile, m_cgFragmentProfile;

	CGprogram m_cgVertexProgram;
	CGprogram m_cgFragmentProgram;

public:
	static void HandleCgError()
	{
		LogError("Cg error: %s", cgGetErrorString(cgGetError()));
	}
	static void InitContext()
	{
		cgSetErrorCallback(HandleCgError);
		m_cgContext = cgCreateContext();

		if(cgGLIsProfileSupported(CG_PROFILE_VP30))
		{
			m_cgVertexProfile = CG_PROFILE_VP30;
			LogInfo("Cg vertex profile: vp30");
		}
		else if(cgGLIsProfileSupported(CG_PROFILE_ARBVP1))
		{
			m_cgVertexProfile = CG_PROFILE_ARBVP1;
			LogInfo("Cg vertex profile: arbvp1");
		}
		else if (cgGLIsProfileSupported(CG_PROFILE_VP20))
		{
			m_cgVertexProfile = CG_PROFILE_VP20;
			LogInfo("Cg vertex profile: vp20");
		}
		else
			LogError("No Cg vertex shader profiles are supported on this system");

		if (cgGLIsProfileSupported(CG_PROFILE_FP30))
		{
			m_cgFragmentProfile = CG_PROFILE_FP30;
			LogInfo("Cg fragment profile: fp30");
		}
		else if(cgGLIsProfileSupported(CG_PROFILE_ARBFP1))
		{
			m_cgFragmentProfile = CG_PROFILE_ARBFP1;
			LogInfo("Cg fragment profile: arbfp1");
		}
		else if (cgGLIsProfileSupported(CG_PROFILE_FP20))
		{
			m_cgFragmentProfile = CG_PROFILE_FP20;
			LogInfo("Cg fragment profile: fp20");
		}
		else
			LogError("No Cg fragment shader profiles are supported on this system");
	}

	static void ReleaseContext()
	{
		cgDestroyContext(m_cgContext);
	}

	CShaderObject()
	{
		m_cgVertexProgram = NULL;
		m_cgFragmentProgram = NULL;
	}
	CShaderObject(const char *pszPath)
	{
		Load(pszPath);
	}
	~CShaderObject()
	{
		Unload();
	}

	void CheckError()
	{
		CGerror err = cgGetError();
		if(err != CG_NO_ERROR)
			LogError("CG error: %s", cgGetErrorString(err));
	}

	void Load(const char *pszPath, const char *pszPath2=NULL)
	{
		char szPath[_MAX_PATH];

		sprintf(szPath, "%sCg.vert", pszPath);
		m_cgVertexProgram = cgCreateProgramFromFile(m_cgContext, CG_SOURCE, szPath, m_cgVertexProfile, NULL, NULL);
		if(cgIsProgram(m_cgVertexProgram))
		{
			if(!cgIsProgramCompiled(m_cgVertexProgram)) 
				cgCompileProgram(m_cgVertexProgram);
			LogInfo(cgGetProgramString(m_cgVertexProgram, CG_COMPILED_PROGRAM));
			cgGLEnableProfile(m_cgVertexProfile);
			cgGLLoadProgram(m_cgVertexProgram);
			cgGLDisableProfile(m_cgVertexProfile);
		}

		sprintf(szPath, "%sCg.frag", pszPath2 ? pszPath2 : pszPath);
		m_cgFragmentProgram = cgCreateProgramFromFile(m_cgContext, CG_SOURCE, szPath, m_cgFragmentProfile, NULL, NULL);
		if(cgIsProgram(m_cgFragmentProgram))
		{
			if(!cgIsProgramCompiled(m_cgFragmentProgram)) 
				cgCompileProgram(m_cgFragmentProgram);
			LogInfo(cgGetProgramString(m_cgFragmentProgram, CG_COMPILED_PROGRAM));
			cgGLEnableProfile(m_cgFragmentProfile);
			cgGLLoadProgram(m_cgFragmentProgram);
			cgGLDisableProfile(m_cgFragmentProfile);
		}
	}

	void Unload()
	{
		if(cgIsProgram(m_cgVertexProgram))
		{
			cgDestroyProgram(m_cgVertexProgram);
			m_cgVertexProgram = NULL;
		}
		if(cgIsProgram(m_cgFragmentProgram))
		{
			cgDestroyProgram(m_cgFragmentProgram);
			m_cgFragmentProgram = NULL;
		}
	}

	void Enable()
	{
		if(cgIsProgram(m_cgVertexProgram))
		{
			cgGLEnableProfile(m_cgVertexProfile);
			cgGLBindProgram(m_cgVertexProgram);
			CGparameter pMatrix = cgGetNamedParameter(m_cgVertexProgram, "gl_ModelViewProjectionMatrix");
			cgGLSetStateMatrixParameter(pMatrix, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);
		}
		if(cgIsProgram(m_cgFragmentProgram))
		{
			cgGLEnableProfile(m_cgFragmentProfile);
			cgGLBindProgram(m_cgFragmentProgram);
		}
	}

	void Disable()
	{
		if(cgIsProgram(m_cgVertexProgram))
			cgGLDisableProfile(m_cgVertexProfile);
		if(cgIsProgram(m_cgFragmentProgram))
			cgGLDisableProfile(m_cgFragmentProfile);
	}

	void SetTextureParameter(const char *pszParameter, int p1)
	{
		CGparameter pVert = cgGetNamedParameter(m_cgVertexProgram, pszParameter);
		if(pVert)
			cgGLSetTextureParameter(pVert, p1);
		CGparameter pFrag = cgGetNamedParameter(m_cgFragmentProgram, pszParameter);
		if(pFrag)
			cgGLSetTextureParameter(pFrag, p1);
	}
	void SetUniformParameter1i(const char *pszParameter, int n1)
	{
	}
	void SetUniformParameter1f(const char *pszParameter, float p1)
	{
		CGparameter pVert = cgGetNamedParameter(m_cgVertexProgram, pszParameter);
		if(pVert)
			cgGLSetParameter1f(pVert, p1);
		CGparameter pFrag = cgGetNamedParameter(m_cgFragmentProgram, pszParameter);
		if(pFrag)
			cgGLSetParameter1f(pFrag, p1);
	}
	void SetUniformParameter3f(const char *pszParameter, float p1, float p2, float p3)
	{
		CGparameter pVert = cgGetNamedParameter(m_cgVertexProgram, pszParameter);
		if(pVert)
			cgGLSetParameter3f(pVert, p1, p2, p3);
		CGparameter pFrag = cgGetNamedParameter(m_cgFragmentProgram, pszParameter);
		if(pFrag)
			cgGLSetParameter3f(pFrag, p1, p2, p3);
	}
};

#else

#include <map>
#include <string>

class CShaderObject
{
protected:
	GLhandleARB m_hProgram;
	GLhandleARB m_hVertexShader;
	GLhandleARB m_hFragmentShader;
	std::map<std::string, GLint> m_mapParameters;

	void LogGLErrors()
	{
		GLenum glErr;
		while((glErr = glGetError()) != GL_NO_ERROR)
			LogError((const char *)gluErrorString(glErr));
	}
	void LogGLInfoLog(GLhandleARB hObj)
	{
		int nBytes;
		glGetObjectParameterivARB(hObj, GL_OBJECT_INFO_LOG_LENGTH_ARB, &nBytes);
		if(nBytes)
		{
			char *pInfo = new char[nBytes];
			glGetInfoLogARB(hObj, nBytes, &nBytes, pInfo);
			LogInfo(pInfo);
			delete pInfo;
		}
	}

public:
	CShaderObject()
	{
		m_hProgram = glCreateProgramObjectARB();
		m_hVertexShader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
		m_hFragmentShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
	}
	~CShaderObject()
	{
		glDeleteObjectARB(m_hFragmentShader);
		glDeleteObjectARB(m_hVertexShader);
		glDeleteObjectARB(m_hProgram);
	}

	bool Load(const char *pszPath, const char *pszPath2=NULL)
	{
		char szPath[_MAX_PATH], *psz;
		int nBytes, bSuccess;

		sprintf(szPath, "%s.vert", pszPath);
		LogInfo("Compiling GLSL shader %s", szPath);
		std::ifstream ifVertexShader(szPath, std::ios::binary);
		ifVertexShader.seekg(0, std::ios::end);
		nBytes = ifVertexShader.tellg();
		ifVertexShader.seekg(0, std::ios::beg);
		psz = new char[nBytes+1];
		ifVertexShader.read(psz, nBytes);
		psz[nBytes] = 0;
		ifVertexShader.close();
		glShaderSourceARB(m_hVertexShader, 1, (const char **)&psz, &nBytes);
		glCompileShaderARB(m_hVertexShader);
		glGetObjectParameterivARB(m_hVertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bSuccess);
		delete psz;
		if(!bSuccess)
		{
			LogError("Failed to compile vertex shader %s", szPath);
			LogGLErrors();
			LogGLInfoLog(m_hVertexShader);
			return false;
		}

		sprintf(szPath, "%s.frag", pszPath2 ? pszPath2 : pszPath);
		LogInfo("Compiling GLSL shader %s", szPath);
		std::ifstream ifFragmentShader(szPath, std::ios::binary);
		ifFragmentShader.seekg(0, std::ios::end);
		nBytes = ifFragmentShader.tellg();
		ifFragmentShader.seekg(0, std::ios::beg);
		psz = new char[nBytes];
		ifFragmentShader.read(psz, nBytes);
		ifFragmentShader.close();
		glShaderSourceARB(m_hFragmentShader, 1, (const char **)&psz, &nBytes);
		glCompileShaderARB(m_hFragmentShader);
		glGetObjectParameterivARB(m_hFragmentShader, GL_OBJECT_COMPILE_STATUS_ARB, &bSuccess);
		delete psz;
		if(!bSuccess)
		{
			LogError("Failed to compile fragment shader %s", szPath);
			LogGLErrors();
			LogGLInfoLog(m_hFragmentShader);
			return false;
		}

		glAttachObjectARB(m_hProgram, m_hVertexShader);
		glAttachObjectARB(m_hProgram, m_hFragmentShader);
		glLinkProgramARB(m_hProgram);

		glGetObjectParameterivARB(m_hProgram, GL_OBJECT_LINK_STATUS_ARB, &bSuccess);
		if(!bSuccess)
		{
			LogError("Failed to link shader %s", szPath);
			LogGLErrors();
			LogGLInfoLog(m_hProgram);
			return false;
		}

		LogGLInfoLog(m_hProgram);
		return true;
	}

	void Enable()
	{
		glUseProgramObjectARB(m_hProgram);
	}
	void Disable()
	{
		glUseProgramObjectARB(NULL);
	}

	GLint GetUniformParameterID(const char *pszParameter)
	{
		std::map<std::string, GLint>::iterator it = m_mapParameters.find(pszParameter);
		if(it == m_mapParameters.end())
		{
			GLint nLoc = glGetUniformLocationARB(m_hProgram, pszParameter);
			it = m_mapParameters.insert(std::pair<std::string, GLint>(pszParameter, nLoc)).first;
		}
		return it->second;
	}

	/*
	void BindTexture(const char *pszParameter, unsigned int nID)
	{
		GLint n = GetUniformParameterID(pszParameter);
		glBindTexture(GL_TEXTURE_2D, nID);
		glUniform1iARB(n, nID);
	}
	*/
	void SetUniformParameter1i(const char *pszParameter, int n1)
	{
		glUniform1iARB(GetUniformParameterID(pszParameter), n1);
	}
	void SetUniformParameter1f(const char *pszParameter, float p1)
	{
		glUniform1fARB(GetUniformParameterID(pszParameter), p1);
	}
	void SetUniformParameter3f(const char *pszParameter, float p1, float p2, float p3)
	{
		glUniform3fARB(GetUniformParameterID(pszParameter), p1, p2, p3);
	}
};

#endif

#endif // __ShaderObject_h__
