#include "../Framework/Common.h"
#include "Application_OGL.h"
#include "GLSLShader.h"

GLSLShader::GLSLShader()
{
  m_FragmentShader = NULL;
  m_VertexShader = NULL;
  m_GeometryShader = NULL;
  m_ProgramObject = NULL;
}

GLSLShader::~GLSLShader()
{
  Destroy();
}

inline GLhandleARB LoadShader(const char *strFile, int iShaderType)
{
  FILE *pFile = NULL;
  pFile = fopen(strFile, "rt");
  if(pFile == NULL)
  {
    char pText[1024];
    pText[0]=0;
    _snprintf(pText, 1024, "Failed to open shader file %s", strFile);
    MessageBoxA(NULL, pText, "Error!", MB_OK);
    return NULL;
  }
  
  // load file contents
  //
  char *strContent = NULL;
  fseek(pFile, 0, SEEK_END);
  long iFileSize = ftell(pFile);
  rewind(pFile);
	if(iFileSize > 0)
  {
		strContent = (char *)malloc(iFileSize+1);
    memset(strContent, 0, iFileSize+1);
		fread(strContent, 1, iFileSize, pFile);
	}
	fclose(pFile);


	GLhandleARB ShaderObject = glCreateShaderObjectARB(iShaderType);
  if(ShaderObject == NULL)
  {
    char pText[1024];
    pText[0]=0;
    _snprintf(pText, 1024, "Failed to create shader object for %s", strFile);
    MessageBoxA(NULL, pText, "Error!", MB_OK);
    return NULL;
  }

	glShaderSourceARB(ShaderObject, 1, (const GLcharARB **)&strContent, NULL);
	free(strContent);

	glCompileShaderARB(ShaderObject);

  char strInfo[1024];
  GLsizei iInfoLength = 0;
  glGetInfoLogARB(ShaderObject, 1024, &iInfoLength, strInfo);
  GLint iCompileSuccess = -1;
  glGetObjectParameterivARB(ShaderObject, GL_OBJECT_COMPILE_STATUS_ARB, &iCompileSuccess);
  if(iCompileSuccess != GL_TRUE)
  {
    char pText[1024];
    pText[0]=0;
    _snprintf(pText, 1024, "Compiling %s failed:\n%s", strFile, strInfo);
    MessageBoxA(NULL, pText, "Error!", MB_OK);
    return NULL;
  }

  return ShaderObject;
}

bool GLSLShader::Load(const char *strVS, const char *strGS, const char *strFS, int iGSMaxVertices)
{
  if(strVS != NULL)
  {
    m_VertexShader = LoadShader(strVS, GL_VERTEX_SHADER_ARB);
    if(m_VertexShader == NULL) return false;
  }

  if(strFS != NULL)
  {
    m_FragmentShader = LoadShader(strFS, GL_FRAGMENT_SHADER_ARB);
    if(m_FragmentShader == NULL) return false;
  }

  if(strGS != NULL)
  {
    m_GeometryShader = LoadShader(strGS, GL_GEOMETRY_SHADER_EXT);
    if(m_GeometryShader == NULL) return false;
  }

	m_ProgramObject = glCreateProgramObjectARB();
  if(m_ProgramObject == NULL)
  {
    MessageBox(NULL, TEXT("Failed to create shader program object!"), TEXT("Error!"), MB_OK);
    return false;
  }

  if(m_VertexShader != NULL)
  {
  	glAttachObjectARB(m_ProgramObject, m_VertexShader);
  }

  if(m_GeometryShader != NULL)
  {
	  glAttachObjectARB(m_ProgramObject, m_GeometryShader);
    glProgramParameteriEXT(m_ProgramObject, GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    glProgramParameteriEXT(m_ProgramObject, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLES);
    glProgramParameteriEXT(m_ProgramObject, GL_GEOMETRY_VERTICES_OUT_EXT, iGSMaxVertices);
  }

  if(m_FragmentShader != NULL)
  {
    glAttachObjectARB(m_ProgramObject, m_FragmentShader);
  }

	glLinkProgramARB(m_ProgramObject);

  char strInfo[1024];
  GLsizei iInfoLength = 0;
  glGetInfoLogARB(m_ProgramObject, 1024, &iInfoLength, strInfo);
  GLint iCompileSuccess = GL_TRUE;
  glGetObjectParameterivARB(m_ProgramObject, GL_OBJECT_LINK_STATUS_ARB, &iCompileSuccess);
  if(iCompileSuccess != GL_TRUE)
  {
    char pText[1024];
    pText[0]=0;
    _snprintf(pText, 1024, "Shader linking failed:\n%s", strInfo);
    MessageBoxA(NULL, pText, "Error!", MB_OK);
    return false;
  }

  return true;
}

void GLSLShader::Destroy(void)
{
  if(m_FragmentShader != NULL)
  {
    glDeleteObjectARB(m_FragmentShader);
    m_FragmentShader = NULL;
  }

  if(m_VertexShader != NULL)
  {
    glDeleteObjectARB(m_VertexShader);
    m_VertexShader = NULL;
  }

  if(m_GeometryShader != NULL)
  {
    glDeleteObjectARB(m_GeometryShader);
    m_GeometryShader = NULL;
  }

  if(m_ProgramObject != NULL)
  {
    glDeleteObjectARB(m_ProgramObject);
    m_ProgramObject = NULL;
  }
}

extern GLSLShader *g_pActiveShader;

void GLSLShader::Activate(void)
{
	glUseProgramObjectARB(m_ProgramObject);
  g_pActiveShader = this;
}

void GLSLShader::Deactivate(void)
{
	glUseProgramObjectARB(NULL);
  g_pActiveShader = NULL;
}

void GLSLShader::SetMatrix(const char *strName, const Matrix &m)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniformMatrix4fvARB(iLocation, 1, false, (float*)&m);
}

void GLSLShader::SetVector(const char *strName, const Vector4 &v)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniform4fARB(iLocation, v.x, v.y, v.z, v.w);
}

void GLSLShader::SetVector(const char *strName, const Vector3 &v)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniform3fARB(iLocation, v.x, v.y, v.z);
}

void GLSLShader::SetFloatArray(const char *strName, const float *p, int iCount)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniform1fvARB(iLocation, iCount, p);
}

void GLSLShader::SetMatrixArray(const char *strName, const Matrix *p, int iCount)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniformMatrix4fvARB(iLocation, iCount, false, (float*)p);
}

void GLSLShader::SetInt(const char *strName, int i)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniform1iARB(iLocation, i);
}

void GLSLShader::SetIntArray(const char *strName, const int *p, int iCount)
{
  GLint iLocation = glGetUniformLocation(m_ProgramObject, strName);
  glUniform1ivARB(iLocation, iCount, p);
}