/*! \file Program.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Program.h.
 */

#include <GL/glew.h>
#include "Program.h"
#include <vector>

Program
  ::Program(void)
    :mIdentifier(0)
{
  ;
} // end Program::Program()

void Program
  ::create(void)
{
  destroy();
  mIdentifier = glCreateProgram();
} // end Program::create()

bool Program
  ::create(const GLuint vertex,
           const GLuint fragment)
{
  create();
  return link(vertex,fragment);
} // end Program::create()

bool Program
  ::create(const GLuint vertex,
           const GLuint geometry,
           const GLuint fragment)
{
  create();
  return link(vertex,geometry,fragment);
} // end Program::create()

void Program
  ::destroy(void)
{
  if(mIdentifier != 0)
  {
    glDeleteProgram(getIdentifier());
    mIdentifier = 0;
  } // end if
} // end Program::destroy()

GLuint Program
  ::getIdentifier(void) const
{
  return mIdentifier;
} // end Program::getIdentifier()

Program
  ::operator GLuint (void) const
{
  return getIdentifier();
} // end operator GLuint ()

void Program
  ::attach(const GLuint s) const
{
  glAttachShader(getIdentifier(), s);
} // end Program::attach()

void Program
  ::detach(const GLuint s) const
{
  glDetachShader(getIdentifier(), s);
} // end Program::detach()

void Program
  ::detachShaders(void) const
{
  // get the number of attached shaders
  GLint num = 0;
  glGetProgramiv(*this, GL_ATTACHED_SHADERS, &num);

  if(num > 0)
  {
    std::vector<GLuint> shaders(num);
    GLsizei count = 0;
    glGetAttachedShaders(*this, num, &count, &shaders[0]);
    for(GLsizei i = 0; i < count; ++i)
    {
      detach(shaders[i]);
    } // end for i
  } // end if
} // end Program::detachShaders()

bool Program
  ::link(void)
{
  glLinkProgram(getIdentifier());

  GLint error = 0;
  glGetProgramiv(getIdentifier(), GL_LINK_STATUS, &error);
  createUniformMap();
  createAttributeMap();

  return error == GL_TRUE;
} // end Program::link()

bool Program
  ::link(const GLuint vertex,
         const GLuint fragment)
{
  // detach all shaders
  detachShaders();

  if(vertex != 0) attach(vertex);
  if(fragment != 0) attach(fragment);

  return link();
} // end Program::link()

bool Program
  ::link(const GLuint vertex,
         const GLuint geometry,
         const GLuint fragment)
{
  // detach all shaders
  detachShaders();

  if(vertex != 0) attach(vertex);
  if(geometry != 0) attach(geometry);
  if(fragment != 0) attach(fragment);

  return link();
} // end Program::link()

void Program
  ::bind(void) const
{
  glUseProgram(getIdentifier());
} // end Program::bind();

void Program
  ::unbind(void) const
{
  glUseProgram(0);
} // end Program::unbind()

void Program
  ::getInfoLog(std::string &log) const
{
  int len = 0;
  glGetProgramiv(getIdentifier(), GL_INFO_LOG_LENGTH, &len);

  if(len != 0)
  {
    char *message = (char*)malloc(len * sizeof(char));

    // get log
    int dummy = 0;
    glGetProgramInfoLog(getIdentifier(), len, &dummy, message);
    log = std::string(message);
    free(message);
  } // end if
} // end Shader::getInfoLog()

void Program
  ::createUniformMap(void)
{
  mUniformMap.clear();

  // get the length of the longest uniform name
  GLint len = 0;
  glGetProgramiv(*this,GL_ACTIVE_UNIFORM_MAX_LENGTH, &len);

  GLchar *name = (GLchar*)malloc(len * sizeof(GLchar));

  // get the number of uniforms
  GLint num = 0;
  glGetProgramiv(*this,GL_ACTIVE_UNIFORMS,&num);
  for(GLint i = 0; i < num; ++i)
  {
    // get the name of uniform i
    GLint written = 0;
    GLint size = 0;
    GLenum type = 0;
    glGetActiveUniform(*this, i, len, &written, &size, &type, name);

    // look up the location of the uniform
    GLint location = getUniformLocation(name);

    // insert into map
    mUniformMap[std::string(name)] = location;
  } // end for i

  free(name);
} // end Program::createUniformMap()

void Program
  ::createAttributeMap(void)
{
  mAttributeMap.clear();

  // get the length of the longest varying name
  GLint len = 0;
  glGetProgramiv(*this,GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &len);

  GLchar *name = (GLchar*)malloc(len * sizeof(GLchar));

  // get the number of attributes
  GLint num = 0;
  glGetProgramiv(*this,GL_ACTIVE_ATTRIBUTES,&num);
  for(GLint i = 0; i < num; ++i)
  {
    // get the name of attribute i
    GLint written = 0;
    GLint size = 0;
    GLenum type = 0;
    glGetActiveAttrib(*this,i,len,&written,&size,&type,name);

    // look up the location of the attribute
    GLint location = getAttribLocation(name);

    // insert into map
    mAttributeMap[std::string(name)] = location;
  } // end for i
} // end Program::createAttributeMap()

GLint Program
  ::getUniformLocation(const GLchar *name) const
{
  return glGetUniformLocation(*this, name);
} // end Program::getUniformLocation()

GLint Program
  ::getAttribLocation(const GLchar *name) const
{
  return glGetAttribLocation(*this, name);
} // end Program::getAttribLocation()

void Program
  ::setUniform4fv(const GLchar *name,
                  const GLfloat *v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform4fv(u->second,1,v);
} // end Program::setUniform4fv()

void Program
  ::setUniform4f(const GLchar *name,
                 const GLfloat v0,
                 const GLfloat v1,
                 const GLfloat v2,
                 const GLfloat v3) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform4f(u->second,v0,v1,v2,v3);
} // end Program::setUniform4fv()

void Program
  ::setUniform3f(const GLchar *name,
                 const GLfloat v0,
                 const GLfloat v1,
                 const GLfloat v2) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform3f(u->second,v0,v1,v2);
} // end Program::setUniform4fv()

void Program
  ::setUniform3fv(const GLchar *name,
                  const GLfloat *v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform3fv(u->second,1,v);
} // end Program::setUniform4fv()

void Program
  ::setUniform2fv(const GLchar *name,
                  const GLfloat *v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform2fv(u->second,1,v);
} // end Program::setUniform2fv()

void Program
  ::setUniform2iv(const GLchar *name,
                  const GLint *v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform2iv(u->second,1,v);
} // end Program::setUniform2iv()

void Program
  ::setUniform1f(const GLchar *name,
                 const GLfloat v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
    glUniform1f(u->second,v);
} // end Program::setUniform4fv()

void Program
  ::setUniform1i(const GLchar *name,
                 const GLint v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
  {
    //glUniform1i(u->second,v);
    glUniform1iv(u->second, 1, &v);
  } // end if
  else
  {
    std::cerr << "Program::setUniform1i(): Couldn't find uniform '" << name << "'" << std::endl;
  } // end else
} // end Program::setUniform1i()

void Program
  ::setUniformMatrix4fv(const GLchar *name,
                        const GLboolean transpose,
                        const GLfloat *v) const
{
  VariableMap::const_iterator u = mUniformMap.find(name);
  if(u != mUniformMap.end())
  {
    glUniformMatrix4fv(u->second, 1, transpose, v);
  } // end if
  else
  {
    std::cerr << "Program::setUniformMatrix4fv(): Couldn't find uniform '" << name << "'" << std::endl;
  } // end else
} // end Program::setUniformMatrix4fv()

void Program
  ::setParameteri(const GLenum pname,
                  const GLint value) const
{
  glProgramParameteriEXT(*this, pname, value);
} // end Program::setParameteri()

std::ostream &operator<<(std::ostream &os, const Program &p)
{
  std::string log;
  p.getInfoLog(log);
  return os << log;
} // end operator<<()

