/*! \file Shader.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Shader.h.
 */

#include "Shader.h"
#include <string>
#include <fstream>

Shader
  ::Shader(void)
    :mIdentifier(0),mType(0)
{
  ;
} // end Shader::Shader()

Shader
  ::Shader(const GLenum type)
    :mIdentifier(0),mType(0)
{
  ;
} // end Shader::Shader()

void Shader
  ::create(void)
{
  destroy();
  mIdentifier = glCreateShader(mType);
} // end Shader::create()

void Shader
  ::create(const GLenum type)
{
  mType = type;
  create();
} // end Shader::create()

bool Shader
  ::create(const GLenum type,
           const char *source)
{
  create(type);
  return compile(source);
} // end Shader::create()

bool Shader
  ::createFromFile(const GLenum type,
                   const char *filename)
{
  create(type);
  return compileFromFile(filename);
} // end Shader::createFromFile()

void Shader
  ::destroy(void)
{
  if(mIdentifier != 0)
  {
    glDeleteShader(getIdentifier());
    mIdentifier = 0;
  } // end if
} // end Shader::destroy()

GLuint Shader
  ::getIdentifier(void) const
{
  return mIdentifier;
} // end Shader::getIdentifier()

Shader
  ::operator GLuint (void) const
{
  return getIdentifier();
} // end Shader::operator GLuint ()

GLenum Shader
  ::getType(void) const
{
  return mType;
} // end Shader::getType()

bool Shader
  ::compile(const char *source)
{
  // set the source
  setSource(source);

  // compile
  return compile();
} // end Shader::init()

bool Shader
  ::compile(void)
{
  glCompileShader(*this);

  // check for error
  GLint error = 0;
  glGetShaderiv(getIdentifier(), GL_COMPILE_STATUS, &error);
  return error == GL_TRUE;
} // end Shader::compile()

bool Shader
  ::compileFromFile(const char *filename)
{
  if(setSourceFromFile(filename))
  {
    return compile();
  } // end if

  return false;
} // end Shader::initFromFile()

void Shader
  ::getInfoLog(std::string &log) const
{
  int len = 0;
  glGetShaderiv(getIdentifier(), GL_INFO_LOG_LENGTH, &len);

  if(len != 0)
  {
    char *message = (char*)malloc(len * sizeof(char));

    // get log
    int dummy = 0;
    glGetShaderInfoLog(getIdentifier(), len, &dummy, message);
    log = std::string(message);
    free(message);
  } // end if
} // end Shader::getInfoLog()

void Shader
  ::getSource(std::string &source) const
{
  // get the length of the source
  GLint len = 0;
  glGetShaderiv(*this, GL_SHADER_SOURCE_LENGTH, &len);
  source.resize(len);
  GLsizei written = 0;
  glGetShaderSource(*this, len, &written, &source[0]);
} // end Shader::getSource()

void Shader
  ::setSource(const char *source) const
{
  glShaderSource(getIdentifier(), 1, &source, 0);
} // end Shader::setSource()

bool Shader
  ::setSourceFromFile(const char *filename) const
{
  std::string source;
  char line[256];
  std::ifstream infile(filename);
  if(infile.is_open())
  {
    while(!infile.eof())
    {
      infile.getline(line, 256);
      source += std::string(line) + std::string("\n");
    } // end while

    setSource(source.c_str());
    return true;
  } // end if

  return false;
} // end Shader::setSourceFromFile()

std::ostream &operator<<(std::ostream &os, const Shader &s)
{
  std::string log;
  s.getInfoLog(log);
  return os << log;
} // end operator<<()

