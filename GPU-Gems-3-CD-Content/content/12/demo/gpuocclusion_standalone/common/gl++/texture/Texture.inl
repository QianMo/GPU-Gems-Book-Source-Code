/*! \file Texture.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Texture.h.
 */

#include "Texture.h"

Texture::Texture(void):Parent()
{
  setTarget(GL_TEXTURE_RECTANGLE_ARB);
  setWidth(1);
  setHeight(1);
  setDepth(0);
  setBorderSize(0);
} // end Texture::Texture()

void Texture
  ::create(const GLenum target)
{
  Parent::create();
  setTarget(target);
} // end Texture::create()

void Texture::setInternalFormat(const GLenum f)
{
  mInternalFormat = f;
} // end Texture::setInternalFormat()

GLenum Texture::getInternalFormat(void) const
{
  return mInternalFormat;
} // end Texture::getInternalFormat()

GLsizei Texture::getWidth(void) const
{
  return mWidth;
} // end Texture::getWidth()

GLsizei Texture::getMaxS(void) const
{
  return (getTarget() == GL_TEXTURE_RECTANGLE_ARB) ?
         (getWidth()):
         1;
} // end Texture::getMaxS()

GLsizei Texture::getHeight(void) const
{
  return mHeight;
} // end Texture::getHeight()

GLsizei Texture::getMaxT(void) const
{
  return (getTarget() == GL_TEXTURE_RECTANGLE_ARB) ?
         (getHeight()):
         1;
} // end Texture::getMaxT()

GLsizei Texture::getDepth(void) const
{
  return mDepth;
} // end Texture::getDepth()

void Texture::init(const GLenum internalFormat,
                   const GLsizei width)
{
  init(internalFormat, width, getBorderSize());
} // end Texture::init()

void Texture::init(const GLenum internalFormat,
                   const GLsizei width,
                   const GLsizei height)
{
  init(internalFormat, width, height, getBorderSize());
} // end Texture::init()

void Texture::texImage3D(const GLenum internalFormat,
                         const GLsizei width,
                         const GLsizei height,
                         const GLsizei depth)
{
  texImage3D(internalFormat, width, height, depth, getBorderSize());
} // end Texture::init()

void Texture::init(const GLenum internalFormat,
                   const GLsizei width,
                   const GLsizei height,
                   const GLint border)
{
  // bind the Texture first
  bind();

  // \todo It'd be nice to templatize this switch
  GLenum externalFormat = GL_RGB;
  GLenum externalType = GL_UNSIGNED_BYTE;
  if(internalFormat == GL_DEPTH_COMPONENT)
  {
    externalFormat = GL_DEPTH_COMPONENT;
    externalType = GL_UNSIGNED_INT;
  } // end if

  glTexImage2D(getTarget(),
               0,
               internalFormat,
               width, height,
               border,
               externalFormat, externalType, 0);

  // unbind the Texture
  unbind();

  setWidth(width);
  setHeight(height);
  setBorderSize(border);
  setInternalFormat(internalFormat);
} // end Texture::init()

void Texture::texImage3D(const GLenum internalFormat,
                         const GLsizei width,
                         const GLsizei height,
                         const GLsizei depth,
                         const GLint border)
{
  // bind the Texture first
  bind();

  // \todo It'd be nice to templatize this switch
  GLenum externalFormat = GL_RGB;
  GLenum externalType = GL_UNSIGNED_BYTE;
  if(internalFormat == GL_DEPTH_COMPONENT)
  {
    externalFormat = GL_DEPTH_COMPONENT;
    externalType = GL_UNSIGNED_INT;
  } // end if

  glTexImage3D(getTarget(),
               0,
               internalFormat,
               width, height, depth,
               border,
               externalFormat, externalType, 0);

  // unbind the Texture
  unbind();

  setWidth(width);
  setHeight(height);
  setDepth(depth);
  setBorderSize(border);
  setInternalFormat(internalFormat);
} // end Texture::init()

template<typename T>
  void Texture::init(const GLenum internalFormat,
                     const GLsizei width,
                     const GLint border,
                     const GLenum externalFormat,
                     const T *pixels)
{
  // bind the Texture first
  bind();
  glTexImage1D(getTarget(),
               0,
               internalFormat,
               width, border,
               externalFormat, ExternalFormat<T>(),
               pixels);

  // unbind the Texture
  unbind();

  setWidth(width);
  setHeight(0);
  setDepth(0);
  setBorderSize(border);
  setInternalFormat(internalFormat);
} // end Texture::init()

template<typename T>
  void Texture::init(const GLenum internalFormat,
                     const GLsizei width,
                     const GLsizei height,
                     const GLint border,
                     const GLenum externalFormat,
                     const T *pixels)
{
  // bind the Texture first
  bind();
  glTexImage2D(getTarget(),
               0,
               internalFormat,
               width, height, border,
               externalFormat, ExternalFormat<T>(),
               pixels);


  // unbind the Texture
  unbind();

  setWidth(width);
  setHeight(height);
  setBorderSize(border);
  setInternalFormat(internalFormat);
} // end Texture::init()

template<typename T>
  void Texture::texImage3D(const GLenum internalFormat,
                           const GLsizei width,
                           const GLsizei height,
                           const GLsizei depth,
                           const GLint border,
                           const GLenum externalFormat,
                           const T *pixels)
{
  // bind the Texture first
  bind();
  glTexImage3D(getTarget(),
               0,
               internalFormat,
               width, height, depth, border,
               externalFormat, ExternalFormat<T>(),
               pixels);


  // unbind the Texture
  unbind();

  setWidth(width);
  setHeight(height);
  setDepth(depth);
  setBorderSize(border);
  setInternalFormat(internalFormat);
} // end Texture::init()

void Texture::setHeight(const GLsizei h)
{
  mHeight = h;
} // end Texture::setHeight()

void Texture::setWidth(const GLsizei w)
{
  mWidth = w;
} // end Texture::setWidth()

void Texture::setDepth(const GLsizei d)
{
  mDepth = d;
} // end Texture::setDepth()

void Texture::setBorderSize(const GLint b)
{
  mBorderSize = b;
} // end Texture::setBorderSize()

GLint Texture::getBorderSize(void) const
{
  return mBorderSize;
} // end Texture::getBorderSize()

void Texture::setTarget(const GLenum t)
{
  return Parent::setTarget(t);
} // end Texture::getTarget()

template<typename Type>
  void Texture::getPixels(const GLenum format, Type *p) const
{
  glGetTexImage(getTarget(),
                0,
                format,
                ExternalFormat<Type>(),
                p);
} // end Texture::getPixels()

void Texture::generateMipmap(void) const
{
  bind();
  glGenerateMipmapEXT(getTarget());
  unbind();
} // end Texture::generateMipmap()

template<>
  Texture::ExternalFormat<GLfloat>::operator GLenum (void) const
{
  return GL_FLOAT;
} // end Texture::ExternalFormat::operator GLenum ()

template<>
  Texture::ExternalFormat<GLhalf>::operator GLenum (void) const
{
  return GL_HALF_FLOAT_ARB;
} // end Texture::ExternalFormat::operator GLenum ()

template<>
  Texture::ExternalFormat<GLbyte>::operator GLenum (void) const
{
  return GL_BYTE;
} // end ExternalFormat::operator GLenum ()

template<>
  Texture::ExternalFormat<GLubyte>::operator GLenum (void) const
{
  return GL_UNSIGNED_BYTE;
} // end Texture::ExternalFormat::operator GLenum ()

template<>
  Texture::ExternalFormat<GLuint>::operator GLenum (void) const
{
  return GL_UNSIGNED_INT;
} // end Texture::ExternalFormat::operator GLenum ()

