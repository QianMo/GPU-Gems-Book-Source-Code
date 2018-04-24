/*! \file GLObject.inl
 *  \author Jared Hoberock
 *  \brief Inline file for GLObject.h.
 */

#include "GLObject.h"

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  GLObject<createCallback, destroyCallback, bindCallback>::GLObject(void)
{
  setIdentifier(0);
  setTarget(0);
} // end GLObject::GLObject()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  GLObject<createCallback, destroyCallback, bindCallback>::GLObject(const GLenum target)
{
  setIdentifier(0);
  setTarget(target);
} // end GLObject::GLObject()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  GLObject<createCallback, destroyCallback, bindCallback>::~GLObject(void)
{
  destroy();
} // end GLObject::~GLObject()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  GLuint GLObject<createCallback, destroyCallback, bindCallback>::getIdentifier(void) const
{
  return mIdentifier;
} // end GLObject::getIdentifier()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  GLenum GLObject<createCallback, destroyCallback, bindCallback>::getTarget(void) const
{
  return mTarget;
} // end GLObject::getTarget()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  void GLObject<createCallback, destroyCallback, bindCallback>::setTarget(const GLenum t)
{
  mTarget = t;
} // end GLObject::setTarget()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  void GLObject<createCallback, destroyCallback, bindCallback>::setIdentifier(const GLuint id)
{
  mIdentifier = id;
} // end GLObject::setIdentifier()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  GLObject<createCallback, destroyCallback, bindCallback>::operator GLuint (void) const
{
  return mIdentifier;
} // end GLObject::operator GLuint()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  void GLObject<createCallback, destroyCallback, bindCallback>::create(void)
{
  destroy();

  GLuint id = 0;
  createCallback(1, &id);
  setIdentifier(id);
} // end GLObject::create()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  void GLObject<createCallback, destroyCallback, bindCallback>::destroy(void)
{
  GLuint id = getIdentifier();
  if(id != 0)
  {
    destroyCallback(1, &id);
  } // end if

  setIdentifier(id);
} // end GLObject::destroy()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  void GLObject<createCallback, destroyCallback, bindCallback>::bind(void) const
{
  bindCallback(getTarget(), getIdentifier());
} // end GLObject::bind()

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback>
  void GLObject<createCallback, destroyCallback, bindCallback>::unbind(void) const
{
  bindCallback(getTarget(), 0);
} // end GLObject::unbind()

