/*! \file GLObject.h
 *  \author Jared Hoberock
 *  \brief Defines the interface for an abstraction of an OpenGL object.
 */

#ifndef GL_OBJECT_H
#define GL_OBJECT_H

#include <GL/glew.h>

typedef void (*CreateCallback)(GLuint, GLuint *);
typedef void (*DestroyCallback)(GLuint, GLuint *);
typedef void (*BindCallback)(GLenum, GLuint);

template<CreateCallback createCallback,
         DestroyCallback destroyCallback,
         BindCallback bindCallback> class GLObject
{
  public:
    /*! \fn GLObject
     *  \brief Null constructor sets mIdentifer and mTarget to 0.
     */
    inline GLObject(void);

    /*! \fn GLObject
     *  \brief Constructor sets mIdentifier to 0 and mTarget
     *         as given.
     *  \param target Sets mTarget.
     */
    inline GLObject(const GLenum target);

    /*! \fn ~GLObject
     *  \brief Null destructor calls destroy.
     */
    inline virtual ~GLObject(void);

    /*! \fn create
     *  \brief This method creates an OpenGL identifier for this GLObject.
     */
    inline void create(void);

    /*! \fn bind
     *  \brief This method binds this GLObject as the current object of its
     *         target.
     */
    inline void bind(void) const;

    /*! \fn unbind
     *  \brief This method unbinds this GLObject as the current object of its
     *         target, and binds 0 in its place.
     */
    inline void unbind(void) const;

    /*! \fn destroy
     *  \brief This method destroys this GLObject's identifier.
     */
    inline void destroy(void);

    /*! \fn getIdentifier
     *  \brief This method returns this GLObject's identifier.
     *  \return mIdentifier
     */
    inline GLuint getIdentifier(void) const;

    /*! \fn setTarget
     *  \brief This method sets this GLObject's target.
     *  \param t Sets mTarget.
     */
    inline void setTarget(const GLenum t);

    /*! \fn getTarget
     *  \brief This method returns this GLObject's target.
     *  \return mTarget
     */
    inline GLenum getTarget(void) const;

    /*! \fn operator GLuint
     *  \brief GLuint cast returns mIdentifer.
     */
    inline operator GLuint (void) const;

  protected:
    /*! \fn setIdentifier
     *  \brief This method sets this GLObject's identifier.
     *  \param id Sets mIdentifer.
     */
    inline void setIdentifier(const GLuint id);

  private:
    /*! The OpenGL identifier of this GLObject.
     */
    GLuint mIdentifier;

    /*! The OpenGL target of this GLObject.
     */
    GLenum mTarget;
}; // end class GLObject

#include "GLObject.inl"

#endif // GL_OBJECT_H

