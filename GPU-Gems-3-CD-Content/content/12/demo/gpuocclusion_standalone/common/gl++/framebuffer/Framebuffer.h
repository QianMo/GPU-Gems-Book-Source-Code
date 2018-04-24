/*! \file Framebuffer.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for abstracting OpenGL Framebuffer objects.
 */

#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <GL/glew.h>
#include <gl++/globject/GLObject.h>

/*! \fn genThunk
 *  \todo Find a way around this.
 */
inline void genThunk(GLuint num, GLuint *ids)
{
  glGenFramebuffersEXT(num,ids);
}  // end genThunk()

/*! \fn deleteThunk
 *  \todo Find a way around this.
 */
inline void deleteThunk(GLuint num, GLuint *ids)
{
  glDeleteFramebuffersEXT(num,ids);
} // end deleteThunk()

/*! \fn bindThunk
 *  \todo Find a way around this.
 */
inline void bindThunk(GLenum target, GLuint id)
{
  glBindFramebufferEXT(target, id);
} // end bindThunk()

class Framebuffer : public GLObject<genThunk,
                                    deleteThunk,
                                    bindThunk>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef GLObject<genThunk, deleteThunk, bindThunk> Parent;

    /*! \fn Framebuffer
     *  Null constructor calls the parent and sets
     *  mTarget to GL_FRAMEBUFFER_EXT.
     */
    inline Framebuffer(void);

    /*! \fn isComplete
     *  \brief This method queries OpenGL to discover whether
     *         or not the currently bound Framebuffer is "complete".
     *  \return glCheckFramebufferStatusEXT() == GL_FRAMEBUFFER_COMPLETE_EXT
     *  \note This method only checks whether or not the currently bound
     *        Framebuffer is complete; not necessarily this Framebuffer.
     */
    inline bool isComplete(void) const;

    /*! \fn attachTexture
     *  \brief This method attaches the specified texture
     *         to the given attachment of this Framebuffer.
     *  \param textureTarget The OpenGL target of the texture
     *                       to attach.
     *  \param attachment The OpenGL enum of the attachment.
     *  \param texture The OpenGL name of the texture to attach.
     *  \param level The mipmap level to attach; defaults to 0.
     *  \todo 1D and 3D textures.
     */
    inline void attachTexture(const GLenum textureTarget,
                              const GLenum attachment,
                              const GLuint texture,
                              const GLint level = 0);
    
    /*! \fn attachTextureLayer
     *  \brief This method attaches the specified texture's
     *         layer to the given attachment of this Framebuffer.
     *  \param attachment The OpenGL enum of the attachment.
     *  \param texture The OpenGL name of the texture to attach.
     *  \param layer The index of the layer to attach.
     *  \param level The mipmap level to attach; defaults to 0.
     */
    inline void attachTextureLayer(const GLenum attachment,
                                   const GLuint texture,
                                   const GLint layer,
                                   const GLint level = 0);

    /*! \fn attachRenderbuffer
     *  \brief This method attaches the specified renderbuffer
     *         to the given attachment of this Framebuffer.
     *  \param renderbufferTarget The OpenGL target of the renderbuffer
     *                            to attach.
     *  \param renderbuffer The OpenGL name of the renderbuffer
     *                      to attach.
     *  \param attachment The OpenGL enum of the attachment.
     */
    inline void attachRenderbuffer(const GLenum renderbufferTarget,
                                   const GLenum attachment,
                                   const GLuint renderbuffer);

    /*! \fn detach
     *  \brief This method detaches whatever is currently bound to
     *         the given framebuffer attachment.
     *  \param attachment The OpenGL enum of the attachment.
     */
    inline void detach(const GLenum attachment);

}; // end class Framebuffer

#include "Framebuffer.inl"

#endif // FRAMEBUFFER_H

