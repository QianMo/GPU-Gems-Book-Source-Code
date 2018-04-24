/*! \file Texture.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for abstracting OpenGL Texture objects.
 */

#ifndef TEXTURE_H
#define TEXTURE_H

#include <GL/glew.h>
#include <gl++/globject/GLObject.h>

/*! \fn genTextureThunk
 *  \todo Find a way around this.
 */
inline void genTextureThunk(GLuint num, GLuint *id)
{
  glGenTextures(num,id);
} // end genTextureThunk()

/*! \fn deleteTextureThunk
 *  \todo Find a way around this.
 */
inline void deleteTextureThunk(GLuint num, GLuint *ids)
{
  glDeleteTextures(num,ids);
} // end deleteTextureThunk()

/*! \fn bindTextureThunk
 *  \todo Find a way around this.
 */
inline void bindTextureThunk(GLenum target, GLuint id)
{
  glBindTexture(target, id);
} // end bindTextureThunk()

class Texture : public GLObject<genTextureThunk,
                                deleteTextureThunk,
                                bindTextureThunk>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef GLObject<genTextureThunk, deleteTextureThunk, bindTextureThunk> Parent;

    /*! \fn Texture
     *  \brief Null constructor calls the parent and sets
     *         mTarget to GL_TEXTURE_RECTANGLE_ARB.
     *         Also:
     *         - calls setWidth(1)
     *         - calls setHeight(1)
     *         - calls setDepth(0)
     *         - calls setBorderSize(0)
     */
    inline Texture(void);

    /*! \fn create
     *  \brief Create creates an OpenGL identifier for this Texture
     *         and sets its target to the one given.
     *  \param target Sets the target of this Texture.
     *         GL_TEXTURE_RECTANGLE_NV by default.
     */
    inline void create(const GLenum target = GL_TEXTURE_RECTANGLE_NV);

    /*! \fn setInternalFormat
     *  \brief This method sets mInternalFormat.
     *  \param f Sets mInternalFormat
     */
    inline void setInternalFormat(const GLenum f);

    /*! \fn getInternalFormat
     *  \brief This method returns mInternalFormat
     *  \return mInternalFormat
     */
    inline GLenum getInternalFormat(void) const;

    /*! \fn getWidth
     *  \brief This method returns mWidth.
     *  \return mWidth
     */
    inline GLsizei getWidth(void) const;

    /*! \fn getMaxS
     *  \brief This method returns the maximum texture coordinate
     *         addressing this Texture's width dimension.
     *  \return 1.0 for 2D texture; getWidth() for rectangular.
     */
    inline GLsizei getMaxS(void) const;

    /*! \fn getHeight
     *  \brief This method returns mHeight.
     *  \return mHeight
     */
    inline GLsizei getHeight(void) const;

    /*! \fn getMaxT
     *  \brief This method returns the maximum texture coordinate
     *         addressing this Texture's height dimension.
     *  \return 1.0 for 2D texture; getHeight() for rectangular.
     */
    inline GLsizei getMaxT(void) const;

    /*! \fn getDepth
     *  \brief This method returns mDepth.
     *  \return mDepth.
     */
    inline GLsizei getDepth(void) const;

    /*! \fn getBorderSize
     *  \brief This method returns mBorderSize.
     *  \return mBorderSize
     */
    inline GLint getBorderSize(void) const;

    /*! \fn setTarget
     *  \brief This method sets mTarget.
     *  \param t Sets Parent::mTarget
     *  \note It makes sense to export setTarget()
     *        as public since the user needs to know
     *        about them in order to specify texture
     *        coordinates correctly.
     */
    inline void setTarget(const GLenum t);

    /*! \fn init
     *  \brief This method allocates 1-dimensional storage for
     *         this Texture.
     *  \param internalFormat The internal format of this Texture.
     *  \param width The width in texels to alocate.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    inline void init(const GLenum internalFormat,
                     const GLsizei width);

    /*! \fn init
     *  \brief This method allocates storage for this Texture.
     *  \param internalFormat The internal format of this Texture.
     *  \param width The width in texels to allocate.
     *  \param height The height in texels to allocate.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    inline void init(const GLenum internalFormat,
                     const GLsizei width,
                     const GLsizei height);

    /*! \fn texImage3D
     *  \brief This method allocates storage for this Texture.
     *  \param internalFormat The internal format of this Texture.
     *  \param width The width in texels to allocate.
     *  \param height The height in texels to allocate.
     *  \param depth The depth in texels to allocate.
     *  \note Also calls the appropriate methods to set each member
     *        as specified by the parameters.
     */
    inline void texImage3D(const GLenum internalFormat,
                           const GLsizei width,
                           const GLsizei height,
                           const GLsizei depth);

    /*! \fn init
     *  \brief This method allocates storage for this Texture.
     *  \param internalFormat The internal format of this Texture.
     *  \param width The width in texels to allocate.
     *  \param height The height in texels to allocate.
     *  \param border The width in texels to allocate for a border.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    inline void init(const GLenum internalFormat,
                     const GLsizei width,
                     const GLsizei height,
                     const GLint border);

    /*! \fn texImage3D
     *  \brief This method allocates storage for this Texture.
     *  \param internalFormat The internal format of this Texture.
     *  \param width The width in texels to allocate.
     *  \param height The height in texels to allocate.
     *  \param depth The depth in texels to allocate.
     *  \param border The width in texels to allocate for a border.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    inline void texImage3D(const GLenum internalFormat,
                           const GLsizei width,
                           const GLsizei height,
                           const GLsizei depth,
                           const GLint border);

    /*! \fn init
     *  \brief This method uplaods the given pixel data to this
     *         Texture.  The Texture is allocated as 1D.
     *  \param internalFormat the internal format of this Texture.
     *  \param width The width in texels to upload.
     *  \param border The size in texels to allocate for a border.
     *  \param externalFormat The format of the pixel data in pixels.
     *  \param pixels A pointer to the pixel data.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    template<typename T>
      inline void init(const GLenum internalFormat,
                       const GLsizei width,
                       const GLint border,
                       const GLenum externalFormat,
                       const T *pixels);

    /*! \fn init
     *  \brief This method uploads the given pixel data to this Texture.
     *  \param internalFormat The internal format of this Texture.
     *  \param width The width in texels to upload.
     *  \param height The height in texels to upload.
     *  \param border The size in texels to allocate for a border.
     *  \param externalFormat The format of the data in pixels.
     *  \param pixels A pointer to the pixel data.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    template<typename T>
      inline void init(const GLenum internalFormat,
                       const GLsizei width,
                       const GLsizei height,
                       const GLint border,
                       const GLenum externalFormat,
                       const T *pixels);

    /*! \fn texImage3D
     *  \brief This method uploads the given pixel data to this Texture.
     *         The Texture is allocated as 3D.
     *  \param internalFormat THe internal format of this Texture.
     *  \param width The width in texels to upload.
     *  \param height The height in texels to upload.
     *  \param depth The depth in texels to upload.
     *  \param border The size in texels to allocate for a border.
     *  \param externalFormat The format of the data in pixels.
     *  \param texels A pointer to the texel data.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    template<typename T>
      inline void texImage3D(const GLenum internalFormat,
                             const GLsizei width,
                             const GLsizei height,
                             const GLsizei depth,
                             const GLint border,
                             const GLenum externalFormat,
                             const T *pixels);

    /*! \fn getPixels
     *  \brief This method returns mipmap level 0's pixels as converted to
     *         the specified Type.
     *  \param format The external format to convert to.
     *  \param p The pixels are returned here.
     */
    template<typename Type> inline void getPixels(const GLenum format, Type *p) const;

    /*! \fn generateMipmap
     *  \brief This method automatically generates a mipmap for this Texture.
     */
    inline void generateMipmap(void) const;

  protected:
    /*! \class ExternalFormat
     *  \brief This type casts to the correct GLenum for an
     *         external format given a type.
     */
    template<typename Type> class ExternalFormat
    {
      public:
        /*! Cast to GLenum operator
         *  \brief This operator returns the correct
         *         GLenum of the external format given Type.
         *  \return As above.
         */
        inline operator GLenum (void) const;
    }; // end class ExternalFormat

    /*! \fn setWidth
     *  \brief This method sets mWidth.
     *  \param w Sets mWidth
     */
    inline void setWidth(const GLsizei w);

    /*! \fn setHeight
     *  \brief This method sets mHeight.
     *  \param h Sets mHeight
     */
    inline void setHeight(const GLsizei h);

    /*! \fn setDepth
     *  \brief This method sets mDepth.
     *  \param d Sets mDepth.
     */
    inline void setDepth(const GLsizei d);

    /*! \fn setBorderSize
     *  \brief This method sets mBorderSize.
     *  \param b Sets mBorderSize
     */
    inline void setBorderSize(const GLint b);

    /*! A Texture has an internal format.
     */
    GLenum mInternalFormat;

    /*! A Texture has a width in texels.
     */
    GLsizei mWidth;

    /*! A Texture has a height in texels.
     */
    GLsizei mHeight;

    /*! A Texture has a depth in texels.
     */
    GLsizei mDepth;

    /*! A Texture has a border size.
     */
    GLint mBorderSize;
}; // end class Texture

#include "Texture.inl"

#endif // TEXTURE_H

