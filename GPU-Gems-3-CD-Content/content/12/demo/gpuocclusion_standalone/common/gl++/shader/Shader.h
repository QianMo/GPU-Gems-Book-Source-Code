/*! \file Shader.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for
 *         abstracting OpenGL Shader objects.
 */

#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>
#include <iostream>

class Shader
{
  public:
    /*! Null constructor sets mIdentifier and mType to 0.
     */
    inline Shader(void);

    /*! This constructor sets mIdentifier to 0 and
     *  mType as given.
     *  \param type Sets mType.
     */
    inline Shader(const GLenum type);

    /*! This method allocates a new OpenGL identifier
     *  for this Shader and destroys the curent one,
     *  if it exists.
     */
    inline void create(void);

    /*! This method allocates a new OpenGL identifier
     *  for this Shader and destroys the current one,
     *  if it exists.
     *  \param type Sets mType.
     */
    inline void create(const GLenum type);

    /*! This method allocates a new OpenGL identifier
     *  for this Shader and compiles from the source
     *  in the given string.
     *  \param type Sets mType.
     *  \param source The source code for this shader.
     *  \return true if this Shader could be sucessfully
     *          created and compiled; false, otherwise.
     */
    inline bool create(const GLenum type,
                       const char *source);

    /*! This method allocates a new OpenGL identifier
     *  for this Shader and compiles from the given
     *  source file.
     *  \param type Sets mType.
     *  \param filename The name of the source file on
     *         disk.
     *  \return true if this Shader could be successfully
     *          created and compiled; false, otherwise.
     */
    inline bool createFromFile(const GLenum type,
                               const char *filename);

    /*! This method destroys the current OpenGL
     *  identifier for this Shader if it exists.
     */
    inline void destroy(void);

    /*! This method returns this Program's identifier.
     *  \return mIdentifier
     */
    inline GLuint getIdentifier(void) const;

    /*! This method returns this Shader's type.
     *  \return mType
     */
    inline GLenum getType(void) const;

    /*! GLuint cast returns mIdentifer.
     *  \return mIdentifer
     */
    inline operator GLuint (void) const;

    /*! This method compiles this Shader given
     *  a source string.
     *  \param source A string containing the source
     *         of this Shader to compile.
     *  \return true if source could be compiled
     *          sucessfully; false, otherwise.
     */
    inline bool compile(const char *source);

    /*! This method compiles this Shader given
     *  the filename of a source file on disk.
     *  \param filename The filename of the source
     *         to compile.
     *  \return true if source could be loaded
     *          and compiled sucessfully; false,
     *          otherwise.
     */
    inline bool compileFromFile(const char *filename);

    /*! This method compiles this Shader from its
     *  current source code.
     *  \return true if this Shader's source could be
     *          successfully compiled; false, otherwise.
     */
    inline bool compile(void);

    /*! This method returns this Shader's info
     *  log as a string.
     *  \param log This Shader's info log is
     *         returned here.
     */
    inline void getInfoLog(std::string &log) const;

    /*! This method returns this Shader's source
     *  \param source This Shader's source is returned here.
     */
    inline void getSource(std::string &source) const;

    /*! This method sets this Shader's source.
     *  \param source This Shader's source.
     */
    inline void setSource(const char *source) const;

    /*! This method sets this Shader's source
     *  from a file on disk.
     *  \param filename The name of the file containing
     *         this Shader's source code.
     *  \return true if this Shader's source code could
     *          be successfully set from disk; false,
     *          otherwise.
     */
    inline bool setSourceFromFile(const char *filename) const;

  protected:
    /*! The OpenGL identifier of this Shader.
     */
    GLuint mIdentifier;

    /*! The type of this Shader.
     */
    GLenum mType;
}; // end Shader

/*! This function outputs the given Shader's current info log to the given ostream.
 *  \param os The stream to output to.
 *  \param s The Shader of interest.
 *  \return os
 */
inline std::ostream &operator<<(std::ostream &os, const Shader &s);

#include "Shader.inl"

#endif // SHADER_H

