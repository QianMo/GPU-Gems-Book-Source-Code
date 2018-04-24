/*! \file Program.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         abstracting OpenGL programs.
 */

#ifndef PROGRAM_H
#define PROGRAM_H

#include <GL/glew.h>
#include <map>
#include <iostream>

class Program
{
  public:
    /*! Null constructor sets mIdentifier to 0.
     */
    inline Program(void);

    /*! This method allocates a new OpenGL
     *  identifier for this Program and destroys
     *  the current one, if it exists.
     *  It also attaches the given
     *  shaders and links this Program.
     *  \param vertex The vertex shader to attach.
     *  \param fragment The fragment shader to attach.
     *  \note To use the fixed pipeline functionality
     *        of any shader, pass 0.
     */
    inline bool create(const GLuint vertex,
                       const GLuint fragment);

    /*! This method allocates a new OpenGL
     *  identifier for this Program and destroys
     *  the current one, if it exists.
     *  It also attaches and the given
     *  shaders and links this Program.
     *  \param vertex The vertex shader to attach.
     *  \param geometry The geometry shader to attach.
     *  \param fragment The fragment shader to attach.
     *  \note To use the fixed pipeline functionality
     *        of any shader, pass 0.
     */
    inline bool create(const GLuint vertex,
                       const GLuint geometry,
                       const GLuint fragment);

    /*! This method allocates a new OpenGL
     *  identifier for this Program and destroys
     *  the current one, if it exists.
     */
    inline void create(void);

    /*! This method destroys the current OpenGL
     *  identifier for this Program.
     */
    inline void destroy(void);

    /*! This method returns this Program's identifier.
     *  \return mIdentifier
     */
    inline GLuint getIdentifier(void) const;

    /*! GLuint cast returns mIdentifer.
     *  \return mIdentifer
     */
    inline operator GLuint (void) const;

    /*! This method attaches a shader object
     *  to this Program.
     *  \param s The OpenGL identifier of the
     *         shader object to attach.
     */
    inline void attach(const GLuint s) const;

    /*! This method detaches a shader object
     *  from this Program.
     *  \param s The OpenGL identifier of the
     *         shader object to detach.
     */
    inline void detach(const GLuint s) const;

    /*! This method detaches all shaders attached
     *  to this Program.
     */
    inline void detachShaders(void) const;

    /*! This method links this Program.
     *  \return true if this Program could be
     *          successfully linked; false, otherwise.
     */
    inline bool link(void);

    /*! This method attaches the given shaders
     *  to this Program and links.
     *  \param vertex The vertex shader to attach.
     *  \param fragment The fragment shader to attach.
     *  \note To use the fixed pipeline functionality
     *        of any shader, pass 0.
     */
    inline bool link(const GLuint vertex,
                     const GLuint fragment);

    /*! This method attaches the given shaders
     *  to this Program and links.
     *  \param vertex The vertex shader to attach.
     *  \param geometry The geometry shader to attach.
     *  \param fragment The fragment shader to attach.
     *  \note To use the fixed pipeline functionality
     *        of any shader, pass 0.
     */
    inline bool link(const GLuint vertex,
                     const GLuint geometry,
                     const GLuint fragment);

    /*! This method binds this Program.
     */
    inline void bind(void) const;

    /*! This method unbinds this Program.
     */
    inline void unbind(void) const;

    /*! This method returns this Program's info
     *  log as a string.
     *  \param log This Shader's info log is
     *         returned here.
     */
    inline void getInfoLog(std::string &log) const;

    /*! This method returns the location of a
     *  uniform variable of this Program.
     *  \param name The name of the uniform.
     *  \return The location of the uniform named name.
     */
    inline GLint getUniformLocation(const GLchar *name) const;

    /*! This method returns the location of
     *  attribute variable of this Program.
     *  \param name The name of the attribute.
     *  \return The location of the attribute named name.
     */
    inline GLint getAttribLocation(const GLchar *name) const;

    /*! This method sets a vec4 uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v0 The first element.
     *  \param v1 The second element.
     *  \param v2 The third element.
     *  \param v3 The fourth element.
     */
    inline void setUniform4f(const GLchar *name,
                             const GLfloat v0,
                             const GLfloat v1,
                             const GLfloat v2,
                             const GLfloat v3) const;

    /*! This method sets a vec4 uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v A pointer to a float4 to set to.
     */
    inline void setUniform4fv(const GLchar *name,
                              const GLfloat *v) const;

    /*! This method sets a vec3 uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v0 The first element.
     *  \param v1 The second element.
     *  \param v2 The third element.
     */
    inline void setUniform3f(const GLchar *name,
                             const GLfloat v0,
                             const GLfloat v1,
                             const GLfloat v2) const;

    /*! This method sets a vec3 uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v A pointer to a float3 to set to.
     */
    inline void setUniform3fv(const GLchar *name,
                              const GLfloat *v) const;

    /*! This method sets a vec2 uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v A pointer to a float2 to set to.
     */
    inline void setUniform2fv(const GLchar *name,
                              const GLfloat *v) const;

    /*! This method sets a uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v The int2 to set to.
     */
    inline void setUniform2iv(const GLchar *name,
                              const GLint *v) const;

    /*! This method sets a uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v The float to set to.
     */
    inline void setUniform1f(const GLchar *name,
                             const GLfloat v) const;

    /*! This method sets a uniform variable.
     *  \param name The name of the uniform to set.
     *  \param v The int to set to.
     */
    inline void setUniform1i(const GLchar *name,
                             const GLint v) const;

    /*! This method sets a uniform matrix variable.
     *  \param name The name of the uniform to set.
     *  \param bool Whether or not to transpose the matrix.
     *  \param v The matrix to set to.
     */
    inline void setUniformMatrix4fv(const GLchar *name,
                                    const GLboolean transpose,
                                    const GLfloat *v) const;

    /*! This method sets a parameter of this
     *  Program.
     *  \param pname The name of the parameter to set.
     *  \param value The value to set the parameter to.
     */
    inline void setParameteri(const GLenum pname,
                              const GLint value) const;

  protected:
    /*! This method creates mUniformMap.
     */
    inline void createUniformMap(void);

    /*! This method creates mAttributeMap.
     */
    inline void createAttributeMap(void);

    /*! The OpenGL identifier of this Program.
     */
    GLuint mIdentifier;

    /*! \typedef VariableMap
     *  \brief Shorthand.
     */
    typedef std::map<std::string,GLint> VariableMap;

    /*! A map from uniform variable names to
     *  uniform variable locations.
     */
    VariableMap mUniformMap;

    /*! A map from attribute variable names to
     *  varying variable locations.
     */
    VariableMap mAttributeMap;
}; // end Program

/*! This function outputs the given Program's current info log to the given ostream.
 *  \param os The ostream to output to.
 *  \param p The Program of interest.
 *  \return os
 */
inline std::ostream &operator<<(std::ostream &os, const Program &p);

#include "Program.inl"

#endif // PROGRAM_H

