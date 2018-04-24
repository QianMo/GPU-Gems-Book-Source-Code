/*! \file DisplayList.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for abstracting OpenGL DisplayList objects.
 */

#ifndef DISPLAY_LIST_H
#define DISPLAY_LIST_H

#include <GL/glew.h>
#include <gl++/globject/GLObject.h>

/*! \fn genDisplayListThunk
 *  \todo XXX Find a way around this.
 */
inline void genDisplayListThunk(GLuint num, GLuint *id)
{
  *id = glGenLists(num);
  for(GLuint i = 1; i < num; ++i)
  {
    id[i] = id[0] + i;
  } // end for i
} // end genDisplayListThunk()

/*! \fn deleteDisplayListThunk
 *  \todo XXX Find a way around this.
 */
inline void deleteDisplayListThunk(GLuint num, GLuint *ids)
{
  for(GLuint i = 0; i < num; ++i)
  {
    glDeleteLists(ids[i], 1);
  } // end for i
} // end deleteDisplayListThunk()

/*! \fn bindDisplayListThunk()
 *  \todo XXX Find a way around this.
 */
inline void bindDisplayListThunk(GLenum compileMode, GLuint id)
{
  if(id == 0)
  {
    // "unbind"
    glEndList();
  } // end if
  else
  {
    // "bind"
    glNewList(id, compileMode);
  } // end else
} // end bindDisplayListThunk()

class DisplayList : public GLObject<genDisplayListThunk,
                                    deleteDisplayListThunk,
                                    bindDisplayListThunk>
{
  public:
    /*! \typdef Parent
     *  \brief Shorthand
     */
    typedef GLObject<genDisplayListThunk,
                     deleteDisplayListThunk,
                     bindDisplayListThunk> Parent;

    /*! \fn DisplayList
     *  \brief Null constructor sets the "target" to GL_COMPILE and
     *         calls the Parent.
     */
    inline DisplayList(void);

    /*! \fn call
     *  \brief This method calls this DisplayList.
     */
    inline void call(void) const;
    
    /*! \fn operator()()
     *  \brief This method calls call().
     */
    inline void operator()(void) const;
}; // end class DisplayList

#include "DisplayList.inl"

#endif // DISPLAY_LIST_H

