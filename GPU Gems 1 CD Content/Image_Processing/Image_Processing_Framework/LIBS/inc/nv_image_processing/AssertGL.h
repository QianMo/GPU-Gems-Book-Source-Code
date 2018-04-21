#ifndef ASSERT_GL_H
#define ASSERT_GL_H
// -------------------------------------------------------------------
// Contents:
//      CG_ASSERT_NO_ERROR macro
//          This macro allows for simple Cg error checking.
//
// Author:
//      Frank Jargstorff (2003)
// -------------------------------------------------------------------


void gl_assert(const char * zFile, unsigned int nLine);


#define GL_ASSERT_NO_ERROR {gl_assert(__FILE__, __LINE__);}

#endif // ASSERT_GL_H