
// --------------------------------------------------------
// Generated by glux perl script (Wed Mar 31 17:19:32 2004)
// 
// Sylvain Lefebvre - 2002 - Sylvain.Lefebvre@imag.fr
// --------------------------------------------------------
#include "glux_no_redefine.h"
#include "glux_ext_defs.h"
#include "gluxLoader.h"
#include "gluxPlugin.h"
// --------------------------------------------------------
//         Plugin creation
// --------------------------------------------------------

#ifndef __GLUX_GL_EXT_draw_range_elements__
#define __GLUX_GL_EXT_draw_range_elements__

GLUX_NEW_PLUGIN(GL_EXT_draw_range_elements);
// --------------------------------------------------------
//           Extension conditions
// --------------------------------------------------------
// --------------------------------------------------------
//           Extension defines
// --------------------------------------------------------
#ifndef GL_MAX_ELEMENTS_VERTICES_EXT
#  define GL_MAX_ELEMENTS_VERTICES_EXT 0x80E8
#endif
#ifndef GL_MAX_ELEMENTS_INDICES_EXT
#  define GL_MAX_ELEMENTS_INDICES_EXT 0x80E9
#endif
// --------------------------------------------------------
//           Extension gl function typedefs
// --------------------------------------------------------
#ifndef __GLUX__GLFCT_glDrawRangeElementsEXT
typedef void (APIENTRYP PFNGLUXDRAWRANGEELEMENTSEXTPROC) (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
#endif
// --------------------------------------------------------
//           Extension gl functions
// --------------------------------------------------------
namespace glux {
#ifndef __GLUX__GLFCT_glDrawRangeElementsEXT
extern PFNGLUXDRAWRANGEELEMENTSEXTPROC glDrawRangeElementsEXT;
#endif
} // namespace glux
// --------------------------------------------------------
#endif
// --------------------------------------------------------
