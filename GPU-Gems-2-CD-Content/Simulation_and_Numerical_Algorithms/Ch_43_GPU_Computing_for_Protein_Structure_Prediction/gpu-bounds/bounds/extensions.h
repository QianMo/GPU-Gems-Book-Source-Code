#ifndef _MP_EXTENSIONS_H
#define	_MP_EXTENSIONS_H

#include "glext-nv.h"
#include "wglext.h"

PFNGLGENOCCLUSIONQUERIESNVPROC		glGenOcclusionQueriesNV;
PFNGLDELETEOCCLUSIONQUERIESNVPROC	glDeleteOcclusionQueriesNV;
PFNGLISOCCLUSIONQUERYNVPROC			glIsOcclusionQueryNV;
PFNGLBEGINOCCLUSIONQUERYNVPROC		glBeginOcclusionQueryNV;
PFNGLENDOCCLUSIONQUERYNVPROC		glEndOcclusionQueryNV;
PFNGLGETOCCLUSIONQUERYIVNVPROC		glGetOcclusionQueryivNV;
PFNGLGETOCCLUSIONQUERYUIVNVPROC		glGetOcclusionQueryuivNV;

PFNWGLALLOCATEMEMORYNVPROC			wglAllocateMemoryNV;
PFNWGLFREEMEMORYNVPROC				wglFreeMemoryNV;

PFNGLVERTEXARRAYRANGENVPROC			glVertexArrayRangeNV;
PFNGLFLUSHVERTEXARRAYRANGENVPROC	glFlushVertexArrayRangeNV;

PFNGLARRAYELEMENTEXTPROC			glArrayElementEXT;
PFNGLDRAWARRAYSEXTPROC				glDrawArraysEXT;
PFNGLVERTEXPOINTEREXTPROC			glVertexPointerEXT;
PFNGLNORMALPOINTEREXTPROC			glNormalPointerEXT;
PFNGLCOLORPOINTEREXTPROC			glColorPointerEXT;
PFNGLINDEXPOINTEREXTPROC			glIndexPointerEXT;
PFNGLTEXCOORDPOINTEREXTPROC			glTexCoordPointerEXT;
PFNGLEDGEFLAGPOINTEREXTPROC			glEdgeFlagPointerEXT;
PFNGLGETPOINTERVEXTPROC				glGetPointervEXT;

PFNWGLCREATEPBUFFERARBPROC			wglCreatePbufferARB;
PFNWGLGETPBUFFERDCARBPROC			wglGetPbufferDCARB;
PFNWGLRELEASEPBUFFERDCARBPROC		wglReleasePbufferDCARB;
PFNWGLDESTROYPBUFFERARBPROC			wglDestroyPbufferARB;
PFNWGLQUERYPBUFFERARBPROC			wglQueryPbufferARB;

PFNWGLGETPIXELFORMATATTRIBIVARBPROC	wglGetPixelFormatAttribivARB;
PFNWGLGETPIXELFORMATATTRIBFVARBPROC	wglGetPixelFormatAttribfvARB;
PFNWGLCHOOSEPIXELFORMATARBPROC		wglChoosePixelFormatARB;

PFNWGLBINDTEXIMAGEARBPROC			wglBindTexImageARB;
PFNWGLRELEASETEXIMAGEARBPROC		wglReleaseTexImageARB;
PFNWGLSETPBUFFERATTRIBARBPROC		wglSetPbufferAttribARB;

PFNGLBLENDEQUATIONEXTPROC			glBlendEquationEXT=NULL;

PFNGLACTIVETEXTUREARBPROC			glActiveTextureARB;
PFNGLMULTITEXCOORD2FARBPROC			glMultiTexCoord2fARB;
PFNGLMULTITEXCOORD4FARBPROC			glMultiTexCoord4fARB;

int PrepareExtensionFunctions()
{
	int retval=1;

	retval&=(NULL!=(glGenOcclusionQueriesNV=(PFNGLGENOCCLUSIONQUERIESNVPROC)wglGetProcAddress("glGenOcclusionQueriesNV")));
	retval&=(NULL!=(glDeleteOcclusionQueriesNV=(PFNGLDELETEOCCLUSIONQUERIESNVPROC)wglGetProcAddress("glDeleteOcclusionQueriesNV")));
	retval&=(NULL!=(glIsOcclusionQueryNV=(PFNGLISOCCLUSIONQUERYNVPROC)wglGetProcAddress("glIsOcclusionQueryNV")));
	retval&=(NULL!=(glBeginOcclusionQueryNV=(PFNGLBEGINOCCLUSIONQUERYNVPROC)wglGetProcAddress("glBeginOcclusionQueryNV")));
	retval&=(NULL!=(glEndOcclusionQueryNV=(PFNGLENDOCCLUSIONQUERYNVPROC)wglGetProcAddress("glEndOcclusionQueryNV")));
	retval&=(NULL!=(glGetOcclusionQueryivNV=(PFNGLGETOCCLUSIONQUERYIVNVPROC)wglGetProcAddress("glGetOcclusionQueryivNV")));
	retval&=(NULL!=(glGetOcclusionQueryuivNV=(PFNGLGETOCCLUSIONQUERYUIVNVPROC)wglGetProcAddress("glGetOcclusionQueryuivNV")));

	retval&=(NULL!=(wglAllocateMemoryNV=(PFNWGLALLOCATEMEMORYNVPROC)wglGetProcAddress("wglAllocateMemoryNV")));
	retval&=(NULL!=(wglFreeMemoryNV=(PFNWGLFREEMEMORYNVPROC)wglGetProcAddress("wglFreeMemoryNV")));

	retval&=(NULL!=(glVertexArrayRangeNV=(PFNGLVERTEXARRAYRANGENVPROC)wglGetProcAddress("glVertexArrayRangeNV")));
	retval&=(NULL!=(glFlushVertexArrayRangeNV=(PFNGLFLUSHVERTEXARRAYRANGENVPROC)wglGetProcAddress("glFlushVertexArrayRangeNV")));

	retval&=(NULL!=(glArrayElementEXT=(PFNGLARRAYELEMENTEXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glDrawArraysEXT=(PFNGLDRAWARRAYSEXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glVertexPointerEXT=(PFNGLVERTEXPOINTEREXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glNormalPointerEXT=(PFNGLNORMALPOINTEREXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glColorPointerEXT=(PFNGLCOLORPOINTEREXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glIndexPointerEXT=(PFNGLINDEXPOINTEREXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glTexCoordPointerEXT=(PFNGLTEXCOORDPOINTEREXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glEdgeFlagPointerEXT=(PFNGLEDGEFLAGPOINTEREXTPROC)wglGetProcAddress("glArrayElementEXT")));
	retval&=(NULL!=(glGetPointervEXT=(PFNGLGETPOINTERVEXTPROC)wglGetProcAddress("glGetPointervEXT")));

	retval&=(NULL!=(wglCreatePbufferARB=(PFNWGLCREATEPBUFFERARBPROC)wglGetProcAddress("wglCreatePbufferARB")));
	retval&=(NULL!=(wglGetPbufferDCARB=(PFNWGLGETPBUFFERDCARBPROC)wglGetProcAddress("wglGetPbufferDCARB")));
	retval&=(NULL!=(wglReleasePbufferDCARB=(PFNWGLRELEASEPBUFFERDCARBPROC)wglGetProcAddress("wglReleasePbufferDCARB")));
	retval&=(NULL!=(wglDestroyPbufferARB=(PFNWGLDESTROYPBUFFERARBPROC)wglGetProcAddress("wglDestroyPbufferARB")));
	retval&=(NULL!=(wglQueryPbufferARB=(PFNWGLQUERYPBUFFERARBPROC)wglGetProcAddress("wglQueryPbufferARB")));

	retval&=(NULL!=(wglGetPixelFormatAttribivARB=(PFNWGLGETPIXELFORMATATTRIBIVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribivARB")));
	retval&=(NULL!=(wglGetPixelFormatAttribfvARB=(PFNWGLGETPIXELFORMATATTRIBFVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribfvARB")));
	retval&=(NULL!=(wglChoosePixelFormatARB=(PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB")));

	retval&=(NULL!=(wglBindTexImageARB=(PFNWGLBINDTEXIMAGEARBPROC)wglGetProcAddress("wglBindTexImageARB")));
	retval&=(NULL!=(wglReleaseTexImageARB=(PFNWGLRELEASETEXIMAGEARBPROC)wglGetProcAddress("wglReleaseTexImageARB")));
	retval&=(NULL!=(wglSetPbufferAttribARB=(PFNWGLSETPBUFFERATTRIBARBPROC)wglGetProcAddress("wglSetPbufferAttribARB")));

	retval&=(NULL!=(glBlendEquationEXT=(PFNGLBLENDEQUATIONEXTPROC)wglGetProcAddress("glBlendEquationEXT")));

	retval&=(NULL!=(glActiveTextureARB=(PFNGLACTIVETEXTUREARBPROC)wglGetProcAddress("glActiveTextureARB")));
	retval&=(NULL!=(glMultiTexCoord2fARB=(PFNGLMULTITEXCOORD2FARBPROC)wglGetProcAddress("glMultiTexCoord2fARB")));
	retval&=(NULL!=(glMultiTexCoord4fARB=(PFNGLMULTITEXCOORD4FARBPROC)wglGetProcAddress("glMultiTexCoord4fARB")));

	return retval;
}

#endif