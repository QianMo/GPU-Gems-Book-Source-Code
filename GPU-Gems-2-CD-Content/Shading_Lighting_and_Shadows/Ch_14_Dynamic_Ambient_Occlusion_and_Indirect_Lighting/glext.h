#ifndef __glext_h_
#define __glext_h_

/*
** Copyright 1998-2002, NVIDIA Corporation.
** All Rights Reserved.
** 
** THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
** NVIDIA, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
** IS SUBJECT TO WRITTEN PRE-APPROVAL BY NVIDIA, CORPORATION.
** 
** 
** Copyright 1992-1999, Silicon Graphics, Inc.
** All Rights Reserved.
** 
** Portions of this file are UNPUBLISHED PROPRIETARY SOURCE CODE of Silicon
** Graphics, Inc.; the contents of this file may not be disclosed to third
** parties, copied or duplicated in any form, in whole or in part, without
** the prior written permission of Silicon Graphics, Inc.
** 
** RESTRICTED RIGHTS LEGEND:
** Use, duplication or disclosure by the Government is subject to
** restrictions as set forth in subdivision (c)(1)(ii) of the Rights in
** Technical Data and Computer Software clause at DFARS 252.227-7013,
** and/or in similar or successor clauses in the FAR, DOD or NASA FAR
** Supplement.  Unpublished - rights reserved under the Copyright Laws of
** the United States.
*/

#if defined(_WIN32) && !defined(APIENTRY) && !defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GLAPI
# ifdef _WIN32
#  define GLAPI __stdcall
# else
#  define GLAPI
# endif
#endif

/*************************************************************/

#ifndef GL_TYPEDEFS_1_5
#define GL_TYPEDEFS_1_5
#if defined(_WIN64)
    typedef __int64 GLintptr;
    typedef __int64 GLsizeiptr;
#elif defined(__ia64__) || defined(__x86_64__)
    typedef long int GLintptr;
    typedef long int GLsizeiptr;
#else
    typedef int GLintptr;
    typedef int GLsizeiptr;
#endif
#endif

typedef unsigned short GLhalf;
typedef unsigned int GLhandleARB;
typedef char GLcharARB;
#if defined(_WIN64)
    typedef __int64 GLintptrARB;
    typedef __int64 GLsizeiptrARB;
#elif defined(__ia64__) || defined(__x86_64__)
    typedef long int GLintptrARB;
    typedef long int GLsizeiptrARB;
#else
    typedef int GLintptrARB;
    typedef int GLsizeiptrARB;
#endif


#ifndef GL_VERSION_1_2
#define GL_VERSION_1_2 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBlendColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
void GLAPI glBlendEquation (GLenum mode);
void GLAPI glDrawRangeElements (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
void GLAPI glColorTable (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *table);
void GLAPI glColorTableParameterfv (GLenum target, GLenum pname, const GLfloat *params);
void GLAPI glColorTableParameteriv (GLenum target, GLenum pname, const GLint *params);
void GLAPI glCopyColorTable (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
void GLAPI glGetColorTable (GLenum target, GLenum format, GLenum type, GLvoid *table);
void GLAPI glGetColorTableParameterfv (GLenum target, GLenum pname, GLfloat *params);
void GLAPI glGetColorTableParameteriv (GLenum target, GLenum pname, GLint *params);
void GLAPI glTexImage3D (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
void GLAPI glTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);
void GLAPI glCopyTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBLENDCOLORPROC) (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
typedef void (GLAPI * PFNGLBLENDEQUATIONPROC) (GLenum mode);
typedef void (GLAPI * PFNGLDRAWRANGEELEMENTSPROC) (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
typedef void (GLAPI * PFNGLCOLORTABLEPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *table);
typedef void (GLAPI * PFNGLCOLORTABLEPARAMETERFVPROC) (GLenum target, GLenum pname, const GLfloat *params);
typedef void (GLAPI * PFNGLCOLORTABLEPARAMETERIVPROC) (GLenum target, GLenum pname, const GLint *params);
typedef void (GLAPI * PFNGLCOPYCOLORTABLEPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
typedef void (GLAPI * PFNGLGETCOLORTABLEPROC) (GLenum target, GLenum format, GLenum type, GLvoid *table);
typedef void (GLAPI * PFNGLGETCOLORTABLEPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETCOLORTABLEPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLTEXIMAGE3DPROC) (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
typedef void (GLAPI * PFNGLTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);
typedef void (GLAPI * PFNGLCOPYTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
#endif


#ifndef GL_ARB_imaging
#define GL_ARB_imaging 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glColorSubTable (GLenum target, GLsizei start, GLsizei count, GLenum format, GLenum type, const GLvoid *data);
void GLAPI glCopyColorSubTable (GLenum target, GLsizei start, GLint x, GLint y, GLsizei width);
void GLAPI glConvolutionFilter1D (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *image);
void GLAPI glConvolutionFilter2D (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *image);
void GLAPI glConvolutionParameterf (GLenum target, GLenum pname, GLfloat params);
void GLAPI glConvolutionParameterfv (GLenum target, GLenum pname, const GLfloat *params);
void GLAPI glConvolutionParameteri (GLenum target, GLenum pname, GLint params);
void GLAPI glConvolutionParameteriv (GLenum target, GLenum pname, const GLint *params);
void GLAPI glCopyConvolutionFilter1D (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
void GLAPI glCopyConvolutionFilter2D (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height);
void GLAPI glGetConvolutionFilter (GLenum target, GLenum format, GLenum type, GLvoid *image);
void GLAPI glGetConvolutionParameterfv (GLenum target, GLenum pname, GLfloat *params);
void GLAPI glGetConvolutionParameteriv (GLenum target, GLenum pname, GLint *params);
void GLAPI glGetSeparableFilter (GLenum target, GLenum format, GLenum type, GLvoid *row, GLvoid *column, GLvoid *span);
void GLAPI glSeparableFilter2D (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *row, const GLvoid *column);
void GLAPI glGetHistogram (GLenum target, GLboolean reset, GLenum format, GLenum type, GLvoid *values);
void GLAPI glGetHistogramParameterfv (GLenum target, GLenum pname, GLfloat *params);
void GLAPI glGetHistogramParameteriv (GLenum target, GLenum pname, GLint *params);
void GLAPI glGetMinmax (GLenum target, GLboolean reset, GLenum format, GLenum type, GLvoid *values);
void GLAPI glGetMinmaxParameterfv (GLenum target, GLenum pname, GLfloat *params);
void GLAPI glGetMinmaxParameteriv (GLenum target, GLenum pname, GLint *params);
void GLAPI glHistogram (GLenum target, GLsizei width, GLenum internalformat, GLboolean sink);
void GLAPI glMinmax (GLenum target, GLenum internalformat, GLboolean sink);
void GLAPI glResetHistogram (GLenum target);
void GLAPI glResetMinmax (GLenum target);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLCOLORSUBTABLEPROC) (GLenum target, GLsizei start, GLsizei count, GLenum format, GLenum type, const GLvoid *data);
typedef void (GLAPI * PFNGLCOPYCOLORSUBTABLEPROC) (GLenum target, GLsizei start, GLint x, GLint y, GLsizei width);
typedef void (GLAPI * PFNGLCONVOLUTIONFILTER1DPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *image);
typedef void (GLAPI * PFNGLCONVOLUTIONFILTER2DPROC) (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *image);
typedef void (GLAPI * PFNGLCONVOLUTIONPARAMETERFPROC) (GLenum target, GLenum pname, GLfloat params);
typedef void (GLAPI * PFNGLCONVOLUTIONPARAMETERFVPROC) (GLenum target, GLenum pname, const GLfloat *params);
typedef void (GLAPI * PFNGLCONVOLUTIONPARAMETERIPROC) (GLenum target, GLenum pname, GLint params);
typedef void (GLAPI * PFNGLCONVOLUTIONPARAMETERIVPROC) (GLenum target, GLenum pname, const GLint *params);
typedef void (GLAPI * PFNGLCOPYCONVOLUTIONFILTER1DPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
typedef void (GLAPI * PFNGLCOPYCONVOLUTIONFILTER2DPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (GLAPI * PFNGLGETCONVOLUTIONFILTERPROC) (GLenum target, GLenum format, GLenum type, GLvoid *image);
typedef void (GLAPI * PFNGLGETCONVOLUTIONPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETCONVOLUTIONPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETSEPARABLEFILTERPROC) (GLenum target, GLenum format, GLenum type, GLvoid *row, GLvoid *column, GLvoid *span);
typedef void (GLAPI * PFNGLSEPARABLEFILTER2DPROC) (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *row, const GLvoid *column);
typedef void (GLAPI * PFNGLGETHISTOGRAMPROC) (GLenum target, GLboolean reset, GLenum format, GLenum type, GLvoid *values);
typedef void (GLAPI * PFNGLGETHISTOGRAMPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETHISTOGRAMPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETMINMAXPROC) (GLenum target, GLboolean reset, GLenum format, GLenum type, GLvoid *values);
typedef void (GLAPI * PFNGLGETMINMAXPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETMINMAXPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLHISTOGRAMPROC) (GLenum target, GLsizei width, GLenum internalformat, GLboolean sink);
typedef void (GLAPI * PFNGLMINMAXPROC) (GLenum target, GLenum internalformat, GLboolean sink);
typedef void (GLAPI * PFNGLRESETHISTOGRAMPROC) (GLenum target);
typedef void (GLAPI * PFNGLRESETMINMAXPROC) (GLenum target);
#endif


#ifndef GL_VERSION_1_3
#define GL_VERSION_1_3 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glActiveTexture (GLenum texture);
void GLAPI glClientActiveTexture (GLenum texture);
void GLAPI glMultiTexCoord1d (GLenum target, GLdouble s);
void GLAPI glMultiTexCoord1dv (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord1f (GLenum target, GLfloat s);
void GLAPI glMultiTexCoord1fv (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord1i (GLenum target, GLint s);
void GLAPI glMultiTexCoord1iv (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord1s (GLenum target, GLshort s);
void GLAPI glMultiTexCoord1sv (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord2d (GLenum target, GLdouble s, GLdouble t);
void GLAPI glMultiTexCoord2dv (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord2f (GLenum target, GLfloat s, GLfloat t);
void GLAPI glMultiTexCoord2fv (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord2i (GLenum target, GLint s, GLint t);
void GLAPI glMultiTexCoord2iv (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord2s (GLenum target, GLshort s, GLshort t);
void GLAPI glMultiTexCoord2sv (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord3d (GLenum target, GLdouble s, GLdouble t, GLdouble r);
void GLAPI glMultiTexCoord3dv (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord3f (GLenum target, GLfloat s, GLfloat t, GLfloat r);
void GLAPI glMultiTexCoord3fv (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord3i (GLenum target, GLint s, GLint t, GLint r);
void GLAPI glMultiTexCoord3iv (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord3s (GLenum target, GLshort s, GLshort t, GLshort r);
void GLAPI glMultiTexCoord3sv (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord4d (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
void GLAPI glMultiTexCoord4dv (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord4f (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
void GLAPI glMultiTexCoord4fv (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord4i (GLenum target, GLint s, GLint t, GLint r, GLint q);
void GLAPI glMultiTexCoord4iv (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord4s (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
void GLAPI glMultiTexCoord4sv (GLenum target, const GLshort *v);
void GLAPI glLoadTransposeMatrixf (const GLfloat *m);
void GLAPI glLoadTransposeMatrixd (const GLdouble *m);
void GLAPI glMultTransposeMatrixf (const GLfloat *m);
void GLAPI glMultTransposeMatrixd (const GLdouble *m);
void GLAPI glCompressedTexImage3D (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexImage2D (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexImage1D (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexSubImage3D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexSubImage1D (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data);
void GLAPI glGetCompressedTexImage (GLenum target, GLint lod, GLvoid *img);
void GLAPI glSampleCoverage (GLclampf value, GLboolean invert);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLACTIVETEXTUREPROC) (GLenum texture);
typedef void (GLAPI * PFNGLCLIENTACTIVETEXTUREPROC) (GLenum texture);
typedef void (GLAPI * PFNGLMULTITEXCOORD1DPROC) (GLenum target, GLdouble s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1FPROC) (GLenum target, GLfloat s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1IPROC) (GLenum target, GLint s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1SPROC) (GLenum target, GLshort s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2DPROC) (GLenum target, GLdouble s, GLdouble t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2FPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2IPROC) (GLenum target, GLint s, GLint t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2SPROC) (GLenum target, GLshort s, GLshort t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3DPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3FPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3IPROC) (GLenum target, GLint s, GLint t, GLint r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3SPROC) (GLenum target, GLshort s, GLshort t, GLshort r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4DPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4FPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4IPROC) (GLenum target, GLint s, GLint t, GLint r, GLint q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4SPROC) (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLLOADTRANSPOSEMATRIXFPROC) (const GLfloat *m);
typedef void (GLAPI * PFNGLLOADTRANSPOSEMATRIXDPROC) (const GLdouble *m);
typedef void (GLAPI * PFNGLMULTTRANSPOSEMATRIXFPROC) (const GLfloat *m);
typedef void (GLAPI * PFNGLMULTTRANSPOSEMATRIXDPROC) (const GLdouble *m);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXIMAGE3DPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXIMAGE2DPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXIMAGE1DPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC) (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLGETCOMPRESSEDTEXIMAGEPROC) (GLenum target, GLint lod, GLvoid *img);
typedef void (GLAPI * PFNGLSAMPLECOVERAGEPROC) (GLclampf value, GLboolean invert);
#endif


#ifndef GL_VERSION_1_4
#define GL_VERSION_1_4 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glPointParameterf (GLenum pname, GLfloat param);
void GLAPI glPointParameterfv (GLenum pname, const GLfloat *params);
void GLAPI glPointParameteri (GLenum pname, GLint param);
void GLAPI glPointParameteriv (GLenum pname, const GLint *params);
void GLAPI glWindowPos2d (GLdouble x, GLdouble y);
void GLAPI glWindowPos2f (GLfloat x, GLfloat y);
void GLAPI glWindowPos2i (GLint x, GLint y);
void GLAPI glWindowPos2s (GLshort x, GLshort y);
void GLAPI glWindowPos2dv (const GLdouble *p);
void GLAPI glWindowPos2fv (const GLfloat *p);
void GLAPI glWindowPos2iv (const GLint *p);
void GLAPI glWindowPos2sv (const GLshort *p);
void GLAPI glWindowPos3d (GLdouble x, GLdouble y, GLdouble z);
void GLAPI glWindowPos3f (GLfloat x, GLfloat y, GLfloat z);
void GLAPI glWindowPos3i (GLint x, GLint y, GLint z);
void GLAPI glWindowPos3s (GLshort x, GLshort y, GLshort z);
void GLAPI glWindowPos3dv (const GLdouble *p);
void GLAPI glWindowPos3fv (const GLfloat *p);
void GLAPI glWindowPos3iv (const GLint *p);
void GLAPI glWindowPos3sv (const GLshort *p);
void GLAPI glBlendFuncSeparate (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
void GLAPI glFogCoordd (GLdouble fog);
void GLAPI glFogCoorddv (const GLdouble *fog);
void GLAPI glFogCoordf (GLfloat fog);
void GLAPI glFogCoordfv (const GLfloat *fog);
void GLAPI glFogCoordPointer (GLenum type, GLsizei stride, const GLvoid *pointer);
void GLAPI glMultiDrawArrays (GLenum mode, const GLint *first, const GLsizei *count, GLsizei primcount);
void GLAPI glMultiDrawElements (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* *indices, GLsizei primcount);
void GLAPI glSecondaryColor3b (GLbyte red, GLbyte green, GLbyte blue);
void GLAPI glSecondaryColor3bv (const GLbyte *v);
void GLAPI glSecondaryColor3d (GLdouble red, GLdouble green, GLdouble blue);
void GLAPI glSecondaryColor3dv (const GLdouble *v);
void GLAPI glSecondaryColor3f (GLfloat red, GLfloat green, GLfloat blue);
void GLAPI glSecondaryColor3fv (const GLfloat *v);
void GLAPI glSecondaryColor3i (GLint red, GLint green, GLint blue);
void GLAPI glSecondaryColor3iv (const GLint *v);
void GLAPI glSecondaryColor3s (GLshort red, GLshort green, GLshort blue);
void GLAPI glSecondaryColor3sv (const GLshort *v);
void GLAPI glSecondaryColor3ub (GLubyte red, GLubyte green, GLubyte blue);
void GLAPI glSecondaryColor3ubv (const GLubyte *v);
void GLAPI glSecondaryColor3ui (GLuint red, GLuint green, GLuint blue);
void GLAPI glSecondaryColor3uiv (const GLuint *v);
void GLAPI glSecondaryColor3us (GLushort red, GLushort green, GLushort blue);
void GLAPI glSecondaryColor3usv (const GLushort *v);
void GLAPI glSecondaryColorPointer (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLPOINTPARAMETERFPROC) (GLenum pname, GLfloat param);
typedef void (GLAPI * PFNGLPOINTPARAMETERFVPROC) (GLenum pname, const GLfloat *params);
typedef void (GLAPI * PFNGLPOINTPARAMETERIPROC) (GLenum pname, GLint param);
typedef void (GLAPI * PFNGLPOINTPARAMETERIVPROC) (GLenum pname, const GLint *params);
typedef void (GLAPI * PFNGLWINDOWPOS2DPROC) (GLdouble x, GLdouble y);
typedef void (GLAPI * PFNGLWINDOWPOS2FPROC) (GLfloat x, GLfloat y);
typedef void (GLAPI * PFNGLWINDOWPOS2IPROC) (GLint x, GLint y);
typedef void (GLAPI * PFNGLWINDOWPOS2SPROC) (GLshort x, GLshort y);
typedef void (GLAPI * PFNGLWINDOWPOS2DVPROC) (const GLdouble *p);
typedef void (GLAPI * PFNGLWINDOWPOS2FVPROC) (const GLfloat *p);
typedef void (GLAPI * PFNGLWINDOWPOS2IVPROC) (const GLint *p);
typedef void (GLAPI * PFNGLWINDOWPOS2SVPROC) (const GLshort *p);
typedef void (GLAPI * PFNGLWINDOWPOS3DPROC) (GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPI * PFNGLWINDOWPOS3FPROC) (GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPI * PFNGLWINDOWPOS3IPROC) (GLint x, GLint y, GLint z);
typedef void (GLAPI * PFNGLWINDOWPOS3SPROC) (GLshort x, GLshort y, GLshort z);
typedef void (GLAPI * PFNGLWINDOWPOS3DVPROC) (const GLdouble *p);
typedef void (GLAPI * PFNGLWINDOWPOS3FVPROC) (const GLfloat *p);
typedef void (GLAPI * PFNGLWINDOWPOS3IVPROC) (const GLint *p);
typedef void (GLAPI * PFNGLWINDOWPOS3SVPROC) (const GLshort *p);
typedef void (GLAPI * PFNGLBLENDFUNCSEPARATEPROC) (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
typedef void (GLAPI * PFNGLFOGCOORDDPROC) (GLdouble fog);
typedef void (GLAPI * PFNGLFOGCOORDDVPROC) (const GLdouble *fog);
typedef void (GLAPI * PFNGLFOGCOORDFPROC) (GLfloat fog);
typedef void (GLAPI * PFNGLFOGCOORDFVPROC) (const GLfloat *fog);
typedef void (GLAPI * PFNGLFOGCOORDPOINTERPROC) (GLenum type, GLsizei stride, const GLvoid *pointer);
typedef void (GLAPI * PFNGLMULTIDRAWARRAYSPROC) (GLenum mode, const GLint *first, const GLsizei *count, GLsizei primcount);
typedef void (GLAPI * PFNGLMULTIDRAWELEMENTSPROC) (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* *indices, GLsizei primcount);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3BPROC) (GLbyte red, GLbyte green, GLbyte blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3BVPROC) (const GLbyte *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3DPROC) (GLdouble red, GLdouble green, GLdouble blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3DVPROC) (const GLdouble *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3FPROC) (GLfloat red, GLfloat green, GLfloat blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3FVPROC) (const GLfloat *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3IPROC) (GLint red, GLint green, GLint blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3IVPROC) (const GLint *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3SPROC) (GLshort red, GLshort green, GLshort blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3SVPROC) (const GLshort *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UBPROC) (GLubyte red, GLubyte green, GLubyte blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UBVPROC) (const GLubyte *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UIPROC) (GLuint red, GLuint green, GLuint blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UIVPROC) (const GLuint *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3USPROC) (GLushort red, GLushort green, GLushort blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3USVPROC) (const GLushort *v);
typedef void (GLAPI * PFNGLSECONDARYCOLORPOINTERPROC) (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
#endif


#ifndef GL_EXT_vertex_array
#define GL_EXT_vertex_array 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glArrayElementEXT (GLint i);
void GLAPI glColorPointerEXT (GLint size, GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
void GLAPI glEdgeFlagPointerEXT (GLsizei stride, GLsizei count, const GLboolean *pointer);
void GLAPI glGetPointervEXT (GLenum pname, GLvoid* *params);
void GLAPI glIndexPointerEXT (GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
void GLAPI glNormalPointerEXT (GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
void GLAPI glTexCoordPointerEXT (GLint size, GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
void GLAPI glVertexPointerEXT (GLint size, GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
void GLAPI glDrawArraysEXT (GLenum mode, GLint first, GLsizei count);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLARRAYELEMENTEXTPROC) (GLint i);
typedef void (GLAPI * PFNGLCOLORPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
typedef void (GLAPI * PFNGLEDGEFLAGPOINTEREXTPROC) (GLsizei stride, GLsizei count, const GLboolean *pointer);
typedef void (GLAPI * PFNGLGETPOINTERVEXTPROC) (GLenum pname, GLvoid* *params);
typedef void (GLAPI * PFNGLINDEXPOINTEREXTPROC) (GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
typedef void (GLAPI * PFNGLNORMALPOINTEREXTPROC) (GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
typedef void (GLAPI * PFNGLTEXCOORDPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
typedef void (GLAPI * PFNGLVERTEXPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLsizei count, const GLvoid *pointer);
typedef void (GLAPI * PFNGLDRAWARRAYSEXTPROC) (GLenum mode, GLint first, GLsizei count);
#endif


#ifndef GL_EXT_texture_object
#define GL_EXT_texture_object 1
#ifdef GL_GLEXT_PROTOTYPES
GLboolean GLAPI glAreTexturesResidentEXT (GLsizei n, const GLuint *textures, GLboolean *residences);
void GLAPI glBindTextureEXT (GLenum target, GLuint texture);
void GLAPI glDeleteTexturesEXT (GLsizei n, const GLuint *textures);
void GLAPI glGenTexturesEXT (GLsizei n, GLuint *textures);
GLboolean GLAPI glIsTextureEXT (GLuint texture);
void GLAPI glPrioritizeTexturesEXT (GLsizei n, const GLuint *textures, const GLclampf *priorities);
#endif /* GL_GLEXT_PROTOTYPES */
typedef GLboolean (GLAPI * PFNGLARETEXTURESRESIDENTEXTPROC) (GLsizei n, const GLuint *textures, GLboolean *residences);
typedef void (GLAPI * PFNGLBINDTEXTUREEXTPROC) (GLenum target, GLuint texture);
typedef void (GLAPI * PFNGLDELETETEXTURESEXTPROC) (GLsizei n, const GLuint *textures);
typedef void (GLAPI * PFNGLGENTEXTURESEXTPROC) (GLsizei n, GLuint *textures);
typedef GLboolean (GLAPI * PFNGLISTEXTUREEXTPROC) (GLuint texture);
typedef void (GLAPI * PFNGLPRIORITIZETEXTURESEXTPROC) (GLsizei n, const GLuint *textures, const GLclampf *priorities);
#endif


#ifndef GL_EXT_compiled_vertex_array
#define GL_EXT_compiled_vertex_array 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glLockArraysEXT (GLint first, GLsizei count);
void GLAPI glUnlockArraysEXT (void);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLLOCKARRAYSEXTPROC) (GLint first, GLsizei count);
typedef void (GLAPI * PFNGLUNLOCKARRAYSEXTPROC) (void);
#endif


#ifndef GL_ARB_multitexture
#define GL_ARB_multitexture 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glActiveTextureARB (GLenum texture);
void GLAPI glClientActiveTextureARB (GLenum texture);
void GLAPI glMultiTexCoord1dARB (GLenum target, GLdouble s);
void GLAPI glMultiTexCoord1dvARB (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord1fARB (GLenum target, GLfloat s);
void GLAPI glMultiTexCoord1fvARB (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord1iARB (GLenum target, GLint s);
void GLAPI glMultiTexCoord1ivARB (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord1sARB (GLenum target, GLshort s);
void GLAPI glMultiTexCoord1svARB (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord2dARB (GLenum target, GLdouble s, GLdouble t);
void GLAPI glMultiTexCoord2dvARB (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord2fARB (GLenum target, GLfloat s, GLfloat t);
void GLAPI glMultiTexCoord2fvARB (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord2iARB (GLenum target, GLint s, GLint t);
void GLAPI glMultiTexCoord2ivARB (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord2sARB (GLenum target, GLshort s, GLshort t);
void GLAPI glMultiTexCoord2svARB (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord3dARB (GLenum target, GLdouble s, GLdouble t, GLdouble r);
void GLAPI glMultiTexCoord3dvARB (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord3fARB (GLenum target, GLfloat s, GLfloat t, GLfloat r);
void GLAPI glMultiTexCoord3fvARB (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord3iARB (GLenum target, GLint s, GLint t, GLint r);
void GLAPI glMultiTexCoord3ivARB (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord3sARB (GLenum target, GLshort s, GLshort t, GLshort r);
void GLAPI glMultiTexCoord3svARB (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord4dARB (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
void GLAPI glMultiTexCoord4dvARB (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord4fARB (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
void GLAPI glMultiTexCoord4fvARB (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord4iARB (GLenum target, GLint s, GLint t, GLint r, GLint q);
void GLAPI glMultiTexCoord4ivARB (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord4sARB (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
void GLAPI glMultiTexCoord4svARB (GLenum target, const GLshort *v);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLACTIVETEXTUREARBPROC) (GLenum texture);
typedef void (GLAPI * PFNGLCLIENTACTIVETEXTUREARBPROC) (GLenum texture);
typedef void (GLAPI * PFNGLMULTITEXCOORD1DARBPROC) (GLenum target, GLdouble s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1FARBPROC) (GLenum target, GLfloat s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1IARBPROC) (GLenum target, GLint s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1SARBPROC) (GLenum target, GLshort s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1SVARBPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2DARBPROC) (GLenum target, GLdouble s, GLdouble t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2FARBPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2IARBPROC) (GLenum target, GLint s, GLint t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2SARBPROC) (GLenum target, GLshort s, GLshort t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2SVARBPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3DARBPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3FARBPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3IARBPROC) (GLenum target, GLint s, GLint t, GLint r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3SARBPROC) (GLenum target, GLshort s, GLshort t, GLshort r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3SVARBPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4DARBPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4FARBPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4IARBPROC) (GLenum target, GLint s, GLint t, GLint r, GLint q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4SARBPROC) (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4SVARBPROC) (GLenum target, const GLshort *v);
#endif


#ifndef GL_ARB_window_pos
#define GL_ARB_window_pos 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glWindowPos2dARB (GLdouble x, GLdouble y);
void GLAPI glWindowPos2fARB (GLfloat x, GLfloat y);
void GLAPI glWindowPos2iARB (GLint x, GLint y);
void GLAPI glWindowPos2sARB (GLshort x, GLshort y);
void GLAPI glWindowPos2dvARB (const GLdouble *p);
void GLAPI glWindowPos2fvARB (const GLfloat *p);
void GLAPI glWindowPos2ivARB (const GLint *p);
void GLAPI glWindowPos2svARB (const GLshort *p);
void GLAPI glWindowPos3dARB (GLdouble x, GLdouble y, GLdouble z);
void GLAPI glWindowPos3fARB (GLfloat x, GLfloat y, GLfloat z);
void GLAPI glWindowPos3iARB (GLint x, GLint y, GLint z);
void GLAPI glWindowPos3sARB (GLshort x, GLshort y, GLshort z);
void GLAPI glWindowPos3dvARB (const GLdouble *p);
void GLAPI glWindowPos3fvARB (const GLfloat *p);
void GLAPI glWindowPos3ivARB (const GLint *p);
void GLAPI glWindowPos3svARB (const GLshort *p);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLWINDOWPOS2DARBPROC) (GLdouble x, GLdouble y);
typedef void (GLAPI * PFNGLWINDOWPOS2FARBPROC) (GLfloat x, GLfloat y);
typedef void (GLAPI * PFNGLWINDOWPOS2IARBPROC) (GLint x, GLint y);
typedef void (GLAPI * PFNGLWINDOWPOS2SARBPROC) (GLshort x, GLshort y);
typedef void (GLAPI * PFNGLWINDOWPOS2DVARBPROC) (const GLdouble *p);
typedef void (GLAPI * PFNGLWINDOWPOS2FVARBPROC) (const GLfloat *p);
typedef void (GLAPI * PFNGLWINDOWPOS2IVARBPROC) (const GLint *p);
typedef void (GLAPI * PFNGLWINDOWPOS2SVARBPROC) (const GLshort *p);
typedef void (GLAPI * PFNGLWINDOWPOS3DARBPROC) (GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPI * PFNGLWINDOWPOS3FARBPROC) (GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPI * PFNGLWINDOWPOS3IARBPROC) (GLint x, GLint y, GLint z);
typedef void (GLAPI * PFNGLWINDOWPOS3SARBPROC) (GLshort x, GLshort y, GLshort z);
typedef void (GLAPI * PFNGLWINDOWPOS3DVARBPROC) (const GLdouble *p);
typedef void (GLAPI * PFNGLWINDOWPOS3FVARBPROC) (const GLfloat *p);
typedef void (GLAPI * PFNGLWINDOWPOS3IVARBPROC) (const GLint *p);
typedef void (GLAPI * PFNGLWINDOWPOS3SVARBPROC) (const GLshort *p);
#endif


#ifndef GL_EXT_texture3D
#define GL_EXT_texture3D 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glTexImage3DEXT (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
void GLAPI glTexSubImage3DEXT (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);
void GLAPI glCopyTexSubImage3DEXT (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLTEXIMAGE3DEXTPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
typedef void (GLAPI * PFNGLTEXSUBIMAGE3DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);
typedef void (GLAPI * PFNGLCOPYTEXSUBIMAGE3DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
#endif


#ifndef GL_EXT_blend_color
#define GL_EXT_blend_color 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBlendColorEXT (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBLENDCOLOREXTPROC) (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
#endif


#ifndef GL_EXT_blend_minmax
#define GL_EXT_blend_minmax 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBlendEquationEXT (GLenum mode);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBLENDEQUATIONEXTPROC) (GLenum mode);
#endif


#ifndef GL_EXT_point_parameters
#define GL_EXT_point_parameters 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glPointParameterfEXT (GLenum pname, GLfloat param);
void GLAPI glPointParameterfvEXT (GLenum pname, const GLfloat *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLPOINTPARAMETERFEXTPROC) (GLenum pname, GLfloat param);
typedef void (GLAPI * PFNGLPOINTPARAMETERFVEXTPROC) (GLenum pname, const GLfloat *params);
#endif


#ifndef GL_EXT_paletted_texture
#define GL_EXT_paletted_texture 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glColorSubTableEXT (GLenum target, GLsizei start, GLsizei count, GLenum format, GLenum type, const GLvoid *table);
void GLAPI glColorTableEXT (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *table);
void GLAPI glGetColorTableEXT (GLenum target, GLenum format, GLenum type, GLvoid *table);
void GLAPI glGetColorTableParameterfvEXT (GLenum target, GLenum pname, GLfloat *params);
void GLAPI glGetColorTableParameterivEXT (GLenum target, GLenum pname, GLint *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLCOLORSUBTABLEEXTPROC) (GLenum target, GLsizei start, GLsizei count, GLenum format, GLenum type, const GLvoid *table);
typedef void (GLAPI * PFNGLCOLORTABLEEXTPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *table);
typedef void (GLAPI * PFNGLGETCOLORTABLEEXTPROC) (GLenum target, GLenum format, GLenum type, GLvoid *table);
typedef void (GLAPI * PFNGLGETCOLORTABLEPARAMETERFVEXTPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETCOLORTABLEPARAMETERIVEXTPROC) (GLenum target, GLenum pname, GLint *params);
#endif


#ifndef GL_WIN_swap_hint
#define GL_WIN_swap_hint 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glAddSwapHintRectWIN (GLint x, GLint y, GLsizei width, GLsizei height);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLADDSWAPHINTRECTWINPROC) (GLint x, GLint y, GLsizei width, GLsizei height);
#endif


#ifndef GL_SGIS_multitexture
#define GL_SGIS_multitexture 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glMultiTexCoord1dSGIS (GLenum target, GLdouble s);
void GLAPI glMultiTexCoord1dvSGIS (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord1fSGIS (GLenum target, GLfloat s);
void GLAPI glMultiTexCoord1fvSGIS (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord1iSGIS (GLenum target, GLint s);
void GLAPI glMultiTexCoord1ivSGIS (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord1sSGIS (GLenum target, GLshort s);
void GLAPI glMultiTexCoord1svSGIS (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord2dSGIS (GLenum target, GLdouble s, GLdouble t);
void GLAPI glMultiTexCoord2dvSGIS (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord2fSGIS (GLenum target, GLfloat s, GLfloat t);
void GLAPI glMultiTexCoord2fvSGIS (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord2iSGIS (GLenum target, GLint s, GLint t);
void GLAPI glMultiTexCoord2ivSGIS (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord2sSGIS (GLenum target, GLshort s, GLshort t);
void GLAPI glMultiTexCoord2svSGIS (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord3dSGIS (GLenum target, GLdouble s, GLdouble t, GLdouble r);
void GLAPI glMultiTexCoord3dvSGIS (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord3fSGIS (GLenum target, GLfloat s, GLfloat t, GLfloat r);
void GLAPI glMultiTexCoord3fvSGIS (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord3iSGIS (GLenum target, GLint s, GLint t, GLint r);
void GLAPI glMultiTexCoord3ivSGIS (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord3sSGIS (GLenum target, GLshort s, GLshort t, GLshort r);
void GLAPI glMultiTexCoord3svSGIS (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoord4dSGIS (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
void GLAPI glMultiTexCoord4dvSGIS (GLenum target, const GLdouble *v);
void GLAPI glMultiTexCoord4fSGIS (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
void GLAPI glMultiTexCoord4fvSGIS (GLenum target, const GLfloat *v);
void GLAPI glMultiTexCoord4iSGIS (GLenum target, GLint s, GLint t, GLint r, GLint q);
void GLAPI glMultiTexCoord4ivSGIS (GLenum target, const GLint *v);
void GLAPI glMultiTexCoord4sSGIS (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
void GLAPI glMultiTexCoord4svSGIS (GLenum target, const GLshort *v);
void GLAPI glMultiTexCoordPointerSGIS (GLenum target, GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
void GLAPI glSelectTextureSGIS (GLenum target);
void GLAPI glSelectTextureCoordSetSGIS (GLenum target);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLMULTITEXCOORD1DSGISPROC) (GLenum target, GLdouble s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1DVSGISPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1FSGISPROC) (GLenum target, GLfloat s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1FVSGISPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1ISGISPROC) (GLenum target, GLint s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1IVSGISPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1SSGISPROC) (GLenum target, GLshort s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1SVSGISPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2DSGISPROC) (GLenum target, GLdouble s, GLdouble t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2DVSGISPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2FSGISPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2FVSGISPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2ISGISPROC) (GLenum target, GLint s, GLint t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2IVSGISPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2SSGISPROC) (GLenum target, GLshort s, GLshort t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2SVSGISPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3DSGISPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3DVSGISPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3FSGISPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3FVSGISPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3ISGISPROC) (GLenum target, GLint s, GLint t, GLint r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3IVSGISPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3SSGISPROC) (GLenum target, GLshort s, GLshort t, GLshort r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3SVSGISPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4DSGISPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4DVSGISPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4FSGISPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4FVSGISPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4ISGISPROC) (GLenum target, GLint s, GLint t, GLint r, GLint q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4IVSGISPROC) (GLenum target, const GLint *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4SSGISPROC) (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4SVSGISPROC) (GLenum target, const GLshort *v);
typedef void (GLAPI * PFNGLMULTITEXCOORDPOINTERSGISPROC) (GLenum target, GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
typedef void (GLAPI * PFNGLSELECTTEXTURESGISPROC) (GLenum target);
typedef void (GLAPI * PFNGLSELECTTEXTURECOORDSETSGISPROC) (GLenum target);
#endif


#ifndef GL_EXT_fog_coord
#define GL_EXT_fog_coord 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glFogCoorddEXT (GLdouble fog);
void GLAPI glFogCoorddvEXT (const GLdouble *fog);
void GLAPI glFogCoordfEXT (GLfloat fog);
void GLAPI glFogCoordfvEXT (const GLfloat *fog);
void GLAPI glFogCoordPointerEXT (GLenum type, GLsizei stride, const GLvoid *pointer);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLFOGCOORDDEXTPROC) (GLdouble fog);
typedef void (GLAPI * PFNGLFOGCOORDDVEXTPROC) (const GLdouble *fog);
typedef void (GLAPI * PFNGLFOGCOORDFEXTPROC) (GLfloat fog);
typedef void (GLAPI * PFNGLFOGCOORDFVEXTPROC) (const GLfloat *fog);
typedef void (GLAPI * PFNGLFOGCOORDPOINTEREXTPROC) (GLenum type, GLsizei stride, const GLvoid *pointer);
#endif


#ifndef GL_EXT_secondary_color
#define GL_EXT_secondary_color 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glSecondaryColor3bEXT (GLbyte red, GLbyte green, GLbyte blue);
void GLAPI glSecondaryColor3bvEXT (const GLbyte *v);
void GLAPI glSecondaryColor3dEXT (GLdouble red, GLdouble green, GLdouble blue);
void GLAPI glSecondaryColor3dvEXT (const GLdouble *v);
void GLAPI glSecondaryColor3fEXT (GLfloat red, GLfloat green, GLfloat blue);
void GLAPI glSecondaryColor3fvEXT (const GLfloat *v);
void GLAPI glSecondaryColor3iEXT (GLint red, GLint green, GLint blue);
void GLAPI glSecondaryColor3ivEXT (const GLint *v);
void GLAPI glSecondaryColor3sEXT (GLshort red, GLshort green, GLshort blue);
void GLAPI glSecondaryColor3svEXT (const GLshort *v);
void GLAPI glSecondaryColor3ubEXT (GLubyte red, GLubyte green, GLubyte blue);
void GLAPI glSecondaryColor3ubvEXT (const GLubyte *v);
void GLAPI glSecondaryColor3uiEXT (GLuint red, GLuint green, GLuint blue);
void GLAPI glSecondaryColor3uivEXT (const GLuint *v);
void GLAPI glSecondaryColor3usEXT (GLushort red, GLushort green, GLushort blue);
void GLAPI glSecondaryColor3usvEXT (const GLushort *v);
void GLAPI glSecondaryColorPointerEXT (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLSECONDARYCOLOR3BEXTPROC) (GLbyte red, GLbyte green, GLbyte blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3BVEXTPROC) (const GLbyte *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3DEXTPROC) (GLdouble red, GLdouble green, GLdouble blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3DVEXTPROC) (const GLdouble *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3FEXTPROC) (GLfloat red, GLfloat green, GLfloat blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3FVEXTPROC) (const GLfloat *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3IEXTPROC) (GLint red, GLint green, GLint blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3IVEXTPROC) (const GLint *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3SEXTPROC) (GLshort red, GLshort green, GLshort blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3SVEXTPROC) (const GLshort *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UBEXTPROC) (GLubyte red, GLubyte green, GLubyte blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UBVEXTPROC) (const GLubyte *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UIEXTPROC) (GLuint red, GLuint green, GLuint blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3UIVEXTPROC) (const GLuint *v);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3USEXTPROC) (GLushort red, GLushort green, GLushort blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3USVEXTPROC) (const GLushort *v);
typedef void (GLAPI * PFNGLSECONDARYCOLORPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
#endif


#ifndef GL_NV_vertex_array_range
#define GL_NV_vertex_array_range 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glFlushVertexArrayRangeNV (void);
void GLAPI glVertexArrayRangeNV (GLsizei size, const GLvoid *pointer);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLFLUSHVERTEXARRAYRANGENVPROC) (void);
typedef void (GLAPI * PFNGLVERTEXARRAYRANGENVPROC) (GLsizei size, const GLvoid *pointer);
#endif


#ifndef GL_NV_register_combiners
#define GL_NV_register_combiners 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glCombinerParameterfvNV (GLenum pname, const GLfloat *params);
void GLAPI glCombinerParameterfNV (GLenum pname, GLfloat param);
void GLAPI glCombinerParameterivNV (GLenum pname, const GLint *params);
void GLAPI glCombinerParameteriNV (GLenum pname, GLint param);
void GLAPI glCombinerInputNV (GLenum stage, GLenum portion, GLenum variable, GLenum input, GLenum mapping, GLenum componentUsage);
void GLAPI glCombinerOutputNV (GLenum stage, GLenum portion, GLenum abOutput, GLenum cdOutput, GLenum sumOutput, GLenum scale, GLenum bias, GLboolean abDotProduct, GLboolean cdDotProduct, GLboolean muxSum);
void GLAPI glFinalCombinerInputNV (GLenum variable, GLenum input, GLenum mapping, GLenum componentUsage);
void GLAPI glGetCombinerInputParameterfvNV (GLenum stage, GLenum portion, GLenum variable, GLenum pname, GLfloat *params);
void GLAPI glGetCombinerInputParameterivNV (GLenum stage, GLenum portion, GLenum variable, GLenum pname, GLint *params);
void GLAPI glGetCombinerOutputParameterfvNV (GLenum stage, GLenum portion, GLenum pname, GLfloat *params);
void GLAPI glGetCombinerOutputParameterivNV (GLenum stage, GLenum portion, GLenum pname, GLint *params);
void GLAPI glGetFinalCombinerInputParameterfvNV (GLenum variable, GLenum pname, GLfloat *params);
void GLAPI glGetFinalCombinerInputParameterivNV (GLenum variable, GLenum pname, GLint *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLCOMBINERPARAMETERFVNVPROC) (GLenum pname, const GLfloat *params);
typedef void (GLAPI * PFNGLCOMBINERPARAMETERFNVPROC) (GLenum pname, GLfloat param);
typedef void (GLAPI * PFNGLCOMBINERPARAMETERIVNVPROC) (GLenum pname, const GLint *params);
typedef void (GLAPI * PFNGLCOMBINERPARAMETERINVPROC) (GLenum pname, GLint param);
typedef void (GLAPI * PFNGLCOMBINERINPUTNVPROC) (GLenum stage, GLenum portion, GLenum variable, GLenum input, GLenum mapping, GLenum componentUsage);
typedef void (GLAPI * PFNGLCOMBINEROUTPUTNVPROC) (GLenum stage, GLenum portion, GLenum abOutput, GLenum cdOutput, GLenum sumOutput, GLenum scale, GLenum bias, GLboolean abDotProduct, GLboolean cdDotProduct, GLboolean muxSum);
typedef void (GLAPI * PFNGLFINALCOMBINERINPUTNVPROC) (GLenum variable, GLenum input, GLenum mapping, GLenum componentUsage);
typedef void (GLAPI * PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC) (GLenum stage, GLenum portion, GLenum variable, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC) (GLenum stage, GLenum portion, GLenum variable, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC) (GLenum stage, GLenum portion, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC) (GLenum stage, GLenum portion, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC) (GLenum variable, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC) (GLenum variable, GLenum pname, GLint *params);
#endif


#ifndef GL_ARB_transpose_matrix
#define GL_ARB_transpose_matrix 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glLoadTransposeMatrixfARB (const GLfloat *m);
void GLAPI glLoadTransposeMatrixdARB (const GLdouble *m);
void GLAPI glMultTransposeMatrixfARB (const GLfloat *m);
void GLAPI glMultTransposeMatrixdARB (const GLdouble *m);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLLOADTRANSPOSEMATRIXFARBPROC) (const GLfloat *m);
typedef void (GLAPI * PFNGLLOADTRANSPOSEMATRIXDARBPROC) (const GLdouble *m);
typedef void (GLAPI * PFNGLMULTTRANSPOSEMATRIXFARBPROC) (const GLfloat *m);
typedef void (GLAPI * PFNGLMULTTRANSPOSEMATRIXDARBPROC) (const GLdouble *m);
#endif


#ifndef GL_ARB_texture_compression
#define GL_ARB_texture_compression 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glCompressedTexImage3DARB (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexImage2DARB (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexImage1DARB (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexSubImage3DARB (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexSubImage2DARB (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data);
void GLAPI glCompressedTexSubImage1DARB (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data);
void GLAPI glGetCompressedTexImageARB (GLenum target, GLint lod, GLvoid *img);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLCOMPRESSEDTEXIMAGE3DARBPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXIMAGE2DARBPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXIMAGE1DARBPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC) (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPI * PFNGLGETCOMPRESSEDTEXIMAGEARBPROC) (GLenum target, GLint lod, GLvoid *img);
#endif


#ifndef GL_NV_vertex_program
#define GL_NV_vertex_program 1
#ifdef GL_GLEXT_PROTOTYPES
GLboolean GLAPI glAreProgramsResidentNV (GLsizei n, const GLuint *programs, GLboolean *residences);
void GLAPI glBindProgramNV (GLenum target, GLuint id);
void GLAPI glDeleteProgramsNV (GLsizei n, const GLuint *programs);
void GLAPI glExecuteProgramNV (GLenum target, GLuint id, const GLfloat *params);
void GLAPI glGenProgramsNV (GLsizei n, GLuint *programs);
void GLAPI glGetProgramParameterdvNV (GLenum target, GLuint index, GLenum pname, GLdouble *params);
void GLAPI glGetProgramParameterfvNV (GLenum target, GLuint index, GLenum pname, GLfloat *params);
void GLAPI glGetProgramivNV (GLuint id, GLenum pname, GLint *params);
void GLAPI glGetProgramStringNV (GLuint id, GLenum pname, GLubyte *program);
void GLAPI glGetTrackMatrixivNV (GLenum target, GLuint address, GLenum pname, GLint *params);
void GLAPI glGetVertexAttribdvNV (GLuint index, GLenum pname, GLdouble *params);
void GLAPI glGetVertexAttribfvNV (GLuint index, GLenum pname, GLfloat *params);
void GLAPI glGetVertexAttribivNV (GLuint index, GLenum pname, GLint *params);
void GLAPI glGetVertexAttribPointervNV (GLuint index, GLenum pname, GLvoid* *pointer);
GLboolean GLAPI glIsProgramNV (GLuint id);
void GLAPI glLoadProgramNV (GLenum target, GLuint id, GLsizei len, const GLubyte *program);
void GLAPI glProgramParameter4dNV (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GLAPI glProgramParameter4dvNV (GLenum target, GLuint index, const GLdouble *v);
void GLAPI glProgramParameter4fNV (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GLAPI glProgramParameter4fvNV (GLenum target, GLuint index, const GLfloat *v);
void GLAPI glProgramParameters4dvNV (GLenum target, GLuint index, GLsizei count, const GLdouble *v);
void GLAPI glProgramParameters4fvNV (GLenum target, GLuint index, GLsizei count, const GLfloat *v);
void GLAPI glRequestResidentProgramsNV (GLsizei n, const GLuint *programs);
void GLAPI glTrackMatrixNV (GLenum target, GLuint address, GLenum matrix, GLenum transform);
void GLAPI glVertexAttribPointerNV (GLuint index, GLint fsize, GLenum type, GLsizei stride, const GLvoid *pointer);
void GLAPI glVertexAttrib1dNV (GLuint index, GLdouble x);
void GLAPI glVertexAttrib1dvNV (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib1fNV (GLuint index, GLfloat x);
void GLAPI glVertexAttrib1fvNV (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib1sNV (GLuint index, GLshort x);
void GLAPI glVertexAttrib1svNV (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib2dNV (GLuint index, GLdouble x, GLdouble y);
void GLAPI glVertexAttrib2dvNV (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib2fNV (GLuint index, GLfloat x, GLfloat y);
void GLAPI glVertexAttrib2fvNV (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib2sNV (GLuint index, GLshort x, GLshort y);
void GLAPI glVertexAttrib2svNV (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib3dNV (GLuint index, GLdouble x, GLdouble y, GLdouble z);
void GLAPI glVertexAttrib3dvNV (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib3fNV (GLuint index, GLfloat x, GLfloat y, GLfloat z);
void GLAPI glVertexAttrib3fvNV (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib3sNV (GLuint index, GLshort x, GLshort y, GLshort z);
void GLAPI glVertexAttrib3svNV (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib4dNV (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GLAPI glVertexAttrib4dvNV (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib4fNV (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GLAPI glVertexAttrib4fvNV (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib4sNV (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
void GLAPI glVertexAttrib4svNV (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib4ubNV (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
void GLAPI glVertexAttrib4ubvNV (GLuint index, const GLubyte *v);
void GLAPI glVertexAttribs1dvNV (GLuint index, GLsizei count, const GLdouble *v);
void GLAPI glVertexAttribs1fvNV (GLuint index, GLsizei count, const GLfloat *v);
void GLAPI glVertexAttribs1svNV (GLuint index, GLsizei count, const GLshort *v);
void GLAPI glVertexAttribs2dvNV (GLuint index, GLsizei count, const GLdouble *v);
void GLAPI glVertexAttribs2fvNV (GLuint index, GLsizei count, const GLfloat *v);
void GLAPI glVertexAttribs2svNV (GLuint index, GLsizei count, const GLshort *v);
void GLAPI glVertexAttribs3dvNV (GLuint index, GLsizei count, const GLdouble *v);
void GLAPI glVertexAttribs3fvNV (GLuint index, GLsizei count, const GLfloat *v);
void GLAPI glVertexAttribs3svNV (GLuint index, GLsizei count, const GLshort *v);
void GLAPI glVertexAttribs4dvNV (GLuint index, GLsizei count, const GLdouble *v);
void GLAPI glVertexAttribs4fvNV (GLuint index, GLsizei count, const GLfloat *v);
void GLAPI glVertexAttribs4svNV (GLuint index, GLsizei count, const GLshort *v);
void GLAPI glVertexAttribs4ubvNV (GLuint index, GLsizei count, const GLubyte *v);
#endif /* GL_GLEXT_PROTOTYPES */
typedef GLboolean (GLAPI * PFNGLAREPROGRAMSRESIDENTNVPROC) (GLsizei n, const GLuint *programs, GLboolean *residences);
typedef void (GLAPI * PFNGLBINDPROGRAMNVPROC) (GLenum target, GLuint id);
typedef void (GLAPI * PFNGLDELETEPROGRAMSNVPROC) (GLsizei n, const GLuint *programs);
typedef void (GLAPI * PFNGLEXECUTEPROGRAMNVPROC) (GLenum target, GLuint id, const GLfloat *params);
typedef void (GLAPI * PFNGLGENPROGRAMSNVPROC) (GLsizei n, GLuint *programs);
typedef void (GLAPI * PFNGLGETPROGRAMPARAMETERDVNVPROC) (GLenum target, GLuint index, GLenum pname, GLdouble *params);
typedef void (GLAPI * PFNGLGETPROGRAMPARAMETERFVNVPROC) (GLenum target, GLuint index, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETPROGRAMIVNVPROC) (GLuint id, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETPROGRAMSTRINGNVPROC) (GLuint id, GLenum pname, GLubyte *program);
typedef void (GLAPI * PFNGLGETTRACKMATRIXIVNVPROC) (GLenum target, GLuint address, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBDVNVPROC) (GLuint index, GLenum pname, GLdouble *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBFVNVPROC) (GLuint index, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBIVNVPROC) (GLuint index, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBPOINTERVNVPROC) (GLuint index, GLenum pname, GLvoid* *pointer);
typedef GLboolean (GLAPI * PFNGLISPROGRAMNVPROC) (GLuint id);
typedef void (GLAPI * PFNGLLOADPROGRAMNVPROC) (GLenum target, GLuint id, GLsizei len, const GLubyte *program);
typedef void (GLAPI * PFNGLPROGRAMPARAMETER4DNVPROC) (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPI * PFNGLPROGRAMPARAMETER4DVNVPROC) (GLenum target, GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLPROGRAMPARAMETER4FNVPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPI * PFNGLPROGRAMPARAMETER4FVNVPROC) (GLenum target, GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLPROGRAMPARAMETERS4DVNVPROC) (GLenum target, GLuint index, GLsizei count, const GLdouble *v);
typedef void (GLAPI * PFNGLPROGRAMPARAMETERS4FVNVPROC) (GLenum target, GLuint index, GLsizei count, const GLfloat *v);
typedef void (GLAPI * PFNGLREQUESTRESIDENTPROGRAMSNVPROC) (GLsizei n, const GLuint *programs);
typedef void (GLAPI * PFNGLTRACKMATRIXNVPROC) (GLenum target, GLuint address, GLenum matrix, GLenum transform);
typedef void (GLAPI * PFNGLVERTEXATTRIBPOINTERNVPROC) (GLuint index, GLint fsize, GLenum type, GLsizei stride, const GLvoid *pointer);
typedef void (GLAPI * PFNGLVERTEXATTRIB1DNVPROC) (GLuint index, GLdouble x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1DVNVPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB1FNVPROC) (GLuint index, GLfloat x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1FVNVPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB1SNVPROC) (GLuint index, GLshort x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1SVNVPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2DNVPROC) (GLuint index, GLdouble x, GLdouble y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2DVNVPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2FNVPROC) (GLuint index, GLfloat x, GLfloat y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2FVNVPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2SNVPROC) (GLuint index, GLshort x, GLshort y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2SVNVPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3DNVPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3DVNVPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3FNVPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3FVNVPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3SNVPROC) (GLuint index, GLshort x, GLshort y, GLshort z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3SVNVPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4DNVPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4DVNVPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4FNVPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4FVNVPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4SNVPROC) (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4SVNVPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4UBNVPROC) (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4UBVNVPROC) (GLuint index, const GLubyte *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS1DVNVPROC) (GLuint index, GLsizei count, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS1FVNVPROC) (GLuint index, GLsizei count, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS1SVNVPROC) (GLuint index, GLsizei count, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS2DVNVPROC) (GLuint index, GLsizei count, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS2FVNVPROC) (GLuint index, GLsizei count, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS2SVNVPROC) (GLuint index, GLsizei count, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS3DVNVPROC) (GLuint index, GLsizei count, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS3FVNVPROC) (GLuint index, GLsizei count, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS3SVNVPROC) (GLuint index, GLsizei count, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS4DVNVPROC) (GLuint index, GLsizei count, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS4FVNVPROC) (GLuint index, GLsizei count, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS4SVNVPROC) (GLuint index, GLsizei count, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS4UBVNVPROC) (GLuint index, GLsizei count, const GLubyte *v);
#endif


#ifndef GL_NV_fence
#define GL_NV_fence 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glDeleteFencesNV (GLsizei n, const GLuint *fences);
void GLAPI glGenFencesNV (GLsizei n, GLuint *fences);
GLboolean GLAPI glIsFenceNV (GLuint fence);
GLboolean GLAPI glTestFenceNV (GLuint fence);
void GLAPI glGetFenceivNV (GLuint fence, GLenum pname, GLint *params);
void GLAPI glFinishFenceNV (GLuint fence);
void GLAPI glSetFenceNV (GLuint fence, GLenum condition);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLDELETEFENCESNVPROC) (GLsizei n, const GLuint *fences);
typedef void (GLAPI * PFNGLGENFENCESNVPROC) (GLsizei n, GLuint *fences);
typedef GLboolean (GLAPI * PFNGLISFENCENVPROC) (GLuint fence);
typedef GLboolean (GLAPI * PFNGLTESTFENCENVPROC) (GLuint fence);
typedef void (GLAPI * PFNGLGETFENCEIVNVPROC) (GLuint fence, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLFINISHFENCENVPROC) (GLuint fence);
typedef void (GLAPI * PFNGLSETFENCENVPROC) (GLuint fence, GLenum condition);
#endif


#ifndef GL_NV_draw_mesh
#define GL_NV_draw_mesh 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glDrawMeshNV (GLenum mode, GLsizei count, GLenum type, GLsizei stride, const GLvoid *indicesTexCoord, const GLvoid *indicesNormal, const GLvoid *indicesVertex);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLDRAWMESHNVPROC) (GLenum mode, GLsizei count, GLenum type, GLsizei stride, const GLvoid *indicesTexCoord, const GLvoid *indicesNormal, const GLvoid *indicesVertex);
#endif


#ifndef GL_Autodesk_valid_back_buffer_hint
#define GL_Autodesk_valid_back_buffer_hint 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glWindowBackBufferHintAutodesk (void);
GLboolean GLAPI glValidBackBufferHintAutodesk (GLint x, GLint y, GLsizei width, GLsizei height);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLWINDOWBACKBUFFERHINTAUTODESKPROC) (void);
typedef GLboolean (GLAPI * PFNGLVALIDBACKBUFFERHINTAUTODESKPROC) (GLint x, GLint y, GLsizei width, GLsizei height);
#endif


#ifndef GL_NV_set_window_stereomode
#define GL_NV_set_window_stereomode 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glSetWindowStereoModeNV (GLboolean displayMode);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLSETWINDOWSTEREOMODENVPROC) (GLboolean displayMode);
#endif


#ifndef GL_NV_register_combiners2
#define GL_NV_register_combiners2 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glCombinerStageParameterfvNV (GLenum stage, GLenum pname, const GLfloat *params);
void GLAPI glGetCombinerStageParameterfvNV (GLenum stage, GLenum pname, GLfloat *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLCOMBINERSTAGEPARAMETERFVNVPROC) (GLenum stage, GLenum pname, const GLfloat *params);
typedef void (GLAPI * PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC) (GLenum stage, GLenum pname, GLfloat *params);
#endif


#ifndef GL_ARB_multisample
#define GL_ARB_multisample 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glSampleCoverageARB (GLclampf value, GLboolean invert);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLSAMPLECOVERAGEARBPROC) (GLclampf value, GLboolean invert);
#endif


#ifndef GL_EXT_draw_range_elements
#define GL_EXT_draw_range_elements 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glDrawRangeElementsEXT (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLDRAWRANGEELEMENTSEXTPROC) (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
#endif


#ifndef GL_NV_pixel_data_range
#define GL_NV_pixel_data_range 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glFlushPixelDataRangeNV (GLenum target);
void GLAPI glPixelDataRangeNV (GLenum target, GLsizei size, const GLvoid *pointer);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLFLUSHPIXELDATARANGENVPROC) (GLenum target);
typedef void (GLAPI * PFNGLPIXELDATARANGENVPROC) (GLenum target, GLsizei size, const GLvoid *pointer);
#endif


#ifndef GL_NV_fragment_program
#define GL_NV_fragment_program 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glGetProgramNamedParameterdvNV (GLuint id, GLsizei len, const GLubyte *name, GLdouble *params);
void GLAPI glGetProgramNamedParameterfvNV (GLuint id, GLsizei len, const GLubyte *name, GLfloat *params);
void GLAPI glProgramNamedParameter4dNV (GLuint id, GLsizei len, const GLubyte *name, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GLAPI glProgramNamedParameter4dvNV (GLuint id, GLsizei len, const GLubyte *name, const GLdouble *v);
void GLAPI glProgramNamedParameter4fNV (GLuint id, GLsizei len, const GLubyte *name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GLAPI glProgramNamedParameter4fvNV (GLuint id, GLsizei len, const GLubyte *name, const GLfloat *v);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC) (GLuint id, GLsizei len, const GLubyte *name, GLdouble *params);
typedef void (GLAPI * PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC) (GLuint id, GLsizei len, const GLubyte *name, GLfloat *params);
typedef void (GLAPI * PFNGLPROGRAMNAMEDPARAMETER4DNVPROC) (GLuint id, GLsizei len, const GLubyte *name, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPI * PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC) (GLuint id, GLsizei len, const GLubyte *name, const GLdouble *v);
typedef void (GLAPI * PFNGLPROGRAMNAMEDPARAMETER4FNVPROC) (GLuint id, GLsizei len, const GLubyte *name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPI * PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC) (GLuint id, GLsizei len, const GLubyte *name, const GLfloat *v);
#endif


#ifndef GL_NV_occlusion_query
#define GL_NV_occlusion_query 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glGenOcclusionQueriesNV (GLsizei n, GLuint *ids);
void GLAPI glDeleteOcclusionQueriesNV (GLsizei n, const GLuint *ids);
GLboolean GLAPI glIsOcclusionQueryNV (GLuint id);
void GLAPI glBeginOcclusionQueryNV (GLuint id);
void GLAPI glEndOcclusionQueryNV (void);
void GLAPI glGetOcclusionQueryivNV (GLuint id, GLenum pname, GLint *params);
void GLAPI glGetOcclusionQueryuivNV (GLuint id, GLenum pname, GLuint *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLGENOCCLUSIONQUERIESNVPROC) (GLsizei n, GLuint *ids);
typedef void (GLAPI * PFNGLDELETEOCCLUSIONQUERIESNVPROC) (GLsizei n, const GLuint *ids);
typedef GLboolean (GLAPI * PFNGLISOCCLUSIONQUERYNVPROC) (GLuint id);
typedef void (GLAPI * PFNGLBEGINOCCLUSIONQUERYNVPROC) (GLuint id);
typedef void (GLAPI * PFNGLENDOCCLUSIONQUERYNVPROC) (void);
typedef void (GLAPI * PFNGLGETOCCLUSIONQUERYIVNVPROC) (GLuint id, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETOCCLUSIONQUERYUIVNVPROC) (GLuint id, GLenum pname, GLuint *params);
#endif


#ifndef GL_NV_point_sprite
#define GL_NV_point_sprite 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glPointParameteriNV (GLenum pname, GLint param);
void GLAPI glPointParameterivNV (GLenum pname, const GLint *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLPOINTPARAMETERINVPROC) (GLenum pname, GLint param);
typedef void (GLAPI * PFNGLPOINTPARAMETERIVNVPROC) (GLenum pname, const GLint *params);
#endif


#ifndef GL_EXT_multi_draw_arrays
#define GL_EXT_multi_draw_arrays 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glMultiDrawArraysEXT (GLenum mode, const GLint *first, const GLsizei *count, GLsizei primcount);
void GLAPI glMultiDrawElementsEXT (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* *indices, GLsizei primcount);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLMULTIDRAWARRAYSEXTPROC) (GLenum mode, const GLint *first, const GLsizei *count, GLsizei primcount);
typedef void (GLAPI * PFNGLMULTIDRAWELEMENTSEXTPROC) (GLenum mode, const GLsizei *count, GLenum type, const GLvoid* *indices, GLsizei primcount);
#endif


#ifndef GL_NV_half_float
#define GL_NV_half_float 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glVertex2hNV (GLhalf x, GLhalf y);
void GLAPI glVertex2hvNV (const GLhalf *v);
void GLAPI glVertex3hNV (GLhalf x, GLhalf y, GLhalf z);
void GLAPI glVertex3hvNV (const GLhalf *v);
void GLAPI glVertex4hNV (GLhalf x, GLhalf y, GLhalf z, GLhalf w);
void GLAPI glVertex4hvNV (const GLhalf *v);
void GLAPI glNormal3hNV (GLhalf nx, GLhalf ny, GLhalf nz);
void GLAPI glNormal3hvNV (const GLhalf *v);
void GLAPI glColor3hNV (GLhalf red, GLhalf green, GLhalf blue);
void GLAPI glColor3hvNV (const GLhalf *v);
void GLAPI glColor4hNV (GLhalf red, GLhalf green, GLhalf blue, GLhalf alpha);
void GLAPI glColor4hvNV (const GLhalf *v);
void GLAPI glTexCoord1hNV (GLhalf s);
void GLAPI glTexCoord1hvNV (const GLhalf *v);
void GLAPI glTexCoord2hNV (GLhalf s, GLhalf t);
void GLAPI glTexCoord2hvNV (const GLhalf *v);
void GLAPI glTexCoord3hNV (GLhalf s, GLhalf t, GLhalf r);
void GLAPI glTexCoord3hvNV (const GLhalf *v);
void GLAPI glTexCoord4hNV (GLhalf s, GLhalf t, GLhalf r, GLhalf q);
void GLAPI glTexCoord4hvNV (const GLhalf *v);
void GLAPI glMultiTexCoord1hNV (GLenum target, GLhalf s);
void GLAPI glMultiTexCoord1hvNV (GLenum target, const GLhalf *v);
void GLAPI glMultiTexCoord2hNV (GLenum target, GLhalf s, GLhalf t);
void GLAPI glMultiTexCoord2hvNV (GLenum target, const GLhalf *v);
void GLAPI glMultiTexCoord3hNV (GLenum target, GLhalf s, GLhalf t, GLhalf r);
void GLAPI glMultiTexCoord3hvNV (GLenum target, const GLhalf *v);
void GLAPI glMultiTexCoord4hNV (GLenum target, GLhalf s, GLhalf t, GLhalf r, GLhalf q);
void GLAPI glMultiTexCoord4hvNV (GLenum target, const GLhalf *v);
void GLAPI glFogCoordhNV (GLhalf fog);
void GLAPI glFogCoordhvNV (const GLhalf *fog);
void GLAPI glSecondaryColor3hNV (GLhalf red, GLhalf green, GLhalf blue);
void GLAPI glSecondaryColor3hvNV (const GLhalf *v);
void GLAPI glVertexAttrib1hNV (GLuint index, GLhalf x);
void GLAPI glVertexAttrib1hvNV (GLuint index, const GLhalf *v);
void GLAPI glVertexAttrib2hNV (GLuint index, GLhalf x, GLhalf y);
void GLAPI glVertexAttrib2hvNV (GLuint index, const GLhalf *v);
void GLAPI glVertexAttrib3hNV (GLuint index, GLhalf x, GLhalf y, GLhalf z);
void GLAPI glVertexAttrib3hvNV (GLuint index, const GLhalf *v);
void GLAPI glVertexAttrib4hNV (GLuint index, GLhalf x, GLhalf y, GLhalf z, GLhalf w);
void GLAPI glVertexAttrib4hvNV (GLuint index, const GLhalf *v);
void GLAPI glVertexAttribs1hvNV (GLuint index, GLsizei count, const GLhalf *v);
void GLAPI glVertexAttribs2hvNV (GLuint index, GLsizei count, const GLhalf *v);
void GLAPI glVertexAttribs3hvNV (GLuint index, GLsizei count, const GLhalf *v);
void GLAPI glVertexAttribs4hvNV (GLuint index, GLsizei count, const GLhalf *v);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLVERTEX2HNVPROC) (GLhalf x, GLhalf y);
typedef void (GLAPI * PFNGLVERTEX2HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEX3HNVPROC) (GLhalf x, GLhalf y, GLhalf z);
typedef void (GLAPI * PFNGLVERTEX3HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEX4HNVPROC) (GLhalf x, GLhalf y, GLhalf z, GLhalf w);
typedef void (GLAPI * PFNGLVERTEX4HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLNORMAL3HNVPROC) (GLhalf nx, GLhalf ny, GLhalf nz);
typedef void (GLAPI * PFNGLNORMAL3HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLCOLOR3HNVPROC) (GLhalf red, GLhalf green, GLhalf blue);
typedef void (GLAPI * PFNGLCOLOR3HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLCOLOR4HNVPROC) (GLhalf red, GLhalf green, GLhalf blue, GLhalf alpha);
typedef void (GLAPI * PFNGLCOLOR4HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLTEXCOORD1HNVPROC) (GLhalf s);
typedef void (GLAPI * PFNGLTEXCOORD1HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLTEXCOORD2HNVPROC) (GLhalf s, GLhalf t);
typedef void (GLAPI * PFNGLTEXCOORD2HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLTEXCOORD3HNVPROC) (GLhalf s, GLhalf t, GLhalf r);
typedef void (GLAPI * PFNGLTEXCOORD3HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLTEXCOORD4HNVPROC) (GLhalf s, GLhalf t, GLhalf r, GLhalf q);
typedef void (GLAPI * PFNGLTEXCOORD4HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD1HNVPROC) (GLenum target, GLhalf s);
typedef void (GLAPI * PFNGLMULTITEXCOORD1HVNVPROC) (GLenum target, const GLhalf *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD2HNVPROC) (GLenum target, GLhalf s, GLhalf t);
typedef void (GLAPI * PFNGLMULTITEXCOORD2HVNVPROC) (GLenum target, const GLhalf *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD3HNVPROC) (GLenum target, GLhalf s, GLhalf t, GLhalf r);
typedef void (GLAPI * PFNGLMULTITEXCOORD3HVNVPROC) (GLenum target, const GLhalf *v);
typedef void (GLAPI * PFNGLMULTITEXCOORD4HNVPROC) (GLenum target, GLhalf s, GLhalf t, GLhalf r, GLhalf q);
typedef void (GLAPI * PFNGLMULTITEXCOORD4HVNVPROC) (GLenum target, const GLhalf *v);
typedef void (GLAPI * PFNGLFOGCOORDHNVPROC) (GLhalf fog);
typedef void (GLAPI * PFNGLFOGCOORDHVNVPROC) (const GLhalf *fog);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3HNVPROC) (GLhalf red, GLhalf green, GLhalf blue);
typedef void (GLAPI * PFNGLSECONDARYCOLOR3HVNVPROC) (const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB1HNVPROC) (GLuint index, GLhalf x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1HVNVPROC) (GLuint index, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2HNVPROC) (GLuint index, GLhalf x, GLhalf y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2HVNVPROC) (GLuint index, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3HNVPROC) (GLuint index, GLhalf x, GLhalf y, GLhalf z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3HVNVPROC) (GLuint index, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4HNVPROC) (GLuint index, GLhalf x, GLhalf y, GLhalf z, GLhalf w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4HVNVPROC) (GLuint index, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS1HVNVPROC) (GLuint index, GLsizei count, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS2HVNVPROC) (GLuint index, GLsizei count, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS3HVNVPROC) (GLuint index, GLsizei count, const GLhalf *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBS4HVNVPROC) (GLuint index, GLsizei count, const GLhalf *v);
#endif


#ifndef GL_EXT_stencil_two_side
#define GL_EXT_stencil_two_side 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glActiveStencilFaceEXT (GLenum face);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLACTIVESTENCILFACEEXTPROC) (GLenum face);
#endif


#ifndef GL_EXT_blend_func_separate
#define GL_EXT_blend_func_separate 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBlendFuncSeparateEXT (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBLENDFUNCSEPARATEEXTPROC) (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
#endif


#ifndef GL_ARB_point_parameters
#define GL_ARB_point_parameters 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glPointParameterfARB (GLenum pname, GLfloat param);
void GLAPI glPointParameterfvARB (GLenum pname, const GLfloat *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLPOINTPARAMETERFARBPROC) (GLenum pname, GLfloat param);
typedef void (GLAPI * PFNGLPOINTPARAMETERFVARBPROC) (GLenum pname, const GLfloat *params);
#endif


#ifndef GL_EXT_depth_bounds_test
#define GL_EXT_depth_bounds_test 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glDepthBoundsEXT (GLclampd zmin, GLclampd zmax);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLDEPTHBOUNDSEXTPROC) (GLclampd zmin, GLclampd zmax);
#endif


#ifndef GL_ARB_vertex_program
#define GL_ARB_vertex_program 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glVertexAttrib1dARB (GLuint index, GLdouble x);
void GLAPI glVertexAttrib1dvARB (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib1fARB (GLuint index, GLfloat x);
void GLAPI glVertexAttrib1fvARB (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib1sARB (GLuint index, GLshort x);
void GLAPI glVertexAttrib1svARB (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib2dARB (GLuint index, GLdouble x, GLdouble y);
void GLAPI glVertexAttrib2dvARB (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib2fARB (GLuint index, GLfloat x, GLfloat y);
void GLAPI glVertexAttrib2fvARB (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib2sARB (GLuint index, GLshort x, GLshort y);
void GLAPI glVertexAttrib2svARB (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib3dARB (GLuint index, GLdouble x, GLdouble y, GLdouble z);
void GLAPI glVertexAttrib3dvARB (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib3fARB (GLuint index, GLfloat x, GLfloat y, GLfloat z);
void GLAPI glVertexAttrib3fvARB (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib3sARB (GLuint index, GLshort x, GLshort y, GLshort z);
void GLAPI glVertexAttrib3svARB (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib4NbvARB (GLuint index, const GLbyte *v);
void GLAPI glVertexAttrib4NivARB (GLuint index, const GLint *v);
void GLAPI glVertexAttrib4NsvARB (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib4NubARB (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
void GLAPI glVertexAttrib4NubvARB (GLuint index, const GLubyte *v);
void GLAPI glVertexAttrib4NuivARB (GLuint index, const GLuint *v);
void GLAPI glVertexAttrib4NusvARB (GLuint index, const GLushort *v);
void GLAPI glVertexAttrib4bvARB (GLuint index, const GLbyte *v);
void GLAPI glVertexAttrib4dARB (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GLAPI glVertexAttrib4dvARB (GLuint index, const GLdouble *v);
void GLAPI glVertexAttrib4fARB (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GLAPI glVertexAttrib4fvARB (GLuint index, const GLfloat *v);
void GLAPI glVertexAttrib4ivARB (GLuint index, const GLint *v);
void GLAPI glVertexAttrib4sARB (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
void GLAPI glVertexAttrib4svARB (GLuint index, const GLshort *v);
void GLAPI glVertexAttrib4ubvARB (GLuint index, const GLubyte *v);
void GLAPI glVertexAttrib4uivARB (GLuint index, const GLuint *v);
void GLAPI glVertexAttrib4usvARB (GLuint index, const GLushort *v);
void GLAPI glVertexAttribPointerARB (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
void GLAPI glEnableVertexAttribArrayARB (GLuint index);
void GLAPI glDisableVertexAttribArrayARB (GLuint index);
void GLAPI glProgramStringARB (GLenum target, GLenum format, GLsizei len, const GLvoid *string);
void GLAPI glBindProgramARB (GLenum target, GLuint program);
void GLAPI glDeleteProgramsARB (GLsizei n, const GLuint *programs);
void GLAPI glGenProgramsARB (GLsizei n, GLuint *programs);
void GLAPI glProgramEnvParameter4dARB (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GLAPI glProgramEnvParameter4dvARB (GLenum target, GLuint index, const GLdouble *params);
void GLAPI glProgramEnvParameter4fARB (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GLAPI glProgramEnvParameter4fvARB (GLenum target, GLuint index, const GLfloat *params);
void GLAPI glProgramLocalParameter4dARB (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
void GLAPI glProgramLocalParameter4dvARB (GLenum target, GLuint index, const GLdouble *params);
void GLAPI glProgramLocalParameter4fARB (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void GLAPI glProgramLocalParameter4fvARB (GLenum target, GLuint index, const GLfloat *params);
void GLAPI glGetProgramEnvParameterdvARB (GLenum target, GLuint index, GLdouble *params);
void GLAPI glGetProgramEnvParameterfvARB (GLenum target, GLuint index, GLfloat *params);
void GLAPI glGetProgramLocalParameterdvARB (GLenum target, GLuint index, GLdouble *params);
void GLAPI glGetProgramLocalParameterfvARB (GLenum target, GLuint index, GLfloat *params);
void GLAPI glGetProgramivARB (GLenum target, GLenum pname, GLint *params);
void GLAPI glGetProgramStringARB (GLenum target, GLenum pname, GLvoid *string);
void GLAPI glGetVertexAttribdvARB (GLuint index, GLenum pname, GLdouble *params);
void GLAPI glGetVertexAttribfvARB (GLuint index, GLenum pname, GLfloat *params);
void GLAPI glGetVertexAttribivARB (GLuint index, GLenum pname, GLint *params);
void GLAPI glGetVertexAttribPointervARB (GLuint index, GLenum pname, GLvoid* *pointer);
GLboolean GLAPI glIsProgramARB (GLuint program);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLVERTEXATTRIB1DARBPROC) (GLuint index, GLdouble x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1DVARBPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB1FARBPROC) (GLuint index, GLfloat x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1FVARBPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB1SARBPROC) (GLuint index, GLshort x);
typedef void (GLAPI * PFNGLVERTEXATTRIB1SVARBPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2DARBPROC) (GLuint index, GLdouble x, GLdouble y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2DVARBPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2FARBPROC) (GLuint index, GLfloat x, GLfloat y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2FVARBPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB2SARBPROC) (GLuint index, GLshort x, GLshort y);
typedef void (GLAPI * PFNGLVERTEXATTRIB2SVARBPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3DARBPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3DVARBPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3FARBPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3FVARBPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB3SARBPROC) (GLuint index, GLshort x, GLshort y, GLshort z);
typedef void (GLAPI * PFNGLVERTEXATTRIB3SVARBPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NBVARBPROC) (GLuint index, const GLbyte *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NIVARBPROC) (GLuint index, const GLint *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NSVARBPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NUBARBPROC) (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NUBVARBPROC) (GLuint index, const GLubyte *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NUIVARBPROC) (GLuint index, const GLuint *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4NUSVARBPROC) (GLuint index, const GLushort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4BVARBPROC) (GLuint index, const GLbyte *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4DARBPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4DVARBPROC) (GLuint index, const GLdouble *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4FARBPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4FVARBPROC) (GLuint index, const GLfloat *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4IVARBPROC) (GLuint index, const GLint *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4SARBPROC) (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (GLAPI * PFNGLVERTEXATTRIB4SVARBPROC) (GLuint index, const GLshort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4UBVARBPROC) (GLuint index, const GLubyte *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4UIVARBPROC) (GLuint index, const GLuint *v);
typedef void (GLAPI * PFNGLVERTEXATTRIB4USVARBPROC) (GLuint index, const GLushort *v);
typedef void (GLAPI * PFNGLVERTEXATTRIBPOINTERARBPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
typedef void (GLAPI * PFNGLENABLEVERTEXATTRIBARRAYARBPROC) (GLuint index);
typedef void (GLAPI * PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) (GLuint index);
typedef void (GLAPI * PFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const GLvoid *string);
typedef void (GLAPI * PFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (GLAPI * PFNGLDELETEPROGRAMSARBPROC) (GLsizei n, const GLuint *programs);
typedef void (GLAPI * PFNGLGENPROGRAMSARBPROC) (GLsizei n, GLuint *programs);
typedef void (GLAPI * PFNGLPROGRAMENVPARAMETER4DARBPROC) (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPI * PFNGLPROGRAMENVPARAMETER4DVARBPROC) (GLenum target, GLuint index, const GLdouble *params);
typedef void (GLAPI * PFNGLPROGRAMENVPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPI * PFNGLPROGRAMENVPARAMETER4FVARBPROC) (GLenum target, GLuint index, const GLfloat *params);
typedef void (GLAPI * PFNGLPROGRAMLOCALPARAMETER4DARBPROC) (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPI * PFNGLPROGRAMLOCALPARAMETER4DVARBPROC) (GLenum target, GLuint index, const GLdouble *params);
typedef void (GLAPI * PFNGLPROGRAMLOCALPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPI * PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) (GLenum target, GLuint index, const GLfloat *params);
typedef void (GLAPI * PFNGLGETPROGRAMENVPARAMETERDVARBPROC) (GLenum target, GLuint index, GLdouble *params);
typedef void (GLAPI * PFNGLGETPROGRAMENVPARAMETERFVARBPROC) (GLenum target, GLuint index, GLfloat *params);
typedef void (GLAPI * PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC) (GLenum target, GLuint index, GLdouble *params);
typedef void (GLAPI * PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC) (GLenum target, GLuint index, GLfloat *params);
typedef void (GLAPI * PFNGLGETPROGRAMIVARBPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETPROGRAMSTRINGARBPROC) (GLenum target, GLenum pname, GLvoid *string);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBDVARBPROC) (GLuint index, GLenum pname, GLdouble *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBFVARBPROC) (GLuint index, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBIVARBPROC) (GLuint index, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETVERTEXATTRIBPOINTERVARBPROC) (GLuint index, GLenum pname, GLvoid* *pointer);
typedef GLboolean (GLAPI * PFNGLISPROGRAMARBPROC) (GLuint program);
#endif


#ifndef GL_NV_primitive_restart
#define GL_NV_primitive_restart 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glPrimitiveRestartNV (void);
void GLAPI glPrimitiveRestartIndexNV (GLuint index);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLPRIMITIVERESTARTNVPROC) (void);
typedef void (GLAPI * PFNGLPRIMITIVERESTARTINDEXNVPROC) (GLuint index);
#endif


#ifndef GL_ARB_vertex_buffer_object
#define GL_ARB_vertex_buffer_object 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBindBufferARB (GLenum target, GLuint buffer);
void GLAPI glBufferDataARB (GLenum target, GLsizeiptrARB size, const GLvoid *data, GLenum usage);
void GLAPI glBufferSubDataARB (GLenum target, GLintptrARB offset, GLsizeiptrARB size, const GLvoid *data);
void GLAPI glDeleteBuffersARB (GLsizei n, const GLuint *buffers);
void GLAPI glGenBuffersARB (GLsizei n, GLuint *buffers);
void GLAPI glGetBufferParameterivARB (GLenum target, GLenum pname, GLint *params);
void GLAPI glGetBufferPointervARB (GLenum target, GLenum pname, GLvoid* *params);
void GLAPI glGetBufferSubDataARB (GLenum target, GLintptrARB offset, GLsizeiptrARB size, GLvoid *data);
GLboolean GLAPI glIsBufferARB (GLuint buffer);
GLvoid* GLAPI glMapBufferARB (GLenum target, GLenum access);
GLboolean GLAPI glUnmapBufferARB (GLenum target);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBINDBUFFERARBPROC) (GLenum target, GLuint buffer);
typedef void (GLAPI * PFNGLBUFFERDATAARBPROC) (GLenum target, GLsizeiptrARB size, const GLvoid *data, GLenum usage);
typedef void (GLAPI * PFNGLBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset, GLsizeiptrARB size, const GLvoid *data);
typedef void (GLAPI * PFNGLDELETEBUFFERSARBPROC) (GLsizei n, const GLuint *buffers);
typedef void (GLAPI * PFNGLGENBUFFERSARBPROC) (GLsizei n, GLuint *buffers);
typedef void (GLAPI * PFNGLGETBUFFERPARAMETERIVARBPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETBUFFERPOINTERVARBPROC) (GLenum target, GLenum pname, GLvoid* *params);
typedef void (GLAPI * PFNGLGETBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset, GLsizeiptrARB size, GLvoid *data);
typedef GLboolean (GLAPI * PFNGLISBUFFERARBPROC) (GLuint buffer);
typedef GLvoid* (GLAPI * PFNGLMAPBUFFERARBPROC) (GLenum target, GLenum access);
typedef GLboolean (GLAPI * PFNGLUNMAPBUFFERARBPROC) (GLenum target);
#endif


#ifndef GL_ARB_vertex_array_set_object
#define GL_ARB_vertex_array_set_object 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBindArraySetARB (GLuint buffer);
void GLAPI glDeleteArraySetsARB (GLsizei n, const GLuint *buffers);
void GLAPI glGenArraySetsARB (GLsizei n, GLuint *buffers);
GLboolean GLAPI glIsArraySetARB (GLuint buffer);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBINDARRAYSETARBPROC) (GLuint buffer);
typedef void (GLAPI * PFNGLDELETEARRAYSETSARBPROC) (GLsizei n, const GLuint *buffers);
typedef void (GLAPI * PFNGLGENARRAYSETSARBPROC) (GLsizei n, GLuint *buffers);
typedef GLboolean (GLAPI * PFNGLISARRAYSETARBPROC) (GLuint buffer);
#endif


#ifndef GL_ARB_occlusion_query
#define GL_ARB_occlusion_query 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glGenQueriesARB (GLsizei n, GLuint *ids);
void GLAPI glDeleteQueriesARB (GLsizei n, const GLuint *ids);
GLboolean GLAPI glIsQueryARB (GLuint id);
void GLAPI glBeginQueryARB (GLenum target, GLuint id);
void GLAPI glEndQueryARB (GLenum target);
void GLAPI glGetQueryObjectivARB (GLuint id, GLenum pname, GLint *params);
void GLAPI glGetQueryObjectuivARB (GLuint id, GLenum pname, GLuint *params);
void GLAPI glGetQueryivARB (GLenum target, GLenum pname, GLint *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLGENQUERIESARBPROC) (GLsizei n, GLuint *ids);
typedef void (GLAPI * PFNGLDELETEQUERIESARBPROC) (GLsizei n, const GLuint *ids);
typedef GLboolean (GLAPI * PFNGLISQUERYARBPROC) (GLuint id);
typedef void (GLAPI * PFNGLBEGINQUERYARBPROC) (GLenum target, GLuint id);
typedef void (GLAPI * PFNGLENDQUERYARBPROC) (GLenum target);
typedef void (GLAPI * PFNGLGETQUERYOBJECTIVARBPROC) (GLuint id, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETQUERYOBJECTUIVARBPROC) (GLuint id, GLenum pname, GLuint *params);
typedef void (GLAPI * PFNGLGETQUERYIVARBPROC) (GLenum target, GLenum pname, GLint *params);
#endif


#ifndef GL_ATI_draw_buffers
#define GL_ATI_draw_buffers 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glDrawBuffersATI (GLsizei n, const GLenum *bufs);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLDRAWBUFFERSATIPROC) (GLsizei n, const GLenum *bufs);
#endif


#ifndef GL_EXT_blend_equation_separate
#define GL_EXT_blend_equation_separate 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBlendEquationSeparateEXT (GLenum modeRGB, GLenum modeAlpha);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBLENDEQUATIONSEPARATEEXTPROC) (GLenum modeRGB, GLenum modeAlpha);
#endif


#ifndef GL_ARB_shader_objects
#define GL_ARB_shader_objects 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glDeleteObjectARB (GLhandleARB obj);
GLhandleARB GLAPI glGetHandleARB (GLenum pname);
void GLAPI glDetachObjectARB (GLhandleARB containerObj, GLhandleARB attachedObj);
GLhandleARB GLAPI glCreateShaderObjectARB (GLenum shaderType);
void GLAPI glShaderSourceARB (GLhandleARB shaderObj, GLsizei count, const GLcharARB* *string, const GLint *length);
void GLAPI glCompileShaderARB (GLhandleARB shaderObj);
GLhandleARB GLAPI glCreateProgramObjectARB (void);
void GLAPI glAttachObjectARB (GLhandleARB containerObj, GLhandleARB attachedObj);
void GLAPI glLinkProgramARB (GLhandleARB programObj);
void GLAPI glUseProgramObjectARB (GLhandleARB programObj);
void GLAPI glValidateProgramARB (GLhandleARB programObj);
void GLAPI glUniform1fARB (GLint location, GLfloat v0);
void GLAPI glUniform2fARB (GLint location, GLfloat v0, GLfloat v1);
void GLAPI glUniform3fARB (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
void GLAPI glUniform4fARB (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
void GLAPI glUniform1iARB (GLint location, GLint v0);
void GLAPI glUniform2iARB (GLint location, GLint v0, GLint v1);
void GLAPI glUniform3iARB (GLint location, GLint v0, GLint v1, GLint v2);
void GLAPI glUniform4iARB (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
void GLAPI glUniform1fvARB (GLint location, GLsizei count, const GLfloat *value);
void GLAPI glUniform2fvARB (GLint location, GLsizei count, const GLfloat *value);
void GLAPI glUniform3fvARB (GLint location, GLsizei count, const GLfloat *value);
void GLAPI glUniform4fvARB (GLint location, GLsizei count, const GLfloat *value);
void GLAPI glUniform1ivARB (GLint location, GLsizei count, const GLint *value);
void GLAPI glUniform2ivARB (GLint location, GLsizei count, const GLint *value);
void GLAPI glUniform3ivARB (GLint location, GLsizei count, const GLint *value);
void GLAPI glUniform4ivARB (GLint location, GLsizei count, const GLint *value);
void GLAPI glUniformMatrix2fvARB (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
void GLAPI glUniformMatrix3fvARB (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
void GLAPI glUniformMatrix4fvARB (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
void GLAPI glGetObjectParameterfvARB (GLhandleARB obj, GLenum pname, GLfloat *params);
void GLAPI glGetObjectParameterivARB (GLhandleARB obj, GLenum pname, GLint *params);
void GLAPI glGetInfoLogARB (GLhandleARB obj, GLsizei maxLength, GLsizei *length, GLcharARB *infoLog);
void GLAPI glGetAttachedObjectsARB (GLhandleARB containerObj, GLsizei maxCount, GLsizei *count, GLhandleARB *obj);
GLint GLAPI glGetUniformLocationARB (GLhandleARB programObj, const GLcharARB *name);
void GLAPI glGetActiveUniformARB (GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei *length, GLsizei *size, GLenum *type, GLcharARB *name);
void GLAPI glGetUniformfvARB (GLhandleARB programObj, GLint location, GLfloat *params);
void GLAPI glGetUniformivARB (GLhandleARB programObj, GLint location, GLint *params);
void GLAPI glGetShaderSourceARB (GLhandleARB obj, GLsizei maxLength, GLsizei *length, GLcharARB *source);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLDELETEOBJECTARBPROC) (GLhandleARB obj);
typedef GLhandleARB (GLAPI * PFNGLGETHANDLEARBPROC) (GLenum pname);
typedef void (GLAPI * PFNGLDETACHOBJECTARBPROC) (GLhandleARB containerObj, GLhandleARB attachedObj);
typedef GLhandleARB (GLAPI * PFNGLCREATESHADEROBJECTARBPROC) (GLenum shaderType);
typedef void (GLAPI * PFNGLSHADERSOURCEARBPROC) (GLhandleARB shaderObj, GLsizei count, const GLcharARB* *string, const GLint *length);
typedef void (GLAPI * PFNGLCOMPILESHADERARBPROC) (GLhandleARB shaderObj);
typedef GLhandleARB (GLAPI * PFNGLCREATEPROGRAMOBJECTARBPROC) (void);
typedef void (GLAPI * PFNGLATTACHOBJECTARBPROC) (GLhandleARB containerObj, GLhandleARB attachedObj);
typedef void (GLAPI * PFNGLLINKPROGRAMARBPROC) (GLhandleARB programObj);
typedef void (GLAPI * PFNGLUSEPROGRAMOBJECTARBPROC) (GLhandleARB programObj);
typedef void (GLAPI * PFNGLVALIDATEPROGRAMARBPROC) (GLhandleARB programObj);
typedef void (GLAPI * PFNGLUNIFORM1FARBPROC) (GLint location, GLfloat v0);
typedef void (GLAPI * PFNGLUNIFORM2FARBPROC) (GLint location, GLfloat v0, GLfloat v1);
typedef void (GLAPI * PFNGLUNIFORM3FARBPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
typedef void (GLAPI * PFNGLUNIFORM4FARBPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
typedef void (GLAPI * PFNGLUNIFORM1IARBPROC) (GLint location, GLint v0);
typedef void (GLAPI * PFNGLUNIFORM2IARBPROC) (GLint location, GLint v0, GLint v1);
typedef void (GLAPI * PFNGLUNIFORM3IARBPROC) (GLint location, GLint v0, GLint v1, GLint v2);
typedef void (GLAPI * PFNGLUNIFORM4IARBPROC) (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
typedef void (GLAPI * PFNGLUNIFORM1FVARBPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (GLAPI * PFNGLUNIFORM2FVARBPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (GLAPI * PFNGLUNIFORM3FVARBPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (GLAPI * PFNGLUNIFORM4FVARBPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (GLAPI * PFNGLUNIFORM1IVARBPROC) (GLint location, GLsizei count, const GLint *value);
typedef void (GLAPI * PFNGLUNIFORM2IVARBPROC) (GLint location, GLsizei count, const GLint *value);
typedef void (GLAPI * PFNGLUNIFORM3IVARBPROC) (GLint location, GLsizei count, const GLint *value);
typedef void (GLAPI * PFNGLUNIFORM4IVARBPROC) (GLint location, GLsizei count, const GLint *value);
typedef void (GLAPI * PFNGLUNIFORMMATRIX2FVARBPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (GLAPI * PFNGLUNIFORMMATRIX3FVARBPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (GLAPI * PFNGLUNIFORMMATRIX4FVARBPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (GLAPI * PFNGLGETOBJECTPARAMETERFVARBPROC) (GLhandleARB obj, GLenum pname, GLfloat *params);
typedef void (GLAPI * PFNGLGETOBJECTPARAMETERIVARBPROC) (GLhandleARB obj, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETINFOLOGARBPROC) (GLhandleARB obj, GLsizei maxLength, GLsizei *length, GLcharARB *infoLog);
typedef void (GLAPI * PFNGLGETATTACHEDOBJECTSARBPROC) (GLhandleARB containerObj, GLsizei maxCount, GLsizei *count, GLhandleARB *obj);
typedef GLint (GLAPI * PFNGLGETUNIFORMLOCATIONARBPROC) (GLhandleARB programObj, const GLcharARB *name);
typedef void (GLAPI * PFNGLGETACTIVEUNIFORMARBPROC) (GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei *length, GLsizei *size, GLenum *type, GLcharARB *name);
typedef void (GLAPI * PFNGLGETUNIFORMFVARBPROC) (GLhandleARB programObj, GLint location, GLfloat *params);
typedef void (GLAPI * PFNGLGETUNIFORMIVARBPROC) (GLhandleARB programObj, GLint location, GLint *params);
typedef void (GLAPI * PFNGLGETSHADERSOURCEARBPROC) (GLhandleARB obj, GLsizei maxLength, GLsizei *length, GLcharARB *source);
#endif


#ifndef GL_ARB_vertex_shader
#define GL_ARB_vertex_shader 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBindAttribLocationARB (GLhandleARB programObj, GLuint index, const GLcharARB *name);
void GLAPI glGetActiveAttribARB (GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei *length, GLsizei *size, GLenum *type, GLcharARB *name);
GLint GLAPI glGetAttribLocationARB (GLhandleARB programObj, const GLcharARB *name);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBINDATTRIBLOCATIONARBPROC) (GLhandleARB programObj, GLuint index, const GLcharARB *name);
typedef void (GLAPI * PFNGLGETACTIVEATTRIBARBPROC) (GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei *length, GLsizei *size, GLenum *type, GLcharARB *name);
typedef GLint (GLAPI * PFNGLGETATTRIBLOCATIONARBPROC) (GLhandleARB programObj, const GLcharARB *name);
#endif


#ifndef GL_VERSION_1_5
#define GL_VERSION_1_5 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBindBuffer (GLenum target, GLuint buffer);
void GLAPI glBufferData (GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);
void GLAPI glBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid *data);
void GLAPI glDeleteBuffers (GLsizei n, const GLuint *buffers);
void GLAPI glGenBuffers (GLsizei n, GLuint *buffers);
void GLAPI glGetBufferParameteriv (GLenum target, GLenum pname, GLint *params);
void GLAPI glGetBufferPointerv (GLenum target, GLenum pname, GLvoid* *params);
void GLAPI glGetBufferSubData (GLenum target, GLintptr offset, GLsizeiptr size, GLvoid *data);
GLboolean GLAPI glIsBuffer (GLuint buffer);
GLvoid* GLAPI glMapBuffer (GLenum target, GLenum access);
GLboolean GLAPI glUnmapBuffer (GLenum target);
void GLAPI glGenQueries (GLsizei n, GLuint *ids);
void GLAPI glDeleteQueries (GLsizei n, const GLuint *ids);
GLboolean GLAPI glIsQuery (GLuint id);
void GLAPI glBeginQuery (GLenum target, GLuint id);
void GLAPI glEndQuery (GLenum target);
void GLAPI glGetQueryObjectiv (GLuint id, GLenum pname, GLint *params);
void GLAPI glGetQueryObjectuiv (GLuint id, GLenum pname, GLuint *params);
void GLAPI glGetQueryiv (GLenum target, GLenum pname, GLint *params);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
typedef void (GLAPI * PFNGLBUFFERDATAPROC) (GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);
typedef void (GLAPI * PFNGLBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid *data);
typedef void (GLAPI * PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);
typedef void (GLAPI * PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (GLAPI * PFNGLGETBUFFERPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETBUFFERPOINTERVPROC) (GLenum target, GLenum pname, GLvoid* *params);
typedef void (GLAPI * PFNGLGETBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, GLvoid *data);
typedef GLboolean (GLAPI * PFNGLISBUFFERPROC) (GLuint buffer);
typedef GLvoid* (GLAPI * PFNGLMAPBUFFERPROC) (GLenum target, GLenum access);
typedef GLboolean (GLAPI * PFNGLUNMAPBUFFERPROC) (GLenum target);
typedef void (GLAPI * PFNGLGENQUERIESPROC) (GLsizei n, GLuint *ids);
typedef void (GLAPI * PFNGLDELETEQUERIESPROC) (GLsizei n, const GLuint *ids);
typedef GLboolean (GLAPI * PFNGLISQUERYPROC) (GLuint id);
typedef void (GLAPI * PFNGLBEGINQUERYPROC) (GLenum target, GLuint id);
typedef void (GLAPI * PFNGLENDQUERYPROC) (GLenum target);
typedef void (GLAPI * PFNGLGETQUERYOBJECTIVPROC) (GLuint id, GLenum pname, GLint *params);
typedef void (GLAPI * PFNGLGETQUERYOBJECTUIVPROC) (GLuint id, GLenum pname, GLuint *params);
typedef void (GLAPI * PFNGLGETQUERYIVPROC) (GLenum target, GLenum pname, GLint *params);
#endif


#ifndef GL_NVX_conditional_render
#define GL_NVX_conditional_render 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBeginConditionalRenderNVX (GLuint id);
void GLAPI glEndConditionalRenderNVX (void);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBEGINCONDITIONALRENDERNVXPROC) (GLuint id);
typedef void (GLAPI * PFNGLENDCONDITIONALRENDERNVXPROC) (void);
#endif


#ifndef GL_EXT_framebuffer_object
#define GL_EXT_framebuffer_object 1
#ifdef GL_GLEXT_PROTOTYPES
void GLAPI glBindFramebufferEXT (GLenum target, GLuint id);
void GLAPI glDeleteFramebuffersEXT (GLsizei n, const GLuint *ids);
void GLAPI glGenFramebuffersEXT (GLsizei n, GLuint *ids);
void GLAPI glFramebufferTextureEXT (GLenum target, GLenum buffer, GLuint id, GLuint level, GLenum face, GLuint image);
void GLAPI glFramebufferStorageEXT (GLenum target, GLenum buffer, GLenum internalFormat, GLint width, GLint height);
void GLAPI glGetFramebufferBufferParameterivEXT (GLenum target, GLenum buffer, GLenum value, GLint *T);
void GLAPI glGetFramebufferBufferParameterfvEXT (GLenum target, GLenum buffer, GLenum value, GLfloat *T);
GLboolean GLAPI glValidateFramebufferEXT (GLenum target);
void GLAPI glGenerateMipmapEXT (GLenum target);
#endif /* GL_GLEXT_PROTOTYPES */
typedef void (GLAPI * PFNGLBINDFRAMEBUFFEREXTPROC) (GLenum target, GLuint id);
typedef void (GLAPI * PFNGLDELETEFRAMEBUFFERSEXTPROC) (GLsizei n, const GLuint *ids);
typedef void (GLAPI * PFNGLGENFRAMEBUFFERSEXTPROC) (GLsizei n, GLuint *ids);
typedef void (GLAPI * PFNGLFRAMEBUFFERTEXTUREEXTPROC) (GLenum target, GLenum buffer, GLuint id, GLuint level, GLenum face, GLuint image);
typedef void (GLAPI * PFNGLFRAMEBUFFERSTORAGEEXTPROC) (GLenum target, GLenum buffer, GLenum internalFormat, GLint width, GLint height);
typedef void (GLAPI * PFNGLGETFRAMEBUFFERBUFFERPARAMETERIVEXTPROC) (GLenum target, GLenum buffer, GLenum value, GLint *T);
typedef void (GLAPI * PFNGLGETFRAMEBUFFERBUFFERPARAMETERFVEXTPROC) (GLenum target, GLenum buffer, GLenum value, GLfloat *T);
typedef GLboolean (GLAPI * PFNGLVALIDATEFRAMEBUFFEREXTPROC) (GLenum target);
typedef void (GLAPI * PFNGLGENERATEMIPMAPEXTPROC) (GLenum target);
#endif

/*************************************************************/

/* Version */
#ifndef GL_VERSION_1_2
#define GL_VERSION_1_2                    1
#endif
#ifndef GL_VERSION_1_3
#define GL_VERSION_1_3                    1
#endif
#ifndef GL_VERSION_1_4
#define GL_VERSION_1_4                    1
#endif
#ifndef GL_VERSION_1_5
#define GL_VERSION_1_5                    1
#endif

/* Extensions */
#ifndef GL_APPLE_transform_hint
#define GL_APPLE_transform_hint           1
#endif
#ifndef GL_ARB_depth_texture
#define GL_ARB_depth_texture              1
#endif
#ifndef GL_ARB_fragment_program
#define GL_ARB_fragment_program           1
#endif
#ifndef GL_ARB_fragment_program_shadow
#define GL_ARB_fragment_program_shadow    1
#endif
#ifndef GL_ARB_fragment_shader
#define GL_ARB_fragment_shader            1
#endif
#ifndef GL_ARB_imaging
#define GL_ARB_imaging                    1
#endif
#ifndef GL_ARB_multisample
#define GL_ARB_multisample                1
#endif
#ifndef GL_ARB_multitexture
#define GL_ARB_multitexture               1
#endif
#ifndef GL_ARB_occlusion_query
#define GL_ARB_occlusion_query            1
#endif
#ifndef GL_ARB_point_parameters
#define GL_ARB_point_parameters           1
#endif
#ifndef GL_ARB_point_sprite
#define GL_ARB_point_sprite               1
#endif
#ifndef GL_ARB_shader_objects
#define GL_ARB_shader_objects             1
#endif
#ifndef GL_ARB_shading_language_100
#define GL_ARB_shading_language_100       1
#endif
#ifndef GL_ARB_shadow
#define GL_ARB_shadow                     1
#endif
#ifndef GL_ARB_shadow_ambient
#define GL_ARB_shadow_ambient             1
#endif
#ifndef GL_ARB_texture_border_clamp
#define GL_ARB_texture_border_clamp       1
#endif
#ifndef GL_ARB_texture_compression
#define GL_ARB_texture_compression        1
#endif
#ifndef GL_ARB_texture_cube_map
#define GL_ARB_texture_cube_map           1
#endif
#ifndef GL_ARB_texture_env_add
#define GL_ARB_texture_env_add            1
#endif
#ifndef GL_ARB_texture_env_combine
#define GL_ARB_texture_env_combine        1
#endif
#ifndef GL_ARB_texture_env_dot3
#define GL_ARB_texture_env_dot3           1
#endif
#ifndef GL_ARB_texture_mirrored_repeat
#define GL_ARB_texture_mirrored_repeat    1
#endif
#ifndef GL_ARB_texture_non_power_of_two
#define GL_ARB_texture_non_power_of_two   1
#endif
#ifndef GL_ARB_texture_rectangle
#define GL_ARB_texture_rectangle          1
#endif
#ifndef GL_ARB_transpose_matrix
#define GL_ARB_transpose_matrix           1
#endif
#ifndef GL_ARB_vertex_buffer_object
#define GL_ARB_vertex_buffer_object       1
#endif
#ifndef GL_ARB_vertex_array_set_object
#define GL_ARB_vertex_array_set_object    1
#endif
#ifndef GL_ARB_vertex_program
#define GL_ARB_vertex_program             1
#endif
#ifndef GL_ARB_vertex_shader
#define GL_ARB_vertex_shader              1
#endif
#ifndef GL_ARB_window_pos
#define GL_ARB_window_pos                 1
#endif
#ifndef GL_ATI_draw_buffers
#define GL_ATI_draw_buffers               1
#endif
#ifndef GL_ATI_pixel_format_float
#define GL_ATI_pixel_format_float         1
#endif
#ifndef GL_ATI_texture_float
#define GL_ATI_texture_float              1
#endif
#ifndef GL_ATI_texture_mirror_once
#define GL_ATI_texture_mirror_once        1
#endif
#ifndef GL_Autodesk_valid_back_buffer_hint
#define GL_Autodesk_valid_back_buffer_hint 1
#endif
#ifndef GL_EXT_Cg_shader
#define GL_EXT_Cg_shader                  1
#endif
#ifndef GL_EXT_abgr
#define GL_EXT_abgr                       1
#endif
#ifndef GL_EXT_bgra
#define GL_EXT_bgra                       1
#endif
#ifndef GL_EXT_blend_color
#define GL_EXT_blend_color                1
#endif
#ifndef GL_EXT_blend_equation_separate
#define GL_EXT_blend_equation_separate    1
#endif
#ifndef GL_EXT_blend_func_separate
#define GL_EXT_blend_func_separate        1
#endif
#ifndef GL_EXT_blend_minmax
#define GL_EXT_blend_minmax               1
#endif
#ifndef GL_EXT_blend_subtract
#define GL_EXT_blend_subtract             1
#endif
#ifndef GL_EXT_clip_volume_hint
#define GL_EXT_clip_volume_hint           1
#endif
#ifndef GL_EXT_color_table
#define GL_EXT_color_table                1
#endif
#ifndef GL_EXT_compiled_vertex_array
#define GL_EXT_compiled_vertex_array      1
#endif
#ifndef GL_EXT_depth_bounds_test
#define GL_EXT_depth_bounds_test          1
#endif
#ifndef GL_EXT_draw_range_elements
#define GL_EXT_draw_range_elements        1
#endif
#ifndef GL_EXT_fog_coord
#define GL_EXT_fog_coord                  1
#endif
#ifndef GL_EXT_framebuffer_object
#define GL_EXT_framebuffer_object         1
#endif
#ifndef GL_EXT_multi_draw_arrays
#define GL_EXT_multi_draw_arrays          1
#endif
#ifndef GL_EXT_packed_pixels
#define GL_EXT_packed_pixels              1
#endif
#ifndef GL_EXT_paletted_texture
#define GL_EXT_paletted_texture           1
#endif
#ifndef GL_EXT_pixel_buffer_object
#define GL_EXT_pixel_buffer_object        1
#endif
#ifndef GL_EXT_point_parameters
#define GL_EXT_point_parameters           1
#endif
#ifndef GL_EXT_rescale_normal
#define GL_EXT_rescale_normal             1
#endif
#ifndef GL_EXT_secondary_color
#define GL_EXT_secondary_color            1
#endif
#ifndef GL_EXT_separate_specular_color
#define GL_EXT_separate_specular_color    1
#endif
#ifndef GL_EXT_shadow_funcs
#define GL_EXT_shadow_funcs               1
#endif
#ifndef GL_EXT_shared_texture_palette
#define GL_EXT_shared_texture_palette     1
#endif
#ifndef GL_EXT_stencil_two_side
#define GL_EXT_stencil_two_side           1
#endif
#ifndef GL_EXT_stencil_wrap
#define GL_EXT_stencil_wrap               1
#endif
#ifndef GL_EXT_texture3D
#define GL_EXT_texture3D                  1
#endif
#ifndef GL_EXT_texture_compression_s3tc
#define GL_EXT_texture_compression_s3tc   1
#endif
#ifndef GL_EXT_texture_cube_map
#define GL_EXT_texture_cube_map           1
#endif
#ifndef GL_EXT_texture_edge_clamp
#define GL_EXT_texture_edge_clamp         1
#endif
#ifndef GL_EXT_texture_env_add
#define GL_EXT_texture_env_add            1
#endif
#ifndef GL_EXT_texture_env_combine
#define GL_EXT_texture_env_combine        1
#endif
#ifndef GL_EXT_texture_env_dot3
#define GL_EXT_texture_env_dot3           1
#endif
#ifndef GL_EXT_texture_filter_anisotropic
#define GL_EXT_texture_filter_anisotropic 1
#endif
#ifndef GL_EXT_texture_lod_bias
#define GL_EXT_texture_lod_bias           1
#endif
#ifndef GL_EXT_texture_mirror_clamp
#define GL_EXT_texture_mirror_clamp       1
#endif
#ifndef GL_EXT_texture_object
#define GL_EXT_texture_object             1
#endif
#ifndef GL_EXT_vertex_array
#define GL_EXT_vertex_array               1
#endif
#ifndef GL_HP_occlusion_test
#define GL_HP_occlusion_test              1
#endif
#ifndef GL_IBM_rasterpos_clip
#define GL_IBM_rasterpos_clip             1
#endif
#ifndef GL_IBM_texture_mirrored_repeat
#define GL_IBM_texture_mirrored_repeat    1
#endif
#ifndef GL_NVX_conditional_render
#define GL_NVX_conditional_render         1
#endif
#ifndef GL_NVX_hrsd_pixels
#define GL_NVX_hrsd_pixels                1
#endif
#ifndef GL_NV_blend_square
#define GL_NV_blend_square                1
#endif
#ifndef GL_NV_centroid_sample
#define GL_NV_centroid_sample             1
#endif
#ifndef GL_NV_copy_depth_to_color
#define GL_NV_copy_depth_to_color         1
#endif
#ifndef GL_NV_depth_clamp
#define GL_NV_depth_clamp                 1
#endif
#ifndef GL_NV_draw_mesh
#define GL_NV_draw_mesh                   1
#endif
#ifndef GL_NV_extended_combiner_program
#define GL_NV_extended_combiner_program   1
#endif
#ifndef GL_NV_fence
#define GL_NV_fence                       1
#endif
#ifndef GL_NV_float_buffer
#define GL_NV_float_buffer                1
#endif
#ifndef GL_NV_fog_distance
#define GL_NV_fog_distance                1
#endif
#ifndef GL_NV_fragment_program
#define GL_NV_fragment_program            1
#endif
#ifndef GL_NV_fragment_program2
#define GL_NV_fragment_program2           1
#endif
#ifndef GL_NV_half_float
#define GL_NV_half_float                  1
#endif
#ifndef GL_NV_light_max_exponent
#define GL_NV_light_max_exponent          1
#endif
#ifndef GL_NV_mac_get_proc_address
#define GL_NV_mac_get_proc_address        1
#endif
#ifndef GL_NV_multisample_filter_hint
#define GL_NV_multisample_filter_hint     1
#endif
#ifndef GL_NV_occlusion_query
#define GL_NV_occlusion_query             1
#endif
#ifndef GL_NV_packed_depth_stencil
#define GL_NV_packed_depth_stencil        1
#endif
#ifndef GL_NV_pixel_data_range
#define GL_NV_pixel_data_range            1
#endif
#ifndef GL_NV_point_sprite
#define GL_NV_point_sprite                1
#endif
#ifndef GL_NV_primitive_restart
#define GL_NV_primitive_restart           1
#endif
#ifndef GL_NV_register_combiners
#define GL_NV_register_combiners          1
#endif
#ifndef GL_NV_register_combiners2
#define GL_NV_register_combiners2         1
#endif
#ifndef GL_NV_set_window_stereomode
#define GL_NV_set_window_stereomode       1
#endif
#ifndef GL_NV_texgen_reflection
#define GL_NV_texgen_reflection           1
#endif
#ifndef GL_NV_texture_compression_vtc
#define GL_NV_texture_compression_vtc     1
#endif
#ifndef GL_NV_texture_env_combine4
#define GL_NV_texture_env_combine4        1
#endif
#ifndef GL_NV_texture_expand_normal
#define GL_NV_texture_expand_normal       1
#endif
#ifndef GL_NV_texture_rectangle
#define GL_NV_texture_rectangle           1
#endif
#ifndef GL_NV_texture_shader
#define GL_NV_texture_shader              1
#endif
#ifndef GL_NV_texture_shader2
#define GL_NV_texture_shader2             1
#endif
#ifndef GL_NV_texture_shader3
#define GL_NV_texture_shader3             1
#endif
#ifndef GL_NV_vertex_array_range
#define GL_NV_vertex_array_range          1
#endif
#ifndef GL_NV_vertex_array_range2
#define GL_NV_vertex_array_range2         1
#endif
#ifndef GL_NV_vertex_program
#define GL_NV_vertex_program              1
#endif
#ifndef GL_NV_vertex_program1_1
#define GL_NV_vertex_program1_1           1
#endif
#ifndef GL_NV_vertex_program2
#define GL_NV_vertex_program2             1
#endif
#ifndef GL_NV_vertex_program3
#define GL_NV_vertex_program3             1
#endif
#ifndef GL_S3_s3tc
#define GL_S3_s3tc                        1
#endif
#ifndef GL_SGIS_generate_mipmap
#define GL_SGIS_generate_mipmap           1
#endif
#ifndef GL_SGIS_multitexture
#define GL_SGIS_multitexture              1
#endif
#ifndef GL_SGIS_texture_lod
#define GL_SGIS_texture_lod               1
#endif
#ifndef GL_SGIX_depth_texture
#define GL_SGIX_depth_texture             1
#endif
#ifndef GL_SGIX_shadow
#define GL_SGIX_shadow                    1
#endif
#ifndef GL_SUN_slice_accum
#define GL_SUN_slice_accum                1
#endif
#ifndef GL_WIN_swap_hint
#define GL_WIN_swap_hint                  1
#endif

/* PixelFormat */
/*      GL_BGR_EXT */
/*      GL_BGRA_EXT */

/* GetPName */
/*      GL_ARRAY_ELEMENT_LOCK_COUNT_EXT */
/*      GL_ARRAY_ELEMENT_LOCK_FIRST_EXT */

/* GetColorTableParameterPNameEXT */
/*      GL_COLOR_TABLE_FORMAT_EXT */
/*      GL_COLOR_TABLE_WIDTH_EXT */
/*      GL_COLOR_TABLE_RED_SIZE_EXT */
/*      GL_COLOR_TABLE_GREEN_SIZE_EXT */
/*      GL_COLOR_TABLE_BLUE_SIZE_EXT */
/*      GL_COLOR_TABLE_ALPHA_SIZE_EXT */
/*      GL_COLOR_TABLE_LUMINANCE_SIZE_EXT */
/*      GL_COLOR_TABLE_INTENSITY_SIZE_EXT */

/* PixelInternalFormat */
/*      GL_COLOR_INDEX1_EXT */
/*      GL_COLOR_INDEX2_EXT */
/*      GL_COLOR_INDEX4_EXT */
/*      GL_COLOR_INDEX8_EXT */
/*      GL_COLOR_INDEX12_EXT */
/*      GL_COLOR_INDEX16_EXT */

/* OpenGL12 */
#define GL_TEXTURE_BINDING_3D             0x806A
#define GL_PACK_SKIP_IMAGES               0x806B
#define GL_PACK_IMAGE_HEIGHT              0x806C
#define GL_UNPACK_SKIP_IMAGES             0x806D
#define GL_UNPACK_IMAGE_HEIGHT            0x806E
#define GL_TEXTURE_3D                     0x806F
#define GL_PROXY_TEXTURE_3D               0x8070
#define GL_TEXTURE_DEPTH                  0x8071
#define GL_TEXTURE_WRAP_R                 0x8072
#define GL_MAX_3D_TEXTURE_SIZE            0x8073
#define GL_BGR                            0x80E0
#define GL_BGRA                           0x80E1
#define GL_UNSIGNED_BYTE_3_3_2            0x8032
#define GL_UNSIGNED_BYTE_2_3_3_REV        0x8362
#define GL_UNSIGNED_SHORT_5_6_5           0x8363
#define GL_UNSIGNED_SHORT_5_6_5_REV       0x8364
#define GL_UNSIGNED_SHORT_4_4_4_4         0x8033
#define GL_UNSIGNED_SHORT_4_4_4_4_REV     0x8365
#define GL_UNSIGNED_SHORT_5_5_5_1         0x8034
#define GL_UNSIGNED_SHORT_1_5_5_5_REV     0x8366
#define GL_UNSIGNED_INT_8_8_8_8           0x8035
#define GL_UNSIGNED_INT_8_8_8_8_REV       0x8367
#define GL_UNSIGNED_INT_10_10_10_2        0x8036
#define GL_UNSIGNED_INT_2_10_10_10_REV    0x8368
#define GL_RESCALE_NORMAL                 0x803A
#define GL_LIGHT_MODEL_COLOR_CONTROL      0x81F8
#define GL_SINGLE_COLOR                   0x81F9
#define GL_SEPARATE_SPECULAR_COLOR        0x81FA
#define GL_CLAMP_TO_EDGE                  0x812F
#define GL_TEXTURE_MIN_LOD                0x813A
#define GL_TEXTURE_MAX_LOD                0x813B
#define GL_TEXTURE_BASE_LEVEL             0x813C
#define GL_TEXTURE_MAX_LEVEL              0x813D
#define GL_MAX_ELEMENTS_VERTICES          0x80E8
#define GL_MAX_ELEMENTS_INDICES           0x80E9
#define GL_ALIASED_POINT_SIZE_RANGE       0x846D
#define GL_ALIASED_LINE_WIDTH_RANGE       0x846E

/* OpenGL13 */
#define GL_ACTIVE_TEXTURE                 0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE          0x84E1
#define GL_MAX_TEXTURE_UNITS              0x84E2
#define GL_TEXTURE0                       0x84C0
#define GL_TEXTURE1                       0x84C1
#define GL_TEXTURE2                       0x84C2
#define GL_TEXTURE3                       0x84C3
#define GL_TEXTURE4                       0x84C4
#define GL_TEXTURE5                       0x84C5
#define GL_TEXTURE6                       0x84C6
#define GL_TEXTURE7                       0x84C7
#define GL_TEXTURE8                       0x84C8
#define GL_TEXTURE9                       0x84C9
#define GL_TEXTURE10                      0x84CA
#define GL_TEXTURE11                      0x84CB
#define GL_TEXTURE12                      0x84CC
#define GL_TEXTURE13                      0x84CD
#define GL_TEXTURE14                      0x84CE
#define GL_TEXTURE15                      0x84CF
#define GL_TEXTURE16                      0x84D0
#define GL_TEXTURE17                      0x84D1
#define GL_TEXTURE18                      0x84D2
#define GL_TEXTURE19                      0x84D3
#define GL_TEXTURE20                      0x84D4
#define GL_TEXTURE21                      0x84D5
#define GL_TEXTURE22                      0x84D6
#define GL_TEXTURE23                      0x84D7
#define GL_TEXTURE24                      0x84D8
#define GL_TEXTURE25                      0x84D9
#define GL_TEXTURE26                      0x84DA
#define GL_TEXTURE27                      0x84DB
#define GL_TEXTURE28                      0x84DC
#define GL_TEXTURE29                      0x84DD
#define GL_TEXTURE30                      0x84DE
#define GL_TEXTURE31                      0x84DF
#define GL_NORMAL_MAP                     0x8511
#define GL_REFLECTION_MAP                 0x8512
#define GL_TEXTURE_CUBE_MAP               0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP       0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X    0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X    0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y    0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y    0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z    0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z    0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP         0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE      0x851C
#define GL_COMBINE                        0x8570
#define GL_COMBINE_RGB                    0x8571
#define GL_COMBINE_ALPHA                  0x8572
#define GL_RGB_SCALE                      0x8573
#define GL_ADD_SIGNED                     0x8574
#define GL_INTERPOLATE                    0x8575
#define GL_CONSTANT                       0x8576
#define GL_PRIMARY_COLOR                  0x8577
#define GL_PREVIOUS                       0x8578
#define GL_SOURCE0_RGB                    0x8580
#define GL_SOURCE1_RGB                    0x8581
#define GL_SOURCE2_RGB                    0x8582
#define GL_SOURCE0_ALPHA                  0x8588
#define GL_SOURCE1_ALPHA                  0x8589
#define GL_SOURCE2_ALPHA                  0x858A
#define GL_OPERAND0_RGB                   0x8590
#define GL_OPERAND1_RGB                   0x8591
#define GL_OPERAND2_RGB                   0x8592
#define GL_OPERAND0_ALPHA                 0x8598
#define GL_OPERAND1_ALPHA                 0x8599
#define GL_OPERAND2_ALPHA                 0x859A
#define GL_SUBTRACT                       0x84E7
#define GL_TRANSPOSE_MODELVIEW_MATRIX     0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX    0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX       0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX         0x84E6
#define GL_COMPRESSED_ALPHA               0x84E9
#define GL_COMPRESSED_LUMINANCE           0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA     0x84EB
#define GL_COMPRESSED_INTENSITY           0x84EC
#define GL_COMPRESSED_RGB                 0x84ED
#define GL_COMPRESSED_RGBA                0x84EE
#define GL_TEXTURE_COMPRESSION_HINT       0x84EF
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE  0x86A0
#define GL_TEXTURE_COMPRESSED             0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS 0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS     0x86A3
#define GL_DOT3_RGB                       0x86AE
#define GL_DOT3_RGBA                      0x86AF
#define GL_CLAMP_TO_BORDER                0x812D
#define GL_MULTISAMPLE                    0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE       0x809E
#define GL_SAMPLE_ALPHA_TO_ONE            0x809F
#define GL_SAMPLE_COVERAGE                0x80A0
#define GL_SAMPLE_BUFFERS                 0x80A8
#define GL_SAMPLES                        0x80A9
#define GL_SAMPLE_COVERAGE_VALUE          0x80AA
#define GL_SAMPLE_COVERAGE_INVERT         0x80AB
#define GL_MULTISAMPLE_BIT                0x20000000

/* EXT_bgra */
#define GL_BGR_EXT                        0x80E0
#define GL_BGRA_EXT                       0x80E1

/* EXT_blend_color */
#define GL_CONSTANT_COLOR_EXT             0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR_EXT   0x8002
#define GL_CONSTANT_ALPHA_EXT             0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA_EXT   0x8004
#define GL_BLEND_COLOR_EXT                0x8005

/* EXT_blend_minmax */
#define GL_FUNC_ADD_EXT                   0x8006
#define GL_MIN_EXT                        0x8007
#define GL_MAX_EXT                        0x8008
#define GL_BLEND_EQUATION_EXT             0x8009

/* EXT_color_table */
#define GL_TABLE_TOO_LARGE_EXT            0x8031
#define GL_COLOR_TABLE_FORMAT_EXT         0x80D8
#define GL_COLOR_TABLE_WIDTH_EXT          0x80D9
#define GL_COLOR_TABLE_RED_SIZE_EXT       0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE_EXT     0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE_EXT      0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE_EXT     0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE_EXT 0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE_EXT 0x80DF

/* EXT_paletted_texture */
#define GL_COLOR_INDEX1_EXT               0x80E2
#define GL_COLOR_INDEX2_EXT               0x80E3
#define GL_COLOR_INDEX4_EXT               0x80E4
#define GL_COLOR_INDEX8_EXT               0x80E5
#define GL_COLOR_INDEX12_EXT              0x80E6
#define GL_COLOR_INDEX16_EXT              0x80E7
#define GL_TEXTURE_INDEX_SIZE_EXT         0x80ED

/* EXT_texture3D */
#define GL_PACK_SKIP_IMAGES               0x806B
#define GL_PACK_SKIP_IMAGES_EXT           0x806B
#define GL_PACK_IMAGE_HEIGHT              0x806C
#define GL_PACK_IMAGE_HEIGHT_EXT          0x806C
#define GL_UNPACK_SKIP_IMAGES             0x806D
#define GL_UNPACK_SKIP_IMAGES_EXT         0x806D
#define GL_UNPACK_IMAGE_HEIGHT            0x806E
#define GL_UNPACK_IMAGE_HEIGHT_EXT        0x806E
#define GL_TEXTURE_3D                     0x806F
#define GL_TEXTURE_3D_EXT                 0x806F
#define GL_PROXY_TEXTURE_3D               0x8070
#define GL_PROXY_TEXTURE_3D_EXT           0x8070
#define GL_TEXTURE_DEPTH                  0x8071
#define GL_TEXTURE_DEPTH_EXT              0x8071
#define GL_TEXTURE_WRAP_R                 0x8072
#define GL_TEXTURE_WRAP_R_EXT             0x8072
#define GL_MAX_3D_TEXTURE_SIZE            0x8073
#define GL_MAX_3D_TEXTURE_SIZE_EXT        0x8073

/* EXT_vertex_array */
#define GL_VERTEX_ARRAY_EXT               0x8074
#define GL_NORMAL_ARRAY_EXT               0x8075
#define GL_COLOR_ARRAY_EXT                0x8076
#define GL_INDEX_ARRAY_EXT                0x8077
#define GL_TEXTURE_COORD_ARRAY_EXT        0x8078
#define GL_EDGE_FLAG_ARRAY_EXT            0x8079
#define GL_VERTEX_ARRAY_SIZE_EXT          0x807A
#define GL_VERTEX_ARRAY_TYPE_EXT          0x807B
#define GL_VERTEX_ARRAY_STRIDE_EXT        0x807C
#define GL_VERTEX_ARRAY_COUNT_EXT         0x807D
#define GL_NORMAL_ARRAY_TYPE_EXT          0x807E
#define GL_NORMAL_ARRAY_STRIDE_EXT        0x807F
#define GL_NORMAL_ARRAY_COUNT_EXT         0x8080
#define GL_COLOR_ARRAY_SIZE_EXT           0x8081
#define GL_COLOR_ARRAY_TYPE_EXT           0x8082
#define GL_COLOR_ARRAY_STRIDE_EXT         0x8083
#define GL_COLOR_ARRAY_COUNT_EXT          0x8084
#define GL_INDEX_ARRAY_TYPE_EXT           0x8085
#define GL_INDEX_ARRAY_STRIDE_EXT         0x8086
#define GL_INDEX_ARRAY_COUNT_EXT          0x8087
#define GL_TEXTURE_COORD_ARRAY_SIZE_EXT   0x8088
#define GL_TEXTURE_COORD_ARRAY_TYPE_EXT   0x8089
#define GL_TEXTURE_COORD_ARRAY_STRIDE_EXT 0x808A
#define GL_TEXTURE_COORD_ARRAY_COUNT_EXT  0x808B
#define GL_EDGE_FLAG_ARRAY_STRIDE_EXT     0x808C
#define GL_EDGE_FLAG_ARRAY_COUNT_EXT      0x808D
#define GL_VERTEX_ARRAY_POINTER_EXT       0x808E
#define GL_NORMAL_ARRAY_POINTER_EXT       0x808F
#define GL_COLOR_ARRAY_POINTER_EXT        0x8090
#define GL_INDEX_ARRAY_POINTER_EXT        0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER_EXT 0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER_EXT    0x8093

/* ARB_imaging */
#define GL_CONSTANT_COLOR                 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR       0x8002
#define GL_CONSTANT_ALPHA                 0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA       0x8004
#define GL_BLEND_COLOR                    0x8005
#define GL_FUNC_ADD                       0x8006
#define GL_MIN                            0x8007
#define GL_MAX                            0x8008
#define GL_BLEND_EQUATION                 0x8009
#define GL_FUNC_SUBTRACT                  0x800A
#define GL_FUNC_REVERSE_SUBTRACT          0x800B
#define GL_COLOR_MATRIX                   0x80B1
#define GL_COLOR_MATRIX_STACK_DEPTH       0x80B2
#define GL_MAX_COLOR_MATRIX_STACK_DEPTH   0x80B3
#define GL_POST_COLOR_MATRIX_RED_SCALE    0x80B4
#define GL_POST_COLOR_MATRIX_GREEN_SCALE  0x80B5
#define GL_POST_COLOR_MATRIX_BLUE_SCALE   0x80B6
#define GL_POST_COLOR_MATRIX_ALPHA_SCALE  0x80B7
#define GL_POST_COLOR_MATRIX_RED_BIAS     0x80B8
#define GL_POST_COLOR_MATRIX_GREEN_BIAS   0x80B9
#define GL_POST_COLOR_MATRIX_BLUE_BIAS    0x80BA
#define GL_POST_COLOR_MATRIX_ALPHA_BIAS   0x80BB
#define GL_COLOR_TABLE                    0x80D0
#define GL_POST_CONVOLUTION_COLOR_TABLE   0x80D1
#define GL_POST_COLOR_MATRIX_COLOR_TABLE  0x80D2
#define GL_PROXY_COLOR_TABLE              0x80D3
#define GL_PROXY_POST_CONVOLUTION_COLOR_TABLE 0x80D4
#define GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE 0x80D5
#define GL_COLOR_TABLE_SCALE              0x80D6
#define GL_COLOR_TABLE_BIAS               0x80D7
#define GL_COLOR_TABLE_FORMAT             0x80D8
#define GL_COLOR_TABLE_WIDTH              0x80D9
#define GL_COLOR_TABLE_RED_SIZE           0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE         0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE          0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE         0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE     0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE     0x80DF
#define GL_CONVOLUTION_1D                 0x8010
#define GL_CONVOLUTION_2D                 0x8011
#define GL_SEPARABLE_2D                   0x8012
#define GL_CONVOLUTION_BORDER_MODE        0x8013
#define GL_CONVOLUTION_FILTER_SCALE       0x8014
#define GL_CONVOLUTION_FILTER_BIAS        0x8015
#define GL_REDUCE                         0x8016
#define GL_CONVOLUTION_FORMAT             0x8017
#define GL_CONVOLUTION_WIDTH              0x8018
#define GL_CONVOLUTION_HEIGHT             0x8019
#define GL_MAX_CONVOLUTION_WIDTH          0x801A
#define GL_MAX_CONVOLUTION_HEIGHT         0x801B
#define GL_POST_CONVOLUTION_RED_SCALE     0x801C
#define GL_POST_CONVOLUTION_GREEN_SCALE   0x801D
#define GL_POST_CONVOLUTION_BLUE_SCALE    0x801E
#define GL_POST_CONVOLUTION_ALPHA_SCALE   0x801F
#define GL_POST_CONVOLUTION_RED_BIAS      0x8020
#define GL_POST_CONVOLUTION_GREEN_BIAS    0x8021
#define GL_POST_CONVOLUTION_BLUE_BIAS     0x8022
#define GL_POST_CONVOLUTION_ALPHA_BIAS    0x8023
#define GL_IGNORE_BORDER                  0x8150
#define GL_CONSTANT_BORDER                0x8151
#define GL_REPLICATE_BORDER               0x8153
#define GL_CONVOLUTION_BORDER_COLOR       0x8154
#define GL_HISTOGRAM                      0x8024
#define GL_PROXY_HISTOGRAM                0x8025
#define GL_HISTOGRAM_WIDTH                0x8026
#define GL_HISTOGRAM_FORMAT               0x8027
#define GL_HISTOGRAM_RED_SIZE             0x8028
#define GL_HISTOGRAM_GREEN_SIZE           0x8029
#define GL_HISTOGRAM_BLUE_SIZE            0x802A
#define GL_HISTOGRAM_ALPHA_SIZE           0x802B
#define GL_HISTOGRAM_LUMINANCE_SIZE       0x802C
#define GL_HISTOGRAM_SINK                 0x802D
#define GL_MINMAX                         0x802E
#define GL_MINMAX_FORMAT                  0x802F
#define GL_MINMAX_SINK                    0x8030

/* EXT_clip_volume_hint */
#define GL_CLIP_VOLUME_CLIPPING_HINT_EXT  0x80F0

/* EXT_point_parameters */
#define GL_POINT_SIZE_MIN_EXT             0x8126
#define GL_POINT_SIZE_MAX_EXT             0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE_EXT  0x8128
#define GL_DISTANCE_ATTENUATION_EXT       0x8129

/* EXT_compiled_vertex_array */
#define GL_ARRAY_ELEMENT_LOCK_FIRST_EXT   0x81A8
#define GL_ARRAY_ELEMENT_LOCK_COUNT_EXT   0x81A9

/* SGIS_multitexture */
#define GL_SELECTED_TEXTURE_SGIS          0x835C
#define GL_MAX_TEXTURES_SGIS              0x835D
#define GL_TEXTURE0_SGIS                  0x835E
#define GL_TEXTURE1_SGIS                  0x835F
#define GL_TEXTURE2_SGIS                  0x8360
#define GL_TEXTURE3_SGIS                  0x8361

/* ARB_multitexture */
#define GL_ACTIVE_TEXTURE_ARB             0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE_ARB      0x84E1
#define GL_MAX_TEXTURE_UNITS_ARB          0x84E2
#define GL_TEXTURE0_ARB                   0x84C0
#define GL_TEXTURE1_ARB                   0x84C1
#define GL_TEXTURE2_ARB                   0x84C2
#define GL_TEXTURE3_ARB                   0x84C3
#define GL_TEXTURE4_ARB                   0x84C4
#define GL_TEXTURE5_ARB                   0x84C5
#define GL_TEXTURE6_ARB                   0x84C6
#define GL_TEXTURE7_ARB                   0x84C7
#define GL_TEXTURE8_ARB                   0x84C8
#define GL_TEXTURE9_ARB                   0x84C9
#define GL_TEXTURE10_ARB                  0x84CA
#define GL_TEXTURE11_ARB                  0x84CB
#define GL_TEXTURE12_ARB                  0x84CC
#define GL_TEXTURE13_ARB                  0x84CD
#define GL_TEXTURE14_ARB                  0x84CE
#define GL_TEXTURE15_ARB                  0x84CF
#define GL_TEXTURE16_ARB                  0x84D0
#define GL_TEXTURE17_ARB                  0x84D1
#define GL_TEXTURE18_ARB                  0x84D2
#define GL_TEXTURE19_ARB                  0x84D3
#define GL_TEXTURE20_ARB                  0x84D4
#define GL_TEXTURE21_ARB                  0x84D5
#define GL_TEXTURE22_ARB                  0x84D6
#define GL_TEXTURE23_ARB                  0x84D7
#define GL_TEXTURE24_ARB                  0x84D8
#define GL_TEXTURE25_ARB                  0x84D9
#define GL_TEXTURE26_ARB                  0x84DA
#define GL_TEXTURE27_ARB                  0x84DB
#define GL_TEXTURE28_ARB                  0x84DC
#define GL_TEXTURE29_ARB                  0x84DD
#define GL_TEXTURE30_ARB                  0x84DE
#define GL_TEXTURE31_ARB                  0x84DF

/* EXT_fog_coord */
#define GL_FOG_COORDINATE_SOURCE_EXT      0x8450
#define GL_FOG_COORDINATE_EXT             0x8451
#define GL_FRAGMENT_DEPTH_EXT             0x8452
#define GL_CURRENT_FOG_COORDINATE_EXT     0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE_EXT  0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE_EXT 0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER_EXT 0x8456
#define GL_FOG_COORDINATE_ARRAY_EXT       0x8457

/* EXT_secondary_color */
#define GL_COLOR_SUM_EXT                  0x8458
#define GL_CURRENT_SECONDARY_COLOR_EXT    0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE_EXT 0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE_EXT 0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE_EXT 0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER_EXT 0x845D
#define GL_SECONDARY_COLOR_ARRAY_EXT      0x845E

/* EXT_separate_specular_color */
#define GL_SINGLE_COLOR_EXT               0x81F9
#define GL_SEPARATE_SPECULAR_COLOR_EXT    0x81FA
#define GL_LIGHT_MODEL_COLOR_CONTROL_EXT  0x81F8

/* EXT_stencil_wrap */
#define GL_INCR_WRAP_EXT                  0x8507
#define GL_DECR_WRAP_EXT                  0x8508

/* NV_texgen_reflection */
#define GL_NORMAL_MAP_NV                  0x8511
#define GL_REFLECTION_MAP_NV              0x8512

/* EXT_texture_cube_map */
#define GL_NORMAL_MAP_EXT                 0x8511
#define GL_REFLECTION_MAP_EXT             0x8512
#define GL_TEXTURE_CUBE_MAP_EXT           0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP_EXT   0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP_EXT     0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_EXT  0x851C

/* ARB_texture_cube_map */
#define GL_NORMAL_MAP_ARB                 0x8511
#define GL_REFLECTION_MAP_ARB             0x8512
#define GL_TEXTURE_CUBE_MAP_ARB           0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP_ARB   0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP_ARB     0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB  0x851C

/* NV_vertex_array_range */
#define GL_VERTEX_ARRAY_RANGE_NV          0x851D
#define GL_VERTEX_ARRAY_RANGE_LENGTH_NV   0x851E
#define GL_VERTEX_ARRAY_RANGE_VALID_NV    0x851F
#define GL_MAX_VERTEX_ARRAY_RANGE_ELEMENT_NV 0x8520
#define GL_VERTEX_ARRAY_RANGE_POINTER_NV  0x8521

/* NV_vertex_array_range2 */
#define GL_VERTEX_ARRAY_RANGE_WITHOUT_FLUSH_NV 0x8533

/* NV_register_combiners */
#define GL_REGISTER_COMBINERS_NV          0x8522
#define GL_COMBINER0_NV                   0x8550
#define GL_COMBINER1_NV                   0x8551
#define GL_COMBINER2_NV                   0x8552
#define GL_COMBINER3_NV                   0x8553
#define GL_COMBINER4_NV                   0x8554
#define GL_COMBINER5_NV                   0x8555
#define GL_COMBINER6_NV                   0x8556
#define GL_COMBINER7_NV                   0x8557
#define GL_VARIABLE_A_NV                  0x8523
#define GL_VARIABLE_B_NV                  0x8524
#define GL_VARIABLE_C_NV                  0x8525
#define GL_VARIABLE_D_NV                  0x8526
#define GL_VARIABLE_E_NV                  0x8527
#define GL_VARIABLE_F_NV                  0x8528
#define GL_VARIABLE_G_NV                  0x8529
#define GL_CONSTANT_COLOR0_NV             0x852A
#define GL_CONSTANT_COLOR1_NV             0x852B
#define GL_PRIMARY_COLOR_NV               0x852C
#define GL_SECONDARY_COLOR_NV             0x852D
#define GL_SPARE0_NV                      0x852E
#define GL_SPARE1_NV                      0x852F
/*      GL_TEXTURE0_ARB */
/*      GL_TEXTURE1_ARB */
#define GL_UNSIGNED_IDENTITY_NV           0x8536
#define GL_UNSIGNED_INVERT_NV             0x8537
#define GL_EXPAND_NORMAL_NV               0x8538
#define GL_EXPAND_NEGATE_NV               0x8539
#define GL_HALF_BIAS_NORMAL_NV            0x853A
#define GL_HALF_BIAS_NEGATE_NV            0x853B
#define GL_SIGNED_IDENTITY_NV             0x853C
#define GL_SIGNED_NEGATE_NV               0x853D
#define GL_E_TIMES_F_NV                   0x8531
#define GL_SPARE0_PLUS_SECONDARY_COLOR_NV 0x8532
#define GL_SCALE_BY_TWO_NV                0x853E
#define GL_SCALE_BY_FOUR_NV               0x853F
#define GL_SCALE_BY_ONE_HALF_NV           0x8540
#define GL_BIAS_BY_NEGATIVE_ONE_HALF_NV   0x8541
#define GL_DISCARD_NV                     0x8530
#define GL_COMBINER_INPUT_NV              0x8542
#define GL_COMBINER_MAPPING_NV            0x8543
#define GL_COMBINER_COMPONENT_USAGE_NV    0x8544
#define GL_COMBINER_AB_DOT_PRODUCT_NV     0x8545
#define GL_COMBINER_CD_DOT_PRODUCT_NV     0x8546
#define GL_COMBINER_MUX_SUM_NV            0x8547
#define GL_COMBINER_SCALE_NV              0x8548
#define GL_COMBINER_BIAS_NV               0x8549
#define GL_COMBINER_AB_OUTPUT_NV          0x854A
#define GL_COMBINER_CD_OUTPUT_NV          0x854B
#define GL_COMBINER_SUM_OUTPUT_NV         0x854C
#define GL_MAX_GENERAL_COMBINERS_NV       0x854D
#define GL_NUM_GENERAL_COMBINERS_NV       0x854E
#define GL_COLOR_SUM_CLAMP_NV             0x854F

/* NV_fog_distance */
#define GL_FOG_DISTANCE_MODE_NV           0x855A
#define GL_EYE_RADIAL_NV                  0x855B
#define GL_EYE_PLANE_ABSOLUTE_NV          0x855C

/* NV_fragment_program */
#define GL_FRAGMENT_PROGRAM_NV            0x8870
#define GL_MAX_TEXTURE_COORDS_NV          0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS_NV     0x8872
#define GL_FRAGMENT_PROGRAM_BINDING_NV    0x8873
#define GL_PROGRAM_ERROR_STRING_NV        0x8874
#define GL_MAX_FRAGMENT_PROGRAM_LOCAL_PARAMETERS_NV 0x8868

/* NV_light_max_exponent */
#define GL_MAX_SHININESS_NV               0x8504
#define GL_MAX_SPOT_EXPONENT_NV           0x8505

/* ARB_texture_env_combine */
#define GL_COMBINE_ARB                    0x8570
#define GL_COMBINE_RGB_ARB                0x8571
#define GL_COMBINE_ALPHA_ARB              0x8572
#define GL_RGB_SCALE_ARB                  0x8573
#define GL_ADD_SIGNED_ARB                 0x8574
#define GL_INTERPOLATE_ARB                0x8575
#define GL_CONSTANT_ARB                   0x8576
#define GL_PRIMARY_COLOR_ARB              0x8577
#define GL_PREVIOUS_ARB                   0x8578
#define GL_SOURCE0_RGB_ARB                0x8580
#define GL_SOURCE1_RGB_ARB                0x8581
#define GL_SOURCE2_RGB_ARB                0x8582
#define GL_SOURCE0_ALPHA_ARB              0x8588
#define GL_SOURCE1_ALPHA_ARB              0x8589
#define GL_SOURCE2_ALPHA_ARB              0x858A
#define GL_OPERAND0_RGB_ARB               0x8590
#define GL_OPERAND1_RGB_ARB               0x8591
#define GL_OPERAND2_RGB_ARB               0x8592
#define GL_OPERAND0_ALPHA_ARB             0x8598
#define GL_OPERAND1_ALPHA_ARB             0x8599
#define GL_OPERAND2_ALPHA_ARB             0x859A
#define GL_SUBTRACT_ARB                   0x84E7

/* EXT_texture_env_combine */
#define GL_COMBINE_EXT                    0x8570
#define GL_COMBINE_RGB_EXT                0x8571
#define GL_COMBINE_ALPHA_EXT              0x8572
#define GL_RGB_SCALE_EXT                  0x8573
#define GL_ADD_SIGNED_EXT                 0x8574
#define GL_INTERPOLATE_EXT                0x8575
#define GL_CONSTANT_EXT                   0x8576
#define GL_PRIMARY_COLOR_EXT              0x8577
#define GL_PREVIOUS_EXT                   0x8578
#define GL_SOURCE0_RGB_EXT                0x8580
#define GL_SOURCE1_RGB_EXT                0x8581
#define GL_SOURCE2_RGB_EXT                0x8582
#define GL_SOURCE0_ALPHA_EXT              0x8588
#define GL_SOURCE1_ALPHA_EXT              0x8589
#define GL_SOURCE2_ALPHA_EXT              0x858A
#define GL_OPERAND0_RGB_EXT               0x8590
#define GL_OPERAND1_RGB_EXT               0x8591
#define GL_OPERAND2_RGB_EXT               0x8592
#define GL_OPERAND0_ALPHA_EXT             0x8598
#define GL_OPERAND1_ALPHA_EXT             0x8599
#define GL_OPERAND2_ALPHA_EXT             0x859A

/* NV_texture_env_combine4 */
#define GL_COMBINE4_NV                    0x8503
#define GL_SOURCE3_RGB_NV                 0x8583
#define GL_SOURCE3_ALPHA_NV               0x858B
#define GL_OPERAND3_RGB_NV                0x8593
#define GL_OPERAND3_ALPHA_NV              0x859B

/* SUN_slice_accum */
#define GL_SLICE_ACCUM_SUN                0x85CC

/* EXT_texture_filter_anisotropic */
#define GL_TEXTURE_MAX_ANISOTROPY_EXT     0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF

/* EXT_texture_lod_bias */
#define GL_MAX_TEXTURE_LOD_BIAS_EXT       0x84FD
#define GL_TEXTURE_FILTER_CONTROL_EXT     0x8500
#define GL_TEXTURE_LOD_BIAS_EXT           0x8501

/* EXT_texture_edge_clamp */
#define GL_CLAMP_TO_EDGE_EXT              0x812F

/* S3_s3tc */
#define GL_RGB_S3TC                       0x83A0
#define GL_RGB4_S3TC                      0x83A1
#define GL_RGBA_S3TC                      0x83A2
#define GL_RGBA4_S3TC                     0x83A3
#define GL_RGBA_DXT5_S3TC                 0x83A4
#define GL_RGBA4_DXT5_S3TC                0x83A5

/* ARB_transpose_matrix */
#define GL_TRANSPOSE_MODELVIEW_MATRIX_ARB 0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX_ARB 0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX_ARB   0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX_ARB     0x84E6

/* ARB_texture_compression */
#define GL_COMPRESSED_ALPHA_ARB           0x84E9
#define GL_COMPRESSED_LUMINANCE_ARB       0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA_ARB 0x84EB
#define GL_COMPRESSED_INTENSITY_ARB       0x84EC
#define GL_COMPRESSED_RGB_ARB             0x84ED
#define GL_COMPRESSED_RGBA_ARB            0x84EE
#define GL_TEXTURE_COMPRESSION_HINT_ARB   0x84EF
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE_ARB 0x86A0
#define GL_TEXTURE_COMPRESSED_ARB         0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS_ARB 0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS_ARB 0x86A3

/* EXT_texture_compression_s3tc */
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT   0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT  0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT  0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT  0x83F3

/* NV_fence */
#define GL_ALL_COMPLETED_NV               0x84F2
#define GL_FENCE_STATUS_NV                0x84F3
#define GL_FENCE_CONDITION_NV             0x84F4

/* NV_mac_get_proc_address */
#define GL_ALL_EXTENSIONS_NV              0x84FB
#define GL_MAC_GET_PROC_ADDRESS_NV        0x84FC

/* NV_vertex_program */
#define GL_VERTEX_PROGRAM_NV              0x8620
#define GL_VERTEX_STATE_PROGRAM_NV        0x8621
#define GL_ATTRIB_ARRAY_SIZE_NV           0x8623
#define GL_ATTRIB_ARRAY_STRIDE_NV         0x8624
#define GL_ATTRIB_ARRAY_TYPE_NV           0x8625
#define GL_CURRENT_ATTRIB_NV              0x8626
#define GL_PROGRAM_LENGTH_NV              0x8627
#define GL_PROGRAM_STRING_NV              0x8628
#define GL_MODELVIEW_PROJECTION_NV        0x8629
#define GL_IDENTITY_NV                    0x862A
#define GL_INVERSE_NV                     0x862B
#define GL_TRANSPOSE_NV                   0x862C
#define GL_INVERSE_TRANSPOSE_NV           0x862D
#define GL_MAX_TRACK_MATRIX_STACK_DEPTH_NV 0x862E
#define GL_MAX_TRACK_MATRICES_NV          0x862F
#define GL_MATRIX0_NV                     0x8630
#define GL_MATRIX1_NV                     0x8631
#define GL_MATRIX2_NV                     0x8632
#define GL_MATRIX3_NV                     0x8633
#define GL_MATRIX4_NV                     0x8634
#define GL_MATRIX5_NV                     0x8635
#define GL_MATRIX6_NV                     0x8636
#define GL_MATRIX7_NV                     0x8637
#define GL_CURRENT_MATRIX_STACK_DEPTH_NV  0x8640
#define GL_CURRENT_MATRIX_NV              0x8641
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV   0x8642
#define GL_VERTEX_PROGRAM_TWO_SIDE_NV     0x8643
#define GL_PROGRAM_PARAMETER_NV           0x8644
#define GL_ATTRIB_ARRAY_POINTER_NV        0x8645
#define GL_PROGRAM_TARGET_NV              0x8646
#define GL_PROGRAM_RESIDENT_NV            0x8647
#define GL_TRACK_MATRIX_NV                0x8648
#define GL_TRACK_MATRIX_TRANSFORM_NV      0x8649
#define GL_VERTEX_PROGRAM_BINDING_NV      0x864A
#define GL_PROGRAM_ERROR_POSITION_NV      0x864B
#define GL_VERTEX_ATTRIB_ARRAY0_NV        0x8650
#define GL_VERTEX_ATTRIB_ARRAY1_NV        0x8651
#define GL_VERTEX_ATTRIB_ARRAY2_NV        0x8652
#define GL_VERTEX_ATTRIB_ARRAY3_NV        0x8653
#define GL_VERTEX_ATTRIB_ARRAY4_NV        0x8654
#define GL_VERTEX_ATTRIB_ARRAY5_NV        0x8655
#define GL_VERTEX_ATTRIB_ARRAY6_NV        0x8656
#define GL_VERTEX_ATTRIB_ARRAY7_NV        0x8657
#define GL_VERTEX_ATTRIB_ARRAY8_NV        0x8658
#define GL_VERTEX_ATTRIB_ARRAY9_NV        0x8659
#define GL_VERTEX_ATTRIB_ARRAY10_NV       0x865A
#define GL_VERTEX_ATTRIB_ARRAY11_NV       0x865B
#define GL_VERTEX_ATTRIB_ARRAY12_NV       0x865C
#define GL_VERTEX_ATTRIB_ARRAY13_NV       0x865D
#define GL_VERTEX_ATTRIB_ARRAY14_NV       0x865E
#define GL_VERTEX_ATTRIB_ARRAY15_NV       0x865F
#define GL_MAP1_VERTEX_ATTRIB0_4_NV       0x8660
#define GL_MAP1_VERTEX_ATTRIB1_4_NV       0x8661
#define GL_MAP1_VERTEX_ATTRIB2_4_NV       0x8662
#define GL_MAP1_VERTEX_ATTRIB3_4_NV       0x8663
#define GL_MAP1_VERTEX_ATTRIB4_4_NV       0x8664
#define GL_MAP1_VERTEX_ATTRIB5_4_NV       0x8665
#define GL_MAP1_VERTEX_ATTRIB6_4_NV       0x8666
#define GL_MAP1_VERTEX_ATTRIB7_4_NV       0x8667
#define GL_MAP1_VERTEX_ATTRIB8_4_NV       0x8668
#define GL_MAP1_VERTEX_ATTRIB9_4_NV       0x8669
#define GL_MAP1_VERTEX_ATTRIB10_4_NV      0x866A
#define GL_MAP1_VERTEX_ATTRIB11_4_NV      0x866B
#define GL_MAP1_VERTEX_ATTRIB12_4_NV      0x866C
#define GL_MAP1_VERTEX_ATTRIB13_4_NV      0x866D
#define GL_MAP1_VERTEX_ATTRIB14_4_NV      0x866E
#define GL_MAP1_VERTEX_ATTRIB15_4_NV      0x866F
#define GL_MAP2_VERTEX_ATTRIB0_4_NV       0x8670
#define GL_MAP2_VERTEX_ATTRIB1_4_NV       0x8671
#define GL_MAP2_VERTEX_ATTRIB2_4_NV       0x8672
#define GL_MAP2_VERTEX_ATTRIB3_4_NV       0x8673
#define GL_MAP2_VERTEX_ATTRIB4_4_NV       0x8674
#define GL_MAP2_VERTEX_ATTRIB5_4_NV       0x8675
#define GL_MAP2_VERTEX_ATTRIB6_4_NV       0x8676
#define GL_MAP2_VERTEX_ATTRIB7_4_NV       0x8677
#define GL_MAP2_VERTEX_ATTRIB8_4_NV       0x8678
#define GL_MAP2_VERTEX_ATTRIB9_4_NV       0x8679
#define GL_MAP2_VERTEX_ATTRIB10_4_NV      0x867A
#define GL_MAP2_VERTEX_ATTRIB11_4_NV      0x867B
#define GL_MAP2_VERTEX_ATTRIB12_4_NV      0x867C
#define GL_MAP2_VERTEX_ATTRIB13_4_NV      0x867D
#define GL_MAP2_VERTEX_ATTRIB14_4_NV      0x867E
#define GL_MAP2_VERTEX_ATTRIB15_4_NV      0x867F

/* NV_texture_shader */
#define GL_OFFSET_TEXTURE_RECTANGLE_NV    0x864C
#define GL_OFFSET_TEXTURE_RECTANGLE_SCALE_NV 0x864D
#define GL_DOT_PRODUCT_TEXTURE_RECTANGLE_NV 0x864E
#define GL_RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV 0x86D9
#define GL_UNSIGNED_INT_S8_S8_8_8_NV      0x86DA
#define GL_UNSIGNED_INT_8_8_S8_S8_REV_NV  0x86DB
#define GL_DSDT_MAG_INTENSITY_NV          0x86DC
#define GL_SHADER_CONSISTENT_NV           0x86DD
#define GL_TEXTURE_SHADER_NV              0x86DE
#define GL_SHADER_OPERATION_NV            0x86DF
#define GL_CULL_MODES_NV                  0x86E0
#define GL_OFFSET_TEXTURE_MATRIX_NV       0x86E1
#define GL_OFFSET_TEXTURE_SCALE_NV        0x86E2
#define GL_OFFSET_TEXTURE_BIAS_NV         0x86E3
#define GL_OFFSET_TEXTURE_2D_MATRIX_NV    GL_OFFSET_TEXTURE_MATRIX_NV
#define GL_OFFSET_TEXTURE_2D_SCALE_NV     GL_OFFSET_TEXTURE_SCALE_NV
#define GL_OFFSET_TEXTURE_2D_BIAS_NV      GL_OFFSET_TEXTURE_BIAS_NV
#define GL_PREVIOUS_TEXTURE_INPUT_NV      0x86E4
#define GL_CONST_EYE_NV                   0x86E5
#define GL_PASS_THROUGH_NV                0x86E6
#define GL_CULL_FRAGMENT_NV               0x86E7
#define GL_OFFSET_TEXTURE_2D_NV           0x86E8
#define GL_DEPENDENT_AR_TEXTURE_2D_NV     0x86E9
#define GL_DEPENDENT_GB_TEXTURE_2D_NV     0x86EA
#define GL_DOT_PRODUCT_NV                 0x86EC
#define GL_DOT_PRODUCT_DEPTH_REPLACE_NV   0x86ED
#define GL_DOT_PRODUCT_TEXTURE_2D_NV      0x86EE
#define GL_DOT_PRODUCT_TEXTURE_CUBE_MAP_NV 0x86F0
#define GL_DOT_PRODUCT_DIFFUSE_CUBE_MAP_NV 0x86F1
#define GL_DOT_PRODUCT_REFLECT_CUBE_MAP_NV 0x86F2
#define GL_DOT_PRODUCT_CONST_EYE_REFLECT_CUBE_MAP_NV 0x86F3
#define GL_HILO_NV                        0x86F4
#define GL_DSDT_NV                        0x86F5
#define GL_DSDT_MAG_NV                    0x86F6
#define GL_DSDT_MAG_VIB_NV                0x86F7
#define GL_HILO16_NV                      0x86F8
#define GL_SIGNED_HILO_NV                 0x86F9
#define GL_SIGNED_HILO16_NV               0x86FA
#define GL_SIGNED_RGBA_NV                 0x86FB
#define GL_SIGNED_RGBA8_NV                0x86FC
#define GL_SIGNED_RGB_NV                  0x86FE
#define GL_SIGNED_RGB8_NV                 0x86FF
#define GL_SIGNED_LUMINANCE_NV            0x8701
#define GL_SIGNED_LUMINANCE8_NV           0x8702
#define GL_SIGNED_LUMINANCE_ALPHA_NV      0x8703
#define GL_SIGNED_LUMINANCE8_ALPHA8_NV    0x8704
#define GL_SIGNED_ALPHA_NV                0x8705
#define GL_SIGNED_ALPHA8_NV               0x8706
#define GL_SIGNED_INTENSITY_NV            0x8707
#define GL_SIGNED_INTENSITY8_NV           0x8708
#define GL_DSDT8_NV                       0x8709
#define GL_DSDT8_MAG8_NV                  0x870A
#define GL_DSDT8_MAG8_INTENSITY8_NV       0x870B
#define GL_SIGNED_RGB_UNSIGNED_ALPHA_NV   0x870C
#define GL_SIGNED_RGB8_UNSIGNED_ALPHA8_NV 0x870D
#define GL_HI_SCALE_NV                    0x870E
#define GL_LO_SCALE_NV                    0x870F
#define GL_DS_SCALE_NV                    0x8710
#define GL_DT_SCALE_NV                    0x8711
#define GL_MAGNITUDE_SCALE_NV             0x8712
#define GL_VIBRANCE_SCALE_NV              0x8713
#define GL_HI_BIAS_NV                     0x8714
#define GL_LO_BIAS_NV                     0x8715
#define GL_DS_BIAS_NV                     0x8716
#define GL_DT_BIAS_NV                     0x8717
#define GL_MAGNITUDE_BIAS_NV              0x8718
#define GL_VIBRANCE_BIAS_NV               0x8719
#define GL_TEXTURE_BORDER_VALUES_NV       0x871A
#define GL_TEXTURE_HI_SIZE_NV             0x871B
#define GL_TEXTURE_LO_SIZE_NV             0x871C
#define GL_TEXTURE_DS_SIZE_NV             0x871D
#define GL_TEXTURE_DT_SIZE_NV             0x871E
#define GL_TEXTURE_MAG_SIZE_NV            0x871F

/* NV_texture_shader2 */
#define GL_DOT_PRODUCT_TEXTURE_3D_NV      0x86EF

/* NV_texture_shader3 */
#define GL_OFFSET_PROJECTIVE_TEXTURE_2D_NV 0x8850
#define GL_OFFSET_PROJECTIVE_TEXTURE_2D_SCALE_NV 0x8851
#define GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_NV 0x8852
#define GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_SCALE_NV 0x8853
#define GL_OFFSET_HILO_TEXTURE_2D_NV      0x8854
#define GL_OFFSET_HILO_TEXTURE_RECTANGLE_NV 0x8855
#define GL_OFFSET_HILO_PROJECTIVE_TEXTURE_2D_NV 0x8856
#define GL_OFFSET_HILO_PROJECTIVE_TEXTURE_RECTANGLE_NV 0x8857
#define GL_DEPENDENT_HILO_TEXTURE_2D_NV   0x8858
#define GL_DEPENDENT_RGB_TEXTURE_3D_NV    0x8859
#define GL_DEPENDENT_RGB_TEXTURE_CUBE_MAP_NV 0x885A
#define GL_DOT_PRODUCT_PASS_THROUGH_NV    0x885B
#define GL_DOT_PRODUCT_TEXTURE_1D_NV      0x885C
#define GL_DOT_PRODUCT_AFFINE_DEPTH_REPLACE_NV 0x885D
#define GL_HILO8_NV                       0x885E
#define GL_SIGNED_HILO8_NV                0x885F
#define GL_FORCE_BLUE_TO_ONE_NV           0x8860

/* NV_register_combiners2 */
#define GL_PER_STAGE_CONSTANTS_NV         0x8535

/* IBM_texture_mirrored_repeat */
#define GL_MIRRORED_REPEAT_IBM            0x8370

/* ARB_texture_env_dot3 */
#define GL_DOT3_RGB_ARB                   0x86AE
#define GL_DOT3_RGBA_ARB                  0x86AF

/* EXT_texture_env_dot3 */
#define GL_DOT3_RGB_EXT                   0x8740
#define GL_DOT3_RGBA_EXT                  0x8741

/* APPLE_transform_hint */
#define GL_TRANSFORM_HINT_APPLE           0x85B1

/* ARB_texture_border_clamp */
#define GL_CLAMP_TO_BORDER_ARB            0x812D

/* NV_texture_rectangle */
#define GL_TEXTURE_RECTANGLE_NV           0x84F5
#define GL_TEXTURE_BINDING_RECTANGLE_NV   0x84F6
#define GL_PROXY_TEXTURE_RECTANGLE_NV     0x84F7
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_NV  0x84F8

/* ARB_texture_rectangle */
#define GL_TEXTURE_RECTANGLE_ARB          0x84F5
#define GL_TEXTURE_BINDING_RECTANGLE_ARB  0x84F6
#define GL_PROXY_TEXTURE_RECTANGLE_ARB    0x84F7
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB 0x84F8

/* ARB_multisample */
#define GL_MULTISAMPLE_ARB                0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE_ARB   0x809E
#define GL_SAMPLE_ALPHA_TO_ONE_ARB        0x809F
#define GL_SAMPLE_COVERAGE_ARB            0x80A0
#define GL_SAMPLE_BUFFERS_ARB             0x80A8
#define GL_SAMPLES_ARB                    0x80A9
#define GL_SAMPLE_COVERAGE_VALUE_ARB      0x80AA
#define GL_SAMPLE_COVERAGE_INVERT_ARB     0x80AB
#define GL_MULTISAMPLE_BIT_ARB            0x20000000

/* NV_multisample_filter_hint */
#define GL_MULTISAMPLE_FILTER_HINT_NV     0x8534

/* NV_packed_depth_stencil */
#define GL_DEPTH_STENCIL_NV               0x84F9
#define GL_UNSIGNED_INT_24_8_NV           0x84FA

/* EXT_draw_range_elements */
#define GL_MAX_ELEMENTS_VERTICES_EXT      0x80E8
#define GL_MAX_ELEMENTS_INDICES_EXT       0x80E9

/* NV_pixel_data_range */
#define GL_WRITE_PIXEL_DATA_RANGE_NV      0x8878
#define GL_READ_PIXEL_DATA_RANGE_NV       0x8879
#define GL_WRITE_PIXEL_DATA_RANGE_LENGTH_NV 0x887A
#define GL_READ_PIXEL_DATA_RANGE_LENGTH_NV 0x887B
#define GL_WRITE_PIXEL_DATA_RANGE_POINTER_NV 0x887C
#define GL_READ_PIXEL_DATA_RANGE_POINTER_NV 0x887D

/* NV_packed_normal */
#define GL_UNSIGNED_INT_S10_S11_S11_REV_NV 0x886B

/* NV_half_float */
#define GL_HALF_FLOAT_NV                  0x140B

/* NV_copy_depth_to_color */
#define GL_DEPTH_STENCIL_TO_RGBA_NV       0x886E
#define GL_DEPTH_STENCIL_TO_BGRA_NV       0x886F

/* HP_occlusion_test */
#define GL_OCCLUSION_TEST_HP              0x8165
#define GL_OCCLUSION_TEST_RESULT_HP       0x8166

/* NV_occlusion_query */
#define GL_PIXEL_COUNTER_BITS_NV          0x8864
#define GL_CURRENT_OCCLUSION_QUERY_ID_NV  0x8865
#define GL_PIXEL_COUNT_NV                 0x8866
#define GL_PIXEL_COUNT_AVAILABLE_NV       0x8867

/* ARB_occlusion_query */
#define GL_QUERY_COUNTER_BITS_ARB         0x8864
#define GL_CURRENT_QUERY_ARB              0x8865
#define GL_QUERY_RESULT_ARB               0x8866
#define GL_QUERY_RESULT_AVAILABLE_ARB     0x8867
#define GL_SAMPLES_PASSED_ARB             0x8914

/* ARB_point_sprite */
#define GL_POINT_SPRITE_ARB               0x8861
#define GL_COORD_REPLACE_ARB              0x8862

/* NV_point_sprite */
#define GL_POINT_SPRITE_NV                0x8861
#define GL_COORD_REPLACE_NV               0x8862
#define GL_POINT_SPRITE_R_MODE_NV         0x8863

/* 3DFX_tbuffer */
#define GL_TBUFFER_WRITE_MASK_3DFX        0x86D8

/* NV_depth_clamp */
#define GL_DEPTH_CLAMP_NV                 0x864F

/* NV_float_buffer */
#define GL_FLOAT_R_NV                     0x8880
#define GL_FLOAT_RG_NV                    0x8881
#define GL_FLOAT_RGB_NV                   0x8882
#define GL_FLOAT_RGBA_NV                  0x8883
#define GL_FLOAT_R16_NV                   0x8884
#define GL_FLOAT_R32_NV                   0x8885
#define GL_FLOAT_RG16_NV                  0x8886
#define GL_FLOAT_RG32_NV                  0x8887
#define GL_FLOAT_RGB16_NV                 0x8888
#define GL_FLOAT_RGB32_NV                 0x8889
#define GL_FLOAT_RGBA16_NV                0x888A
#define GL_FLOAT_RGBA32_NV                0x888B
#define GL_TEXTURE_FLOAT_COMPONENTS_NV    0x888C
#define GL_FLOAT_CLEAR_COLOR_VALUE_NV     0x888D
#define GL_FLOAT_RGBA_MODE_NV             0x888E

/* EXT_stencil_two_side */
#define GL_STENCIL_TEST_TWO_SIDE_EXT      0x8910
#define GL_ACTIVE_STENCIL_FACE_EXT        0x8911

/* EXT_blend_func_separate */
#define GL_BLEND_DST_RGB_EXT              0x80C8
#define GL_BLEND_SRC_RGB_EXT              0x80C9
#define GL_BLEND_DST_ALPHA_EXT            0x80CA
#define GL_BLEND_SRC_ALPHA_EXT            0x80CB

/* ARB_texture_mirrored_repeat */
#define GL_MIRRORED_REPEAT_ARB            0x8370

/* ARB_depth_texture */
#define GL_DEPTH_COMPONENT16_ARB          0x81A5
#define GL_DEPTH_COMPONENT24_ARB          0x81A6
#define GL_DEPTH_COMPONENT32_ARB          0x81A7
#define GL_TEXTURE_DEPTH_SIZE_ARB         0x884A
#define GL_DEPTH_TEXTURE_MODE_ARB         0x884B

/* ARB_shadow */
#define GL_TEXTURE_COMPARE_MODE_ARB       0x884C
#define GL_TEXTURE_COMPARE_FUNC_ARB       0x884D
#define GL_COMPARE_R_TO_TEXTURE_ARB       0x884E

/* ARB_shadow_ambient */
#define GL_TEXTURE_COMPARE_FAIL_VALUE_ARB 0x80BF

/* NV_force_software */
#define GL_FORCE_SOFTWARE_NV              0x6007

/* ARB_point_parameters */
#define GL_POINT_SIZE_MIN_ARB             0x8126
#define GL_POINT_SIZE_MAX_ARB             0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE_ARB  0x8128
#define GL_POINT_DISTANCE_ATTENUATION_ARB 0x8129

/* EXT_depth_bounds_test */
#define GL_DEPTH_BOUNDS_TEST_EXT          0x8890
#define GL_DEPTH_BOUNDS_EXT               0x8891

/* ARB_vertex_program */
#define GL_VERTEX_PROGRAM_ARB             0x8620
#define GL_VERTEX_PROGRAM_POINT_SIZE_ARB  0x8642
#define GL_VERTEX_PROGRAM_TWO_SIDE_ARB    0x8643
#define GL_COLOR_SUM_ARB                  0x8458
#define GL_PROGRAM_FORMAT_ASCII_ARB       0x8875
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED_ARB 0x8622
#define GL_VERTEX_ATTRIB_ARRAY_SIZE_ARB   0x8623
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE_ARB 0x8624
#define GL_VERTEX_ATTRIB_ARRAY_TYPE_ARB   0x8625
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED_ARB 0x886A
#define GL_CURRENT_VERTEX_ATTRIB_ARB      0x8626
#define GL_VERTEX_ATTRIB_ARRAY_POINTER_ARB 0x8645
#define GL_PROGRAM_LENGTH_ARB             0x8627
#define GL_PROGRAM_FORMAT_ARB             0x8876
#define GL_PROGRAM_BINDING_ARB            0x8677
#define GL_PROGRAM_INSTRUCTIONS_ARB       0x88A0
#define GL_MAX_PROGRAM_INSTRUCTIONS_ARB   0x88A1
#define GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB 0x88A2
#define GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB 0x88A3
#define GL_PROGRAM_TEMPORARIES_ARB        0x88A4
#define GL_MAX_PROGRAM_TEMPORARIES_ARB    0x88A5
#define GL_PROGRAM_NATIVE_TEMPORARIES_ARB 0x88A6
#define GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB 0x88A7
#define GL_PROGRAM_PARAMETERS_ARB         0x88A8
#define GL_MAX_PROGRAM_PARAMETERS_ARB     0x88A9
#define GL_PROGRAM_NATIVE_PARAMETERS_ARB  0x88AA
#define GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB 0x88AB
#define GL_PROGRAM_ATTRIBS_ARB            0x88AC
#define GL_MAX_PROGRAM_ATTRIBS_ARB        0x88AD
#define GL_PROGRAM_NATIVE_ATTRIBS_ARB     0x88AE
#define GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB 0x88AF
#define GL_PROGRAM_ADDRESS_REGISTERS_ARB  0x88B0
#define GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB 0x88B1
#define GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB 0x88B2
#define GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB 0x88B3
#define GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB 0x88B4
#define GL_MAX_PROGRAM_ENV_PARAMETERS_ARB 0x88B5
#define GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB 0x88B6
#define GL_PROGRAM_STRING_ARB             0x8628
#define GL_PROGRAM_ERROR_POSITION_ARB     0x864B
#define GL_CURRENT_MATRIX_ARB             0x8641
#define GL_TRANSPOSE_CURRENT_MATRIX_ARB   0x88B7
#define GL_CURRENT_MATRIX_STACK_DEPTH_ARB 0x8640
#define GL_MAX_VERTEX_ATTRIBS_ARB         0x8869
#define GL_MAX_PROGRAM_MATRICES_ARB       0x862F
#define GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB 0x862E
#define GL_PROGRAM_ERROR_STRING_ARB       0x8874
#define GL_MATRIX0_ARB                    0x88C0
#define GL_MATRIX1_ARB                    0x88C1
#define GL_MATRIX2_ARB                    0x88C2
#define GL_MATRIX3_ARB                    0x88C3
#define GL_MATRIX4_ARB                    0x88C4
#define GL_MATRIX5_ARB                    0x88C5
#define GL_MATRIX6_ARB                    0x88C6
#define GL_MATRIX7_ARB                    0x88C7
#define GL_MATRIX8_ARB                    0x88C8
#define GL_MATRIX9_ARB                    0x88C9
#define GL_MATRIX10_ARB                   0x88CA
#define GL_MATRIX11_ARB                   0x88CB
#define GL_MATRIX12_ARB                   0x88CC
#define GL_MATRIX13_ARB                   0x88CD
#define GL_MATRIX14_ARB                   0x88CE
#define GL_MATRIX15_ARB                   0x88CF
#define GL_MATRIX16_ARB                   0x88D0
#define GL_MATRIX17_ARB                   0x88D1
#define GL_MATRIX18_ARB                   0x88D2
#define GL_MATRIX19_ARB                   0x88D3
#define GL_MATRIX20_ARB                   0x88D4
#define GL_MATRIX21_ARB                   0x88D5
#define GL_MATRIX22_ARB                   0x88D6
#define GL_MATRIX23_ARB                   0x88D7
#define GL_MATRIX24_ARB                   0x88D8
#define GL_MATRIX25_ARB                   0x88D9
#define GL_MATRIX26_ARB                   0x88DA
#define GL_MATRIX27_ARB                   0x88DB
#define GL_MATRIX28_ARB                   0x88DC
#define GL_MATRIX29_ARB                   0x88DD
#define GL_MATRIX30_ARB                   0x88DE
#define GL_MATRIX31_ARB                   0x88DF

/* OpenGL14 */
#define GL_POINT_SIZE_MIN                 0x8126
#define GL_POINT_SIZE_MAX                 0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE      0x8128
#define GL_POINT_DISTANCE_ATTENUATION     0x8129
#define GL_FOG_COORDINATE_SOURCE          0x8450
#define GL_FOG_COORDINATE                 0x8451
#define GL_FRAGMENT_DEPTH                 0x8452
#define GL_CURRENT_FOG_COORDINATE         0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE      0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE    0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER   0x8456
#define GL_FOG_COORDINATE_ARRAY           0x8457
#define GL_COLOR_SUM                      0x8458
#define GL_CURRENT_SECONDARY_COLOR        0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE     0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE     0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE   0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER  0x845D
#define GL_SECONDARY_COLOR_ARRAY          0x845E
#define GL_INCR_WRAP                      0x8507
#define GL_DECR_WRAP                      0x8508
#define GL_MAX_TEXTURE_LOD_BIAS           0x84FD
#define GL_TEXTURE_FILTER_CONTROL         0x8500
#define GL_TEXTURE_LOD_BIAS               0x8501
#define GL_GENERATE_MIPMAP_SGIS           0x8191
#define GL_GENERATE_MIPMAP_HINT_SGIS      0x8192
#define GL_BLEND_DST_RGB                  0x80C8
#define GL_BLEND_SRC_RGB                  0x80C9
#define GL_BLEND_DST_ALPHA                0x80CA
#define GL_BLEND_SRC_ALPHA                0x80CB
#define GL_MIRRORED_REPEAT                0x8370
#define GL_DEPTH_COMPONENT16              0x81A5
#define GL_DEPTH_COMPONENT24              0x81A6
#define GL_DEPTH_COMPONENT32              0x81A7
#define GL_TEXTURE_DEPTH_SIZE             0x884A
#define GL_DEPTH_TEXTURE_MODE             0x884B
#define GL_TEXTURE_COMPARE_MODE           0x884C
#define GL_TEXTURE_COMPARE_FUNC           0x884D
#define GL_COMPARE_R_TO_TEXTURE           0x884E

/* NV_primitive_restart */
#define GL_PRIMITIVE_RESTART_NV           0x8558
#define GL_PRIMITIVE_RESTART_INDEX_NV     0x8559

/* SGIS_texture_color_mask */
#define GL_TEXTURE_COLOR_WRITEMASK_SGIS   0x81EF

/* NV_texture_expand_normal */
#define GL_TEXTURE_UNSIGNED_REMAP_MODE_NV 0x888F

/* ARB_fragment_program */
#define GL_FRAGMENT_PROGRAM_ARB           0x8804
/*      GL_PROGRAM_FORMAT_ASCII_ARB */
/*      GL_PROGRAM_LENGTH_ARB */
/*      GL_PROGRAM_FORMAT_ARB */
/*      GL_PROGRAM_BINDING_ARB */
/*      GL_PROGRAM_INSTRUCTIONS_ARB */
/*      GL_MAX_PROGRAM_INSTRUCTIONS_ARB */
/*      GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB */
/*      GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB */
/*      GL_PROGRAM_TEMPORARIES_ARB */
/*      GL_MAX_PROGRAM_TEMPORARIES_ARB */
/*      GL_PROGRAM_NATIVE_TEMPORARIES_ARB */
/*      GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB */
/*      GL_PROGRAM_PARAMETERS_ARB */
/*      GL_MAX_PROGRAM_PARAMETERS_ARB */
/*      GL_PROGRAM_NATIVE_PARAMETERS_ARB */
/*      GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB */
/*      GL_PROGRAM_ATTRIBS_ARB */
/*      GL_MAX_PROGRAM_ATTRIBS_ARB */
/*      GL_PROGRAM_NATIVE_ATTRIBS_ARB */
/*      GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB */
/*      GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB */
/*      GL_MAX_PROGRAM_ENV_PARAMETERS_ARB */
/*      GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB */
#define GL_PROGRAM_ALU_INSTRUCTIONS_ARB   0x8805
#define GL_PROGRAM_TEX_INSTRUCTIONS_ARB   0x8806
#define GL_PROGRAM_TEX_INDIRECTIONS_ARB   0x8807
#define GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB 0x8808
#define GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB 0x8809
#define GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB 0x880A
#define GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB 0x880B
#define GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB 0x880C
#define GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB 0x880D
#define GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB 0x880E
#define GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB 0x880F
#define GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB 0x8810
/*      GL_PROGRAM_STRING_ARB */
/*      GL_PROGRAM_ERROR_POSITION_ARB */
/*      GL_CURRENT_MATRIX_ARB */
/*      GL_TRANSPOSE_CURRENT_MATRIX_ARB */
/*      GL_CURRENT_MATRIX_STACK_DEPTH_ARB */
/*      GL_MAX_PROGRAM_MATRICES_ARB */
/*      GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB */
#define GL_MAX_TEXTURE_COORDS_ARB         0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS_ARB    0x8872
/*      GL_PROGRAM_ERROR_STRING_ARB */
/*      GL_MATRIX0_ARB */
/*      GL_MATRIX1_ARB */
/*      GL_MATRIX2_ARB */
/*      GL_MATRIX3_ARB */
/*      GL_MATRIX4_ARB */
/*      GL_MATRIX5_ARB */
/*      GL_MATRIX6_ARB */
/*      GL_MATRIX7_ARB */
/*      GL_MATRIX8_ARB */
/*      GL_MATRIX9_ARB */
/*      GL_MATRIX10_ARB */
/*      GL_MATRIX11_ARB */
/*      GL_MATRIX12_ARB */
/*      GL_MATRIX13_ARB */
/*      GL_MATRIX14_ARB */
/*      GL_MATRIX15_ARB */
/*      GL_MATRIX16_ARB */
/*      GL_MATRIX17_ARB */
/*      GL_MATRIX18_ARB */
/*      GL_MATRIX19_ARB */
/*      GL_MATRIX20_ARB */
/*      GL_MATRIX21_ARB */
/*      GL_MATRIX22_ARB */
/*      GL_MATRIX23_ARB */
/*      GL_MATRIX24_ARB */
/*      GL_MATRIX25_ARB */
/*      GL_MATRIX26_ARB */
/*      GL_MATRIX27_ARB */
/*      GL_MATRIX28_ARB */
/*      GL_MATRIX29_ARB */
/*      GL_MATRIX30_ARB */
/*      GL_MATRIX31_ARB */
/*      GL_PROGRAM_ERROR_STRING_ARB */
/*      GL_MATRIX0_ARB */
/*      GL_MATRIX1_ARB */
/*      GL_MATRIX2_ARB */
/*      GL_MATRIX3_ARB */
/*      GL_MATRIX4_ARB */
/*      GL_MATRIX5_ARB */
/*      GL_MATRIX6_ARB */
/*      GL_MATRIX7_ARB */
/*      GL_MATRIX8_ARB */
/*      GL_MATRIX9_ARB */
/*      GL_MATRIX10_ARB */
/*      GL_MATRIX11_ARB */
/*      GL_MATRIX12_ARB */
/*      GL_MATRIX13_ARB */
/*      GL_MATRIX14_ARB */
/*      GL_MATRIX15_ARB */
/*      GL_MATRIX16_ARB */
/*      GL_MATRIX17_ARB */
/*      GL_MATRIX18_ARB */
/*      GL_MATRIX19_ARB */
/*      GL_MATRIX20_ARB */
/*      GL_MATRIX21_ARB */
/*      GL_MATRIX22_ARB */
/*      GL_MATRIX23_ARB */
/*      GL_MATRIX24_ARB */
/*      GL_MATRIX25_ARB */
/*      GL_MATRIX26_ARB */
/*      GL_MATRIX27_ARB */
/*      GL_MATRIX28_ARB */
/*      GL_MATRIX29_ARB */
/*      GL_MATRIX30_ARB */
/*      GL_MATRIX31_ARB */

/* ARB_vertex_buffer_object */
#define GL_ARRAY_BUFFER_ARB               0x8892
#define GL_ELEMENT_ARRAY_BUFFER_ARB       0x8893
#define GL_ARRAY_BUFFER_BINDING_ARB       0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING_ARB 0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING_ARB 0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING_ARB 0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING_ARB 0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING_ARB 0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING_ARB 0x889A
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING_ARB 0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING_ARB 0x889C
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING_ARB 0x889D
#define GL_WEIGHT_ARRAY_BUFFER_BINDING_ARB 0x889E
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING_ARB 0x889F
#define GL_STREAM_DRAW_ARB                0x88E0
#define GL_STREAM_READ_ARB                0x88E1
#define GL_STREAM_COPY_ARB                0x88E2
#define GL_STATIC_DRAW_ARB                0x88E4
#define GL_STATIC_READ_ARB                0x88E5
#define GL_STATIC_COPY_ARB                0x88E6
#define GL_DYNAMIC_DRAW_ARB               0x88E8
#define GL_DYNAMIC_READ_ARB               0x88E9
#define GL_DYNAMIC_COPY_ARB               0x88EA
#define GL_READ_ONLY_ARB                  0x88B8
#define GL_WRITE_ONLY_ARB                 0x88B9
#define GL_READ_WRITE_ARB                 0x88BA
#define GL_BUFFER_SIZE_ARB                0x8764
#define GL_BUFFER_USAGE_ARB               0x8765
#define GL_BUFFER_ACCESS_ARB              0x88BB
#define GL_BUFFER_MAPPED_ARB              0x88BC
#define GL_BUFFER_MAP_POINTER_ARB         0x88BD

/* EXT_pixel_buffer_object */
#define GL_PIXEL_PACK_BUFFER_EXT          0x88EB
#define GL_PIXEL_UNPACK_BUFFER_EXT        0x88EC
#define GL_PIXEL_PACK_BUFFER_BINDING_EXT  0x88ED
#define GL_PIXEL_UNPACK_BUFFER_BINDING_EXT 0x88EF

/* NVX_hrsd_pixels */
#define GL_HRSD_PIXELS_NVX                0x6400
#define GL_HRSD_SCALE_NVX                 0x6401

/* IBM_rasterpos_clip */
#define GL_RASTER_POSITION_UNCLIPPED_IBM  0x19262

/* ATI_texture_mirror_once */
#define GL_MIRROR_CLAMP_ATI               0x8742
#define GL_MIRROR_CLAMP_TO_EDGE_ATI       0x8743

/* ATI_texture_float */
#define GL_RGBA_FLOAT32_ATI               0x8814
#define GL_RGB_FLOAT32_ATI                0x8815
#define GL_ALPHA_FLOAT32_ATI              0x8816
#define GL_INTENSITY_FLOAT32_ATI          0x8817
#define GL_LUMINANCE_FLOAT32_ATI          0x8818
#define GL_LUMINANCE_ALPHA_FLOAT32_ATI    0x8819
#define GL_RGBA_FLOAT16_ATI               0x881A
#define GL_RGB_FLOAT16_ATI                0x881B
#define GL_ALPHA_FLOAT16_ATI              0x881C
#define GL_INTENSITY_FLOAT16_ATI          0x881D
#define GL_LUMINANCE_FLOAT16_ATI          0x881E
#define GL_LUMINANCE_ALPHA_FLOAT16_ATI    0x881F

/* ATI_pixel_format_float */
#define GL_RGBA_FLOAT_MODE_ATI            0x8820
#define GL_COLOR_CLEAR_UNCLAMPED_VALUE_ATI 0x8835

/* ATI_draw_buffers */
#define GL_MAX_DRAW_BUFFERS_ATI           0x8824
#define GL_DRAW_BUFFER0_ATI               0x8825
#define GL_DRAW_BUFFER1_ATI               0x8826
#define GL_DRAW_BUFFER2_ATI               0x8827
#define GL_DRAW_BUFFER3_ATI               0x8828
#define GL_DRAW_BUFFER4_ATI               0x8829
#define GL_DRAW_BUFFER5_ATI               0x882A
#define GL_DRAW_BUFFER6_ATI               0x882B
#define GL_DRAW_BUFFER7_ATI               0x882C
#define GL_DRAW_BUFFER8_ATI               0x882D
#define GL_DRAW_BUFFER9_ATI               0x882E
#define GL_DRAW_BUFFER10_ATI              0x882F
#define GL_DRAW_BUFFER11_ATI              0x8830
#define GL_DRAW_BUFFER12_ATI              0x8831
#define GL_DRAW_BUFFER13_ATI              0x8832
#define GL_DRAW_BUFFER14_ATI              0x8833
#define GL_DRAW_BUFFER15_ATI              0x8834

/* EXT_texture_mirror_clamp */
#define GL_MIRROR_CLAMP_EXT               0x8742
#define GL_MIRROR_CLAMP_TO_EDGE_EXT       0x8743
#define GL_MIRROR_CLAMP_TO_BORDER_EXT     0x8912

/* EXT_blend_equation_separate */
#define GL_BLEND_EQUATION_RGB_EXT         0x8009
#define GL_BLEND_EQUATION_ALPHA_EXT       0x883D

/* NV_centroid_sample */
#define GL_TEXTURE_COORD_CONTROL_NV       0x891A
#define GL_CENTROID_SAMPLE_NV             0x891B

/* ARB_shader_objects */
#define GL_PROGRAM_OBJECT_ARB             0x8B40
#define GL_SHADER_OBJECT_ARB              0x8B48
#define GL_OBJECT_TYPE_ARB                0x8B4E
#define GL_OBJECT_SUBTYPE_ARB             0x8B4F
#define GL_OBJECT_DELETE_STATUS_ARB       0x8B80
#define GL_OBJECT_COMPILE_STATUS_ARB      0x8B81
#define GL_OBJECT_LINK_STATUS_ARB         0x8B82
#define GL_OBJECT_VALIDATE_STATUS_ARB     0x8B83
#define GL_OBJECT_INFO_LOG_LENGTH_ARB     0x8B84
#define GL_OBJECT_ATTACHED_OBJECTS_ARB    0x8B85
#define GL_OBJECT_ACTIVE_UNIFORMS_ARB     0x8B86
#define GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB 0x8B87
#define GL_OBJECT_SHADER_SOURCE_LENGTH_ARB 0x8B88
#define GL_FLOAT_VEC2_ARB                 0x8B50
#define GL_FLOAT_VEC3_ARB                 0x8B51
#define GL_FLOAT_VEC4_ARB                 0x8B52
#define GL_INT_VEC2_ARB                   0x8B53
#define GL_INT_VEC3_ARB                   0x8B54
#define GL_INT_VEC4_ARB                   0x8B55
#define GL_BOOL_ARB                       0x8B56
#define GL_BOOL_VEC2_ARB                  0x8B57
#define GL_BOOL_VEC3_ARB                  0x8B58
#define GL_BOOL_VEC4_ARB                  0x8B59
#define GL_FLOAT_MAT2_ARB                 0x8B5A
#define GL_FLOAT_MAT3_ARB                 0x8B5B
#define GL_FLOAT_MAT4_ARB                 0x8B5C
#define GL_SAMPLER_1D_ARB                 0x8B5D
#define GL_SAMPLER_2D_ARB                 0x8B5E
#define GL_SAMPLER_3D_ARB                 0x8B5F
#define GL_SAMPLER_CUBE_ARB               0x8B60
#define GL_SAMPLER_1D_SHADOW_ARB          0x8B61
#define GL_SAMPLER_2D_SHADOW_ARB          0x8B62

/* ARB_shading_language_100 */
#define GL_SHADING_LANGUAGE_VERSION_ARB   0x8B8C

/* ARB_vertex_shader */
#define GL_VERTEX_SHADER_ARB              0x8B31
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB 0x8B4A
#define GL_MAX_VARYING_FLOATS_ARB         0x8B4B
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB 0x8B4C
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB 0x8B4D
#define GL_OBJECT_ACTIVE_ATTRIBUTES_ARB   0x8B89
#define GL_OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH_ARB 0x8B8A

/* ARB_fragment_shader */
#define GL_FRAGMENT_SHADER_ARB            0x8B30
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB 0x8B49

/* EXT_Cg_shader */
#define GL_CG_VERTEX_SHADER_EXT           0x890E
#define GL_CG_FRAGMENT_SHADER_EXT         0x890F

/* OpenGL15 */
#define GL_FOG_COORD_SRC                  0x8450
#define GL_FOG_COORD                      0x8451
#define GL_CURRENT_FOG_COORD              0x8453
#define GL_FOG_COORD_ARRAY_TYPE           0x8454
#define GL_FOG_COORD_ARRAY_STRIDE         0x8455
#define GL_FOG_COORD_ARRAY_POINTER        0x8456
#define GL_FOG_COORD_ARRAY                0x8457
#define GL_SRC0_RGB                       0x8580
#define GL_SRC1_RGB                       0x8581
#define GL_SRC2_RGB                       0x8582
#define GL_SRC0_ALPHA                     0x8588
#define GL_SRC1_ALPHA                     0x8589
#define GL_SRC2_ALPHA                     0x858A
#define GL_QUERY_COUNTER_BITS             0x8864
#define GL_CURRENT_QUERY                  0x8865
#define GL_QUERY_RESULT                   0x8866
#define GL_QUERY_RESULT_AVAILABLE         0x8867
#define GL_SAMPLES_PASSED                 0x8914
#define GL_ARRAY_BUFFER                   0x8892
#define GL_ELEMENT_ARRAY_BUFFER           0x8893
#define GL_ARRAY_BUFFER_BINDING           0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING   0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING    0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING    0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING     0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING     0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING 0x889A
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING 0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING 0x889C
#define GL_FOG_COORD_ARRAY_BUFFER_BINDING 0x889D
#define GL_STREAM_DRAW                    0x88E0
#define GL_STREAM_READ                    0x88E1
#define GL_STREAM_COPY                    0x88E2
#define GL_STATIC_DRAW                    0x88E4
#define GL_STATIC_READ                    0x88E5
#define GL_STATIC_COPY                    0x88E6
#define GL_DYNAMIC_DRAW                   0x88E8
#define GL_DYNAMIC_READ                   0x88E9
#define GL_DYNAMIC_COPY                   0x88EA
#define GL_READ_ONLY                      0x88B8
#define GL_WRITE_ONLY                     0x88B9
#define GL_READ_WRITE                     0x88BA
#define GL_BUFFER_SIZE                    0x8764
#define GL_BUFFER_USAGE                   0x8765
#define GL_BUFFER_ACCESS                  0x88BB
#define GL_BUFFER_MAPPED                  0x88BC
#define GL_BUFFER_MAP_POINTER             0x88BD

/* NV_vertex_program2_option */
#define GL_MAX_PROGRAM_EXEC_INSTRUCTIONS_NV 0x88F4
#define GL_MAX_PROGRAM_CALL_DEPTH_NV      0x88F5

/* NV_fragment_program2 */
/*      GL_MAX_PROGRAM_EXEC_INSTRUCTIONS_NV */
/*      GL_MAX_PROGRAM_CALL_DEPTH_NV */
#define GL_MAX_PROGRAM_IF_DEPTH_NV        0x88F6
#define GL_MAX_PROGRAM_LOOP_DEPTH_NV      0x88F7
#define GL_MAX_PROGRAM_LOOP_COUNT_NV      0x88F8

/* EXT_framebuffer_object */
#define GL_FRAMEBUFFER_EXT                0X6000
#define GL_TEXTURE_FACE_EXT               0x6001
#define GL_TEXTURE_IMAGE_EXT              0x6002
#define GL_TEXTURE_LEVEL_EXT              0x6003
#define GL_BUFFER_TYPE_EXT                0x6006
#define GL_BUFFER_NAME_EXT                0x6007
#define GL_TEXTURE_BUFFER_TYPE            0x6008
#define GL_INTRINSIC_BUFFER_TYPE          0x6009
#define GL_INTRINSIC_WIDTH_EXT            0x600A
#define GL_INTRINSIC_HEIGHT_EXT           0x600B

/*************************************************************/



#endif /* __glext_h_ */
