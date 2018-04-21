/*
** The OpenGL Extension Wrangler Library
** Copyright (C) 2003, 2002, Milan Ikits <milan.ikits@ieee.org>
** Copyright (C) 2003, 2002, Marcelo E. Magallon <mmagallo@debian.org>
** Copyright (C) 2002, Lev Povalahev <levp@gmx.net>
** All rights reserved.
** 
** Redistribution and use in source and binary forms, with or without 
** modification, are permitted provided that the following conditions are met:
** 
** * Redistributions of source code must retain the above copyright notice, 
**   this list of conditions and the following disclaimer.
** * Redistributions in binary form must reproduce the above copyright notice, 
**   this list of conditions and the following disclaimer in the documentation 
**   and/or other materials provided with the distribution.
** * The name of the author may be used to endorse or promote products 
**   derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
** LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
** CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
** INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
** CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
** ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
** THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
** License Applicability. Except to the extent portions of this file are
** made subject to an alternative license as permitted in the SGI Free
** Software License B, Version 1.1 (the "License"), the contents of this
** file are subject only to the provisions of the License. You may not use
** this file except in compliance with the License. You may obtain a copy
** of the License at Silicon Graphics, Inc., attn: Legal Services, 1600
** Amphitheatre Parkway, Mountain View, CA 94043-1351, or at:
** 
** http://oss.sgi.com/projects/FreeB
** 
** Note that, as provided in the License, the Software is distributed on an
** "AS IS" basis, with ALL EXPRESS AND IMPLIED WARRANTIES AND CONDITIONS
** DISCLAIMED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES AND
** CONDITIONS OF MERCHANTABILITY, SATISFACTORY QUALITY, FITNESS FOR A
** PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
** 
** Original Code. The Original Code is: OpenGL Sample Implementation,
** Version 1.2.1, released January 26, 2000, developed by Silicon Graphics,
** Inc. The Original Code is Copyright (c) 1991-2000 Silicon Graphics, Inc.
** Copyright in any portions created by third parties is as indicated
** elsewhere herein. All Rights Reserved.
** 
** Additional Notice Provisions: This software was created using the
** OpenGL(R) version 1.2.1 Sample Implementation published by SGI, but has
** not been independently verified as being compliant with the OpenGL(R)
** version 1.2.1 Specification.
*/

#ifndef __wglew_h__
#define __wglew_h__
#define __WGLEW_H__

#ifdef __wglext_h_
#error wglext.h included before wglew.h
#endif

#define __wglext_h_

#if !defined(APIENTRY) && !defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

/*
 * GLEW_STATIC needs to be set when using the static version.
 * GLEW_BUILD is set when building the DLL version.
 */
#ifdef GLEW_STATIC
#  define GLEWAPI extern
#else
#  ifdef GLEW_BUILD
#    define GLEWAPI extern __declspec(dllexport)
#  else
#    define GLEWAPI extern __declspec(dllimport)
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------- WGL_3DFX_multisample ------------------------- */

#ifndef WGL_3DFX_multisample
#define WGL_3DFX_multisample 1

#define WGL_SAMPLE_BUFFERS_3DFX 0x2060
#define WGL_SAMPLES_3DFX 0x2061

GLEWAPI GLboolean WGLEW_3DFX_multisample;

#endif /* WGL_3DFX_multisample */

/* ------------------------- WGL_ARB_buffer_region ------------------------- */

#ifndef WGL_ARB_buffer_region
#define WGL_ARB_buffer_region 1

#define WGL_FRONT_COLOR_BUFFER_BIT_ARB 0x00000001
#define WGL_BACK_COLOR_BUFFER_BIT_ARB 0x00000002
#define WGL_DEPTH_BUFFER_BIT_ARB 0x00000004
#define WGL_STENCIL_BUFFER_BIT_ARB 0x00000008

typedef HANDLE (WINAPI * PFNWGLCREATEBUFFERREGIONARBPROC) (HDC hDC, int iLayerPlane, UINT uType);
typedef VOID (WINAPI * PFNWGLDELETEBUFFERREGIONARBPROC) (HANDLE hRegion);
typedef BOOL (WINAPI * PFNWGLRESTOREBUFFERREGIONARBPROC) (HANDLE hRegion, int x, int y, int width, int height, int xSrc, int ySrc);
typedef BOOL (WINAPI * PFNWGLSAVEBUFFERREGIONARBPROC) (HANDLE hRegion, int x, int y, int width, int height);

GLEWAPI PFNWGLCREATEBUFFERREGIONARBPROC wglewCreateBufferRegionARB;
GLEWAPI PFNWGLDELETEBUFFERREGIONARBPROC wglewDeleteBufferRegionARB;
GLEWAPI PFNWGLRESTOREBUFFERREGIONARBPROC wglewRestoreBufferRegionARB;
GLEWAPI PFNWGLSAVEBUFFERREGIONARBPROC wglewSaveBufferRegionARB;

#define wglCreateBufferRegionARB wglewCreateBufferRegionARB
#define wglDeleteBufferRegionARB wglewDeleteBufferRegionARB
#define wglRestoreBufferRegionARB wglewRestoreBufferRegionARB
#define wglSaveBufferRegionARB wglewSaveBufferRegionARB

GLEWAPI GLboolean WGLEW_ARB_buffer_region;

#endif /* WGL_ARB_buffer_region */

/* ----------------------- WGL_ARB_extensions_string ----------------------- */

#ifndef WGL_ARB_extensions_string
#define WGL_ARB_extensions_string 1

typedef const char* (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);

GLEWAPI PFNWGLGETEXTENSIONSSTRINGARBPROC wglewGetExtensionsStringARB;

#define wglGetExtensionsStringARB wglewGetExtensionsStringARB

GLEWAPI GLboolean WGLEW_ARB_extensions_string;

#endif /* WGL_ARB_extensions_string */

/* ----------------------- WGL_ARB_make_current_read ----------------------- */

#ifndef WGL_ARB_make_current_read
#define WGL_ARB_make_current_read 1

typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCARBPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTARBPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);

GLEWAPI PFNWGLGETCURRENTREADDCARBPROC wglewGetCurrentReadDCARB;
GLEWAPI PFNWGLMAKECONTEXTCURRENTARBPROC wglewMakeContextCurrentARB;

#define wglGetCurrentReadDCARB wglewGetCurrentReadDCARB
#define wglMakeContextCurrentARB wglewMakeContextCurrentARB

GLEWAPI GLboolean WGLEW_ARB_make_current_read;

#endif /* WGL_ARB_make_current_read */

/* -------------------------- WGL_ARB_multisample -------------------------- */

#ifndef WGL_ARB_multisample
#define WGL_ARB_multisample 1

#define WGL_SAMPLE_BUFFERS_ARB 0x2041
#define WGL_SAMPLES_ARB 0x2042

GLEWAPI GLboolean WGLEW_ARB_multisample;

#endif /* WGL_ARB_multisample */

/* ---------------------------- WGL_ARB_pbuffer ---------------------------- */

#ifndef WGL_ARB_pbuffer
#define WGL_ARB_pbuffer 1

#define WGL_DRAW_TO_PBUFFER_ARB 0x202D
#define WGL_MAX_PBUFFER_PIXELS_ARB 0x202E
#define WGL_MAX_PBUFFER_WIDTH_ARB 0x202F
#define WGL_MAX_PBUFFER_HEIGHT_ARB 0x2030
#define WGL_PBUFFER_LARGEST_ARB 0x2033
#define WGL_PBUFFER_WIDTH_ARB 0x2034
#define WGL_PBUFFER_HEIGHT_ARB 0x2035
#define WGL_PBUFFER_LOST_ARB 0x2036

DECLARE_HANDLE(HPBUFFERARB);

typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int* piAttribList);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFERARBPROC) (HPBUFFERARB hPbuffer, int iAttribute, int* piValue);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer, HDC hDC);

GLEWAPI PFNWGLCREATEPBUFFERARBPROC wglewCreatePbufferARB;
GLEWAPI PFNWGLDESTROYPBUFFERARBPROC wglewDestroyPbufferARB;
GLEWAPI PFNWGLGETPBUFFERDCARBPROC wglewGetPbufferDCARB;
GLEWAPI PFNWGLQUERYPBUFFERARBPROC wglewQueryPbufferARB;
GLEWAPI PFNWGLRELEASEPBUFFERDCARBPROC wglewReleasePbufferDCARB;

#define wglCreatePbufferARB wglewCreatePbufferARB
#define wglDestroyPbufferARB wglewDestroyPbufferARB
#define wglGetPbufferDCARB wglewGetPbufferDCARB
#define wglQueryPbufferARB wglewQueryPbufferARB
#define wglReleasePbufferDCARB wglewReleasePbufferDCARB

GLEWAPI GLboolean WGLEW_ARB_pbuffer;

#endif /* WGL_ARB_pbuffer */

/* -------------------------- WGL_ARB_pixel_format ------------------------- */

#ifndef WGL_ARB_pixel_format
#define WGL_ARB_pixel_format 1

#define WGL_NUMBER_PIXEL_FORMATS_ARB 0x2000
#define WGL_DRAW_TO_WINDOW_ARB 0x2001
#define WGL_DRAW_TO_BITMAP_ARB 0x2002
#define WGL_ACCELERATION_ARB 0x2003
#define WGL_NEED_PALETTE_ARB 0x2004
#define WGL_NEED_SYSTEM_PALETTE_ARB 0x2005
#define WGL_SWAP_LAYER_BUFFERS_ARB 0x2006
#define WGL_SWAP_METHOD_ARB 0x2007
#define WGL_NUMBER_OVERLAYS_ARB 0x2008
#define WGL_NUMBER_UNDERLAYS_ARB 0x2009
#define WGL_TRANSPARENT_ARB 0x200A
#define WGL_SHARE_DEPTH_ARB 0x200C
#define WGL_SHARE_STENCIL_ARB 0x200D
#define WGL_SHARE_ACCUM_ARB 0x200E
#define WGL_SUPPORT_GDI_ARB 0x200F
#define WGL_SUPPORT_OPENGL_ARB 0x2010
#define WGL_DOUBLE_BUFFER_ARB 0x2011
#define WGL_STEREO_ARB 0x2012
#define WGL_PIXEL_TYPE_ARB 0x2013
#define WGL_COLOR_BITS_ARB 0x2014
#define WGL_RED_BITS_ARB 0x2015
#define WGL_RED_SHIFT_ARB 0x2016
#define WGL_GREEN_BITS_ARB 0x2017
#define WGL_GREEN_SHIFT_ARB 0x2018
#define WGL_BLUE_BITS_ARB 0x2019
#define WGL_BLUE_SHIFT_ARB 0x201A
#define WGL_ALPHA_BITS_ARB 0x201B
#define WGL_ALPHA_SHIFT_ARB 0x201C
#define WGL_ACCUM_BITS_ARB 0x201D
#define WGL_ACCUM_RED_BITS_ARB 0x201E
#define WGL_ACCUM_GREEN_BITS_ARB 0x201F
#define WGL_ACCUM_BLUE_BITS_ARB 0x2020
#define WGL_ACCUM_ALPHA_BITS_ARB 0x2021
#define WGL_DEPTH_BITS_ARB 0x2022
#define WGL_STENCIL_BITS_ARB 0x2023
#define WGL_AUX_BUFFERS_ARB 0x2024
#define WGL_NO_ACCELERATION_ARB 0x2025
#define WGL_GENERIC_ACCELERATION_ARB 0x2026
#define WGL_FULL_ACCELERATION_ARB 0x2027
#define WGL_SWAP_EXCHANGE_ARB 0x2028
#define WGL_SWAP_COPY_ARB 0x2029
#define WGL_SWAP_UNDEFINED_ARB 0x202A
#define WGL_TYPE_RGBA_ARB 0x202B
#define WGL_TYPE_COLORINDEX_ARB 0x202C
#define WGL_TRANSPARENT_RED_VALUE_ARB 0x2037
#define WGL_TRANSPARENT_GREEN_VALUE_ARB 0x2038
#define WGL_TRANSPARENT_BLUE_VALUE_ARB 0x2039
#define WGL_TRANSPARENT_ALPHA_VALUE_ARB 0x203A
#define WGL_TRANSPARENT_INDEX_VALUE_ARB 0x203B

typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int* piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, int *piValues);

GLEWAPI PFNWGLCHOOSEPIXELFORMATARBPROC wglewChoosePixelFormatARB;
GLEWAPI PFNWGLGETPIXELFORMATATTRIBFVARBPROC wglewGetPixelFormatAttribfvARB;
GLEWAPI PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglewGetPixelFormatAttribivARB;

#define wglChoosePixelFormatARB wglewChoosePixelFormatARB
#define wglGetPixelFormatAttribfvARB wglewGetPixelFormatAttribfvARB
#define wglGetPixelFormatAttribivARB wglewGetPixelFormatAttribivARB

GLEWAPI GLboolean WGLEW_ARB_pixel_format;

#endif /* WGL_ARB_pixel_format */

/* ------------------------- WGL_ARB_render_texture ------------------------ */

#ifndef WGL_ARB_render_texture
#define WGL_ARB_render_texture 1

#define WGL_BIND_TO_TEXTURE_RGB_ARB 0x2070
#define WGL_BIND_TO_TEXTURE_RGBA_ARB 0x2071
#define WGL_TEXTURE_FORMAT_ARB 0x2072
#define WGL_TEXTURE_TARGET_ARB 0x2073
#define WGL_MIPMAP_TEXTURE_ARB 0x2074
#define WGL_TEXTURE_RGB_ARB 0x2075
#define WGL_TEXTURE_RGBA_ARB 0x2076
#define WGL_NO_TEXTURE_ARB 0x2077
#define WGL_TEXTURE_CUBE_MAP_ARB 0x2078
#define WGL_TEXTURE_1D_ARB 0x2079
#define WGL_TEXTURE_2D_ARB 0x207A
#define WGL_MIPMAP_LEVEL_ARB 0x207B
#define WGL_CUBE_MAP_FACE_ARB 0x207C
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x207E
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x207F
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x2080
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x2081
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x2082
#define WGL_FRONT_LEFT_ARB 0x2083
#define WGL_FRONT_RIGHT_ARB 0x2084
#define WGL_BACK_LEFT_ARB 0x2085
#define WGL_BACK_RIGHT_ARB 0x2086
#define WGL_AUX0_ARB 0x2087
#define WGL_AUX1_ARB 0x2088
#define WGL_AUX2_ARB 0x2089
#define WGL_AUX3_ARB 0x208A
#define WGL_AUX4_ARB 0x208B
#define WGL_AUX5_ARB 0x208C
#define WGL_AUX6_ARB 0x208D
#define WGL_AUX7_ARB 0x208E
#define WGL_AUX8_ARB 0x208F
#define WGL_AUX9_ARB 0x2090

typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLSETPBUFFERATTRIBARBPROC) (HPBUFFERARB hPbuffer, const int* piAttribList);

GLEWAPI PFNWGLBINDTEXIMAGEARBPROC wglewBindTexImageARB;
GLEWAPI PFNWGLRELEASETEXIMAGEARBPROC wglewReleaseTexImageARB;
GLEWAPI PFNWGLSETPBUFFERATTRIBARBPROC wglewSetPbufferAttribARB;

#define wglBindTexImageARB wglewBindTexImageARB
#define wglReleaseTexImageARB wglewReleaseTexImageARB
#define wglSetPbufferAttribARB wglewSetPbufferAttribARB

GLEWAPI GLboolean WGLEW_ARB_render_texture;

#endif /* WGL_ARB_render_texture */

/* ----------------------- WGL_ATI_pixel_format_float ---------------------- */

#ifndef WGL_ATI_pixel_format_float
#define WGL_ATI_pixel_format_float 1

#define WGL_TYPE_RGBA_FLOAT_ATI 0x21A0
#define GL_RGBA_FLOAT_MODE_ATI 0x8820
#define GL_COLOR_CLEAR_UNCLAMPED_VALUE_ATI 0x8835

GLEWAPI GLboolean WGLEW_ATI_pixel_format_float;

#endif /* WGL_ATI_pixel_format_float */

/* -------------------------- WGL_EXT_depth_float -------------------------- */

#ifndef WGL_EXT_depth_float
#define WGL_EXT_depth_float 1

#define WGL_DEPTH_FLOAT_EXT 0x2040

GLEWAPI GLboolean WGLEW_EXT_depth_float;

#endif /* WGL_EXT_depth_float */

/* ---------------------- WGL_EXT_display_color_table ---------------------- */

#ifndef WGL_EXT_display_color_table
#define WGL_EXT_display_color_table 1

typedef GLboolean (WINAPI * PFNWGLBINDDISPLAYCOLORTABLEEXTPROC) (GLushort id);
typedef GLboolean (WINAPI * PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC) (GLushort id);
typedef void (WINAPI * PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC) (GLushort id);
typedef GLboolean (WINAPI * PFNWGLLOADDISPLAYCOLORTABLEEXTPROC) (GLushort* table, GLuint length);

GLEWAPI PFNWGLBINDDISPLAYCOLORTABLEEXTPROC wglewBindDisplayColorTableEXT;
GLEWAPI PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC wglewCreateDisplayColorTableEXT;
GLEWAPI PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC wglewDestroyDisplayColorTableEXT;
GLEWAPI PFNWGLLOADDISPLAYCOLORTABLEEXTPROC wglewLoadDisplayColorTableEXT;

#define wglBindDisplayColorTableEXT wglewBindDisplayColorTableEXT
#define wglCreateDisplayColorTableEXT wglewCreateDisplayColorTableEXT
#define wglDestroyDisplayColorTableEXT wglewDestroyDisplayColorTableEXT
#define wglLoadDisplayColorTableEXT wglewLoadDisplayColorTableEXT

GLEWAPI GLboolean WGLEW_EXT_display_color_table;

#endif /* WGL_EXT_display_color_table */

/* ----------------------- WGL_EXT_extensions_string ----------------------- */

#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1

typedef const char* (WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (void);

GLEWAPI PFNWGLGETEXTENSIONSSTRINGEXTPROC wglewGetExtensionsStringEXT;

#define wglGetExtensionsStringEXT wglewGetExtensionsStringEXT

GLEWAPI GLboolean WGLEW_EXT_extensions_string;

#endif /* WGL_EXT_extensions_string */

/* ----------------------- WGL_EXT_make_current_read ----------------------- */

#ifndef WGL_EXT_make_current_read
#define WGL_EXT_make_current_read 1

typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCEXTPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTEXTPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);

GLEWAPI PFNWGLGETCURRENTREADDCEXTPROC wglewGetCurrentReadDCEXT;
GLEWAPI PFNWGLMAKECONTEXTCURRENTEXTPROC wglewMakeContextCurrentEXT;

#define wglGetCurrentReadDCEXT wglewGetCurrentReadDCEXT
#define wglMakeContextCurrentEXT wglewMakeContextCurrentEXT

GLEWAPI GLboolean WGLEW_EXT_make_current_read;

#endif /* WGL_EXT_make_current_read */

/* -------------------------- WGL_EXT_multisample -------------------------- */

#ifndef WGL_EXT_multisample
#define WGL_EXT_multisample 1

#define WGL_SAMPLE_BUFFERS_EXT 0x2041
#define WGL_SAMPLES_EXT 0x2042

GLEWAPI GLboolean WGLEW_EXT_multisample;

#endif /* WGL_EXT_multisample */

/* ---------------------------- WGL_EXT_pbuffer ---------------------------- */

#ifndef WGL_EXT_pbuffer
#define WGL_EXT_pbuffer 1

#define WGL_DRAW_TO_PBUFFER_EXT 0x202D
#define WGL_MAX_PBUFFER_PIXELS_EXT 0x202E
#define WGL_MAX_PBUFFER_WIDTH_EXT 0x202F
#define WGL_MAX_PBUFFER_HEIGHT_EXT 0x2030
#define WGL_OPTIMAL_PBUFFER_WIDTH_EXT 0x2031
#define WGL_OPTIMAL_PBUFFER_HEIGHT_EXT 0x2032
#define WGL_PBUFFER_LARGEST_EXT 0x2033
#define WGL_PBUFFER_WIDTH_EXT 0x2034
#define WGL_PBUFFER_HEIGHT_EXT 0x2035

DECLARE_HANDLE(HPBUFFEREXT);

typedef HPBUFFEREXT (WINAPI * PFNWGLCREATEPBUFFEREXTPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int* piAttribList);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFEREXTPROC) (HPBUFFEREXT hPbuffer);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCEXTPROC) (HPBUFFEREXT hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFEREXTPROC) (HPBUFFEREXT hPbuffer, int iAttribute, int* piValue);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCEXTPROC) (HPBUFFEREXT hPbuffer, HDC hDC);

GLEWAPI PFNWGLCREATEPBUFFEREXTPROC wglewCreatePbufferEXT;
GLEWAPI PFNWGLDESTROYPBUFFEREXTPROC wglewDestroyPbufferEXT;
GLEWAPI PFNWGLGETPBUFFERDCEXTPROC wglewGetPbufferDCEXT;
GLEWAPI PFNWGLQUERYPBUFFEREXTPROC wglewQueryPbufferEXT;
GLEWAPI PFNWGLRELEASEPBUFFERDCEXTPROC wglewReleasePbufferDCEXT;

#define wglCreatePbufferEXT wglewCreatePbufferEXT
#define wglDestroyPbufferEXT wglewDestroyPbufferEXT
#define wglGetPbufferDCEXT wglewGetPbufferDCEXT
#define wglQueryPbufferEXT wglewQueryPbufferEXT
#define wglReleasePbufferDCEXT wglewReleasePbufferDCEXT

GLEWAPI GLboolean WGLEW_EXT_pbuffer;

#endif /* WGL_EXT_pbuffer */

/* -------------------------- WGL_EXT_pixel_format ------------------------- */

#ifndef WGL_EXT_pixel_format
#define WGL_EXT_pixel_format 1

#define WGL_NUMBER_PIXEL_FORMATS_EXT 0x2000
#define WGL_DRAW_TO_WINDOW_EXT 0x2001
#define WGL_DRAW_TO_BITMAP_EXT 0x2002
#define WGL_ACCELERATION_EXT 0x2003
#define WGL_NEED_PALETTE_EXT 0x2004
#define WGL_NEED_SYSTEM_PALETTE_EXT 0x2005
#define WGL_SWAP_LAYER_BUFFERS_EXT 0x2006
#define WGL_SWAP_METHOD_EXT 0x2007
#define WGL_NUMBER_OVERLAYS_EXT 0x2008
#define WGL_NUMBER_UNDERLAYS_EXT 0x2009
#define WGL_TRANSPARENT_EXT 0x200A
#define WGL_TRANSPARENT_VALUE_EXT 0x200B
#define WGL_SHARE_DEPTH_EXT 0x200C
#define WGL_SHARE_STENCIL_EXT 0x200D
#define WGL_SHARE_ACCUM_EXT 0x200E
#define WGL_SUPPORT_GDI_EXT 0x200F
#define WGL_SUPPORT_OPENGL_EXT 0x2010
#define WGL_DOUBLE_BUFFER_EXT 0x2011
#define WGL_STEREO_EXT 0x2012
#define WGL_PIXEL_TYPE_EXT 0x2013
#define WGL_COLOR_BITS_EXT 0x2014
#define WGL_RED_BITS_EXT 0x2015
#define WGL_RED_SHIFT_EXT 0x2016
#define WGL_GREEN_BITS_EXT 0x2017
#define WGL_GREEN_SHIFT_EXT 0x2018
#define WGL_BLUE_BITS_EXT 0x2019
#define WGL_BLUE_SHIFT_EXT 0x201A
#define WGL_ALPHA_BITS_EXT 0x201B
#define WGL_ALPHA_SHIFT_EXT 0x201C
#define WGL_ACCUM_BITS_EXT 0x201D
#define WGL_ACCUM_RED_BITS_EXT 0x201E
#define WGL_ACCUM_GREEN_BITS_EXT 0x201F
#define WGL_ACCUM_BLUE_BITS_EXT 0x2020
#define WGL_ACCUM_ALPHA_BITS_EXT 0x2021
#define WGL_DEPTH_BITS_EXT 0x2022
#define WGL_STENCIL_BITS_EXT 0x2023
#define WGL_AUX_BUFFERS_EXT 0x2024
#define WGL_NO_ACCELERATION_EXT 0x2025
#define WGL_GENERIC_ACCELERATION_EXT 0x2026
#define WGL_FULL_ACCELERATION_EXT 0x2027
#define WGL_SWAP_EXCHANGE_EXT 0x2028
#define WGL_SWAP_COPY_EXT 0x2029
#define WGL_SWAP_UNDEFINED_EXT 0x202A
#define WGL_TYPE_RGBA_EXT 0x202B
#define WGL_TYPE_COLORINDEX_EXT 0x202C

typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATEXTPROC) (HDC hdc, const int* piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVEXTPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int* piAttributes, FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVEXTPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int* piAttributes, int *piValues);

GLEWAPI PFNWGLCHOOSEPIXELFORMATEXTPROC wglewChoosePixelFormatEXT;
GLEWAPI PFNWGLGETPIXELFORMATATTRIBFVEXTPROC wglewGetPixelFormatAttribfvEXT;
GLEWAPI PFNWGLGETPIXELFORMATATTRIBIVEXTPROC wglewGetPixelFormatAttribivEXT;

#define wglChoosePixelFormatEXT wglewChoosePixelFormatEXT
#define wglGetPixelFormatAttribfvEXT wglewGetPixelFormatAttribfvEXT
#define wglGetPixelFormatAttribivEXT wglewGetPixelFormatAttribivEXT

GLEWAPI GLboolean WGLEW_EXT_pixel_format;

#endif /* WGL_EXT_pixel_format */

/* -------------------------- WGL_EXT_swap_control ------------------------- */

#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1

typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);
typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);

GLEWAPI PFNWGLGETSWAPINTERVALEXTPROC wglewGetSwapIntervalEXT;
GLEWAPI PFNWGLSWAPINTERVALEXTPROC wglewSwapIntervalEXT;

#define wglGetSwapIntervalEXT wglewGetSwapIntervalEXT
#define wglSwapIntervalEXT wglewSwapIntervalEXT

GLEWAPI GLboolean WGLEW_EXT_swap_control;

#endif /* WGL_EXT_swap_control */

/* --------------------- WGL_I3D_digital_video_control --------------------- */

#ifndef WGL_I3D_digital_video_control
#define WGL_I3D_digital_video_control 1

#define WGL_DIGITAL_VIDEO_CURSOR_ALPHA_FRAMEBUFFER_I3D 0x2050
#define WGL_DIGITAL_VIDEO_CURSOR_ALPHA_VALUE_I3D 0x2051
#define WGL_DIGITAL_VIDEO_CURSOR_INCLUDED_I3D 0x2052
#define WGL_DIGITAL_VIDEO_GAMMA_CORRECTED_I3D 0x2053

typedef BOOL (WINAPI * PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC) (HDC hDC, int iAttribute, int* piValue);
typedef BOOL (WINAPI * PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC) (HDC hDC, int iAttribute, const int* piValue);

GLEWAPI PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC wglewGetDigitalVideoParametersI3D;
GLEWAPI PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC wglewSetDigitalVideoParametersI3D;

#define wglGetDigitalVideoParametersI3D wglewGetDigitalVideoParametersI3D
#define wglSetDigitalVideoParametersI3D wglewSetDigitalVideoParametersI3D

GLEWAPI GLboolean WGLEW_I3D_digital_video_control;

#endif /* WGL_I3D_digital_video_control */

/* ----------------------------- WGL_I3D_gamma ----------------------------- */

#ifndef WGL_I3D_gamma
#define WGL_I3D_gamma 1

#define WGL_GAMMA_TABLE_SIZE_I3D 0x204E
#define WGL_GAMMA_EXCLUDE_DESKTOP_I3D 0x204F

typedef BOOL (WINAPI * PFNWGLGETGAMMATABLEI3DPROC) (HDC hDC, int iEntries, USHORT* puRed, USHORT *puGreen, USHORT *puBlue);
typedef BOOL (WINAPI * PFNWGLGETGAMMATABLEPARAMETERSI3DPROC) (HDC hDC, int iAttribute, int* piValue);
typedef BOOL (WINAPI * PFNWGLSETGAMMATABLEI3DPROC) (HDC hDC, int iEntries, const USHORT* puRed, const USHORT *puGreen, const USHORT *puBlue);
typedef BOOL (WINAPI * PFNWGLSETGAMMATABLEPARAMETERSI3DPROC) (HDC hDC, int iAttribute, const int* piValue);

GLEWAPI PFNWGLGETGAMMATABLEI3DPROC wglewGetGammaTableI3D;
GLEWAPI PFNWGLGETGAMMATABLEPARAMETERSI3DPROC wglewGetGammaTableParametersI3D;
GLEWAPI PFNWGLSETGAMMATABLEI3DPROC wglewSetGammaTableI3D;
GLEWAPI PFNWGLSETGAMMATABLEPARAMETERSI3DPROC wglewSetGammaTableParametersI3D;

#define wglGetGammaTableI3D wglewGetGammaTableI3D
#define wglGetGammaTableParametersI3D wglewGetGammaTableParametersI3D
#define wglSetGammaTableI3D wglewSetGammaTableI3D
#define wglSetGammaTableParametersI3D wglewSetGammaTableParametersI3D

GLEWAPI GLboolean WGLEW_I3D_gamma;

#endif /* WGL_I3D_gamma */

/* ---------------------------- WGL_I3D_genlock ---------------------------- */

#ifndef WGL_I3D_genlock
#define WGL_I3D_genlock 1

#define WGL_GENLOCK_SOURCE_MULTIVIEW_I3D 0x2044
#define WGL_GENLOCK_SOURCE_EXTERNAL_SYNC_I3D 0x2045
#define WGL_GENLOCK_SOURCE_EXTERNAL_FIELD_I3D 0x2046
#define WGL_GENLOCK_SOURCE_EXTERNAL_TTL_I3D 0x2047
#define WGL_GENLOCK_SOURCE_DIGITAL_SYNC_I3D 0x2048
#define WGL_GENLOCK_SOURCE_DIGITAL_FIELD_I3D 0x2049
#define WGL_GENLOCK_SOURCE_EDGE_FALLING_I3D 0x204A
#define WGL_GENLOCK_SOURCE_EDGE_RISING_I3D 0x204B
#define WGL_GENLOCK_SOURCE_EDGE_BOTH_I3D 0x204C

typedef BOOL (WINAPI * PFNWGLDISABLEGENLOCKI3DPROC) (HDC hDC);
typedef BOOL (WINAPI * PFNWGLENABLEGENLOCKI3DPROC) (HDC hDC);
typedef BOOL (WINAPI * PFNWGLGENLOCKSAMPLERATEI3DPROC) (HDC hDC, UINT uRate);
typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEDELAYI3DPROC) (HDC hDC, UINT uDelay);
typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEEDGEI3DPROC) (HDC hDC, UINT uEdge);
typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEI3DPROC) (HDC hDC, UINT uSource);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSAMPLERATEI3DPROC) (HDC hDC, UINT* uRate);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEDELAYI3DPROC) (HDC hDC, UINT* uDelay);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEEDGEI3DPROC) (HDC hDC, UINT* uEdge);
typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEI3DPROC) (HDC hDC, UINT* uSource);
typedef BOOL (WINAPI * PFNWGLISENABLEDGENLOCKI3DPROC) (HDC hDC, BOOL* pFlag);
typedef BOOL (WINAPI * PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC) (HDC hDC, UINT* uMaxLineDelay, UINT *uMaxPixelDelay);

GLEWAPI PFNWGLDISABLEGENLOCKI3DPROC wglewDisableGenlockI3D;
GLEWAPI PFNWGLENABLEGENLOCKI3DPROC wglewEnableGenlockI3D;
GLEWAPI PFNWGLGENLOCKSAMPLERATEI3DPROC wglewGenlockSampleRateI3D;
GLEWAPI PFNWGLGENLOCKSOURCEDELAYI3DPROC wglewGenlockSourceDelayI3D;
GLEWAPI PFNWGLGENLOCKSOURCEEDGEI3DPROC wglewGenlockSourceEdgeI3D;
GLEWAPI PFNWGLGENLOCKSOURCEI3DPROC wglewGenlockSourceI3D;
GLEWAPI PFNWGLGETGENLOCKSAMPLERATEI3DPROC wglewGetGenlockSampleRateI3D;
GLEWAPI PFNWGLGETGENLOCKSOURCEDELAYI3DPROC wglewGetGenlockSourceDelayI3D;
GLEWAPI PFNWGLGETGENLOCKSOURCEEDGEI3DPROC wglewGetGenlockSourceEdgeI3D;
GLEWAPI PFNWGLGETGENLOCKSOURCEI3DPROC wglewGetGenlockSourceI3D;
GLEWAPI PFNWGLISENABLEDGENLOCKI3DPROC wglewIsEnabledGenlockI3D;
GLEWAPI PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC wglewQueryGenlockMaxSourceDelayI3D;

#define wglDisableGenlockI3D wglewDisableGenlockI3D
#define wglEnableGenlockI3D wglewEnableGenlockI3D
#define wglGenlockSampleRateI3D wglewGenlockSampleRateI3D
#define wglGenlockSourceDelayI3D wglewGenlockSourceDelayI3D
#define wglGenlockSourceEdgeI3D wglewGenlockSourceEdgeI3D
#define wglGenlockSourceI3D wglewGenlockSourceI3D
#define wglGetGenlockSampleRateI3D wglewGetGenlockSampleRateI3D
#define wglGetGenlockSourceDelayI3D wglewGetGenlockSourceDelayI3D
#define wglGetGenlockSourceEdgeI3D wglewGetGenlockSourceEdgeI3D
#define wglGetGenlockSourceI3D wglewGetGenlockSourceI3D
#define wglIsEnabledGenlockI3D wglewIsEnabledGenlockI3D
#define wglQueryGenlockMaxSourceDelayI3D wglewQueryGenlockMaxSourceDelayI3D

GLEWAPI GLboolean WGLEW_I3D_genlock;

#endif /* WGL_I3D_genlock */

/* -------------------------- WGL_I3D_image_buffer ------------------------- */

#ifndef WGL_I3D_image_buffer
#define WGL_I3D_image_buffer 1

#define WGL_IMAGE_BUFFER_MIN_ACCESS_I3D 0x00000001
#define WGL_IMAGE_BUFFER_LOCK_I3D 0x00000002

typedef BOOL (WINAPI * PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC) (HDC hdc, HANDLE* pEvent, LPVOID *pAddress, DWORD *pSize, UINT count);
typedef LPVOID (WINAPI * PFNWGLCREATEIMAGEBUFFERI3DPROC) (HDC hDC, DWORD dwSize, UINT uFlags);
typedef BOOL (WINAPI * PFNWGLDESTROYIMAGEBUFFERI3DPROC) (HDC hDC, LPVOID pAddress);
typedef BOOL (WINAPI * PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC) (HDC hdc, LPVOID* pAddress, UINT count);

GLEWAPI PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC wglewAssociateImageBufferEventsI3D;
GLEWAPI PFNWGLCREATEIMAGEBUFFERI3DPROC wglewCreateImageBufferI3D;
GLEWAPI PFNWGLDESTROYIMAGEBUFFERI3DPROC wglewDestroyImageBufferI3D;
GLEWAPI PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC wglewReleaseImageBufferEventsI3D;

#define wglAssociateImageBufferEventsI3D wglewAssociateImageBufferEventsI3D
#define wglCreateImageBufferI3D wglewCreateImageBufferI3D
#define wglDestroyImageBufferI3D wglewDestroyImageBufferI3D
#define wglReleaseImageBufferEventsI3D wglewReleaseImageBufferEventsI3D

GLEWAPI GLboolean WGLEW_I3D_image_buffer;

#endif /* WGL_I3D_image_buffer */

/* ------------------------ WGL_I3D_swap_frame_lock ------------------------ */

#ifndef WGL_I3D_swap_frame_lock
#define WGL_I3D_swap_frame_lock 1

typedef BOOL (WINAPI * PFNWGLDISABLEFRAMELOCKI3DPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLENABLEFRAMELOCKI3DPROC) (VOID);
typedef BOOL (WINAPI * PFNWGLISENABLEDFRAMELOCKI3DPROC) (BOOL* pFlag);
typedef BOOL (WINAPI * PFNWGLQUERYFRAMELOCKMASTERI3DPROC) (BOOL* pFlag);

GLEWAPI PFNWGLDISABLEFRAMELOCKI3DPROC wglewDisableFrameLockI3D;
GLEWAPI PFNWGLENABLEFRAMELOCKI3DPROC wglewEnableFrameLockI3D;
GLEWAPI PFNWGLISENABLEDFRAMELOCKI3DPROC wglewIsEnabledFrameLockI3D;
GLEWAPI PFNWGLQUERYFRAMELOCKMASTERI3DPROC wglewQueryFrameLockMasterI3D;

#define wglDisableFrameLockI3D wglewDisableFrameLockI3D
#define wglEnableFrameLockI3D wglewEnableFrameLockI3D
#define wglIsEnabledFrameLockI3D wglewIsEnabledFrameLockI3D
#define wglQueryFrameLockMasterI3D wglewQueryFrameLockMasterI3D

GLEWAPI GLboolean WGLEW_I3D_swap_frame_lock;

#endif /* WGL_I3D_swap_frame_lock */

/* ------------------------ WGL_I3D_swap_frame_usage ----------------------- */

#ifndef WGL_I3D_swap_frame_usage
#define WGL_I3D_swap_frame_usage 1

typedef BOOL (WINAPI * PFNWGLBEGINFRAMETRACKINGI3DPROC) (void);
typedef BOOL (WINAPI * PFNWGLENDFRAMETRACKINGI3DPROC) (void);
typedef BOOL (WINAPI * PFNWGLGETFRAMEUSAGEI3DPROC) (float* pUsage);
typedef BOOL (WINAPI * PFNWGLQUERYFRAMETRACKINGI3DPROC) (DWORD* pFrameCount, DWORD *pMissedFrames, float *pLastMissedUsage);

GLEWAPI PFNWGLBEGINFRAMETRACKINGI3DPROC wglewBeginFrameTrackingI3D;
GLEWAPI PFNWGLENDFRAMETRACKINGI3DPROC wglewEndFrameTrackingI3D;
GLEWAPI PFNWGLGETFRAMEUSAGEI3DPROC wglewGetFrameUsageI3D;
GLEWAPI PFNWGLQUERYFRAMETRACKINGI3DPROC wglewQueryFrameTrackingI3D;

#define wglBeginFrameTrackingI3D wglewBeginFrameTrackingI3D
#define wglEndFrameTrackingI3D wglewEndFrameTrackingI3D
#define wglGetFrameUsageI3D wglewGetFrameUsageI3D
#define wglQueryFrameTrackingI3D wglewQueryFrameTrackingI3D

GLEWAPI GLboolean WGLEW_I3D_swap_frame_usage;

#endif /* WGL_I3D_swap_frame_usage */

/* -------------------------- WGL_NV_float_buffer -------------------------- */

#ifndef WGL_NV_float_buffer
#define WGL_NV_float_buffer 1

#define WGL_FLOAT_COMPONENTS_NV 0x20B0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV 0x20B1
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV 0x20B2
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV 0x20B3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV 0x20B4
#define WGL_TEXTURE_FLOAT_R_NV 0x20B5
#define WGL_TEXTURE_FLOAT_RG_NV 0x20B6
#define WGL_TEXTURE_FLOAT_RGB_NV 0x20B7
#define WGL_TEXTURE_FLOAT_RGBA_NV 0x20B8

GLEWAPI GLboolean WGLEW_NV_float_buffer;

#endif /* WGL_NV_float_buffer */

/* ---------------------- WGL_NV_render_depth_texture ---------------------- */

#ifndef WGL_NV_render_depth_texture
#define WGL_NV_render_depth_texture 1

#define WGL_NO_TEXTURE_ARB 0x2077
#define WGL_BIND_TO_TEXTURE_DEPTH_NV 0x20A3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_DEPTH_NV 0x20A4
#define WGL_DEPTH_TEXTURE_FORMAT_NV 0x20A5
#define WGL_TEXTURE_DEPTH_COMPONENT_NV 0x20A6
#define WGL_DEPTH_COMPONENT_NV 0x20A7

GLEWAPI GLboolean WGLEW_NV_render_depth_texture;

#endif /* WGL_NV_render_depth_texture */

/* -------------------- WGL_NV_render_texture_rectangle -------------------- */

#ifndef WGL_NV_render_texture_rectangle
#define WGL_NV_render_texture_rectangle 1

#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV 0x20A0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV 0x20A1
#define WGL_TEXTURE_RECTANGLE_NV 0x20A2

GLEWAPI GLboolean WGLEW_NV_render_texture_rectangle;

#endif /* WGL_NV_render_texture_rectangle */

/* ----------------------- WGL_NV_vertex_array_range ----------------------- */

#ifndef WGL_NV_vertex_array_range
#define WGL_NV_vertex_array_range 1

typedef void * (WINAPI * PFNWGLALLOCATEMEMORYNVPROC) (GLsizei size, GLfloat readFrequency, GLfloat writeFrequency, GLfloat priority);
typedef void (WINAPI * PFNWGLFREEMEMORYNVPROC) (void *pointer);

GLEWAPI PFNWGLALLOCATEMEMORYNVPROC wglewAllocateMemoryNV;
GLEWAPI PFNWGLFREEMEMORYNVPROC wglewFreeMemoryNV;

#define wglAllocateMemoryNV wglewAllocateMemoryNV
#define wglFreeMemoryNV wglewFreeMemoryNV

GLEWAPI GLboolean WGLEW_NV_vertex_array_range;

#endif /* WGL_NV_vertex_array_range */

/* -------------------------- WGL_OML_sync_control ------------------------- */

#ifndef WGL_OML_sync_control
#define WGL_OML_sync_control 1

typedef BOOL (WINAPI * PFNWGLGETMSCRATEOMLPROC) (HDC hdc, INT32* numerator, INT32 *denominator);
typedef BOOL (WINAPI * PFNWGLGETSYNCVALUESOMLPROC) (HDC hdc, INT64* ust, INT64 *msc, INT64 *sbc);
typedef INT64 (WINAPI * PFNWGLSWAPBUFFERSMSCOMLPROC) (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder);
typedef INT64 (WINAPI * PFNWGLSWAPLAYERBUFFERSMSCOMLPROC) (HDC hdc, INT fuPlanes, INT64 target_msc, INT64 divisor, INT64 remainder);
typedef BOOL (WINAPI * PFNWGLWAITFORMSCOMLPROC) (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder, INT64* ust, INT64 *msc, INT64 *sbc);
typedef BOOL (WINAPI * PFNWGLWAITFORSBCOMLPROC) (HDC hdc, INT64 target_sbc, INT64* ust, INT64 *msc, INT64 *sbc);

GLEWAPI PFNWGLGETMSCRATEOMLPROC wglewGetMscRateOML;
GLEWAPI PFNWGLGETSYNCVALUESOMLPROC wglewGetSyncValuesOML;
GLEWAPI PFNWGLSWAPBUFFERSMSCOMLPROC wglewSwapBuffersMscOML;
GLEWAPI PFNWGLSWAPLAYERBUFFERSMSCOMLPROC wglewSwapLayerBuffersMscOML;
GLEWAPI PFNWGLWAITFORMSCOMLPROC wglewWaitForMscOML;
GLEWAPI PFNWGLWAITFORSBCOMLPROC wglewWaitForSbcOML;

#define wglGetMscRateOML wglewGetMscRateOML
#define wglGetSyncValuesOML wglewGetSyncValuesOML
#define wglSwapBuffersMscOML wglewSwapBuffersMscOML
#define wglSwapLayerBuffersMscOML wglewSwapLayerBuffersMscOML
#define wglWaitForMscOML wglewWaitForMscOML
#define wglWaitForSbcOML wglewWaitForSbcOML

GLEWAPI GLboolean WGLEW_OML_sync_control;

#endif /* WGL_OML_sync_control */

/* ------------------------------------------------------------------------- */

GLEWAPI GLboolean wglewGetExtension (const GLubyte* name);

#ifdef __cplusplus
}
#endif

#undef GLEWAPI

#endif /* __wglew_h__ */
