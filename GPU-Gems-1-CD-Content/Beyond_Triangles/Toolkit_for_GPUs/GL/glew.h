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

#ifndef __glew_h__
#define __glew_h__
#define __GLEW_H__

#if defined(__gl_h_) || defined(__GL_H__)
#error gl.h included before glew.h
#endif
#if defined(__glext_h_) || defined(__GLEXT_H_)
#error glext.h included before glew.h
#endif
#if defined(__gl_ATI_h_)
#error glATI.h included before glew.h
#endif

#define __gl_h_
#define __GL_H__
#define __glext_h_
#define __GLEXT_H_
#define __gl_ATI_h_

#if defined(_WIN32)

/*
 * GLEW does not include <windows.h> to avoid name space pollution.
 * GL needs GLAPI and GLAPIENTRY, GLU needs APIENTRY, CALLBACK, and wchar_t
 * defined properly.
 */
/* <windef.h> */
#ifndef APIENTRY
#define GLEW_APIENTRY_DEFINED
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define APIENTRY __stdcall
#  elif (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#    define APIENTRY __stdcall
#  else
#    define APIENTRY
#  endif
#endif
#ifndef GLAPI
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define GLAPI extern
#  endif
#endif
/* <winnt.h> */
#ifndef CALLBACK
#define GLEW_CALLBACK_DEFINED
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define CALLBACK __attribute__ ((__stdcall__))
#  elif (defined(_M_MRX000) || defined(_M_IX86) || defined(_M_ALPHA) || defined(_M_PPC)) && !defined(MIDL_PASS)
#    define CALLBACK __stdcall
#  else
#    define CALLBACK
#  endif
#endif
/* <wingdi.h> and <winnt.h> */
#ifndef WINGDIAPI
#define GLEW_WINGDIAPI_DEFINED
#define WINGDIAPI __declspec(dllimport)
#endif
/* <ctype.h> */
#if !defined(__CYGWIN__) && !defined(__MINGW32__) && !defined(_WCHAR_T_DEFINED)
#  ifndef _WCHAR_T_DEFINED
     typedef unsigned short wchar_t;
#    define _WCHAR_T_DEFINED
#  endif
#endif

#ifndef GLAPI
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define GLAPI extern
#  else
#    define GLAPI WINGDIAPI
#  endif
#endif

#ifndef GLAPIENTRY
#define GLAPIENTRY APIENTRY
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

#else /* _UNIX */

#define GLEW_APIENTRY_DEFINED
#define APIENTRY
#define GLEWAPI extern

/* <glu.h> */
#ifndef GLAPI
#define GLAPI extern
#endif
#ifndef GLAPIENTRY
#define GLAPIENTRY
#endif

#endif /* _WIN32 */

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------- GL_VERSION_1_1 ---------------------------- */

#ifndef GL_VERSION_1_1
#define GL_VERSION_1_1 1

typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef signed char GLbyte;
typedef short GLshort;
typedef int GLint;
typedef int GLsizei;
typedef unsigned char GLubyte;
typedef unsigned short GLushort;
typedef unsigned int GLuint;
typedef float GLfloat;
typedef float GLclampf;
typedef double GLdouble;
typedef double GLclampd;
typedef void GLvoid;

#define GL_ACCUM 0x0100
#define GL_LOAD 0x0101
#define GL_RETURN 0x0102
#define GL_MULT 0x0103
#define GL_ADD 0x0104
#define GL_NEVER 0x0200
#define GL_LESS 0x0201
#define GL_EQUAL 0x0202
#define GL_LEQUAL 0x0203
#define GL_GREATER 0x0204
#define GL_NOTEQUAL 0x0205
#define GL_GEQUAL 0x0206
#define GL_ALWAYS 0x0207
#define GL_CURRENT_BIT 0x00000001
#define GL_POINT_BIT 0x00000002
#define GL_LINE_BIT 0x00000004
#define GL_POLYGON_BIT 0x00000008
#define GL_POLYGON_STIPPLE_BIT 0x00000010
#define GL_PIXEL_MODE_BIT 0x00000020
#define GL_LIGHTING_BIT 0x00000040
#define GL_FOG_BIT 0x00000080
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_ACCUM_BUFFER_BIT 0x00000200
#define GL_STENCIL_BUFFER_BIT 0x00000400
#define GL_VIEWPORT_BIT 0x00000800
#define GL_TRANSFORM_BIT 0x00001000
#define GL_ENABLE_BIT 0x00002000
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_HINT_BIT 0x00008000
#define GL_EVAL_BIT 0x00010000
#define GL_LIST_BIT 0x00020000
#define GL_TEXTURE_BIT 0x00040000
#define GL_SCISSOR_BIT 0x00080000
#define GL_ALL_ATTRIB_BITS 0x000fffff
#define GL_POINTS 0x0000
#define GL_LINES 0x0001
#define GL_LINE_LOOP 0x0002
#define GL_LINE_STRIP 0x0003
#define GL_TRIANGLES 0x0004
#define GL_TRIANGLE_STRIP 0x0005
#define GL_TRIANGLE_FAN 0x0006
#define GL_QUADS 0x0007
#define GL_QUAD_STRIP 0x0008
#define GL_POLYGON 0x0009
#define GL_ZERO 0
#define GL_ONE 1
#define GL_SRC_COLOR 0x0300
#define GL_ONE_MINUS_SRC_COLOR 0x0301
#define GL_SRC_ALPHA 0x0302
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#define GL_DST_ALPHA 0x0304
#define GL_ONE_MINUS_DST_ALPHA 0x0305
#define GL_DST_COLOR 0x0306
#define GL_ONE_MINUS_DST_COLOR 0x0307
#define GL_SRC_ALPHA_SATURATE 0x0308
#define GL_TRUE 1
#define GL_FALSE 0
#define GL_CLIP_PLANE0 0x3000
#define GL_CLIP_PLANE1 0x3001
#define GL_CLIP_PLANE2 0x3002
#define GL_CLIP_PLANE3 0x3003
#define GL_CLIP_PLANE4 0x3004
#define GL_CLIP_PLANE5 0x3005
#define GL_BYTE 0x1400
#define GL_UNSIGNED_BYTE 0x1401
#define GL_SHORT 0x1402
#define GL_UNSIGNED_SHORT 0x1403
#define GL_INT 0x1404
#define GL_UNSIGNED_INT 0x1405
#define GL_FLOAT 0x1406
#define GL_2_BYTES 0x1407
#define GL_3_BYTES 0x1408
#define GL_4_BYTES 0x1409
#define GL_DOUBLE 0x140A
#define GL_NONE 0
#define GL_FRONT_LEFT 0x0400
#define GL_FRONT_RIGHT 0x0401
#define GL_BACK_LEFT 0x0402
#define GL_BACK_RIGHT 0x0403
#define GL_FRONT 0x0404
#define GL_BACK 0x0405
#define GL_LEFT 0x0406
#define GL_RIGHT 0x0407
#define GL_FRONT_AND_BACK 0x0408
#define GL_AUX0 0x0409
#define GL_AUX1 0x040A
#define GL_AUX2 0x040B
#define GL_AUX3 0x040C
#define GL_NO_ERROR 0
#define GL_INVALID_ENUM 0x0500
#define GL_INVALID_VALUE 0x0501
#define GL_INVALID_OPERATION 0x0502
#define GL_STACK_OVERFLOW 0x0503
#define GL_STACK_UNDERFLOW 0x0504
#define GL_OUT_OF_MEMORY 0x0505
#define GL_2D 0x0600
#define GL_3D 0x0601
#define GL_3D_COLOR 0x0602
#define GL_3D_COLOR_TEXTURE 0x0603
#define GL_4D_COLOR_TEXTURE 0x0604
#define GL_PASS_THROUGH_TOKEN 0x0700
#define GL_POINT_TOKEN 0x0701
#define GL_LINE_TOKEN 0x0702
#define GL_POLYGON_TOKEN 0x0703
#define GL_BITMAP_TOKEN 0x0704
#define GL_DRAW_PIXEL_TOKEN 0x0705
#define GL_COPY_PIXEL_TOKEN 0x0706
#define GL_LINE_RESET_TOKEN 0x0707
#define GL_EXP 0x0800
#define GL_EXP2 0x0801
#define GL_CW 0x0900
#define GL_CCW 0x0901
#define GL_COEFF 0x0A00
#define GL_ORDER 0x0A01
#define GL_DOMAIN 0x0A02
#define GL_CURRENT_COLOR 0x0B00
#define GL_CURRENT_INDEX 0x0B01
#define GL_CURRENT_NORMAL 0x0B02
#define GL_CURRENT_TEXTURE_COORDS 0x0B03
#define GL_CURRENT_RASTER_COLOR 0x0B04
#define GL_CURRENT_RASTER_INDEX 0x0B05
#define GL_CURRENT_RASTER_TEXTURE_COORDS 0x0B06
#define GL_CURRENT_RASTER_POSITION 0x0B07
#define GL_CURRENT_RASTER_POSITION_VALID 0x0B08
#define GL_CURRENT_RASTER_DISTANCE 0x0B09
#define GL_POINT_SMOOTH 0x0B10
#define GL_POINT_SIZE 0x0B11
#define GL_POINT_SIZE_RANGE 0x0B12
#define GL_POINT_SIZE_GRANULARITY 0x0B13
#define GL_LINE_SMOOTH 0x0B20
#define GL_LINE_WIDTH 0x0B21
#define GL_LINE_WIDTH_RANGE 0x0B22
#define GL_LINE_WIDTH_GRANULARITY 0x0B23
#define GL_LINE_STIPPLE 0x0B24
#define GL_LINE_STIPPLE_PATTERN 0x0B25
#define GL_LINE_STIPPLE_REPEAT 0x0B26
#define GL_LIST_MODE 0x0B30
#define GL_MAX_LIST_NESTING 0x0B31
#define GL_LIST_BASE 0x0B32
#define GL_LIST_INDEX 0x0B33
#define GL_POLYGON_MODE 0x0B40
#define GL_POLYGON_SMOOTH 0x0B41
#define GL_POLYGON_STIPPLE 0x0B42
#define GL_EDGE_FLAG 0x0B43
#define GL_CULL_FACE 0x0B44
#define GL_CULL_FACE_MODE 0x0B45
#define GL_FRONT_FACE 0x0B46
#define GL_LIGHTING 0x0B50
#define GL_LIGHT_MODEL_LOCAL_VIEWER 0x0B51
#define GL_LIGHT_MODEL_TWO_SIDE 0x0B52
#define GL_LIGHT_MODEL_AMBIENT 0x0B53
#define GL_SHADE_MODEL 0x0B54
#define GL_COLOR_MATERIAL_FACE 0x0B55
#define GL_COLOR_MATERIAL_PARAMETER 0x0B56
#define GL_COLOR_MATERIAL 0x0B57
#define GL_FOG 0x0B60
#define GL_FOG_INDEX 0x0B61
#define GL_FOG_DENSITY 0x0B62
#define GL_FOG_START 0x0B63
#define GL_FOG_END 0x0B64
#define GL_FOG_MODE 0x0B65
#define GL_FOG_COLOR 0x0B66
#define GL_DEPTH_RANGE 0x0B70
#define GL_DEPTH_TEST 0x0B71
#define GL_DEPTH_WRITEMASK 0x0B72
#define GL_DEPTH_CLEAR_VALUE 0x0B73
#define GL_DEPTH_FUNC 0x0B74
#define GL_ACCUM_CLEAR_VALUE 0x0B80
#define GL_STENCIL_TEST 0x0B90
#define GL_STENCIL_CLEAR_VALUE 0x0B91
#define GL_STENCIL_FUNC 0x0B92
#define GL_STENCIL_VALUE_MASK 0x0B93
#define GL_STENCIL_FAIL 0x0B94
#define GL_STENCIL_PASS_DEPTH_FAIL 0x0B95
#define GL_STENCIL_PASS_DEPTH_PASS 0x0B96
#define GL_STENCIL_REF 0x0B97
#define GL_STENCIL_WRITEMASK 0x0B98
#define GL_MATRIX_MODE 0x0BA0
#define GL_NORMALIZE 0x0BA1
#define GL_VIEWPORT 0x0BA2
#define GL_MODELVIEW_STACK_DEPTH 0x0BA3
#define GL_PROJECTION_STACK_DEPTH 0x0BA4
#define GL_TEXTURE_STACK_DEPTH 0x0BA5
#define GL_MODELVIEW_MATRIX 0x0BA6
#define GL_PROJECTION_MATRIX 0x0BA7
#define GL_TEXTURE_MATRIX 0x0BA8
#define GL_ATTRIB_STACK_DEPTH 0x0BB0
#define GL_CLIENT_ATTRIB_STACK_DEPTH 0x0BB1
#define GL_ALPHA_TEST 0x0BC0
#define GL_ALPHA_TEST_FUNC 0x0BC1
#define GL_ALPHA_TEST_REF 0x0BC2
#define GL_DITHER 0x0BD0
#define GL_BLEND_DST 0x0BE0
#define GL_BLEND_SRC 0x0BE1
#define GL_BLEND 0x0BE2
#define GL_LOGIC_OP_MODE 0x0BF0
#define GL_INDEX_LOGIC_OP 0x0BF1
#define GL_COLOR_LOGIC_OP 0x0BF2
#define GL_AUX_BUFFERS 0x0C00
#define GL_DRAW_BUFFER 0x0C01
#define GL_READ_BUFFER 0x0C02
#define GL_SCISSOR_BOX 0x0C10
#define GL_SCISSOR_TEST 0x0C11
#define GL_INDEX_CLEAR_VALUE 0x0C20
#define GL_INDEX_WRITEMASK 0x0C21
#define GL_COLOR_CLEAR_VALUE 0x0C22
#define GL_COLOR_WRITEMASK 0x0C23
#define GL_INDEX_MODE 0x0C30
#define GL_RGBA_MODE 0x0C31
#define GL_DOUBLEBUFFER 0x0C32
#define GL_STEREO 0x0C33
#define GL_RENDER_MODE 0x0C40
#define GL_PERSPECTIVE_CORRECTION_HINT 0x0C50
#define GL_POINT_SMOOTH_HINT 0x0C51
#define GL_LINE_SMOOTH_HINT 0x0C52
#define GL_POLYGON_SMOOTH_HINT 0x0C53
#define GL_FOG_HINT 0x0C54
#define GL_TEXTURE_GEN_S 0x0C60
#define GL_TEXTURE_GEN_T 0x0C61
#define GL_TEXTURE_GEN_R 0x0C62
#define GL_TEXTURE_GEN_Q 0x0C63
#define GL_PIXEL_MAP_I_TO_I 0x0C70
#define GL_PIXEL_MAP_S_TO_S 0x0C71
#define GL_PIXEL_MAP_I_TO_R 0x0C72
#define GL_PIXEL_MAP_I_TO_G 0x0C73
#define GL_PIXEL_MAP_I_TO_B 0x0C74
#define GL_PIXEL_MAP_I_TO_A 0x0C75
#define GL_PIXEL_MAP_R_TO_R 0x0C76
#define GL_PIXEL_MAP_G_TO_G 0x0C77
#define GL_PIXEL_MAP_B_TO_B 0x0C78
#define GL_PIXEL_MAP_A_TO_A 0x0C79
#define GL_PIXEL_MAP_I_TO_I_SIZE 0x0CB0
#define GL_PIXEL_MAP_S_TO_S_SIZE 0x0CB1
#define GL_PIXEL_MAP_I_TO_R_SIZE 0x0CB2
#define GL_PIXEL_MAP_I_TO_G_SIZE 0x0CB3
#define GL_PIXEL_MAP_I_TO_B_SIZE 0x0CB4
#define GL_PIXEL_MAP_I_TO_A_SIZE 0x0CB5
#define GL_PIXEL_MAP_R_TO_R_SIZE 0x0CB6
#define GL_PIXEL_MAP_G_TO_G_SIZE 0x0CB7
#define GL_PIXEL_MAP_B_TO_B_SIZE 0x0CB8
#define GL_PIXEL_MAP_A_TO_A_SIZE 0x0CB9
#define GL_UNPACK_SWAP_BYTES 0x0CF0
#define GL_UNPACK_LSB_FIRST 0x0CF1
#define GL_UNPACK_ROW_LENGTH 0x0CF2
#define GL_UNPACK_SKIP_ROWS 0x0CF3
#define GL_UNPACK_SKIP_PIXELS 0x0CF4
#define GL_UNPACK_ALIGNMENT 0x0CF5
#define GL_PACK_SWAP_BYTES 0x0D00
#define GL_PACK_LSB_FIRST 0x0D01
#define GL_PACK_ROW_LENGTH 0x0D02
#define GL_PACK_SKIP_ROWS 0x0D03
#define GL_PACK_SKIP_PIXELS 0x0D04
#define GL_PACK_ALIGNMENT 0x0D05
#define GL_MAP_COLOR 0x0D10
#define GL_MAP_STENCIL 0x0D11
#define GL_INDEX_SHIFT 0x0D12
#define GL_INDEX_OFFSET 0x0D13
#define GL_RED_SCALE 0x0D14
#define GL_RED_BIAS 0x0D15
#define GL_ZOOM_X 0x0D16
#define GL_ZOOM_Y 0x0D17
#define GL_GREEN_SCALE 0x0D18
#define GL_GREEN_BIAS 0x0D19
#define GL_BLUE_SCALE 0x0D1A
#define GL_BLUE_BIAS 0x0D1B
#define GL_ALPHA_SCALE 0x0D1C
#define GL_ALPHA_BIAS 0x0D1D
#define GL_DEPTH_SCALE 0x0D1E
#define GL_DEPTH_BIAS 0x0D1F
#define GL_MAX_EVAL_ORDER 0x0D30
#define GL_MAX_LIGHTS 0x0D31
#define GL_MAX_CLIP_PLANES 0x0D32
#define GL_MAX_TEXTURE_SIZE 0x0D33
#define GL_MAX_PIXEL_MAP_TABLE 0x0D34
#define GL_MAX_ATTRIB_STACK_DEPTH 0x0D35
#define GL_MAX_MODELVIEW_STACK_DEPTH 0x0D36
#define GL_MAX_NAME_STACK_DEPTH 0x0D37
#define GL_MAX_PROJECTION_STACK_DEPTH 0x0D38
#define GL_MAX_TEXTURE_STACK_DEPTH 0x0D39
#define GL_MAX_VIEWPORT_DIMS 0x0D3A
#define GL_MAX_CLIENT_ATTRIB_STACK_DEPTH 0x0D3B
#define GL_SUBPIXEL_BITS 0x0D50
#define GL_INDEX_BITS 0x0D51
#define GL_RED_BITS 0x0D52
#define GL_GREEN_BITS 0x0D53
#define GL_BLUE_BITS 0x0D54
#define GL_ALPHA_BITS 0x0D55
#define GL_DEPTH_BITS 0x0D56
#define GL_STENCIL_BITS 0x0D57
#define GL_ACCUM_RED_BITS 0x0D58
#define GL_ACCUM_GREEN_BITS 0x0D59
#define GL_ACCUM_BLUE_BITS 0x0D5A
#define GL_ACCUM_ALPHA_BITS 0x0D5B
#define GL_NAME_STACK_DEPTH 0x0D70
#define GL_AUTO_NORMAL 0x0D80
#define GL_MAP1_COLOR_4 0x0D90
#define GL_MAP1_INDEX 0x0D91
#define GL_MAP1_NORMAL 0x0D92
#define GL_MAP1_TEXTURE_COORD_1 0x0D93
#define GL_MAP1_TEXTURE_COORD_2 0x0D94
#define GL_MAP1_TEXTURE_COORD_3 0x0D95
#define GL_MAP1_TEXTURE_COORD_4 0x0D96
#define GL_MAP1_VERTEX_3 0x0D97
#define GL_MAP1_VERTEX_4 0x0D98
#define GL_MAP2_COLOR_4 0x0DB0
#define GL_MAP2_INDEX 0x0DB1
#define GL_MAP2_NORMAL 0x0DB2
#define GL_MAP2_TEXTURE_COORD_1 0x0DB3
#define GL_MAP2_TEXTURE_COORD_2 0x0DB4
#define GL_MAP2_TEXTURE_COORD_3 0x0DB5
#define GL_MAP2_TEXTURE_COORD_4 0x0DB6
#define GL_MAP2_VERTEX_3 0x0DB7
#define GL_MAP2_VERTEX_4 0x0DB8
#define GL_MAP1_GRID_DOMAIN 0x0DD0
#define GL_MAP1_GRID_SEGMENTS 0x0DD1
#define GL_MAP2_GRID_DOMAIN 0x0DD2
#define GL_MAP2_GRID_SEGMENTS 0x0DD3
#define GL_TEXTURE_1D 0x0DE0
#define GL_TEXTURE_2D 0x0DE1
#define GL_FEEDBACK_BUFFER_POINTER 0x0DF0
#define GL_FEEDBACK_BUFFER_SIZE 0x0DF1
#define GL_FEEDBACK_BUFFER_TYPE 0x0DF2
#define GL_SELECTION_BUFFER_POINTER 0x0DF3
#define GL_SELECTION_BUFFER_SIZE 0x0DF4
#define GL_TEXTURE_WIDTH 0x1000
#define GL_TEXTURE_HEIGHT 0x1001
#define GL_TEXTURE_INTERNAL_FORMAT 0x1003
#define GL_TEXTURE_BORDER_COLOR 0x1004
#define GL_TEXTURE_BORDER 0x1005
#define GL_DONT_CARE 0x1100
#define GL_FASTEST 0x1101
#define GL_NICEST 0x1102
#define GL_LIGHT0 0x4000
#define GL_LIGHT1 0x4001
#define GL_LIGHT2 0x4002
#define GL_LIGHT3 0x4003
#define GL_LIGHT4 0x4004
#define GL_LIGHT5 0x4005
#define GL_LIGHT6 0x4006
#define GL_LIGHT7 0x4007
#define GL_AMBIENT 0x1200
#define GL_DIFFUSE 0x1201
#define GL_SPECULAR 0x1202
#define GL_POSITION 0x1203
#define GL_SPOT_DIRECTION 0x1204
#define GL_SPOT_EXPONENT 0x1205
#define GL_SPOT_CUTOFF 0x1206
#define GL_CONSTANT_ATTENUATION 0x1207
#define GL_LINEAR_ATTENUATION 0x1208
#define GL_QUADRATIC_ATTENUATION 0x1209
#define GL_COMPILE 0x1300
#define GL_COMPILE_AND_EXECUTE 0x1301
#define GL_CLEAR 0x1500
#define GL_AND 0x1501
#define GL_AND_REVERSE 0x1502
#define GL_COPY 0x1503
#define GL_AND_INVERTED 0x1504
#define GL_NOOP 0x1505
#define GL_XOR 0x1506
#define GL_OR 0x1507
#define GL_NOR 0x1508
#define GL_EQUIV 0x1509
#define GL_INVERT 0x150A
#define GL_OR_REVERSE 0x150B
#define GL_COPY_INVERTED 0x150C
#define GL_OR_INVERTED 0x150D
#define GL_NAND 0x150E
#define GL_SET 0x150F
#define GL_EMISSION 0x1600
#define GL_SHININESS 0x1601
#define GL_AMBIENT_AND_DIFFUSE 0x1602
#define GL_COLOR_INDEXES 0x1603
#define GL_MODELVIEW 0x1700
#define GL_PROJECTION 0x1701
#define GL_TEXTURE 0x1702
#define GL_COLOR 0x1800
#define GL_DEPTH 0x1801
#define GL_STENCIL 0x1802
#define GL_COLOR_INDEX 0x1900
#define GL_STENCIL_INDEX 0x1901
#define GL_DEPTH_COMPONENT 0x1902
#define GL_RED 0x1903
#define GL_GREEN 0x1904
#define GL_BLUE 0x1905
#define GL_ALPHA 0x1906
#define GL_RGB 0x1907
#define GL_RGBA 0x1908
#define GL_LUMINANCE 0x1909
#define GL_LUMINANCE_ALPHA 0x190A
#define GL_BITMAP 0x1A00
#define GL_POINT 0x1B00
#define GL_LINE 0x1B01
#define GL_FILL 0x1B02
#define GL_RENDER 0x1C00
#define GL_FEEDBACK 0x1C01
#define GL_SELECT 0x1C02
#define GL_FLAT 0x1D00
#define GL_SMOOTH 0x1D01
#define GL_KEEP 0x1E00
#define GL_REPLACE 0x1E01
#define GL_INCR 0x1E02
#define GL_DECR 0x1E03
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_EXTENSIONS 0x1F03
#define GL_S 0x2000
#define GL_T 0x2001
#define GL_R 0x2002
#define GL_Q 0x2003
#define GL_MODULATE 0x2100
#define GL_DECAL 0x2101
#define GL_TEXTURE_ENV_MODE 0x2200
#define GL_TEXTURE_ENV_COLOR 0x2201
#define GL_TEXTURE_ENV 0x2300
#define GL_EYE_LINEAR 0x2400
#define GL_OBJECT_LINEAR 0x2401
#define GL_SPHERE_MAP 0x2402
#define GL_TEXTURE_GEN_MODE 0x2500
#define GL_OBJECT_PLANE 0x2501
#define GL_EYE_PLANE 0x2502
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_NEAREST_MIPMAP_NEAREST 0x2700
#define GL_LINEAR_MIPMAP_NEAREST 0x2701
#define GL_NEAREST_MIPMAP_LINEAR 0x2702
#define GL_LINEAR_MIPMAP_LINEAR 0x2703
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_CLAMP 0x2900
#define GL_REPEAT 0x2901
#define GL_CLIENT_PIXEL_STORE_BIT 0x00000001
#define GL_CLIENT_VERTEX_ARRAY_BIT 0x00000002
#define GL_CLIENT_ALL_ATTRIB_BITS 0xffffffff
#define GL_POLYGON_OFFSET_FACTOR 0x8038
#define GL_POLYGON_OFFSET_UNITS 0x2A00
#define GL_POLYGON_OFFSET_POINT 0x2A01
#define GL_POLYGON_OFFSET_LINE 0x2A02
#define GL_POLYGON_OFFSET_FILL 0x8037
#define GL_ALPHA4 0x803B
#define GL_ALPHA8 0x803C
#define GL_ALPHA12 0x803D
#define GL_ALPHA16 0x803E
#define GL_LUMINANCE4 0x803F
#define GL_LUMINANCE8 0x8040
#define GL_LUMINANCE12 0x8041
#define GL_LUMINANCE16 0x8042
#define GL_LUMINANCE4_ALPHA4 0x8043
#define GL_LUMINANCE6_ALPHA2 0x8044
#define GL_LUMINANCE8_ALPHA8 0x8045
#define GL_LUMINANCE12_ALPHA4 0x8046
#define GL_LUMINANCE12_ALPHA12 0x8047
#define GL_LUMINANCE16_ALPHA16 0x8048
#define GL_INTENSITY 0x8049
#define GL_INTENSITY4 0x804A
#define GL_INTENSITY8 0x804B
#define GL_INTENSITY12 0x804C
#define GL_INTENSITY16 0x804D
#define GL_R3_G3_B2 0x2A10
#define GL_RGB4 0x804F
#define GL_RGB5 0x8050
#define GL_RGB8 0x8051
#define GL_RGB10 0x8052
#define GL_RGB12 0x8053
#define GL_RGB16 0x8054
#define GL_RGBA2 0x8055
#define GL_RGBA4 0x8056
#define GL_RGB5_A1 0x8057
#define GL_RGBA8 0x8058
#define GL_RGB10_A2 0x8059
#define GL_RGBA12 0x805A
#define GL_RGBA16 0x805B
#define GL_TEXTURE_RED_SIZE 0x805C
#define GL_TEXTURE_GREEN_SIZE 0x805D
#define GL_TEXTURE_BLUE_SIZE 0x805E
#define GL_TEXTURE_ALPHA_SIZE 0x805F
#define GL_TEXTURE_LUMINANCE_SIZE 0x8060
#define GL_TEXTURE_INTENSITY_SIZE 0x8061
#define GL_PROXY_TEXTURE_1D 0x8063
#define GL_PROXY_TEXTURE_2D 0x8064
#define GL_TEXTURE_PRIORITY 0x8066
#define GL_TEXTURE_RESIDENT 0x8067
#define GL_TEXTURE_BINDING_1D 0x8068
#define GL_TEXTURE_BINDING_2D 0x8069
#define GL_VERTEX_ARRAY 0x8074
#define GL_NORMAL_ARRAY 0x8075
#define GL_COLOR_ARRAY 0x8076
#define GL_INDEX_ARRAY 0x8077
#define GL_TEXTURE_COORD_ARRAY 0x8078
#define GL_EDGE_FLAG_ARRAY 0x8079
#define GL_VERTEX_ARRAY_SIZE 0x807A
#define GL_VERTEX_ARRAY_TYPE 0x807B
#define GL_VERTEX_ARRAY_STRIDE 0x807C
#define GL_NORMAL_ARRAY_TYPE 0x807E
#define GL_NORMAL_ARRAY_STRIDE 0x807F
#define GL_COLOR_ARRAY_SIZE 0x8081
#define GL_COLOR_ARRAY_TYPE 0x8082
#define GL_COLOR_ARRAY_STRIDE 0x8083
#define GL_INDEX_ARRAY_TYPE 0x8085
#define GL_INDEX_ARRAY_STRIDE 0x8086
#define GL_TEXTURE_COORD_ARRAY_SIZE 0x8088
#define GL_TEXTURE_COORD_ARRAY_TYPE 0x8089
#define GL_TEXTURE_COORD_ARRAY_STRIDE 0x808A
#define GL_EDGE_FLAG_ARRAY_STRIDE 0x808C
#define GL_VERTEX_ARRAY_POINTER 0x808E
#define GL_NORMAL_ARRAY_POINTER 0x808F
#define GL_COLOR_ARRAY_POINTER 0x8090
#define GL_INDEX_ARRAY_POINTER 0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER 0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER 0x8093
#define GL_V2F 0x2A20
#define GL_V3F 0x2A21
#define GL_C4UB_V2F 0x2A22
#define GL_C4UB_V3F 0x2A23
#define GL_C3F_V3F 0x2A24
#define GL_N3F_V3F 0x2A25
#define GL_C4F_N3F_V3F 0x2A26
#define GL_T2F_V3F 0x2A27
#define GL_T4F_V4F 0x2A28
#define GL_T2F_C4UB_V3F 0x2A29
#define GL_T2F_C3F_V3F 0x2A2A
#define GL_T2F_N3F_V3F 0x2A2B
#define GL_T2F_C4F_N3F_V3F 0x2A2C
#define GL_T4F_C4F_N3F_V4F 0x2A2D
#define GL_LOGIC_OP GL_INDEX_LOGIC_OP
#define GL_TEXTURE_COMPONENTS GL_TEXTURE_INTERNAL_FORMAT
#define GL_COLOR_INDEX1_EXT 0x80E2
#define GL_COLOR_INDEX2_EXT 0x80E3
#define GL_COLOR_INDEX4_EXT 0x80E4
#define GL_COLOR_INDEX8_EXT 0x80E5
#define GL_COLOR_INDEX12_EXT 0x80E6
#define GL_COLOR_INDEX16_EXT 0x80E7

GLAPI void GLAPIENTRY glAccum (GLenum op, GLfloat value);
GLAPI void GLAPIENTRY glAlphaFunc (GLenum func, GLclampf ref);
GLAPI GLboolean GLAPIENTRY glAreTexturesResident (GLsizei n, const GLuint *textures, GLboolean *residences);
GLAPI void GLAPIENTRY glArrayElement (GLint i);
GLAPI void GLAPIENTRY glBegin (GLenum mode);
GLAPI void GLAPIENTRY glBindTexture (GLenum target, GLuint texture);
GLAPI void GLAPIENTRY glBitmap (GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte *bitmap);
GLAPI void GLAPIENTRY glBlendFunc (GLenum sfactor, GLenum dfactor);
GLAPI void GLAPIENTRY glCallList (GLuint list);
GLAPI void GLAPIENTRY glCallLists (GLsizei n, GLenum type, const GLvoid *lists);
GLAPI void GLAPIENTRY glClear (GLbitfield mask);
GLAPI void GLAPIENTRY glClearAccum (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
GLAPI void GLAPIENTRY glClearColor (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
GLAPI void GLAPIENTRY glClearDepth (GLclampd depth);
GLAPI void GLAPIENTRY glClearIndex (GLfloat c);
GLAPI void GLAPIENTRY glClearStencil (GLint s);
GLAPI void GLAPIENTRY glClipPlane (GLenum plane, const GLdouble *equation);
GLAPI void GLAPIENTRY glColor3b (GLbyte red, GLbyte green, GLbyte blue);
GLAPI void GLAPIENTRY glColor3bv (const GLbyte *v);
GLAPI void GLAPIENTRY glColor3d (GLdouble red, GLdouble green, GLdouble blue);
GLAPI void GLAPIENTRY glColor3dv (const GLdouble *v);
GLAPI void GLAPIENTRY glColor3f (GLfloat red, GLfloat green, GLfloat blue);
GLAPI void GLAPIENTRY glColor3fv (const GLfloat *v);
GLAPI void GLAPIENTRY glColor3i (GLint red, GLint green, GLint blue);
GLAPI void GLAPIENTRY glColor3iv (const GLint *v);
GLAPI void GLAPIENTRY glColor3s (GLshort red, GLshort green, GLshort blue);
GLAPI void GLAPIENTRY glColor3sv (const GLshort *v);
GLAPI void GLAPIENTRY glColor3ub (GLubyte red, GLubyte green, GLubyte blue);
GLAPI void GLAPIENTRY glColor3ubv (const GLubyte *v);
GLAPI void GLAPIENTRY glColor3ui (GLuint red, GLuint green, GLuint blue);
GLAPI void GLAPIENTRY glColor3uiv (const GLuint *v);
GLAPI void GLAPIENTRY glColor3us (GLushort red, GLushort green, GLushort blue);
GLAPI void GLAPIENTRY glColor3usv (const GLushort *v);
GLAPI void GLAPIENTRY glColor4b (GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha);
GLAPI void GLAPIENTRY glColor4bv (const GLbyte *v);
GLAPI void GLAPIENTRY glColor4d (GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha);
GLAPI void GLAPIENTRY glColor4dv (const GLdouble *v);
GLAPI void GLAPIENTRY glColor4f (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
GLAPI void GLAPIENTRY glColor4fv (const GLfloat *v);
GLAPI void GLAPIENTRY glColor4i (GLint red, GLint green, GLint blue, GLint alpha);
GLAPI void GLAPIENTRY glColor4iv (const GLint *v);
GLAPI void GLAPIENTRY glColor4s (GLshort red, GLshort green, GLshort blue, GLshort alpha);
GLAPI void GLAPIENTRY glColor4sv (const GLshort *v);
GLAPI void GLAPIENTRY glColor4ub (GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha);
GLAPI void GLAPIENTRY glColor4ubv (const GLubyte *v);
GLAPI void GLAPIENTRY glColor4ui (GLuint red, GLuint green, GLuint blue, GLuint alpha);
GLAPI void GLAPIENTRY glColor4uiv (const GLuint *v);
GLAPI void GLAPIENTRY glColor4us (GLushort red, GLushort green, GLushort blue, GLushort alpha);
GLAPI void GLAPIENTRY glColor4usv (const GLushort *v);
GLAPI void GLAPIENTRY glColorMask (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
GLAPI void GLAPIENTRY glColorMaterial (GLenum face, GLenum mode);
GLAPI void GLAPIENTRY glColorPointer (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
GLAPI void GLAPIENTRY glCopyPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum type);
GLAPI void GLAPIENTRY glCopyTexImage1D (GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y, GLsizei width, GLint border);
GLAPI void GLAPIENTRY glCopyTexImage2D (GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
GLAPI void GLAPIENTRY glCopyTexSubImage1D (GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
GLAPI void GLAPIENTRY glCopyTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
GLAPI void GLAPIENTRY glCullFace (GLenum mode);
GLAPI void GLAPIENTRY glDeleteLists (GLuint list, GLsizei range);
GLAPI void GLAPIENTRY glDeleteTextures (GLsizei n, const GLuint *textures);
GLAPI void GLAPIENTRY glDepthFunc (GLenum func);
GLAPI void GLAPIENTRY glDepthMask (GLboolean flag);
GLAPI void GLAPIENTRY glDepthRange (GLclampd zNear, GLclampd zFar);
GLAPI void GLAPIENTRY glDisable (GLenum cap);
GLAPI void GLAPIENTRY glDisableClientState (GLenum array);
GLAPI void GLAPIENTRY glDrawArrays (GLenum mode, GLint first, GLsizei count);
GLAPI void GLAPIENTRY glDrawBuffer (GLenum mode);
GLAPI void GLAPIENTRY glDrawElements (GLenum mode, GLsizei count, GLenum type, const GLvoid *indices);
GLAPI void GLAPIENTRY glDrawPixels (GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels);
GLAPI void GLAPIENTRY glEdgeFlag (GLboolean flag);
GLAPI void GLAPIENTRY glEdgeFlagPointer (GLsizei stride, const GLvoid *pointer);
GLAPI void GLAPIENTRY glEdgeFlagv (const GLboolean *flag);
GLAPI void GLAPIENTRY glEnable (GLenum cap);
GLAPI void GLAPIENTRY glEnableClientState (GLenum array);
GLAPI void GLAPIENTRY glEnd (void);
GLAPI void GLAPIENTRY glEndList (void);
GLAPI void GLAPIENTRY glEvalCoord1d (GLdouble u);
GLAPI void GLAPIENTRY glEvalCoord1dv (const GLdouble *u);
GLAPI void GLAPIENTRY glEvalCoord1f (GLfloat u);
GLAPI void GLAPIENTRY glEvalCoord1fv (const GLfloat *u);
GLAPI void GLAPIENTRY glEvalCoord2d (GLdouble u, GLdouble v);
GLAPI void GLAPIENTRY glEvalCoord2dv (const GLdouble *u);
GLAPI void GLAPIENTRY glEvalCoord2f (GLfloat u, GLfloat v);
GLAPI void GLAPIENTRY glEvalCoord2fv (const GLfloat *u);
GLAPI void GLAPIENTRY glEvalMesh1 (GLenum mode, GLint i1, GLint i2);
GLAPI void GLAPIENTRY glEvalMesh2 (GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2);
GLAPI void GLAPIENTRY glEvalPoint1 (GLint i);
GLAPI void GLAPIENTRY glEvalPoint2 (GLint i, GLint j);
GLAPI void GLAPIENTRY glFeedbackBuffer (GLsizei size, GLenum type, GLfloat *buffer);
GLAPI void GLAPIENTRY glFinish (void);
GLAPI void GLAPIENTRY glFlush (void);
GLAPI void GLAPIENTRY glFogf (GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glFogfv (GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glFogi (GLenum pname, GLint param);
GLAPI void GLAPIENTRY glFogiv (GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glFrontFace (GLenum mode);
GLAPI void GLAPIENTRY glFrustum (GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
GLAPI GLuint GLAPIENTRY glGenLists (GLsizei range);
GLAPI void GLAPIENTRY glGenTextures (GLsizei n, GLuint *textures);
GLAPI void GLAPIENTRY glGetBooleanv (GLenum pname, GLboolean *params);
GLAPI void GLAPIENTRY glGetClipPlane (GLenum plane, GLdouble *equation);
GLAPI void GLAPIENTRY glGetDoublev (GLenum pname, GLdouble *params);
GLAPI GLenum GLAPIENTRY glGetError (void);
GLAPI void GLAPIENTRY glGetFloatv (GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetIntegerv (GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glGetLightfv (GLenum light, GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetLightiv (GLenum light, GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glGetMapdv (GLenum target, GLenum query, GLdouble *v);
GLAPI void GLAPIENTRY glGetMapfv (GLenum target, GLenum query, GLfloat *v);
GLAPI void GLAPIENTRY glGetMapiv (GLenum target, GLenum query, GLint *v);
GLAPI void GLAPIENTRY glGetMaterialfv (GLenum face, GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetMaterialiv (GLenum face, GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glGetPixelMapfv (GLenum map, GLfloat *values);
GLAPI void GLAPIENTRY glGetPixelMapuiv (GLenum map, GLuint *values);
GLAPI void GLAPIENTRY glGetPixelMapusv (GLenum map, GLushort *values);
GLAPI void GLAPIENTRY glGetPointerv (GLenum pname, GLvoid* *params);
GLAPI void GLAPIENTRY glGetPolygonStipple (GLubyte *mask);
GLAPI const GLubyte * GLAPIENTRY glGetString (GLenum name);
GLAPI void GLAPIENTRY glGetTexEnvfv (GLenum target, GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetTexEnviv (GLenum target, GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glGetTexGendv (GLenum coord, GLenum pname, GLdouble *params);
GLAPI void GLAPIENTRY glGetTexGenfv (GLenum coord, GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetTexGeniv (GLenum coord, GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glGetTexImage (GLenum target, GLint level, GLenum format, GLenum type, GLvoid *pixels);
GLAPI void GLAPIENTRY glGetTexLevelParameterfv (GLenum target, GLint level, GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetTexLevelParameteriv (GLenum target, GLint level, GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glGetTexParameterfv (GLenum target, GLenum pname, GLfloat *params);
GLAPI void GLAPIENTRY glGetTexParameteriv (GLenum target, GLenum pname, GLint *params);
GLAPI void GLAPIENTRY glHint (GLenum target, GLenum mode);
GLAPI void GLAPIENTRY glIndexMask (GLuint mask);
GLAPI void GLAPIENTRY glIndexPointer (GLenum type, GLsizei stride, const GLvoid *pointer);
GLAPI void GLAPIENTRY glIndexd (GLdouble c);
GLAPI void GLAPIENTRY glIndexdv (const GLdouble *c);
GLAPI void GLAPIENTRY glIndexf (GLfloat c);
GLAPI void GLAPIENTRY glIndexfv (const GLfloat *c);
GLAPI void GLAPIENTRY glIndexi (GLint c);
GLAPI void GLAPIENTRY glIndexiv (const GLint *c);
GLAPI void GLAPIENTRY glIndexs (GLshort c);
GLAPI void GLAPIENTRY glIndexsv (const GLshort *c);
GLAPI void GLAPIENTRY glIndexub (GLubyte c);
GLAPI void GLAPIENTRY glIndexubv (const GLubyte *c);
GLAPI void GLAPIENTRY glInitNames (void);
GLAPI void GLAPIENTRY glInterleavedArrays (GLenum format, GLsizei stride, const GLvoid *pointer);
GLAPI GLboolean GLAPIENTRY glIsEnabled (GLenum cap);
GLAPI GLboolean GLAPIENTRY glIsList (GLuint list);
GLAPI GLboolean GLAPIENTRY glIsTexture (GLuint texture);
GLAPI void GLAPIENTRY glLightModelf (GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glLightModelfv (GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glLightModeli (GLenum pname, GLint param);
GLAPI void GLAPIENTRY glLightModeliv (GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glLightf (GLenum light, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glLightfv (GLenum light, GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glLighti (GLenum light, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glLightiv (GLenum light, GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glLineStipple (GLint factor, GLushort pattern);
GLAPI void GLAPIENTRY glLineWidth (GLfloat width);
GLAPI void GLAPIENTRY glListBase (GLuint base);
GLAPI void GLAPIENTRY glLoadIdentity (void);
GLAPI void GLAPIENTRY glLoadMatrixd (const GLdouble *m);
GLAPI void GLAPIENTRY glLoadMatrixf (const GLfloat *m);
GLAPI void GLAPIENTRY glLoadName (GLuint name);
GLAPI void GLAPIENTRY glLogicOp (GLenum opcode);
GLAPI void GLAPIENTRY glMap1d (GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble *points);
GLAPI void GLAPIENTRY glMap1f (GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat *points);
GLAPI void GLAPIENTRY glMap2d (GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble *points);
GLAPI void GLAPIENTRY glMap2f (GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder, GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat *points);
GLAPI void GLAPIENTRY glMapGrid1d (GLint un, GLdouble u1, GLdouble u2);
GLAPI void GLAPIENTRY glMapGrid1f (GLint un, GLfloat u1, GLfloat u2);
GLAPI void GLAPIENTRY glMapGrid2d (GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2);
GLAPI void GLAPIENTRY glMapGrid2f (GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2);
GLAPI void GLAPIENTRY glMaterialf (GLenum face, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glMaterialfv (GLenum face, GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glMateriali (GLenum face, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glMaterialiv (GLenum face, GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glMatrixMode (GLenum mode);
GLAPI void GLAPIENTRY glMultMatrixd (const GLdouble *m);
GLAPI void GLAPIENTRY glMultMatrixf (const GLfloat *m);
GLAPI void GLAPIENTRY glNewList (GLuint list, GLenum mode);
GLAPI void GLAPIENTRY glNormal3b (GLbyte nx, GLbyte ny, GLbyte nz);
GLAPI void GLAPIENTRY glNormal3bv (const GLbyte *v);
GLAPI void GLAPIENTRY glNormal3d (GLdouble nx, GLdouble ny, GLdouble nz);
GLAPI void GLAPIENTRY glNormal3dv (const GLdouble *v);
GLAPI void GLAPIENTRY glNormal3f (GLfloat nx, GLfloat ny, GLfloat nz);
GLAPI void GLAPIENTRY glNormal3fv (const GLfloat *v);
GLAPI void GLAPIENTRY glNormal3i (GLint nx, GLint ny, GLint nz);
GLAPI void GLAPIENTRY glNormal3iv (const GLint *v);
GLAPI void GLAPIENTRY glNormal3s (GLshort nx, GLshort ny, GLshort nz);
GLAPI void GLAPIENTRY glNormal3sv (const GLshort *v);
GLAPI void GLAPIENTRY glNormalPointer (GLenum type, GLsizei stride, const GLvoid *pointer);
GLAPI void GLAPIENTRY glOrtho (GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
GLAPI void GLAPIENTRY glPassThrough (GLfloat token);
GLAPI void GLAPIENTRY glPixelMapfv (GLenum map, GLsizei mapsize, const GLfloat *values);
GLAPI void GLAPIENTRY glPixelMapuiv (GLenum map, GLsizei mapsize, const GLuint *values);
GLAPI void GLAPIENTRY glPixelMapusv (GLenum map, GLsizei mapsize, const GLushort *values);
GLAPI void GLAPIENTRY glPixelStoref (GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glPixelStorei (GLenum pname, GLint param);
GLAPI void GLAPIENTRY glPixelTransferf (GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glPixelTransferi (GLenum pname, GLint param);
GLAPI void GLAPIENTRY glPixelZoom (GLfloat xfactor, GLfloat yfactor);
GLAPI void GLAPIENTRY glPointSize (GLfloat size);
GLAPI void GLAPIENTRY glPolygonMode (GLenum face, GLenum mode);
GLAPI void GLAPIENTRY glPolygonOffset (GLfloat factor, GLfloat units);
GLAPI void GLAPIENTRY glPolygonStipple (const GLubyte *mask);
GLAPI void GLAPIENTRY glPopAttrib (void);
GLAPI void GLAPIENTRY glPopClientAttrib (void);
GLAPI void GLAPIENTRY glPopMatrix (void);
GLAPI void GLAPIENTRY glPopName (void);
GLAPI void GLAPIENTRY glPrioritizeTextures (GLsizei n, const GLuint *textures, const GLclampf *priorities);
GLAPI void GLAPIENTRY glPushAttrib (GLbitfield mask);
GLAPI void GLAPIENTRY glPushClientAttrib (GLbitfield mask);
GLAPI void GLAPIENTRY glPushMatrix (void);
GLAPI void GLAPIENTRY glPushName (GLuint name);
GLAPI void GLAPIENTRY glRasterPos2d (GLdouble x, GLdouble y);
GLAPI void GLAPIENTRY glRasterPos2dv (const GLdouble *v);
GLAPI void GLAPIENTRY glRasterPos2f (GLfloat x, GLfloat y);
GLAPI void GLAPIENTRY glRasterPos2fv (const GLfloat *v);
GLAPI void GLAPIENTRY glRasterPos2i (GLint x, GLint y);
GLAPI void GLAPIENTRY glRasterPos2iv (const GLint *v);
GLAPI void GLAPIENTRY glRasterPos2s (GLshort x, GLshort y);
GLAPI void GLAPIENTRY glRasterPos2sv (const GLshort *v);
GLAPI void GLAPIENTRY glRasterPos3d (GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glRasterPos3dv (const GLdouble *v);
GLAPI void GLAPIENTRY glRasterPos3f (GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glRasterPos3fv (const GLfloat *v);
GLAPI void GLAPIENTRY glRasterPos3i (GLint x, GLint y, GLint z);
GLAPI void GLAPIENTRY glRasterPos3iv (const GLint *v);
GLAPI void GLAPIENTRY glRasterPos3s (GLshort x, GLshort y, GLshort z);
GLAPI void GLAPIENTRY glRasterPos3sv (const GLshort *v);
GLAPI void GLAPIENTRY glRasterPos4d (GLdouble x, GLdouble y, GLdouble z, GLdouble w);
GLAPI void GLAPIENTRY glRasterPos4dv (const GLdouble *v);
GLAPI void GLAPIENTRY glRasterPos4f (GLfloat x, GLfloat y, GLfloat z, GLfloat w);
GLAPI void GLAPIENTRY glRasterPos4fv (const GLfloat *v);
GLAPI void GLAPIENTRY glRasterPos4i (GLint x, GLint y, GLint z, GLint w);
GLAPI void GLAPIENTRY glRasterPos4iv (const GLint *v);
GLAPI void GLAPIENTRY glRasterPos4s (GLshort x, GLshort y, GLshort z, GLshort w);
GLAPI void GLAPIENTRY glRasterPos4sv (const GLshort *v);
GLAPI void GLAPIENTRY glReadBuffer (GLenum mode);
GLAPI void GLAPIENTRY glReadPixels (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *pixels);
GLAPI void GLAPIENTRY glRectd (GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2);
GLAPI void GLAPIENTRY glRectdv (const GLdouble *v1, const GLdouble *v2);
GLAPI void GLAPIENTRY glRectf (GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
GLAPI void GLAPIENTRY glRectfv (const GLfloat *v1, const GLfloat *v2);
GLAPI void GLAPIENTRY glRecti (GLint x1, GLint y1, GLint x2, GLint y2);
GLAPI void GLAPIENTRY glRectiv (const GLint *v1, const GLint *v2);
GLAPI void GLAPIENTRY glRects (GLshort x1, GLshort y1, GLshort x2, GLshort y2);
GLAPI void GLAPIENTRY glRectsv (const GLshort *v1, const GLshort *v2);
GLAPI GLint GLAPIENTRY glRenderMode (GLenum mode);
GLAPI void GLAPIENTRY glRotated (GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glRotatef (GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glScaled (GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glScalef (GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glScissor (GLint x, GLint y, GLsizei width, GLsizei height);
GLAPI void GLAPIENTRY glSelectBuffer (GLsizei size, GLuint *buffer);
GLAPI void GLAPIENTRY glShadeModel (GLenum mode);
GLAPI void GLAPIENTRY glStencilFunc (GLenum func, GLint ref, GLuint mask);
GLAPI void GLAPIENTRY glStencilMask (GLuint mask);
GLAPI void GLAPIENTRY glStencilOp (GLenum fail, GLenum zfail, GLenum zpass);
GLAPI void GLAPIENTRY glTexCoord1d (GLdouble s);
GLAPI void GLAPIENTRY glTexCoord1dv (const GLdouble *v);
GLAPI void GLAPIENTRY glTexCoord1f (GLfloat s);
GLAPI void GLAPIENTRY glTexCoord1fv (const GLfloat *v);
GLAPI void GLAPIENTRY glTexCoord1i (GLint s);
GLAPI void GLAPIENTRY glTexCoord1iv (const GLint *v);
GLAPI void GLAPIENTRY glTexCoord1s (GLshort s);
GLAPI void GLAPIENTRY glTexCoord1sv (const GLshort *v);
GLAPI void GLAPIENTRY glTexCoord2d (GLdouble s, GLdouble t);
GLAPI void GLAPIENTRY glTexCoord2dv (const GLdouble *v);
GLAPI void GLAPIENTRY glTexCoord2f (GLfloat s, GLfloat t);
GLAPI void GLAPIENTRY glTexCoord2fv (const GLfloat *v);
GLAPI void GLAPIENTRY glTexCoord2i (GLint s, GLint t);
GLAPI void GLAPIENTRY glTexCoord2iv (const GLint *v);
GLAPI void GLAPIENTRY glTexCoord2s (GLshort s, GLshort t);
GLAPI void GLAPIENTRY glTexCoord2sv (const GLshort *v);
GLAPI void GLAPIENTRY glTexCoord3d (GLdouble s, GLdouble t, GLdouble r);
GLAPI void GLAPIENTRY glTexCoord3dv (const GLdouble *v);
GLAPI void GLAPIENTRY glTexCoord3f (GLfloat s, GLfloat t, GLfloat r);
GLAPI void GLAPIENTRY glTexCoord3fv (const GLfloat *v);
GLAPI void GLAPIENTRY glTexCoord3i (GLint s, GLint t, GLint r);
GLAPI void GLAPIENTRY glTexCoord3iv (const GLint *v);
GLAPI void GLAPIENTRY glTexCoord3s (GLshort s, GLshort t, GLshort r);
GLAPI void GLAPIENTRY glTexCoord3sv (const GLshort *v);
GLAPI void GLAPIENTRY glTexCoord4d (GLdouble s, GLdouble t, GLdouble r, GLdouble q);
GLAPI void GLAPIENTRY glTexCoord4dv (const GLdouble *v);
GLAPI void GLAPIENTRY glTexCoord4f (GLfloat s, GLfloat t, GLfloat r, GLfloat q);
GLAPI void GLAPIENTRY glTexCoord4fv (const GLfloat *v);
GLAPI void GLAPIENTRY glTexCoord4i (GLint s, GLint t, GLint r, GLint q);
GLAPI void GLAPIENTRY glTexCoord4iv (const GLint *v);
GLAPI void GLAPIENTRY glTexCoord4s (GLshort s, GLshort t, GLshort r, GLshort q);
GLAPI void GLAPIENTRY glTexCoord4sv (const GLshort *v);
GLAPI void GLAPIENTRY glTexCoordPointer (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
GLAPI void GLAPIENTRY glTexEnvf (GLenum target, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glTexEnvfv (GLenum target, GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glTexEnvi (GLenum target, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glTexEnviv (GLenum target, GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glTexGend (GLenum coord, GLenum pname, GLdouble param);
GLAPI void GLAPIENTRY glTexGendv (GLenum coord, GLenum pname, const GLdouble *params);
GLAPI void GLAPIENTRY glTexGenf (GLenum coord, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glTexGenfv (GLenum coord, GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glTexGeni (GLenum coord, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glTexGeniv (GLenum coord, GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glTexImage1D (GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
GLAPI void GLAPIENTRY glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
GLAPI void GLAPIENTRY glTexParameterf (GLenum target, GLenum pname, GLfloat param);
GLAPI void GLAPIENTRY glTexParameterfv (GLenum target, GLenum pname, const GLfloat *params);
GLAPI void GLAPIENTRY glTexParameteri (GLenum target, GLenum pname, GLint param);
GLAPI void GLAPIENTRY glTexParameteriv (GLenum target, GLenum pname, const GLint *params);
GLAPI void GLAPIENTRY glTexSubImage1D (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const GLvoid *pixels);
GLAPI void GLAPIENTRY glTexSubImage2D (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels);
GLAPI void GLAPIENTRY glTranslated (GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glTranslatef (GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glVertex2d (GLdouble x, GLdouble y);
GLAPI void GLAPIENTRY glVertex2dv (const GLdouble *v);
GLAPI void GLAPIENTRY glVertex2f (GLfloat x, GLfloat y);
GLAPI void GLAPIENTRY glVertex2fv (const GLfloat *v);
GLAPI void GLAPIENTRY glVertex2i (GLint x, GLint y);
GLAPI void GLAPIENTRY glVertex2iv (const GLint *v);
GLAPI void GLAPIENTRY glVertex2s (GLshort x, GLshort y);
GLAPI void GLAPIENTRY glVertex2sv (const GLshort *v);
GLAPI void GLAPIENTRY glVertex3d (GLdouble x, GLdouble y, GLdouble z);
GLAPI void GLAPIENTRY glVertex3dv (const GLdouble *v);
GLAPI void GLAPIENTRY glVertex3f (GLfloat x, GLfloat y, GLfloat z);
GLAPI void GLAPIENTRY glVertex3fv (const GLfloat *v);
GLAPI void GLAPIENTRY glVertex3i (GLint x, GLint y, GLint z);
GLAPI void GLAPIENTRY glVertex3iv (const GLint *v);
GLAPI void GLAPIENTRY glVertex3s (GLshort x, GLshort y, GLshort z);
GLAPI void GLAPIENTRY glVertex3sv (const GLshort *v);
GLAPI void GLAPIENTRY glVertex4d (GLdouble x, GLdouble y, GLdouble z, GLdouble w);
GLAPI void GLAPIENTRY glVertex4dv (const GLdouble *v);
GLAPI void GLAPIENTRY glVertex4f (GLfloat x, GLfloat y, GLfloat z, GLfloat w);
GLAPI void GLAPIENTRY glVertex4fv (const GLfloat *v);
GLAPI void GLAPIENTRY glVertex4i (GLint x, GLint y, GLint z, GLint w);
GLAPI void GLAPIENTRY glVertex4iv (const GLint *v);
GLAPI void GLAPIENTRY glVertex4s (GLshort x, GLshort y, GLshort z, GLshort w);
GLAPI void GLAPIENTRY glVertex4sv (const GLshort *v);
GLAPI void GLAPIENTRY glVertexPointer (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
GLAPI void GLAPIENTRY glViewport (GLint x, GLint y, GLsizei width, GLsizei height);

GLEWAPI GLboolean GLEW_VERSION_1_1;
#define glew_VERSION_1_1 GLEW_VERSION_1_1

#endif /* GL_VERSION_1_1 */

/* ------------------------------------------------------------------------- */

/* this is where we can safely include GLU */
#include <GL/glu.h>
/* ----------------------------- GL_VERSION_1_2 ---------------------------- */

#ifndef GL_VERSION_1_2
#define GL_VERSION_1_2 1

#define GL_SMOOTH_POINT_SIZE_RANGE 0x0B12
#define GL_SMOOTH_POINT_SIZE_GRANULARITY 0x0B13
#define GL_SMOOTH_LINE_WIDTH_RANGE 0x0B22
#define GL_SMOOTH_LINE_WIDTH_GRANULARITY 0x0B23
#define GL_UNSIGNED_BYTE_3_3_2 0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4 0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1 0x8034
#define GL_UNSIGNED_INT_8_8_8_8 0x8035
#define GL_UNSIGNED_INT_10_10_10_2 0x8036
#define GL_RESCALE_NORMAL 0x803A
#define GL_TEXTURE_BINDING_3D 0x806A
#define GL_PACK_SKIP_IMAGES 0x806B
#define GL_PACK_IMAGE_HEIGHT 0x806C
#define GL_UNPACK_SKIP_IMAGES 0x806D
#define GL_UNPACK_IMAGE_HEIGHT 0x806E
#define GL_TEXTURE_3D 0x806F
#define GL_PROXY_TEXTURE_3D 0x8070
#define GL_TEXTURE_DEPTH 0x8071
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_MAX_3D_TEXTURE_SIZE 0x8073
#define GL_BGR 0x80E0
#define GL_BGRA 0x80E1
#define GL_MAX_ELEMENTS_VERTICES 0x80E8
#define GL_MAX_ELEMENTS_INDICES 0x80E9
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_TEXTURE_MIN_LOD 0x813A
#define GL_TEXTURE_MAX_LOD 0x813B
#define GL_TEXTURE_BASE_LEVEL 0x813C
#define GL_TEXTURE_MAX_LEVEL 0x813D
#define GL_LIGHT_MODEL_COLOR_CONTROL 0x81F8
#define GL_SINGLE_COLOR 0x81F9
#define GL_SEPARATE_SPECULAR_COLOR 0x81FA
#define GL_UNSIGNED_BYTE_2_3_3_REV 0x8362
#define GL_UNSIGNED_SHORT_5_6_5 0x8363
#define GL_UNSIGNED_SHORT_5_6_5_REV 0x8364
#define GL_UNSIGNED_SHORT_4_4_4_4_REV 0x8365
#define GL_UNSIGNED_SHORT_1_5_5_5_REV 0x8366
#define GL_UNSIGNED_INT_8_8_8_8_REV 0x8367
#define GL_UNSIGNED_INT_2_10_10_10_REV 0x8368
#define GL_ALIASED_POINT_SIZE_RANGE 0x846D
#define GL_ALIASED_LINE_WIDTH_RANGE 0x846E

typedef void (GLAPIENTRY * PFNGLCOPYTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (GLAPIENTRY * PFNGLDRAWRANGEELEMENTSPROC) (GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const GLvoid *indices);
typedef void (GLAPIENTRY * PFNGLTEXIMAGE3DPROC) (GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
typedef void (GLAPIENTRY * PFNGLTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);

GLEWAPI PFNGLCOPYTEXSUBIMAGE3DPROC glewCopyTexSubImage3D;
GLEWAPI PFNGLDRAWRANGEELEMENTSPROC glewDrawRangeElements;
GLEWAPI PFNGLTEXIMAGE3DPROC glewTexImage3D;
GLEWAPI PFNGLTEXSUBIMAGE3DPROC glewTexSubImage3D;

#define glCopyTexSubImage3D glewCopyTexSubImage3D
#define glDrawRangeElements glewDrawRangeElements
#define glTexImage3D glewTexImage3D
#define glTexSubImage3D glewTexSubImage3D

GLEWAPI GLboolean GLEW_VERSION_1_2;

#endif /* GL_VERSION_1_2 */

/* ----------------------------- GL_VERSION_1_3 ---------------------------- */

#ifndef GL_VERSION_1_3
#define GL_VERSION_1_3 1

#define GL_MULTISAMPLE 0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE 0x809E
#define GL_SAMPLE_ALPHA_TO_ONE 0x809F
#define GL_SAMPLE_COVERAGE 0x80A0
#define GL_SAMPLE_BUFFERS 0x80A8
#define GL_SAMPLES 0x80A9
#define GL_SAMPLE_COVERAGE_VALUE 0x80AA
#define GL_SAMPLE_COVERAGE_INVERT 0x80AB
#define GL_CLAMP_TO_BORDER 0x812D
#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE1 0x84C1
#define GL_TEXTURE2 0x84C2
#define GL_TEXTURE3 0x84C3
#define GL_TEXTURE4 0x84C4
#define GL_TEXTURE5 0x84C5
#define GL_TEXTURE6 0x84C6
#define GL_TEXTURE7 0x84C7
#define GL_TEXTURE8 0x84C8
#define GL_TEXTURE9 0x84C9
#define GL_TEXTURE10 0x84CA
#define GL_TEXTURE11 0x84CB
#define GL_TEXTURE12 0x84CC
#define GL_TEXTURE13 0x84CD
#define GL_TEXTURE14 0x84CE
#define GL_TEXTURE15 0x84CF
#define GL_TEXTURE16 0x84D0
#define GL_TEXTURE17 0x84D1
#define GL_TEXTURE18 0x84D2
#define GL_TEXTURE19 0x84D3
#define GL_TEXTURE20 0x84D4
#define GL_TEXTURE21 0x84D5
#define GL_TEXTURE22 0x84D6
#define GL_TEXTURE23 0x84D7
#define GL_TEXTURE24 0x84D8
#define GL_TEXTURE25 0x84D9
#define GL_TEXTURE26 0x84DA
#define GL_TEXTURE27 0x84DB
#define GL_TEXTURE28 0x84DC
#define GL_TEXTURE29 0x84DD
#define GL_TEXTURE30 0x84DE
#define GL_TEXTURE31 0x84DF
#define GL_ACTIVE_TEXTURE 0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE 0x84E1
#define GL_MAX_TEXTURE_UNITS 0x84E2
#define GL_TRANSPOSE_MODELVIEW_MATRIX 0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX 0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX 0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX 0x84E6
#define GL_SUBTRACT 0x84E7
#define GL_COMPRESSED_ALPHA 0x84E9
#define GL_COMPRESSED_LUMINANCE 0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA 0x84EB
#define GL_COMPRESSED_INTENSITY 0x84EC
#define GL_COMPRESSED_RGB 0x84ED
#define GL_COMPRESSED_RGBA 0x84EE
#define GL_TEXTURE_COMPRESSION_HINT 0x84EF
#define GL_NORMAL_MAP 0x8511
#define GL_REFLECTION_MAP 0x8512
#define GL_TEXTURE_CUBE_MAP 0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP 0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP 0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE 0x851C
#define GL_COMBINE 0x8570
#define GL_COMBINE_RGB 0x8571
#define GL_COMBINE_ALPHA 0x8572
#define GL_RGB_SCALE 0x8573
#define GL_ADD_SIGNED 0x8574
#define GL_INTERPOLATE 0x8575
#define GL_CONSTANT 0x8576
#define GL_PRIMARY_COLOR 0x8577
#define GL_PREVIOUS 0x8578
#define GL_SOURCE0_RGB 0x8580
#define GL_SOURCE1_RGB 0x8581
#define GL_SOURCE2_RGB 0x8582
#define GL_SOURCE0_ALPHA 0x8588
#define GL_SOURCE1_ALPHA 0x8589
#define GL_SOURCE2_ALPHA 0x858A
#define GL_OPERAND0_RGB 0x8590
#define GL_OPERAND1_RGB 0x8591
#define GL_OPERAND2_RGB 0x8592
#define GL_OPERAND0_ALPHA 0x8598
#define GL_OPERAND1_ALPHA 0x8599
#define GL_OPERAND2_ALPHA 0x859A
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE 0x86A0
#define GL_TEXTURE_COMPRESSED 0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS 0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS 0x86A3
#define GL_DOT3_RGB 0x86AE
#define GL_DOT3_RGBA 0x86AF
#define GL_MULTISAMPLE_BIT 0x20000000

typedef void (GLAPIENTRY * PFNGLACTIVETEXTUREPROC) (GLenum texture);
typedef void (GLAPIENTRY * PFNGLCLIENTACTIVETEXTUREPROC) (GLenum texture);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXIMAGE1DPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXIMAGE2DPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXIMAGE3DPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC) (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLGETCOMPRESSEDTEXIMAGEPROC) (GLenum target, GLint lod, GLvoid *img);
typedef void (GLAPIENTRY * PFNGLLOADTRANSPOSEMATRIXDPROC) (const GLdouble m[16]);
typedef void (GLAPIENTRY * PFNGLLOADTRANSPOSEMATRIXFPROC) (const GLfloat m[16]);
typedef void (GLAPIENTRY * PFNGLMULTTRANSPOSEMATRIXDPROC) (const GLdouble m[16]);
typedef void (GLAPIENTRY * PFNGLMULTTRANSPOSEMATRIXFPROC) (const GLfloat m[16]);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1DPROC) (GLenum target, GLdouble s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1FPROC) (GLenum target, GLfloat s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1IPROC) (GLenum target, GLint s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1SPROC) (GLenum target, GLshort s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2DPROC) (GLenum target, GLdouble s, GLdouble t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2FPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2IPROC) (GLenum target, GLint s, GLint t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2SPROC) (GLenum target, GLshort s, GLshort t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3DPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3FPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3IPROC) (GLenum target, GLint s, GLint t, GLint r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3SPROC) (GLenum target, GLshort s, GLshort t, GLshort r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4DPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4DVPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4FPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4FVPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4IPROC) (GLenum target, GLint s, GLint t, GLint r, GLint q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4IVPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4SPROC) (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4SVPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLSAMPLECOVERAGEPROC) (GLclampf value, GLboolean invert);

GLEWAPI PFNGLACTIVETEXTUREPROC glewActiveTexture;
GLEWAPI PFNGLCLIENTACTIVETEXTUREPROC glewClientActiveTexture;
GLEWAPI PFNGLCOMPRESSEDTEXIMAGE1DPROC glewCompressedTexImage1D;
GLEWAPI PFNGLCOMPRESSEDTEXIMAGE2DPROC glewCompressedTexImage2D;
GLEWAPI PFNGLCOMPRESSEDTEXIMAGE3DPROC glewCompressedTexImage3D;
GLEWAPI PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC glewCompressedTexSubImage1D;
GLEWAPI PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC glewCompressedTexSubImage2D;
GLEWAPI PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC glewCompressedTexSubImage3D;
GLEWAPI PFNGLGETCOMPRESSEDTEXIMAGEPROC glewGetCompressedTexImage;
GLEWAPI PFNGLLOADTRANSPOSEMATRIXDPROC glewLoadTransposeMatrixd;
GLEWAPI PFNGLLOADTRANSPOSEMATRIXFPROC glewLoadTransposeMatrixf;
GLEWAPI PFNGLMULTTRANSPOSEMATRIXDPROC glewMultTransposeMatrixd;
GLEWAPI PFNGLMULTTRANSPOSEMATRIXFPROC glewMultTransposeMatrixf;
GLEWAPI PFNGLMULTITEXCOORD1DPROC glewMultiTexCoord1d;
GLEWAPI PFNGLMULTITEXCOORD1DVPROC glewMultiTexCoord1dv;
GLEWAPI PFNGLMULTITEXCOORD1FPROC glewMultiTexCoord1f;
GLEWAPI PFNGLMULTITEXCOORD1FVPROC glewMultiTexCoord1fv;
GLEWAPI PFNGLMULTITEXCOORD1IPROC glewMultiTexCoord1i;
GLEWAPI PFNGLMULTITEXCOORD1IVPROC glewMultiTexCoord1iv;
GLEWAPI PFNGLMULTITEXCOORD1SPROC glewMultiTexCoord1s;
GLEWAPI PFNGLMULTITEXCOORD1SVPROC glewMultiTexCoord1sv;
GLEWAPI PFNGLMULTITEXCOORD2DPROC glewMultiTexCoord2d;
GLEWAPI PFNGLMULTITEXCOORD2DVPROC glewMultiTexCoord2dv;
GLEWAPI PFNGLMULTITEXCOORD2FPROC glewMultiTexCoord2f;
GLEWAPI PFNGLMULTITEXCOORD2FVPROC glewMultiTexCoord2fv;
GLEWAPI PFNGLMULTITEXCOORD2IPROC glewMultiTexCoord2i;
GLEWAPI PFNGLMULTITEXCOORD2IVPROC glewMultiTexCoord2iv;
GLEWAPI PFNGLMULTITEXCOORD2SPROC glewMultiTexCoord2s;
GLEWAPI PFNGLMULTITEXCOORD2SVPROC glewMultiTexCoord2sv;
GLEWAPI PFNGLMULTITEXCOORD3DPROC glewMultiTexCoord3d;
GLEWAPI PFNGLMULTITEXCOORD3DVPROC glewMultiTexCoord3dv;
GLEWAPI PFNGLMULTITEXCOORD3FPROC glewMultiTexCoord3f;
GLEWAPI PFNGLMULTITEXCOORD3FVPROC glewMultiTexCoord3fv;
GLEWAPI PFNGLMULTITEXCOORD3IPROC glewMultiTexCoord3i;
GLEWAPI PFNGLMULTITEXCOORD3IVPROC glewMultiTexCoord3iv;
GLEWAPI PFNGLMULTITEXCOORD3SPROC glewMultiTexCoord3s;
GLEWAPI PFNGLMULTITEXCOORD3SVPROC glewMultiTexCoord3sv;
GLEWAPI PFNGLMULTITEXCOORD4DPROC glewMultiTexCoord4d;
GLEWAPI PFNGLMULTITEXCOORD4DVPROC glewMultiTexCoord4dv;
GLEWAPI PFNGLMULTITEXCOORD4FPROC glewMultiTexCoord4f;
GLEWAPI PFNGLMULTITEXCOORD4FVPROC glewMultiTexCoord4fv;
GLEWAPI PFNGLMULTITEXCOORD4IPROC glewMultiTexCoord4i;
GLEWAPI PFNGLMULTITEXCOORD4IVPROC glewMultiTexCoord4iv;
GLEWAPI PFNGLMULTITEXCOORD4SPROC glewMultiTexCoord4s;
GLEWAPI PFNGLMULTITEXCOORD4SVPROC glewMultiTexCoord4sv;
GLEWAPI PFNGLSAMPLECOVERAGEPROC glewSampleCoverage;

#define glActiveTexture glewActiveTexture
#define glClientActiveTexture glewClientActiveTexture
#define glCompressedTexImage1D glewCompressedTexImage1D
#define glCompressedTexImage2D glewCompressedTexImage2D
#define glCompressedTexImage3D glewCompressedTexImage3D
#define glCompressedTexSubImage1D glewCompressedTexSubImage1D
#define glCompressedTexSubImage2D glewCompressedTexSubImage2D
#define glCompressedTexSubImage3D glewCompressedTexSubImage3D
#define glGetCompressedTexImage glewGetCompressedTexImage
#define glLoadTransposeMatrixd glewLoadTransposeMatrixd
#define glLoadTransposeMatrixf glewLoadTransposeMatrixf
#define glMultTransposeMatrixd glewMultTransposeMatrixd
#define glMultTransposeMatrixf glewMultTransposeMatrixf
#define glMultiTexCoord1d glewMultiTexCoord1d
#define glMultiTexCoord1dv glewMultiTexCoord1dv
#define glMultiTexCoord1f glewMultiTexCoord1f
#define glMultiTexCoord1fv glewMultiTexCoord1fv
#define glMultiTexCoord1i glewMultiTexCoord1i
#define glMultiTexCoord1iv glewMultiTexCoord1iv
#define glMultiTexCoord1s glewMultiTexCoord1s
#define glMultiTexCoord1sv glewMultiTexCoord1sv
#define glMultiTexCoord2d glewMultiTexCoord2d
#define glMultiTexCoord2dv glewMultiTexCoord2dv
#define glMultiTexCoord2f glewMultiTexCoord2f
#define glMultiTexCoord2fv glewMultiTexCoord2fv
#define glMultiTexCoord2i glewMultiTexCoord2i
#define glMultiTexCoord2iv glewMultiTexCoord2iv
#define glMultiTexCoord2s glewMultiTexCoord2s
#define glMultiTexCoord2sv glewMultiTexCoord2sv
#define glMultiTexCoord3d glewMultiTexCoord3d
#define glMultiTexCoord3dv glewMultiTexCoord3dv
#define glMultiTexCoord3f glewMultiTexCoord3f
#define glMultiTexCoord3fv glewMultiTexCoord3fv
#define glMultiTexCoord3i glewMultiTexCoord3i
#define glMultiTexCoord3iv glewMultiTexCoord3iv
#define glMultiTexCoord3s glewMultiTexCoord3s
#define glMultiTexCoord3sv glewMultiTexCoord3sv
#define glMultiTexCoord4d glewMultiTexCoord4d
#define glMultiTexCoord4dv glewMultiTexCoord4dv
#define glMultiTexCoord4f glewMultiTexCoord4f
#define glMultiTexCoord4fv glewMultiTexCoord4fv
#define glMultiTexCoord4i glewMultiTexCoord4i
#define glMultiTexCoord4iv glewMultiTexCoord4iv
#define glMultiTexCoord4s glewMultiTexCoord4s
#define glMultiTexCoord4sv glewMultiTexCoord4sv
#define glSampleCoverage glewSampleCoverage

GLEWAPI GLboolean GLEW_VERSION_1_3;

#endif /* GL_VERSION_1_3 */

/* ----------------------------- GL_VERSION_1_4 ---------------------------- */

#ifndef GL_VERSION_1_4
#define GL_VERSION_1_4 1

#define GL_BLEND_DST_RGB 0x80C8
#define GL_BLEND_SRC_RGB 0x80C9
#define GL_BLEND_DST_ALPHA 0x80CA
#define GL_BLEND_SRC_ALPHA 0x80CB
#define GL_POINT_SIZE_MIN 0x8126
#define GL_POINT_SIZE_MAX 0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE 0x8128
#define GL_POINT_DISTANCE_ATTENUATION 0x8129
#define GL_GENERATE_MIPMAP 0x8191
#define GL_GENERATE_MIPMAP_HINT 0x8192
#define GL_DEPTH_COMPONENT16 0x81A5
#define GL_DEPTH_COMPONENT24 0x81A6
#define GL_DEPTH_COMPONENT32 0x81A7
#define GL_MIRRORED_REPEAT 0x8370
#define GL_FOG_COORDINATE_SOURCE 0x8450
#define GL_FOG_COORDINATE 0x8451
#define GL_FRAGMENT_DEPTH 0x8452
#define GL_CURRENT_FOG_COORDINATE 0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE 0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE 0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER 0x8456
#define GL_FOG_COORDINATE_ARRAY 0x8457
#define GL_COLOR_SUM 0x8458
#define GL_CURRENT_SECONDARY_COLOR 0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE 0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE 0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE 0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER 0x845D
#define GL_SECONDARY_COLOR_ARRAY 0x845E
#define GL_MAX_TEXTURE_LOD_BIAS 0x84FD
#define GL_TEXTURE_FILTER_CONTROL 0x8500
#define GL_TEXTURE_LOD_BIAS 0x8501
#define GL_INCR_WRAP 0x8507
#define GL_DECR_WRAP 0x8508
#define GL_TEXTURE_DEPTH_SIZE 0x884A
#define GL_DEPTH_TEXTURE_MODE 0x884B
#define GL_TEXTURE_COMPARE_MODE 0x884C
#define GL_TEXTURE_COMPARE_FUNC 0x884D
#define GL_COMPARE_R_TO_TEXTURE 0x884E

typedef void (GLAPIENTRY * PFNGLBLENDFUNCSEPARATEPROC) (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
typedef void (GLAPIENTRY * PFNGLFOGCOORDPOINTERPROC) (GLenum type, GLsizei stride, const GLvoid *pointer);
typedef void (GLAPIENTRY * PFNGLFOGCOORDDPROC) (GLdouble coord);
typedef void (GLAPIENTRY * PFNGLFOGCOORDDVPROC) (const GLdouble *coord);
typedef void (GLAPIENTRY * PFNGLFOGCOORDFPROC) (GLfloat coord);
typedef void (GLAPIENTRY * PFNGLFOGCOORDFVPROC) (const GLfloat *coord);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWARRAYSPROC) (GLenum mode, GLint *first, GLsizei *count, GLsizei primcount);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWELEMENTSPROC) (GLenum mode, GLsizei *count, GLenum type, const GLvoid **indices, GLsizei primcount);
typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERFPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERFVPROC) (GLenum pname, GLfloat *params);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3BPROC) (GLbyte red, GLbyte green, GLbyte blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3BVPROC) (const GLbyte *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3DPROC) (GLdouble red, GLdouble green, GLdouble blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3DVPROC) (const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3FPROC) (GLfloat red, GLfloat green, GLfloat blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3FVPROC) (const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3IPROC) (GLint red, GLint green, GLint blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3IVPROC) (const GLint *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3SPROC) (GLshort red, GLshort green, GLshort blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3SVPROC) (const GLshort *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UBPROC) (GLubyte red, GLubyte green, GLubyte blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UBVPROC) (const GLubyte *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UIPROC) (GLuint red, GLuint green, GLuint blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UIVPROC) (const GLuint *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3USPROC) (GLushort red, GLushort green, GLushort blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3USVPROC) (const GLushort *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLORPOINTERPROC) (GLint size, GLenum type, GLsizei stride, GLvoid *pointer);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2DPROC) (GLdouble x, GLdouble y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2DVPROC) (const GLdouble *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2FPROC) (GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2FVPROC) (const GLfloat *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2IPROC) (GLint x, GLint y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2IVPROC) (const GLint *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2SPROC) (GLshort x, GLshort y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2SVPROC) (const GLshort *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3DPROC) (GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3DVPROC) (const GLdouble *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3FPROC) (GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3FVPROC) (const GLfloat *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3IPROC) (GLint x, GLint y, GLint z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3IVPROC) (const GLint *p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3SPROC) (GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3SVPROC) (const GLshort *p);

GLEWAPI PFNGLBLENDFUNCSEPARATEPROC glewBlendFuncSeparate;
GLEWAPI PFNGLFOGCOORDPOINTERPROC glewFogCoordPointer;
GLEWAPI PFNGLFOGCOORDDPROC glewFogCoordd;
GLEWAPI PFNGLFOGCOORDDVPROC glewFogCoorddv;
GLEWAPI PFNGLFOGCOORDFPROC glewFogCoordf;
GLEWAPI PFNGLFOGCOORDFVPROC glewFogCoordfv;
GLEWAPI PFNGLMULTIDRAWARRAYSPROC glewMultiDrawArrays;
GLEWAPI PFNGLMULTIDRAWELEMENTSPROC glewMultiDrawElements;
GLEWAPI PFNGLPOINTPARAMETERFPROC glewPointParameterf;
GLEWAPI PFNGLPOINTPARAMETERFVPROC glewPointParameterfv;
GLEWAPI PFNGLSECONDARYCOLOR3BPROC glewSecondaryColor3b;
GLEWAPI PFNGLSECONDARYCOLOR3BVPROC glewSecondaryColor3bv;
GLEWAPI PFNGLSECONDARYCOLOR3DPROC glewSecondaryColor3d;
GLEWAPI PFNGLSECONDARYCOLOR3DVPROC glewSecondaryColor3dv;
GLEWAPI PFNGLSECONDARYCOLOR3FPROC glewSecondaryColor3f;
GLEWAPI PFNGLSECONDARYCOLOR3FVPROC glewSecondaryColor3fv;
GLEWAPI PFNGLSECONDARYCOLOR3IPROC glewSecondaryColor3i;
GLEWAPI PFNGLSECONDARYCOLOR3IVPROC glewSecondaryColor3iv;
GLEWAPI PFNGLSECONDARYCOLOR3SPROC glewSecondaryColor3s;
GLEWAPI PFNGLSECONDARYCOLOR3SVPROC glewSecondaryColor3sv;
GLEWAPI PFNGLSECONDARYCOLOR3UBPROC glewSecondaryColor3ub;
GLEWAPI PFNGLSECONDARYCOLOR3UBVPROC glewSecondaryColor3ubv;
GLEWAPI PFNGLSECONDARYCOLOR3UIPROC glewSecondaryColor3ui;
GLEWAPI PFNGLSECONDARYCOLOR3UIVPROC glewSecondaryColor3uiv;
GLEWAPI PFNGLSECONDARYCOLOR3USPROC glewSecondaryColor3us;
GLEWAPI PFNGLSECONDARYCOLOR3USVPROC glewSecondaryColor3usv;
GLEWAPI PFNGLSECONDARYCOLORPOINTERPROC glewSecondaryColorPointer;
GLEWAPI PFNGLWINDOWPOS2DPROC glewWindowPos2d;
GLEWAPI PFNGLWINDOWPOS2DVPROC glewWindowPos2dv;
GLEWAPI PFNGLWINDOWPOS2FPROC glewWindowPos2f;
GLEWAPI PFNGLWINDOWPOS2FVPROC glewWindowPos2fv;
GLEWAPI PFNGLWINDOWPOS2IPROC glewWindowPos2i;
GLEWAPI PFNGLWINDOWPOS2IVPROC glewWindowPos2iv;
GLEWAPI PFNGLWINDOWPOS2SPROC glewWindowPos2s;
GLEWAPI PFNGLWINDOWPOS2SVPROC glewWindowPos2sv;
GLEWAPI PFNGLWINDOWPOS3DPROC glewWindowPos3d;
GLEWAPI PFNGLWINDOWPOS3DVPROC glewWindowPos3dv;
GLEWAPI PFNGLWINDOWPOS3FPROC glewWindowPos3f;
GLEWAPI PFNGLWINDOWPOS3FVPROC glewWindowPos3fv;
GLEWAPI PFNGLWINDOWPOS3IPROC glewWindowPos3i;
GLEWAPI PFNGLWINDOWPOS3IVPROC glewWindowPos3iv;
GLEWAPI PFNGLWINDOWPOS3SPROC glewWindowPos3s;
GLEWAPI PFNGLWINDOWPOS3SVPROC glewWindowPos3sv;

#define glBlendFuncSeparate glewBlendFuncSeparate
#define glFogCoordPointer glewFogCoordPointer
#define glFogCoordd glewFogCoordd
#define glFogCoorddv glewFogCoorddv
#define glFogCoordf glewFogCoordf
#define glFogCoordfv glewFogCoordfv
#define glMultiDrawArrays glewMultiDrawArrays
#define glMultiDrawElements glewMultiDrawElements
#define glPointParameterf glewPointParameterf
#define glPointParameterfv glewPointParameterfv
#define glSecondaryColor3b glewSecondaryColor3b
#define glSecondaryColor3bv glewSecondaryColor3bv
#define glSecondaryColor3d glewSecondaryColor3d
#define glSecondaryColor3dv glewSecondaryColor3dv
#define glSecondaryColor3f glewSecondaryColor3f
#define glSecondaryColor3fv glewSecondaryColor3fv
#define glSecondaryColor3i glewSecondaryColor3i
#define glSecondaryColor3iv glewSecondaryColor3iv
#define glSecondaryColor3s glewSecondaryColor3s
#define glSecondaryColor3sv glewSecondaryColor3sv
#define glSecondaryColor3ub glewSecondaryColor3ub
#define glSecondaryColor3ubv glewSecondaryColor3ubv
#define glSecondaryColor3ui glewSecondaryColor3ui
#define glSecondaryColor3uiv glewSecondaryColor3uiv
#define glSecondaryColor3us glewSecondaryColor3us
#define glSecondaryColor3usv glewSecondaryColor3usv
#define glSecondaryColorPointer glewSecondaryColorPointer
#define glWindowPos2d glewWindowPos2d
#define glWindowPos2dv glewWindowPos2dv
#define glWindowPos2f glewWindowPos2f
#define glWindowPos2fv glewWindowPos2fv
#define glWindowPos2i glewWindowPos2i
#define glWindowPos2iv glewWindowPos2iv
#define glWindowPos2s glewWindowPos2s
#define glWindowPos2sv glewWindowPos2sv
#define glWindowPos3d glewWindowPos3d
#define glWindowPos3dv glewWindowPos3dv
#define glWindowPos3f glewWindowPos3f
#define glWindowPos3fv glewWindowPos3fv
#define glWindowPos3i glewWindowPos3i
#define glWindowPos3iv glewWindowPos3iv
#define glWindowPos3s glewWindowPos3s
#define glWindowPos3sv glewWindowPos3sv

GLEWAPI GLboolean GLEW_VERSION_1_4;

#endif /* GL_VERSION_1_4 */

/* -------------------------- GL_3DFX_multisample -------------------------- */

#ifndef GL_3DFX_multisample
#define GL_3DFX_multisample 1

#define GLX_SAMPLE_BUFFERS_3DFX 0x8050
#define GLX_SAMPLES_3DFX 0x8051
#define GL_MULTISAMPLE_3DFX 0x86B2
#define GL_SAMPLE_BUFFERS_3DFX 0x86B3
#define GL_SAMPLES_3DFX 0x86B4
#define GL_MULTISAMPLE_BIT_3DFX 0x20000000

GLEWAPI GLboolean GLEW_3DFX_multisample;

#endif /* GL_3DFX_multisample */

/* ---------------------------- GL_3DFX_tbuffer ---------------------------- */

#ifndef GL_3DFX_tbuffer
#define GL_3DFX_tbuffer 1

typedef void (GLAPIENTRY * PFNGLTBUFFERMASK3DFXPROC) (GLuint mask);

GLEWAPI PFNGLTBUFFERMASK3DFXPROC glewTbufferMask3DFX;

#define glTbufferMask3DFX glewTbufferMask3DFX

GLEWAPI GLboolean GLEW_3DFX_tbuffer;

#endif /* GL_3DFX_tbuffer */

/* -------------------- GL_3DFX_texture_compression_FXT1 ------------------- */

#ifndef GL_3DFX_texture_compression_FXT1
#define GL_3DFX_texture_compression_FXT1 1

#define GL_COMPRESSED_RGB_FXT1_3DFX 0x86B0
#define GL_COMPRESSED_RGBA_FXT1_3DFX 0x86B1

GLEWAPI GLboolean GLEW_3DFX_texture_compression_FXT1;

#endif /* GL_3DFX_texture_compression_FXT1 */

/* ------------------------ GL_APPLE_client_storage ------------------------ */

#ifndef GL_APPLE_client_storage
#define GL_APPLE_client_storage 1

#define GL_UNPACK_CLIENT_STORAGE_APPLE 0x85B2

GLEWAPI GLboolean GLEW_APPLE_client_storage;

#endif /* GL_APPLE_client_storage */

/* ------------------------- GL_APPLE_element_array ------------------------ */

#ifndef GL_APPLE_element_array
#define GL_APPLE_element_array 1

#define GL_ELEMENT_ARRAY_APPLE 0x8768
#define GL_ELEMENT_ARRAY_TYPE_APPLE 0x8769
#define GL_ELEMENT_ARRAY_POINTER_APPLE 0x876A

typedef void (GLAPIENTRY * PFNGLDRAWELEMENTARRAYAPPLEPROC) (GLenum mode, GLint first, GLsizei count);
typedef void (GLAPIENTRY * PFNGLDRAWRANGEELEMENTARRAYAPPLEPROC) (GLenum mode, GLuint start, GLuint end, GLint first, GLsizei count);
typedef void (GLAPIENTRY * PFNGLELEMENTPOINTERAPPLEPROC) (GLenum type, const void* pointer);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWELEMENTARRAYAPPLEPROC) (GLenum mode, const GLint* first, const GLsizei *count, GLsizei primcount);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWRANGEELEMENTARRAYAPPLEPROC) (GLenum mode, GLuint start, GLuint end, const GLint* first, const GLsizei *count, GLsizei primcount);

GLEWAPI PFNGLDRAWELEMENTARRAYAPPLEPROC glewDrawElementArrayAPPLE;
GLEWAPI PFNGLDRAWRANGEELEMENTARRAYAPPLEPROC glewDrawRangeElementArrayAPPLE;
GLEWAPI PFNGLELEMENTPOINTERAPPLEPROC glewElementPointerAPPLE;
GLEWAPI PFNGLMULTIDRAWELEMENTARRAYAPPLEPROC glewMultiDrawElementArrayAPPLE;
GLEWAPI PFNGLMULTIDRAWRANGEELEMENTARRAYAPPLEPROC glewMultiDrawRangeElementArrayAPPLE;

#define glDrawElementArrayAPPLE glewDrawElementArrayAPPLE
#define glDrawRangeElementArrayAPPLE glewDrawRangeElementArrayAPPLE
#define glElementPointerAPPLE glewElementPointerAPPLE
#define glMultiDrawElementArrayAPPLE glewMultiDrawElementArrayAPPLE
#define glMultiDrawRangeElementArrayAPPLE glewMultiDrawRangeElementArrayAPPLE

GLEWAPI GLboolean GLEW_APPLE_element_array;

#endif /* GL_APPLE_element_array */

/* ----------------------------- GL_APPLE_fence ---------------------------- */

#ifndef GL_APPLE_fence
#define GL_APPLE_fence 1

#define GL_DRAW_PIXELS_APPLE 0x8A0A
#define GL_FENCE_APPLE 0x8A0B

typedef void (GLAPIENTRY * PFNGLDELETEFENCESAPPLEPROC) (GLsizei n, const GLuint* fences);
typedef void (GLAPIENTRY * PFNGLFINISHFENCEAPPLEPROC) (GLuint fence);
typedef void (GLAPIENTRY * PFNGLFINISHOBJECTAPPLEPROC) (GLenum object, GLint name);
typedef void (GLAPIENTRY * PFNGLGENFENCESAPPLEPROC) (GLsizei n, GLuint* fences);
typedef GLboolean (GLAPIENTRY * PFNGLISFENCEAPPLEPROC) (GLuint fence);
typedef void (GLAPIENTRY * PFNGLSETFENCEAPPLEPROC) (GLuint fence);
typedef GLboolean (GLAPIENTRY * PFNGLTESTFENCEAPPLEPROC) (GLuint fence);
typedef GLboolean (GLAPIENTRY * PFNGLTESTOBJECTAPPLEPROC) (GLenum object, GLuint name);

GLEWAPI PFNGLDELETEFENCESAPPLEPROC glewDeleteFencesAPPLE;
GLEWAPI PFNGLFINISHFENCEAPPLEPROC glewFinishFenceAPPLE;
GLEWAPI PFNGLFINISHOBJECTAPPLEPROC glewFinishObjectAPPLE;
GLEWAPI PFNGLGENFENCESAPPLEPROC glewGenFencesAPPLE;
GLEWAPI PFNGLISFENCEAPPLEPROC glewIsFenceAPPLE;
GLEWAPI PFNGLSETFENCEAPPLEPROC glewSetFenceAPPLE;
GLEWAPI PFNGLTESTFENCEAPPLEPROC glewTestFenceAPPLE;
GLEWAPI PFNGLTESTOBJECTAPPLEPROC glewTestObjectAPPLE;

#define glDeleteFencesAPPLE glewDeleteFencesAPPLE
#define glFinishFenceAPPLE glewFinishFenceAPPLE
#define glFinishObjectAPPLE glewFinishObjectAPPLE
#define glGenFencesAPPLE glewGenFencesAPPLE
#define glIsFenceAPPLE glewIsFenceAPPLE
#define glSetFenceAPPLE glewSetFenceAPPLE
#define glTestFenceAPPLE glewTestFenceAPPLE
#define glTestObjectAPPLE glewTestObjectAPPLE

GLEWAPI GLboolean GLEW_APPLE_fence;

#endif /* GL_APPLE_fence */

/* ------------------------- GL_APPLE_float_pixels ------------------------- */

#ifndef GL_APPLE_float_pixels
#define GL_APPLE_float_pixels 1

#define GL_HALF_APPLE 0x140B
#define GL_RGBA_FLOAT32_APPLE 0x8814
#define GL_RGB_FLOAT32_APPLE 0x8815
#define GL_ALPHA_FLOAT32_APPLE 0x8816
#define GL_INTENSITY_FLOAT32_APPLE 0x8817
#define GL_LUMINANCE_FLOAT32_APPLE 0x8818
#define GL_LUMINANCE_ALPHA_FLOAT32_APPLE 0x8819
#define GL_RGBA_FLOAT16_APPLE 0x881A
#define GL_RGB_FLOAT16_APPLE 0x881B
#define GL_ALPHA_FLOAT16_APPLE 0x881C
#define GL_INTENSITY_FLOAT16_APPLE 0x881D
#define GL_LUMINANCE_FLOAT16_APPLE 0x881E
#define GL_LUMINANCE_ALPHA_FLOAT16_APPLE 0x881F
#define GL_COLOR_FLOAT_APPLE 0x8A0F

GLEWAPI GLboolean GLEW_APPLE_float_pixels;

#endif /* GL_APPLE_float_pixels */

/* ------------------------ GL_APPLE_specular_vector ----------------------- */

#ifndef GL_APPLE_specular_vector
#define GL_APPLE_specular_vector 1

#define GL_LIGHT_MODEL_SPECULAR_VECTOR_APPLE 0x85B0

GLEWAPI GLboolean GLEW_APPLE_specular_vector;

#endif /* GL_APPLE_specular_vector */

/* ------------------------- GL_APPLE_texture_range ------------------------ */

#ifndef GL_APPLE_texture_range
#define GL_APPLE_texture_range 1

#define GL_TEXTURE_RANGE_LENGTH_APPLE 0x85B7
#define GL_TEXTURE_RANGE_POINTER_APPLE 0x85B8
#define GL_TEXTURE_STORAGE_HINT_APPLE 0x85BC
#define GL_STORAGE_PRIVATE_APPLE 0x85BD
#define GL_STORAGE_CACHED_APPLE 0x85BE
#define GL_STORAGE_SHARED_APPLE 0x85BF

typedef void (GLAPIENTRY * PFNGLGETTEXPARAMETERPOINTERVAPPLEPROC) (GLenum target, GLenum pname, GLvoid **params);
typedef void (GLAPIENTRY * PFNGLTEXTURERANGEAPPLEPROC) (GLenum target, GLsizei length, GLvoid *pointer);

GLEWAPI PFNGLGETTEXPARAMETERPOINTERVAPPLEPROC glewGetTexParameterPointervAPPLE;
GLEWAPI PFNGLTEXTURERANGEAPPLEPROC glewTextureRangeAPPLE;

#define glGetTexParameterPointervAPPLE glewGetTexParameterPointervAPPLE
#define glTextureRangeAPPLE glewTextureRangeAPPLE

GLEWAPI GLboolean GLEW_APPLE_texture_range;

#endif /* GL_APPLE_texture_range */

/* ------------------------ GL_APPLE_transform_hint ------------------------ */

#ifndef GL_APPLE_transform_hint
#define GL_APPLE_transform_hint 1

#define GL_TRANSFORM_HINT_APPLE 0x85B1

GLEWAPI GLboolean GLEW_APPLE_transform_hint;

#endif /* GL_APPLE_transform_hint */

/* ---------------------- GL_APPLE_vertex_array_object --------------------- */

#ifndef GL_APPLE_vertex_array_object
#define GL_APPLE_vertex_array_object 1

#define GL_VERTEX_ARRAY_BINDING_APPLE 0x85B5

typedef void (GLAPIENTRY * PFNGLBINDVERTEXARRAYAPPLEPROC) (GLuint array);
typedef void (GLAPIENTRY * PFNGLDELETEVERTEXARRAYSAPPLEPROC) (GLsizei n, const GLuint* arrays);
typedef void (GLAPIENTRY * PFNGLGENVERTEXARRAYSAPPLEPROC) (GLsizei n, const GLuint* arrays);
typedef GLboolean (GLAPIENTRY * PFNGLISVERTEXARRAYAPPLEPROC) (GLuint array);

GLEWAPI PFNGLBINDVERTEXARRAYAPPLEPROC glewBindVertexArrayAPPLE;
GLEWAPI PFNGLDELETEVERTEXARRAYSAPPLEPROC glewDeleteVertexArraysAPPLE;
GLEWAPI PFNGLGENVERTEXARRAYSAPPLEPROC glewGenVertexArraysAPPLE;
GLEWAPI PFNGLISVERTEXARRAYAPPLEPROC glewIsVertexArrayAPPLE;

#define glBindVertexArrayAPPLE glewBindVertexArrayAPPLE
#define glDeleteVertexArraysAPPLE glewDeleteVertexArraysAPPLE
#define glGenVertexArraysAPPLE glewGenVertexArraysAPPLE
#define glIsVertexArrayAPPLE glewIsVertexArrayAPPLE

GLEWAPI GLboolean GLEW_APPLE_vertex_array_object;

#endif /* GL_APPLE_vertex_array_object */

/* ---------------------- GL_APPLE_vertex_array_range ---------------------- */

#ifndef GL_APPLE_vertex_array_range
#define GL_APPLE_vertex_array_range 1

#define GL_VERTEX_ARRAY_RANGE_APPLE 0x851D
#define GL_VERTEX_ARRAY_RANGE_LENGTH_APPLE 0x851E
#define GL_VERTEX_ARRAY_STORAGE_HINT_APPLE 0x851F
#define GL_MAX_VERTEX_ARRAY_RANGE_ELEMENT_APPLE 0x8520
#define GL_VERTEX_ARRAY_RANGE_POINTER_APPLE 0x8521
#define GL_STORAGE_CACHED_APPLE 0x85BE
#define GL_STORAGE_SHARED_APPLE 0x85BF

typedef void (GLAPIENTRY * PFNGLFLUSHVERTEXARRAYRANGEAPPLEPROC) (GLsizei length, void* pointer);
typedef void (GLAPIENTRY * PFNGLVERTEXARRAYPARAMETERIAPPLEPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLVERTEXARRAYRANGEAPPLEPROC) (GLsizei length, void* pointer);

GLEWAPI PFNGLFLUSHVERTEXARRAYRANGEAPPLEPROC glewFlushVertexArrayRangeAPPLE;
GLEWAPI PFNGLVERTEXARRAYPARAMETERIAPPLEPROC glewVertexArrayParameteriAPPLE;
GLEWAPI PFNGLVERTEXARRAYRANGEAPPLEPROC glewVertexArrayRangeAPPLE;

#define glFlushVertexArrayRangeAPPLE glewFlushVertexArrayRangeAPPLE
#define glVertexArrayParameteriAPPLE glewVertexArrayParameteriAPPLE
#define glVertexArrayRangeAPPLE glewVertexArrayRangeAPPLE

GLEWAPI GLboolean GLEW_APPLE_vertex_array_range;

#endif /* GL_APPLE_vertex_array_range */

/* --------------------------- GL_APPLE_ycbcr_422 -------------------------- */

#ifndef GL_APPLE_ycbcr_422
#define GL_APPLE_ycbcr_422 1

#define GL_YCBCR_422_APPLE 0x85B9
#define GL_UNSIGNED_SHORT_8_8_APPLE 0x85BA
#define GL_UNSIGNED_SHORT_8_8_REV_APPLE 0x85BB

GLEWAPI GLboolean GLEW_APPLE_ycbcr_422;

#endif /* GL_APPLE_ycbcr_422 */

/* -------------------------- GL_ARB_depth_texture ------------------------- */

#ifndef GL_ARB_depth_texture
#define GL_ARB_depth_texture 1

#define GL_DEPTH_COMPONENT16_ARB 0x81A5
#define GL_DEPTH_COMPONENT24_ARB 0x81A6
#define GL_DEPTH_COMPONENT32_ARB 0x81A7
#define GL_TEXTURE_DEPTH_SIZE_ARB 0x884A
#define GL_DEPTH_TEXTURE_MODE_ARB 0x884B

GLEWAPI GLboolean GLEW_ARB_depth_texture;

#endif /* GL_ARB_depth_texture */

/* ------------------------ GL_ARB_fragment_program ------------------------ */

#ifndef GL_ARB_fragment_program
#define GL_ARB_fragment_program 1

#define GL_FRAGMENT_PROGRAM_ARB 0x8804
#define GL_PROGRAM_ALU_INSTRUCTIONS_ARB 0x8805
#define GL_PROGRAM_TEX_INSTRUCTIONS_ARB 0x8806
#define GL_PROGRAM_TEX_INDIRECTIONS_ARB 0x8807
#define GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB 0x8808
#define GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB 0x8809
#define GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB 0x880A
#define GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB 0x880B
#define GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB 0x880C
#define GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB 0x880D
#define GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB 0x880E
#define GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB 0x880F
#define GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB 0x8810
#define GL_MAX_TEXTURE_COORDS_ARB 0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS_ARB 0x8872

GLEWAPI GLboolean GLEW_ARB_fragment_program;

#endif /* GL_ARB_fragment_program */

/* ------------------------- GL_ARB_fragment_shader ------------------------ */

#ifndef GL_ARB_fragment_shader
#define GL_ARB_fragment_shader 1

#define GL_FRAGMENT_SHADER_ARB 0x8B30
#define GL_MAX_FRAGMENT_UNIFORM_COMPONENTS_ARB 0x8B49

GLEWAPI GLboolean GLEW_ARB_fragment_shader;

#endif /* GL_ARB_fragment_shader */

/* ----------------------------- GL_ARB_imaging ---------------------------- */

#ifndef GL_ARB_imaging
#define GL_ARB_imaging 1

#define GL_CONSTANT_COLOR 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR 0x8002
#define GL_CONSTANT_ALPHA 0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA 0x8004
#define GL_BLEND_COLOR 0x8005
#define GL_FUNC_ADD 0x8006
#define GL_MIN 0x8007
#define GL_MAX 0x8008
#define GL_BLEND_EQUATION 0x8009
#define GL_FUNC_SUBTRACT 0x800A
#define GL_FUNC_REVERSE_SUBTRACT 0x800B
#define GL_CONVOLUTION_1D 0x8010
#define GL_CONVOLUTION_2D 0x8011
#define GL_SEPARABLE_2D 0x8012
#define GL_CONVOLUTION_BORDER_MODE 0x8013
#define GL_CONVOLUTION_FILTER_SCALE 0x8014
#define GL_CONVOLUTION_FILTER_BIAS 0x8015
#define GL_REDUCE 0x8016
#define GL_CONVOLUTION_FORMAT 0x8017
#define GL_CONVOLUTION_WIDTH 0x8018
#define GL_CONVOLUTION_HEIGHT 0x8019
#define GL_MAX_CONVOLUTION_WIDTH 0x801A
#define GL_MAX_CONVOLUTION_HEIGHT 0x801B
#define GL_POST_CONVOLUTION_RED_SCALE 0x801C
#define GL_POST_CONVOLUTION_GREEN_SCALE 0x801D
#define GL_POST_CONVOLUTION_BLUE_SCALE 0x801E
#define GL_POST_CONVOLUTION_ALPHA_SCALE 0x801F
#define GL_POST_CONVOLUTION_RED_BIAS 0x8020
#define GL_POST_CONVOLUTION_GREEN_BIAS 0x8021
#define GL_POST_CONVOLUTION_BLUE_BIAS 0x8022
#define GL_POST_CONVOLUTION_ALPHA_BIAS 0x8023
#define GL_HISTOGRAM 0x8024
#define GL_PROXY_HISTOGRAM 0x8025
#define GL_HISTOGRAM_WIDTH 0x8026
#define GL_HISTOGRAM_FORMAT 0x8027
#define GL_HISTOGRAM_RED_SIZE 0x8028
#define GL_HISTOGRAM_GREEN_SIZE 0x8029
#define GL_HISTOGRAM_BLUE_SIZE 0x802A
#define GL_HISTOGRAM_ALPHA_SIZE 0x802B
#define GL_HISTOGRAM_LUMINANCE_SIZE 0x802C
#define GL_HISTOGRAM_SINK 0x802D
#define GL_MINMAX 0x802E
#define GL_MINMAX_FORMAT 0x802F
#define GL_MINMAX_SINK 0x8030
#define GL_TABLE_TOO_LARGE 0x8031
#define GL_COLOR_MATRIX 0x80B1
#define GL_COLOR_MATRIX_STACK_DEPTH 0x80B2
#define GL_MAX_COLOR_MATRIX_STACK_DEPTH 0x80B3
#define GL_POST_COLOR_MATRIX_RED_SCALE 0x80B4
#define GL_POST_COLOR_MATRIX_GREEN_SCALE 0x80B5
#define GL_POST_COLOR_MATRIX_BLUE_SCALE 0x80B6
#define GL_POST_COLOR_MATRIX_ALPHA_SCALE 0x80B7
#define GL_POST_COLOR_MATRIX_RED_BIAS 0x80B8
#define GL_POST_COLOR_MATRIX_GREEN_BIAS 0x80B9
#define GL_POST_COLOR_MATRIX_BLUE_BIAS 0x80BA
#define GL_POST_COLOR_MATRIX_ALPHA_BIAS 0x80BB
#define GL_COLOR_TABLE 0x80D0
#define GL_POST_CONVOLUTION_COLOR_TABLE 0x80D1
#define GL_POST_COLOR_MATRIX_COLOR_TABLE 0x80D2
#define GL_PROXY_COLOR_TABLE 0x80D3
#define GL_PROXY_POST_CONVOLUTION_COLOR_TABLE 0x80D4
#define GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE 0x80D5
#define GL_COLOR_TABLE_SCALE 0x80D6
#define GL_COLOR_TABLE_BIAS 0x80D7
#define GL_COLOR_TABLE_FORMAT 0x80D8
#define GL_COLOR_TABLE_WIDTH 0x80D9
#define GL_COLOR_TABLE_RED_SIZE 0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE 0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE 0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE 0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE 0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE 0x80DF
#define GL_IGNORE_BORDER 0x8150
#define GL_CONSTANT_BORDER 0x8151
#define GL_WRAP_BORDER 0x8152
#define GL_REPLICATE_BORDER 0x8153
#define GL_CONVOLUTION_BORDER_COLOR 0x8154

typedef void (GLAPIENTRY * PFNGLBLENDCOLORPROC) (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
typedef void (GLAPIENTRY * PFNGLBLENDEQUATIONPROC) (GLenum mode);
typedef void (GLAPIENTRY * PFNGLCOLORSUBTABLEPROC) (GLenum target, GLsizei start, GLsizei count, GLenum format, GLenum type, const GLvoid *data);
typedef void (GLAPIENTRY * PFNGLCOLORTABLEPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *table);
typedef void (GLAPIENTRY * PFNGLCOLORTABLEPARAMETERFVPROC) (GLenum target, GLenum pname, const GLfloat *params);
typedef void (GLAPIENTRY * PFNGLCOLORTABLEPARAMETERIVPROC) (GLenum target, GLenum pname, const GLint *params);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONFILTER1DPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *image);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONFILTER2DPROC) (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *image);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERFPROC) (GLenum target, GLenum pname, GLfloat params);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERFVPROC) (GLenum target, GLenum pname, const GLfloat *params);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERIPROC) (GLenum target, GLenum pname, GLint params);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERIVPROC) (GLenum target, GLenum pname, const GLint *params);
typedef void (GLAPIENTRY * PFNGLCOPYCOLORSUBTABLEPROC) (GLenum target, GLsizei start, GLint x, GLint y, GLsizei width);
typedef void (GLAPIENTRY * PFNGLCOPYCOLORTABLEPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
typedef void (GLAPIENTRY * PFNGLCOPYCONVOLUTIONFILTER1DPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
typedef void (GLAPIENTRY * PFNGLCOPYCONVOLUTIONFILTER2DPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPROC) (GLenum target, GLenum format, GLenum type, GLvoid *table);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETCONVOLUTIONFILTERPROC) (GLenum target, GLenum format, GLenum type, GLvoid *image);
typedef void (GLAPIENTRY * PFNGLGETCONVOLUTIONPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPIENTRY * PFNGLGETCONVOLUTIONPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETHISTOGRAMPROC) (GLenum target, GLboolean reset, GLenum format, GLenum type, GLvoid *values);
typedef void (GLAPIENTRY * PFNGLGETHISTOGRAMPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPIENTRY * PFNGLGETHISTOGRAMPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETMINMAXPROC) (GLenum target, GLboolean reset, GLenum format, GLenum types, GLvoid *values);
typedef void (GLAPIENTRY * PFNGLGETMINMAXPARAMETERFVPROC) (GLenum target, GLenum pname, GLfloat *params);
typedef void (GLAPIENTRY * PFNGLGETMINMAXPARAMETERIVPROC) (GLenum target, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETSEPARABLEFILTERPROC) (GLenum target, GLenum format, GLenum type, GLvoid *row, GLvoid *column, GLvoid *span);
typedef void (GLAPIENTRY * PFNGLHISTOGRAMPROC) (GLenum target, GLsizei width, GLenum internalformat, GLboolean sink);
typedef void (GLAPIENTRY * PFNGLMINMAXPROC) (GLenum target, GLenum internalformat, GLboolean sink);
typedef void (GLAPIENTRY * PFNGLRESETHISTOGRAMPROC) (GLenum target);
typedef void (GLAPIENTRY * PFNGLRESETMINMAXPROC) (GLenum target);
typedef void (GLAPIENTRY * PFNGLSEPARABLEFILTER2DPROC) (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *row, const GLvoid *column);

GLEWAPI PFNGLBLENDCOLORPROC glewBlendColor;
GLEWAPI PFNGLBLENDEQUATIONPROC glewBlendEquation;
GLEWAPI PFNGLCOLORSUBTABLEPROC glewColorSubTable;
GLEWAPI PFNGLCOLORTABLEPROC glewColorTable;
GLEWAPI PFNGLCOLORTABLEPARAMETERFVPROC glewColorTableParameterfv;
GLEWAPI PFNGLCOLORTABLEPARAMETERIVPROC glewColorTableParameteriv;
GLEWAPI PFNGLCONVOLUTIONFILTER1DPROC glewConvolutionFilter1D;
GLEWAPI PFNGLCONVOLUTIONFILTER2DPROC glewConvolutionFilter2D;
GLEWAPI PFNGLCONVOLUTIONPARAMETERFPROC glewConvolutionParameterf;
GLEWAPI PFNGLCONVOLUTIONPARAMETERFVPROC glewConvolutionParameterfv;
GLEWAPI PFNGLCONVOLUTIONPARAMETERIPROC glewConvolutionParameteri;
GLEWAPI PFNGLCONVOLUTIONPARAMETERIVPROC glewConvolutionParameteriv;
GLEWAPI PFNGLCOPYCOLORSUBTABLEPROC glewCopyColorSubTable;
GLEWAPI PFNGLCOPYCOLORTABLEPROC glewCopyColorTable;
GLEWAPI PFNGLCOPYCONVOLUTIONFILTER1DPROC glewCopyConvolutionFilter1D;
GLEWAPI PFNGLCOPYCONVOLUTIONFILTER2DPROC glewCopyConvolutionFilter2D;
GLEWAPI PFNGLGETCOLORTABLEPROC glewGetColorTable;
GLEWAPI PFNGLGETCOLORTABLEPARAMETERFVPROC glewGetColorTableParameterfv;
GLEWAPI PFNGLGETCOLORTABLEPARAMETERIVPROC glewGetColorTableParameteriv;
GLEWAPI PFNGLGETCONVOLUTIONFILTERPROC glewGetConvolutionFilter;
GLEWAPI PFNGLGETCONVOLUTIONPARAMETERFVPROC glewGetConvolutionParameterfv;
GLEWAPI PFNGLGETCONVOLUTIONPARAMETERIVPROC glewGetConvolutionParameteriv;
GLEWAPI PFNGLGETHISTOGRAMPROC glewGetHistogram;
GLEWAPI PFNGLGETHISTOGRAMPARAMETERFVPROC glewGetHistogramParameterfv;
GLEWAPI PFNGLGETHISTOGRAMPARAMETERIVPROC glewGetHistogramParameteriv;
GLEWAPI PFNGLGETMINMAXPROC glewGetMinmax;
GLEWAPI PFNGLGETMINMAXPARAMETERFVPROC glewGetMinmaxParameterfv;
GLEWAPI PFNGLGETMINMAXPARAMETERIVPROC glewGetMinmaxParameteriv;
GLEWAPI PFNGLGETSEPARABLEFILTERPROC glewGetSeparableFilter;
GLEWAPI PFNGLHISTOGRAMPROC glewHistogram;
GLEWAPI PFNGLMINMAXPROC glewMinmax;
GLEWAPI PFNGLRESETHISTOGRAMPROC glewResetHistogram;
GLEWAPI PFNGLRESETMINMAXPROC glewResetMinmax;
GLEWAPI PFNGLSEPARABLEFILTER2DPROC glewSeparableFilter2D;

#define glBlendColor glewBlendColor
#define glBlendEquation glewBlendEquation
#define glColorSubTable glewColorSubTable
#define glColorTable glewColorTable
#define glColorTableParameterfv glewColorTableParameterfv
#define glColorTableParameteriv glewColorTableParameteriv
#define glConvolutionFilter1D glewConvolutionFilter1D
#define glConvolutionFilter2D glewConvolutionFilter2D
#define glConvolutionParameterf glewConvolutionParameterf
#define glConvolutionParameterfv glewConvolutionParameterfv
#define glConvolutionParameteri glewConvolutionParameteri
#define glConvolutionParameteriv glewConvolutionParameteriv
#define glCopyColorSubTable glewCopyColorSubTable
#define glCopyColorTable glewCopyColorTable
#define glCopyConvolutionFilter1D glewCopyConvolutionFilter1D
#define glCopyConvolutionFilter2D glewCopyConvolutionFilter2D
#define glGetColorTable glewGetColorTable
#define glGetColorTableParameterfv glewGetColorTableParameterfv
#define glGetColorTableParameteriv glewGetColorTableParameteriv
#define glGetConvolutionFilter glewGetConvolutionFilter
#define glGetConvolutionParameterfv glewGetConvolutionParameterfv
#define glGetConvolutionParameteriv glewGetConvolutionParameteriv
#define glGetHistogram glewGetHistogram
#define glGetHistogramParameterfv glewGetHistogramParameterfv
#define glGetHistogramParameteriv glewGetHistogramParameteriv
#define glGetMinmax glewGetMinmax
#define glGetMinmaxParameterfv glewGetMinmaxParameterfv
#define glGetMinmaxParameteriv glewGetMinmaxParameteriv
#define glGetSeparableFilter glewGetSeparableFilter
#define glHistogram glewHistogram
#define glMinmax glewMinmax
#define glResetHistogram glewResetHistogram
#define glResetMinmax glewResetMinmax
#define glSeparableFilter2D glewSeparableFilter2D

GLEWAPI GLboolean GLEW_ARB_imaging;

#endif /* GL_ARB_imaging */

/* ------------------------- GL_ARB_matrix_palette ------------------------- */

#ifndef GL_ARB_matrix_palette
#define GL_ARB_matrix_palette 1

#define GL_MATRIX_PALETTE_ARB 0x8840
#define GL_MAX_MATRIX_PALETTE_STACK_DEPTH_ARB 0x8841
#define GL_MAX_PALETTE_MATRICES_ARB 0x8842
#define GL_CURRENT_PALETTE_MATRIX_ARB 0x8843
#define GL_MATRIX_INDEX_ARRAY_ARB 0x8844
#define GL_CURRENT_MATRIX_INDEX_ARB 0x8845
#define GL_MATRIX_INDEX_ARRAY_SIZE_ARB 0x8846
#define GL_MATRIX_INDEX_ARRAY_TYPE_ARB 0x8847
#define GL_MATRIX_INDEX_ARRAY_STRIDE_ARB 0x8848
#define GL_MATRIX_INDEX_ARRAY_POINTER_ARB 0x8849

typedef void (GLAPIENTRY * PFNGLCURRENTPALETTEMATRIXARBPROC) (GLint index);
typedef void (GLAPIENTRY * PFNGLMATRIXINDEXPOINTERARBPROC) (GLint size, GLenum type, GLsizei stride, GLvoid *pointer);
typedef void (GLAPIENTRY * PFNGLMATRIXINDEXUBVARBPROC) (GLint size, GLubyte *indices);
typedef void (GLAPIENTRY * PFNGLMATRIXINDEXUIVARBPROC) (GLint size, GLuint *indices);
typedef void (GLAPIENTRY * PFNGLMATRIXINDEXUSVARBPROC) (GLint size, GLushort *indices);

GLEWAPI PFNGLCURRENTPALETTEMATRIXARBPROC glewCurrentPaletteMatrixARB;
GLEWAPI PFNGLMATRIXINDEXPOINTERARBPROC glewMatrixIndexPointerARB;
GLEWAPI PFNGLMATRIXINDEXUBVARBPROC glewMatrixIndexubvARB;
GLEWAPI PFNGLMATRIXINDEXUIVARBPROC glewMatrixIndexuivARB;
GLEWAPI PFNGLMATRIXINDEXUSVARBPROC glewMatrixIndexusvARB;

#define glCurrentPaletteMatrixARB glewCurrentPaletteMatrixARB
#define glMatrixIndexPointerARB glewMatrixIndexPointerARB
#define glMatrixIndexubvARB glewMatrixIndexubvARB
#define glMatrixIndexuivARB glewMatrixIndexuivARB
#define glMatrixIndexusvARB glewMatrixIndexusvARB

GLEWAPI GLboolean GLEW_ARB_matrix_palette;

#endif /* GL_ARB_matrix_palette */

/* --------------------------- GL_ARB_multisample -------------------------- */

#ifndef GL_ARB_multisample
#define GL_ARB_multisample 1

#define GL_MULTISAMPLE_ARB 0x809D
#define GL_SAMPLE_ALPHA_TO_COVERAGE_ARB 0x809E
#define GL_SAMPLE_ALPHA_TO_ONE_ARB 0x809F
#define GL_SAMPLE_COVERAGE_ARB 0x80A0
#define GL_SAMPLE_BUFFERS_ARB 0x80A8
#define GL_SAMPLES_ARB 0x80A9
#define GL_SAMPLE_COVERAGE_VALUE_ARB 0x80AA
#define GL_SAMPLE_COVERAGE_INVERT_ARB 0x80AB
#define GLX_SAMPLE_BUFFERS_ARB 100000
#define GLX_SAMPLES_ARB 100001
#define GL_MULTISAMPLE_BIT_ARB 0x20000000

typedef void (GLAPIENTRY * PFNGLSAMPLECOVERAGEARBPROC) (GLclampf value, GLboolean invert);

GLEWAPI PFNGLSAMPLECOVERAGEARBPROC glewSampleCoverageARB;

#define glSampleCoverageARB glewSampleCoverageARB

GLEWAPI GLboolean GLEW_ARB_multisample;

#endif /* GL_ARB_multisample */

/* -------------------------- GL_ARB_multitexture -------------------------- */

#ifndef GL_ARB_multitexture
#define GL_ARB_multitexture 1

#define GL_TEXTURE0_ARB 0x84C0
#define GL_TEXTURE1_ARB 0x84C1
#define GL_TEXTURE2_ARB 0x84C2
#define GL_TEXTURE3_ARB 0x84C3
#define GL_TEXTURE4_ARB 0x84C4
#define GL_TEXTURE5_ARB 0x84C5
#define GL_TEXTURE6_ARB 0x84C6
#define GL_TEXTURE7_ARB 0x84C7
#define GL_TEXTURE8_ARB 0x84C8
#define GL_TEXTURE9_ARB 0x84C9
#define GL_TEXTURE10_ARB 0x84CA
#define GL_TEXTURE11_ARB 0x84CB
#define GL_TEXTURE12_ARB 0x84CC
#define GL_TEXTURE13_ARB 0x84CD
#define GL_TEXTURE14_ARB 0x84CE
#define GL_TEXTURE15_ARB 0x84CF
#define GL_TEXTURE16_ARB 0x84D0
#define GL_TEXTURE17_ARB 0x84D1
#define GL_TEXTURE18_ARB 0x84D2
#define GL_TEXTURE19_ARB 0x84D3
#define GL_TEXTURE20_ARB 0x84D4
#define GL_TEXTURE21_ARB 0x84D5
#define GL_TEXTURE22_ARB 0x84D6
#define GL_TEXTURE23_ARB 0x84D7
#define GL_TEXTURE24_ARB 0x84D8
#define GL_TEXTURE25_ARB 0x84D9
#define GL_TEXTURE26_ARB 0x84DA
#define GL_TEXTURE27_ARB 0x84DB
#define GL_TEXTURE28_ARB 0x84DC
#define GL_TEXTURE29_ARB 0x84DD
#define GL_TEXTURE30_ARB 0x84DE
#define GL_TEXTURE31_ARB 0x84DF
#define GL_ACTIVE_TEXTURE_ARB 0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE_ARB 0x84E1
#define GL_MAX_TEXTURE_UNITS_ARB 0x84E2

typedef void (GLAPIENTRY * PFNGLACTIVETEXTUREARBPROC) (GLenum texture);
typedef void (GLAPIENTRY * PFNGLCLIENTACTIVETEXTUREARBPROC) (GLenum texture);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1DARBPROC) (GLenum target, GLdouble s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1FARBPROC) (GLenum target, GLfloat s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1IARBPROC) (GLenum target, GLint s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1SARBPROC) (GLenum target, GLshort s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1SVARBPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2DARBPROC) (GLenum target, GLdouble s, GLdouble t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2FARBPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2IARBPROC) (GLenum target, GLint s, GLint t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2SARBPROC) (GLenum target, GLshort s, GLshort t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2SVARBPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3DARBPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3FARBPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3IARBPROC) (GLenum target, GLint s, GLint t, GLint r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3SARBPROC) (GLenum target, GLshort s, GLshort t, GLshort r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3SVARBPROC) (GLenum target, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4DARBPROC) (GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4DVARBPROC) (GLenum target, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4FARBPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4FVARBPROC) (GLenum target, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4IARBPROC) (GLenum target, GLint s, GLint t, GLint r, GLint q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4IVARBPROC) (GLenum target, const GLint *v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4SARBPROC) (GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4SVARBPROC) (GLenum target, const GLshort *v);

GLEWAPI PFNGLACTIVETEXTUREARBPROC glewActiveTextureARB;
GLEWAPI PFNGLCLIENTACTIVETEXTUREARBPROC glewClientActiveTextureARB;
GLEWAPI PFNGLMULTITEXCOORD1DARBPROC glewMultiTexCoord1dARB;
GLEWAPI PFNGLMULTITEXCOORD1DVARBPROC glewMultiTexCoord1dvARB;
GLEWAPI PFNGLMULTITEXCOORD1FARBPROC glewMultiTexCoord1fARB;
GLEWAPI PFNGLMULTITEXCOORD1FVARBPROC glewMultiTexCoord1fvARB;
GLEWAPI PFNGLMULTITEXCOORD1IARBPROC glewMultiTexCoord1iARB;
GLEWAPI PFNGLMULTITEXCOORD1IVARBPROC glewMultiTexCoord1ivARB;
GLEWAPI PFNGLMULTITEXCOORD1SARBPROC glewMultiTexCoord1sARB;
GLEWAPI PFNGLMULTITEXCOORD1SVARBPROC glewMultiTexCoord1svARB;
GLEWAPI PFNGLMULTITEXCOORD2DARBPROC glewMultiTexCoord2dARB;
GLEWAPI PFNGLMULTITEXCOORD2DVARBPROC glewMultiTexCoord2dvARB;
GLEWAPI PFNGLMULTITEXCOORD2FARBPROC glewMultiTexCoord2fARB;
GLEWAPI PFNGLMULTITEXCOORD2FVARBPROC glewMultiTexCoord2fvARB;
GLEWAPI PFNGLMULTITEXCOORD2IARBPROC glewMultiTexCoord2iARB;
GLEWAPI PFNGLMULTITEXCOORD2IVARBPROC glewMultiTexCoord2ivARB;
GLEWAPI PFNGLMULTITEXCOORD2SARBPROC glewMultiTexCoord2sARB;
GLEWAPI PFNGLMULTITEXCOORD2SVARBPROC glewMultiTexCoord2svARB;
GLEWAPI PFNGLMULTITEXCOORD3DARBPROC glewMultiTexCoord3dARB;
GLEWAPI PFNGLMULTITEXCOORD3DVARBPROC glewMultiTexCoord3dvARB;
GLEWAPI PFNGLMULTITEXCOORD3FARBPROC glewMultiTexCoord3fARB;
GLEWAPI PFNGLMULTITEXCOORD3FVARBPROC glewMultiTexCoord3fvARB;
GLEWAPI PFNGLMULTITEXCOORD3IARBPROC glewMultiTexCoord3iARB;
GLEWAPI PFNGLMULTITEXCOORD3IVARBPROC glewMultiTexCoord3ivARB;
GLEWAPI PFNGLMULTITEXCOORD3SARBPROC glewMultiTexCoord3sARB;
GLEWAPI PFNGLMULTITEXCOORD3SVARBPROC glewMultiTexCoord3svARB;
GLEWAPI PFNGLMULTITEXCOORD4DARBPROC glewMultiTexCoord4dARB;
GLEWAPI PFNGLMULTITEXCOORD4DVARBPROC glewMultiTexCoord4dvARB;
GLEWAPI PFNGLMULTITEXCOORD4FARBPROC glewMultiTexCoord4fARB;
GLEWAPI PFNGLMULTITEXCOORD4FVARBPROC glewMultiTexCoord4fvARB;
GLEWAPI PFNGLMULTITEXCOORD4IARBPROC glewMultiTexCoord4iARB;
GLEWAPI PFNGLMULTITEXCOORD4IVARBPROC glewMultiTexCoord4ivARB;
GLEWAPI PFNGLMULTITEXCOORD4SARBPROC glewMultiTexCoord4sARB;
GLEWAPI PFNGLMULTITEXCOORD4SVARBPROC glewMultiTexCoord4svARB;

#define glActiveTextureARB glewActiveTextureARB
#define glClientActiveTextureARB glewClientActiveTextureARB
#define glMultiTexCoord1dARB glewMultiTexCoord1dARB
#define glMultiTexCoord1dvARB glewMultiTexCoord1dvARB
#define glMultiTexCoord1fARB glewMultiTexCoord1fARB
#define glMultiTexCoord1fvARB glewMultiTexCoord1fvARB
#define glMultiTexCoord1iARB glewMultiTexCoord1iARB
#define glMultiTexCoord1ivARB glewMultiTexCoord1ivARB
#define glMultiTexCoord1sARB glewMultiTexCoord1sARB
#define glMultiTexCoord1svARB glewMultiTexCoord1svARB
#define glMultiTexCoord2dARB glewMultiTexCoord2dARB
#define glMultiTexCoord2dvARB glewMultiTexCoord2dvARB
#define glMultiTexCoord2fARB glewMultiTexCoord2fARB
#define glMultiTexCoord2fvARB glewMultiTexCoord2fvARB
#define glMultiTexCoord2iARB glewMultiTexCoord2iARB
#define glMultiTexCoord2ivARB glewMultiTexCoord2ivARB
#define glMultiTexCoord2sARB glewMultiTexCoord2sARB
#define glMultiTexCoord2svARB glewMultiTexCoord2svARB
#define glMultiTexCoord3dARB glewMultiTexCoord3dARB
#define glMultiTexCoord3dvARB glewMultiTexCoord3dvARB
#define glMultiTexCoord3fARB glewMultiTexCoord3fARB
#define glMultiTexCoord3fvARB glewMultiTexCoord3fvARB
#define glMultiTexCoord3iARB glewMultiTexCoord3iARB
#define glMultiTexCoord3ivARB glewMultiTexCoord3ivARB
#define glMultiTexCoord3sARB glewMultiTexCoord3sARB
#define glMultiTexCoord3svARB glewMultiTexCoord3svARB
#define glMultiTexCoord4dARB glewMultiTexCoord4dARB
#define glMultiTexCoord4dvARB glewMultiTexCoord4dvARB
#define glMultiTexCoord4fARB glewMultiTexCoord4fARB
#define glMultiTexCoord4fvARB glewMultiTexCoord4fvARB
#define glMultiTexCoord4iARB glewMultiTexCoord4iARB
#define glMultiTexCoord4ivARB glewMultiTexCoord4ivARB
#define glMultiTexCoord4sARB glewMultiTexCoord4sARB
#define glMultiTexCoord4svARB glewMultiTexCoord4svARB

GLEWAPI GLboolean GLEW_ARB_multitexture;

#endif /* GL_ARB_multitexture */

/* ------------------------- GL_ARB_occlusion_query ------------------------ */

#ifndef GL_ARB_occlusion_query
#define GL_ARB_occlusion_query 1

#define GL_QUERY_COUNTER_BITS_ARB 0x8864
#define GL_CURRENT_QUERY_ARB 0x8865
#define GL_QUERY_RESULT_ARB 0x8866
#define GL_QUERY_RESULT_AVAILABLE_ARB 0x8867
#define GL_SAMPLES_PASSED_ARB 0x8914

typedef void (GLAPIENTRY * PFNGLBEGINQUERYARBPROC) (GLenum target, GLuint id);
typedef void (GLAPIENTRY * PFNGLDELETEQUERIESARBPROC) (GLsizei n, const GLuint* ids);
typedef void (GLAPIENTRY * PFNGLENDQUERYARBPROC) (GLenum target);
typedef void (GLAPIENTRY * PFNGLGENQUERIESARBPROC) (GLsizei n, GLuint* ids);
typedef void (GLAPIENTRY * PFNGLGETQUERYOBJECTIVARBPROC) (GLuint id, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETQUERYOBJECTUIVARBPROC) (GLuint id, GLenum pname, GLuint* params);
typedef void (GLAPIENTRY * PFNGLGETQUERYIVARBPROC) (GLenum target, GLenum pname, GLint* params);
typedef GLboolean (GLAPIENTRY * PFNGLISQUERYARBPROC) (GLuint id);

GLEWAPI PFNGLBEGINQUERYARBPROC glewBeginQueryARB;
GLEWAPI PFNGLDELETEQUERIESARBPROC glewDeleteQueriesARB;
GLEWAPI PFNGLENDQUERYARBPROC glewEndQueryARB;
GLEWAPI PFNGLGENQUERIESARBPROC glewGenQueriesARB;
GLEWAPI PFNGLGETQUERYOBJECTIVARBPROC glewGetQueryObjectivARB;
GLEWAPI PFNGLGETQUERYOBJECTUIVARBPROC glewGetQueryObjectuivARB;
GLEWAPI PFNGLGETQUERYIVARBPROC glewGetQueryivARB;
GLEWAPI PFNGLISQUERYARBPROC glewIsQueryARB;

#define glBeginQueryARB glewBeginQueryARB
#define glDeleteQueriesARB glewDeleteQueriesARB
#define glEndQueryARB glewEndQueryARB
#define glGenQueriesARB glewGenQueriesARB
#define glGetQueryObjectivARB glewGetQueryObjectivARB
#define glGetQueryObjectuivARB glewGetQueryObjectuivARB
#define glGetQueryivARB glewGetQueryivARB
#define glIsQueryARB glewIsQueryARB

GLEWAPI GLboolean GLEW_ARB_occlusion_query;

#endif /* GL_ARB_occlusion_query */

/* ------------------------ GL_ARB_point_parameters ------------------------ */

#ifndef GL_ARB_point_parameters
#define GL_ARB_point_parameters 1

#define GL_POINT_SIZE_MIN_ARB 0x8126
#define GL_POINT_SIZE_MAX_ARB 0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE_ARB 0x8128
#define GL_POINT_DISTANCE_ATTENUATION_ARB 0x8129

typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERFARBPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERFVARBPROC) (GLenum pname, GLfloat* params);

GLEWAPI PFNGLPOINTPARAMETERFARBPROC glewPointParameterfARB;
GLEWAPI PFNGLPOINTPARAMETERFVARBPROC glewPointParameterfvARB;

#define glPointParameterfARB glewPointParameterfARB
#define glPointParameterfvARB glewPointParameterfvARB

GLEWAPI GLboolean GLEW_ARB_point_parameters;

#endif /* GL_ARB_point_parameters */

/* -------------------------- GL_ARB_point_sprite -------------------------- */

#ifndef GL_ARB_point_sprite
#define GL_ARB_point_sprite 1

#define GL_POINT_SPRITE_ARB 0x8861
#define GL_COORD_REPLACE_ARB 0x8862

GLEWAPI GLboolean GLEW_ARB_point_sprite;

#endif /* GL_ARB_point_sprite */

/* ------------------------- GL_ARB_shader_objects ------------------------- */

#ifndef GL_ARB_shader_objects
#define GL_ARB_shader_objects 1

#define GL_PROGRAM_OBJECT_ARB 0x8B40
#define GL_SHADER_OBJECT_ARB 0x8B48
#define GL_OBJECT_TYPE_ARB 0x8B4E
#define GL_OBJECT_SUBTYPE_ARB 0x8B4F
#define GL_BOOL_ARB 0x8B56
#define GL_BOOL_VEC2_ARB 0x8B57
#define GL_BOOL_VEC3_ARB 0x8B58
#define GL_BOOL_VEC4_ARB 0x8B59
#define GL_OBJECT_DELETE_STATUS_ARB 0x8B80
#define GL_OBJECT_COMPILE_STATUS_ARB 0x8B81
#define GL_OBJECT_LINK_STATUS_ARB 0x8B82
#define GL_OBJECT_VALIDATE_STATUS_ARB 0x8B83
#define GL_OBJECT_INFO_LOG_LENGTH_ARB 0x8B84
#define GL_OBJECT_ATTACHED_OBJECTS_ARB 0x8B85
#define GL_OBJECT_ACTIVE_UNIFORMS_ARB 0x8B86
#define GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB 0x8B87
#define GL_OBJECT_SHADER_SOURCE_LENGTH_ARB 0x8B88

typedef char GLcharARB;
typedef int GLhandleARB;

typedef void (GLAPIENTRY * PFNGLATTACHOBJECTARBPROC) (GLhandleARB containerObj, GLhandleARB obj);
typedef void (GLAPIENTRY * PFNGLCOMPILESHADERARBPROC) (GLhandleARB shaderObj);
typedef GLhandleARB (GLAPIENTRY * PFNGLCREATEPROGRAMOBJECTARBPROC) (void);
typedef GLhandleARB (GLAPIENTRY * PFNGLCREATESHADEROBJECTARBPROC) (GLenum shaderType);
typedef void (GLAPIENTRY * PFNGLDELETEOBJECTARBPROC) (GLhandleARB obj);
typedef void (GLAPIENTRY * PFNGLDETACHOBJECTARBPROC) (GLhandleARB containerObj, GLhandleARB attachedObj);
typedef void (GLAPIENTRY * PFNGLGETACTIVEUNIFORMARBPROC) (GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei* length, GLint *size, GLenum *type, GLcharARB *name);
typedef void (GLAPIENTRY * PFNGLGETATTACHEDOBJECTSARBPROC) (GLhandleARB containerObj, GLsizei maxCount, GLsizei* count, GLhandleARB *obj);
typedef GLhandleARB (GLAPIENTRY * PFNGLGETHANDLEARBPROC) (GLenum pname);
typedef void (GLAPIENTRY * PFNGLGETINFOLOGARBPROC) (GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB *infoLog);
typedef void (GLAPIENTRY * PFNGLGETOBJECTPARAMETERFVARBPROC) (GLhandleARB obj, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETOBJECTPARAMETERIVARBPROC) (GLhandleARB obj, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETSHADERSOURCEARBPROC) (GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB *source);
typedef GLint (GLAPIENTRY * PFNGLGETUNIFORMLOCATIONARBPROC) (GLhandleARB programObj, const GLcharARB* name);
typedef void (GLAPIENTRY * PFNGLGETUNIFORMFVARBPROC) (GLhandleARB programObj, GLint location, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETUNIFORMIVARBPROC) (GLhandleARB programObj, GLint location, GLint* params);
typedef void (GLAPIENTRY * PFNGLLINKPROGRAMARBPROC) (GLhandleARB programObj);
typedef void (GLAPIENTRY * PFNGLSHADERSOURCEARBPROC) (GLhandleARB shaderObj, GLsizei count, const GLcharARB ** string, const GLint *length);
typedef void (GLAPIENTRY * PFNGLUNIFORM1FARBPROC) (GLint location, GLfloat v0);
typedef void (GLAPIENTRY * PFNGLUNIFORM1FVARBPROC) (GLint location, GLsizei count, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM1IARBPROC) (GLint location, GLint v0);
typedef void (GLAPIENTRY * PFNGLUNIFORM1IVARBPROC) (GLint location, GLsizei count, GLint* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM2FARBPROC) (GLint location, GLfloat v0, GLfloat v1);
typedef void (GLAPIENTRY * PFNGLUNIFORM2FVARBPROC) (GLint location, GLsizei count, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM2IARBPROC) (GLint location, GLint v0, GLint v1);
typedef void (GLAPIENTRY * PFNGLUNIFORM2IVARBPROC) (GLint location, GLsizei count, GLint* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM3FARBPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
typedef void (GLAPIENTRY * PFNGLUNIFORM3FVARBPROC) (GLint location, GLsizei count, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM3IARBPROC) (GLint location, GLint v0, GLint v1, GLint v2);
typedef void (GLAPIENTRY * PFNGLUNIFORM3IVARBPROC) (GLint location, GLsizei count, GLint* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM4FARBPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
typedef void (GLAPIENTRY * PFNGLUNIFORM4FVARBPROC) (GLint location, GLsizei count, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUNIFORM4IARBPROC) (GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
typedef void (GLAPIENTRY * PFNGLUNIFORM4IVARBPROC) (GLint location, GLsizei count, GLint* value);
typedef void (GLAPIENTRY * PFNGLUNIFORMMATRIX2FVARBPROC) (GLint location, GLsizei count, GLboolean transpose, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUNIFORMMATRIX3FVARBPROC) (GLint location, GLsizei count, GLboolean transpose, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUNIFORMMATRIX4FVARBPROC) (GLint location, GLsizei count, GLboolean transpose, GLfloat* value);
typedef void (GLAPIENTRY * PFNGLUSEPROGRAMOBJECTARBPROC) (GLhandleARB programObj);
typedef void (GLAPIENTRY * PFNGLVALIDATEPROGRAMARBPROC) (GLhandleARB programObj);

GLEWAPI PFNGLATTACHOBJECTARBPROC glewAttachObjectARB;
GLEWAPI PFNGLCOMPILESHADERARBPROC glewCompileShaderARB;
GLEWAPI PFNGLCREATEPROGRAMOBJECTARBPROC glewCreateProgramObjectARB;
GLEWAPI PFNGLCREATESHADEROBJECTARBPROC glewCreateShaderObjectARB;
GLEWAPI PFNGLDELETEOBJECTARBPROC glewDeleteObjectARB;
GLEWAPI PFNGLDETACHOBJECTARBPROC glewDetachObjectARB;
GLEWAPI PFNGLGETACTIVEUNIFORMARBPROC glewGetActiveUniformARB;
GLEWAPI PFNGLGETATTACHEDOBJECTSARBPROC glewGetAttachedObjectsARB;
GLEWAPI PFNGLGETHANDLEARBPROC glewGetHandleARB;
GLEWAPI PFNGLGETINFOLOGARBPROC glewGetInfoLogARB;
GLEWAPI PFNGLGETOBJECTPARAMETERFVARBPROC glewGetObjectParameterfvARB;
GLEWAPI PFNGLGETOBJECTPARAMETERIVARBPROC glewGetObjectParameterivARB;
GLEWAPI PFNGLGETSHADERSOURCEARBPROC glewGetShaderSourceARB;
GLEWAPI PFNGLGETUNIFORMLOCATIONARBPROC glewGetUniformLocationARB;
GLEWAPI PFNGLGETUNIFORMFVARBPROC glewGetUniformfvARB;
GLEWAPI PFNGLGETUNIFORMIVARBPROC glewGetUniformivARB;
GLEWAPI PFNGLLINKPROGRAMARBPROC glewLinkProgramARB;
GLEWAPI PFNGLSHADERSOURCEARBPROC glewShaderSourceARB;
GLEWAPI PFNGLUNIFORM1FARBPROC glewUniform1fARB;
GLEWAPI PFNGLUNIFORM1FVARBPROC glewUniform1fvARB;
GLEWAPI PFNGLUNIFORM1IARBPROC glewUniform1iARB;
GLEWAPI PFNGLUNIFORM1IVARBPROC glewUniform1ivARB;
GLEWAPI PFNGLUNIFORM2FARBPROC glewUniform2fARB;
GLEWAPI PFNGLUNIFORM2FVARBPROC glewUniform2fvARB;
GLEWAPI PFNGLUNIFORM2IARBPROC glewUniform2iARB;
GLEWAPI PFNGLUNIFORM2IVARBPROC glewUniform2ivARB;
GLEWAPI PFNGLUNIFORM3FARBPROC glewUniform3fARB;
GLEWAPI PFNGLUNIFORM3FVARBPROC glewUniform3fvARB;
GLEWAPI PFNGLUNIFORM3IARBPROC glewUniform3iARB;
GLEWAPI PFNGLUNIFORM3IVARBPROC glewUniform3ivARB;
GLEWAPI PFNGLUNIFORM4FARBPROC glewUniform4fARB;
GLEWAPI PFNGLUNIFORM4FVARBPROC glewUniform4fvARB;
GLEWAPI PFNGLUNIFORM4IARBPROC glewUniform4iARB;
GLEWAPI PFNGLUNIFORM4IVARBPROC glewUniform4ivARB;
GLEWAPI PFNGLUNIFORMMATRIX2FVARBPROC glewUniformMatrix2fvARB;
GLEWAPI PFNGLUNIFORMMATRIX3FVARBPROC glewUniformMatrix3fvARB;
GLEWAPI PFNGLUNIFORMMATRIX4FVARBPROC glewUniformMatrix4fvARB;
GLEWAPI PFNGLUSEPROGRAMOBJECTARBPROC glewUseProgramObjectARB;
GLEWAPI PFNGLVALIDATEPROGRAMARBPROC glewValidateProgramARB;

#define glAttachObjectARB glewAttachObjectARB
#define glCompileShaderARB glewCompileShaderARB
#define glCreateProgramObjectARB glewCreateProgramObjectARB
#define glCreateShaderObjectARB glewCreateShaderObjectARB
#define glDeleteObjectARB glewDeleteObjectARB
#define glDetachObjectARB glewDetachObjectARB
#define glGetActiveUniformARB glewGetActiveUniformARB
#define glGetAttachedObjectsARB glewGetAttachedObjectsARB
#define glGetHandleARB glewGetHandleARB
#define glGetInfoLogARB glewGetInfoLogARB
#define glGetObjectParameterfvARB glewGetObjectParameterfvARB
#define glGetObjectParameterivARB glewGetObjectParameterivARB
#define glGetShaderSourceARB glewGetShaderSourceARB
#define glGetUniformLocationARB glewGetUniformLocationARB
#define glGetUniformfvARB glewGetUniformfvARB
#define glGetUniformivARB glewGetUniformivARB
#define glLinkProgramARB glewLinkProgramARB
#define glShaderSourceARB glewShaderSourceARB
#define glUniform1fARB glewUniform1fARB
#define glUniform1fvARB glewUniform1fvARB
#define glUniform1iARB glewUniform1iARB
#define glUniform1ivARB glewUniform1ivARB
#define glUniform2fARB glewUniform2fARB
#define glUniform2fvARB glewUniform2fvARB
#define glUniform2iARB glewUniform2iARB
#define glUniform2ivARB glewUniform2ivARB
#define glUniform3fARB glewUniform3fARB
#define glUniform3fvARB glewUniform3fvARB
#define glUniform3iARB glewUniform3iARB
#define glUniform3ivARB glewUniform3ivARB
#define glUniform4fARB glewUniform4fARB
#define glUniform4fvARB glewUniform4fvARB
#define glUniform4iARB glewUniform4iARB
#define glUniform4ivARB glewUniform4ivARB
#define glUniformMatrix2fvARB glewUniformMatrix2fvARB
#define glUniformMatrix3fvARB glewUniformMatrix3fvARB
#define glUniformMatrix4fvARB glewUniformMatrix4fvARB
#define glUseProgramObjectARB glewUseProgramObjectARB
#define glValidateProgramARB glewValidateProgramARB

GLEWAPI GLboolean GLEW_ARB_shader_objects;

#endif /* GL_ARB_shader_objects */

/* ---------------------- GL_ARB_shading_language_100 ---------------------- */

#ifndef GL_ARB_shading_language_100
#define GL_ARB_shading_language_100 1

GLEWAPI GLboolean GLEW_ARB_shading_language_100;

#endif /* GL_ARB_shading_language_100 */

/* ----------------------------- GL_ARB_shadow ----------------------------- */

#ifndef GL_ARB_shadow
#define GL_ARB_shadow 1

#define GL_TEXTURE_COMPARE_MODE_ARB 0x884C
#define GL_TEXTURE_COMPARE_FUNC_ARB 0x884D
#define GL_COMPARE_R_TO_TEXTURE_ARB 0x884E

GLEWAPI GLboolean GLEW_ARB_shadow;

#endif /* GL_ARB_shadow */

/* ------------------------- GL_ARB_shadow_ambient ------------------------- */

#ifndef GL_ARB_shadow_ambient
#define GL_ARB_shadow_ambient 1

#define GL_TEXTURE_COMPARE_FAIL_VALUE_ARB 0x80BF

GLEWAPI GLboolean GLEW_ARB_shadow_ambient;

#endif /* GL_ARB_shadow_ambient */

/* ---------------------- GL_ARB_texture_border_clamp ---------------------- */

#ifndef GL_ARB_texture_border_clamp
#define GL_ARB_texture_border_clamp 1

#define GL_CLAMP_TO_BORDER_ARB 0x812D

GLEWAPI GLboolean GLEW_ARB_texture_border_clamp;

#endif /* GL_ARB_texture_border_clamp */

/* ----------------------- GL_ARB_texture_compression ---------------------- */

#ifndef GL_ARB_texture_compression
#define GL_ARB_texture_compression 1

#define GL_COMPRESSED_ALPHA_ARB 0x84E9
#define GL_COMPRESSED_LUMINANCE_ARB 0x84EA
#define GL_COMPRESSED_LUMINANCE_ALPHA_ARB 0x84EB
#define GL_COMPRESSED_INTENSITY_ARB 0x84EC
#define GL_COMPRESSED_RGB_ARB 0x84ED
#define GL_COMPRESSED_RGBA_ARB 0x84EE
#define GL_TEXTURE_COMPRESSION_HINT_ARB 0x84EF
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE_ARB 0x86A0
#define GL_TEXTURE_COMPRESSED_ARB 0x86A1
#define GL_NUM_COMPRESSED_TEXTURE_FORMATS_ARB 0x86A2
#define GL_COMPRESSED_TEXTURE_FORMATS_ARB 0x86A3

typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXIMAGE1DARBPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void* data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXIMAGE2DARBPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void* data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXIMAGE3DARBPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void* data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC) (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data);
typedef void (GLAPIENTRY * PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data);
typedef void (GLAPIENTRY * PFNGLGETCOMPRESSEDTEXIMAGEARBPROC) (GLenum target, GLint lod, void* img);

GLEWAPI PFNGLCOMPRESSEDTEXIMAGE1DARBPROC glewCompressedTexImage1DARB;
GLEWAPI PFNGLCOMPRESSEDTEXIMAGE2DARBPROC glewCompressedTexImage2DARB;
GLEWAPI PFNGLCOMPRESSEDTEXIMAGE3DARBPROC glewCompressedTexImage3DARB;
GLEWAPI PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC glewCompressedTexSubImage1DARB;
GLEWAPI PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC glewCompressedTexSubImage2DARB;
GLEWAPI PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC glewCompressedTexSubImage3DARB;
GLEWAPI PFNGLGETCOMPRESSEDTEXIMAGEARBPROC glewGetCompressedTexImageARB;

#define glCompressedTexImage1DARB glewCompressedTexImage1DARB
#define glCompressedTexImage2DARB glewCompressedTexImage2DARB
#define glCompressedTexImage3DARB glewCompressedTexImage3DARB
#define glCompressedTexSubImage1DARB glewCompressedTexSubImage1DARB
#define glCompressedTexSubImage2DARB glewCompressedTexSubImage2DARB
#define glCompressedTexSubImage3DARB glewCompressedTexSubImage3DARB
#define glGetCompressedTexImageARB glewGetCompressedTexImageARB

GLEWAPI GLboolean GLEW_ARB_texture_compression;

#endif /* GL_ARB_texture_compression */

/* ------------------------ GL_ARB_texture_cube_map ------------------------ */

#ifndef GL_ARB_texture_cube_map
#define GL_ARB_texture_cube_map 1

#define GL_NORMAL_MAP_ARB 0x8511
#define GL_REFLECTION_MAP_ARB 0x8512
#define GL_TEXTURE_CUBE_MAP_ARB 0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP_ARB 0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP_ARB 0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_ARB 0x851C

GLEWAPI GLboolean GLEW_ARB_texture_cube_map;

#endif /* GL_ARB_texture_cube_map */

/* ------------------------- GL_ARB_texture_env_add ------------------------ */

#ifndef GL_ARB_texture_env_add
#define GL_ARB_texture_env_add 1

GLEWAPI GLboolean GLEW_ARB_texture_env_add;

#endif /* GL_ARB_texture_env_add */

/* ----------------------- GL_ARB_texture_env_combine ---------------------- */

#ifndef GL_ARB_texture_env_combine
#define GL_ARB_texture_env_combine 1

#define GL_SUBTRACT_ARB 0x84E7
#define GL_COMBINE_ARB 0x8570
#define GL_COMBINE_RGB_ARB 0x8571
#define GL_COMBINE_ALPHA_ARB 0x8572
#define GL_RGB_SCALE_ARB 0x8573
#define GL_ADD_SIGNED_ARB 0x8574
#define GL_INTERPOLATE_ARB 0x8575
#define GL_CONSTANT_ARB 0x8576
#define GL_PRIMARY_COLOR_ARB 0x8577
#define GL_PREVIOUS_ARB 0x8578
#define GL_SOURCE0_RGB_ARB 0x8580
#define GL_SOURCE1_RGB_ARB 0x8581
#define GL_SOURCE2_RGB_ARB 0x8582
#define GL_SOURCE0_ALPHA_ARB 0x8588
#define GL_SOURCE1_ALPHA_ARB 0x8589
#define GL_SOURCE2_ALPHA_ARB 0x858A
#define GL_OPERAND0_RGB_ARB 0x8590
#define GL_OPERAND1_RGB_ARB 0x8591
#define GL_OPERAND2_RGB_ARB 0x8592
#define GL_OPERAND0_ALPHA_ARB 0x8598
#define GL_OPERAND1_ALPHA_ARB 0x8599
#define GL_OPERAND2_ALPHA_ARB 0x859A

GLEWAPI GLboolean GLEW_ARB_texture_env_combine;

#endif /* GL_ARB_texture_env_combine */

/* ---------------------- GL_ARB_texture_env_crossbar ---------------------- */

#ifndef GL_ARB_texture_env_crossbar
#define GL_ARB_texture_env_crossbar 1

GLEWAPI GLboolean GLEW_ARB_texture_env_crossbar;

#endif /* GL_ARB_texture_env_crossbar */

/* ------------------------ GL_ARB_texture_env_dot3 ------------------------ */

#ifndef GL_ARB_texture_env_dot3
#define GL_ARB_texture_env_dot3 1

#define GL_DOT3_RGB_ARB 0x86AE
#define GL_DOT3_RGBA_ARB 0x86AF

GLEWAPI GLboolean GLEW_ARB_texture_env_dot3;

#endif /* GL_ARB_texture_env_dot3 */

/* --------------------- GL_ARB_texture_mirrored_repeat -------------------- */

#ifndef GL_ARB_texture_mirrored_repeat
#define GL_ARB_texture_mirrored_repeat 1

#define GL_MIRRORED_REPEAT_ARB 0x8370

GLEWAPI GLboolean GLEW_ARB_texture_mirrored_repeat;

#endif /* GL_ARB_texture_mirrored_repeat */

/* -------------------- GL_ARB_texture_non_power_of_two -------------------- */

#ifndef GL_ARB_texture_non_power_of_two
#define GL_ARB_texture_non_power_of_two 1

GLEWAPI GLboolean GLEW_ARB_texture_non_power_of_two;

#endif /* GL_ARB_texture_non_power_of_two */

/* ------------------------ GL_ARB_transpose_matrix ------------------------ */

#ifndef GL_ARB_transpose_matrix
#define GL_ARB_transpose_matrix 1

#define GL_TRANSPOSE_MODELVIEW_MATRIX_ARB 0x84E3
#define GL_TRANSPOSE_PROJECTION_MATRIX_ARB 0x84E4
#define GL_TRANSPOSE_TEXTURE_MATRIX_ARB 0x84E5
#define GL_TRANSPOSE_COLOR_MATRIX_ARB 0x84E6

GLEWAPI GLboolean GLEW_ARB_transpose_matrix;

#endif /* GL_ARB_transpose_matrix */

/* -------------------------- GL_ARB_vertex_blend -------------------------- */

#ifndef GL_ARB_vertex_blend
#define GL_ARB_vertex_blend 1

#define GL_MODELVIEW0_ARB 0x1700
#define GL_MODELVIEW1_ARB 0x850A
#define GL_MAX_VERTEX_UNITS_ARB 0x86A4
#define GL_ACTIVE_VERTEX_UNITS_ARB 0x86A5
#define GL_VERTEX_BLEND_ARB 0x86A7
#define GL_CURRENT_WEIGHT_ARB 0x86A8
#define GL_WEIGHT_ARRAY_TYPE_ARB 0x86A9
#define GL_WEIGHT_ARRAY_STRIDE_ARB 0x86AA
#define GL_WEIGHT_ARRAY_SIZE_ARB 0x86AB
#define GL_WEIGHT_ARRAY_POINTER_ARB 0x86AC
#define GL_WEIGHT_ARRAY_ARB 0x86AD
#define GL_MODELVIEW2_ARB 0x8722
#define GL_MODELVIEW3_ARB 0x8723
#define GL_MODELVIEW4_ARB 0x8724
#define GL_MODELVIEW5_ARB 0x8725
#define GL_MODELVIEW6_ARB 0x8726
#define GL_MODELVIEW7_ARB 0x8727
#define GL_MODELVIEW8_ARB 0x8728
#define GL_MODELVIEW9_ARB 0x8729
#define GL_MODELVIEW10_ARB 0x872A
#define GL_MODELVIEW11_ARB 0x872B
#define GL_MODELVIEW12_ARB 0x872C
#define GL_MODELVIEW13_ARB 0x872D
#define GL_MODELVIEW14_ARB 0x872E
#define GL_MODELVIEW15_ARB 0x872F
#define GL_MODELVIEW16_ARB 0x8730
#define GL_MODELVIEW17_ARB 0x8731
#define GL_MODELVIEW18_ARB 0x8732
#define GL_MODELVIEW19_ARB 0x8733
#define GL_MODELVIEW20_ARB 0x8734
#define GL_MODELVIEW21_ARB 0x8735
#define GL_MODELVIEW22_ARB 0x8736
#define GL_MODELVIEW23_ARB 0x8737
#define GL_MODELVIEW24_ARB 0x8738
#define GL_MODELVIEW25_ARB 0x8739
#define GL_MODELVIEW26_ARB 0x873A
#define GL_MODELVIEW27_ARB 0x873B
#define GL_MODELVIEW28_ARB 0x873C
#define GL_MODELVIEW29_ARB 0x873D
#define GL_MODELVIEW30_ARB 0x873E
#define GL_MODELVIEW31_ARB 0x873F

typedef void (GLAPIENTRY * PFNGLVERTEXBLENDARBPROC) (GLint count);
typedef void (GLAPIENTRY * PFNGLWEIGHTPOINTERARBPROC) (GLint size, GLenum type, GLsizei stride, GLvoid *pointer);
typedef void (GLAPIENTRY * PFNGLWEIGHTBVARBPROC) (GLint size, GLbyte *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTDVARBPROC) (GLint size, GLdouble *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTFVARBPROC) (GLint size, GLfloat *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTIVARBPROC) (GLint size, GLint *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTSVARBPROC) (GLint size, GLshort *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTUBVARBPROC) (GLint size, GLubyte *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTUIVARBPROC) (GLint size, GLuint *weights);
typedef void (GLAPIENTRY * PFNGLWEIGHTUSVARBPROC) (GLint size, GLushort *weights);

GLEWAPI PFNGLVERTEXBLENDARBPROC glewVertexBlendARB;
GLEWAPI PFNGLWEIGHTPOINTERARBPROC glewWeightPointerARB;
GLEWAPI PFNGLWEIGHTBVARBPROC glewWeightbvARB;
GLEWAPI PFNGLWEIGHTDVARBPROC glewWeightdvARB;
GLEWAPI PFNGLWEIGHTFVARBPROC glewWeightfvARB;
GLEWAPI PFNGLWEIGHTIVARBPROC glewWeightivARB;
GLEWAPI PFNGLWEIGHTSVARBPROC glewWeightsvARB;
GLEWAPI PFNGLWEIGHTUBVARBPROC glewWeightubvARB;
GLEWAPI PFNGLWEIGHTUIVARBPROC glewWeightuivARB;
GLEWAPI PFNGLWEIGHTUSVARBPROC glewWeightusvARB;

#define glVertexBlendARB glewVertexBlendARB
#define glWeightPointerARB glewWeightPointerARB
#define glWeightbvARB glewWeightbvARB
#define glWeightdvARB glewWeightdvARB
#define glWeightfvARB glewWeightfvARB
#define glWeightivARB glewWeightivARB
#define glWeightsvARB glewWeightsvARB
#define glWeightubvARB glewWeightubvARB
#define glWeightuivARB glewWeightuivARB
#define glWeightusvARB glewWeightusvARB

GLEWAPI GLboolean GLEW_ARB_vertex_blend;

#endif /* GL_ARB_vertex_blend */

/* ---------------------- GL_ARB_vertex_buffer_object ---------------------- */

#ifndef GL_ARB_vertex_buffer_object
#define GL_ARB_vertex_buffer_object 1

#define GL_BUFFER_SIZE_ARB 0x8764
#define GL_BUFFER_USAGE_ARB 0x8765
#define GL_ARRAY_BUFFER_ARB 0x8892
#define GL_ELEMENT_ARRAY_BUFFER_ARB 0x8893
#define GL_ARRAY_BUFFER_BINDING_ARB 0x8894
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
#define GL_READ_ONLY_ARB 0x88B8
#define GL_WRITE_ONLY_ARB 0x88B9
#define GL_READ_WRITE_ARB 0x88BA
#define GL_BUFFER_ACCESS_ARB 0x88BB
#define GL_BUFFER_MAPPED_ARB 0x88BC
#define GL_BUFFER_MAP_POINTER_ARB 0x88BD
#define GL_STREAM_DRAW_ARB 0x88E0
#define GL_STREAM_READ_ARB 0x88E1
#define GL_STREAM_COPY_ARB 0x88E2
#define GL_STATIC_DRAW_ARB 0x88E4
#define GL_STATIC_READ_ARB 0x88E5
#define GL_STATIC_COPY_ARB 0x88E6
#define GL_DYNAMIC_DRAW_ARB 0x88E8
#define GL_DYNAMIC_READ_ARB 0x88E9
#define GL_DYNAMIC_COPY_ARB 0x88EA

typedef int GLsizeiptrARB;
typedef int GLintptrARB;

typedef void (GLAPIENTRY * PFNGLBINDBUFFERARBPROC) (GLenum target, GLuint buffer);
typedef void (GLAPIENTRY * PFNGLBUFFERDATAARBPROC) (GLenum target, GLsizeiptrARB size, const void* data, GLenum usage);
typedef void (GLAPIENTRY * PFNGLBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset, GLsizeiptrARB size, const void* data);
typedef void (GLAPIENTRY * PFNGLDELETEBUFFERSARBPROC) (GLsizei n, const GLuint* buffers);
typedef void (GLAPIENTRY * PFNGLGENBUFFERSARBPROC) (GLsizei n, GLuint* buffers);
typedef void (GLAPIENTRY * PFNGLGETBUFFERPARAMETERIVARBPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETBUFFERPOINTERVARBPROC) (GLenum target, GLenum pname, GLvoid** params);
typedef void (GLAPIENTRY * PFNGLGETBUFFERSUBDATAARBPROC) (GLenum target, GLintptrARB offset, GLsizeiptrARB size, void* data);
typedef GLboolean (GLAPIENTRY * PFNGLISBUFFERARBPROC) (GLuint buffer);
typedef GLvoid * (GLAPIENTRY * PFNGLMAPBUFFERARBPROC) (GLenum target, GLenum access);
typedef GLboolean (GLAPIENTRY * PFNGLUNMAPBUFFERARBPROC) (GLenum target);

GLEWAPI PFNGLBINDBUFFERARBPROC glewBindBufferARB;
GLEWAPI PFNGLBUFFERDATAARBPROC glewBufferDataARB;
GLEWAPI PFNGLBUFFERSUBDATAARBPROC glewBufferSubDataARB;
GLEWAPI PFNGLDELETEBUFFERSARBPROC glewDeleteBuffersARB;
GLEWAPI PFNGLGENBUFFERSARBPROC glewGenBuffersARB;
GLEWAPI PFNGLGETBUFFERPARAMETERIVARBPROC glewGetBufferParameterivARB;
GLEWAPI PFNGLGETBUFFERPOINTERVARBPROC glewGetBufferPointervARB;
GLEWAPI PFNGLGETBUFFERSUBDATAARBPROC glewGetBufferSubDataARB;
GLEWAPI PFNGLISBUFFERARBPROC glewIsBufferARB;
GLEWAPI PFNGLMAPBUFFERARBPROC glewMapBufferARB;
GLEWAPI PFNGLUNMAPBUFFERARBPROC glewUnmapBufferARB;

#define glBindBufferARB glewBindBufferARB
#define glBufferDataARB glewBufferDataARB
#define glBufferSubDataARB glewBufferSubDataARB
#define glDeleteBuffersARB glewDeleteBuffersARB
#define glGenBuffersARB glewGenBuffersARB
#define glGetBufferParameterivARB glewGetBufferParameterivARB
#define glGetBufferPointervARB glewGetBufferPointervARB
#define glGetBufferSubDataARB glewGetBufferSubDataARB
#define glIsBufferARB glewIsBufferARB
#define glMapBufferARB glewMapBufferARB
#define glUnmapBufferARB glewUnmapBufferARB

GLEWAPI GLboolean GLEW_ARB_vertex_buffer_object;

#endif /* GL_ARB_vertex_buffer_object */

/* ------------------------- GL_ARB_vertex_program ------------------------- */

#ifndef GL_ARB_vertex_program
#define GL_ARB_vertex_program 1

#define GL_COLOR_SUM_ARB 0x8458
#define GL_VERTEX_PROGRAM_ARB 0x8620
#define GL_VERTEX_ATTRIB_ARRAY_ENABLED_ARB 0x8622
#define GL_VERTEX_ATTRIB_ARRAY_SIZE_ARB 0x8623
#define GL_VERTEX_ATTRIB_ARRAY_STRIDE_ARB 0x8624
#define GL_VERTEX_ATTRIB_ARRAY_TYPE_ARB 0x8625
#define GL_CURRENT_VERTEX_ATTRIB_ARB 0x8626
#define GL_PROGRAM_LENGTH_ARB 0x8627
#define GL_PROGRAM_STRING_ARB 0x8628
#define GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB 0x862E
#define GL_MAX_PROGRAM_MATRICES_ARB 0x862F
#define GL_CURRENT_MATRIX_STACK_DEPTH_ARB 0x8640
#define GL_CURRENT_MATRIX_ARB 0x8641
#define GL_VERTEX_PROGRAM_POINT_SIZE_ARB 0x8642
#define GL_VERTEX_PROGRAM_TWO_SIDE_ARB 0x8643
#define GL_VERTEX_ATTRIB_ARRAY_POINTER_ARB 0x8645
#define GL_PROGRAM_ERROR_POSITION_ARB 0x864B
#define GL_PROGRAM_BINDING_ARB 0x8677
#define GL_MAX_VERTEX_ATTRIBS_ARB 0x8869
#define GL_VERTEX_ATTRIB_ARRAY_NORMALIZED_ARB 0x886A
#define GL_PROGRAM_ERROR_STRING_ARB 0x8874
#define GL_PROGRAM_FORMAT_ASCII_ARB 0x8875
#define GL_PROGRAM_FORMAT_ARB 0x8876
#define GL_PROGRAM_INSTRUCTIONS_ARB 0x88A0
#define GL_MAX_PROGRAM_INSTRUCTIONS_ARB 0x88A1
#define GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB 0x88A2
#define GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB 0x88A3
#define GL_PROGRAM_TEMPORARIES_ARB 0x88A4
#define GL_MAX_PROGRAM_TEMPORARIES_ARB 0x88A5
#define GL_PROGRAM_NATIVE_TEMPORARIES_ARB 0x88A6
#define GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB 0x88A7
#define GL_PROGRAM_PARAMETERS_ARB 0x88A8
#define GL_MAX_PROGRAM_PARAMETERS_ARB 0x88A9
#define GL_PROGRAM_NATIVE_PARAMETERS_ARB 0x88AA
#define GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB 0x88AB
#define GL_PROGRAM_ATTRIBS_ARB 0x88AC
#define GL_MAX_PROGRAM_ATTRIBS_ARB 0x88AD
#define GL_PROGRAM_NATIVE_ATTRIBS_ARB 0x88AE
#define GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB 0x88AF
#define GL_PROGRAM_ADDRESS_REGISTERS_ARB 0x88B0
#define GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB 0x88B1
#define GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB 0x88B2
#define GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB 0x88B3
#define GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB 0x88B4
#define GL_MAX_PROGRAM_ENV_PARAMETERS_ARB 0x88B5
#define GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB 0x88B6
#define GL_TRANSPOSE_CURRENT_MATRIX_ARB 0x88B7
#define GL_MATRIX0_ARB 0x88C0
#define GL_MATRIX1_ARB 0x88C1
#define GL_MATRIX2_ARB 0x88C2
#define GL_MATRIX3_ARB 0x88C3
#define GL_MATRIX4_ARB 0x88C4
#define GL_MATRIX5_ARB 0x88C5
#define GL_MATRIX6_ARB 0x88C6
#define GL_MATRIX7_ARB 0x88C7
#define GL_MATRIX8_ARB 0x88C8
#define GL_MATRIX9_ARB 0x88C9
#define GL_MATRIX10_ARB 0x88CA
#define GL_MATRIX11_ARB 0x88CB
#define GL_MATRIX12_ARB 0x88CC
#define GL_MATRIX13_ARB 0x88CD
#define GL_MATRIX14_ARB 0x88CE
#define GL_MATRIX15_ARB 0x88CF
#define GL_MATRIX16_ARB 0x88D0
#define GL_MATRIX17_ARB 0x88D1
#define GL_MATRIX18_ARB 0x88D2
#define GL_MATRIX19_ARB 0x88D3
#define GL_MATRIX20_ARB 0x88D4
#define GL_MATRIX21_ARB 0x88D5
#define GL_MATRIX22_ARB 0x88D6
#define GL_MATRIX23_ARB 0x88D7
#define GL_MATRIX24_ARB 0x88D8
#define GL_MATRIX25_ARB 0x88D9
#define GL_MATRIX26_ARB 0x88DA
#define GL_MATRIX27_ARB 0x88DB
#define GL_MATRIX28_ARB 0x88DC
#define GL_MATRIX29_ARB 0x88DD
#define GL_MATRIX30_ARB 0x88DE
#define GL_MATRIX31_ARB 0x88DF

typedef void (GLAPIENTRY * PFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (GLAPIENTRY * PFNGLDELETEPROGRAMSARBPROC) (GLsizei n, const GLuint* programs);
typedef void (GLAPIENTRY * PFNGLDISABLEVERTEXATTRIBARRAYARBPROC) (GLuint index);
typedef void (GLAPIENTRY * PFNGLENABLEVERTEXATTRIBARRAYARBPROC) (GLuint index);
typedef void (GLAPIENTRY * PFNGLGENPROGRAMSARBPROC) (GLsizei n, GLuint* programs);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMENVPARAMETERDVARBPROC) (GLenum target, GLuint index, GLdouble* params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMENVPARAMETERFVARBPROC) (GLenum target, GLuint index, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC) (GLenum target, GLuint index, GLdouble* params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC) (GLenum target, GLuint index, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMSTRINGARBPROC) (GLenum target, GLenum pname, void* string);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMIVARBPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBPOINTERVARBPROC) (GLuint index, GLenum pname, GLvoid** pointer);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBDVARBPROC) (GLuint index, GLenum pname, GLdouble* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBFVARBPROC) (GLuint index, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBIVARBPROC) (GLuint index, GLenum pname, GLint* params);
typedef GLboolean (GLAPIENTRY * PFNGLISPROGRAMARBPROC) (GLuint program);
typedef void (GLAPIENTRY * PFNGLPROGRAMENVPARAMETER4DARBPROC) (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLPROGRAMENVPARAMETER4DVARBPROC) (GLenum target, GLuint index, const GLdouble* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMENVPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLPROGRAMENVPARAMETER4FVARBPROC) (GLenum target, GLuint index, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMLOCALPARAMETER4DARBPROC) (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLPROGRAMLOCALPARAMETER4DVARBPROC) (GLenum target, GLuint index, const GLdouble* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMLOCALPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLPROGRAMLOCALPARAMETER4FVARBPROC) (GLenum target, GLuint index, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const void* string);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1DARBPROC) (GLuint index, GLdouble x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1DVARBPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1FARBPROC) (GLuint index, GLfloat x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1FVARBPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1SARBPROC) (GLuint index, GLshort x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1SVARBPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2DARBPROC) (GLuint index, GLdouble x, GLdouble y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2DVARBPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2FARBPROC) (GLuint index, GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2FVARBPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2SARBPROC) (GLuint index, GLshort x, GLshort y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2SVARBPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3DARBPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3DVARBPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3FARBPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3FVARBPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3SARBPROC) (GLuint index, GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3SVARBPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NBVARBPROC) (GLuint index, const GLbyte* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NIVARBPROC) (GLuint index, const GLint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NSVARBPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUBARBPROC) (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUBVARBPROC) (GLuint index, const GLubyte* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUIVARBPROC) (GLuint index, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUSVARBPROC) (GLuint index, const GLushort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4BVARBPROC) (GLuint index, const GLbyte* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4DARBPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4DVARBPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4FARBPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4FVARBPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4IVARBPROC) (GLuint index, const GLint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4SARBPROC) (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4SVARBPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4UBVARBPROC) (GLuint index, const GLubyte* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4UIVARBPROC) (GLuint index, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4USVARBPROC) (GLuint index, const GLushort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBPOINTERARBPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);

GLEWAPI PFNGLBINDPROGRAMARBPROC glewBindProgramARB;
GLEWAPI PFNGLDELETEPROGRAMSARBPROC glewDeleteProgramsARB;
GLEWAPI PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glewDisableVertexAttribArrayARB;
GLEWAPI PFNGLENABLEVERTEXATTRIBARRAYARBPROC glewEnableVertexAttribArrayARB;
GLEWAPI PFNGLGENPROGRAMSARBPROC glewGenProgramsARB;
GLEWAPI PFNGLGETPROGRAMENVPARAMETERDVARBPROC glewGetProgramEnvParameterdvARB;
GLEWAPI PFNGLGETPROGRAMENVPARAMETERFVARBPROC glewGetProgramEnvParameterfvARB;
GLEWAPI PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC glewGetProgramLocalParameterdvARB;
GLEWAPI PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC glewGetProgramLocalParameterfvARB;
GLEWAPI PFNGLGETPROGRAMSTRINGARBPROC glewGetProgramStringARB;
GLEWAPI PFNGLGETPROGRAMIVARBPROC glewGetProgramivARB;
GLEWAPI PFNGLGETVERTEXATTRIBPOINTERVARBPROC glewGetVertexAttribPointervARB;
GLEWAPI PFNGLGETVERTEXATTRIBDVARBPROC glewGetVertexAttribdvARB;
GLEWAPI PFNGLGETVERTEXATTRIBFVARBPROC glewGetVertexAttribfvARB;
GLEWAPI PFNGLGETVERTEXATTRIBIVARBPROC glewGetVertexAttribivARB;
GLEWAPI PFNGLISPROGRAMARBPROC glewIsProgramARB;
GLEWAPI PFNGLPROGRAMENVPARAMETER4DARBPROC glewProgramEnvParameter4dARB;
GLEWAPI PFNGLPROGRAMENVPARAMETER4DVARBPROC glewProgramEnvParameter4dvARB;
GLEWAPI PFNGLPROGRAMENVPARAMETER4FARBPROC glewProgramEnvParameter4fARB;
GLEWAPI PFNGLPROGRAMENVPARAMETER4FVARBPROC glewProgramEnvParameter4fvARB;
GLEWAPI PFNGLPROGRAMLOCALPARAMETER4DARBPROC glewProgramLocalParameter4dARB;
GLEWAPI PFNGLPROGRAMLOCALPARAMETER4DVARBPROC glewProgramLocalParameter4dvARB;
GLEWAPI PFNGLPROGRAMLOCALPARAMETER4FARBPROC glewProgramLocalParameter4fARB;
GLEWAPI PFNGLPROGRAMLOCALPARAMETER4FVARBPROC glewProgramLocalParameter4fvARB;
GLEWAPI PFNGLPROGRAMSTRINGARBPROC glewProgramStringARB;
GLEWAPI PFNGLVERTEXATTRIB1DARBPROC glewVertexAttrib1dARB;
GLEWAPI PFNGLVERTEXATTRIB1DVARBPROC glewVertexAttrib1dvARB;
GLEWAPI PFNGLVERTEXATTRIB1FARBPROC glewVertexAttrib1fARB;
GLEWAPI PFNGLVERTEXATTRIB1FVARBPROC glewVertexAttrib1fvARB;
GLEWAPI PFNGLVERTEXATTRIB1SARBPROC glewVertexAttrib1sARB;
GLEWAPI PFNGLVERTEXATTRIB1SVARBPROC glewVertexAttrib1svARB;
GLEWAPI PFNGLVERTEXATTRIB2DARBPROC glewVertexAttrib2dARB;
GLEWAPI PFNGLVERTEXATTRIB2DVARBPROC glewVertexAttrib2dvARB;
GLEWAPI PFNGLVERTEXATTRIB2FARBPROC glewVertexAttrib2fARB;
GLEWAPI PFNGLVERTEXATTRIB2FVARBPROC glewVertexAttrib2fvARB;
GLEWAPI PFNGLVERTEXATTRIB2SARBPROC glewVertexAttrib2sARB;
GLEWAPI PFNGLVERTEXATTRIB2SVARBPROC glewVertexAttrib2svARB;
GLEWAPI PFNGLVERTEXATTRIB3DARBPROC glewVertexAttrib3dARB;
GLEWAPI PFNGLVERTEXATTRIB3DVARBPROC glewVertexAttrib3dvARB;
GLEWAPI PFNGLVERTEXATTRIB3FARBPROC glewVertexAttrib3fARB;
GLEWAPI PFNGLVERTEXATTRIB3FVARBPROC glewVertexAttrib3fvARB;
GLEWAPI PFNGLVERTEXATTRIB3SARBPROC glewVertexAttrib3sARB;
GLEWAPI PFNGLVERTEXATTRIB3SVARBPROC glewVertexAttrib3svARB;
GLEWAPI PFNGLVERTEXATTRIB4NBVARBPROC glewVertexAttrib4NbvARB;
GLEWAPI PFNGLVERTEXATTRIB4NIVARBPROC glewVertexAttrib4NivARB;
GLEWAPI PFNGLVERTEXATTRIB4NSVARBPROC glewVertexAttrib4NsvARB;
GLEWAPI PFNGLVERTEXATTRIB4NUBARBPROC glewVertexAttrib4NubARB;
GLEWAPI PFNGLVERTEXATTRIB4NUBVARBPROC glewVertexAttrib4NubvARB;
GLEWAPI PFNGLVERTEXATTRIB4NUIVARBPROC glewVertexAttrib4NuivARB;
GLEWAPI PFNGLVERTEXATTRIB4NUSVARBPROC glewVertexAttrib4NusvARB;
GLEWAPI PFNGLVERTEXATTRIB4BVARBPROC glewVertexAttrib4bvARB;
GLEWAPI PFNGLVERTEXATTRIB4DARBPROC glewVertexAttrib4dARB;
GLEWAPI PFNGLVERTEXATTRIB4DVARBPROC glewVertexAttrib4dvARB;
GLEWAPI PFNGLVERTEXATTRIB4FARBPROC glewVertexAttrib4fARB;
GLEWAPI PFNGLVERTEXATTRIB4FVARBPROC glewVertexAttrib4fvARB;
GLEWAPI PFNGLVERTEXATTRIB4IVARBPROC glewVertexAttrib4ivARB;
GLEWAPI PFNGLVERTEXATTRIB4SARBPROC glewVertexAttrib4sARB;
GLEWAPI PFNGLVERTEXATTRIB4SVARBPROC glewVertexAttrib4svARB;
GLEWAPI PFNGLVERTEXATTRIB4UBVARBPROC glewVertexAttrib4ubvARB;
GLEWAPI PFNGLVERTEXATTRIB4UIVARBPROC glewVertexAttrib4uivARB;
GLEWAPI PFNGLVERTEXATTRIB4USVARBPROC glewVertexAttrib4usvARB;
GLEWAPI PFNGLVERTEXATTRIBPOINTERARBPROC glewVertexAttribPointerARB;

#define glBindProgramARB glewBindProgramARB
#define glDeleteProgramsARB glewDeleteProgramsARB
#define glDisableVertexAttribArrayARB glewDisableVertexAttribArrayARB
#define glEnableVertexAttribArrayARB glewEnableVertexAttribArrayARB
#define glGenProgramsARB glewGenProgramsARB
#define glGetProgramEnvParameterdvARB glewGetProgramEnvParameterdvARB
#define glGetProgramEnvParameterfvARB glewGetProgramEnvParameterfvARB
#define glGetProgramLocalParameterdvARB glewGetProgramLocalParameterdvARB
#define glGetProgramLocalParameterfvARB glewGetProgramLocalParameterfvARB
#define glGetProgramStringARB glewGetProgramStringARB
#define glGetProgramivARB glewGetProgramivARB
#define glGetVertexAttribPointervARB glewGetVertexAttribPointervARB
#define glGetVertexAttribdvARB glewGetVertexAttribdvARB
#define glGetVertexAttribfvARB glewGetVertexAttribfvARB
#define glGetVertexAttribivARB glewGetVertexAttribivARB
#define glIsProgramARB glewIsProgramARB
#define glProgramEnvParameter4dARB glewProgramEnvParameter4dARB
#define glProgramEnvParameter4dvARB glewProgramEnvParameter4dvARB
#define glProgramEnvParameter4fARB glewProgramEnvParameter4fARB
#define glProgramEnvParameter4fvARB glewProgramEnvParameter4fvARB
#define glProgramLocalParameter4dARB glewProgramLocalParameter4dARB
#define glProgramLocalParameter4dvARB glewProgramLocalParameter4dvARB
#define glProgramLocalParameter4fARB glewProgramLocalParameter4fARB
#define glProgramLocalParameter4fvARB glewProgramLocalParameter4fvARB
#define glProgramStringARB glewProgramStringARB
#define glVertexAttrib1dARB glewVertexAttrib1dARB
#define glVertexAttrib1dvARB glewVertexAttrib1dvARB
#define glVertexAttrib1fARB glewVertexAttrib1fARB
#define glVertexAttrib1fvARB glewVertexAttrib1fvARB
#define glVertexAttrib1sARB glewVertexAttrib1sARB
#define glVertexAttrib1svARB glewVertexAttrib1svARB
#define glVertexAttrib2dARB glewVertexAttrib2dARB
#define glVertexAttrib2dvARB glewVertexAttrib2dvARB
#define glVertexAttrib2fARB glewVertexAttrib2fARB
#define glVertexAttrib2fvARB glewVertexAttrib2fvARB
#define glVertexAttrib2sARB glewVertexAttrib2sARB
#define glVertexAttrib2svARB glewVertexAttrib2svARB
#define glVertexAttrib3dARB glewVertexAttrib3dARB
#define glVertexAttrib3dvARB glewVertexAttrib3dvARB
#define glVertexAttrib3fARB glewVertexAttrib3fARB
#define glVertexAttrib3fvARB glewVertexAttrib3fvARB
#define glVertexAttrib3sARB glewVertexAttrib3sARB
#define glVertexAttrib3svARB glewVertexAttrib3svARB
#define glVertexAttrib4NbvARB glewVertexAttrib4NbvARB
#define glVertexAttrib4NivARB glewVertexAttrib4NivARB
#define glVertexAttrib4NsvARB glewVertexAttrib4NsvARB
#define glVertexAttrib4NubARB glewVertexAttrib4NubARB
#define glVertexAttrib4NubvARB glewVertexAttrib4NubvARB
#define glVertexAttrib4NuivARB glewVertexAttrib4NuivARB
#define glVertexAttrib4NusvARB glewVertexAttrib4NusvARB
#define glVertexAttrib4bvARB glewVertexAttrib4bvARB
#define glVertexAttrib4dARB glewVertexAttrib4dARB
#define glVertexAttrib4dvARB glewVertexAttrib4dvARB
#define glVertexAttrib4fARB glewVertexAttrib4fARB
#define glVertexAttrib4fvARB glewVertexAttrib4fvARB
#define glVertexAttrib4ivARB glewVertexAttrib4ivARB
#define glVertexAttrib4sARB glewVertexAttrib4sARB
#define glVertexAttrib4svARB glewVertexAttrib4svARB
#define glVertexAttrib4ubvARB glewVertexAttrib4ubvARB
#define glVertexAttrib4uivARB glewVertexAttrib4uivARB
#define glVertexAttrib4usvARB glewVertexAttrib4usvARB
#define glVertexAttribPointerARB glewVertexAttribPointerARB

GLEWAPI GLboolean GLEW_ARB_vertex_program;

#endif /* GL_ARB_vertex_program */

/* -------------------------- GL_ARB_vertex_shader ------------------------- */

#ifndef GL_ARB_vertex_shader
#define GL_ARB_vertex_shader 1

#define GL_VERTEX_SHADER_ARB 0x8B31
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS_ARB 0x8B4A
#define GL_MAX_VARYING_FLOATS_ARB 0x8B4B
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB 0x8B4C
#define GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS_ARB 0x8B4D
#define GL_OBJECT_ACTIVE_ATTRIBUTES_ARB 0x8B89
#define GL_OBJECT_ACTIVE_ATTRIBUTE_MAX_LENGTH_ARB 0x8B8A

typedef void (GLAPIENTRY * PFNGLBINDATTRIBLOCATIONARBPROC) (GLhandleARB programObj, GLuint index, const GLcharARB* name);
typedef void (GLAPIENTRY * PFNGLGETACTIVEATTRIBARBPROC) (GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei* length, GLint *size, GLenum *type, GLcharARB *name);
typedef GLint (GLAPIENTRY * PFNGLGETATTRIBLOCATIONARBPROC) (GLhandleARB programObj, const GLcharARB* name);

GLEWAPI PFNGLBINDATTRIBLOCATIONARBPROC glewBindAttribLocationARB;
GLEWAPI PFNGLGETACTIVEATTRIBARBPROC glewGetActiveAttribARB;
GLEWAPI PFNGLGETATTRIBLOCATIONARBPROC glewGetAttribLocationARB;

#define glBindAttribLocationARB glewBindAttribLocationARB
#define glGetActiveAttribARB glewGetActiveAttribARB
#define glGetAttribLocationARB glewGetAttribLocationARB

GLEWAPI GLboolean GLEW_ARB_vertex_shader;

#endif /* GL_ARB_vertex_shader */

/* --------------------------- GL_ARB_window_pos --------------------------- */

#ifndef GL_ARB_window_pos
#define GL_ARB_window_pos 1

typedef void (GLAPIENTRY * PFNGLWINDOWPOS2DARBPROC) (GLdouble x, GLdouble y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2DVARBPROC) (const GLdouble* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2FARBPROC) (GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2FVARBPROC) (const GLfloat* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2IARBPROC) (GLint x, GLint y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2IVARBPROC) (const GLint* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2SARBPROC) (GLshort x, GLshort y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2SVARBPROC) (const GLshort* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3DARBPROC) (GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3DVARBPROC) (const GLdouble* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3FARBPROC) (GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3FVARBPROC) (const GLfloat* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3IARBPROC) (GLint x, GLint y, GLint z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3IVARBPROC) (const GLint* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3SARBPROC) (GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3SVARBPROC) (const GLshort* p);

GLEWAPI PFNGLWINDOWPOS2DARBPROC glewWindowPos2dARB;
GLEWAPI PFNGLWINDOWPOS2DVARBPROC glewWindowPos2dvARB;
GLEWAPI PFNGLWINDOWPOS2FARBPROC glewWindowPos2fARB;
GLEWAPI PFNGLWINDOWPOS2FVARBPROC glewWindowPos2fvARB;
GLEWAPI PFNGLWINDOWPOS2IARBPROC glewWindowPos2iARB;
GLEWAPI PFNGLWINDOWPOS2IVARBPROC glewWindowPos2ivARB;
GLEWAPI PFNGLWINDOWPOS2SARBPROC glewWindowPos2sARB;
GLEWAPI PFNGLWINDOWPOS2SVARBPROC glewWindowPos2svARB;
GLEWAPI PFNGLWINDOWPOS3DARBPROC glewWindowPos3dARB;
GLEWAPI PFNGLWINDOWPOS3DVARBPROC glewWindowPos3dvARB;
GLEWAPI PFNGLWINDOWPOS3FARBPROC glewWindowPos3fARB;
GLEWAPI PFNGLWINDOWPOS3FVARBPROC glewWindowPos3fvARB;
GLEWAPI PFNGLWINDOWPOS3IARBPROC glewWindowPos3iARB;
GLEWAPI PFNGLWINDOWPOS3IVARBPROC glewWindowPos3ivARB;
GLEWAPI PFNGLWINDOWPOS3SARBPROC glewWindowPos3sARB;
GLEWAPI PFNGLWINDOWPOS3SVARBPROC glewWindowPos3svARB;

#define glWindowPos2dARB glewWindowPos2dARB
#define glWindowPos2dvARB glewWindowPos2dvARB
#define glWindowPos2fARB glewWindowPos2fARB
#define glWindowPos2fvARB glewWindowPos2fvARB
#define glWindowPos2iARB glewWindowPos2iARB
#define glWindowPos2ivARB glewWindowPos2ivARB
#define glWindowPos2sARB glewWindowPos2sARB
#define glWindowPos2svARB glewWindowPos2svARB
#define glWindowPos3dARB glewWindowPos3dARB
#define glWindowPos3dvARB glewWindowPos3dvARB
#define glWindowPos3fARB glewWindowPos3fARB
#define glWindowPos3fvARB glewWindowPos3fvARB
#define glWindowPos3iARB glewWindowPos3iARB
#define glWindowPos3ivARB glewWindowPos3ivARB
#define glWindowPos3sARB glewWindowPos3sARB
#define glWindowPos3svARB glewWindowPos3svARB

GLEWAPI GLboolean GLEW_ARB_window_pos;

#endif /* GL_ARB_window_pos */

/* ------------------------- GL_ATIX_point_sprites ------------------------- */

#ifndef GL_ATIX_point_sprites
#define GL_ATIX_point_sprites 1

GLEWAPI GLboolean GLEW_ATIX_point_sprites;

#endif /* GL_ATIX_point_sprites */

/* ---------------------- GL_ATIX_texture_env_combine3 --------------------- */

#ifndef GL_ATIX_texture_env_combine3
#define GL_ATIX_texture_env_combine3 1

#define GL_MODULATE_ADD_ATIX 0x8744
#define GL_MODULATE_SIGNED_ADD_ATIX 0x8745
#define GL_MODULATE_SUBTRACT_ATIX 0x8746

GLEWAPI GLboolean GLEW_ATIX_texture_env_combine3;

#endif /* GL_ATIX_texture_env_combine3 */

/* ----------------------- GL_ATIX_texture_env_route ----------------------- */

#ifndef GL_ATIX_texture_env_route
#define GL_ATIX_texture_env_route 1

#define GL_SECONDARY_COLOR_ATIX 0x8747
#define GL_TEXTURE_OUTPUT_RGB_ATIX 0x8748
#define GL_TEXTURE_OUTPUT_ALPHA_ATIX 0x8749

GLEWAPI GLboolean GLEW_ATIX_texture_env_route;

#endif /* GL_ATIX_texture_env_route */

/* ---------------- GL_ATIX_vertex_shader_output_point_size ---------------- */

#ifndef GL_ATIX_vertex_shader_output_point_size
#define GL_ATIX_vertex_shader_output_point_size 1

#define GL_OUTPUT_POINT_SIZE_ATIX 0x610E

GLEWAPI GLboolean GLEW_ATIX_vertex_shader_output_point_size;

#endif /* GL_ATIX_vertex_shader_output_point_size */

/* -------------------------- GL_ATI_draw_buffers -------------------------- */

#ifndef GL_ATI_draw_buffers
#define GL_ATI_draw_buffers 1

#define GL_MAX_DRAW_BUFFERS_ATI 0x8824
#define GL_DRAW_BUFFER0_ATI 0x8825
#define GL_DRAW_BUFFER1_ATI 0x8826
#define GL_DRAW_BUFFER2_ATI 0x8827
#define GL_DRAW_BUFFER3_ATI 0x8828
#define GL_DRAW_BUFFER4_ATI 0x8829
#define GL_DRAW_BUFFER5_ATI 0x882A
#define GL_DRAW_BUFFER6_ATI 0x882B
#define GL_DRAW_BUFFER7_ATI 0x882C
#define GL_DRAW_BUFFER8_ATI 0x882D
#define GL_DRAW_BUFFER9_ATI 0x882E
#define GL_DRAW_BUFFER10_ATI 0x882F
#define GL_DRAW_BUFFER11_ATI 0x8830
#define GL_DRAW_BUFFER12_ATI 0x8831
#define GL_DRAW_BUFFER13_ATI 0x8832
#define GL_DRAW_BUFFER14_ATI 0x8833
#define GL_DRAW_BUFFER15_ATI 0x8834

typedef void (GLAPIENTRY * PFNGLDRAWBUFFERSATIPROC) (GLsizei n, const GLenum* bufs);

GLEWAPI PFNGLDRAWBUFFERSATIPROC glewDrawBuffersATI;

#define glDrawBuffersATI glewDrawBuffersATI

GLEWAPI GLboolean GLEW_ATI_draw_buffers;

#endif /* GL_ATI_draw_buffers */

/* -------------------------- GL_ATI_element_array ------------------------- */

#ifndef GL_ATI_element_array
#define GL_ATI_element_array 1

#define GL_ELEMENT_ARRAY_ATI 0x8768
#define GL_ELEMENT_ARRAY_TYPE_ATI 0x8769
#define GL_ELEMENT_ARRAY_POINTER_ATI 0x876A

typedef void (GLAPIENTRY * PFNGLDRAWELEMENTARRAYATIPROC) (GLenum mode, GLsizei count);
typedef void (GLAPIENTRY * PFNGLDRAWRANGEELEMENTARRAYATIPROC) (GLenum mode, GLuint start, GLuint end, GLsizei count);
typedef void (GLAPIENTRY * PFNGLELEMENTPOINTERATIPROC) (GLenum type, const void* pointer);

GLEWAPI PFNGLDRAWELEMENTARRAYATIPROC glewDrawElementArrayATI;
GLEWAPI PFNGLDRAWRANGEELEMENTARRAYATIPROC glewDrawRangeElementArrayATI;
GLEWAPI PFNGLELEMENTPOINTERATIPROC glewElementPointerATI;

#define glDrawElementArrayATI glewDrawElementArrayATI
#define glDrawRangeElementArrayATI glewDrawRangeElementArrayATI
#define glElementPointerATI glewElementPointerATI

GLEWAPI GLboolean GLEW_ATI_element_array;

#endif /* GL_ATI_element_array */

/* ------------------------- GL_ATI_envmap_bumpmap ------------------------- */

#ifndef GL_ATI_envmap_bumpmap
#define GL_ATI_envmap_bumpmap 1

#define GL_BUMP_ROT_MATRIX_ATI 0x8775
#define GL_BUMP_ROT_MATRIX_SIZE_ATI 0x8776
#define GL_BUMP_NUM_TEX_UNITS_ATI 0x8777
#define GL_BUMP_TEX_UNITS_ATI 0x8778
#define GL_DUDV_ATI 0x8779
#define GL_DU8DV8_ATI 0x877A
#define GL_BUMP_ENVMAP_ATI 0x877B
#define GL_BUMP_TARGET_ATI 0x877C

typedef void (GLAPIENTRY * PFNGLGETTEXBUMPPARAMETERFVATIPROC) (GLenum pname, GLfloat *param);
typedef void (GLAPIENTRY * PFNGLGETTEXBUMPPARAMETERIVATIPROC) (GLenum pname, GLint *param);
typedef void (GLAPIENTRY * PFNGLTEXBUMPPARAMETERFVATIPROC) (GLenum pname, GLfloat *param);
typedef void (GLAPIENTRY * PFNGLTEXBUMPPARAMETERIVATIPROC) (GLenum pname, GLint *param);

GLEWAPI PFNGLGETTEXBUMPPARAMETERFVATIPROC glewGetTexBumpParameterfvATI;
GLEWAPI PFNGLGETTEXBUMPPARAMETERIVATIPROC glewGetTexBumpParameterivATI;
GLEWAPI PFNGLTEXBUMPPARAMETERFVATIPROC glewTexBumpParameterfvATI;
GLEWAPI PFNGLTEXBUMPPARAMETERIVATIPROC glewTexBumpParameterivATI;

#define glGetTexBumpParameterfvATI glewGetTexBumpParameterfvATI
#define glGetTexBumpParameterivATI glewGetTexBumpParameterivATI
#define glTexBumpParameterfvATI glewTexBumpParameterfvATI
#define glTexBumpParameterivATI glewTexBumpParameterivATI

GLEWAPI GLboolean GLEW_ATI_envmap_bumpmap;

#endif /* GL_ATI_envmap_bumpmap */

/* ------------------------- GL_ATI_fragment_shader ------------------------ */

#ifndef GL_ATI_fragment_shader
#define GL_ATI_fragment_shader 1

#define GL_RED_BIT_ATI 0x00000001
#define GL_2X_BIT_ATI 0x00000001
#define GL_4X_BIT_ATI 0x00000002
#define GL_GREEN_BIT_ATI 0x00000002
#define GL_COMP_BIT_ATI 0x00000002
#define GL_BLUE_BIT_ATI 0x00000004
#define GL_8X_BIT_ATI 0x00000004
#define GL_NEGATE_BIT_ATI 0x00000004
#define GL_BIAS_BIT_ATI 0x00000008
#define GL_HALF_BIT_ATI 0x00000008
#define GL_QUARTER_BIT_ATI 0x00000010
#define GL_EIGHTH_BIT_ATI 0x00000020
#define GL_SATURATE_BIT_ATI 0x00000040
#define GL_FRAGMENT_SHADER_ATI 0x8920
#define GL_REG_0_ATI 0x8921
#define GL_REG_1_ATI 0x8922
#define GL_REG_2_ATI 0x8923
#define GL_REG_3_ATI 0x8924
#define GL_REG_4_ATI 0x8925
#define GL_REG_5_ATI 0x8926
#define GL_CON_0_ATI 0x8941
#define GL_CON_1_ATI 0x8942
#define GL_CON_2_ATI 0x8943
#define GL_CON_3_ATI 0x8944
#define GL_CON_4_ATI 0x8945
#define GL_CON_5_ATI 0x8946
#define GL_CON_6_ATI 0x8947
#define GL_CON_7_ATI 0x8948
#define GL_MOV_ATI 0x8961
#define GL_ADD_ATI 0x8963
#define GL_MUL_ATI 0x8964
#define GL_SUB_ATI 0x8965
#define GL_DOT3_ATI 0x8966
#define GL_DOT4_ATI 0x8967
#define GL_MAD_ATI 0x8968
#define GL_LERP_ATI 0x8969
#define GL_CND_ATI 0x896A
#define GL_CND0_ATI 0x896B
#define GL_DOT2_ADD_ATI 0x896C
#define GL_SECONDARY_INTERPOLATOR_ATI 0x896D
#define GL_SWIZZLE_STR_ATI 0x8976
#define GL_SWIZZLE_STQ_ATI 0x8977
#define GL_SWIZZLE_STR_DR_ATI 0x8978
#define GL_SWIZZLE_STQ_DQ_ATI 0x8979

typedef void (GLAPIENTRY * PFNGLALPHAFRAGMENTOP1ATIPROC) (GLenum op, GLuint dst, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod);
typedef void (GLAPIENTRY * PFNGLALPHAFRAGMENTOP2ATIPROC) (GLenum op, GLuint dst, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod);
typedef void (GLAPIENTRY * PFNGLALPHAFRAGMENTOP3ATIPROC) (GLenum op, GLuint dst, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod, GLuint arg3, GLuint arg3Rep, GLuint arg3Mod);
typedef void (GLAPIENTRY * PFNGLBEGINFRAGMENTSHADERATIPROC) (void);
typedef void (GLAPIENTRY * PFNGLBINDFRAGMENTSHADERATIPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLCOLORFRAGMENTOP1ATIPROC) (GLenum op, GLuint dst, GLuint dstMask, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod);
typedef void (GLAPIENTRY * PFNGLCOLORFRAGMENTOP2ATIPROC) (GLenum op, GLuint dst, GLuint dstMask, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod);
typedef void (GLAPIENTRY * PFNGLCOLORFRAGMENTOP3ATIPROC) (GLenum op, GLuint dst, GLuint dstMask, GLuint dstMod, GLuint arg1, GLuint arg1Rep, GLuint arg1Mod, GLuint arg2, GLuint arg2Rep, GLuint arg2Mod, GLuint arg3, GLuint arg3Rep, GLuint arg3Mod);
typedef void (GLAPIENTRY * PFNGLDELETEFRAGMENTSHADERATIPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLENDFRAGMENTSHADERATIPROC) (void);
typedef GLuint (GLAPIENTRY * PFNGLGENFRAGMENTSHADERSATIPROC) (GLuint range);
typedef void (GLAPIENTRY * PFNGLPASSTEXCOORDATIPROC) (GLuint dst, GLuint coord, GLenum swizzle);
typedef void (GLAPIENTRY * PFNGLSAMPLEMAPATIPROC) (GLuint dst, GLuint interp, GLenum swizzle);
typedef void (GLAPIENTRY * PFNGLSETFRAGMENTSHADERCONSTANTATIPROC) (GLuint dst, const GLfloat* value);

GLEWAPI PFNGLALPHAFRAGMENTOP1ATIPROC glewAlphaFragmentOp1ATI;
GLEWAPI PFNGLALPHAFRAGMENTOP2ATIPROC glewAlphaFragmentOp2ATI;
GLEWAPI PFNGLALPHAFRAGMENTOP3ATIPROC glewAlphaFragmentOp3ATI;
GLEWAPI PFNGLBEGINFRAGMENTSHADERATIPROC glewBeginFragmentShaderATI;
GLEWAPI PFNGLBINDFRAGMENTSHADERATIPROC glewBindFragmentShaderATI;
GLEWAPI PFNGLCOLORFRAGMENTOP1ATIPROC glewColorFragmentOp1ATI;
GLEWAPI PFNGLCOLORFRAGMENTOP2ATIPROC glewColorFragmentOp2ATI;
GLEWAPI PFNGLCOLORFRAGMENTOP3ATIPROC glewColorFragmentOp3ATI;
GLEWAPI PFNGLDELETEFRAGMENTSHADERATIPROC glewDeleteFragmentShaderATI;
GLEWAPI PFNGLENDFRAGMENTSHADERATIPROC glewEndFragmentShaderATI;
GLEWAPI PFNGLGENFRAGMENTSHADERSATIPROC glewGenFragmentShadersATI;
GLEWAPI PFNGLPASSTEXCOORDATIPROC glewPassTexCoordATI;
GLEWAPI PFNGLSAMPLEMAPATIPROC glewSampleMapATI;
GLEWAPI PFNGLSETFRAGMENTSHADERCONSTANTATIPROC glewSetFragmentShaderConstantATI;

#define glAlphaFragmentOp1ATI glewAlphaFragmentOp1ATI
#define glAlphaFragmentOp2ATI glewAlphaFragmentOp2ATI
#define glAlphaFragmentOp3ATI glewAlphaFragmentOp3ATI
#define glBeginFragmentShaderATI glewBeginFragmentShaderATI
#define glBindFragmentShaderATI glewBindFragmentShaderATI
#define glColorFragmentOp1ATI glewColorFragmentOp1ATI
#define glColorFragmentOp2ATI glewColorFragmentOp2ATI
#define glColorFragmentOp3ATI glewColorFragmentOp3ATI
#define glDeleteFragmentShaderATI glewDeleteFragmentShaderATI
#define glEndFragmentShaderATI glewEndFragmentShaderATI
#define glGenFragmentShadersATI glewGenFragmentShadersATI
#define glPassTexCoordATI glewPassTexCoordATI
#define glSampleMapATI glewSampleMapATI
#define glSetFragmentShaderConstantATI glewSetFragmentShaderConstantATI

GLEWAPI GLboolean GLEW_ATI_fragment_shader;

#endif /* GL_ATI_fragment_shader */

/* ------------------------ GL_ATI_map_object_buffer ----------------------- */

#ifndef GL_ATI_map_object_buffer
#define GL_ATI_map_object_buffer 1

typedef void* (GLAPIENTRY * PFNGLMAPOBJECTBUFFERATIPROC) (GLuint buffer);
typedef void (GLAPIENTRY * PFNGLUNMAPOBJECTBUFFERATIPROC) (GLuint buffer);

GLEWAPI PFNGLMAPOBJECTBUFFERATIPROC glewMapObjectBufferATI;
GLEWAPI PFNGLUNMAPOBJECTBUFFERATIPROC glewUnmapObjectBufferATI;

#define glMapObjectBufferATI glewMapObjectBufferATI
#define glUnmapObjectBufferATI glewUnmapObjectBufferATI

GLEWAPI GLboolean GLEW_ATI_map_object_buffer;

#endif /* GL_ATI_map_object_buffer */

/* -------------------------- GL_ATI_pn_triangles -------------------------- */

#ifndef GL_ATI_pn_triangles
#define GL_ATI_pn_triangles 1

#define GL_PN_TRIANGLES_ATI 0x87F0
#define GL_MAX_PN_TRIANGLES_TESSELATION_LEVEL_ATI 0x87F1
#define GL_PN_TRIANGLES_POINT_MODE_ATI 0x87F2
#define GL_PN_TRIANGLES_NORMAL_MODE_ATI 0x87F3
#define GL_PN_TRIANGLES_TESSELATION_LEVEL_ATI 0x87F4
#define GL_PN_TRIANGLES_POINT_MODE_LINEAR_ATI 0x87F5
#define GL_PN_TRIANGLES_POINT_MODE_CUBIC_ATI 0x87F6
#define GL_PN_TRIANGLES_NORMAL_MODE_LINEAR_ATI 0x87F7
#define GL_PN_TRIANGLES_NORMAL_MODE_QUADRATIC_ATI 0x87F8

typedef void (GLAPIENTRY * PFNGLPNTRIANGLESFATIPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLPNTRIANGLESIATIPROC) (GLenum pname, GLint param);

GLEWAPI PFNGLPNTRIANGLESFATIPROC glPNTrianglewesfATI;
GLEWAPI PFNGLPNTRIANGLESIATIPROC glPNTrianglewesiATI;

#define glPNTrianglesfATI glPNTrianglewesfATI
#define glPNTrianglesiATI glPNTrianglewesiATI

GLEWAPI GLboolean GLEW_ATI_pn_triangles;

#endif /* GL_ATI_pn_triangles */

/* ------------------------ GL_ATI_separate_stencil ------------------------ */

#ifndef GL_ATI_separate_stencil
#define GL_ATI_separate_stencil 1

#define GL_STENCIL_BACK_FUNC_ATI 0x8800
#define GL_STENCIL_BACK_FAIL_ATI 0x8801
#define GL_STENCIL_BACK_PASS_DEPTH_FAIL_ATI 0x8802
#define GL_STENCIL_BACK_PASS_DEPTH_PASS_ATI 0x8803

typedef void (GLAPIENTRY * PFNGLSTENCILFUNCSEPARATEATIPROC) (GLenum frontfunc, GLenum backfunc, GLint ref, GLuint mask);
typedef void (GLAPIENTRY * PFNGLSTENCILOPSEPARATEATIPROC) (GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);

GLEWAPI PFNGLSTENCILFUNCSEPARATEATIPROC glewStencilFuncSeparateATI;
GLEWAPI PFNGLSTENCILOPSEPARATEATIPROC glewStencilOpSeparateATI;

#define glStencilFuncSeparateATI glewStencilFuncSeparateATI
#define glStencilOpSeparateATI glewStencilOpSeparateATI

GLEWAPI GLboolean GLEW_ATI_separate_stencil;

#endif /* GL_ATI_separate_stencil */

/* ---------------------- GL_ATI_text_fragment_shader ---------------------- */

#ifndef GL_ATI_text_fragment_shader
#define GL_ATI_text_fragment_shader 1

#define GL_TEXT_FRAGMENT_SHADER_ATI 0x8200

GLEWAPI GLboolean GLEW_ATI_text_fragment_shader;

#endif /* GL_ATI_text_fragment_shader */

/* ---------------------- GL_ATI_texture_env_combine3 ---------------------- */

#ifndef GL_ATI_texture_env_combine3
#define GL_ATI_texture_env_combine3 1

#define GL_MODULATE_ADD_ATI 0x8744
#define GL_MODULATE_SIGNED_ADD_ATI 0x8745
#define GL_MODULATE_SUBTRACT_ATI 0x8746

GLEWAPI GLboolean GLEW_ATI_texture_env_combine3;

#endif /* GL_ATI_texture_env_combine3 */

/* -------------------------- GL_ATI_texture_float ------------------------- */

#ifndef GL_ATI_texture_float
#define GL_ATI_texture_float 1

#define GL_RGBA_FLOAT32_ATI 0x8814
#define GL_RGB_FLOAT32_ATI 0x8815
#define GL_ALPHA_FLOAT32_ATI 0x8816
#define GL_INTENSITY_FLOAT32_ATI 0x8817
#define GL_LUMINANCE_FLOAT32_ATI 0x8818
#define GL_LUMINANCE_ALPHA_FLOAT32_ATI 0x8819
#define GL_RGBA_FLOAT16_ATI 0x881A
#define GL_RGB_FLOAT16_ATI 0x881B
#define GL_ALPHA_FLOAT16_ATI 0x881C
#define GL_INTENSITY_FLOAT16_ATI 0x881D
#define GL_LUMINANCE_FLOAT16_ATI 0x881E
#define GL_LUMINANCE_ALPHA_FLOAT16_ATI 0x881F

GLEWAPI GLboolean GLEW_ATI_texture_float;

#endif /* GL_ATI_texture_float */

/* ----------------------- GL_ATI_texture_mirror_once ---------------------- */

#ifndef GL_ATI_texture_mirror_once
#define GL_ATI_texture_mirror_once 1

#define GL_MIRROR_CLAMP_ATI 0x8742
#define GL_MIRROR_CLAMP_TO_EDGE_ATI 0x8743

GLEWAPI GLboolean GLEW_ATI_texture_mirror_once;

#endif /* GL_ATI_texture_mirror_once */

/* ----------------------- GL_ATI_vertex_array_object ---------------------- */

#ifndef GL_ATI_vertex_array_object
#define GL_ATI_vertex_array_object 1

#define GL_STATIC_ATI 0x8760
#define GL_DYNAMIC_ATI 0x8761
#define GL_PRESERVE_ATI 0x8762
#define GL_DISCARD_ATI 0x8763
#define GL_OBJECT_BUFFER_SIZE_ATI 0x8764
#define GL_OBJECT_BUFFER_USAGE_ATI 0x8765
#define GL_ARRAY_OBJECT_BUFFER_ATI 0x8766
#define GL_ARRAY_OBJECT_OFFSET_ATI 0x8767

typedef void (GLAPIENTRY * PFNGLARRAYOBJECTATIPROC) (GLenum array, GLint size, GLenum type, GLsizei stride, GLuint buffer, GLuint offset);
typedef void (GLAPIENTRY * PFNGLFREEOBJECTBUFFERATIPROC) (GLuint buffer);
typedef void (GLAPIENTRY * PFNGLGETARRAYOBJECTFVATIPROC) (GLenum array, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETARRAYOBJECTIVATIPROC) (GLenum array, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETOBJECTBUFFERFVATIPROC) (GLuint buffer, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETOBJECTBUFFERIVATIPROC) (GLuint buffer, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETVARIANTARRAYOBJECTFVATIPROC) (GLuint id, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETVARIANTARRAYOBJECTIVATIPROC) (GLuint id, GLenum pname, GLint* params);
typedef GLboolean (GLAPIENTRY * PFNGLISOBJECTBUFFERATIPROC) (GLuint buffer);
typedef GLuint (GLAPIENTRY * PFNGLNEWOBJECTBUFFERATIPROC) (GLsizei size, const void* pointer, GLenum usage);
typedef void (GLAPIENTRY * PFNGLUPDATEOBJECTBUFFERATIPROC) (GLuint buffer, GLuint offset, GLsizei size, const void* pointer, GLenum preserve);
typedef void (GLAPIENTRY * PFNGLVARIANTARRAYOBJECTATIPROC) (GLuint id, GLenum type, GLsizei stride, GLuint buffer, GLuint offset);

GLEWAPI PFNGLARRAYOBJECTATIPROC glewArrayObjectATI;
GLEWAPI PFNGLFREEOBJECTBUFFERATIPROC glewFreeObjectBufferATI;
GLEWAPI PFNGLGETARRAYOBJECTFVATIPROC glewGetArrayObjectfvATI;
GLEWAPI PFNGLGETARRAYOBJECTIVATIPROC glewGetArrayObjectivATI;
GLEWAPI PFNGLGETOBJECTBUFFERFVATIPROC glewGetObjectBufferfvATI;
GLEWAPI PFNGLGETOBJECTBUFFERIVATIPROC glewGetObjectBufferivATI;
GLEWAPI PFNGLGETVARIANTARRAYOBJECTFVATIPROC glewGetVariantArrayObjectfvATI;
GLEWAPI PFNGLGETVARIANTARRAYOBJECTIVATIPROC glewGetVariantArrayObjectivATI;
GLEWAPI PFNGLISOBJECTBUFFERATIPROC glewIsObjectBufferATI;
GLEWAPI PFNGLNEWOBJECTBUFFERATIPROC glewNewObjectBufferATI;
GLEWAPI PFNGLUPDATEOBJECTBUFFERATIPROC glewUpdateObjectBufferATI;
GLEWAPI PFNGLVARIANTARRAYOBJECTATIPROC glewVariantArrayObjectATI;

#define glArrayObjectATI glewArrayObjectATI
#define glFreeObjectBufferATI glewFreeObjectBufferATI
#define glGetArrayObjectfvATI glewGetArrayObjectfvATI
#define glGetArrayObjectivATI glewGetArrayObjectivATI
#define glGetObjectBufferfvATI glewGetObjectBufferfvATI
#define glGetObjectBufferivATI glewGetObjectBufferivATI
#define glGetVariantArrayObjectfvATI glewGetVariantArrayObjectfvATI
#define glGetVariantArrayObjectivATI glewGetVariantArrayObjectivATI
#define glIsObjectBufferATI glewIsObjectBufferATI
#define glNewObjectBufferATI glewNewObjectBufferATI
#define glUpdateObjectBufferATI glewUpdateObjectBufferATI
#define glVariantArrayObjectATI glewVariantArrayObjectATI

GLEWAPI GLboolean GLEW_ATI_vertex_array_object;

#endif /* GL_ATI_vertex_array_object */

/* ------------------- GL_ATI_vertex_attrib_array_object ------------------- */

#ifndef GL_ATI_vertex_attrib_array_object
#define GL_ATI_vertex_attrib_array_object 1

typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBARRAYOBJECTFVATIPROC) (GLuint index, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBARRAYOBJECTIVATIPROC) (GLuint index, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBARRAYOBJECTATIPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLuint buffer, GLuint offset);

GLEWAPI PFNGLGETVERTEXATTRIBARRAYOBJECTFVATIPROC glewGetVertexAttribArrayObjectfvATI;
GLEWAPI PFNGLGETVERTEXATTRIBARRAYOBJECTIVATIPROC glewGetVertexAttribArrayObjectivATI;
GLEWAPI PFNGLVERTEXATTRIBARRAYOBJECTATIPROC glewVertexAttribArrayObjectATI;

#define glGetVertexAttribArrayObjectfvATI glewGetVertexAttribArrayObjectfvATI
#define glGetVertexAttribArrayObjectivATI glewGetVertexAttribArrayObjectivATI
#define glVertexAttribArrayObjectATI glewVertexAttribArrayObjectATI

GLEWAPI GLboolean GLEW_ATI_vertex_attrib_array_object;

#endif /* GL_ATI_vertex_attrib_array_object */

/* ------------------------- GL_ATI_vertex_streams ------------------------- */

#ifndef GL_ATI_vertex_streams
#define GL_ATI_vertex_streams 1

#define GL_MAX_VERTEX_STREAMS_ATI 0x876B
#define GL_VERTEX_SOURCE_ATI 0x876C
#define GL_VERTEX_STREAM0_ATI 0x876D
#define GL_VERTEX_STREAM1_ATI 0x876E
#define GL_VERTEX_STREAM2_ATI 0x876F
#define GL_VERTEX_STREAM3_ATI 0x8770
#define GL_VERTEX_STREAM4_ATI 0x8771
#define GL_VERTEX_STREAM5_ATI 0x8772
#define GL_VERTEX_STREAM6_ATI 0x8773
#define GL_VERTEX_STREAM7_ATI 0x8774

typedef void (GLAPIENTRY * PFNGLCLIENTACTIVEVERTEXSTREAMATIPROC) (GLenum stream);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3BATIPROC) (GLenum stream, GLbyte x, GLbyte y, GLbyte z);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3BVATIPROC) (GLenum stream, const GLbyte *v);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3DATIPROC) (GLenum stream, GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3DVATIPROC) (GLenum stream, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3FATIPROC) (GLenum stream, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3FVATIPROC) (GLenum stream, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3IATIPROC) (GLenum stream, GLint x, GLint y, GLint z);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3IVATIPROC) (GLenum stream, const GLint *v);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3SATIPROC) (GLenum stream, GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLNORMALSTREAM3SVATIPROC) (GLenum stream, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLVERTEXBLENDENVFATIPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLVERTEXBLENDENVIATIPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2DATIPROC) (GLenum stream, GLdouble x, GLdouble y);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2DVATIPROC) (GLenum stream, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2FATIPROC) (GLenum stream, GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2FVATIPROC) (GLenum stream, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2IATIPROC) (GLenum stream, GLint x, GLint y);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2IVATIPROC) (GLenum stream, const GLint *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2SATIPROC) (GLenum stream, GLshort x, GLshort y);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM2SVATIPROC) (GLenum stream, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3DATIPROC) (GLenum stream, GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3DVATIPROC) (GLenum stream, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3FATIPROC) (GLenum stream, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3FVATIPROC) (GLenum stream, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3IATIPROC) (GLenum stream, GLint x, GLint y, GLint z);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3IVATIPROC) (GLenum stream, const GLint *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3SATIPROC) (GLenum stream, GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM3SVATIPROC) (GLenum stream, const GLshort *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4DATIPROC) (GLenum stream, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4DVATIPROC) (GLenum stream, const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4FATIPROC) (GLenum stream, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4FVATIPROC) (GLenum stream, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4IATIPROC) (GLenum stream, GLint x, GLint y, GLint z, GLint w);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4IVATIPROC) (GLenum stream, const GLint *v);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4SATIPROC) (GLenum stream, GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (GLAPIENTRY * PFNGLVERTEXSTREAM4SVATIPROC) (GLenum stream, const GLshort *v);

GLEWAPI PFNGLCLIENTACTIVEVERTEXSTREAMATIPROC glewClientActiveVertexStreamATI;
GLEWAPI PFNGLNORMALSTREAM3BATIPROC glewNormalStream3bATI;
GLEWAPI PFNGLNORMALSTREAM3BVATIPROC glewNormalStream3bvATI;
GLEWAPI PFNGLNORMALSTREAM3DATIPROC glewNormalStream3dATI;
GLEWAPI PFNGLNORMALSTREAM3DVATIPROC glewNormalStream3dvATI;
GLEWAPI PFNGLNORMALSTREAM3FATIPROC glewNormalStream3fATI;
GLEWAPI PFNGLNORMALSTREAM3FVATIPROC glewNormalStream3fvATI;
GLEWAPI PFNGLNORMALSTREAM3IATIPROC glewNormalStream3iATI;
GLEWAPI PFNGLNORMALSTREAM3IVATIPROC glewNormalStream3ivATI;
GLEWAPI PFNGLNORMALSTREAM3SATIPROC glewNormalStream3sATI;
GLEWAPI PFNGLNORMALSTREAM3SVATIPROC glewNormalStream3svATI;
GLEWAPI PFNGLVERTEXBLENDENVFATIPROC glewVertexBlendEnvfATI;
GLEWAPI PFNGLVERTEXBLENDENVIATIPROC glewVertexBlendEnviATI;
GLEWAPI PFNGLVERTEXSTREAM2DATIPROC glewVertexStream2dATI;
GLEWAPI PFNGLVERTEXSTREAM2DVATIPROC glewVertexStream2dvATI;
GLEWAPI PFNGLVERTEXSTREAM2FATIPROC glewVertexStream2fATI;
GLEWAPI PFNGLVERTEXSTREAM2FVATIPROC glewVertexStream2fvATI;
GLEWAPI PFNGLVERTEXSTREAM2IATIPROC glewVertexStream2iATI;
GLEWAPI PFNGLVERTEXSTREAM2IVATIPROC glewVertexStream2ivATI;
GLEWAPI PFNGLVERTEXSTREAM2SATIPROC glewVertexStream2sATI;
GLEWAPI PFNGLVERTEXSTREAM2SVATIPROC glewVertexStream2svATI;
GLEWAPI PFNGLVERTEXSTREAM3DATIPROC glewVertexStream3dATI;
GLEWAPI PFNGLVERTEXSTREAM3DVATIPROC glewVertexStream3dvATI;
GLEWAPI PFNGLVERTEXSTREAM3FATIPROC glewVertexStream3fATI;
GLEWAPI PFNGLVERTEXSTREAM3FVATIPROC glewVertexStream3fvATI;
GLEWAPI PFNGLVERTEXSTREAM3IATIPROC glewVertexStream3iATI;
GLEWAPI PFNGLVERTEXSTREAM3IVATIPROC glewVertexStream3ivATI;
GLEWAPI PFNGLVERTEXSTREAM3SATIPROC glewVertexStream3sATI;
GLEWAPI PFNGLVERTEXSTREAM3SVATIPROC glewVertexStream3svATI;
GLEWAPI PFNGLVERTEXSTREAM4DATIPROC glewVertexStream4dATI;
GLEWAPI PFNGLVERTEXSTREAM4DVATIPROC glewVertexStream4dvATI;
GLEWAPI PFNGLVERTEXSTREAM4FATIPROC glewVertexStream4fATI;
GLEWAPI PFNGLVERTEXSTREAM4FVATIPROC glewVertexStream4fvATI;
GLEWAPI PFNGLVERTEXSTREAM4IATIPROC glewVertexStream4iATI;
GLEWAPI PFNGLVERTEXSTREAM4IVATIPROC glewVertexStream4ivATI;
GLEWAPI PFNGLVERTEXSTREAM4SATIPROC glewVertexStream4sATI;
GLEWAPI PFNGLVERTEXSTREAM4SVATIPROC glewVertexStream4svATI;

#define glClientActiveVertexStreamATI glewClientActiveVertexStreamATI
#define glNormalStream3bATI glewNormalStream3bATI
#define glNormalStream3bvATI glewNormalStream3bvATI
#define glNormalStream3dATI glewNormalStream3dATI
#define glNormalStream3dvATI glewNormalStream3dvATI
#define glNormalStream3fATI glewNormalStream3fATI
#define glNormalStream3fvATI glewNormalStream3fvATI
#define glNormalStream3iATI glewNormalStream3iATI
#define glNormalStream3ivATI glewNormalStream3ivATI
#define glNormalStream3sATI glewNormalStream3sATI
#define glNormalStream3svATI glewNormalStream3svATI
#define glVertexBlendEnvfATI glewVertexBlendEnvfATI
#define glVertexBlendEnviATI glewVertexBlendEnviATI
#define glVertexStream2dATI glewVertexStream2dATI
#define glVertexStream2dvATI glewVertexStream2dvATI
#define glVertexStream2fATI glewVertexStream2fATI
#define glVertexStream2fvATI glewVertexStream2fvATI
#define glVertexStream2iATI glewVertexStream2iATI
#define glVertexStream2ivATI glewVertexStream2ivATI
#define glVertexStream2sATI glewVertexStream2sATI
#define glVertexStream2svATI glewVertexStream2svATI
#define glVertexStream3dATI glewVertexStream3dATI
#define glVertexStream3dvATI glewVertexStream3dvATI
#define glVertexStream3fATI glewVertexStream3fATI
#define glVertexStream3fvATI glewVertexStream3fvATI
#define glVertexStream3iATI glewVertexStream3iATI
#define glVertexStream3ivATI glewVertexStream3ivATI
#define glVertexStream3sATI glewVertexStream3sATI
#define glVertexStream3svATI glewVertexStream3svATI
#define glVertexStream4dATI glewVertexStream4dATI
#define glVertexStream4dvATI glewVertexStream4dvATI
#define glVertexStream4fATI glewVertexStream4fATI
#define glVertexStream4fvATI glewVertexStream4fvATI
#define glVertexStream4iATI glewVertexStream4iATI
#define glVertexStream4ivATI glewVertexStream4ivATI
#define glVertexStream4sATI glewVertexStream4sATI
#define glVertexStream4svATI glewVertexStream4svATI

GLEWAPI GLboolean GLEW_ATI_vertex_streams;

#endif /* GL_ATI_vertex_streams */

/* --------------------------- GL_EXT_422_pixels --------------------------- */

#ifndef GL_EXT_422_pixels
#define GL_EXT_422_pixels 1

#define GL_422_EXT 0x80CC
#define GL_422_REV_EXT 0x80CD
#define GL_422_AVERAGE_EXT 0x80CE
#define GL_422_REV_AVERAGE_EXT 0x80CF

GLEWAPI GLboolean GLEW_EXT_422_pixels;

#endif /* GL_EXT_422_pixels */

/* ------------------------------ GL_EXT_abgr ------------------------------ */

#ifndef GL_EXT_abgr
#define GL_EXT_abgr 1

#define GL_ABGR_EXT 0x8000

GLEWAPI GLboolean GLEW_EXT_abgr;

#endif /* GL_EXT_abgr */

/* ------------------------------ GL_EXT_bgra ------------------------------ */

#ifndef GL_EXT_bgra
#define GL_EXT_bgra 1

#define GL_BGR_EXT 0x80E0
#define GL_BGRA_EXT 0x80E1

GLEWAPI GLboolean GLEW_EXT_bgra;

#endif /* GL_EXT_bgra */

/* --------------------------- GL_EXT_blend_color -------------------------- */

#ifndef GL_EXT_blend_color
#define GL_EXT_blend_color 1

#define GL_CONSTANT_COLOR_EXT 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR_EXT 0x8002
#define GL_CONSTANT_ALPHA_EXT 0x8003
#define GL_ONE_MINUS_CONSTANT_ALPHA_EXT 0x8004
#define GL_BLEND_COLOR_EXT 0x8005

typedef void (GLAPIENTRY * PFNGLBLENDCOLOREXTPROC) (GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);

GLEWAPI PFNGLBLENDCOLOREXTPROC glewBlendColorEXT;

#define glBlendColorEXT glewBlendColorEXT

GLEWAPI GLboolean GLEW_EXT_blend_color;

#endif /* GL_EXT_blend_color */

/* ----------------------- GL_EXT_blend_func_separate ---------------------- */

#ifndef GL_EXT_blend_func_separate
#define GL_EXT_blend_func_separate 1

#define GL_BLEND_DST_RGB_EXT 0x80C8
#define GL_BLEND_SRC_RGB_EXT 0x80C9
#define GL_BLEND_DST_ALPHA_EXT 0x80CA
#define GL_BLEND_SRC_ALPHA_EXT 0x80CB

typedef void (GLAPIENTRY * PFNGLBLENDFUNCSEPARATEEXTPROC) (GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);

GLEWAPI PFNGLBLENDFUNCSEPARATEEXTPROC glewBlendFuncSeparateEXT;

#define glBlendFuncSeparateEXT glewBlendFuncSeparateEXT

GLEWAPI GLboolean GLEW_EXT_blend_func_separate;

#endif /* GL_EXT_blend_func_separate */

/* ------------------------- GL_EXT_blend_logic_op ------------------------- */

#ifndef GL_EXT_blend_logic_op
#define GL_EXT_blend_logic_op 1

GLEWAPI GLboolean GLEW_EXT_blend_logic_op;

#endif /* GL_EXT_blend_logic_op */

/* -------------------------- GL_EXT_blend_minmax -------------------------- */

#ifndef GL_EXT_blend_minmax
#define GL_EXT_blend_minmax 1

#define GL_FUNC_ADD_EXT 0x8006
#define GL_MIN_EXT 0x8007
#define GL_MAX_EXT 0x8008
#define GL_BLEND_EQUATION_EXT 0x8009

typedef void (GLAPIENTRY * PFNGLBLENDEQUATIONEXTPROC) (GLenum mode);

GLEWAPI PFNGLBLENDEQUATIONEXTPROC glewBlendEquationEXT;

#define glBlendEquationEXT glewBlendEquationEXT

GLEWAPI GLboolean GLEW_EXT_blend_minmax;

#endif /* GL_EXT_blend_minmax */

/* ------------------------- GL_EXT_blend_subtract ------------------------- */

#ifndef GL_EXT_blend_subtract
#define GL_EXT_blend_subtract 1

#define GL_FUNC_SUBTRACT_EXT 0x800A
#define GL_FUNC_REVERSE_SUBTRACT_EXT 0x800B

GLEWAPI GLboolean GLEW_EXT_blend_subtract;

#endif /* GL_EXT_blend_subtract */

/* ------------------------ GL_EXT_clip_volume_hint ------------------------ */

#ifndef GL_EXT_clip_volume_hint
#define GL_EXT_clip_volume_hint 1

#define GL_CLIP_VOLUME_CLIPPING_HINT_EXT 0x80F0

GLEWAPI GLboolean GLEW_EXT_clip_volume_hint;

#endif /* GL_EXT_clip_volume_hint */

/* ------------------------------ GL_EXT_cmyka ----------------------------- */

#ifndef GL_EXT_cmyka
#define GL_EXT_cmyka 1

#define GL_CMYK_EXT 0x800C
#define GL_CMYKA_EXT 0x800D
#define GL_PACK_CMYK_HINT_EXT 0x800E
#define GL_UNPACK_CMYK_HINT_EXT 0x800F

GLEWAPI GLboolean GLEW_EXT_cmyka;

#endif /* GL_EXT_cmyka */

/* ------------------------- GL_EXT_color_subtable ------------------------- */

#ifndef GL_EXT_color_subtable
#define GL_EXT_color_subtable 1

typedef void (GLAPIENTRY * PFNGLCOLORSUBTABLEEXTPROC) (GLenum target, GLsizei start, GLsizei count, GLenum format, GLenum type, const void* data);
typedef void (GLAPIENTRY * PFNGLCOPYCOLORSUBTABLEEXTPROC) (GLenum target, GLsizei start, GLint x, GLint y, GLsizei width);

GLEWAPI PFNGLCOLORSUBTABLEEXTPROC glewColorSubTableEXT;
GLEWAPI PFNGLCOPYCOLORSUBTABLEEXTPROC glewCopyColorSubTableEXT;

#define glColorSubTableEXT glewColorSubTableEXT
#define glCopyColorSubTableEXT glewCopyColorSubTableEXT

GLEWAPI GLboolean GLEW_EXT_color_subtable;

#endif /* GL_EXT_color_subtable */

/* ---------------------- GL_EXT_compiled_vertex_array --------------------- */

#ifndef GL_EXT_compiled_vertex_array
#define GL_EXT_compiled_vertex_array 1

typedef void (GLAPIENTRY * PFNGLLOCKARRAYSEXTPROC) (GLint first, GLsizei count);
typedef void (GLAPIENTRY * PFNGLUNLOCKARRAYSEXTPROC) (void);

GLEWAPI PFNGLLOCKARRAYSEXTPROC glewLockArraysEXT;
GLEWAPI PFNGLUNLOCKARRAYSEXTPROC glewUnlockArraysEXT;

#define glLockArraysEXT glewLockArraysEXT
#define glUnlockArraysEXT glewUnlockArraysEXT

GLEWAPI GLboolean GLEW_EXT_compiled_vertex_array;

#endif /* GL_EXT_compiled_vertex_array */

/* --------------------------- GL_EXT_convolution -------------------------- */

#ifndef GL_EXT_convolution
#define GL_EXT_convolution 1

#define GL_CONVOLUTION_1D_EXT 0x8010
#define GL_CONVOLUTION_2D_EXT 0x8011
#define GL_SEPARABLE_2D_EXT 0x8012
#define GL_CONVOLUTION_BORDER_MODE_EXT 0x8013
#define GL_CONVOLUTION_FILTER_SCALE_EXT 0x8014
#define GL_CONVOLUTION_FILTER_BIAS_EXT 0x8015
#define GL_REDUCE_EXT 0x8016
#define GL_CONVOLUTION_FORMAT_EXT 0x8017
#define GL_CONVOLUTION_WIDTH_EXT 0x8018
#define GL_CONVOLUTION_HEIGHT_EXT 0x8019
#define GL_MAX_CONVOLUTION_WIDTH_EXT 0x801A
#define GL_MAX_CONVOLUTION_HEIGHT_EXT 0x801B
#define GL_POST_CONVOLUTION_RED_SCALE_EXT 0x801C
#define GL_POST_CONVOLUTION_GREEN_SCALE_EXT 0x801D
#define GL_POST_CONVOLUTION_BLUE_SCALE_EXT 0x801E
#define GL_POST_CONVOLUTION_ALPHA_SCALE_EXT 0x801F
#define GL_POST_CONVOLUTION_RED_BIAS_EXT 0x8020
#define GL_POST_CONVOLUTION_GREEN_BIAS_EXT 0x8021
#define GL_POST_CONVOLUTION_BLUE_BIAS_EXT 0x8022
#define GL_POST_CONVOLUTION_ALPHA_BIAS_EXT 0x8023

typedef void (GLAPIENTRY * PFNGLCONVOLUTIONFILTER1DEXTPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const void* image);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONFILTER2DEXTPROC) (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* image);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERFEXTPROC) (GLenum target, GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERFVEXTPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERIEXTPROC) (GLenum target, GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLCONVOLUTIONPARAMETERIVEXTPROC) (GLenum target, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLCOPYCONVOLUTIONFILTER1DEXTPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
typedef void (GLAPIENTRY * PFNGLCOPYCONVOLUTIONFILTER2DEXTPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (GLAPIENTRY * PFNGLGETCONVOLUTIONFILTEREXTPROC) (GLenum target, GLenum format, GLenum type, void* image);
typedef void (GLAPIENTRY * PFNGLGETCONVOLUTIONPARAMETERFVEXTPROC) (GLenum target, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETCONVOLUTIONPARAMETERIVEXTPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETSEPARABLEFILTEREXTPROC) (GLenum target, GLenum format, GLenum type, void* row, void* column, void* span);
typedef void (GLAPIENTRY * PFNGLSEPARABLEFILTER2DEXTPROC) (GLenum target, GLenum internalformat, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* row, const void* column);

GLEWAPI PFNGLCONVOLUTIONFILTER1DEXTPROC glewConvolutionFilter1DEXT;
GLEWAPI PFNGLCONVOLUTIONFILTER2DEXTPROC glewConvolutionFilter2DEXT;
GLEWAPI PFNGLCONVOLUTIONPARAMETERFEXTPROC glewConvolutionParameterfEXT;
GLEWAPI PFNGLCONVOLUTIONPARAMETERFVEXTPROC glewConvolutionParameterfvEXT;
GLEWAPI PFNGLCONVOLUTIONPARAMETERIEXTPROC glewConvolutionParameteriEXT;
GLEWAPI PFNGLCONVOLUTIONPARAMETERIVEXTPROC glewConvolutionParameterivEXT;
GLEWAPI PFNGLCOPYCONVOLUTIONFILTER1DEXTPROC glewCopyConvolutionFilter1DEXT;
GLEWAPI PFNGLCOPYCONVOLUTIONFILTER2DEXTPROC glewCopyConvolutionFilter2DEXT;
GLEWAPI PFNGLGETCONVOLUTIONFILTEREXTPROC glewGetConvolutionFilterEXT;
GLEWAPI PFNGLGETCONVOLUTIONPARAMETERFVEXTPROC glewGetConvolutionParameterfvEXT;
GLEWAPI PFNGLGETCONVOLUTIONPARAMETERIVEXTPROC glewGetConvolutionParameterivEXT;
GLEWAPI PFNGLGETSEPARABLEFILTEREXTPROC glewGetSeparableFilterEXT;
GLEWAPI PFNGLSEPARABLEFILTER2DEXTPROC glewSeparableFilter2DEXT;

#define glConvolutionFilter1DEXT glewConvolutionFilter1DEXT
#define glConvolutionFilter2DEXT glewConvolutionFilter2DEXT
#define glConvolutionParameterfEXT glewConvolutionParameterfEXT
#define glConvolutionParameterfvEXT glewConvolutionParameterfvEXT
#define glConvolutionParameteriEXT glewConvolutionParameteriEXT
#define glConvolutionParameterivEXT glewConvolutionParameterivEXT
#define glCopyConvolutionFilter1DEXT glewCopyConvolutionFilter1DEXT
#define glCopyConvolutionFilter2DEXT glewCopyConvolutionFilter2DEXT
#define glGetConvolutionFilterEXT glewGetConvolutionFilterEXT
#define glGetConvolutionParameterfvEXT glewGetConvolutionParameterfvEXT
#define glGetConvolutionParameterivEXT glewGetConvolutionParameterivEXT
#define glGetSeparableFilterEXT glewGetSeparableFilterEXT
#define glSeparableFilter2DEXT glewSeparableFilter2DEXT

GLEWAPI GLboolean GLEW_EXT_convolution;

#endif /* GL_EXT_convolution */

/* ------------------------ GL_EXT_coordinate_frame ------------------------ */

#ifndef GL_EXT_coordinate_frame
#define GL_EXT_coordinate_frame 1

#define GL_TANGENT_ARRAY_EXT 0x8439
#define GL_BINORMAL_ARRAY_EXT 0x843A
#define GL_CURRENT_TANGENT_EXT 0x843B
#define GL_CURRENT_BINORMAL_EXT 0x843C
#define GL_TANGENT_ARRAY_TYPE_EXT 0x843E
#define GL_TANGENT_ARRAY_STRIDE_EXT 0x843F
#define GL_BINORMAL_ARRAY_TYPE_EXT 0x8440
#define GL_BINORMAL_ARRAY_STRIDE_EXT 0x8441
#define GL_TANGENT_ARRAY_POINTER_EXT 0x8442
#define GL_BINORMAL_ARRAY_POINTER_EXT 0x8443
#define GL_MAP1_TANGENT_EXT 0x8444
#define GL_MAP2_TANGENT_EXT 0x8445
#define GL_MAP1_BINORMAL_EXT 0x8446
#define GL_MAP2_BINORMAL_EXT 0x8447

typedef void (GLAPIENTRY * PFNGLBINORMALPOINTEREXTPROC) (GLenum type, GLsizei stride, void* pointer);
typedef void (GLAPIENTRY * PFNGLTANGENTPOINTEREXTPROC) (GLenum type, GLsizei stride, void* pointer);

GLEWAPI PFNGLBINORMALPOINTEREXTPROC glewBinormalPointerEXT;
GLEWAPI PFNGLTANGENTPOINTEREXTPROC glewTangentPointerEXT;

#define glBinormalPointerEXT glewBinormalPointerEXT
#define glTangentPointerEXT glewTangentPointerEXT

GLEWAPI GLboolean GLEW_EXT_coordinate_frame;

#endif /* GL_EXT_coordinate_frame */

/* -------------------------- GL_EXT_copy_texture -------------------------- */

#ifndef GL_EXT_copy_texture
#define GL_EXT_copy_texture 1

typedef void (GLAPIENTRY * PFNGLCOPYTEXIMAGE1DEXTPROC) (GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border);
typedef void (GLAPIENTRY * PFNGLCOPYTEXIMAGE2DEXTPROC) (GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
typedef void (GLAPIENTRY * PFNGLCOPYTEXSUBIMAGE1DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
typedef void (GLAPIENTRY * PFNGLCOPYTEXSUBIMAGE2DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
typedef void (GLAPIENTRY * PFNGLCOPYTEXSUBIMAGE3DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);

GLEWAPI PFNGLCOPYTEXIMAGE1DEXTPROC glewCopyTexImage1DEXT;
GLEWAPI PFNGLCOPYTEXIMAGE2DEXTPROC glewCopyTexImage2DEXT;
GLEWAPI PFNGLCOPYTEXSUBIMAGE1DEXTPROC glewCopyTexSubImage1DEXT;
GLEWAPI PFNGLCOPYTEXSUBIMAGE2DEXTPROC glewCopyTexSubImage2DEXT;
GLEWAPI PFNGLCOPYTEXSUBIMAGE3DEXTPROC glewCopyTexSubImage3DEXT;

#define glCopyTexImage1DEXT glewCopyTexImage1DEXT
#define glCopyTexImage2DEXT glewCopyTexImage2DEXT
#define glCopyTexSubImage1DEXT glewCopyTexSubImage1DEXT
#define glCopyTexSubImage2DEXT glewCopyTexSubImage2DEXT
#define glCopyTexSubImage3DEXT glewCopyTexSubImage3DEXT

GLEWAPI GLboolean GLEW_EXT_copy_texture;

#endif /* GL_EXT_copy_texture */

/* --------------------------- GL_EXT_cull_vertex -------------------------- */

#ifndef GL_EXT_cull_vertex
#define GL_EXT_cull_vertex 1

typedef void (GLAPIENTRY * PFNGLCULLPARAMETERDVEXTPROC) (GLenum pname, GLdouble* params);
typedef void (GLAPIENTRY * PFNGLCULLPARAMETERFVEXTPROC) (GLenum pname, GLfloat* params);

GLEWAPI PFNGLCULLPARAMETERDVEXTPROC glewCullParameterdvEXT;
GLEWAPI PFNGLCULLPARAMETERFVEXTPROC glewCullParameterfvEXT;

#define glCullParameterdvEXT glewCullParameterdvEXT
#define glCullParameterfvEXT glewCullParameterfvEXT

GLEWAPI GLboolean GLEW_EXT_cull_vertex;

#endif /* GL_EXT_cull_vertex */

/* ------------------------ GL_EXT_depth_bounds_test ----------------------- */

#ifndef GL_EXT_depth_bounds_test
#define GL_EXT_depth_bounds_test 1

#define GL_DEPTH_BOUNDS_TEST_EXT 0x8890
#define GL_DEPTH_BOUNDS_EXT 0x8891

typedef void (GLAPIENTRY * PFNGLDEPTHBOUNDSEXTPROC) (GLclampd zmin, GLclampd zmax);

GLEWAPI PFNGLDEPTHBOUNDSEXTPROC glewDepthBoundsEXT;

#define glDepthBoundsEXT glewDepthBoundsEXT

GLEWAPI GLboolean GLEW_EXT_depth_bounds_test;

#endif /* GL_EXT_depth_bounds_test */

/* ----------------------- GL_EXT_draw_range_elements ---------------------- */

#ifndef GL_EXT_draw_range_elements
#define GL_EXT_draw_range_elements 1

GLEWAPI GLboolean GLEW_EXT_draw_range_elements;

#endif /* GL_EXT_draw_range_elements */

/* ---------------------------- GL_EXT_fog_coord --------------------------- */

#ifndef GL_EXT_fog_coord
#define GL_EXT_fog_coord 1

#define GL_FOG_COORDINATE_SOURCE_EXT 0x8450
#define GL_FOG_COORDINATE_EXT 0x8451
#define GL_FRAGMENT_DEPTH_EXT 0x8452
#define GL_CURRENT_FOG_COORDINATE_EXT 0x8453
#define GL_FOG_COORDINATE_ARRAY_TYPE_EXT 0x8454
#define GL_FOG_COORDINATE_ARRAY_STRIDE_EXT 0x8455
#define GL_FOG_COORDINATE_ARRAY_POINTER_EXT 0x8456
#define GL_FOG_COORDINATE_ARRAY_EXT 0x8457

typedef void (GLAPIENTRY * PFNGLFOGCOORDPOINTEREXTPROC) (GLenum type, GLsizei stride, const GLvoid *pointer);
typedef void (GLAPIENTRY * PFNGLFOGCOORDDEXTPROC) (GLdouble coord);
typedef void (GLAPIENTRY * PFNGLFOGCOORDDVEXTPROC) (const GLdouble *coord);
typedef void (GLAPIENTRY * PFNGLFOGCOORDFEXTPROC) (GLfloat coord);
typedef void (GLAPIENTRY * PFNGLFOGCOORDFVEXTPROC) (const GLfloat *coord);

GLEWAPI PFNGLFOGCOORDPOINTEREXTPROC glewFogCoordPointerEXT;
GLEWAPI PFNGLFOGCOORDDEXTPROC glewFogCoorddEXT;
GLEWAPI PFNGLFOGCOORDDVEXTPROC glewFogCoorddvEXT;
GLEWAPI PFNGLFOGCOORDFEXTPROC glewFogCoordfEXT;
GLEWAPI PFNGLFOGCOORDFVEXTPROC glewFogCoordfvEXT;

#define glFogCoordPointerEXT glewFogCoordPointerEXT
#define glFogCoorddEXT glewFogCoorddEXT
#define glFogCoorddvEXT glewFogCoorddvEXT
#define glFogCoordfEXT glewFogCoordfEXT
#define glFogCoordfvEXT glewFogCoordfvEXT

GLEWAPI GLboolean GLEW_EXT_fog_coord;

#endif /* GL_EXT_fog_coord */

/* ------------------------ GL_EXT_fragment_lighting ----------------------- */

#ifndef GL_EXT_fragment_lighting
#define GL_EXT_fragment_lighting 1

#define GL_FRAGMENT_LIGHTING_EXT 0x8400
#define GL_FRAGMENT_COLOR_MATERIAL_EXT 0x8401
#define GL_FRAGMENT_COLOR_MATERIAL_FACE_EXT 0x8402
#define GL_FRAGMENT_COLOR_MATERIAL_PARAMETER_EXT 0x8403
#define GL_MAX_FRAGMENT_LIGHTS_EXT 0x8404
#define GL_MAX_ACTIVE_LIGHTS_EXT 0x8405
#define GL_CURRENT_RASTER_NORMAL_EXT 0x8406
#define GL_LIGHT_ENV_MODE_EXT 0x8407
#define GL_FRAGMENT_LIGHT_MODEL_LOCAL_VIEWER_EXT 0x8408
#define GL_FRAGMENT_LIGHT_MODEL_TWO_SIDE_EXT 0x8409
#define GL_FRAGMENT_LIGHT_MODEL_AMBIENT_EXT 0x840A
#define GL_FRAGMENT_LIGHT_MODEL_NORMAL_INTERPOLATION_EXT 0x840B
#define GL_FRAGMENT_LIGHT0_EXT 0x840C
#define GL_FRAGMENT_LIGHT7_EXT 0x8413

typedef void (GLAPIENTRY * PFNGLFRAGMENTCOLORMATERIALEXTPROC) (GLenum face, GLenum mode);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELFEXTPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELFVEXTPROC) (GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELIEXTPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELIVEXTPROC) (GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTFEXTPROC) (GLenum light, GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTFVEXTPROC) (GLenum light, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTIEXTPROC) (GLenum light, GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTIVEXTPROC) (GLenum light, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALFEXTPROC) (GLenum face, GLenum pname, const GLfloat param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALFVEXTPROC) (GLenum face, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALIEXTPROC) (GLenum face, GLenum pname, const GLint param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALIVEXTPROC) (GLenum face, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTLIGHTFVEXTPROC) (GLenum light, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTLIGHTIVEXTPROC) (GLenum light, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTMATERIALFVEXTPROC) (GLenum face, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTMATERIALIVEXTPROC) (GLenum face, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLLIGHTENVIEXTPROC) (GLenum pname, GLint param);

GLEWAPI PFNGLFRAGMENTCOLORMATERIALEXTPROC glewFragmentColorMaterialEXT;
GLEWAPI PFNGLFRAGMENTLIGHTMODELFEXTPROC glewFragmentLightModelfEXT;
GLEWAPI PFNGLFRAGMENTLIGHTMODELFVEXTPROC glewFragmentLightModelfvEXT;
GLEWAPI PFNGLFRAGMENTLIGHTMODELIEXTPROC glewFragmentLightModeliEXT;
GLEWAPI PFNGLFRAGMENTLIGHTMODELIVEXTPROC glewFragmentLightModelivEXT;
GLEWAPI PFNGLFRAGMENTLIGHTFEXTPROC glewFragmentLightfEXT;
GLEWAPI PFNGLFRAGMENTLIGHTFVEXTPROC glewFragmentLightfvEXT;
GLEWAPI PFNGLFRAGMENTLIGHTIEXTPROC glewFragmentLightiEXT;
GLEWAPI PFNGLFRAGMENTLIGHTIVEXTPROC glewFragmentLightivEXT;
GLEWAPI PFNGLFRAGMENTMATERIALFEXTPROC glewFragmentMaterialfEXT;
GLEWAPI PFNGLFRAGMENTMATERIALFVEXTPROC glewFragmentMaterialfvEXT;
GLEWAPI PFNGLFRAGMENTMATERIALIEXTPROC glewFragmentMaterialiEXT;
GLEWAPI PFNGLFRAGMENTMATERIALIVEXTPROC glewFragmentMaterialivEXT;
GLEWAPI PFNGLGETFRAGMENTLIGHTFVEXTPROC glewGetFragmentLightfvEXT;
GLEWAPI PFNGLGETFRAGMENTLIGHTIVEXTPROC glewGetFragmentLightivEXT;
GLEWAPI PFNGLGETFRAGMENTMATERIALFVEXTPROC glewGetFragmentMaterialfvEXT;
GLEWAPI PFNGLGETFRAGMENTMATERIALIVEXTPROC glewGetFragmentMaterialivEXT;
GLEWAPI PFNGLLIGHTENVIEXTPROC glewLightEnviEXT;

#define glFragmentColorMaterialEXT glewFragmentColorMaterialEXT
#define glFragmentLightModelfEXT glewFragmentLightModelfEXT
#define glFragmentLightModelfvEXT glewFragmentLightModelfvEXT
#define glFragmentLightModeliEXT glewFragmentLightModeliEXT
#define glFragmentLightModelivEXT glewFragmentLightModelivEXT
#define glFragmentLightfEXT glewFragmentLightfEXT
#define glFragmentLightfvEXT glewFragmentLightfvEXT
#define glFragmentLightiEXT glewFragmentLightiEXT
#define glFragmentLightivEXT glewFragmentLightivEXT
#define glFragmentMaterialfEXT glewFragmentMaterialfEXT
#define glFragmentMaterialfvEXT glewFragmentMaterialfvEXT
#define glFragmentMaterialiEXT glewFragmentMaterialiEXT
#define glFragmentMaterialivEXT glewFragmentMaterialivEXT
#define glGetFragmentLightfvEXT glewGetFragmentLightfvEXT
#define glGetFragmentLightivEXT glewGetFragmentLightivEXT
#define glGetFragmentMaterialfvEXT glewGetFragmentMaterialfvEXT
#define glGetFragmentMaterialivEXT glewGetFragmentMaterialivEXT
#define glLightEnviEXT glewLightEnviEXT

GLEWAPI GLboolean GLEW_EXT_fragment_lighting;

#endif /* GL_EXT_fragment_lighting */

/* ---------------------------- GL_EXT_histogram --------------------------- */

#ifndef GL_EXT_histogram
#define GL_EXT_histogram 1

#define GL_HISTOGRAM_EXT 0x8024
#define GL_PROXY_HISTOGRAM_EXT 0x8025
#define GL_HISTOGRAM_WIDTH_EXT 0x8026
#define GL_HISTOGRAM_FORMAT_EXT 0x8027
#define GL_HISTOGRAM_RED_SIZE_EXT 0x8028
#define GL_HISTOGRAM_GREEN_SIZE_EXT 0x8029
#define GL_HISTOGRAM_BLUE_SIZE_EXT 0x802A
#define GL_HISTOGRAM_ALPHA_SIZE_EXT 0x802B
#define GL_HISTOGRAM_LUMINANCE_SIZE_EXT 0x802C
#define GL_HISTOGRAM_SINK_EXT 0x802D
#define GL_MINMAX_EXT 0x802E
#define GL_MINMAX_FORMAT_EXT 0x802F
#define GL_MINMAX_SINK_EXT 0x8030

typedef void (GLAPIENTRY * PFNGLGETHISTOGRAMEXTPROC) (GLenum target, GLboolean reset, GLenum format, GLenum type, void* values);
typedef void (GLAPIENTRY * PFNGLGETHISTOGRAMPARAMETERFVEXTPROC) (GLenum target, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETHISTOGRAMPARAMETERIVEXTPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETMINMAXEXTPROC) (GLenum target, GLboolean reset, GLenum format, GLenum type, void* values);
typedef void (GLAPIENTRY * PFNGLGETMINMAXPARAMETERFVEXTPROC) (GLenum target, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETMINMAXPARAMETERIVEXTPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLHISTOGRAMEXTPROC) (GLenum target, GLsizei width, GLenum internalformat, GLboolean sink);
typedef void (GLAPIENTRY * PFNGLMINMAXEXTPROC) (GLenum target, GLenum internalformat, GLboolean sink);
typedef void (GLAPIENTRY * PFNGLRESETHISTOGRAMEXTPROC) (GLenum target);
typedef void (GLAPIENTRY * PFNGLRESETMINMAXEXTPROC) (GLenum target);

GLEWAPI PFNGLGETHISTOGRAMEXTPROC glewGetHistogramEXT;
GLEWAPI PFNGLGETHISTOGRAMPARAMETERFVEXTPROC glewGetHistogramParameterfvEXT;
GLEWAPI PFNGLGETHISTOGRAMPARAMETERIVEXTPROC glewGetHistogramParameterivEXT;
GLEWAPI PFNGLGETMINMAXEXTPROC glewGetMinmaxEXT;
GLEWAPI PFNGLGETMINMAXPARAMETERFVEXTPROC glewGetMinmaxParameterfvEXT;
GLEWAPI PFNGLGETMINMAXPARAMETERIVEXTPROC glewGetMinmaxParameterivEXT;
GLEWAPI PFNGLHISTOGRAMEXTPROC glewHistogramEXT;
GLEWAPI PFNGLMINMAXEXTPROC glewMinmaxEXT;
GLEWAPI PFNGLRESETHISTOGRAMEXTPROC glewResetHistogramEXT;
GLEWAPI PFNGLRESETMINMAXEXTPROC glewResetMinmaxEXT;

#define glGetHistogramEXT glewGetHistogramEXT
#define glGetHistogramParameterfvEXT glewGetHistogramParameterfvEXT
#define glGetHistogramParameterivEXT glewGetHistogramParameterivEXT
#define glGetMinmaxEXT glewGetMinmaxEXT
#define glGetMinmaxParameterfvEXT glewGetMinmaxParameterfvEXT
#define glGetMinmaxParameterivEXT glewGetMinmaxParameterivEXT
#define glHistogramEXT glewHistogramEXT
#define glMinmaxEXT glewMinmaxEXT
#define glResetHistogramEXT glewResetHistogramEXT
#define glResetMinmaxEXT glewResetMinmaxEXT

GLEWAPI GLboolean GLEW_EXT_histogram;

#endif /* GL_EXT_histogram */

/* ----------------------- GL_EXT_index_array_formats ---------------------- */

#ifndef GL_EXT_index_array_formats
#define GL_EXT_index_array_formats 1

GLEWAPI GLboolean GLEW_EXT_index_array_formats;

#endif /* GL_EXT_index_array_formats */

/* --------------------------- GL_EXT_index_func --------------------------- */

#ifndef GL_EXT_index_func
#define GL_EXT_index_func 1

typedef void (GLAPIENTRY * PFNGLINDEXFUNCEXTPROC) (GLenum func, GLfloat ref);

GLEWAPI PFNGLINDEXFUNCEXTPROC glewIndexFuncEXT;

#define glIndexFuncEXT glewIndexFuncEXT

GLEWAPI GLboolean GLEW_EXT_index_func;

#endif /* GL_EXT_index_func */

/* ------------------------- GL_EXT_index_material ------------------------- */

#ifndef GL_EXT_index_material
#define GL_EXT_index_material 1

typedef void (GLAPIENTRY * PFNGLINDEXMATERIALEXTPROC) (GLenum face, GLenum mode);

GLEWAPI PFNGLINDEXMATERIALEXTPROC glewIndexMaterialEXT;

#define glIndexMaterialEXT glewIndexMaterialEXT

GLEWAPI GLboolean GLEW_EXT_index_material;

#endif /* GL_EXT_index_material */

/* -------------------------- GL_EXT_index_texture ------------------------- */

#ifndef GL_EXT_index_texture
#define GL_EXT_index_texture 1

GLEWAPI GLboolean GLEW_EXT_index_texture;

#endif /* GL_EXT_index_texture */

/* -------------------------- GL_EXT_light_texture ------------------------- */

#ifndef GL_EXT_light_texture
#define GL_EXT_light_texture 1

#define GL_FRAGMENT_MATERIAL_EXT 0x8349
#define GL_FRAGMENT_NORMAL_EXT 0x834A
#define GL_FRAGMENT_COLOR_EXT 0x834C
#define GL_ATTENUATION_EXT 0x834D
#define GL_SHADOW_ATTENUATION_EXT 0x834E
#define GL_TEXTURE_APPLICATION_MODE_EXT 0x834F
#define GL_TEXTURE_LIGHT_EXT 0x8350
#define GL_TEXTURE_MATERIAL_FACE_EXT 0x8351
#define GL_TEXTURE_MATERIAL_PARAMETER_EXT 0x8352
#define GL_FRAGMENT_DEPTH_EXT 0x8452

typedef void (GLAPIENTRY * PFNGLAPPLYTEXTUREEXTPROC) (GLenum mode);
typedef void (GLAPIENTRY * PFNGLTEXTURELIGHTEXTPROC) (GLenum pname);
typedef void (GLAPIENTRY * PFNGLTEXTUREMATERIALEXTPROC) (GLenum face, GLenum mode);

GLEWAPI PFNGLAPPLYTEXTUREEXTPROC glewApplyTextureEXT;
GLEWAPI PFNGLTEXTURELIGHTEXTPROC glewTextureLightEXT;
GLEWAPI PFNGLTEXTUREMATERIALEXTPROC glewTextureMaterialEXT;

#define glApplyTextureEXT glewApplyTextureEXT
#define glTextureLightEXT glewTextureLightEXT
#define glTextureMaterialEXT glewTextureMaterialEXT

GLEWAPI GLboolean GLEW_EXT_light_texture;

#endif /* GL_EXT_light_texture */

/* ------------------------- GL_EXT_misc_attribute ------------------------- */

#ifndef GL_EXT_misc_attribute
#define GL_EXT_misc_attribute 1

GLEWAPI GLboolean GLEW_EXT_misc_attribute;

#endif /* GL_EXT_misc_attribute */

/* ------------------------ GL_EXT_multi_draw_arrays ----------------------- */

#ifndef GL_EXT_multi_draw_arrays
#define GL_EXT_multi_draw_arrays 1

typedef void (GLAPIENTRY * PFNGLMULTIDRAWARRAYSEXTPROC) (GLenum mode, GLint* first, GLsizei *count, GLsizei primcount);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWELEMENTSEXTPROC) (GLenum mode, GLsizei* count, GLenum type, const GLvoid **indices, GLsizei primcount);

GLEWAPI PFNGLMULTIDRAWARRAYSEXTPROC glewMultiDrawArraysEXT;
GLEWAPI PFNGLMULTIDRAWELEMENTSEXTPROC glewMultiDrawElementsEXT;

#define glMultiDrawArraysEXT glewMultiDrawArraysEXT
#define glMultiDrawElementsEXT glewMultiDrawElementsEXT

GLEWAPI GLboolean GLEW_EXT_multi_draw_arrays;

#endif /* GL_EXT_multi_draw_arrays */

/* --------------------------- GL_EXT_multisample -------------------------- */

#ifndef GL_EXT_multisample
#define GL_EXT_multisample 1

#define GL_MULTISAMPLE_EXT 0x809D
#define GL_SAMPLE_ALPHA_TO_MASK_EXT 0x809E
#define GL_SAMPLE_ALPHA_TO_ONE_EXT 0x809F
#define GL_SAMPLE_MASK_EXT 0x80A0
#define GL_1PASS_EXT 0x80A1
#define GL_2PASS_0_EXT 0x80A2
#define GL_2PASS_1_EXT 0x80A3
#define GL_4PASS_0_EXT 0x80A4
#define GL_4PASS_1_EXT 0x80A5
#define GL_4PASS_2_EXT 0x80A6
#define GL_4PASS_3_EXT 0x80A7
#define GL_SAMPLE_BUFFERS_EXT 0x80A8
#define GL_SAMPLES_EXT 0x80A9
#define GL_SAMPLE_MASK_VALUE_EXT 0x80AA
#define GL_SAMPLE_MASK_INVERT_EXT 0x80AB
#define GL_SAMPLE_PATTERN_EXT 0x80AC
#define GL_MULTISAMPLE_BIT_EXT 0x20000000

typedef void (GLAPIENTRY * PFNGLSAMPLEMASKEXTPROC) (GLclampf value, GLboolean invert);
typedef void (GLAPIENTRY * PFNGLSAMPLEPATTERNEXTPROC) (GLenum pattern);

GLEWAPI PFNGLSAMPLEMASKEXTPROC glewSampleMaskEXT;
GLEWAPI PFNGLSAMPLEPATTERNEXTPROC glewSamplePatternEXT;

#define glSampleMaskEXT glewSampleMaskEXT
#define glSamplePatternEXT glewSamplePatternEXT

GLEWAPI GLboolean GLEW_EXT_multisample;

#endif /* GL_EXT_multisample */

/* -------------------------- GL_EXT_packed_pixels ------------------------- */

#ifndef GL_EXT_packed_pixels
#define GL_EXT_packed_pixels 1

#define GL_UNSIGNED_BYTE_3_3_2_EXT 0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4_EXT 0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1_EXT 0x8034
#define GL_UNSIGNED_INT_8_8_8_8_EXT 0x8035
#define GL_UNSIGNED_INT_10_10_10_2_EXT 0x8036

GLEWAPI GLboolean GLEW_EXT_packed_pixels;

#endif /* GL_EXT_packed_pixels */

/* ------------------------ GL_EXT_paletted_texture ------------------------ */

#ifndef GL_EXT_paletted_texture
#define GL_EXT_paletted_texture 1

#define GL_TEXTURE_1D 0x0DE0
#define GL_TEXTURE_2D 0x0DE1
#define GL_PROXY_TEXTURE_1D 0x8063
#define GL_PROXY_TEXTURE_2D 0x8064
#define GL_TEXTURE_3D_EXT 0x806F
#define GL_PROXY_TEXTURE_3D_EXT 0x8070
#define GL_COLOR_TABLE_FORMAT_EXT 0x80D8
#define GL_COLOR_TABLE_WIDTH_EXT 0x80D9
#define GL_COLOR_TABLE_RED_SIZE_EXT 0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE_EXT 0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE_EXT 0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE_EXT 0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE_EXT 0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE_EXT 0x80DF
#define GL_COLOR_INDEX1_EXT 0x80E2
#define GL_COLOR_INDEX2_EXT 0x80E3
#define GL_COLOR_INDEX4_EXT 0x80E4
#define GL_COLOR_INDEX8_EXT 0x80E5
#define GL_COLOR_INDEX12_EXT 0x80E6
#define GL_COLOR_INDEX16_EXT 0x80E7
#define GL_TEXTURE_INDEX_SIZE_EXT 0x80ED
#define GL_TEXTURE_CUBE_MAP_ARB 0x8513
#define GL_PROXY_TEXTURE_CUBE_MAP_ARB 0x851B

typedef void (GLAPIENTRY * PFNGLCOLORTABLEEXTPROC) (GLenum target, GLenum internalFormat, GLsizei width, GLenum format, GLenum type, const void* data);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEEXTPROC) (GLenum target, GLenum format, GLenum type, void* data);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPARAMETERFVEXTPROC) (GLenum target, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPARAMETERIVEXTPROC) (GLenum target, GLenum pname, GLint* params);

GLEWAPI PFNGLCOLORTABLEEXTPROC glewColorTableEXT;
GLEWAPI PFNGLGETCOLORTABLEEXTPROC glewGetColorTableEXT;
GLEWAPI PFNGLGETCOLORTABLEPARAMETERFVEXTPROC glewGetColorTableParameterfvEXT;
GLEWAPI PFNGLGETCOLORTABLEPARAMETERIVEXTPROC glewGetColorTableParameterivEXT;

#define glColorTableEXT glewColorTableEXT
#define glGetColorTableEXT glewGetColorTableEXT
#define glGetColorTableParameterfvEXT glewGetColorTableParameterfvEXT
#define glGetColorTableParameterivEXT glewGetColorTableParameterivEXT

GLEWAPI GLboolean GLEW_EXT_paletted_texture;

#endif /* GL_EXT_paletted_texture */

/* ------------------------- GL_EXT_pixel_transform ------------------------ */

#ifndef GL_EXT_pixel_transform
#define GL_EXT_pixel_transform 1

#define GL_PIXEL_TRANSFORM_2D_EXT 0x8330
#define GL_PIXEL_MAG_FILTER_EXT 0x8331
#define GL_PIXEL_MIN_FILTER_EXT 0x8332
#define GL_PIXEL_CUBIC_WEIGHT_EXT 0x8333
#define GL_CUBIC_EXT 0x8334
#define GL_AVERAGE_EXT 0x8335
#define GL_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT 0x8336
#define GL_MAX_PIXEL_TRANSFORM_2D_STACK_DEPTH_EXT 0x8337
#define GL_PIXEL_TRANSFORM_2D_MATRIX_EXT 0x8338

typedef void (GLAPIENTRY * PFNGLGETPIXELTRANSFORMPARAMETERFVEXTPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETPIXELTRANSFORMPARAMETERIVEXTPROC) (GLenum target, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLPIXELTRANSFORMPARAMETERFEXTPROC) (GLenum target, GLenum pname, const GLfloat param);
typedef void (GLAPIENTRY * PFNGLPIXELTRANSFORMPARAMETERFVEXTPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLPIXELTRANSFORMPARAMETERIEXTPROC) (GLenum target, GLenum pname, const GLint param);
typedef void (GLAPIENTRY * PFNGLPIXELTRANSFORMPARAMETERIVEXTPROC) (GLenum target, GLenum pname, const GLint* params);

GLEWAPI PFNGLGETPIXELTRANSFORMPARAMETERFVEXTPROC glewGetPixelTransformParameterfvEXT;
GLEWAPI PFNGLGETPIXELTRANSFORMPARAMETERIVEXTPROC glewGetPixelTransformParameterivEXT;
GLEWAPI PFNGLPIXELTRANSFORMPARAMETERFEXTPROC glewPixelTransformParameterfEXT;
GLEWAPI PFNGLPIXELTRANSFORMPARAMETERFVEXTPROC glewPixelTransformParameterfvEXT;
GLEWAPI PFNGLPIXELTRANSFORMPARAMETERIEXTPROC glewPixelTransformParameteriEXT;
GLEWAPI PFNGLPIXELTRANSFORMPARAMETERIVEXTPROC glewPixelTransformParameterivEXT;

#define glGetPixelTransformParameterfvEXT glewGetPixelTransformParameterfvEXT
#define glGetPixelTransformParameterivEXT glewGetPixelTransformParameterivEXT
#define glPixelTransformParameterfEXT glewPixelTransformParameterfEXT
#define glPixelTransformParameterfvEXT glewPixelTransformParameterfvEXT
#define glPixelTransformParameteriEXT glewPixelTransformParameteriEXT
#define glPixelTransformParameterivEXT glewPixelTransformParameterivEXT

GLEWAPI GLboolean GLEW_EXT_pixel_transform;

#endif /* GL_EXT_pixel_transform */

/* ------------------- GL_EXT_pixel_transform_color_table ------------------ */

#ifndef GL_EXT_pixel_transform_color_table
#define GL_EXT_pixel_transform_color_table 1

GLEWAPI GLboolean GLEW_EXT_pixel_transform_color_table;

#endif /* GL_EXT_pixel_transform_color_table */

/* ------------------------ GL_EXT_point_parameters ------------------------ */

#ifndef GL_EXT_point_parameters
#define GL_EXT_point_parameters 1

#define GL_POINT_SIZE_MIN_EXT 0x8126
#define GL_POINT_SIZE_MAX_EXT 0x8127
#define GL_POINT_FADE_THRESHOLD_SIZE_EXT 0x8128
#define GL_DISTANCE_ATTENUATION_EXT 0x8129

typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERFEXTPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERFVEXTPROC) (GLenum pname, GLfloat* params);

GLEWAPI PFNGLPOINTPARAMETERFEXTPROC glewPointParameterfEXT;
GLEWAPI PFNGLPOINTPARAMETERFVEXTPROC glewPointParameterfvEXT;

#define glPointParameterfEXT glewPointParameterfEXT
#define glPointParameterfvEXT glewPointParameterfvEXT

GLEWAPI GLboolean GLEW_EXT_point_parameters;

#endif /* GL_EXT_point_parameters */

/* ------------------------- GL_EXT_polygon_offset ------------------------- */

#ifndef GL_EXT_polygon_offset
#define GL_EXT_polygon_offset 1

#define GL_POLYGON_OFFSET_EXT 0x8037
#define GL_POLYGON_OFFSET_FACTOR_EXT 0x8038
#define GL_POLYGON_OFFSET_BIAS_EXT 0x8039

typedef void (GLAPIENTRY * PFNGLPOLYGONOFFSETEXTPROC) (GLfloat factor, GLfloat bias);

GLEWAPI PFNGLPOLYGONOFFSETEXTPROC glewPolygonOffsetEXT;

#define glPolygonOffsetEXT glewPolygonOffsetEXT

GLEWAPI GLboolean GLEW_EXT_polygon_offset;

#endif /* GL_EXT_polygon_offset */

/* ------------------------- GL_EXT_rescale_normal ------------------------- */

#ifndef GL_EXT_rescale_normal
#define GL_EXT_rescale_normal 1

GLEWAPI GLboolean GLEW_EXT_rescale_normal;

#endif /* GL_EXT_rescale_normal */

/* -------------------------- GL_EXT_scene_marker -------------------------- */

#ifndef GL_EXT_scene_marker
#define GL_EXT_scene_marker 1

typedef void (GLAPIENTRY * PFNGLBEGINSCENEEXTPROC) (void);
typedef void (GLAPIENTRY * PFNGLENDSCENEEXTPROC) (void);

GLEWAPI PFNGLBEGINSCENEEXTPROC glewBeginSceneEXT;
GLEWAPI PFNGLENDSCENEEXTPROC glewEndSceneEXT;

#define glBeginSceneEXT glewBeginSceneEXT
#define glEndSceneEXT glewEndSceneEXT

GLEWAPI GLboolean GLEW_EXT_scene_marker;

#endif /* GL_EXT_scene_marker */

/* ------------------------- GL_EXT_secondary_color ------------------------ */

#ifndef GL_EXT_secondary_color
#define GL_EXT_secondary_color 1

#define GL_COLOR_SUM_EXT 0x8458
#define GL_CURRENT_SECONDARY_COLOR_EXT 0x8459
#define GL_SECONDARY_COLOR_ARRAY_SIZE_EXT 0x845A
#define GL_SECONDARY_COLOR_ARRAY_TYPE_EXT 0x845B
#define GL_SECONDARY_COLOR_ARRAY_STRIDE_EXT 0x845C
#define GL_SECONDARY_COLOR_ARRAY_POINTER_EXT 0x845D
#define GL_SECONDARY_COLOR_ARRAY_EXT 0x845E

typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3BEXTPROC) (GLbyte red, GLbyte green, GLbyte blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3BVEXTPROC) (const GLbyte *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3DEXTPROC) (GLdouble red, GLdouble green, GLdouble blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3DVEXTPROC) (const GLdouble *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3FEXTPROC) (GLfloat red, GLfloat green, GLfloat blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3FVEXTPROC) (const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3IEXTPROC) (GLint red, GLint green, GLint blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3IVEXTPROC) (const GLint *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3SEXTPROC) (GLshort red, GLshort green, GLshort blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3SVEXTPROC) (const GLshort *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UBEXTPROC) (GLubyte red, GLubyte green, GLubyte blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UBVEXTPROC) (const GLubyte *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UIEXTPROC) (GLuint red, GLuint green, GLuint blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3UIVEXTPROC) (const GLuint *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3USEXTPROC) (GLushort red, GLushort green, GLushort blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3USVEXTPROC) (const GLushort *v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLORPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLvoid *pointer);

GLEWAPI PFNGLSECONDARYCOLOR3BEXTPROC glewSecondaryColor3bEXT;
GLEWAPI PFNGLSECONDARYCOLOR3BVEXTPROC glewSecondaryColor3bvEXT;
GLEWAPI PFNGLSECONDARYCOLOR3DEXTPROC glewSecondaryColor3dEXT;
GLEWAPI PFNGLSECONDARYCOLOR3DVEXTPROC glewSecondaryColor3dvEXT;
GLEWAPI PFNGLSECONDARYCOLOR3FEXTPROC glewSecondaryColor3fEXT;
GLEWAPI PFNGLSECONDARYCOLOR3FVEXTPROC glewSecondaryColor3fvEXT;
GLEWAPI PFNGLSECONDARYCOLOR3IEXTPROC glewSecondaryColor3iEXT;
GLEWAPI PFNGLSECONDARYCOLOR3IVEXTPROC glewSecondaryColor3ivEXT;
GLEWAPI PFNGLSECONDARYCOLOR3SEXTPROC glewSecondaryColor3sEXT;
GLEWAPI PFNGLSECONDARYCOLOR3SVEXTPROC glewSecondaryColor3svEXT;
GLEWAPI PFNGLSECONDARYCOLOR3UBEXTPROC glewSecondaryColor3ubEXT;
GLEWAPI PFNGLSECONDARYCOLOR3UBVEXTPROC glewSecondaryColor3ubvEXT;
GLEWAPI PFNGLSECONDARYCOLOR3UIEXTPROC glewSecondaryColor3uiEXT;
GLEWAPI PFNGLSECONDARYCOLOR3UIVEXTPROC glewSecondaryColor3uivEXT;
GLEWAPI PFNGLSECONDARYCOLOR3USEXTPROC glewSecondaryColor3usEXT;
GLEWAPI PFNGLSECONDARYCOLOR3USVEXTPROC glewSecondaryColor3usvEXT;
GLEWAPI PFNGLSECONDARYCOLORPOINTEREXTPROC glewSecondaryColorPointerEXT;

#define glSecondaryColor3bEXT glewSecondaryColor3bEXT
#define glSecondaryColor3bvEXT glewSecondaryColor3bvEXT
#define glSecondaryColor3dEXT glewSecondaryColor3dEXT
#define glSecondaryColor3dvEXT glewSecondaryColor3dvEXT
#define glSecondaryColor3fEXT glewSecondaryColor3fEXT
#define glSecondaryColor3fvEXT glewSecondaryColor3fvEXT
#define glSecondaryColor3iEXT glewSecondaryColor3iEXT
#define glSecondaryColor3ivEXT glewSecondaryColor3ivEXT
#define glSecondaryColor3sEXT glewSecondaryColor3sEXT
#define glSecondaryColor3svEXT glewSecondaryColor3svEXT
#define glSecondaryColor3ubEXT glewSecondaryColor3ubEXT
#define glSecondaryColor3ubvEXT glewSecondaryColor3ubvEXT
#define glSecondaryColor3uiEXT glewSecondaryColor3uiEXT
#define glSecondaryColor3uivEXT glewSecondaryColor3uivEXT
#define glSecondaryColor3usEXT glewSecondaryColor3usEXT
#define glSecondaryColor3usvEXT glewSecondaryColor3usvEXT
#define glSecondaryColorPointerEXT glewSecondaryColorPointerEXT

GLEWAPI GLboolean GLEW_EXT_secondary_color;

#endif /* GL_EXT_secondary_color */

/* --------------------- GL_EXT_separate_specular_color -------------------- */

#ifndef GL_EXT_separate_specular_color
#define GL_EXT_separate_specular_color 1

#define GL_LIGHT_MODEL_COLOR_CONTROL_EXT 0x81F8
#define GL_SINGLE_COLOR_EXT 0x81F9
#define GL_SEPARATE_SPECULAR_COLOR_EXT 0x81FA

GLEWAPI GLboolean GLEW_EXT_separate_specular_color;

#endif /* GL_EXT_separate_specular_color */

/* -------------------------- GL_EXT_shadow_funcs -------------------------- */

#ifndef GL_EXT_shadow_funcs
#define GL_EXT_shadow_funcs 1

GLEWAPI GLboolean GLEW_EXT_shadow_funcs;

#endif /* GL_EXT_shadow_funcs */

/* --------------------- GL_EXT_shared_texture_palette --------------------- */

#ifndef GL_EXT_shared_texture_palette
#define GL_EXT_shared_texture_palette 1

#define GL_SHARED_TEXTURE_PALETTE_EXT 0x81FB

GLEWAPI GLboolean GLEW_EXT_shared_texture_palette;

#endif /* GL_EXT_shared_texture_palette */

/* ------------------------ GL_EXT_stencil_two_side ------------------------ */

#ifndef GL_EXT_stencil_two_side
#define GL_EXT_stencil_two_side 1

#define GL_STENCIL_TEST_TWO_SIDE_EXT 0x8910
#define GL_ACTIVE_STENCIL_FACE_EXT 0x8911

typedef void (GLAPIENTRY * PFNGLACTIVESTENCILFACEEXTPROC) (GLenum face);

GLEWAPI PFNGLACTIVESTENCILFACEEXTPROC glewActiveStencilFaceEXT;

#define glActiveStencilFaceEXT glewActiveStencilFaceEXT

GLEWAPI GLboolean GLEW_EXT_stencil_two_side;

#endif /* GL_EXT_stencil_two_side */

/* -------------------------- GL_EXT_stencil_wrap -------------------------- */

#ifndef GL_EXT_stencil_wrap
#define GL_EXT_stencil_wrap 1

#define GL_INCR_WRAP_EXT 0x8507
#define GL_DECR_WRAP_EXT 0x8508

GLEWAPI GLboolean GLEW_EXT_stencil_wrap;

#endif /* GL_EXT_stencil_wrap */

/* --------------------------- GL_EXT_subtexture --------------------------- */

#ifndef GL_EXT_subtexture
#define GL_EXT_subtexture 1

typedef void (GLAPIENTRY * PFNGLTEXSUBIMAGE1DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels);
typedef void (GLAPIENTRY * PFNGLTEXSUBIMAGE2DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels);
typedef void (GLAPIENTRY * PFNGLTEXSUBIMAGE3DEXTPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels);

GLEWAPI PFNGLTEXSUBIMAGE1DEXTPROC glewTexSubImage1DEXT;
GLEWAPI PFNGLTEXSUBIMAGE2DEXTPROC glewTexSubImage2DEXT;
GLEWAPI PFNGLTEXSUBIMAGE3DEXTPROC glewTexSubImage3DEXT;

#define glTexSubImage1DEXT glewTexSubImage1DEXT
#define glTexSubImage2DEXT glewTexSubImage2DEXT
#define glTexSubImage3DEXT glewTexSubImage3DEXT

GLEWAPI GLboolean GLEW_EXT_subtexture;

#endif /* GL_EXT_subtexture */

/* ----------------------------- GL_EXT_texture ---------------------------- */

#ifndef GL_EXT_texture
#define GL_EXT_texture 1

#define GL_ALPHA4_EXT 0x803B
#define GL_ALPHA8_EXT 0x803C
#define GL_ALPHA12_EXT 0x803D
#define GL_ALPHA16_EXT 0x803E
#define GL_LUMINANCE4_EXT 0x803F
#define GL_LUMINANCE8_EXT 0x8040
#define GL_LUMINANCE12_EXT 0x8041
#define GL_LUMINANCE16_EXT 0x8042
#define GL_LUMINANCE4_ALPHA4_EXT 0x8043
#define GL_LUMINANCE6_ALPHA2_EXT 0x8044
#define GL_LUMINANCE8_ALPHA8_EXT 0x8045
#define GL_LUMINANCE12_ALPHA4_EXT 0x8046
#define GL_LUMINANCE12_ALPHA12_EXT 0x8047
#define GL_LUMINANCE16_ALPHA16_EXT 0x8048
#define GL_INTENSITY_EXT 0x8049
#define GL_INTENSITY4_EXT 0x804A
#define GL_INTENSITY8_EXT 0x804B
#define GL_INTENSITY12_EXT 0x804C
#define GL_INTENSITY16_EXT 0x804D
#define GL_RGB2_EXT 0x804E
#define GL_RGB4_EXT 0x804F
#define GL_RGB5_EXT 0x8050
#define GL_RGB8_EXT 0x8051
#define GL_RGB10_EXT 0x8052
#define GL_RGB12_EXT 0x8053
#define GL_RGB16_EXT 0x8054
#define GL_RGBA2_EXT 0x8055
#define GL_RGBA4_EXT 0x8056
#define GL_RGB5_A1_EXT 0x8057
#define GL_RGBA8_EXT 0x8058
#define GL_RGB10_A2_EXT 0x8059
#define GL_RGBA12_EXT 0x805A
#define GL_RGBA16_EXT 0x805B
#define GL_TEXTURE_RED_SIZE_EXT 0x805C
#define GL_TEXTURE_GREEN_SIZE_EXT 0x805D
#define GL_TEXTURE_BLUE_SIZE_EXT 0x805E
#define GL_TEXTURE_ALPHA_SIZE_EXT 0x805F
#define GL_TEXTURE_LUMINANCE_SIZE_EXT 0x8060
#define GL_TEXTURE_INTENSITY_SIZE_EXT 0x8061
#define GL_REPLACE_EXT 0x8062
#define GL_PROXY_TEXTURE_1D_EXT 0x8063
#define GL_PROXY_TEXTURE_2D_EXT 0x8064

GLEWAPI GLboolean GLEW_EXT_texture;

#endif /* GL_EXT_texture */

/* ---------------------------- GL_EXT_texture3D --------------------------- */

#ifndef GL_EXT_texture3D
#define GL_EXT_texture3D 1

#define GL_PACK_SKIP_IMAGES_EXT 0x806B
#define GL_PACK_IMAGE_HEIGHT_EXT 0x806C
#define GL_UNPACK_SKIP_IMAGES_EXT 0x806D
#define GL_UNPACK_IMAGE_HEIGHT_EXT 0x806E
#define GL_TEXTURE_3D_EXT 0x806F
#define GL_PROXY_TEXTURE_3D_EXT 0x8070
#define GL_TEXTURE_DEPTH_EXT 0x8071
#define GL_TEXTURE_WRAP_R_EXT 0x8072
#define GL_MAX_3D_TEXTURE_SIZE_EXT 0x8073

typedef void (GLAPIENTRY * PFNGLTEXIMAGE3DEXTPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void* pixels);

GLEWAPI PFNGLTEXIMAGE3DEXTPROC glewTexImage3DEXT;

#define glTexImage3DEXT glewTexImage3DEXT

GLEWAPI GLboolean GLEW_EXT_texture3D;

#endif /* GL_EXT_texture3D */

/* -------------------- GL_EXT_texture_compression_s3tc -------------------- */

#ifndef GL_EXT_texture_compression_s3tc
#define GL_EXT_texture_compression_s3tc 1

#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3

GLEWAPI GLboolean GLEW_EXT_texture_compression_s3tc;

#endif /* GL_EXT_texture_compression_s3tc */

/* ------------------------ GL_EXT_texture_cube_map ------------------------ */

#ifndef GL_EXT_texture_cube_map
#define GL_EXT_texture_cube_map 1

#define GL_NORMAL_MAP_EXT 0x8511
#define GL_REFLECTION_MAP_EXT 0x8512
#define GL_TEXTURE_CUBE_MAP_EXT 0x8513
#define GL_TEXTURE_BINDING_CUBE_MAP_EXT 0x8514
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT 0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT 0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT 0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT 0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT 0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT 0x851A
#define GL_PROXY_TEXTURE_CUBE_MAP_EXT 0x851B
#define GL_MAX_CUBE_MAP_TEXTURE_SIZE_EXT 0x851C

GLEWAPI GLboolean GLEW_EXT_texture_cube_map;

#endif /* GL_EXT_texture_cube_map */

/* ----------------------- GL_EXT_texture_edge_clamp ----------------------- */

#ifndef GL_EXT_texture_edge_clamp
#define GL_EXT_texture_edge_clamp 1

#define GL_CLAMP_TO_EDGE_EXT 0x812F

GLEWAPI GLboolean GLEW_EXT_texture_edge_clamp;

#endif /* GL_EXT_texture_edge_clamp */

/* --------------------------- GL_EXT_texture_env -------------------------- */

#ifndef GL_EXT_texture_env
#define GL_EXT_texture_env 1

#define GL_TEXTURE_ENV0_EXT 0
#define GL_ENV_BLEND_EXT 0
#define GL_TEXTURE_ENV_SHIFT_EXT 0
#define GL_ENV_REPLACE_EXT 0
#define GL_ENV_ADD_EXT 0
#define GL_ENV_SUBTRACT_EXT 0
#define GL_TEXTURE_ENV_MODE_ALPHA_EXT 0
#define GL_ENV_REVERSE_SUBTRACT_EXT 0
#define GL_ENV_REVERSE_BLEND_EXT 0
#define GL_ENV_COPY_EXT 0
#define GL_ENV_MODULATE_EXT 0

GLEWAPI GLboolean GLEW_EXT_texture_env;

#endif /* GL_EXT_texture_env */

/* ------------------------- GL_EXT_texture_env_add ------------------------ */

#ifndef GL_EXT_texture_env_add
#define GL_EXT_texture_env_add 1

GLEWAPI GLboolean GLEW_EXT_texture_env_add;

#endif /* GL_EXT_texture_env_add */

/* ----------------------- GL_EXT_texture_env_combine ---------------------- */

#ifndef GL_EXT_texture_env_combine
#define GL_EXT_texture_env_combine 1

#define GL_COMBINE_EXT 0x8570
#define GL_COMBINE_RGB_EXT 0x8571
#define GL_COMBINE_ALPHA_EXT 0x8572
#define GL_RGB_SCALE_EXT 0x8573
#define GL_ADD_SIGNED_EXT 0x8574
#define GL_INTERPOLATE_EXT 0x8575
#define GL_CONSTANT_EXT 0x8576
#define GL_PRIMARY_COLOR_EXT 0x8577
#define GL_PREVIOUS_EXT 0x8578
#define GL_SOURCE0_RGB_EXT 0x8580
#define GL_SOURCE1_RGB_EXT 0x8581
#define GL_SOURCE2_RGB_EXT 0x8582
#define GL_SOURCE0_ALPHA_EXT 0x8588
#define GL_SOURCE1_ALPHA_EXT 0x8589
#define GL_SOURCE2_ALPHA_EXT 0x858A
#define GL_OPERAND0_RGB_EXT 0x8590
#define GL_OPERAND1_RGB_EXT 0x8591
#define GL_OPERAND2_RGB_EXT 0x8592
#define GL_OPERAND0_ALPHA_EXT 0x8598
#define GL_OPERAND1_ALPHA_EXT 0x8599
#define GL_OPERAND2_ALPHA_EXT 0x859A

GLEWAPI GLboolean GLEW_EXT_texture_env_combine;

#endif /* GL_EXT_texture_env_combine */

/* ------------------------ GL_EXT_texture_env_dot3 ------------------------ */

#ifndef GL_EXT_texture_env_dot3
#define GL_EXT_texture_env_dot3 1

#define GL_DOT3_RGB_EXT 0x8740
#define GL_DOT3_RGBA_EXT 0x8741

GLEWAPI GLboolean GLEW_EXT_texture_env_dot3;

#endif /* GL_EXT_texture_env_dot3 */

/* ------------------- GL_EXT_texture_filter_anisotropic ------------------- */

#ifndef GL_EXT_texture_filter_anisotropic
#define GL_EXT_texture_filter_anisotropic 1

#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF

GLEWAPI GLboolean GLEW_EXT_texture_filter_anisotropic;

#endif /* GL_EXT_texture_filter_anisotropic */

/* ------------------------ GL_EXT_texture_lod_bias ------------------------ */

#ifndef GL_EXT_texture_lod_bias
#define GL_EXT_texture_lod_bias 1

#define GL_MAX_TEXTURE_LOD_BIAS_EXT 0x84FD
#define GL_TEXTURE_FILTER_CONTROL_EXT 0x8500
#define GL_TEXTURE_LOD_BIAS_EXT 0x8501

GLEWAPI GLboolean GLEW_EXT_texture_lod_bias;

#endif /* GL_EXT_texture_lod_bias */

/* ---------------------- GL_EXT_texture_mirror_clamp ---------------------- */

#ifndef GL_EXT_texture_mirror_clamp
#define GL_EXT_texture_mirror_clamp 1

#define GL_MIRROR_CLAMP_EXT 0x8742
#define GL_MIRROR_CLAMP_TO_EDGE_EXT 0x8743
#define GL_MIRROR_CLAMP_TO_BORDER_EXT 0x8912

GLEWAPI GLboolean GLEW_EXT_texture_mirror_clamp;

#endif /* GL_EXT_texture_mirror_clamp */

/* ------------------------- GL_EXT_texture_object ------------------------- */

#ifndef GL_EXT_texture_object
#define GL_EXT_texture_object 1

#define GL_TEXTURE_PRIORITY_EXT 0x8066
#define GL_TEXTURE_RESIDENT_EXT 0x8067
#define GL_TEXTURE_1D_BINDING_EXT 0x8068
#define GL_TEXTURE_2D_BINDING_EXT 0x8069
#define GL_TEXTURE_3D_BINDING_EXT 0x806A

typedef GLboolean (GLAPIENTRY * PFNGLARETEXTURESRESIDENTEXTPROC) (GLsizei n, const GLuint* textures, GLboolean* residences);
typedef void (GLAPIENTRY * PFNGLBINDTEXTUREEXTPROC) (GLenum target, GLuint texture);
typedef void (GLAPIENTRY * PFNGLDELETETEXTURESEXTPROC) (GLsizei n, const GLuint* textures);
typedef void (GLAPIENTRY * PFNGLGENTEXTURESEXTPROC) (GLsizei n, GLuint* textures);
typedef GLboolean (GLAPIENTRY * PFNGLISTEXTUREEXTPROC) (GLuint texture);
typedef void (GLAPIENTRY * PFNGLPRIORITIZETEXTURESEXTPROC) (GLsizei n, const GLuint* textures, const GLclampf* priorities);

GLEWAPI PFNGLARETEXTURESRESIDENTEXTPROC glewAreTexturesResidentEXT;
GLEWAPI PFNGLBINDTEXTUREEXTPROC glewBindTextureEXT;
GLEWAPI PFNGLDELETETEXTURESEXTPROC glewDeleteTexturesEXT;
GLEWAPI PFNGLGENTEXTURESEXTPROC glewGenTexturesEXT;
GLEWAPI PFNGLISTEXTUREEXTPROC glewIsTextureEXT;
GLEWAPI PFNGLPRIORITIZETEXTURESEXTPROC glewPrioritizeTexturesEXT;

#define glAreTexturesResidentEXT glewAreTexturesResidentEXT
#define glBindTextureEXT glewBindTextureEXT
#define glDeleteTexturesEXT glewDeleteTexturesEXT
#define glGenTexturesEXT glewGenTexturesEXT
#define glIsTextureEXT glewIsTextureEXT
#define glPrioritizeTexturesEXT glewPrioritizeTexturesEXT

GLEWAPI GLboolean GLEW_EXT_texture_object;

#endif /* GL_EXT_texture_object */

/* --------------------- GL_EXT_texture_perturb_normal --------------------- */

#ifndef GL_EXT_texture_perturb_normal
#define GL_EXT_texture_perturb_normal 1

#define GL_PERTURB_EXT 0x85AE
#define GL_TEXTURE_NORMAL_EXT 0x85AF

typedef void (GLAPIENTRY * PFNGLTEXTURENORMALEXTPROC) (GLenum mode);

GLEWAPI PFNGLTEXTURENORMALEXTPROC glewTextureNormalEXT;

#define glTextureNormalEXT glewTextureNormalEXT

GLEWAPI GLboolean GLEW_EXT_texture_perturb_normal;

#endif /* GL_EXT_texture_perturb_normal */

/* ------------------------ GL_EXT_texture_rectangle ----------------------- */

#ifndef GL_EXT_texture_rectangle
#define GL_EXT_texture_rectangle 1

#define GL_TEXTURE_RECTANGLE_EXT 0x85B3
#define GL_TEXTURE_BINDING_RECTANGLE_EXT 0x85B4
#define GL_PROXY_TEXTURE_RECTANGLE_EXT 0x85B5
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_EXT 0x85B6

GLEWAPI GLboolean GLEW_EXT_texture_rectangle;

#endif /* GL_EXT_texture_rectangle */

/* -------------------------- GL_EXT_vertex_array -------------------------- */

#ifndef GL_EXT_vertex_array
#define GL_EXT_vertex_array 1

#define GL_DOUBLE_EXT 0x140A
#define GL_VERTEX_ARRAY_EXT 0x8074
#define GL_NORMAL_ARRAY_EXT 0x8075
#define GL_COLOR_ARRAY_EXT 0x8076
#define GL_INDEX_ARRAY_EXT 0x8077
#define GL_TEXTURE_COORD_ARRAY_EXT 0x8078
#define GL_EDGE_FLAG_ARRAY_EXT 0x8079
#define GL_VERTEX_ARRAY_SIZE_EXT 0x807A
#define GL_VERTEX_ARRAY_TYPE_EXT 0x807B
#define GL_VERTEX_ARRAY_STRIDE_EXT 0x807C
#define GL_VERTEX_ARRAY_COUNT_EXT 0x807D
#define GL_NORMAL_ARRAY_TYPE_EXT 0x807E
#define GL_NORMAL_ARRAY_STRIDE_EXT 0x807F
#define GL_NORMAL_ARRAY_COUNT_EXT 0x8080
#define GL_COLOR_ARRAY_SIZE_EXT 0x8081
#define GL_COLOR_ARRAY_TYPE_EXT 0x8082
#define GL_COLOR_ARRAY_STRIDE_EXT 0x8083
#define GL_COLOR_ARRAY_COUNT_EXT 0x8084
#define GL_INDEX_ARRAY_TYPE_EXT 0x8085
#define GL_INDEX_ARRAY_STRIDE_EXT 0x8086
#define GL_INDEX_ARRAY_COUNT_EXT 0x8087
#define GL_TEXTURE_COORD_ARRAY_SIZE_EXT 0x8088
#define GL_TEXTURE_COORD_ARRAY_TYPE_EXT 0x8089
#define GL_TEXTURE_COORD_ARRAY_STRIDE_EXT 0x808A
#define GL_TEXTURE_COORD_ARRAY_COUNT_EXT 0x808B
#define GL_EDGE_FLAG_ARRAY_STRIDE_EXT 0x808C
#define GL_EDGE_FLAG_ARRAY_COUNT_EXT 0x808D
#define GL_VERTEX_ARRAY_POINTER_EXT 0x808E
#define GL_NORMAL_ARRAY_POINTER_EXT 0x808F
#define GL_COLOR_ARRAY_POINTER_EXT 0x8090
#define GL_INDEX_ARRAY_POINTER_EXT 0x8091
#define GL_TEXTURE_COORD_ARRAY_POINTER_EXT 0x8092
#define GL_EDGE_FLAG_ARRAY_POINTER_EXT 0x8093

typedef void (GLAPIENTRY * PFNGLARRAYELEMENTEXTPROC) (GLint i);
typedef void (GLAPIENTRY * PFNGLCOLORPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLsizei count, const void* pointer);
typedef void (GLAPIENTRY * PFNGLDRAWARRAYSEXTPROC) (GLenum mode, GLint first, GLsizei count);
typedef void (GLAPIENTRY * PFNGLEDGEFLAGPOINTEREXTPROC) (GLsizei stride, GLsizei count, const GLboolean* pointer);
typedef void (GLAPIENTRY * PFNGLGETPOINTERVEXTPROC) (GLenum pname, void** params);
typedef void (GLAPIENTRY * PFNGLINDEXPOINTEREXTPROC) (GLenum type, GLsizei stride, GLsizei count, const void* pointer);
typedef void (GLAPIENTRY * PFNGLNORMALPOINTEREXTPROC) (GLenum type, GLsizei stride, GLsizei count, const void* pointer);
typedef void (GLAPIENTRY * PFNGLTEXCOORDPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLsizei count, const void* pointer);
typedef void (GLAPIENTRY * PFNGLVERTEXPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, GLsizei count, const void* pointer);

GLEWAPI PFNGLARRAYELEMENTEXTPROC glewArrayElementEXT;
GLEWAPI PFNGLCOLORPOINTEREXTPROC glewColorPointerEXT;
GLEWAPI PFNGLDRAWARRAYSEXTPROC glewDrawArraysEXT;
GLEWAPI PFNGLEDGEFLAGPOINTEREXTPROC glewEdgeFlagPointerEXT;
GLEWAPI PFNGLGETPOINTERVEXTPROC glewGetPointervEXT;
GLEWAPI PFNGLINDEXPOINTEREXTPROC glewIndexPointerEXT;
GLEWAPI PFNGLNORMALPOINTEREXTPROC glewNormalPointerEXT;
GLEWAPI PFNGLTEXCOORDPOINTEREXTPROC glewTexCoordPointerEXT;
GLEWAPI PFNGLVERTEXPOINTEREXTPROC glewVertexPointerEXT;

#define glArrayElementEXT glewArrayElementEXT
#define glColorPointerEXT glewColorPointerEXT
#define glDrawArraysEXT glewDrawArraysEXT
#define glEdgeFlagPointerEXT glewEdgeFlagPointerEXT
#define glGetPointervEXT glewGetPointervEXT
#define glIndexPointerEXT glewIndexPointerEXT
#define glNormalPointerEXT glewNormalPointerEXT
#define glTexCoordPointerEXT glewTexCoordPointerEXT
#define glVertexPointerEXT glewVertexPointerEXT

GLEWAPI GLboolean GLEW_EXT_vertex_array;

#endif /* GL_EXT_vertex_array */

/* -------------------------- GL_EXT_vertex_shader ------------------------- */

#ifndef GL_EXT_vertex_shader
#define GL_EXT_vertex_shader 1

#define GL_VERTEX_SHADER_EXT 0x8780
#define GL_VERTEX_SHADER_BINDING_EXT 0x8781
#define GL_OP_INDEX_EXT 0x8782
#define GL_OP_NEGATE_EXT 0x8783
#define GL_OP_DOT3_EXT 0x8784
#define GL_OP_DOT4_EXT 0x8785
#define GL_OP_MUL_EXT 0x8786
#define GL_OP_ADD_EXT 0x8787
#define GL_OP_MADD_EXT 0x8788
#define GL_OP_FRAC_EXT 0x8789
#define GL_OP_MAX_EXT 0x878A
#define GL_OP_MIN_EXT 0x878B
#define GL_OP_SET_GE_EXT 0x878C
#define GL_OP_SET_LT_EXT 0x878D
#define GL_OP_CLAMP_EXT 0x878E
#define GL_OP_FLOOR_EXT 0x878F
#define GL_OP_ROUND_EXT 0x8790
#define GL_OP_EXP_BASE_2_EXT 0x8791
#define GL_OP_LOG_BASE_2_EXT 0x8792
#define GL_OP_POWER_EXT 0x8793
#define GL_OP_RECIP_EXT 0x8794
#define GL_OP_RECIP_SQRT_EXT 0x8795
#define GL_OP_SUB_EXT 0x8796
#define GL_OP_CROSS_PRODUCT_EXT 0x8797
#define GL_OP_MULTIPLY_MATRIX_EXT 0x8798
#define GL_OP_MOV_EXT 0x8799
#define GL_OUTPUT_VERTEX_EXT 0x879A
#define GL_OUTPUT_COLOR0_EXT 0x879B
#define GL_OUTPUT_COLOR1_EXT 0x879C
#define GL_OUTPUT_TEXTURE_COORD0_EXT 0x879D
#define GL_OUTPUT_TEXTURE_COORD1_EXT 0x879E
#define GL_OUTPUT_TEXTURE_COORD2_EXT 0x879F
#define GL_OUTPUT_TEXTURE_COORD3_EXT 0x87A0
#define GL_OUTPUT_TEXTURE_COORD4_EXT 0x87A1
#define GL_OUTPUT_TEXTURE_COORD5_EXT 0x87A2
#define GL_OUTPUT_TEXTURE_COORD6_EXT 0x87A3
#define GL_OUTPUT_TEXTURE_COORD7_EXT 0x87A4
#define GL_OUTPUT_TEXTURE_COORD8_EXT 0x87A5
#define GL_OUTPUT_TEXTURE_COORD9_EXT 0x87A6
#define GL_OUTPUT_TEXTURE_COORD10_EXT 0x87A7
#define GL_OUTPUT_TEXTURE_COORD11_EXT 0x87A8
#define GL_OUTPUT_TEXTURE_COORD12_EXT 0x87A9
#define GL_OUTPUT_TEXTURE_COORD13_EXT 0x87AA
#define GL_OUTPUT_TEXTURE_COORD14_EXT 0x87AB
#define GL_OUTPUT_TEXTURE_COORD15_EXT 0x87AC
#define GL_OUTPUT_TEXTURE_COORD16_EXT 0x87AD
#define GL_OUTPUT_TEXTURE_COORD17_EXT 0x87AE
#define GL_OUTPUT_TEXTURE_COORD18_EXT 0x87AF
#define GL_OUTPUT_TEXTURE_COORD19_EXT 0x87B0
#define GL_OUTPUT_TEXTURE_COORD20_EXT 0x87B1
#define GL_OUTPUT_TEXTURE_COORD21_EXT 0x87B2
#define GL_OUTPUT_TEXTURE_COORD22_EXT 0x87B3
#define GL_OUTPUT_TEXTURE_COORD23_EXT 0x87B4
#define GL_OUTPUT_TEXTURE_COORD24_EXT 0x87B5
#define GL_OUTPUT_TEXTURE_COORD25_EXT 0x87B6
#define GL_OUTPUT_TEXTURE_COORD26_EXT 0x87B7
#define GL_OUTPUT_TEXTURE_COORD27_EXT 0x87B8
#define GL_OUTPUT_TEXTURE_COORD28_EXT 0x87B9
#define GL_OUTPUT_TEXTURE_COORD29_EXT 0x87BA
#define GL_OUTPUT_TEXTURE_COORD30_EXT 0x87BB
#define GL_OUTPUT_TEXTURE_COORD31_EXT 0x87BC
#define GL_OUTPUT_FOG_EXT 0x87BD
#define GL_SCALAR_EXT 0x87BE
#define GL_VECTOR_EXT 0x87BF
#define GL_MATRIX_EXT 0x87C0
#define GL_VARIANT_EXT 0x87C1
#define GL_INVARIANT_EXT 0x87C2
#define GL_LOCAL_CONSTANT_EXT 0x87C3
#define GL_LOCAL_EXT 0x87C4
#define GL_MAX_VERTEX_SHADER_INSTRUCTIONS_EXT 0x87C5
#define GL_MAX_VERTEX_SHADER_VARIANTS_EXT 0x87C6
#define GL_MAX_VERTEX_SHADER_INVARIANTS_EXT 0x87C7
#define GL_MAX_VERTEX_SHADER_LOCAL_CONSTANTS_EXT 0x87C8
#define GL_MAX_VERTEX_SHADER_LOCALS_EXT 0x87C9
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_INSTRUCTIONS_EXT 0x87CA
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_VARIANTS_EXT 0x87CB
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_INVARIANTS_EXT 0x87CC
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_LOCAL_CONSTANTS_EXT 0x87CD
#define GL_MAX_OPTIMIZED_VERTEX_SHADER_LOCALS_EXT 0x87CE
#define GL_VERTEX_SHADER_INSTRUCTIONS_EXT 0x87CF
#define GL_VERTEX_SHADER_VARIANTS_EXT 0x87D0
#define GL_VERTEX_SHADER_INVARIANTS_EXT 0x87D1
#define GL_VERTEX_SHADER_LOCAL_CONSTANTS_EXT 0x87D2
#define GL_VERTEX_SHADER_LOCALS_EXT 0x87D3
#define GL_VERTEX_SHADER_OPTIMIZED_EXT 0x87D4
#define GL_X_EXT 0x87D5
#define GL_Y_EXT 0x87D6
#define GL_Z_EXT 0x87D7
#define GL_W_EXT 0x87D8
#define GL_NEGATIVE_X_EXT 0x87D9
#define GL_NEGATIVE_Y_EXT 0x87DA
#define GL_NEGATIVE_Z_EXT 0x87DB
#define GL_NEGATIVE_W_EXT 0x87DC
#define GL_ZERO_EXT 0x87DD
#define GL_ONE_EXT 0x87DE
#define GL_NEGATIVE_ONE_EXT 0x87DF
#define GL_NORMALIZED_RANGE_EXT 0x87E0
#define GL_FULL_RANGE_EXT 0x87E1
#define GL_CURRENT_VERTEX_EXT 0x87E2
#define GL_MVP_MATRIX_EXT 0x87E3
#define GL_VARIANT_VALUE_EXT 0x87E4
#define GL_VARIANT_DATATYPE_EXT 0x87E5
#define GL_VARIANT_ARRAY_STRIDE_EXT 0x87E6
#define GL_VARIANT_ARRAY_TYPE_EXT 0x87E7
#define GL_VARIANT_ARRAY_EXT 0x87E8
#define GL_VARIANT_ARRAY_POINTER_EXT 0x87E9
#define GL_INVARIANT_VALUE_EXT 0x87EA
#define GL_INVARIANT_DATATYPE_EXT 0x87EB
#define GL_LOCAL_CONSTANT_VALUE_EXT 0x87EC
#define GL_LOCAL_CONSTANT_DATATYPE_EXT 0x87ED

typedef void (GLAPIENTRY * PFNGLBEGINVERTEXSHADEREXTPROC) (void);
typedef GLuint (GLAPIENTRY * PFNGLBINDLIGHTPARAMETEREXTPROC) (GLenum light, GLenum value);
typedef GLuint (GLAPIENTRY * PFNGLBINDMATERIALPARAMETEREXTPROC) (GLenum face, GLenum value);
typedef GLuint (GLAPIENTRY * PFNGLBINDPARAMETEREXTPROC) (GLenum value);
typedef GLuint (GLAPIENTRY * PFNGLBINDTEXGENPARAMETEREXTPROC) (GLenum unit, GLenum coord, GLenum value);
typedef GLuint (GLAPIENTRY * PFNGLBINDTEXTUREUNITPARAMETEREXTPROC) (GLenum unit, GLenum value);
typedef void (GLAPIENTRY * PFNGLBINDVERTEXSHADEREXTPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLDELETEVERTEXSHADEREXTPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLDISABLEVARIANTCLIENTSTATEEXTPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLENABLEVARIANTCLIENTSTATEEXTPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLENDVERTEXSHADEREXTPROC) (void);
typedef void (GLAPIENTRY * PFNGLEXTRACTCOMPONENTEXTPROC) (GLuint res, GLuint src, GLuint num);
typedef GLuint (GLAPIENTRY * PFNGLGENSYMBOLSEXTPROC) (GLenum dataType, GLenum storageType, GLenum range, GLuint components);
typedef GLuint (GLAPIENTRY * PFNGLGENVERTEXSHADERSEXTPROC) (GLuint range);
typedef void (GLAPIENTRY * PFNGLGETINVARIANTBOOLEANVEXTPROC) (GLuint id, GLenum value, GLboolean *data);
typedef void (GLAPIENTRY * PFNGLGETINVARIANTFLOATVEXTPROC) (GLuint id, GLenum value, GLfloat *data);
typedef void (GLAPIENTRY * PFNGLGETINVARIANTINTEGERVEXTPROC) (GLuint id, GLenum value, GLint *data);
typedef void (GLAPIENTRY * PFNGLGETLOCALCONSTANTBOOLEANVEXTPROC) (GLuint id, GLenum value, GLboolean *data);
typedef void (GLAPIENTRY * PFNGLGETLOCALCONSTANTFLOATVEXTPROC) (GLuint id, GLenum value, GLfloat *data);
typedef void (GLAPIENTRY * PFNGLGETLOCALCONSTANTINTEGERVEXTPROC) (GLuint id, GLenum value, GLint *data);
typedef void (GLAPIENTRY * PFNGLGETVARIANTBOOLEANVEXTPROC) (GLuint id, GLenum value, GLboolean *data);
typedef void (GLAPIENTRY * PFNGLGETVARIANTFLOATVEXTPROC) (GLuint id, GLenum value, GLfloat *data);
typedef void (GLAPIENTRY * PFNGLGETVARIANTINTEGERVEXTPROC) (GLuint id, GLenum value, GLint *data);
typedef void (GLAPIENTRY * PFNGLGETVARIANTPOINTERVEXTPROC) (GLuint id, GLenum value, GLvoid **data);
typedef void (GLAPIENTRY * PFNGLINSERTCOMPONENTEXTPROC) (GLuint res, GLuint src, GLuint num);
typedef GLboolean (GLAPIENTRY * PFNGLISVARIANTENABLEDEXTPROC) (GLuint id, GLenum cap);
typedef void (GLAPIENTRY * PFNGLSETINVARIANTEXTPROC) (GLuint id, GLenum type, GLvoid *addr);
typedef void (GLAPIENTRY * PFNGLSETLOCALCONSTANTEXTPROC) (GLuint id, GLenum type, GLvoid *addr);
typedef void (GLAPIENTRY * PFNGLSHADEROP1EXTPROC) (GLenum op, GLuint res, GLuint arg1);
typedef void (GLAPIENTRY * PFNGLSHADEROP2EXTPROC) (GLenum op, GLuint res, GLuint arg1, GLuint arg2);
typedef void (GLAPIENTRY * PFNGLSHADEROP3EXTPROC) (GLenum op, GLuint res, GLuint arg1, GLuint arg2, GLuint arg3);
typedef void (GLAPIENTRY * PFNGLSWIZZLEEXTPROC) (GLuint res, GLuint in, GLenum outX, GLenum outY, GLenum outZ, GLenum outW);
typedef void (GLAPIENTRY * PFNGLVARIANTPOINTEREXTPROC) (GLuint id, GLenum type, GLuint stride, GLvoid *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTBVEXTPROC) (GLuint id, GLbyte *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTDVEXTPROC) (GLuint id, GLdouble *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTFVEXTPROC) (GLuint id, GLfloat *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTIVEXTPROC) (GLuint id, GLint *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTSVEXTPROC) (GLuint id, GLshort *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTUBVEXTPROC) (GLuint id, GLubyte *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTUIVEXTPROC) (GLuint id, GLuint *addr);
typedef void (GLAPIENTRY * PFNGLVARIANTUSVEXTPROC) (GLuint id, GLushort *addr);
typedef void (GLAPIENTRY * PFNGLWRITEMASKEXTPROC) (GLuint res, GLuint in, GLenum outX, GLenum outY, GLenum outZ, GLenum outW);

GLEWAPI PFNGLBEGINVERTEXSHADEREXTPROC glewBeginVertexShaderEXT;
GLEWAPI PFNGLBINDLIGHTPARAMETEREXTPROC glewBindLightParameterEXT;
GLEWAPI PFNGLBINDMATERIALPARAMETEREXTPROC glewBindMaterialParameterEXT;
GLEWAPI PFNGLBINDPARAMETEREXTPROC glewBindParameterEXT;
GLEWAPI PFNGLBINDTEXGENPARAMETEREXTPROC glewBindTexGenParameterEXT;
GLEWAPI PFNGLBINDTEXTUREUNITPARAMETEREXTPROC glewBindTextureUnitParameterEXT;
GLEWAPI PFNGLBINDVERTEXSHADEREXTPROC glewBindVertexShaderEXT;
GLEWAPI PFNGLDELETEVERTEXSHADEREXTPROC glewDeleteVertexShaderEXT;
GLEWAPI PFNGLDISABLEVARIANTCLIENTSTATEEXTPROC glewDisableVariantClientStateEXT;
GLEWAPI PFNGLENABLEVARIANTCLIENTSTATEEXTPROC glewEnableVariantClientStateEXT;
GLEWAPI PFNGLENDVERTEXSHADEREXTPROC glewEndVertexShaderEXT;
GLEWAPI PFNGLEXTRACTCOMPONENTEXTPROC glewExtractComponentEXT;
GLEWAPI PFNGLGENSYMBOLSEXTPROC glewGenSymbolsEXT;
GLEWAPI PFNGLGENVERTEXSHADERSEXTPROC glewGenVertexShadersEXT;
GLEWAPI PFNGLGETINVARIANTBOOLEANVEXTPROC glewGetInvariantBooleanvEXT;
GLEWAPI PFNGLGETINVARIANTFLOATVEXTPROC glewGetInvariantFloatvEXT;
GLEWAPI PFNGLGETINVARIANTINTEGERVEXTPROC glewGetInvariantIntegervEXT;
GLEWAPI PFNGLGETLOCALCONSTANTBOOLEANVEXTPROC glewGetLocalConstantBooleanvEXT;
GLEWAPI PFNGLGETLOCALCONSTANTFLOATVEXTPROC glewGetLocalConstantFloatvEXT;
GLEWAPI PFNGLGETLOCALCONSTANTINTEGERVEXTPROC glewGetLocalConstantIntegervEXT;
GLEWAPI PFNGLGETVARIANTBOOLEANVEXTPROC glewGetVariantBooleanvEXT;
GLEWAPI PFNGLGETVARIANTFLOATVEXTPROC glewGetVariantFloatvEXT;
GLEWAPI PFNGLGETVARIANTINTEGERVEXTPROC glewGetVariantIntegervEXT;
GLEWAPI PFNGLGETVARIANTPOINTERVEXTPROC glewGetVariantPointervEXT;
GLEWAPI PFNGLINSERTCOMPONENTEXTPROC glewInsertComponentEXT;
GLEWAPI PFNGLISVARIANTENABLEDEXTPROC glewIsVariantEnabledEXT;
GLEWAPI PFNGLSETINVARIANTEXTPROC glewSetInvariantEXT;
GLEWAPI PFNGLSETLOCALCONSTANTEXTPROC glewSetLocalConstantEXT;
GLEWAPI PFNGLSHADEROP1EXTPROC glewShaderOp1EXT;
GLEWAPI PFNGLSHADEROP2EXTPROC glewShaderOp2EXT;
GLEWAPI PFNGLSHADEROP3EXTPROC glewShaderOp3EXT;
GLEWAPI PFNGLSWIZZLEEXTPROC glewSwizzleEXT;
GLEWAPI PFNGLVARIANTPOINTEREXTPROC glewVariantPointerEXT;
GLEWAPI PFNGLVARIANTBVEXTPROC glewVariantbvEXT;
GLEWAPI PFNGLVARIANTDVEXTPROC glewVariantdvEXT;
GLEWAPI PFNGLVARIANTFVEXTPROC glewVariantfvEXT;
GLEWAPI PFNGLVARIANTIVEXTPROC glewVariantivEXT;
GLEWAPI PFNGLVARIANTSVEXTPROC glewVariantsvEXT;
GLEWAPI PFNGLVARIANTUBVEXTPROC glewVariantubvEXT;
GLEWAPI PFNGLVARIANTUIVEXTPROC glewVariantuivEXT;
GLEWAPI PFNGLVARIANTUSVEXTPROC glewVariantusvEXT;
GLEWAPI PFNGLWRITEMASKEXTPROC glewWriteMaskEXT;

#define glBeginVertexShaderEXT glewBeginVertexShaderEXT
#define glBindLightParameterEXT glewBindLightParameterEXT
#define glBindMaterialParameterEXT glewBindMaterialParameterEXT
#define glBindParameterEXT glewBindParameterEXT
#define glBindTexGenParameterEXT glewBindTexGenParameterEXT
#define glBindTextureUnitParameterEXT glewBindTextureUnitParameterEXT
#define glBindVertexShaderEXT glewBindVertexShaderEXT
#define glDeleteVertexShaderEXT glewDeleteVertexShaderEXT
#define glDisableVariantClientStateEXT glewDisableVariantClientStateEXT
#define glEnableVariantClientStateEXT glewEnableVariantClientStateEXT
#define glEndVertexShaderEXT glewEndVertexShaderEXT
#define glExtractComponentEXT glewExtractComponentEXT
#define glGenSymbolsEXT glewGenSymbolsEXT
#define glGenVertexShadersEXT glewGenVertexShadersEXT
#define glGetInvariantBooleanvEXT glewGetInvariantBooleanvEXT
#define glGetInvariantFloatvEXT glewGetInvariantFloatvEXT
#define glGetInvariantIntegervEXT glewGetInvariantIntegervEXT
#define glGetLocalConstantBooleanvEXT glewGetLocalConstantBooleanvEXT
#define glGetLocalConstantFloatvEXT glewGetLocalConstantFloatvEXT
#define glGetLocalConstantIntegervEXT glewGetLocalConstantIntegervEXT
#define glGetVariantBooleanvEXT glewGetVariantBooleanvEXT
#define glGetVariantFloatvEXT glewGetVariantFloatvEXT
#define glGetVariantIntegervEXT glewGetVariantIntegervEXT
#define glGetVariantPointervEXT glewGetVariantPointervEXT
#define glInsertComponentEXT glewInsertComponentEXT
#define glIsVariantEnabledEXT glewIsVariantEnabledEXT
#define glSetInvariantEXT glewSetInvariantEXT
#define glSetLocalConstantEXT glewSetLocalConstantEXT
#define glShaderOp1EXT glewShaderOp1EXT
#define glShaderOp2EXT glewShaderOp2EXT
#define glShaderOp3EXT glewShaderOp3EXT
#define glSwizzleEXT glewSwizzleEXT
#define glVariantPointerEXT glewVariantPointerEXT
#define glVariantbvEXT glewVariantbvEXT
#define glVariantdvEXT glewVariantdvEXT
#define glVariantfvEXT glewVariantfvEXT
#define glVariantivEXT glewVariantivEXT
#define glVariantsvEXT glewVariantsvEXT
#define glVariantubvEXT glewVariantubvEXT
#define glVariantuivEXT glewVariantuivEXT
#define glVariantusvEXT glewVariantusvEXT
#define glWriteMaskEXT glewWriteMaskEXT

GLEWAPI GLboolean GLEW_EXT_vertex_shader;

#endif /* GL_EXT_vertex_shader */

/* ------------------------ GL_EXT_vertex_weighting ------------------------ */

#ifndef GL_EXT_vertex_weighting
#define GL_EXT_vertex_weighting 1

#define GL_MODELVIEW0_STACK_DEPTH_EXT 0x0BA3
#define GL_MODELVIEW0_MATRIX_EXT 0x0BA6
#define GL_MODELVIEW0_EXT 0x1700
#define GL_MODELVIEW1_STACK_DEPTH_EXT 0x8502
#define GL_MODELVIEW1_MATRIX_EXT 0x8506
#define GL_VERTEX_WEIGHTING_EXT 0x8509
#define GL_MODELVIEW1_EXT 0x850A
#define GL_CURRENT_VERTEX_WEIGHT_EXT 0x850B
#define GL_VERTEX_WEIGHT_ARRAY_EXT 0x850C
#define GL_VERTEX_WEIGHT_ARRAY_SIZE_EXT 0x850D
#define GL_VERTEX_WEIGHT_ARRAY_TYPE_EXT 0x850E
#define GL_VERTEX_WEIGHT_ARRAY_STRIDE_EXT 0x850F
#define GL_VERTEX_WEIGHT_ARRAY_POINTER_EXT 0x8510

typedef void (GLAPIENTRY * PFNGLVERTEXWEIGHTPOINTEREXTPROC) (GLint size, GLenum type, GLsizei stride, void* pointer);
typedef void (GLAPIENTRY * PFNGLVERTEXWEIGHTFEXTPROC) (GLfloat weight);
typedef void (GLAPIENTRY * PFNGLVERTEXWEIGHTFVEXTPROC) (GLfloat* weight);

GLEWAPI PFNGLVERTEXWEIGHTPOINTEREXTPROC glewVertexWeightPointerEXT;
GLEWAPI PFNGLVERTEXWEIGHTFEXTPROC glewVertexWeightfEXT;
GLEWAPI PFNGLVERTEXWEIGHTFVEXTPROC glewVertexWeightfvEXT;

#define glVertexWeightPointerEXT glewVertexWeightPointerEXT
#define glVertexWeightfEXT glewVertexWeightfEXT
#define glVertexWeightfvEXT glewVertexWeightfvEXT

GLEWAPI GLboolean GLEW_EXT_vertex_weighting;

#endif /* GL_EXT_vertex_weighting */

/* --------------------- GL_HP_convolution_border_modes -------------------- */

#ifndef GL_HP_convolution_border_modes
#define GL_HP_convolution_border_modes 1

GLEWAPI GLboolean GLEW_HP_convolution_border_modes;

#endif /* GL_HP_convolution_border_modes */

/* ------------------------- GL_HP_image_transform ------------------------- */

#ifndef GL_HP_image_transform
#define GL_HP_image_transform 1

typedef void (GLAPIENTRY * PFNGLGETIMAGETRANSFORMPARAMETERFVHPPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETIMAGETRANSFORMPARAMETERIVHPPROC) (GLenum target, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLIMAGETRANSFORMPARAMETERFHPPROC) (GLenum target, GLenum pname, const GLfloat param);
typedef void (GLAPIENTRY * PFNGLIMAGETRANSFORMPARAMETERFVHPPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLIMAGETRANSFORMPARAMETERIHPPROC) (GLenum target, GLenum pname, const GLint param);
typedef void (GLAPIENTRY * PFNGLIMAGETRANSFORMPARAMETERIVHPPROC) (GLenum target, GLenum pname, const GLint* params);

GLEWAPI PFNGLGETIMAGETRANSFORMPARAMETERFVHPPROC glewGetImageTransformParameterfvHP;
GLEWAPI PFNGLGETIMAGETRANSFORMPARAMETERIVHPPROC glewGetImageTransformParameterivHP;
GLEWAPI PFNGLIMAGETRANSFORMPARAMETERFHPPROC glewImageTransformParameterfHP;
GLEWAPI PFNGLIMAGETRANSFORMPARAMETERFVHPPROC glewImageTransformParameterfvHP;
GLEWAPI PFNGLIMAGETRANSFORMPARAMETERIHPPROC glewImageTransformParameteriHP;
GLEWAPI PFNGLIMAGETRANSFORMPARAMETERIVHPPROC glewImageTransformParameterivHP;

#define glGetImageTransformParameterfvHP glewGetImageTransformParameterfvHP
#define glGetImageTransformParameterivHP glewGetImageTransformParameterivHP
#define glImageTransformParameterfHP glewImageTransformParameterfHP
#define glImageTransformParameterfvHP glewImageTransformParameterfvHP
#define glImageTransformParameteriHP glewImageTransformParameteriHP
#define glImageTransformParameterivHP glewImageTransformParameterivHP

GLEWAPI GLboolean GLEW_HP_image_transform;

#endif /* GL_HP_image_transform */

/* -------------------------- GL_HP_occlusion_test ------------------------- */

#ifndef GL_HP_occlusion_test
#define GL_HP_occlusion_test 1

#define GL_OCCLUSION_TEST_HP 0x8165
#define GL_OCCLUSION_TEST_RESULT_HP 0x8166

GLEWAPI GLboolean GLEW_HP_occlusion_test;

#endif /* GL_HP_occlusion_test */

/* ------------------------- GL_HP_texture_lighting ------------------------ */

#ifndef GL_HP_texture_lighting
#define GL_HP_texture_lighting 1

GLEWAPI GLboolean GLEW_HP_texture_lighting;

#endif /* GL_HP_texture_lighting */

/* --------------------------- GL_IBM_cull_vertex -------------------------- */

#ifndef GL_IBM_cull_vertex
#define GL_IBM_cull_vertex 1

#define GL_CULL_VERTEX_IBM 103050

GLEWAPI GLboolean GLEW_IBM_cull_vertex;

#endif /* GL_IBM_cull_vertex */

/* ---------------------- GL_IBM_multimode_draw_arrays --------------------- */

#ifndef GL_IBM_multimode_draw_arrays
#define GL_IBM_multimode_draw_arrays 1

typedef void (GLAPIENTRY * PFNGLMULTIMODEDRAWARRAYSIBMPROC) (GLenum* mode, GLint *first, GLsizei *count, GLsizei primcount, GLint modestride);
typedef void (GLAPIENTRY * PFNGLMULTIMODEDRAWELEMENTSIBMPROC) (GLenum* mode, GLsizei *count, GLenum type, const GLvoid **indices, GLsizei primcount, GLint modestride);

GLEWAPI PFNGLMULTIMODEDRAWARRAYSIBMPROC glewMultiModeDrawArraysIBM;
GLEWAPI PFNGLMULTIMODEDRAWELEMENTSIBMPROC glewMultiModeDrawElementsIBM;

#define glMultiModeDrawArraysIBM glewMultiModeDrawArraysIBM
#define glMultiModeDrawElementsIBM glewMultiModeDrawElementsIBM

GLEWAPI GLboolean GLEW_IBM_multimode_draw_arrays;

#endif /* GL_IBM_multimode_draw_arrays */

/* ------------------------- GL_IBM_rasterpos_clip ------------------------- */

#ifndef GL_IBM_rasterpos_clip
#define GL_IBM_rasterpos_clip 1

#define GL_RASTER_POSITION_UNCLIPPED_IBM 103010

GLEWAPI GLboolean GLEW_IBM_rasterpos_clip;

#endif /* GL_IBM_rasterpos_clip */

/* --------------------------- GL_IBM_static_data -------------------------- */

#ifndef GL_IBM_static_data
#define GL_IBM_static_data 1

#define GL_ALL_STATIC_DATA_IBM 103060
#define GL_STATIC_VERTEX_ARRAY_IBM 103061

GLEWAPI GLboolean GLEW_IBM_static_data;

#endif /* GL_IBM_static_data */

/* --------------------- GL_IBM_texture_mirrored_repeat -------------------- */

#ifndef GL_IBM_texture_mirrored_repeat
#define GL_IBM_texture_mirrored_repeat 1

#define GL_MIRRORED_REPEAT_IBM 0x8370

GLEWAPI GLboolean GLEW_IBM_texture_mirrored_repeat;

#endif /* GL_IBM_texture_mirrored_repeat */

/* ----------------------- GL_IBM_vertex_array_lists ----------------------- */

#ifndef GL_IBM_vertex_array_lists
#define GL_IBM_vertex_array_lists 1

#define GL_VERTEX_ARRAY_LIST_IBM 103070
#define GL_NORMAL_ARRAY_LIST_IBM 103071
#define GL_COLOR_ARRAY_LIST_IBM 103072
#define GL_INDEX_ARRAY_LIST_IBM 103073
#define GL_TEXTURE_COORD_ARRAY_LIST_IBM 103074
#define GL_EDGE_FLAG_ARRAY_LIST_IBM 103075
#define GL_FOG_COORDINATE_ARRAY_LIST_IBM 103076
#define GL_SECONDARY_COLOR_ARRAY_LIST_IBM 103077
#define GL_VERTEX_ARRAY_LIST_STRIDE_IBM 103080
#define GL_NORMAL_ARRAY_LIST_STRIDE_IBM 103081
#define GL_COLOR_ARRAY_LIST_STRIDE_IBM 103082
#define GL_INDEX_ARRAY_LIST_STRIDE_IBM 103083
#define GL_TEXTURE_COORD_ARRAY_LIST_STRIDE_IBM 103084
#define GL_EDGE_FLAG_ARRAY_LIST_STRIDE_IBM 103085
#define GL_FOG_COORDINATE_ARRAY_LIST_STRIDE_IBM 103086
#define GL_SECONDARY_COLOR_ARRAY_LIST_STRIDE_IBM 103087

typedef void (GLAPIENTRY * PFNGLCOLORPOINTERLISTIBMPROC) (GLint size, GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLEDGEFLAGPOINTERLISTIBMPROC) (GLint stride, const GLboolean ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLFOGCOORDPOINTERLISTIBMPROC) (GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLINDEXPOINTERLISTIBMPROC) (GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLNORMALPOINTERLISTIBMPROC) (GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLORPOINTERLISTIBMPROC) (GLint size, GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLTEXCOORDPOINTERLISTIBMPROC) (GLint size, GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);
typedef void (GLAPIENTRY * PFNGLVERTEXPOINTERLISTIBMPROC) (GLint size, GLenum type, GLint stride, const GLvoid ** pointer, GLint ptrstride);

GLEWAPI PFNGLCOLORPOINTERLISTIBMPROC glewColorPointerListIBM;
GLEWAPI PFNGLEDGEFLAGPOINTERLISTIBMPROC glewEdgeFlagPointerListIBM;
GLEWAPI PFNGLFOGCOORDPOINTERLISTIBMPROC glewFogCoordPointerListIBM;
GLEWAPI PFNGLINDEXPOINTERLISTIBMPROC glewIndexPointerListIBM;
GLEWAPI PFNGLNORMALPOINTERLISTIBMPROC glewNormalPointerListIBM;
GLEWAPI PFNGLSECONDARYCOLORPOINTERLISTIBMPROC glewSecondaryColorPointerListIBM;
GLEWAPI PFNGLTEXCOORDPOINTERLISTIBMPROC glewTexCoordPointerListIBM;
GLEWAPI PFNGLVERTEXPOINTERLISTIBMPROC glewVertexPointerListIBM;

#define glColorPointerListIBM glewColorPointerListIBM
#define glEdgeFlagPointerListIBM glewEdgeFlagPointerListIBM
#define glFogCoordPointerListIBM glewFogCoordPointerListIBM
#define glIndexPointerListIBM glewIndexPointerListIBM
#define glNormalPointerListIBM glewNormalPointerListIBM
#define glSecondaryColorPointerListIBM glewSecondaryColorPointerListIBM
#define glTexCoordPointerListIBM glewTexCoordPointerListIBM
#define glVertexPointerListIBM glewVertexPointerListIBM

GLEWAPI GLboolean GLEW_IBM_vertex_array_lists;

#endif /* GL_IBM_vertex_array_lists */

/* -------------------------- GL_INGR_color_clamp -------------------------- */

#ifndef GL_INGR_color_clamp
#define GL_INGR_color_clamp 1

#define GL_RED_MIN_CLAMP_INGR 0x8560
#define GL_GREEN_MIN_CLAMP_INGR 0x8561
#define GL_BLUE_MIN_CLAMP_INGR 0x8562
#define GL_ALPHA_MIN_CLAMP_INGR 0x8563
#define GL_RED_MAX_CLAMP_INGR 0x8564
#define GL_GREEN_MAX_CLAMP_INGR 0x8565
#define GL_BLUE_MAX_CLAMP_INGR 0x8566
#define GL_ALPHA_MAX_CLAMP_INGR 0x8567

GLEWAPI GLboolean GLEW_INGR_color_clamp;

#endif /* GL_INGR_color_clamp */

/* ------------------------- GL_INGR_interlace_read ------------------------ */

#ifndef GL_INGR_interlace_read
#define GL_INGR_interlace_read 1

#define GL_INTERLACE_READ_INGR 0x8568

GLEWAPI GLboolean GLEW_INGR_interlace_read;

#endif /* GL_INGR_interlace_read */

/* ------------------------ GL_INTEL_parallel_arrays ----------------------- */

#ifndef GL_INTEL_parallel_arrays
#define GL_INTEL_parallel_arrays 1

#define GL_PARALLEL_ARRAYS_INTEL 0x83F4
#define GL_VERTEX_ARRAY_PARALLEL_POINTERS_INTEL 0x83F5
#define GL_NORMAL_ARRAY_PARALLEL_POINTERS_INTEL 0x83F6
#define GL_COLOR_ARRAY_PARALLEL_POINTERS_INTEL 0x83F7
#define GL_TEXTURE_COORD_ARRAY_PARALLEL_POINTERS_INTEL 0x83F8

typedef void (GLAPIENTRY * PFNGLCOLORPOINTERVINTELPROC) (GLint size, GLenum type, const void** pointer);
typedef void (GLAPIENTRY * PFNGLNORMALPOINTERVINTELPROC) (GLenum type, const void** pointer);
typedef void (GLAPIENTRY * PFNGLTEXCOORDPOINTERVINTELPROC) (GLint size, GLenum type, const void** pointer);
typedef void (GLAPIENTRY * PFNGLVERTEXPOINTERVINTELPROC) (GLint size, GLenum type, const void** pointer);

GLEWAPI PFNGLCOLORPOINTERVINTELPROC glewColorPointervINTEL;
GLEWAPI PFNGLNORMALPOINTERVINTELPROC glewNormalPointervINTEL;
GLEWAPI PFNGLTEXCOORDPOINTERVINTELPROC glewTexCoordPointervINTEL;
GLEWAPI PFNGLVERTEXPOINTERVINTELPROC glewVertexPointervINTEL;

#define glColorPointervINTEL glewColorPointervINTEL
#define glNormalPointervINTEL glewNormalPointervINTEL
#define glTexCoordPointervINTEL glewTexCoordPointervINTEL
#define glVertexPointervINTEL glewVertexPointervINTEL

GLEWAPI GLboolean GLEW_INTEL_parallel_arrays;

#endif /* GL_INTEL_parallel_arrays */

/* ------------------------ GL_INTEL_texture_scissor ----------------------- */

#ifndef GL_INTEL_texture_scissor
#define GL_INTEL_texture_scissor 1

typedef void (GLAPIENTRY * PFNGLTEXSCISSORFUNCINTELPROC) (GLenum target, GLenum lfunc, GLenum hfunc);
typedef void (GLAPIENTRY * PFNGLTEXSCISSORINTELPROC) (GLenum target, GLclampf tlow, GLclampf thigh);

GLEWAPI PFNGLTEXSCISSORFUNCINTELPROC glewTexScissorFuncINTEL;
GLEWAPI PFNGLTEXSCISSORINTELPROC glewTexScissorINTEL;

#define glTexScissorFuncINTEL glewTexScissorFuncINTEL
#define glTexScissorINTEL glewTexScissorINTEL

GLEWAPI GLboolean GLEW_INTEL_texture_scissor;

#endif /* GL_INTEL_texture_scissor */

/* ------------------------- GL_MESA_resize_buffers ------------------------ */

#ifndef GL_MESA_resize_buffers
#define GL_MESA_resize_buffers 1

typedef void (GLAPIENTRY * PFNGLRESIZEBUFFERSMESAPROC) (void);

GLEWAPI PFNGLRESIZEBUFFERSMESAPROC glewResizeBuffersMESA;

#define glResizeBuffersMESA glewResizeBuffersMESA

GLEWAPI GLboolean GLEW_MESA_resize_buffers;

#endif /* GL_MESA_resize_buffers */

/* --------------------------- GL_MESA_window_pos -------------------------- */

#ifndef GL_MESA_window_pos
#define GL_MESA_window_pos 1

typedef void (GLAPIENTRY * PFNGLWINDOWPOS2DMESAPROC) (GLdouble x, GLdouble y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2DVMESAPROC) (const GLdouble* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2FMESAPROC) (GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2FVMESAPROC) (const GLfloat* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2IMESAPROC) (GLint x, GLint y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2IVMESAPROC) (const GLint* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2SMESAPROC) (GLshort x, GLshort y);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS2SVMESAPROC) (const GLshort* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3DMESAPROC) (GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3DVMESAPROC) (const GLdouble* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3FMESAPROC) (GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3FVMESAPROC) (const GLfloat* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3IMESAPROC) (GLint x, GLint y, GLint z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3IVMESAPROC) (const GLint* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3SMESAPROC) (GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS3SVMESAPROC) (const GLshort* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4DMESAPROC) (GLdouble x, GLdouble y, GLdouble z, GLdouble);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4DVMESAPROC) (const GLdouble* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4FMESAPROC) (GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4FVMESAPROC) (const GLfloat* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4IMESAPROC) (GLint x, GLint y, GLint z, GLint w);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4IVMESAPROC) (const GLint* p);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4SMESAPROC) (GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (GLAPIENTRY * PFNGLWINDOWPOS4SVMESAPROC) (const GLshort* p);

GLEWAPI PFNGLWINDOWPOS2DMESAPROC glewWindowPos2dMESA;
GLEWAPI PFNGLWINDOWPOS2DVMESAPROC glewWindowPos2dvMESA;
GLEWAPI PFNGLWINDOWPOS2FMESAPROC glewWindowPos2fMESA;
GLEWAPI PFNGLWINDOWPOS2FVMESAPROC glewWindowPos2fvMESA;
GLEWAPI PFNGLWINDOWPOS2IMESAPROC glewWindowPos2iMESA;
GLEWAPI PFNGLWINDOWPOS2IVMESAPROC glewWindowPos2ivMESA;
GLEWAPI PFNGLWINDOWPOS2SMESAPROC glewWindowPos2sMESA;
GLEWAPI PFNGLWINDOWPOS2SVMESAPROC glewWindowPos2svMESA;
GLEWAPI PFNGLWINDOWPOS3DMESAPROC glewWindowPos3dMESA;
GLEWAPI PFNGLWINDOWPOS3DVMESAPROC glewWindowPos3dvMESA;
GLEWAPI PFNGLWINDOWPOS3FMESAPROC glewWindowPos3fMESA;
GLEWAPI PFNGLWINDOWPOS3FVMESAPROC glewWindowPos3fvMESA;
GLEWAPI PFNGLWINDOWPOS3IMESAPROC glewWindowPos3iMESA;
GLEWAPI PFNGLWINDOWPOS3IVMESAPROC glewWindowPos3ivMESA;
GLEWAPI PFNGLWINDOWPOS3SMESAPROC glewWindowPos3sMESA;
GLEWAPI PFNGLWINDOWPOS3SVMESAPROC glewWindowPos3svMESA;
GLEWAPI PFNGLWINDOWPOS4DMESAPROC glewWindowPos4dMESA;
GLEWAPI PFNGLWINDOWPOS4DVMESAPROC glewWindowPos4dvMESA;
GLEWAPI PFNGLWINDOWPOS4FMESAPROC glewWindowPos4fMESA;
GLEWAPI PFNGLWINDOWPOS4FVMESAPROC glewWindowPos4fvMESA;
GLEWAPI PFNGLWINDOWPOS4IMESAPROC glewWindowPos4iMESA;
GLEWAPI PFNGLWINDOWPOS4IVMESAPROC glewWindowPos4ivMESA;
GLEWAPI PFNGLWINDOWPOS4SMESAPROC glewWindowPos4sMESA;
GLEWAPI PFNGLWINDOWPOS4SVMESAPROC glewWindowPos4svMESA;

#define glWindowPos2dMESA glewWindowPos2dMESA
#define glWindowPos2dvMESA glewWindowPos2dvMESA
#define glWindowPos2fMESA glewWindowPos2fMESA
#define glWindowPos2fvMESA glewWindowPos2fvMESA
#define glWindowPos2iMESA glewWindowPos2iMESA
#define glWindowPos2ivMESA glewWindowPos2ivMESA
#define glWindowPos2sMESA glewWindowPos2sMESA
#define glWindowPos2svMESA glewWindowPos2svMESA
#define glWindowPos3dMESA glewWindowPos3dMESA
#define glWindowPos3dvMESA glewWindowPos3dvMESA
#define glWindowPos3fMESA glewWindowPos3fMESA
#define glWindowPos3fvMESA glewWindowPos3fvMESA
#define glWindowPos3iMESA glewWindowPos3iMESA
#define glWindowPos3ivMESA glewWindowPos3ivMESA
#define glWindowPos3sMESA glewWindowPos3sMESA
#define glWindowPos3svMESA glewWindowPos3svMESA
#define glWindowPos4dMESA glewWindowPos4dMESA
#define glWindowPos4dvMESA glewWindowPos4dvMESA
#define glWindowPos4fMESA glewWindowPos4fMESA
#define glWindowPos4fvMESA glewWindowPos4fvMESA
#define glWindowPos4iMESA glewWindowPos4iMESA
#define glWindowPos4ivMESA glewWindowPos4ivMESA
#define glWindowPos4sMESA glewWindowPos4sMESA
#define glWindowPos4svMESA glewWindowPos4svMESA

GLEWAPI GLboolean GLEW_MESA_window_pos;

#endif /* GL_MESA_window_pos */

/* --------------------------- GL_NV_blend_square -------------------------- */

#ifndef GL_NV_blend_square
#define GL_NV_blend_square 1

GLEWAPI GLboolean GLEW_NV_blend_square;

#endif /* GL_NV_blend_square */

/* ----------------------- GL_NV_copy_depth_to_color ----------------------- */

#ifndef GL_NV_copy_depth_to_color
#define GL_NV_copy_depth_to_color 1

#define GL_DEPTH_STENCIL_TO_RGBA_NV 0x886E
#define GL_DEPTH_STENCIL_TO_BGRA_NV 0x886F

GLEWAPI GLboolean GLEW_NV_copy_depth_to_color;

#endif /* GL_NV_copy_depth_to_color */

/* --------------------------- GL_NV_depth_clamp --------------------------- */

#ifndef GL_NV_depth_clamp
#define GL_NV_depth_clamp 1

#define GL_DEPTH_CLAMP_NV 0x864F

GLEWAPI GLboolean GLEW_NV_depth_clamp;

#endif /* GL_NV_depth_clamp */

/* -------------------------- GL_NV_element_array -------------------------- */

#ifndef GL_NV_element_array
#define GL_NV_element_array 1

#define GL_ELEMENT_ARRAY_TYPE_NV 0x8769
#define GL_ELEMENT_ARRAY_POINTER_NV 0x876A

typedef void (GLAPIENTRY * PFNGLDRAWELEMENTARRAYNVPROC) (GLenum mode, GLint first, GLsizei count);
typedef void (GLAPIENTRY * PFNGLDRAWRANGEELEMENTARRAYNVPROC) (GLenum mode, GLuint start, GLuint end, GLint first, GLsizei count);
typedef void (GLAPIENTRY * PFNGLELEMENTPOINTERNVPROC) (GLenum type, const GLvoid *pointer);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWELEMENTARRAYNVPROC) (GLenum mode, const GLint *first, const GLsizei *count, GLsizei primcount);
typedef void (GLAPIENTRY * PFNGLMULTIDRAWRANGEELEMENTARRAYNVPROC) (GLenum mode, GLuint start, GLuint end, const GLint *first, const GLsizei *count, GLsizei primcount);

GLEWAPI PFNGLDRAWELEMENTARRAYNVPROC glewDrawElementArrayNV;
GLEWAPI PFNGLDRAWRANGEELEMENTARRAYNVPROC glewDrawRangeElementArrayNV;
GLEWAPI PFNGLELEMENTPOINTERNVPROC glewElementPointerNV;
GLEWAPI PFNGLMULTIDRAWELEMENTARRAYNVPROC glewMultiDrawElementArrayNV;
GLEWAPI PFNGLMULTIDRAWRANGEELEMENTARRAYNVPROC glewMultiDrawRangeElementArrayNV;

#define glDrawElementArrayNV glewDrawElementArrayNV
#define glDrawRangeElementArrayNV glewDrawRangeElementArrayNV
#define glElementPointerNV glewElementPointerNV
#define glMultiDrawElementArrayNV glewMultiDrawElementArrayNV
#define glMultiDrawRangeElementArrayNV glewMultiDrawRangeElementArrayNV

GLEWAPI GLboolean GLEW_NV_element_array;

#endif /* GL_NV_element_array */

/* ---------------------------- GL_NV_evaluators --------------------------- */

#ifndef GL_NV_evaluators
#define GL_NV_evaluators 1

#define GL_EVAL_2D_NV 0x86C0
#define GL_EVAL_TRIANGULAR_2D_NV 0x86C1
#define GL_MAP_TESSELLATION_NV 0x86C2
#define GL_MAP_ATTRIB_U_ORDER_NV 0x86C3
#define GL_MAP_ATTRIB_V_ORDER_NV 0x86C4
#define GL_EVAL_FRACTIONAL_TESSELLATION_NV 0x86C5
#define GL_EVAL_VERTEX_ATTRIB0_NV 0x86C6
#define GL_EVAL_VERTEX_ATTRIB1_NV 0x86C7
#define GL_EVAL_VERTEX_ATTRIB2_NV 0x86C8
#define GL_EVAL_VERTEX_ATTRIB3_NV 0x86C9
#define GL_EVAL_VERTEX_ATTRIB4_NV 0x86CA
#define GL_EVAL_VERTEX_ATTRIB5_NV 0x86CB
#define GL_EVAL_VERTEX_ATTRIB6_NV 0x86CC
#define GL_EVAL_VERTEX_ATTRIB7_NV 0x86CD
#define GL_EVAL_VERTEX_ATTRIB8_NV 0x86CE
#define GL_EVAL_VERTEX_ATTRIB9_NV 0x86CF
#define GL_EVAL_VERTEX_ATTRIB10_NV 0x86D0
#define GL_EVAL_VERTEX_ATTRIB11_NV 0x86D1
#define GL_EVAL_VERTEX_ATTRIB12_NV 0x86D2
#define GL_EVAL_VERTEX_ATTRIB13_NV 0x86D3
#define GL_EVAL_VERTEX_ATTRIB14_NV 0x86D4
#define GL_EVAL_VERTEX_ATTRIB15_NV 0x86D5
#define GL_MAX_MAP_TESSELLATION_NV 0x86D6
#define GL_MAX_RATIONAL_EVAL_ORDER_NV 0x86D7

typedef void (GLAPIENTRY * PFNGLEVALMAPSNVPROC) (GLenum target, GLenum mode);
typedef void (GLAPIENTRY * PFNGLGETMAPATTRIBPARAMETERFVNVPROC) (GLenum target, GLuint index, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETMAPATTRIBPARAMETERIVNVPROC) (GLenum target, GLuint index, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETMAPCONTROLPOINTSNVPROC) (GLenum target, GLuint index, GLenum type, GLsizei ustride, GLsizei vstride, GLboolean packed, void* points);
typedef void (GLAPIENTRY * PFNGLGETMAPPARAMETERFVNVPROC) (GLenum target, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETMAPPARAMETERIVNVPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLMAPCONTROLPOINTSNVPROC) (GLenum target, GLuint index, GLenum type, GLsizei ustride, GLsizei vstride, GLint uorder, GLint vorder, GLboolean packed, const void* points);
typedef void (GLAPIENTRY * PFNGLMAPPARAMETERFVNVPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLMAPPARAMETERIVNVPROC) (GLenum target, GLenum pname, const GLint* params);

GLEWAPI PFNGLEVALMAPSNVPROC glewEvalMapsNV;
GLEWAPI PFNGLGETMAPATTRIBPARAMETERFVNVPROC glewGetMapAttribParameterfvNV;
GLEWAPI PFNGLGETMAPATTRIBPARAMETERIVNVPROC glewGetMapAttribParameterivNV;
GLEWAPI PFNGLGETMAPCONTROLPOINTSNVPROC glewGetMapControlPointsNV;
GLEWAPI PFNGLGETMAPPARAMETERFVNVPROC glewGetMapParameterfvNV;
GLEWAPI PFNGLGETMAPPARAMETERIVNVPROC glewGetMapParameterivNV;
GLEWAPI PFNGLMAPCONTROLPOINTSNVPROC glewMapControlPointsNV;
GLEWAPI PFNGLMAPPARAMETERFVNVPROC glewMapParameterfvNV;
GLEWAPI PFNGLMAPPARAMETERIVNVPROC glewMapParameterivNV;

#define glEvalMapsNV glewEvalMapsNV
#define glGetMapAttribParameterfvNV glewGetMapAttribParameterfvNV
#define glGetMapAttribParameterivNV glewGetMapAttribParameterivNV
#define glGetMapControlPointsNV glewGetMapControlPointsNV
#define glGetMapParameterfvNV glewGetMapParameterfvNV
#define glGetMapParameterivNV glewGetMapParameterivNV
#define glMapControlPointsNV glewMapControlPointsNV
#define glMapParameterfvNV glewMapParameterfvNV
#define glMapParameterivNV glewMapParameterivNV

GLEWAPI GLboolean GLEW_NV_evaluators;

#endif /* GL_NV_evaluators */

/* ------------------------------ GL_NV_fence ------------------------------ */

#ifndef GL_NV_fence
#define GL_NV_fence 1

#define GL_ALL_COMPLETED_NV 0x84F2
#define GL_FENCE_STATUS_NV 0x84F3
#define GL_FENCE_CONDITION_NV 0x84F4

typedef void (GLAPIENTRY * PFNGLDELETEFENCESNVPROC) (GLsizei n, const GLuint* fences);
typedef void (GLAPIENTRY * PFNGLFINISHFENCENVPROC) (GLuint fence);
typedef void (GLAPIENTRY * PFNGLGENFENCESNVPROC) (GLsizei n, GLuint* fences);
typedef void (GLAPIENTRY * PFNGLGETFENCEIVNVPROC) (GLuint fence, GLenum pname, GLint* params);
typedef GLboolean (GLAPIENTRY * PFNGLISFENCENVPROC) (GLuint fence);
typedef void (GLAPIENTRY * PFNGLSETFENCENVPROC) (GLuint fence, GLenum condition);
typedef GLboolean (GLAPIENTRY * PFNGLTESTFENCENVPROC) (GLuint fence);

GLEWAPI PFNGLDELETEFENCESNVPROC glewDeleteFencesNV;
GLEWAPI PFNGLFINISHFENCENVPROC glewFinishFenceNV;
GLEWAPI PFNGLGENFENCESNVPROC glewGenFencesNV;
GLEWAPI PFNGLGETFENCEIVNVPROC glewGetFenceivNV;
GLEWAPI PFNGLISFENCENVPROC glewIsFenceNV;
GLEWAPI PFNGLSETFENCENVPROC glewSetFenceNV;
GLEWAPI PFNGLTESTFENCENVPROC glewTestFenceNV;

#define glDeleteFencesNV glewDeleteFencesNV
#define glFinishFenceNV glewFinishFenceNV
#define glGenFencesNV glewGenFencesNV
#define glGetFenceivNV glewGetFenceivNV
#define glIsFenceNV glewIsFenceNV
#define glSetFenceNV glewSetFenceNV
#define glTestFenceNV glewTestFenceNV

GLEWAPI GLboolean GLEW_NV_fence;

#endif /* GL_NV_fence */

/* --------------------------- GL_NV_float_buffer -------------------------- */

#ifndef GL_NV_float_buffer
#define GL_NV_float_buffer 1

#define GL_FLOAT_R_NV 0x8880
#define GL_FLOAT_RG_NV 0x8881
#define GL_FLOAT_RGB_NV 0x8882
#define GL_FLOAT_RGBA_NV 0x8883
#define GL_FLOAT_R16_NV 0x8884
#define GL_FLOAT_R32_NV 0x8885
#define GL_FLOAT_RG16_NV 0x8886
#define GL_FLOAT_RG32_NV 0x8887
#define GL_FLOAT_RGB16_NV 0x8888
#define GL_FLOAT_RGB32_NV 0x8889
#define GL_FLOAT_RGBA16_NV 0x888A
#define GL_FLOAT_RGBA32_NV 0x888B
#define GL_TEXTURE_FLOAT_COMPONENTS_NV 0x888C
#define GL_FLOAT_CLEAR_COLOR_VALUE_NV 0x888D
#define GL_FLOAT_RGBA_MODE_NV 0x888E

GLEWAPI GLboolean GLEW_NV_float_buffer;

#endif /* GL_NV_float_buffer */

/* --------------------------- GL_NV_fog_distance -------------------------- */

#ifndef GL_NV_fog_distance
#define GL_NV_fog_distance 1

#define GL_FOG_DISTANCE_MODE_NV 0x855A
#define GL_EYE_RADIAL_NV 0x855B
#define GL_EYE_PLANE_ABSOLUTE_NV 0x855C

GLEWAPI GLboolean GLEW_NV_fog_distance;

#endif /* GL_NV_fog_distance */

/* ------------------------- GL_NV_fragment_program ------------------------ */

#ifndef GL_NV_fragment_program
#define GL_NV_fragment_program 1

#define GL_MAX_FRAGMENT_PROGRAM_LOCAL_PARAMETERS_NV 0x8868
#define GL_FRAGMENT_PROGRAM_NV 0x8870
#define GL_MAX_TEXTURE_COORDS_NV 0x8871
#define GL_MAX_TEXTURE_IMAGE_UNITS_NV 0x8872
#define GL_FRAGMENT_PROGRAM_BINDING_NV 0x8873
#define GL_PROGRAM_ERROR_STRING_NV 0x8874

typedef void (GLAPIENTRY * PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC) (GLuint id, GLsizei len, const GLubyte* name, GLdouble *params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC) (GLuint id, GLsizei len, const GLubyte* name, GLfloat *params);
typedef void (GLAPIENTRY * PFNGLPROGRAMNAMEDPARAMETER4DNVPROC) (GLuint id, GLsizei len, const GLubyte* name, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC) (GLuint id, GLsizei len, const GLubyte* name, const GLdouble v[]);
typedef void (GLAPIENTRY * PFNGLPROGRAMNAMEDPARAMETER4FNVPROC) (GLuint id, GLsizei len, const GLubyte* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC) (GLuint id, GLsizei len, const GLubyte* name, const GLfloat v[]);

GLEWAPI PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC glewGetProgramNamedParameterdvNV;
GLEWAPI PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC glewGetProgramNamedParameterfvNV;
GLEWAPI PFNGLPROGRAMNAMEDPARAMETER4DNVPROC glewProgramNamedParameter4dNV;
GLEWAPI PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC glewProgramNamedParameter4dvNV;
GLEWAPI PFNGLPROGRAMNAMEDPARAMETER4FNVPROC glewProgramNamedParameter4fNV;
GLEWAPI PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC glewProgramNamedParameter4fvNV;

#define glGetProgramNamedParameterdvNV glewGetProgramNamedParameterdvNV
#define glGetProgramNamedParameterfvNV glewGetProgramNamedParameterfvNV
#define glProgramNamedParameter4dNV glewProgramNamedParameter4dNV
#define glProgramNamedParameter4dvNV glewProgramNamedParameter4dvNV
#define glProgramNamedParameter4fNV glewProgramNamedParameter4fNV
#define glProgramNamedParameter4fvNV glewProgramNamedParameter4fvNV

GLEWAPI GLboolean GLEW_NV_fragment_program;

#endif /* GL_NV_fragment_program */

/* ---------------------------- GL_NV_half_float --------------------------- */

#ifndef GL_NV_half_float
#define GL_NV_half_float 1

#define GL_HALF_FLOAT_NV 0x140B

typedef unsigned short GLhalf;

typedef void (GLAPIENTRY * PFNGLCOLOR3HNVPROC) (GLuint red, GLuint green, GLuint blue);
typedef void (GLAPIENTRY * PFNGLCOLOR3HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLCOLOR4HNVPROC) (GLuint red, GLuint green, GLuint blue, GLuint alpha);
typedef void (GLAPIENTRY * PFNGLCOLOR4HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLFOGCOORDHNVPROC) (GLuint fog);
typedef void (GLAPIENTRY * PFNGLFOGCOORDHVNVPROC) (const GLuint* fog);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1HNVPROC) (GLenum target, GLuint s);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1HVNVPROC) (GLenum target, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2HNVPROC) (GLenum target, GLuint s, GLuint t);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2HVNVPROC) (GLenum target, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3HNVPROC) (GLenum target, GLuint s, GLuint t, GLuint r);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3HVNVPROC) (GLenum target, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4HNVPROC) (GLenum target, GLuint s, GLuint t, GLuint r, GLuint q);
typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD4HVNVPROC) (GLenum target, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLNORMAL3HNVPROC) (GLuint nx, GLuint ny, GLuint nz);
typedef void (GLAPIENTRY * PFNGLNORMAL3HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3HNVPROC) (GLuint red, GLuint green, GLuint blue);
typedef void (GLAPIENTRY * PFNGLSECONDARYCOLOR3HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD1HNVPROC) (GLuint s);
typedef void (GLAPIENTRY * PFNGLTEXCOORD1HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2HNVPROC) (GLuint s, GLuint t);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD3HNVPROC) (GLuint s, GLuint t, GLuint r);
typedef void (GLAPIENTRY * PFNGLTEXCOORD3HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD4HNVPROC) (GLuint s, GLuint t, GLuint r, GLuint q);
typedef void (GLAPIENTRY * PFNGLTEXCOORD4HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEX2HNVPROC) (GLuint x, GLuint y);
typedef void (GLAPIENTRY * PFNGLVERTEX2HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEX3HNVPROC) (GLuint x, GLuint y, GLuint z);
typedef void (GLAPIENTRY * PFNGLVERTEX3HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEX4HNVPROC) (GLuint x, GLuint y, GLuint z, GLuint w);
typedef void (GLAPIENTRY * PFNGLVERTEX4HVNVPROC) (const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1HNVPROC) (GLuint index, GLuint x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1HVNVPROC) (GLuint index, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2HNVPROC) (GLuint index, GLuint x, GLuint y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2HVNVPROC) (GLuint index, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3HNVPROC) (GLuint index, GLuint x, GLuint y, GLuint z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3HVNVPROC) (GLuint index, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4HNVPROC) (GLuint index, GLuint x, GLuint y, GLuint z, GLuint w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4HVNVPROC) (GLuint index, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS1HVNVPROC) (GLuint index, GLsizei n, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS2HVNVPROC) (GLuint index, GLsizei n, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS3HVNVPROC) (GLuint index, GLsizei n, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS4HVNVPROC) (GLuint index, GLsizei n, const GLuint* v);
typedef void (GLAPIENTRY * PFNGLVERTEXWEIGHTHNVPROC) (GLuint weight);
typedef void (GLAPIENTRY * PFNGLVERTEXWEIGHTHVNVPROC) (const GLuint* weight);

GLEWAPI PFNGLCOLOR3HNVPROC glewColor3hNV;
GLEWAPI PFNGLCOLOR3HVNVPROC glewColor3hvNV;
GLEWAPI PFNGLCOLOR4HNVPROC glewColor4hNV;
GLEWAPI PFNGLCOLOR4HVNVPROC glewColor4hvNV;
GLEWAPI PFNGLFOGCOORDHNVPROC glewFogCoordhNV;
GLEWAPI PFNGLFOGCOORDHVNVPROC glewFogCoordhvNV;
GLEWAPI PFNGLMULTITEXCOORD1HNVPROC glewMultiTexCoord1hNV;
GLEWAPI PFNGLMULTITEXCOORD1HVNVPROC glewMultiTexCoord1hvNV;
GLEWAPI PFNGLMULTITEXCOORD2HNVPROC glewMultiTexCoord2hNV;
GLEWAPI PFNGLMULTITEXCOORD2HVNVPROC glewMultiTexCoord2hvNV;
GLEWAPI PFNGLMULTITEXCOORD3HNVPROC glewMultiTexCoord3hNV;
GLEWAPI PFNGLMULTITEXCOORD3HVNVPROC glewMultiTexCoord3hvNV;
GLEWAPI PFNGLMULTITEXCOORD4HNVPROC glewMultiTexCoord4hNV;
GLEWAPI PFNGLMULTITEXCOORD4HVNVPROC glewMultiTexCoord4hvNV;
GLEWAPI PFNGLNORMAL3HNVPROC glewNormal3hNV;
GLEWAPI PFNGLNORMAL3HVNVPROC glewNormal3hvNV;
GLEWAPI PFNGLSECONDARYCOLOR3HNVPROC glewSecondaryColor3hNV;
GLEWAPI PFNGLSECONDARYCOLOR3HVNVPROC glewSecondaryColor3hvNV;
GLEWAPI PFNGLTEXCOORD1HNVPROC glewTexCoord1hNV;
GLEWAPI PFNGLTEXCOORD1HVNVPROC glewTexCoord1hvNV;
GLEWAPI PFNGLTEXCOORD2HNVPROC glewTexCoord2hNV;
GLEWAPI PFNGLTEXCOORD2HVNVPROC glewTexCoord2hvNV;
GLEWAPI PFNGLTEXCOORD3HNVPROC glewTexCoord3hNV;
GLEWAPI PFNGLTEXCOORD3HVNVPROC glewTexCoord3hvNV;
GLEWAPI PFNGLTEXCOORD4HNVPROC glewTexCoord4hNV;
GLEWAPI PFNGLTEXCOORD4HVNVPROC glewTexCoord4hvNV;
GLEWAPI PFNGLVERTEX2HNVPROC glewVertex2hNV;
GLEWAPI PFNGLVERTEX2HVNVPROC glewVertex2hvNV;
GLEWAPI PFNGLVERTEX3HNVPROC glewVertex3hNV;
GLEWAPI PFNGLVERTEX3HVNVPROC glewVertex3hvNV;
GLEWAPI PFNGLVERTEX4HNVPROC glewVertex4hNV;
GLEWAPI PFNGLVERTEX4HVNVPROC glewVertex4hvNV;
GLEWAPI PFNGLVERTEXATTRIB1HNVPROC glewVertexAttrib1hNV;
GLEWAPI PFNGLVERTEXATTRIB1HVNVPROC glewVertexAttrib1hvNV;
GLEWAPI PFNGLVERTEXATTRIB2HNVPROC glewVertexAttrib2hNV;
GLEWAPI PFNGLVERTEXATTRIB2HVNVPROC glewVertexAttrib2hvNV;
GLEWAPI PFNGLVERTEXATTRIB3HNVPROC glewVertexAttrib3hNV;
GLEWAPI PFNGLVERTEXATTRIB3HVNVPROC glewVertexAttrib3hvNV;
GLEWAPI PFNGLVERTEXATTRIB4HNVPROC glewVertexAttrib4hNV;
GLEWAPI PFNGLVERTEXATTRIB4HVNVPROC glewVertexAttrib4hvNV;
GLEWAPI PFNGLVERTEXATTRIBS1HVNVPROC glewVertexAttribs1hvNV;
GLEWAPI PFNGLVERTEXATTRIBS2HVNVPROC glewVertexAttribs2hvNV;
GLEWAPI PFNGLVERTEXATTRIBS3HVNVPROC glewVertexAttribs3hvNV;
GLEWAPI PFNGLVERTEXATTRIBS4HVNVPROC glewVertexAttribs4hvNV;
GLEWAPI PFNGLVERTEXWEIGHTHNVPROC glewVertexWeighthNV;
GLEWAPI PFNGLVERTEXWEIGHTHVNVPROC glewVertexWeighthvNV;

#define glColor3hNV glewColor3hNV
#define glColor3hvNV glewColor3hvNV
#define glColor4hNV glewColor4hNV
#define glColor4hvNV glewColor4hvNV
#define glFogCoordhNV glewFogCoordhNV
#define glFogCoordhvNV glewFogCoordhvNV
#define glMultiTexCoord1hNV glewMultiTexCoord1hNV
#define glMultiTexCoord1hvNV glewMultiTexCoord1hvNV
#define glMultiTexCoord2hNV glewMultiTexCoord2hNV
#define glMultiTexCoord2hvNV glewMultiTexCoord2hvNV
#define glMultiTexCoord3hNV glewMultiTexCoord3hNV
#define glMultiTexCoord3hvNV glewMultiTexCoord3hvNV
#define glMultiTexCoord4hNV glewMultiTexCoord4hNV
#define glMultiTexCoord4hvNV glewMultiTexCoord4hvNV
#define glNormal3hNV glewNormal3hNV
#define glNormal3hvNV glewNormal3hvNV
#define glSecondaryColor3hNV glewSecondaryColor3hNV
#define glSecondaryColor3hvNV glewSecondaryColor3hvNV
#define glTexCoord1hNV glewTexCoord1hNV
#define glTexCoord1hvNV glewTexCoord1hvNV
#define glTexCoord2hNV glewTexCoord2hNV
#define glTexCoord2hvNV glewTexCoord2hvNV
#define glTexCoord3hNV glewTexCoord3hNV
#define glTexCoord3hvNV glewTexCoord3hvNV
#define glTexCoord4hNV glewTexCoord4hNV
#define glTexCoord4hvNV glewTexCoord4hvNV
#define glVertex2hNV glewVertex2hNV
#define glVertex2hvNV glewVertex2hvNV
#define glVertex3hNV glewVertex3hNV
#define glVertex3hvNV glewVertex3hvNV
#define glVertex4hNV glewVertex4hNV
#define glVertex4hvNV glewVertex4hvNV
#define glVertexAttrib1hNV glewVertexAttrib1hNV
#define glVertexAttrib1hvNV glewVertexAttrib1hvNV
#define glVertexAttrib2hNV glewVertexAttrib2hNV
#define glVertexAttrib2hvNV glewVertexAttrib2hvNV
#define glVertexAttrib3hNV glewVertexAttrib3hNV
#define glVertexAttrib3hvNV glewVertexAttrib3hvNV
#define glVertexAttrib4hNV glewVertexAttrib4hNV
#define glVertexAttrib4hvNV glewVertexAttrib4hvNV
#define glVertexAttribs1hvNV glewVertexAttribs1hvNV
#define glVertexAttribs2hvNV glewVertexAttribs2hvNV
#define glVertexAttribs3hvNV glewVertexAttribs3hvNV
#define glVertexAttribs4hvNV glewVertexAttribs4hvNV
#define glVertexWeighthNV glewVertexWeighthNV
#define glVertexWeighthvNV glewVertexWeighthvNV

GLEWAPI GLboolean GLEW_NV_half_float;

#endif /* GL_NV_half_float */

/* ------------------------ GL_NV_light_max_exponent ----------------------- */

#ifndef GL_NV_light_max_exponent
#define GL_NV_light_max_exponent 1

#define GL_MAX_SHININESS_NV 0x8504
#define GL_MAX_SPOT_EXPONENT_NV 0x8505

GLEWAPI GLboolean GLEW_NV_light_max_exponent;

#endif /* GL_NV_light_max_exponent */

/* --------------------- GL_NV_multisample_filter_hint --------------------- */

#ifndef GL_NV_multisample_filter_hint
#define GL_NV_multisample_filter_hint 1

#define GL_MULTISAMPLE_FILTER_HINT_NV 0x8534

GLEWAPI GLboolean GLEW_NV_multisample_filter_hint;

#endif /* GL_NV_multisample_filter_hint */

/* ------------------------- GL_NV_occlusion_query ------------------------- */

#ifndef GL_NV_occlusion_query
#define GL_NV_occlusion_query 1

#define GL_PIXEL_COUNTER_BITS_NV 0x8864
#define GL_CURRENT_OCCLUSION_QUERY_ID_NV 0x8865
#define GL_PIXEL_COUNT_NV 0x8866
#define GL_PIXEL_COUNT_AVAILABLE_NV 0x8867

typedef void (GLAPIENTRY * PFNGLBEGINOCCLUSIONQUERYNVPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLDELETEOCCLUSIONQUERIESNVPROC) (GLsizei n, const GLuint* ids);
typedef void (GLAPIENTRY * PFNGLENDOCCLUSIONQUERYNVPROC) (void);
typedef void (GLAPIENTRY * PFNGLGENOCCLUSIONQUERIESNVPROC) (GLsizei n, GLuint* ids);
typedef void (GLAPIENTRY * PFNGLGETOCCLUSIONQUERYIVNVPROC) (GLuint id, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETOCCLUSIONQUERYUIVNVPROC) (GLuint id, GLenum pname, GLuint* params);
typedef GLboolean (GLAPIENTRY * PFNGLISOCCLUSIONQUERYNVPROC) (GLuint id);

GLEWAPI PFNGLBEGINOCCLUSIONQUERYNVPROC glewBeginOcclusionQueryNV;
GLEWAPI PFNGLDELETEOCCLUSIONQUERIESNVPROC glewDeleteOcclusionQueriesNV;
GLEWAPI PFNGLENDOCCLUSIONQUERYNVPROC glewEndOcclusionQueryNV;
GLEWAPI PFNGLGENOCCLUSIONQUERIESNVPROC glewGenOcclusionQueriesNV;
GLEWAPI PFNGLGETOCCLUSIONQUERYIVNVPROC glewGetOcclusionQueryivNV;
GLEWAPI PFNGLGETOCCLUSIONQUERYUIVNVPROC glewGetOcclusionQueryuivNV;
GLEWAPI PFNGLISOCCLUSIONQUERYNVPROC glewIsOcclusionQueryNV;

#define glBeginOcclusionQueryNV glewBeginOcclusionQueryNV
#define glDeleteOcclusionQueriesNV glewDeleteOcclusionQueriesNV
#define glEndOcclusionQueryNV glewEndOcclusionQueryNV
#define glGenOcclusionQueriesNV glewGenOcclusionQueriesNV
#define glGetOcclusionQueryivNV glewGetOcclusionQueryivNV
#define glGetOcclusionQueryuivNV glewGetOcclusionQueryuivNV
#define glIsOcclusionQueryNV glewIsOcclusionQueryNV

GLEWAPI GLboolean GLEW_NV_occlusion_query;

#endif /* GL_NV_occlusion_query */

/* ----------------------- GL_NV_packed_depth_stencil ---------------------- */

#ifndef GL_NV_packed_depth_stencil
#define GL_NV_packed_depth_stencil 1

#define GL_DEPTH_STENCIL_NV 0x84F9
#define GL_UNSIGNED_INT_24_8_NV 0x84FA

GLEWAPI GLboolean GLEW_NV_packed_depth_stencil;

#endif /* GL_NV_packed_depth_stencil */

/* ------------------------- GL_NV_pixel_data_range ------------------------ */

#ifndef GL_NV_pixel_data_range
#define GL_NV_pixel_data_range 1

#define GL_WRITE_PIXEL_DATA_RANGE_NV 0x8878
#define GL_READ_PIXEL_DATA_RANGE_NV 0x8879
#define GL_WRITE_PIXEL_DATA_RANGE_LENGTH_NV 0x887A
#define GL_READ_PIXEL_DATA_RANGE_LENGTH_NV 0x887B
#define GL_WRITE_PIXEL_DATA_RANGE_POINTER_NV 0x887C
#define GL_READ_PIXEL_DATA_RANGE_POINTER_NV 0x887D

typedef void (GLAPIENTRY * PFNGLFLUSHPIXELDATARANGENVPROC) (GLenum target);
typedef void (GLAPIENTRY * PFNGLPIXELDATARANGENVPROC) (GLenum target, GLsizei length, void* pointer);

GLEWAPI PFNGLFLUSHPIXELDATARANGENVPROC glewFlushPixelDataRangeNV;
GLEWAPI PFNGLPIXELDATARANGENVPROC glewPixelDataRangeNV;

#define glFlushPixelDataRangeNV glewFlushPixelDataRangeNV
#define glPixelDataRangeNV glewPixelDataRangeNV

GLEWAPI GLboolean GLEW_NV_pixel_data_range;

#endif /* GL_NV_pixel_data_range */

/* --------------------------- GL_NV_point_sprite -------------------------- */

#ifndef GL_NV_point_sprite
#define GL_NV_point_sprite 1

#define GL_POINT_SPRITE_NV 0x8861
#define GL_COORD_REPLACE_NV 0x8862
#define GL_POINT_SPRITE_R_MODE_NV 0x8863

typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERINVPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLPOINTPARAMETERIVNVPROC) (GLenum pname, const GLint* params);

GLEWAPI PFNGLPOINTPARAMETERINVPROC glewPointParameteriNV;
GLEWAPI PFNGLPOINTPARAMETERIVNVPROC glewPointParameterivNV;

#define glPointParameteriNV glewPointParameteriNV
#define glPointParameterivNV glewPointParameterivNV

GLEWAPI GLboolean GLEW_NV_point_sprite;

#endif /* GL_NV_point_sprite */

/* ------------------------ GL_NV_primitive_restart ------------------------ */

#ifndef GL_NV_primitive_restart
#define GL_NV_primitive_restart 1

#define GL_PRIMITIVE_RESTART_NV 0x8558
#define GL_PRIMITIVE_RESTART_INDEX_NV 0x8559

typedef void (GLAPIENTRY * PFNGLPRIMITIVERESTARTINDEXNVPROC) (GLuint index);
typedef void (GLAPIENTRY * PFNGLPRIMITIVERESTARTNVPROC) (void);

GLEWAPI PFNGLPRIMITIVERESTARTINDEXNVPROC glewPrimitiveRestartIndexNV;
GLEWAPI PFNGLPRIMITIVERESTARTNVPROC glewPrimitiveRestartNV;

#define glPrimitiveRestartIndexNV glewPrimitiveRestartIndexNV
#define glPrimitiveRestartNV glewPrimitiveRestartNV

GLEWAPI GLboolean GLEW_NV_primitive_restart;

#endif /* GL_NV_primitive_restart */

/* ------------------------ GL_NV_register_combiners ----------------------- */

#ifndef GL_NV_register_combiners
#define GL_NV_register_combiners 1

#define GL_REGISTER_COMBINERS_NV 0x8522
#define GL_VARIABLE_A_NV 0x8523
#define GL_VARIABLE_B_NV 0x8524
#define GL_VARIABLE_C_NV 0x8525
#define GL_VARIABLE_D_NV 0x8526
#define GL_VARIABLE_E_NV 0x8527
#define GL_VARIABLE_F_NV 0x8528
#define GL_VARIABLE_G_NV 0x8529
#define GL_CONSTANT_COLOR0_NV 0x852A
#define GL_CONSTANT_COLOR1_NV 0x852B
#define GL_PRIMARY_COLOR_NV 0x852C
#define GL_SECONDARY_COLOR_NV 0x852D
#define GL_SPARE0_NV 0x852E
#define GL_SPARE1_NV 0x852F
#define GL_DISCARD_NV 0x8530
#define GL_E_TIMES_F_NV 0x8531
#define GL_SPARE0_PLUS_SECONDARY_COLOR_NV 0x8532
#define GL_UNSIGNED_IDENTITY_NV 0x8536
#define GL_UNSIGNED_INVERT_NV 0x8537
#define GL_EXPAND_NORMAL_NV 0x8538
#define GL_EXPAND_NEGATE_NV 0x8539
#define GL_HALF_BIAS_NORMAL_NV 0x853A
#define GL_HALF_BIAS_NEGATE_NV 0x853B
#define GL_SIGNED_IDENTITY_NV 0x853C
#define GL_SIGNED_NEGATE_NV 0x853D
#define GL_SCALE_BY_TWO_NV 0x853E
#define GL_SCALE_BY_FOUR_NV 0x853F
#define GL_SCALE_BY_ONE_HALF_NV 0x8540
#define GL_BIAS_BY_NEGATIVE_ONE_HALF_NV 0x8541
#define GL_COMBINER_INPUT_NV 0x8542
#define GL_COMBINER_MAPPING_NV 0x8543
#define GL_COMBINER_COMPONENT_USAGE_NV 0x8544
#define GL_COMBINER_AB_DOT_PRODUCT_NV 0x8545
#define GL_COMBINER_CD_DOT_PRODUCT_NV 0x8546
#define GL_COMBINER_MUX_SUM_NV 0x8547
#define GL_COMBINER_SCALE_NV 0x8548
#define GL_COMBINER_BIAS_NV 0x8549
#define GL_COMBINER_AB_OUTPUT_NV 0x854A
#define GL_COMBINER_CD_OUTPUT_NV 0x854B
#define GL_COMBINER_SUM_OUTPUT_NV 0x854C
#define GL_MAX_GENERAL_COMBINERS_NV 0x854D
#define GL_NUM_GENERAL_COMBINERS_NV 0x854E
#define GL_COLOR_SUM_CLAMP_NV 0x854F
#define GL_COMBINER0_NV 0x8550
#define GL_COMBINER1_NV 0x8551
#define GL_COMBINER2_NV 0x8552
#define GL_COMBINER3_NV 0x8553
#define GL_COMBINER4_NV 0x8554
#define GL_COMBINER5_NV 0x8555
#define GL_COMBINER6_NV 0x8556
#define GL_COMBINER7_NV 0x8557

typedef void (GLAPIENTRY * PFNGLCOMBINERINPUTNVPROC) (GLenum stage, GLenum portion, GLenum variable, GLenum input, GLenum mapping, GLenum componentUsage);
typedef void (GLAPIENTRY * PFNGLCOMBINEROUTPUTNVPROC) (GLenum stage, GLenum portion, GLenum abOutput, GLenum cdOutput, GLenum sumOutput, GLenum scale, GLenum bias, GLboolean abDotProduct, GLboolean cdDotProduct, GLboolean muxSum);
typedef void (GLAPIENTRY * PFNGLCOMBINERPARAMETERFNVPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLCOMBINERPARAMETERFVNVPROC) (GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLCOMBINERPARAMETERINVPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLCOMBINERPARAMETERIVNVPROC) (GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLFINALCOMBINERINPUTNVPROC) (GLenum variable, GLenum input, GLenum mapping, GLenum componentUsage);
typedef void (GLAPIENTRY * PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC) (GLenum stage, GLenum portion, GLenum variable, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC) (GLenum stage, GLenum portion, GLenum variable, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC) (GLenum stage, GLenum portion, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC) (GLenum stage, GLenum portion, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC) (GLenum variable, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC) (GLenum variable, GLenum pname, GLint* params);

GLEWAPI PFNGLCOMBINERINPUTNVPROC glewCombinerInputNV;
GLEWAPI PFNGLCOMBINEROUTPUTNVPROC glewCombinerOutputNV;
GLEWAPI PFNGLCOMBINERPARAMETERFNVPROC glewCombinerParameterfNV;
GLEWAPI PFNGLCOMBINERPARAMETERFVNVPROC glewCombinerParameterfvNV;
GLEWAPI PFNGLCOMBINERPARAMETERINVPROC glewCombinerParameteriNV;
GLEWAPI PFNGLCOMBINERPARAMETERIVNVPROC glewCombinerParameterivNV;
GLEWAPI PFNGLFINALCOMBINERINPUTNVPROC glewFinalCombinerInputNV;
GLEWAPI PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC glewGetCombinerInputParameterfvNV;
GLEWAPI PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC glewGetCombinerInputParameterivNV;
GLEWAPI PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC glewGetCombinerOutputParameterfvNV;
GLEWAPI PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC glewGetCombinerOutputParameterivNV;
GLEWAPI PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC glewGetFinalCombinerInputParameterfvNV;
GLEWAPI PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC glewGetFinalCombinerInputParameterivNV;

#define glCombinerInputNV glewCombinerInputNV
#define glCombinerOutputNV glewCombinerOutputNV
#define glCombinerParameterfNV glewCombinerParameterfNV
#define glCombinerParameterfvNV glewCombinerParameterfvNV
#define glCombinerParameteriNV glewCombinerParameteriNV
#define glCombinerParameterivNV glewCombinerParameterivNV
#define glFinalCombinerInputNV glewFinalCombinerInputNV
#define glGetCombinerInputParameterfvNV glewGetCombinerInputParameterfvNV
#define glGetCombinerInputParameterivNV glewGetCombinerInputParameterivNV
#define glGetCombinerOutputParameterfvNV glewGetCombinerOutputParameterfvNV
#define glGetCombinerOutputParameterivNV glewGetCombinerOutputParameterivNV
#define glGetFinalCombinerInputParameterfvNV glewGetFinalCombinerInputParameterfvNV
#define glGetFinalCombinerInputParameterivNV glewGetFinalCombinerInputParameterivNV

GLEWAPI GLboolean GLEW_NV_register_combiners;

#endif /* GL_NV_register_combiners */

/* ----------------------- GL_NV_register_combiners2 ----------------------- */

#ifndef GL_NV_register_combiners2
#define GL_NV_register_combiners2 1

#define GL_PER_STAGE_CONSTANTS_NV 0x8535

typedef void (GLAPIENTRY * PFNGLCOMBINERSTAGEPARAMETERFVNVPROC) (GLenum stage, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC) (GLenum stage, GLenum pname, GLfloat* params);

GLEWAPI PFNGLCOMBINERSTAGEPARAMETERFVNVPROC glewCombinerStageParameterfvNV;
GLEWAPI PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC glewGetCombinerStageParameterfvNV;

#define glCombinerStageParameterfvNV glewCombinerStageParameterfvNV
#define glGetCombinerStageParameterfvNV glewGetCombinerStageParameterfvNV

GLEWAPI GLboolean GLEW_NV_register_combiners2;

#endif /* GL_NV_register_combiners2 */

/* -------------------------- GL_NV_texgen_emboss -------------------------- */

#ifndef GL_NV_texgen_emboss
#define GL_NV_texgen_emboss 1

#define GL_EMBOSS_LIGHT_NV 0x855D
#define GL_EMBOSS_CONSTANT_NV 0x855E
#define GL_EMBOSS_MAP_NV 0x855F

GLEWAPI GLboolean GLEW_NV_texgen_emboss;

#endif /* GL_NV_texgen_emboss */

/* ------------------------ GL_NV_texgen_reflection ------------------------ */

#ifndef GL_NV_texgen_reflection
#define GL_NV_texgen_reflection 1

#define GL_NORMAL_MAP_NV 0x8511
#define GL_REFLECTION_MAP_NV 0x8512

GLEWAPI GLboolean GLEW_NV_texgen_reflection;

#endif /* GL_NV_texgen_reflection */

/* --------------------- GL_NV_texture_compression_vtc --------------------- */

#ifndef GL_NV_texture_compression_vtc
#define GL_NV_texture_compression_vtc 1

GLEWAPI GLboolean GLEW_NV_texture_compression_vtc;

#endif /* GL_NV_texture_compression_vtc */

/* ----------------------- GL_NV_texture_env_combine4 ---------------------- */

#ifndef GL_NV_texture_env_combine4
#define GL_NV_texture_env_combine4 1

#define GL_COMBINE4_NV 0x8503
#define GL_SOURCE3_RGB_NV 0x8583
#define GL_SOURCE3_ALPHA_NV 0x858B
#define GL_OPERAND3_RGB_NV 0x8593
#define GL_OPERAND3_ALPHA_NV 0x859B

GLEWAPI GLboolean GLEW_NV_texture_env_combine4;

#endif /* GL_NV_texture_env_combine4 */

/* ---------------------- GL_NV_texture_expand_normal ---------------------- */

#ifndef GL_NV_texture_expand_normal
#define GL_NV_texture_expand_normal 1

#define GL_TEXTURE_UNSIGNED_REMAP_MODE_NV 0x888F

GLEWAPI GLboolean GLEW_NV_texture_expand_normal;

#endif /* GL_NV_texture_expand_normal */

/* ------------------------ GL_NV_texture_rectangle ------------------------ */

#ifndef GL_NV_texture_rectangle
#define GL_NV_texture_rectangle 1

#define GL_TEXTURE_RECTANGLE_NV 0x84F5
#define GL_TEXTURE_BINDING_RECTANGLE_NV 0x84F6
#define GL_PROXY_TEXTURE_RECTANGLE_NV 0x84F7
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_NV 0x84F8

GLEWAPI GLboolean GLEW_NV_texture_rectangle;

#endif /* GL_NV_texture_rectangle */

/* -------------------------- GL_NV_texture_shader ------------------------- */

#ifndef GL_NV_texture_shader
#define GL_NV_texture_shader 1

#define GL_OFFSET_TEXTURE_RECTANGLE_NV 0x864C
#define GL_OFFSET_TEXTURE_RECTANGLE_SCALE_NV 0x864D
#define GL_DOT_PRODUCT_TEXTURE_RECTANGLE_NV 0x864E
#define GL_RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV 0x86D9
#define GL_UNSIGNED_INT_S8_S8_8_8_NV 0x86DA
#define GL_UNSIGNED_INT_8_8_S8_S8_REV_NV 0x86DB
#define GL_DSDT_MAG_INTENSITY_NV 0x86DC
#define GL_SHADER_CONSISTENT_NV 0x86DD
#define GL_TEXTURE_SHADER_NV 0x86DE
#define GL_SHADER_OPERATION_NV 0x86DF
#define GL_CULL_MODES_NV 0x86E0
#define GL_OFFSET_TEXTURE_MATRIX_NV 0x86E1
#define GL_OFFSET_TEXTURE_SCALE_NV 0x86E2
#define GL_OFFSET_TEXTURE_BIAS_NV 0x86E3
#define GL_PREVIOUS_TEXTURE_INPUT_NV 0x86E4
#define GL_CONST_EYE_NV 0x86E5
#define GL_PASS_THROUGH_NV 0x86E6
#define GL_CULL_FRAGMENT_NV 0x86E7
#define GL_OFFSET_TEXTURE_2D_NV 0x86E8
#define GL_DEPENDENT_AR_TEXTURE_2D_NV 0x86E9
#define GL_DEPENDENT_GB_TEXTURE_2D_NV 0x86EA
#define GL_DOT_PRODUCT_NV 0x86EC
#define GL_DOT_PRODUCT_DEPTH_REPLACE_NV 0x86ED
#define GL_DOT_PRODUCT_TEXTURE_2D_NV 0x86EE
#define GL_DOT_PRODUCT_TEXTURE_CUBE_MAP_NV 0x86F0
#define GL_DOT_PRODUCT_DIFFUSE_CUBE_MAP_NV 0x86F1
#define GL_DOT_PRODUCT_REFLECT_CUBE_MAP_NV 0x86F2
#define GL_DOT_PRODUCT_CONST_EYE_REFLECT_CUBE_MAP_NV 0x86F3
#define GL_HILO_NV 0x86F4
#define GL_DSDT_NV 0x86F5
#define GL_DSDT_MAG_NV 0x86F6
#define GL_DSDT_MAG_VIB_NV 0x86F7
#define GL_HILO16_NV 0x86F8
#define GL_SIGNED_HILO_NV 0x86F9
#define GL_SIGNED_HILO16_NV 0x86FA
#define GL_SIGNED_RGBA_NV 0x86FB
#define GL_SIGNED_RGBA8_NV 0x86FC
#define GL_SIGNED_RGB_NV 0x86FE
#define GL_SIGNED_RGB8_NV 0x86FF
#define GL_SIGNED_LUMINANCE_NV 0x8701
#define GL_SIGNED_LUMINANCE8_NV 0x8702
#define GL_SIGNED_LUMINANCE_ALPHA_NV 0x8703
#define GL_SIGNED_LUMINANCE8_ALPHA8_NV 0x8704
#define GL_SIGNED_ALPHA_NV 0x8705
#define GL_SIGNED_ALPHA8_NV 0x8706
#define GL_SIGNED_INTENSITY_NV 0x8707
#define GL_SIGNED_INTENSITY8_NV 0x8708
#define GL_DSDT8_NV 0x8709
#define GL_DSDT8_MAG8_NV 0x870A
#define GL_DSDT8_MAG8_INTENSITY8_NV 0x870B
#define GL_SIGNED_RGB_UNSIGNED_ALPHA_NV 0x870C
#define GL_SIGNED_RGB8_UNSIGNED_ALPHA8_NV 0x870D
#define GL_HI_SCALE_NV 0x870E
#define GL_LO_SCALE_NV 0x870F
#define GL_DS_SCALE_NV 0x8710
#define GL_DT_SCALE_NV 0x8711
#define GL_MAGNITUDE_SCALE_NV 0x8712
#define GL_VIBRANCE_SCALE_NV 0x8713
#define GL_HI_BIAS_NV 0x8714
#define GL_LO_BIAS_NV 0x8715
#define GL_DS_BIAS_NV 0x8716
#define GL_DT_BIAS_NV 0x8717
#define GL_MAGNITUDE_BIAS_NV 0x8718
#define GL_VIBRANCE_BIAS_NV 0x8719
#define GL_TEXTURE_BORDER_VALUES_NV 0x871A
#define GL_TEXTURE_HI_SIZE_NV 0x871B
#define GL_TEXTURE_LO_SIZE_NV 0x871C
#define GL_TEXTURE_DS_SIZE_NV 0x871D
#define GL_TEXTURE_DT_SIZE_NV 0x871E
#define GL_TEXTURE_MAG_SIZE_NV 0x871F

GLEWAPI GLboolean GLEW_NV_texture_shader;

#endif /* GL_NV_texture_shader */

/* ------------------------- GL_NV_texture_shader2 ------------------------- */

#ifndef GL_NV_texture_shader2
#define GL_NV_texture_shader2 1

#define GL_UNSIGNED_INT_S8_S8_8_8_NV 0x86DA
#define GL_UNSIGNED_INT_8_8_S8_S8_REV_NV 0x86DB
#define GL_DSDT_MAG_INTENSITY_NV 0x86DC
#define GL_DOT_PRODUCT_TEXTURE_3D_NV 0x86EF
#define GL_HILO_NV 0x86F4
#define GL_DSDT_NV 0x86F5
#define GL_DSDT_MAG_NV 0x86F6
#define GL_DSDT_MAG_VIB_NV 0x86F7
#define GL_HILO16_NV 0x86F8
#define GL_SIGNED_HILO_NV 0x86F9
#define GL_SIGNED_HILO16_NV 0x86FA
#define GL_SIGNED_RGBA_NV 0x86FB
#define GL_SIGNED_RGBA8_NV 0x86FC
#define GL_SIGNED_RGB_NV 0x86FE
#define GL_SIGNED_RGB8_NV 0x86FF
#define GL_SIGNED_LUMINANCE_NV 0x8701
#define GL_SIGNED_LUMINANCE8_NV 0x8702
#define GL_SIGNED_LUMINANCE_ALPHA_NV 0x8703
#define GL_SIGNED_LUMINANCE8_ALPHA8_NV 0x8704
#define GL_SIGNED_ALPHA_NV 0x8705
#define GL_SIGNED_ALPHA8_NV 0x8706
#define GL_SIGNED_INTENSITY_NV 0x8707
#define GL_SIGNED_INTENSITY8_NV 0x8708
#define GL_DSDT8_NV 0x8709
#define GL_DSDT8_MAG8_NV 0x870A
#define GL_DSDT8_MAG8_INTENSITY8_NV 0x870B
#define GL_SIGNED_RGB_UNSIGNED_ALPHA_NV 0x870C
#define GL_SIGNED_RGB8_UNSIGNED_ALPHA8_NV 0x870D

GLEWAPI GLboolean GLEW_NV_texture_shader2;

#endif /* GL_NV_texture_shader2 */

/* ------------------------- GL_NV_texture_shader3 ------------------------- */

#ifndef GL_NV_texture_shader3
#define GL_NV_texture_shader3 1

#define GL_OFFSET_PROJECTIVE_TEXTURE_2D_NV 0x8850
#define GL_OFFSET_PROJECTIVE_TEXTURE_2D_SCALE_NV 0x8851
#define GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_NV 0x8852
#define GL_OFFSET_PROJECTIVE_TEXTURE_RECTANGLE_SCALE_NV 0x8853
#define GL_OFFSET_HILO_TEXTURE_2D_NV 0x8854
#define GL_OFFSET_HILO_TEXTURE_RECTANGLE_NV 0x8855
#define GL_OFFSET_HILO_PROJECTIVE_TEXTURE_2D_NV 0x8856
#define GL_OFFSET_HILO_PROJECTIVE_TEXTURE_RECTANGLE_NV 0x8857
#define GL_DEPENDENT_HILO_TEXTURE_2D_NV 0x8858
#define GL_DEPENDENT_RGB_TEXTURE_3D_NV 0x8859
#define GL_DEPENDENT_RGB_TEXTURE_CUBE_MAP_NV 0x885A
#define GL_DOT_PRODUCT_PASS_THROUGH_NV 0x885B
#define GL_DOT_PRODUCT_TEXTURE_1D_NV 0x885C
#define GL_DOT_PRODUCT_AFFINE_DEPTH_REPLACE_NV 0x885D
#define GL_HILO8_NV 0x885E
#define GL_SIGNED_HILO8_NV 0x885F
#define GL_FORCE_BLUE_TO_ONE_NV 0x8860

GLEWAPI GLboolean GLEW_NV_texture_shader3;

#endif /* GL_NV_texture_shader3 */

/* ------------------------ GL_NV_vertex_array_range ----------------------- */

#ifndef GL_NV_vertex_array_range
#define GL_NV_vertex_array_range 1

#define GL_VERTEX_ARRAY_RANGE_NV 0x851D
#define GL_VERTEX_ARRAY_RANGE_LENGTH_NV 0x851E
#define GL_VERTEX_ARRAY_RANGE_VALID_NV 0x851F
#define GL_MAX_VERTEX_ARRAY_RANGE_ELEMENT_NV 0x8520
#define GL_VERTEX_ARRAY_RANGE_POINTER_NV 0x8521

typedef void (GLAPIENTRY * PFNGLFLUSHVERTEXARRAYRANGENVPROC) (void);
typedef void (GLAPIENTRY * PFNGLVERTEXARRAYRANGENVPROC) (GLsizei length, void* pointer);

GLEWAPI PFNGLFLUSHVERTEXARRAYRANGENVPROC glewFlushVertexArrayRangeNV;
GLEWAPI PFNGLVERTEXARRAYRANGENVPROC glewVertexArrayRangeNV;

#define glFlushVertexArrayRangeNV glewFlushVertexArrayRangeNV
#define glVertexArrayRangeNV glewVertexArrayRangeNV

GLEWAPI GLboolean GLEW_NV_vertex_array_range;

#endif /* GL_NV_vertex_array_range */

/* ----------------------- GL_NV_vertex_array_range2 ----------------------- */

#ifndef GL_NV_vertex_array_range2
#define GL_NV_vertex_array_range2 1

#define GL_VERTEX_ARRAY_RANGE_WITHOUT_FLUSH_NV 0x8533

GLEWAPI GLboolean GLEW_NV_vertex_array_range2;

#endif /* GL_NV_vertex_array_range2 */

/* -------------------------- GL_NV_vertex_program ------------------------- */

#ifndef GL_NV_vertex_program
#define GL_NV_vertex_program 1

#define GL_VERTEX_PROGRAM_NV 0x8620
#define GL_VERTEX_STATE_PROGRAM_NV 0x8621
#define GL_ATTRIB_ARRAY_SIZE_NV 0x8623
#define GL_ATTRIB_ARRAY_STRIDE_NV 0x8624
#define GL_ATTRIB_ARRAY_TYPE_NV 0x8625
#define GL_CURRENT_ATTRIB_NV 0x8626
#define GL_PROGRAM_LENGTH_NV 0x8627
#define GL_PROGRAM_STRING_NV 0x8628
#define GL_MODELVIEW_PROJECTION_NV 0x8629
#define GL_IDENTITY_NV 0x862A
#define GL_INVERSE_NV 0x862B
#define GL_TRANSPOSE_NV 0x862C
#define GL_INVERSE_TRANSPOSE_NV 0x862D
#define GL_MAX_TRACK_MATRIX_STACK_DEPTH_NV 0x862E
#define GL_MAX_TRACK_MATRICES_NV 0x862F
#define GL_MATRIX0_NV 0x8630
#define GL_MATRIX1_NV 0x8631
#define GL_MATRIX2_NV 0x8632
#define GL_MATRIX3_NV 0x8633
#define GL_MATRIX4_NV 0x8634
#define GL_MATRIX5_NV 0x8635
#define GL_MATRIX6_NV 0x8636
#define GL_MATRIX7_NV 0x8637
#define GL_CURRENT_MATRIX_STACK_DEPTH_NV 0x8640
#define GL_CURRENT_MATRIX_NV 0x8641
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642
#define GL_VERTEX_PROGRAM_TWO_SIDE_NV 0x8643
#define GL_PROGRAM_PARAMETER_NV 0x8644
#define GL_ATTRIB_ARRAY_POINTER_NV 0x8645
#define GL_PROGRAM_TARGET_NV 0x8646
#define GL_PROGRAM_RESIDENT_NV 0x8647
#define GL_TRACK_MATRIX_NV 0x8648
#define GL_TRACK_MATRIX_TRANSFORM_NV 0x8649
#define GL_VERTEX_PROGRAM_BINDING_NV 0x864A
#define GL_PROGRAM_ERROR_POSITION_NV 0x864B
#define GL_VERTEX_ATTRIB_ARRAY0_NV 0x8650
#define GL_VERTEX_ATTRIB_ARRAY1_NV 0x8651
#define GL_VERTEX_ATTRIB_ARRAY2_NV 0x8652
#define GL_VERTEX_ATTRIB_ARRAY3_NV 0x8653
#define GL_VERTEX_ATTRIB_ARRAY4_NV 0x8654
#define GL_VERTEX_ATTRIB_ARRAY5_NV 0x8655
#define GL_VERTEX_ATTRIB_ARRAY6_NV 0x8656
#define GL_VERTEX_ATTRIB_ARRAY7_NV 0x8657
#define GL_VERTEX_ATTRIB_ARRAY8_NV 0x8658
#define GL_VERTEX_ATTRIB_ARRAY9_NV 0x8659
#define GL_VERTEX_ATTRIB_ARRAY10_NV 0x865A
#define GL_VERTEX_ATTRIB_ARRAY11_NV 0x865B
#define GL_VERTEX_ATTRIB_ARRAY12_NV 0x865C
#define GL_VERTEX_ATTRIB_ARRAY13_NV 0x865D
#define GL_VERTEX_ATTRIB_ARRAY14_NV 0x865E
#define GL_VERTEX_ATTRIB_ARRAY15_NV 0x865F
#define GL_MAP1_VERTEX_ATTRIB0_4_NV 0x8660
#define GL_MAP1_VERTEX_ATTRIB1_4_NV 0x8661
#define GL_MAP1_VERTEX_ATTRIB2_4_NV 0x8662
#define GL_MAP1_VERTEX_ATTRIB3_4_NV 0x8663
#define GL_MAP1_VERTEX_ATTRIB4_4_NV 0x8664
#define GL_MAP1_VERTEX_ATTRIB5_4_NV 0x8665
#define GL_MAP1_VERTEX_ATTRIB6_4_NV 0x8666
#define GL_MAP1_VERTEX_ATTRIB7_4_NV 0x8667
#define GL_MAP1_VERTEX_ATTRIB8_4_NV 0x8668
#define GL_MAP1_VERTEX_ATTRIB9_4_NV 0x8669
#define GL_MAP1_VERTEX_ATTRIB10_4_NV 0x866A
#define GL_MAP1_VERTEX_ATTRIB11_4_NV 0x866B
#define GL_MAP1_VERTEX_ATTRIB12_4_NV 0x866C
#define GL_MAP1_VERTEX_ATTRIB13_4_NV 0x866D
#define GL_MAP1_VERTEX_ATTRIB14_4_NV 0x866E
#define GL_MAP1_VERTEX_ATTRIB15_4_NV 0x866F
#define GL_MAP2_VERTEX_ATTRIB0_4_NV 0x8670
#define GL_MAP2_VERTEX_ATTRIB1_4_NV 0x8671
#define GL_MAP2_VERTEX_ATTRIB2_4_NV 0x8672
#define GL_MAP2_VERTEX_ATTRIB3_4_NV 0x8673
#define GL_MAP2_VERTEX_ATTRIB4_4_NV 0x8674
#define GL_MAP2_VERTEX_ATTRIB5_4_NV 0x8675
#define GL_MAP2_VERTEX_ATTRIB6_4_NV 0x8676
#define GL_MAP2_VERTEX_ATTRIB7_4_NV 0x8677
#define GL_MAP2_VERTEX_ATTRIB8_4_NV 0x8678
#define GL_MAP2_VERTEX_ATTRIB9_4_NV 0x8679
#define GL_MAP2_VERTEX_ATTRIB10_4_NV 0x867A
#define GL_MAP2_VERTEX_ATTRIB11_4_NV 0x867B
#define GL_MAP2_VERTEX_ATTRIB12_4_NV 0x867C
#define GL_MAP2_VERTEX_ATTRIB13_4_NV 0x867D
#define GL_MAP2_VERTEX_ATTRIB14_4_NV 0x867E
#define GL_MAP2_VERTEX_ATTRIB15_4_NV 0x867F

typedef GLboolean (GLAPIENTRY * PFNGLAREPROGRAMSRESIDENTNVPROC) (GLsizei n, const GLuint* ids, GLboolean *residences);
typedef void (GLAPIENTRY * PFNGLBINDPROGRAMNVPROC) (GLenum target, GLuint id);
typedef void (GLAPIENTRY * PFNGLDELETEPROGRAMSNVPROC) (GLsizei n, const GLuint* ids);
typedef void (GLAPIENTRY * PFNGLEXECUTEPROGRAMNVPROC) (GLenum target, GLuint id, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGENPROGRAMSNVPROC) (GLsizei n, GLuint* ids);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMPARAMETERDVNVPROC) (GLenum target, GLuint index, GLenum pname, GLdouble* params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMPARAMETERFVNVPROC) (GLenum target, GLuint index, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMSTRINGNVPROC) (GLuint id, GLenum pname, GLubyte* program);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMIVNVPROC) (GLuint id, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETTRACKMATRIXIVNVPROC) (GLenum target, GLuint address, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBPOINTERVNVPROC) (GLuint index, GLenum pname, GLvoid** pointer);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBDVNVPROC) (GLuint index, GLenum pname, GLdouble* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBFVNVPROC) (GLuint index, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBIVNVPROC) (GLuint index, GLenum pname, GLint* params);
typedef GLboolean (GLAPIENTRY * PFNGLISPROGRAMNVPROC) (GLuint id);
typedef void (GLAPIENTRY * PFNGLLOADPROGRAMNVPROC) (GLenum target, GLuint id, GLsizei len, const GLubyte* program);
typedef void (GLAPIENTRY * PFNGLPROGRAMPARAMETER4DNVPROC) (GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLPROGRAMPARAMETER4DVNVPROC) (GLenum target, GLuint index, const GLdouble* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMPARAMETER4FNVPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLPROGRAMPARAMETER4FVNVPROC) (GLenum target, GLuint index, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMPARAMETERS4DVNVPROC) (GLenum target, GLuint index, GLuint num, const GLdouble* params);
typedef void (GLAPIENTRY * PFNGLPROGRAMPARAMETERS4FVNVPROC) (GLenum target, GLuint index, GLuint num, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLREQUESTRESIDENTPROGRAMSNVPROC) (GLsizei n, GLuint* ids);
typedef void (GLAPIENTRY * PFNGLTRACKMATRIXNVPROC) (GLenum target, GLuint address, GLenum matrix, GLenum transform);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1DNVPROC) (GLuint index, GLdouble x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1DVNVPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1FNVPROC) (GLuint index, GLfloat x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1FVNVPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1SNVPROC) (GLuint index, GLshort x);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1SVNVPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2DNVPROC) (GLuint index, GLdouble x, GLdouble y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2DVNVPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2FNVPROC) (GLuint index, GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2FVNVPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2SNVPROC) (GLuint index, GLshort x, GLshort y);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2SVNVPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3DNVPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3DVNVPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3FNVPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3FVNVPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3SNVPROC) (GLuint index, GLshort x, GLshort y, GLshort z);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3SVNVPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4DNVPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4DVNVPROC) (GLuint index, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4FNVPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4FVNVPROC) (GLuint index, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4SNVPROC) (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4SVNVPROC) (GLuint index, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4UBNVPROC) (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4UBVNVPROC) (GLuint index, const GLubyte* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBPOINTERNVPROC) (GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS1DVNVPROC) (GLuint index, GLsizei n, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS1FVNVPROC) (GLuint index, GLsizei n, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS1SVNVPROC) (GLuint index, GLsizei n, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS2DVNVPROC) (GLuint index, GLsizei n, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS2FVNVPROC) (GLuint index, GLsizei n, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS2SVNVPROC) (GLuint index, GLsizei n, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS3DVNVPROC) (GLuint index, GLsizei n, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS3FVNVPROC) (GLuint index, GLsizei n, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS3SVNVPROC) (GLuint index, GLsizei n, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS4DVNVPROC) (GLuint index, GLsizei n, const GLdouble* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS4FVNVPROC) (GLuint index, GLsizei n, const GLfloat* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS4SVNVPROC) (GLuint index, GLsizei n, const GLshort* v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBS4UBVNVPROC) (GLuint index, GLsizei n, const GLubyte* v);

GLEWAPI PFNGLAREPROGRAMSRESIDENTNVPROC glewAreProgramsResidentNV;
GLEWAPI PFNGLBINDPROGRAMNVPROC glewBindProgramNV;
GLEWAPI PFNGLDELETEPROGRAMSNVPROC glewDeleteProgramsNV;
GLEWAPI PFNGLEXECUTEPROGRAMNVPROC glewExecuteProgramNV;
GLEWAPI PFNGLGENPROGRAMSNVPROC glewGenProgramsNV;
GLEWAPI PFNGLGETPROGRAMPARAMETERDVNVPROC glewGetProgramParameterdvNV;
GLEWAPI PFNGLGETPROGRAMPARAMETERFVNVPROC glewGetProgramParameterfvNV;
GLEWAPI PFNGLGETPROGRAMSTRINGNVPROC glewGetProgramStringNV;
GLEWAPI PFNGLGETPROGRAMIVNVPROC glewGetProgramivNV;
GLEWAPI PFNGLGETTRACKMATRIXIVNVPROC glewGetTrackMatrixivNV;
GLEWAPI PFNGLGETVERTEXATTRIBPOINTERVNVPROC glewGetVertexAttribPointervNV;
GLEWAPI PFNGLGETVERTEXATTRIBDVNVPROC glewGetVertexAttribdvNV;
GLEWAPI PFNGLGETVERTEXATTRIBFVNVPROC glewGetVertexAttribfvNV;
GLEWAPI PFNGLGETVERTEXATTRIBIVNVPROC glewGetVertexAttribivNV;
GLEWAPI PFNGLISPROGRAMNVPROC glewIsProgramNV;
GLEWAPI PFNGLLOADPROGRAMNVPROC glewLoadProgramNV;
GLEWAPI PFNGLPROGRAMPARAMETER4DNVPROC glewProgramParameter4dNV;
GLEWAPI PFNGLPROGRAMPARAMETER4DVNVPROC glewProgramParameter4dvNV;
GLEWAPI PFNGLPROGRAMPARAMETER4FNVPROC glewProgramParameter4fNV;
GLEWAPI PFNGLPROGRAMPARAMETER4FVNVPROC glewProgramParameter4fvNV;
GLEWAPI PFNGLPROGRAMPARAMETERS4DVNVPROC glewProgramParameters4dvNV;
GLEWAPI PFNGLPROGRAMPARAMETERS4FVNVPROC glewProgramParameters4fvNV;
GLEWAPI PFNGLREQUESTRESIDENTPROGRAMSNVPROC glewRequestResidentProgramsNV;
GLEWAPI PFNGLTRACKMATRIXNVPROC glewTrackMatrixNV;
GLEWAPI PFNGLVERTEXATTRIB1DNVPROC glewVertexAttrib1dNV;
GLEWAPI PFNGLVERTEXATTRIB1DVNVPROC glewVertexAttrib1dvNV;
GLEWAPI PFNGLVERTEXATTRIB1FNVPROC glewVertexAttrib1fNV;
GLEWAPI PFNGLVERTEXATTRIB1FVNVPROC glewVertexAttrib1fvNV;
GLEWAPI PFNGLVERTEXATTRIB1SNVPROC glewVertexAttrib1sNV;
GLEWAPI PFNGLVERTEXATTRIB1SVNVPROC glewVertexAttrib1svNV;
GLEWAPI PFNGLVERTEXATTRIB2DNVPROC glewVertexAttrib2dNV;
GLEWAPI PFNGLVERTEXATTRIB2DVNVPROC glewVertexAttrib2dvNV;
GLEWAPI PFNGLVERTEXATTRIB2FNVPROC glewVertexAttrib2fNV;
GLEWAPI PFNGLVERTEXATTRIB2FVNVPROC glewVertexAttrib2fvNV;
GLEWAPI PFNGLVERTEXATTRIB2SNVPROC glewVertexAttrib2sNV;
GLEWAPI PFNGLVERTEXATTRIB2SVNVPROC glewVertexAttrib2svNV;
GLEWAPI PFNGLVERTEXATTRIB3DNVPROC glewVertexAttrib3dNV;
GLEWAPI PFNGLVERTEXATTRIB3DVNVPROC glewVertexAttrib3dvNV;
GLEWAPI PFNGLVERTEXATTRIB3FNVPROC glewVertexAttrib3fNV;
GLEWAPI PFNGLVERTEXATTRIB3FVNVPROC glewVertexAttrib3fvNV;
GLEWAPI PFNGLVERTEXATTRIB3SNVPROC glewVertexAttrib3sNV;
GLEWAPI PFNGLVERTEXATTRIB3SVNVPROC glewVertexAttrib3svNV;
GLEWAPI PFNGLVERTEXATTRIB4DNVPROC glewVertexAttrib4dNV;
GLEWAPI PFNGLVERTEXATTRIB4DVNVPROC glewVertexAttrib4dvNV;
GLEWAPI PFNGLVERTEXATTRIB4FNVPROC glewVertexAttrib4fNV;
GLEWAPI PFNGLVERTEXATTRIB4FVNVPROC glewVertexAttrib4fvNV;
GLEWAPI PFNGLVERTEXATTRIB4SNVPROC glewVertexAttrib4sNV;
GLEWAPI PFNGLVERTEXATTRIB4SVNVPROC glewVertexAttrib4svNV;
GLEWAPI PFNGLVERTEXATTRIB4UBNVPROC glewVertexAttrib4ubNV;
GLEWAPI PFNGLVERTEXATTRIB4UBVNVPROC glewVertexAttrib4ubvNV;
GLEWAPI PFNGLVERTEXATTRIBPOINTERNVPROC glewVertexAttribPointerNV;
GLEWAPI PFNGLVERTEXATTRIBS1DVNVPROC glewVertexAttribs1dvNV;
GLEWAPI PFNGLVERTEXATTRIBS1FVNVPROC glewVertexAttribs1fvNV;
GLEWAPI PFNGLVERTEXATTRIBS1SVNVPROC glewVertexAttribs1svNV;
GLEWAPI PFNGLVERTEXATTRIBS2DVNVPROC glewVertexAttribs2dvNV;
GLEWAPI PFNGLVERTEXATTRIBS2FVNVPROC glewVertexAttribs2fvNV;
GLEWAPI PFNGLVERTEXATTRIBS2SVNVPROC glewVertexAttribs2svNV;
GLEWAPI PFNGLVERTEXATTRIBS3DVNVPROC glewVertexAttribs3dvNV;
GLEWAPI PFNGLVERTEXATTRIBS3FVNVPROC glewVertexAttribs3fvNV;
GLEWAPI PFNGLVERTEXATTRIBS3SVNVPROC glewVertexAttribs3svNV;
GLEWAPI PFNGLVERTEXATTRIBS4DVNVPROC glewVertexAttribs4dvNV;
GLEWAPI PFNGLVERTEXATTRIBS4FVNVPROC glewVertexAttribs4fvNV;
GLEWAPI PFNGLVERTEXATTRIBS4SVNVPROC glewVertexAttribs4svNV;
GLEWAPI PFNGLVERTEXATTRIBS4UBVNVPROC glewVertexAttribs4ubvNV;

#define glAreProgramsResidentNV glewAreProgramsResidentNV
#define glBindProgramNV glewBindProgramNV
#define glDeleteProgramsNV glewDeleteProgramsNV
#define glExecuteProgramNV glewExecuteProgramNV
#define glGenProgramsNV glewGenProgramsNV
#define glGetProgramParameterdvNV glewGetProgramParameterdvNV
#define glGetProgramParameterfvNV glewGetProgramParameterfvNV
#define glGetProgramStringNV glewGetProgramStringNV
#define glGetProgramivNV glewGetProgramivNV
#define glGetTrackMatrixivNV glewGetTrackMatrixivNV
#define glGetVertexAttribPointervNV glewGetVertexAttribPointervNV
#define glGetVertexAttribdvNV glewGetVertexAttribdvNV
#define glGetVertexAttribfvNV glewGetVertexAttribfvNV
#define glGetVertexAttribivNV glewGetVertexAttribivNV
#define glIsProgramNV glewIsProgramNV
#define glLoadProgramNV glewLoadProgramNV
#define glProgramParameter4dNV glewProgramParameter4dNV
#define glProgramParameter4dvNV glewProgramParameter4dvNV
#define glProgramParameter4fNV glewProgramParameter4fNV
#define glProgramParameter4fvNV glewProgramParameter4fvNV
#define glProgramParameters4dvNV glewProgramParameters4dvNV
#define glProgramParameters4fvNV glewProgramParameters4fvNV
#define glRequestResidentProgramsNV glewRequestResidentProgramsNV
#define glTrackMatrixNV glewTrackMatrixNV
#define glVertexAttrib1dNV glewVertexAttrib1dNV
#define glVertexAttrib1dvNV glewVertexAttrib1dvNV
#define glVertexAttrib1fNV glewVertexAttrib1fNV
#define glVertexAttrib1fvNV glewVertexAttrib1fvNV
#define glVertexAttrib1sNV glewVertexAttrib1sNV
#define glVertexAttrib1svNV glewVertexAttrib1svNV
#define glVertexAttrib2dNV glewVertexAttrib2dNV
#define glVertexAttrib2dvNV glewVertexAttrib2dvNV
#define glVertexAttrib2fNV glewVertexAttrib2fNV
#define glVertexAttrib2fvNV glewVertexAttrib2fvNV
#define glVertexAttrib2sNV glewVertexAttrib2sNV
#define glVertexAttrib2svNV glewVertexAttrib2svNV
#define glVertexAttrib3dNV glewVertexAttrib3dNV
#define glVertexAttrib3dvNV glewVertexAttrib3dvNV
#define glVertexAttrib3fNV glewVertexAttrib3fNV
#define glVertexAttrib3fvNV glewVertexAttrib3fvNV
#define glVertexAttrib3sNV glewVertexAttrib3sNV
#define glVertexAttrib3svNV glewVertexAttrib3svNV
#define glVertexAttrib4dNV glewVertexAttrib4dNV
#define glVertexAttrib4dvNV glewVertexAttrib4dvNV
#define glVertexAttrib4fNV glewVertexAttrib4fNV
#define glVertexAttrib4fvNV glewVertexAttrib4fvNV
#define glVertexAttrib4sNV glewVertexAttrib4sNV
#define glVertexAttrib4svNV glewVertexAttrib4svNV
#define glVertexAttrib4ubNV glewVertexAttrib4ubNV
#define glVertexAttrib4ubvNV glewVertexAttrib4ubvNV
#define glVertexAttribPointerNV glewVertexAttribPointerNV
#define glVertexAttribs1dvNV glewVertexAttribs1dvNV
#define glVertexAttribs1fvNV glewVertexAttribs1fvNV
#define glVertexAttribs1svNV glewVertexAttribs1svNV
#define glVertexAttribs2dvNV glewVertexAttribs2dvNV
#define glVertexAttribs2fvNV glewVertexAttribs2fvNV
#define glVertexAttribs2svNV glewVertexAttribs2svNV
#define glVertexAttribs3dvNV glewVertexAttribs3dvNV
#define glVertexAttribs3fvNV glewVertexAttribs3fvNV
#define glVertexAttribs3svNV glewVertexAttribs3svNV
#define glVertexAttribs4dvNV glewVertexAttribs4dvNV
#define glVertexAttribs4fvNV glewVertexAttribs4fvNV
#define glVertexAttribs4svNV glewVertexAttribs4svNV
#define glVertexAttribs4ubvNV glewVertexAttribs4ubvNV

GLEWAPI GLboolean GLEW_NV_vertex_program;

#endif /* GL_NV_vertex_program */

/* ------------------------ GL_NV_vertex_program1_1 ------------------------ */

#ifndef GL_NV_vertex_program1_1
#define GL_NV_vertex_program1_1 1

GLEWAPI GLboolean GLEW_NV_vertex_program1_1;

#endif /* GL_NV_vertex_program1_1 */

/* ------------------------- GL_NV_vertex_program2 ------------------------- */

#ifndef GL_NV_vertex_program2
#define GL_NV_vertex_program2 1

GLEWAPI GLboolean GLEW_NV_vertex_program2;

#endif /* GL_NV_vertex_program2 */

/* ---------------------------- GL_OML_interlace --------------------------- */

#ifndef GL_OML_interlace
#define GL_OML_interlace 1

#define GL_INTERLACE_OML 0x8980
#define GL_INTERLACE_READ_OML 0x8981

GLEWAPI GLboolean GLEW_OML_interlace;

#endif /* GL_OML_interlace */

/* ---------------------------- GL_OML_resample ---------------------------- */

#ifndef GL_OML_resample
#define GL_OML_resample 1

#define GL_PACK_RESAMPLE_OML 0x8984
#define GL_UNPACK_RESAMPLE_OML 0x8985
#define GL_RESAMPLE_REPLICATE_OML 0x8986
#define GL_RESAMPLE_ZERO_FILL_OML 0x8987
#define GL_RESAMPLE_AVERAGE_OML 0x8988
#define GL_RESAMPLE_DECIMATE_OML 0x8989

GLEWAPI GLboolean GLEW_OML_resample;

#endif /* GL_OML_resample */

/* ---------------------------- GL_OML_subsample --------------------------- */

#ifndef GL_OML_subsample
#define GL_OML_subsample 1

#define GL_FORMAT_SUBSAMPLE_24_24_OML 0x8982
#define GL_FORMAT_SUBSAMPLE_244_244_OML 0x8983

GLEWAPI GLboolean GLEW_OML_subsample;

#endif /* GL_OML_subsample */

/* --------------------------- GL_PGI_misc_hints --------------------------- */

#ifndef GL_PGI_misc_hints
#define GL_PGI_misc_hints 1

#define GL_PREFER_DOUBLEBUFFER_HINT_PGI 107000
#define GL_CONSERVE_MEMORY_HINT_PGI 107005
#define GL_RECLAIM_MEMORY_HINT_PGI 107006
#define GL_NATIVE_GRAPHICS_HANDLE_PGI 107010
#define GL_NATIVE_GRAPHICS_BEGIN_HINT_PGI 107011
#define GL_NATIVE_GRAPHICS_END_HINT_PGI 107012
#define GL_ALWAYS_FAST_HINT_PGI 107020
#define GL_ALWAYS_SOFT_HINT_PGI 107021
#define GL_ALLOW_DRAW_OBJ_HINT_PGI 107022
#define GL_ALLOW_DRAW_WIN_HINT_PGI 107023
#define GL_ALLOW_DRAW_FRG_HINT_PGI 107024
#define GL_ALLOW_DRAW_MEM_HINT_PGI 107025
#define GL_STRICT_DEPTHFUNC_HINT_PGI 107030
#define GL_STRICT_LIGHTING_HINT_PGI 107031
#define GL_STRICT_SCISSOR_HINT_PGI 107032
#define GL_FULL_STIPPLE_HINT_PGI 107033
#define GL_CLIP_NEAR_HINT_PGI 107040
#define GL_CLIP_FAR_HINT_PGI 107041
#define GL_WIDE_LINE_HINT_PGI 107042
#define GL_BACK_NORMALS_HINT_PGI 107043

GLEWAPI GLboolean GLEW_PGI_misc_hints;

#endif /* GL_PGI_misc_hints */

/* -------------------------- GL_PGI_vertex_hints -------------------------- */

#ifndef GL_PGI_vertex_hints
#define GL_PGI_vertex_hints 1

#define GL_VERTEX23_BIT_PGI 0x00000004
#define GL_VERTEX4_BIT_PGI 0x00000008
#define GL_COLOR3_BIT_PGI 0x00010000
#define GL_COLOR4_BIT_PGI 0x00020000
#define GL_EDGEFLAG_BIT_PGI 0x00040000
#define GL_INDEX_BIT_PGI 0x00080000
#define GL_MAT_AMBIENT_BIT_PGI 0x00100000
#define GL_VERTEX_DATA_HINT_PGI 107050
#define GL_VERTEX_CONSISTENT_HINT_PGI 107051
#define GL_MATERIAL_SIDE_HINT_PGI 107052
#define GL_MAX_VERTEX_HINT_PGI 107053
#define GL_MAT_AMBIENT_AND_DIFFUSE_BIT_PGI 0x00200000
#define GL_MAT_DIFFUSE_BIT_PGI 0x00400000
#define GL_MAT_EMISSION_BIT_PGI 0x00800000
#define GL_MAT_COLOR_INDEXES_BIT_PGI 0x01000000
#define GL_MAT_SHININESS_BIT_PGI 0x02000000
#define GL_MAT_SPECULAR_BIT_PGI 0x04000000
#define GL_NORMAL_BIT_PGI 0x08000000
#define GL_TEXCOORD1_BIT_PGI 0x10000000
#define GL_TEXCOORD2_BIT_PGI 0x20000000
#define GL_TEXCOORD3_BIT_PGI 0x40000000
#define GL_TEXCOORD4_BIT_PGI 0x80000000

GLEWAPI GLboolean GLEW_PGI_vertex_hints;

#endif /* GL_PGI_vertex_hints */

/* ----------------------- GL_REND_screen_coordinates ---------------------- */

#ifndef GL_REND_screen_coordinates
#define GL_REND_screen_coordinates 1

#define GL_SCREEN_COORDINATES_REND 0x8490
#define GL_INVERTED_SCREEN_W_REND 0x8491

GLEWAPI GLboolean GLEW_REND_screen_coordinates;

#endif /* GL_REND_screen_coordinates */

/* ------------------------------- GL_S3_s3tc ------------------------------ */

#ifndef GL_S3_s3tc
#define GL_S3_s3tc 1

#define GL_RGB_S3TC 0x83A0
#define GL_RGB4_S3TC 0x83A1
#define GL_RGBA_S3TC 0x83A2
#define GL_RGBA4_S3TC 0x83A3
#define GL_RGBA_DXT5_S3TC 0x83A4
#define GL_RGBA4_DXT5_S3TC 0x83A5

GLEWAPI GLboolean GLEW_S3_s3tc;

#endif /* GL_S3_s3tc */

/* -------------------------- GL_SGIS_color_range -------------------------- */

#ifndef GL_SGIS_color_range
#define GL_SGIS_color_range 1

#define GLX_MIN_RED_SGIS 0
#define GLX_MAX_GREEN_SGIS 0
#define GLX_MIN_BLUE_SGIS 0
#define GLX_MAX_RED_SGIS 0
#define GLX_MAX_ALPHA_SGIS 0
#define GLX_MIN_GREEN_SGIS 0
#define GLX_MIN_ALPHA_SGIS 0
#define GLX_EXTENDED_RANGE_SGIS 0
#define GLX_MAX_BLUE_SGIS 0
#define GL_EXTENDED_RANGE_SGIS 0x85A5
#define GL_MIN_RED_SGIS 0x85A6
#define GL_MAX_RED_SGIS 0x85A7
#define GL_MIN_GREEN_SGIS 0x85A8
#define GL_MAX_GREEN_SGIS 0x85A9
#define GL_MIN_BLUE_SGIS 0x85AA
#define GL_MAX_BLUE_SGIS 0x85AB
#define GL_MIN_ALPHA_SGIS 0x85AC
#define GL_MAX_ALPHA_SGIS 0x85AD

GLEWAPI GLboolean GLEW_SGIS_color_range;

#endif /* GL_SGIS_color_range */

/* ------------------------- GL_SGIS_detail_texture ------------------------ */

#ifndef GL_SGIS_detail_texture
#define GL_SGIS_detail_texture 1

typedef void (GLAPIENTRY * PFNGLDETAILTEXFUNCSGISPROC) (GLenum target, GLsizei n, const GLfloat* points);
typedef void (GLAPIENTRY * PFNGLGETDETAILTEXFUNCSGISPROC) (GLenum target, GLfloat* points);

GLEWAPI PFNGLDETAILTEXFUNCSGISPROC glewDetailTexFuncSGIS;
GLEWAPI PFNGLGETDETAILTEXFUNCSGISPROC glewGetDetailTexFuncSGIS;

#define glDetailTexFuncSGIS glewDetailTexFuncSGIS
#define glGetDetailTexFuncSGIS glewGetDetailTexFuncSGIS

GLEWAPI GLboolean GLEW_SGIS_detail_texture;

#endif /* GL_SGIS_detail_texture */

/* -------------------------- GL_SGIS_fog_function ------------------------- */

#ifndef GL_SGIS_fog_function
#define GL_SGIS_fog_function 1

typedef void (GLAPIENTRY * PFNGLFOGFUNCSGISPROC) (GLsizei n, const GLfloat* points);
typedef void (GLAPIENTRY * PFNGLGETFOGFUNCSGISPROC) (GLfloat* points);

GLEWAPI PFNGLFOGFUNCSGISPROC glewFogFuncSGIS;
GLEWAPI PFNGLGETFOGFUNCSGISPROC glewGetFogFuncSGIS;

#define glFogFuncSGIS glewFogFuncSGIS
#define glGetFogFuncSGIS glewGetFogFuncSGIS

GLEWAPI GLboolean GLEW_SGIS_fog_function;

#endif /* GL_SGIS_fog_function */

/* ------------------------ GL_SGIS_generate_mipmap ------------------------ */

#ifndef GL_SGIS_generate_mipmap
#define GL_SGIS_generate_mipmap 1

#define GL_GENERATE_MIPMAP_SGIS 0x8191
#define GL_GENERATE_MIPMAP_HINT_SGIS 0x8192

GLEWAPI GLboolean GLEW_SGIS_generate_mipmap;

#endif /* GL_SGIS_generate_mipmap */

/* -------------------------- GL_SGIS_multisample -------------------------- */

#ifndef GL_SGIS_multisample
#define GL_SGIS_multisample 1

#define GL_MULTISAMPLE_SGIS 0x809D
#define GL_SAMPLE_ALPHA_TO_MASK_SGIS 0x809E
#define GL_SAMPLE_ALPHA_TO_ONE_SGIS 0x809F
#define GL_SAMPLE_MASK_SGIS 0x80A0
#define GL_1PASS_SGIS 0x80A1
#define GL_2PASS_0_SGIS 0x80A2
#define GL_2PASS_1_SGIS 0x80A3
#define GL_4PASS_0_SGIS 0x80A4
#define GL_4PASS_1_SGIS 0x80A5
#define GL_4PASS_2_SGIS 0x80A6
#define GL_4PASS_3_SGIS 0x80A7
#define GL_SAMPLE_BUFFERS_SGIS 0x80A8
#define GL_SAMPLES_SGIS 0x80A9
#define GL_SAMPLE_MASK_VALUE_SGIS 0x80AA
#define GL_SAMPLE_MASK_INVERT_SGIS 0x80AB
#define GL_SAMPLE_PATTERN_SGIS 0x80AC
#define GLX_SAMPLE_BUFFERS_SGIS 100000
#define GLX_SAMPLES_SGIS 100001
#define GL_MULTISAMPLE_BIT_EXT 0x20000000

typedef void (GLAPIENTRY * PFNGLSAMPLEMASKSGISPROC) (GLclampf value, GLboolean invert);
typedef void (GLAPIENTRY * PFNGLSAMPLEPATTERNSGISPROC) (GLenum pattern);

GLEWAPI PFNGLSAMPLEMASKSGISPROC glewSampleMaskSGIS;
GLEWAPI PFNGLSAMPLEPATTERNSGISPROC glewSamplePatternSGIS;

#define glSampleMaskSGIS glewSampleMaskSGIS
#define glSamplePatternSGIS glewSamplePatternSGIS

GLEWAPI GLboolean GLEW_SGIS_multisample;

#endif /* GL_SGIS_multisample */

/* ------------------------- GL_SGIS_pixel_texture ------------------------- */

#ifndef GL_SGIS_pixel_texture
#define GL_SGIS_pixel_texture 1

GLEWAPI GLboolean GLEW_SGIS_pixel_texture;

#endif /* GL_SGIS_pixel_texture */

/* ------------------------ GL_SGIS_sharpen_texture ------------------------ */

#ifndef GL_SGIS_sharpen_texture
#define GL_SGIS_sharpen_texture 1

typedef void (GLAPIENTRY * PFNGLGETSHARPENTEXFUNCSGISPROC) (GLenum target, GLfloat* points);
typedef void (GLAPIENTRY * PFNGLSHARPENTEXFUNCSGISPROC) (GLenum target, GLsizei n, const GLfloat* points);

GLEWAPI PFNGLGETSHARPENTEXFUNCSGISPROC glewGetSharpenTexFuncSGIS;
GLEWAPI PFNGLSHARPENTEXFUNCSGISPROC glewSharpenTexFuncSGIS;

#define glGetSharpenTexFuncSGIS glewGetSharpenTexFuncSGIS
#define glSharpenTexFuncSGIS glewSharpenTexFuncSGIS

GLEWAPI GLboolean GLEW_SGIS_sharpen_texture;

#endif /* GL_SGIS_sharpen_texture */

/* --------------------------- GL_SGIS_texture4D --------------------------- */

#ifndef GL_SGIS_texture4D
#define GL_SGIS_texture4D 1

typedef void (GLAPIENTRY * PFNGLTEXIMAGE4DSGISPROC) (GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLsizei extent, GLint border, GLenum format, GLenum type, const void* pixels);
typedef void (GLAPIENTRY * PFNGLTEXSUBIMAGE4DSGISPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint woffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei extent, GLenum format, GLenum type, const void* pixels);

GLEWAPI PFNGLTEXIMAGE4DSGISPROC glewTexImage4DSGIS;
GLEWAPI PFNGLTEXSUBIMAGE4DSGISPROC glewTexSubImage4DSGIS;

#define glTexImage4DSGIS glewTexImage4DSGIS
#define glTexSubImage4DSGIS glewTexSubImage4DSGIS

GLEWAPI GLboolean GLEW_SGIS_texture4D;

#endif /* GL_SGIS_texture4D */

/* ---------------------- GL_SGIS_texture_border_clamp --------------------- */

#ifndef GL_SGIS_texture_border_clamp
#define GL_SGIS_texture_border_clamp 1

#define GL_CLAMP_TO_BORDER_SGIS 0x812D

GLEWAPI GLboolean GLEW_SGIS_texture_border_clamp;

#endif /* GL_SGIS_texture_border_clamp */

/* ----------------------- GL_SGIS_texture_edge_clamp ---------------------- */

#ifndef GL_SGIS_texture_edge_clamp
#define GL_SGIS_texture_edge_clamp 1

#define GL_CLAMP_TO_EDGE_SGIS 0x812F

GLEWAPI GLboolean GLEW_SGIS_texture_edge_clamp;

#endif /* GL_SGIS_texture_edge_clamp */

/* ------------------------ GL_SGIS_texture_filter4 ------------------------ */

#ifndef GL_SGIS_texture_filter4
#define GL_SGIS_texture_filter4 1

typedef void (GLAPIENTRY * PFNGLGETTEXFILTERFUNCSGISPROC) (GLenum target, GLenum filter, GLfloat* weights);
typedef void (GLAPIENTRY * PFNGLTEXFILTERFUNCSGISPROC) (GLenum target, GLenum filter, GLsizei n, const GLfloat* weights);

GLEWAPI PFNGLGETTEXFILTERFUNCSGISPROC glewGetTexFilterFuncSGIS;
GLEWAPI PFNGLTEXFILTERFUNCSGISPROC glewTexFilterFuncSGIS;

#define glGetTexFilterFuncSGIS glewGetTexFilterFuncSGIS
#define glTexFilterFuncSGIS glewTexFilterFuncSGIS

GLEWAPI GLboolean GLEW_SGIS_texture_filter4;

#endif /* GL_SGIS_texture_filter4 */

/* -------------------------- GL_SGIS_texture_lod -------------------------- */

#ifndef GL_SGIS_texture_lod
#define GL_SGIS_texture_lod 1

#define GL_TEXTURE_MIN_LOD_SGIS 0x813A
#define GL_TEXTURE_MAX_LOD_SGIS 0x813B
#define GL_TEXTURE_BASE_LEVEL_SGIS 0x813C
#define GL_TEXTURE_MAX_LEVEL_SGIS 0x813D

GLEWAPI GLboolean GLEW_SGIS_texture_lod;

#endif /* GL_SGIS_texture_lod */

/* ------------------------- GL_SGIS_texture_select ------------------------ */

#ifndef GL_SGIS_texture_select
#define GL_SGIS_texture_select 1

GLEWAPI GLboolean GLEW_SGIS_texture_select;

#endif /* GL_SGIS_texture_select */

/* ----------------------------- GL_SGIX_async ----------------------------- */

#ifndef GL_SGIX_async
#define GL_SGIX_async 1

#define GL_ASYNC_MARKER_SGIX 0x8329

typedef void (GLAPIENTRY * PFNGLASYNCMARKERSGIXPROC) (GLuint marker);
typedef void (GLAPIENTRY * PFNGLDELETEASYNCMARKERSSGIXPROC) (GLuint marker, GLsizei range);
typedef GLint (GLAPIENTRY * PFNGLFINISHASYNCSGIXPROC) (GLuint* markerp);
typedef GLuint (GLAPIENTRY * PFNGLGENASYNCMARKERSSGIXPROC) (GLsizei range);
typedef GLboolean (GLAPIENTRY * PFNGLISASYNCMARKERSGIXPROC) (GLuint marker);
typedef GLint (GLAPIENTRY * PFNGLPOLLASYNCSGIXPROC) (GLuint* markerp);

GLEWAPI PFNGLASYNCMARKERSGIXPROC glewAsyncMarkerSGIX;
GLEWAPI PFNGLDELETEASYNCMARKERSSGIXPROC glewDeleteAsyncMarkersSGIX;
GLEWAPI PFNGLFINISHASYNCSGIXPROC glewFinishAsyncSGIX;
GLEWAPI PFNGLGENASYNCMARKERSSGIXPROC glewGenAsyncMarkersSGIX;
GLEWAPI PFNGLISASYNCMARKERSGIXPROC glewIsAsyncMarkerSGIX;
GLEWAPI PFNGLPOLLASYNCSGIXPROC glewPollAsyncSGIX;

#define glAsyncMarkerSGIX glewAsyncMarkerSGIX
#define glDeleteAsyncMarkersSGIX glewDeleteAsyncMarkersSGIX
#define glFinishAsyncSGIX glewFinishAsyncSGIX
#define glGenAsyncMarkersSGIX glewGenAsyncMarkersSGIX
#define glIsAsyncMarkerSGIX glewIsAsyncMarkerSGIX
#define glPollAsyncSGIX glewPollAsyncSGIX

GLEWAPI GLboolean GLEW_SGIX_async;

#endif /* GL_SGIX_async */

/* ------------------------ GL_SGIX_async_histogram ------------------------ */

#ifndef GL_SGIX_async_histogram
#define GL_SGIX_async_histogram 1

#define GL_ASYNC_HISTOGRAM_SGIX 0x832C
#define GL_MAX_ASYNC_HISTOGRAM_SGIX 0x832D

GLEWAPI GLboolean GLEW_SGIX_async_histogram;

#endif /* GL_SGIX_async_histogram */

/* -------------------------- GL_SGIX_async_pixel -------------------------- */

#ifndef GL_SGIX_async_pixel
#define GL_SGIX_async_pixel 1

#define GL_ASYNC_TEX_IMAGE_SGIX 0x835C
#define GL_ASYNC_DRAW_PIXELS_SGIX 0x835D
#define GL_ASYNC_READ_PIXELS_SGIX 0x835E
#define GL_MAX_ASYNC_TEX_IMAGE_SGIX 0x835F
#define GL_MAX_ASYNC_DRAW_PIXELS_SGIX 0x8360
#define GL_MAX_ASYNC_READ_PIXELS_SGIX 0x8361

GLEWAPI GLboolean GLEW_SGIX_async_pixel;

#endif /* GL_SGIX_async_pixel */

/* ----------------------- GL_SGIX_blend_alpha_minmax ---------------------- */

#ifndef GL_SGIX_blend_alpha_minmax
#define GL_SGIX_blend_alpha_minmax 1

#define GL_ALPHA_MIN_SGIX 0x8320
#define GL_ALPHA_MAX_SGIX 0x8321

GLEWAPI GLboolean GLEW_SGIX_blend_alpha_minmax;

#endif /* GL_SGIX_blend_alpha_minmax */

/* ---------------------------- GL_SGIX_clipmap ---------------------------- */

#ifndef GL_SGIX_clipmap
#define GL_SGIX_clipmap 1

GLEWAPI GLboolean GLEW_SGIX_clipmap;

#endif /* GL_SGIX_clipmap */

/* ------------------------- GL_SGIX_depth_texture ------------------------- */

#ifndef GL_SGIX_depth_texture
#define GL_SGIX_depth_texture 1

#define GL_DEPTH_COMPONENT16_SGIX 0x81A5
#define GL_DEPTH_COMPONENT24_SGIX 0x81A6
#define GL_DEPTH_COMPONENT32_SGIX 0x81A7

GLEWAPI GLboolean GLEW_SGIX_depth_texture;

#endif /* GL_SGIX_depth_texture */

/* -------------------------- GL_SGIX_flush_raster ------------------------- */

#ifndef GL_SGIX_flush_raster
#define GL_SGIX_flush_raster 1

typedef void (GLAPIENTRY * PFNGLFLUSHRASTERSGIXPROC) (void);

GLEWAPI PFNGLFLUSHRASTERSGIXPROC glewFlushRasterSGIX;

#define glFlushRasterSGIX glewFlushRasterSGIX

GLEWAPI GLboolean GLEW_SGIX_flush_raster;

#endif /* GL_SGIX_flush_raster */

/* --------------------------- GL_SGIX_fog_offset -------------------------- */

#ifndef GL_SGIX_fog_offset
#define GL_SGIX_fog_offset 1

#define GL_FOG_OFFSET_SGIX 0x8198
#define GL_FOG_OFFSET_VALUE_SGIX 0x8199

GLEWAPI GLboolean GLEW_SGIX_fog_offset;

#endif /* GL_SGIX_fog_offset */

/* -------------------------- GL_SGIX_fog_texture -------------------------- */

#ifndef GL_SGIX_fog_texture
#define GL_SGIX_fog_texture 1

#define GL_TEXTURE_FOG_SGIX 0
#define GL_FOG_PATCHY_FACTOR_SGIX 0
#define GL_FRAGMENT_FOG_SGIX 0

typedef void (GLAPIENTRY * PFNGLTEXTUREFOGSGIXPROC) (GLenum pname);

GLEWAPI PFNGLTEXTUREFOGSGIXPROC glewTextureFogSGIX;

#define glTextureFogSGIX glewTextureFogSGIX

GLEWAPI GLboolean GLEW_SGIX_fog_texture;

#endif /* GL_SGIX_fog_texture */

/* ------------------- GL_SGIX_fragment_specular_lighting ------------------ */

#ifndef GL_SGIX_fragment_specular_lighting
#define GL_SGIX_fragment_specular_lighting 1

typedef void (GLAPIENTRY * PFNGLFRAGMENTCOLORMATERIALSGIXPROC) (GLenum face, GLenum mode);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELFSGIXPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELFVSGIXPROC) (GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELISGIXPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTMODELIVSGIXPROC) (GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTFSGIXPROC) (GLenum light, GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTFVSGIXPROC) (GLenum light, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTISGIXPROC) (GLenum light, GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTLIGHTIVSGIXPROC) (GLenum light, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALFSGIXPROC) (GLenum face, GLenum pname, const GLfloat param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALFVSGIXPROC) (GLenum face, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALISGIXPROC) (GLenum face, GLenum pname, const GLint param);
typedef void (GLAPIENTRY * PFNGLFRAGMENTMATERIALIVSGIXPROC) (GLenum face, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTLIGHTFVSGIXPROC) (GLenum light, GLenum value, GLfloat* data);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTLIGHTIVSGIXPROC) (GLenum light, GLenum value, GLint* data);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTMATERIALFVSGIXPROC) (GLenum face, GLenum pname, const GLfloat* data);
typedef void (GLAPIENTRY * PFNGLGETFRAGMENTMATERIALIVSGIXPROC) (GLenum face, GLenum pname, const GLint* data);

GLEWAPI PFNGLFRAGMENTCOLORMATERIALSGIXPROC glewFragmentColorMaterialSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTMODELFSGIXPROC glewFragmentLightModelfSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTMODELFVSGIXPROC glewFragmentLightModelfvSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTMODELISGIXPROC glewFragmentLightModeliSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTMODELIVSGIXPROC glewFragmentLightModelivSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTFSGIXPROC glewFragmentLightfSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTFVSGIXPROC glewFragmentLightfvSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTISGIXPROC glewFragmentLightiSGIX;
GLEWAPI PFNGLFRAGMENTLIGHTIVSGIXPROC glewFragmentLightivSGIX;
GLEWAPI PFNGLFRAGMENTMATERIALFSGIXPROC glewFragmentMaterialfSGIX;
GLEWAPI PFNGLFRAGMENTMATERIALFVSGIXPROC glewFragmentMaterialfvSGIX;
GLEWAPI PFNGLFRAGMENTMATERIALISGIXPROC glewFragmentMaterialiSGIX;
GLEWAPI PFNGLFRAGMENTMATERIALIVSGIXPROC glewFragmentMaterialivSGIX;
GLEWAPI PFNGLGETFRAGMENTLIGHTFVSGIXPROC glewGetFragmentLightfvSGIX;
GLEWAPI PFNGLGETFRAGMENTLIGHTIVSGIXPROC glewGetFragmentLightivSGIX;
GLEWAPI PFNGLGETFRAGMENTMATERIALFVSGIXPROC glewGetFragmentMaterialfvSGIX;
GLEWAPI PFNGLGETFRAGMENTMATERIALIVSGIXPROC glewGetFragmentMaterialivSGIX;

#define glFragmentColorMaterialSGIX glewFragmentColorMaterialSGIX
#define glFragmentLightModelfSGIX glewFragmentLightModelfSGIX
#define glFragmentLightModelfvSGIX glewFragmentLightModelfvSGIX
#define glFragmentLightModeliSGIX glewFragmentLightModeliSGIX
#define glFragmentLightModelivSGIX glewFragmentLightModelivSGIX
#define glFragmentLightfSGIX glewFragmentLightfSGIX
#define glFragmentLightfvSGIX glewFragmentLightfvSGIX
#define glFragmentLightiSGIX glewFragmentLightiSGIX
#define glFragmentLightivSGIX glewFragmentLightivSGIX
#define glFragmentMaterialfSGIX glewFragmentMaterialfSGIX
#define glFragmentMaterialfvSGIX glewFragmentMaterialfvSGIX
#define glFragmentMaterialiSGIX glewFragmentMaterialiSGIX
#define glFragmentMaterialivSGIX glewFragmentMaterialivSGIX
#define glGetFragmentLightfvSGIX glewGetFragmentLightfvSGIX
#define glGetFragmentLightivSGIX glewGetFragmentLightivSGIX
#define glGetFragmentMaterialfvSGIX glewGetFragmentMaterialfvSGIX
#define glGetFragmentMaterialivSGIX glewGetFragmentMaterialivSGIX

GLEWAPI GLboolean GLEW_SGIX_fragment_specular_lighting;

#endif /* GL_SGIX_fragment_specular_lighting */

/* --------------------------- GL_SGIX_framezoom --------------------------- */

#ifndef GL_SGIX_framezoom
#define GL_SGIX_framezoom 1

typedef void (GLAPIENTRY * PFNGLFRAMEZOOMSGIXPROC) (GLint factor);

GLEWAPI PFNGLFRAMEZOOMSGIXPROC glewFrameZoomSGIX;

#define glFrameZoomSGIX glewFrameZoomSGIX

GLEWAPI GLboolean GLEW_SGIX_framezoom;

#endif /* GL_SGIX_framezoom */

/* --------------------------- GL_SGIX_interlace --------------------------- */

#ifndef GL_SGIX_interlace
#define GL_SGIX_interlace 1

#define GL_INTERLACE_SGIX 0x8094

GLEWAPI GLboolean GLEW_SGIX_interlace;

#endif /* GL_SGIX_interlace */

/* ------------------------- GL_SGIX_ir_instrument1 ------------------------ */

#ifndef GL_SGIX_ir_instrument1
#define GL_SGIX_ir_instrument1 1

GLEWAPI GLboolean GLEW_SGIX_ir_instrument1;

#endif /* GL_SGIX_ir_instrument1 */

/* ------------------------- GL_SGIX_list_priority ------------------------- */

#ifndef GL_SGIX_list_priority
#define GL_SGIX_list_priority 1

GLEWAPI GLboolean GLEW_SGIX_list_priority;

#endif /* GL_SGIX_list_priority */

/* ------------------------- GL_SGIX_pixel_texture ------------------------- */

#ifndef GL_SGIX_pixel_texture
#define GL_SGIX_pixel_texture 1

typedef void (GLAPIENTRY * PFNGLPIXELTEXGENSGIXPROC) (GLenum mode);

GLEWAPI PFNGLPIXELTEXGENSGIXPROC glewPixelTexGenSGIX;

#define glPixelTexGenSGIX glewPixelTexGenSGIX

GLEWAPI GLboolean GLEW_SGIX_pixel_texture;

#endif /* GL_SGIX_pixel_texture */

/* ----------------------- GL_SGIX_pixel_texture_bits ---------------------- */

#ifndef GL_SGIX_pixel_texture_bits
#define GL_SGIX_pixel_texture_bits 1

GLEWAPI GLboolean GLEW_SGIX_pixel_texture_bits;

#endif /* GL_SGIX_pixel_texture_bits */

/* ------------------------ GL_SGIX_reference_plane ------------------------ */

#ifndef GL_SGIX_reference_plane
#define GL_SGIX_reference_plane 1

typedef void (GLAPIENTRY * PFNGLREFERENCEPLANESGIXPROC) (const GLdouble* equation);

GLEWAPI PFNGLREFERENCEPLANESGIXPROC glewReferencePlaneSGIX;

#define glReferencePlaneSGIX glewReferencePlaneSGIX

GLEWAPI GLboolean GLEW_SGIX_reference_plane;

#endif /* GL_SGIX_reference_plane */

/* ---------------------------- GL_SGIX_resample --------------------------- */

#ifndef GL_SGIX_resample
#define GL_SGIX_resample 1

#define GL_PACK_RESAMPLE_SGIX 0x842E
#define GL_UNPACK_RESAMPLE_SGIX 0x842F
#define GL_RESAMPLE_DECIMATE_SGIX 0x8430
#define GL_RESAMPLE_REPLICATE_SGIX 0x8433
#define GL_RESAMPLE_ZERO_FILL_SGIX 0x8434

GLEWAPI GLboolean GLEW_SGIX_resample;

#endif /* GL_SGIX_resample */

/* ----------------------------- GL_SGIX_shadow ---------------------------- */

#ifndef GL_SGIX_shadow
#define GL_SGIX_shadow 1

GLEWAPI GLboolean GLEW_SGIX_shadow;

#endif /* GL_SGIX_shadow */

/* ------------------------- GL_SGIX_shadow_ambient ------------------------ */

#ifndef GL_SGIX_shadow_ambient
#define GL_SGIX_shadow_ambient 1

#define GL_SHADOW_AMBIENT_SGIX 0x80BF

GLEWAPI GLboolean GLEW_SGIX_shadow_ambient;

#endif /* GL_SGIX_shadow_ambient */

/* ----------------------------- GL_SGIX_sprite ---------------------------- */

#ifndef GL_SGIX_sprite
#define GL_SGIX_sprite 1

typedef void (GLAPIENTRY * PFNGLSPRITEPARAMETERFSGIXPROC) (GLenum pname, GLfloat param);
typedef void (GLAPIENTRY * PFNGLSPRITEPARAMETERFVSGIXPROC) (GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLSPRITEPARAMETERISGIXPROC) (GLenum pname, GLint param);
typedef void (GLAPIENTRY * PFNGLSPRITEPARAMETERIVSGIXPROC) (GLenum pname, GLint* params);

GLEWAPI PFNGLSPRITEPARAMETERFSGIXPROC glewSpriteParameterfSGIX;
GLEWAPI PFNGLSPRITEPARAMETERFVSGIXPROC glewSpriteParameterfvSGIX;
GLEWAPI PFNGLSPRITEPARAMETERISGIXPROC glewSpriteParameteriSGIX;
GLEWAPI PFNGLSPRITEPARAMETERIVSGIXPROC glewSpriteParameterivSGIX;

#define glSpriteParameterfSGIX glewSpriteParameterfSGIX
#define glSpriteParameterfvSGIX glewSpriteParameterfvSGIX
#define glSpriteParameteriSGIX glewSpriteParameteriSGIX
#define glSpriteParameterivSGIX glewSpriteParameterivSGIX

GLEWAPI GLboolean GLEW_SGIX_sprite;

#endif /* GL_SGIX_sprite */

/* ----------------------- GL_SGIX_tag_sample_buffer ----------------------- */

#ifndef GL_SGIX_tag_sample_buffer
#define GL_SGIX_tag_sample_buffer 1

typedef void (GLAPIENTRY * PFNGLTAGSAMPLEBUFFERSGIXPROC) (void);

GLEWAPI PFNGLTAGSAMPLEBUFFERSGIXPROC glewTagSampleBufferSGIX;

#define glTagSampleBufferSGIX glewTagSampleBufferSGIX

GLEWAPI GLboolean GLEW_SGIX_tag_sample_buffer;

#endif /* GL_SGIX_tag_sample_buffer */

/* ------------------------ GL_SGIX_texture_add_env ------------------------ */

#ifndef GL_SGIX_texture_add_env
#define GL_SGIX_texture_add_env 1

GLEWAPI GLboolean GLEW_SGIX_texture_add_env;

#endif /* GL_SGIX_texture_add_env */

/* -------------------- GL_SGIX_texture_coordinate_clamp ------------------- */

#ifndef GL_SGIX_texture_coordinate_clamp
#define GL_SGIX_texture_coordinate_clamp 1

#define GL_TEXTURE_MAX_CLAMP_S_SGIX 0x8369
#define GL_TEXTURE_MAX_CLAMP_T_SGIX 0x836A
#define GL_TEXTURE_MAX_CLAMP_R_SGIX 0x836B

GLEWAPI GLboolean GLEW_SGIX_texture_coordinate_clamp;

#endif /* GL_SGIX_texture_coordinate_clamp */

/* ------------------------ GL_SGIX_texture_lod_bias ----------------------- */

#ifndef GL_SGIX_texture_lod_bias
#define GL_SGIX_texture_lod_bias 1

GLEWAPI GLboolean GLEW_SGIX_texture_lod_bias;

#endif /* GL_SGIX_texture_lod_bias */

/* ---------------------- GL_SGIX_texture_multi_buffer --------------------- */

#ifndef GL_SGIX_texture_multi_buffer
#define GL_SGIX_texture_multi_buffer 1

#define GL_TEXTURE_MULTI_BUFFER_HINT_SGIX 0x812E

GLEWAPI GLboolean GLEW_SGIX_texture_multi_buffer;

#endif /* GL_SGIX_texture_multi_buffer */

/* ------------------------- GL_SGIX_texture_range ------------------------- */

#ifndef GL_SGIX_texture_range
#define GL_SGIX_texture_range 1

#define GL_RGB_SIGNED_SGIX 0x85E0
#define GL_RGBA_SIGNED_SGIX 0x85E1
#define GL_ALPHA_SIGNED_SGIX 0x85E2
#define GL_LUMINANCE_SIGNED_SGIX 0x85E3
#define GL_INTENSITY_SIGNED_SGIX 0x85E4
#define GL_LUMINANCE_ALPHA_SIGNED_SGIX 0x85E5
#define GL_RGB16_SIGNED_SGIX 0x85E6
#define GL_RGBA16_SIGNED_SGIX 0x85E7
#define GL_ALPHA16_SIGNED_SGIX 0x85E8
#define GL_LUMINANCE16_SIGNED_SGIX 0x85E9
#define GL_INTENSITY16_SIGNED_SGIX 0x85EA
#define GL_LUMINANCE16_ALPHA16_SIGNED_SGIX 0x85EB
#define GL_RGB_EXTENDED_RANGE_SGIX 0x85EC
#define GL_RGBA_EXTENDED_RANGE_SGIX 0x85ED
#define GL_ALPHA_EXTENDED_RANGE_SGIX 0x85EE
#define GL_LUMINANCE_EXTENDED_RANGE_SGIX 0x85EF
#define GL_INTENSITY_EXTENDED_RANGE_SGIX 0x85F0
#define GL_LUMINANCE_ALPHA_EXTENDED_RANGE_SGIX 0x85F1
#define GL_RGB16_EXTENDED_RANGE_SGIX 0x85F2
#define GL_RGBA16_EXTENDED_RANGE_SGIX 0x85F3
#define GL_ALPHA16_EXTENDED_RANGE_SGIX 0x85F4
#define GL_LUMINANCE16_EXTENDED_RANGE_SGIX 0x85F5
#define GL_INTENSITY16_EXTENDED_RANGE_SGIX 0x85F6
#define GL_LUMINANCE16_ALPHA16_EXTENDED_RANGE_SGIX 0x85F7
#define GL_MIN_LUMINANCE_SGIS 0x85F8
#define GL_MAX_LUMINANCE_SGIS 0x85F9
#define GL_MIN_INTENSITY_SGIS 0x85FA
#define GL_MAX_INTENSITY_SGIS 0x85FB

GLEWAPI GLboolean GLEW_SGIX_texture_range;

#endif /* GL_SGIX_texture_range */

/* ----------------------- GL_SGIX_texture_scale_bias ---------------------- */

#ifndef GL_SGIX_texture_scale_bias
#define GL_SGIX_texture_scale_bias 1

#define GL_POST_TEXTURE_FILTER_BIAS_SGIX 0x8179
#define GL_POST_TEXTURE_FILTER_SCALE_SGIX 0x817A
#define GL_POST_TEXTURE_FILTER_BIAS_RANGE_SGIX 0x817B
#define GL_POST_TEXTURE_FILTER_SCALE_RANGE_SGIX 0x817C

GLEWAPI GLboolean GLEW_SGIX_texture_scale_bias;

#endif /* GL_SGIX_texture_scale_bias */

/* ------------------------- GL_SGIX_vertex_preclip ------------------------ */

#ifndef GL_SGIX_vertex_preclip
#define GL_SGIX_vertex_preclip 1

#define GL_VERTEX_PRECLIP_SGIX 0x83EE
#define GL_VERTEX_PRECLIP_HINT_SGIX 0x83EF

GLEWAPI GLboolean GLEW_SGIX_vertex_preclip;

#endif /* GL_SGIX_vertex_preclip */

/* ---------------------- GL_SGIX_vertex_preclip_hint ---------------------- */

#ifndef GL_SGIX_vertex_preclip_hint
#define GL_SGIX_vertex_preclip_hint 1

#define GL_VERTEX_PRECLIP_SGIX 0x83EE
#define GL_VERTEX_PRECLIP_HINT_SGIX 0x83EF

GLEWAPI GLboolean GLEW_SGIX_vertex_preclip_hint;

#endif /* GL_SGIX_vertex_preclip_hint */

/* ----------------------------- GL_SGIX_ycrcb ----------------------------- */

#ifndef GL_SGIX_ycrcb
#define GL_SGIX_ycrcb 1

GLEWAPI GLboolean GLEW_SGIX_ycrcb;

#endif /* GL_SGIX_ycrcb */

/* -------------------------- GL_SGI_color_matrix -------------------------- */

#ifndef GL_SGI_color_matrix
#define GL_SGI_color_matrix 1

#define GL_COLOR_MATRIX_SGI 0x80B1
#define GL_COLOR_MATRIX_STACK_DEPTH_SGI 0x80B2
#define GL_MAX_COLOR_MATRIX_STACK_DEPTH_SGI 0x80B3
#define GL_POST_COLOR_MATRIX_RED_SCALE_SGI 0x80B4
#define GL_POST_COLOR_MATRIX_GREEN_SCALE_SGI 0x80B5
#define GL_POST_COLOR_MATRIX_BLUE_SCALE_SGI 0x80B6
#define GL_POST_COLOR_MATRIX_ALPHA_SCALE_SGI 0x80B7
#define GL_POST_COLOR_MATRIX_RED_BIAS_SGI 0x80B8
#define GL_POST_COLOR_MATRIX_GREEN_BIAS_SGI 0x80B9
#define GL_POST_COLOR_MATRIX_BLUE_BIAS_SGI 0x80BA
#define GL_POST_COLOR_MATRIX_ALPHA_BIAS_SGI 0x80BB

GLEWAPI GLboolean GLEW_SGI_color_matrix;

#endif /* GL_SGI_color_matrix */

/* --------------------------- GL_SGI_color_table -------------------------- */

#ifndef GL_SGI_color_table
#define GL_SGI_color_table 1

#define GL_COLOR_TABLE_SGI 0x80D0
#define GL_POST_CONVOLUTION_COLOR_TABLE_SGI 0x80D1
#define GL_POST_COLOR_MATRIX_COLOR_TABLE_SGI 0x80D2
#define GL_PROXY_COLOR_TABLE_SGI 0x80D3
#define GL_PROXY_POST_CONVOLUTION_COLOR_TABLE_SGI 0x80D4
#define GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE_SGI 0x80D5
#define GL_COLOR_TABLE_SCALE_SGI 0x80D6
#define GL_COLOR_TABLE_BIAS_SGI 0x80D7
#define GL_COLOR_TABLE_FORMAT_SGI 0x80D8
#define GL_COLOR_TABLE_WIDTH_SGI 0x80D9
#define GL_COLOR_TABLE_RED_SIZE_SGI 0x80DA
#define GL_COLOR_TABLE_GREEN_SIZE_SGI 0x80DB
#define GL_COLOR_TABLE_BLUE_SIZE_SGI 0x80DC
#define GL_COLOR_TABLE_ALPHA_SIZE_SGI 0x80DD
#define GL_COLOR_TABLE_LUMINANCE_SIZE_SGI 0x80DE
#define GL_COLOR_TABLE_INTENSITY_SIZE_SGI 0x80DF

typedef void (GLAPIENTRY * PFNGLCOLORTABLEPARAMETERFVSGIPROC) (GLenum target, GLenum pname, const GLfloat* params);
typedef void (GLAPIENTRY * PFNGLCOLORTABLEPARAMETERIVSGIPROC) (GLenum target, GLenum pname, const GLint* params);
typedef void (GLAPIENTRY * PFNGLCOLORTABLESGIPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const void* table);
typedef void (GLAPIENTRY * PFNGLCOPYCOLORTABLESGIPROC) (GLenum target, GLenum internalformat, GLint x, GLint y, GLsizei width);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPARAMETERFVSGIPROC) (GLenum target, GLenum pname, GLfloat* params);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLEPARAMETERIVSGIPROC) (GLenum target, GLenum pname, GLint* params);
typedef void (GLAPIENTRY * PFNGLGETCOLORTABLESGIPROC) (GLenum target, GLenum format, GLenum type, void* table);

GLEWAPI PFNGLCOLORTABLEPARAMETERFVSGIPROC glewColorTableParameterfvSGI;
GLEWAPI PFNGLCOLORTABLEPARAMETERIVSGIPROC glewColorTableParameterivSGI;
GLEWAPI PFNGLCOLORTABLESGIPROC glewColorTableSGI;
GLEWAPI PFNGLCOPYCOLORTABLESGIPROC glewCopyColorTableSGI;
GLEWAPI PFNGLGETCOLORTABLEPARAMETERFVSGIPROC glewGetColorTableParameterfvSGI;
GLEWAPI PFNGLGETCOLORTABLEPARAMETERIVSGIPROC glewGetColorTableParameterivSGI;
GLEWAPI PFNGLGETCOLORTABLESGIPROC glewGetColorTableSGI;

#define glColorTableParameterfvSGI glewColorTableParameterfvSGI
#define glColorTableParameterivSGI glewColorTableParameterivSGI
#define glColorTableSGI glewColorTableSGI
#define glCopyColorTableSGI glewCopyColorTableSGI
#define glGetColorTableParameterfvSGI glewGetColorTableParameterfvSGI
#define glGetColorTableParameterivSGI glewGetColorTableParameterivSGI
#define glGetColorTableSGI glewGetColorTableSGI

GLEWAPI GLboolean GLEW_SGI_color_table;

#endif /* GL_SGI_color_table */

/* ----------------------- GL_SGI_texture_color_table ---------------------- */

#ifndef GL_SGI_texture_color_table
#define GL_SGI_texture_color_table 1

#define GL_TEXTURE_COLOR_TABLE_SGI 0x80BC
#define GL_PROXY_TEXTURE_COLOR_TABLE_SGI 0x80BD

GLEWAPI GLboolean GLEW_SGI_texture_color_table;

#endif /* GL_SGI_texture_color_table */

/* ------------------------- GL_SUNX_constant_data ------------------------- */

#ifndef GL_SUNX_constant_data
#define GL_SUNX_constant_data 1

#define GL_UNPACK_CONSTANT_DATA_SUNX 0x81D5
#define GL_TEXTURE_CONSTANT_DATA_SUNX 0x81D6

typedef void (GLAPIENTRY * PFNGLFINISHTEXTURESUNXPROC) (void);

GLEWAPI PFNGLFINISHTEXTURESUNXPROC glewFinishTextureSUNX;

#define glFinishTextureSUNX glewFinishTextureSUNX

GLEWAPI GLboolean GLEW_SUNX_constant_data;

#endif /* GL_SUNX_constant_data */

/* -------------------- GL_SUN_convolution_border_modes -------------------- */

#ifndef GL_SUN_convolution_border_modes
#define GL_SUN_convolution_border_modes 1

#define GL_WRAP_BORDER_SUN 0x81D4

GLEWAPI GLboolean GLEW_SUN_convolution_border_modes;

#endif /* GL_SUN_convolution_border_modes */

/* -------------------------- GL_SUN_global_alpha -------------------------- */

#ifndef GL_SUN_global_alpha
#define GL_SUN_global_alpha 1

#define GL_GLOBAL_ALPHA_SUN 0x81D9
#define GL_GLOBAL_ALPHA_FACTOR_SUN 0x81DA

typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORBSUNPROC) (GLbyte factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORDSUNPROC) (GLdouble factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORFSUNPROC) (GLfloat factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORISUNPROC) (GLint factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORSSUNPROC) (GLshort factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORUBSUNPROC) (GLubyte factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORUISUNPROC) (GLuint factor);
typedef void (GLAPIENTRY * PFNGLGLOBALALPHAFACTORUSSUNPROC) (GLushort factor);

GLEWAPI PFNGLGLOBALALPHAFACTORBSUNPROC glewGlobalAlphaFactorbSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORDSUNPROC glewGlobalAlphaFactordSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORFSUNPROC glewGlobalAlphaFactorfSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORISUNPROC glewGlobalAlphaFactoriSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORSSUNPROC glewGlobalAlphaFactorsSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORUBSUNPROC glewGlobalAlphaFactorubSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORUISUNPROC glewGlobalAlphaFactoruiSUN;
GLEWAPI PFNGLGLOBALALPHAFACTORUSSUNPROC glewGlobalAlphaFactorusSUN;

#define glGlobalAlphaFactorbSUN glewGlobalAlphaFactorbSUN
#define glGlobalAlphaFactordSUN glewGlobalAlphaFactordSUN
#define glGlobalAlphaFactorfSUN glewGlobalAlphaFactorfSUN
#define glGlobalAlphaFactoriSUN glewGlobalAlphaFactoriSUN
#define glGlobalAlphaFactorsSUN glewGlobalAlphaFactorsSUN
#define glGlobalAlphaFactorubSUN glewGlobalAlphaFactorubSUN
#define glGlobalAlphaFactoruiSUN glewGlobalAlphaFactoruiSUN
#define glGlobalAlphaFactorusSUN glewGlobalAlphaFactorusSUN

GLEWAPI GLboolean GLEW_SUN_global_alpha;

#endif /* GL_SUN_global_alpha */

/* --------------------------- GL_SUN_mesh_array --------------------------- */

#ifndef GL_SUN_mesh_array
#define GL_SUN_mesh_array 1

#define GL_QUAD_MESH_SUN 0x8614
#define GL_TRIANGLE_MESH_SUN 0x8615

GLEWAPI GLboolean GLEW_SUN_mesh_array;

#endif /* GL_SUN_mesh_array */

/* --------------------------- GL_SUN_slice_accum -------------------------- */

#ifndef GL_SUN_slice_accum
#define GL_SUN_slice_accum 1

#define GL_SLICE_ACCUM_SUN 0x85CC

GLEWAPI GLboolean GLEW_SUN_slice_accum;

#endif /* GL_SUN_slice_accum */

/* -------------------------- GL_SUN_triangle_list ------------------------- */

#ifndef GL_SUN_triangle_list
#define GL_SUN_triangle_list 1

#define GL_RESTART_SUN 0x01
#define GL_REPLACE_MIDDLE_SUN 0x02
#define GL_REPLACE_OLDEST_SUN 0x03
#define GL_TRIANGLE_LIST_SUN 0x81D7
#define GL_REPLACEMENT_CODE_SUN 0x81D8
#define GL_REPLACEMENT_CODE_ARRAY_SUN 0x85C0
#define GL_REPLACEMENT_CODE_ARRAY_TYPE_SUN 0x85C1
#define GL_REPLACEMENT_CODE_ARRAY_STRIDE_SUN 0x85C2
#define GL_REPLACEMENT_CODE_ARRAY_POINTER_SUN 0x85C3
#define GL_R1UI_V3F_SUN 0x85C4
#define GL_R1UI_C4UB_V3F_SUN 0x85C5
#define GL_R1UI_C3F_V3F_SUN 0x85C6
#define GL_R1UI_N3F_V3F_SUN 0x85C7
#define GL_R1UI_C4F_N3F_V3F_SUN 0x85C8
#define GL_R1UI_T2F_V3F_SUN 0x85C9
#define GL_R1UI_T2F_N3F_V3F_SUN 0x85CA
#define GL_R1UI_T2F_C4F_N3F_V3F_SUN 0x85CB

typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEPOINTERSUNPROC) (GLenum type, GLsizei stride, const void* pointer);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUBSUNPROC) (GLubyte code);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUBVSUNPROC) (const GLubyte* code);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUISUNPROC) (GLuint code);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUIVSUNPROC) (const GLuint* code);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUSSUNPROC) (GLushort code);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUSVSUNPROC) (const GLushort* code);

GLEWAPI PFNGLREPLACEMENTCODEPOINTERSUNPROC glewReplacementCodePointerSUN;
GLEWAPI PFNGLREPLACEMENTCODEUBSUNPROC glewReplacementCodeubSUN;
GLEWAPI PFNGLREPLACEMENTCODEUBVSUNPROC glewReplacementCodeubvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUISUNPROC glewReplacementCodeuiSUN;
GLEWAPI PFNGLREPLACEMENTCODEUIVSUNPROC glewReplacementCodeuivSUN;
GLEWAPI PFNGLREPLACEMENTCODEUSSUNPROC glewReplacementCodeusSUN;
GLEWAPI PFNGLREPLACEMENTCODEUSVSUNPROC glewReplacementCodeusvSUN;

#define glReplacementCodePointerSUN glewReplacementCodePointerSUN
#define glReplacementCodeubSUN glewReplacementCodeubSUN
#define glReplacementCodeubvSUN glewReplacementCodeubvSUN
#define glReplacementCodeuiSUN glewReplacementCodeuiSUN
#define glReplacementCodeuivSUN glewReplacementCodeuivSUN
#define glReplacementCodeusSUN glewReplacementCodeusSUN
#define glReplacementCodeusvSUN glewReplacementCodeusvSUN

GLEWAPI GLboolean GLEW_SUN_triangle_list;

#endif /* GL_SUN_triangle_list */

/* ----------------------------- GL_SUN_vertex ----------------------------- */

#ifndef GL_SUN_vertex
#define GL_SUN_vertex 1

typedef void (GLAPIENTRY * PFNGLCOLOR3FVERTEX3FSUNPROC) (GLfloat r, GLfloat g, GLfloat b, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLCOLOR3FVERTEX3FVSUNPROC) (const GLfloat* c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLCOLOR4FNORMAL3FVERTEX3FSUNPROC) (GLfloat r, GLfloat g, GLfloat b, GLfloat a, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLCOLOR4FNORMAL3FVERTEX3FVSUNPROC) (const GLfloat* c, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLCOLOR4UBVERTEX2FSUNPROC) (GLubyte r, GLubyte g, GLubyte b, GLubyte a, GLfloat x, GLfloat y);
typedef void (GLAPIENTRY * PFNGLCOLOR4UBVERTEX2FVSUNPROC) (const GLubyte* c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLCOLOR4UBVERTEX3FSUNPROC) (GLubyte r, GLubyte g, GLubyte b, GLubyte a, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLCOLOR4UBVERTEX3FVSUNPROC) (const GLubyte* c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLNORMAL3FVERTEX3FSUNPROC) (GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLNORMAL3FVERTEX3FVSUNPROC) (const GLfloat* n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FSUNPROC) (GLuint rc, GLfloat r, GLfloat g, GLfloat b, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FSUNPROC) (GLuint rc, GLfloat r, GLfloat g, GLfloat b, GLfloat a, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *c, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FSUNPROC) (GLuint rc, GLubyte r, GLubyte g, GLubyte b, GLubyte a, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FVSUNPROC) (const GLuint* rc, const GLubyte *c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FSUNPROC) (GLuint rc, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC) (GLuint rc, GLfloat s, GLfloat t, GLfloat r, GLfloat g, GLfloat b, GLfloat a, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *tc, const GLfloat *c, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FSUNPROC) (GLuint rc, GLfloat s, GLfloat t, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *tc, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FSUNPROC) (GLuint rc, GLfloat s, GLfloat t, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *tc, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUIVERTEX3FSUNPROC) (GLuint rc, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLREPLACEMENTCODEUIVERTEX3FVSUNPROC) (const GLuint* rc, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FCOLOR3FVERTEX3FSUNPROC) (GLfloat s, GLfloat t, GLfloat r, GLfloat g, GLfloat b, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FCOLOR3FVERTEX3FVSUNPROC) (const GLfloat* tc, const GLfloat *c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC) (GLfloat s, GLfloat t, GLfloat r, GLfloat g, GLfloat b, GLfloat a, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC) (const GLfloat* tc, const GLfloat *c, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FCOLOR4UBVERTEX3FSUNPROC) (GLfloat s, GLfloat t, GLubyte r, GLubyte g, GLubyte b, GLubyte a, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FCOLOR4UBVERTEX3FVSUNPROC) (const GLfloat* tc, const GLubyte *c, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FNORMAL3FVERTEX3FSUNPROC) (GLfloat s, GLfloat t, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FNORMAL3FVERTEX3FVSUNPROC) (const GLfloat* tc, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FVERTEX3FSUNPROC) (GLfloat s, GLfloat t, GLfloat x, GLfloat y, GLfloat z);
typedef void (GLAPIENTRY * PFNGLTEXCOORD2FVERTEX3FVSUNPROC) (const GLfloat* tc, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FSUNPROC) (GLfloat s, GLfloat t, GLfloat p, GLfloat q, GLfloat r, GLfloat g, GLfloat b, GLfloat a, GLfloat nx, GLfloat ny, GLfloat nz, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FVSUNPROC) (const GLfloat* tc, const GLfloat *c, const GLfloat *n, const GLfloat *v);
typedef void (GLAPIENTRY * PFNGLTEXCOORD4FVERTEX4FSUNPROC) (GLfloat s, GLfloat t, GLfloat p, GLfloat q, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (GLAPIENTRY * PFNGLTEXCOORD4FVERTEX4FVSUNPROC) (const GLfloat* tc, const GLfloat *v);

GLEWAPI PFNGLCOLOR3FVERTEX3FSUNPROC glewColor3fVertex3fSUN;
GLEWAPI PFNGLCOLOR3FVERTEX3FVSUNPROC glewColor3fVertex3fvSUN;
GLEWAPI PFNGLCOLOR4FNORMAL3FVERTEX3FSUNPROC glewColor4fNormal3fVertex3fSUN;
GLEWAPI PFNGLCOLOR4FNORMAL3FVERTEX3FVSUNPROC glewColor4fNormal3fVertex3fvSUN;
GLEWAPI PFNGLCOLOR4UBVERTEX2FSUNPROC glewColor4ubVertex2fSUN;
GLEWAPI PFNGLCOLOR4UBVERTEX2FVSUNPROC glewColor4ubVertex2fvSUN;
GLEWAPI PFNGLCOLOR4UBVERTEX3FSUNPROC glewColor4ubVertex3fSUN;
GLEWAPI PFNGLCOLOR4UBVERTEX3FVSUNPROC glewColor4ubVertex3fvSUN;
GLEWAPI PFNGLNORMAL3FVERTEX3FSUNPROC glewNormal3fVertex3fSUN;
GLEWAPI PFNGLNORMAL3FVERTEX3FVSUNPROC glewNormal3fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FSUNPROC glewReplacementCodeuiColor3fVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FVSUNPROC glewReplacementCodeuiColor3fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FSUNPROC glewReplacementCodeuiColor4fNormal3fVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FVSUNPROC glewReplacementCodeuiColor4fNormal3fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FSUNPROC glewReplacementCodeuiColor4ubVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FVSUNPROC glewReplacementCodeuiColor4ubVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FSUNPROC glewReplacementCodeuiNormal3fVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FVSUNPROC glewReplacementCodeuiNormal3fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC glewReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC glewReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FSUNPROC glewReplacementCodeuiTexCoord2fNormal3fVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FVSUNPROC glewReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FSUNPROC glewReplacementCodeuiTexCoord2fVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FVSUNPROC glewReplacementCodeuiTexCoord2fVertex3fvSUN;
GLEWAPI PFNGLREPLACEMENTCODEUIVERTEX3FSUNPROC glewReplacementCodeuiVertex3fSUN;
GLEWAPI PFNGLREPLACEMENTCODEUIVERTEX3FVSUNPROC glewReplacementCodeuiVertex3fvSUN;
GLEWAPI PFNGLTEXCOORD2FCOLOR3FVERTEX3FSUNPROC glewTexCoord2fColor3fVertex3fSUN;
GLEWAPI PFNGLTEXCOORD2FCOLOR3FVERTEX3FVSUNPROC glewTexCoord2fColor3fVertex3fvSUN;
GLEWAPI PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC glewTexCoord2fColor4fNormal3fVertex3fSUN;
GLEWAPI PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC glewTexCoord2fColor4fNormal3fVertex3fvSUN;
GLEWAPI PFNGLTEXCOORD2FCOLOR4UBVERTEX3FSUNPROC glewTexCoord2fColor4ubVertex3fSUN;
GLEWAPI PFNGLTEXCOORD2FCOLOR4UBVERTEX3FVSUNPROC glewTexCoord2fColor4ubVertex3fvSUN;
GLEWAPI PFNGLTEXCOORD2FNORMAL3FVERTEX3FSUNPROC glewTexCoord2fNormal3fVertex3fSUN;
GLEWAPI PFNGLTEXCOORD2FNORMAL3FVERTEX3FVSUNPROC glewTexCoord2fNormal3fVertex3fvSUN;
GLEWAPI PFNGLTEXCOORD2FVERTEX3FSUNPROC glewTexCoord2fVertex3fSUN;
GLEWAPI PFNGLTEXCOORD2FVERTEX3FVSUNPROC glewTexCoord2fVertex3fvSUN;
GLEWAPI PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FSUNPROC glewTexCoord4fColor4fNormal3fVertex4fSUN;
GLEWAPI PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FVSUNPROC glewTexCoord4fColor4fNormal3fVertex4fvSUN;
GLEWAPI PFNGLTEXCOORD4FVERTEX4FSUNPROC glewTexCoord4fVertex4fSUN;
GLEWAPI PFNGLTEXCOORD4FVERTEX4FVSUNPROC glewTexCoord4fVertex4fvSUN;

#define glColor3fVertex3fSUN glewColor3fVertex3fSUN
#define glColor3fVertex3fvSUN glewColor3fVertex3fvSUN
#define glColor4fNormal3fVertex3fSUN glewColor4fNormal3fVertex3fSUN
#define glColor4fNormal3fVertex3fvSUN glewColor4fNormal3fVertex3fvSUN
#define glColor4ubVertex2fSUN glewColor4ubVertex2fSUN
#define glColor4ubVertex2fvSUN glewColor4ubVertex2fvSUN
#define glColor4ubVertex3fSUN glewColor4ubVertex3fSUN
#define glColor4ubVertex3fvSUN glewColor4ubVertex3fvSUN
#define glNormal3fVertex3fSUN glewNormal3fVertex3fSUN
#define glNormal3fVertex3fvSUN glewNormal3fVertex3fvSUN
#define glReplacementCodeuiColor3fVertex3fSUN glewReplacementCodeuiColor3fVertex3fSUN
#define glReplacementCodeuiColor3fVertex3fvSUN glewReplacementCodeuiColor3fVertex3fvSUN
#define glReplacementCodeuiColor4fNormal3fVertex3fSUN glewReplacementCodeuiColor4fNormal3fVertex3fSUN
#define glReplacementCodeuiColor4fNormal3fVertex3fvSUN glewReplacementCodeuiColor4fNormal3fVertex3fvSUN
#define glReplacementCodeuiColor4ubVertex3fSUN glewReplacementCodeuiColor4ubVertex3fSUN
#define glReplacementCodeuiColor4ubVertex3fvSUN glewReplacementCodeuiColor4ubVertex3fvSUN
#define glReplacementCodeuiNormal3fVertex3fSUN glewReplacementCodeuiNormal3fVertex3fSUN
#define glReplacementCodeuiNormal3fVertex3fvSUN glewReplacementCodeuiNormal3fVertex3fvSUN
#define glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN glewReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN
#define glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN glewReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN
#define glReplacementCodeuiTexCoord2fNormal3fVertex3fSUN glewReplacementCodeuiTexCoord2fNormal3fVertex3fSUN
#define glReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN glewReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN
#define glReplacementCodeuiTexCoord2fVertex3fSUN glewReplacementCodeuiTexCoord2fVertex3fSUN
#define glReplacementCodeuiTexCoord2fVertex3fvSUN glewReplacementCodeuiTexCoord2fVertex3fvSUN
#define glReplacementCodeuiVertex3fSUN glewReplacementCodeuiVertex3fSUN
#define glReplacementCodeuiVertex3fvSUN glewReplacementCodeuiVertex3fvSUN
#define glTexCoord2fColor3fVertex3fSUN glewTexCoord2fColor3fVertex3fSUN
#define glTexCoord2fColor3fVertex3fvSUN glewTexCoord2fColor3fVertex3fvSUN
#define glTexCoord2fColor4fNormal3fVertex3fSUN glewTexCoord2fColor4fNormal3fVertex3fSUN
#define glTexCoord2fColor4fNormal3fVertex3fvSUN glewTexCoord2fColor4fNormal3fVertex3fvSUN
#define glTexCoord2fColor4ubVertex3fSUN glewTexCoord2fColor4ubVertex3fSUN
#define glTexCoord2fColor4ubVertex3fvSUN glewTexCoord2fColor4ubVertex3fvSUN
#define glTexCoord2fNormal3fVertex3fSUN glewTexCoord2fNormal3fVertex3fSUN
#define glTexCoord2fNormal3fVertex3fvSUN glewTexCoord2fNormal3fVertex3fvSUN
#define glTexCoord2fVertex3fSUN glewTexCoord2fVertex3fSUN
#define glTexCoord2fVertex3fvSUN glewTexCoord2fVertex3fvSUN
#define glTexCoord4fColor4fNormal3fVertex4fSUN glewTexCoord4fColor4fNormal3fVertex4fSUN
#define glTexCoord4fColor4fNormal3fVertex4fvSUN glewTexCoord4fColor4fNormal3fVertex4fvSUN
#define glTexCoord4fVertex4fSUN glewTexCoord4fVertex4fSUN
#define glTexCoord4fVertex4fvSUN glewTexCoord4fVertex4fvSUN

GLEWAPI GLboolean GLEW_SUN_vertex;

#endif /* GL_SUN_vertex */

/* -------------------------- GL_WIN_phong_shading ------------------------- */

#ifndef GL_WIN_phong_shading
#define GL_WIN_phong_shading 1

#define GL_PHONG_WIN 0x80EA
#define GL_PHONG_HINT_WIN 0x80EB

GLEWAPI GLboolean GLEW_WIN_phong_shading;

#endif /* GL_WIN_phong_shading */

/* -------------------------- GL_WIN_specular_fog -------------------------- */

#ifndef GL_WIN_specular_fog
#define GL_WIN_specular_fog 1

#define GL_FOG_SPECULAR_TEXTURE_WIN 0x80EC

GLEWAPI GLboolean GLEW_WIN_specular_fog;

#endif /* GL_WIN_specular_fog */

/* ---------------------------- GL_WIN_swap_hint --------------------------- */

#ifndef GL_WIN_swap_hint
#define GL_WIN_swap_hint 1

typedef void (GLAPIENTRY * PFNGLADDSWAPHINTRECTWINPROC) (GLint x, GLint y, GLsizei width, GLsizei height);

GLEWAPI PFNGLADDSWAPHINTRECTWINPROC glewAddSwapHintRectWIN;

#define glAddSwapHintRectWIN glewAddSwapHintRectWIN

GLEWAPI GLboolean GLEW_WIN_swap_hint;

#endif /* GL_WIN_swap_hint */

/* ------------------------------------------------------------------------- */

/* error codes */
#define GLEW_OK 0
#define GLEW_NO_ERROR 0
#define GLEW_ERROR_NO_GL_VERSION 1  /* missing GL version */
#define GLEW_ERROR_GL_VERSION_10_ONLY 2  /* GL 1.1 and up are not supported */
#define GLEW_ERROR_GLX_VERSION_11_ONLY 3  /* GLX 1.2 and up are not supported */

/* string codes */
#define GLEW_VERSION 1

/* API */
GLEWAPI GLboolean glewExperimental;
GLEWAPI GLenum glewInit ();
GLEWAPI GLboolean glewGetExtension (const GLubyte* name);
GLEWAPI const GLubyte* glewGetErrorString (GLenum error);
GLEWAPI const GLubyte* glewGetString (GLenum name);

#ifdef __cplusplus
}
#endif

#ifdef GLEW_APIENTRY_DEFINED
#undef GLEW_APIENTRY_DEFINED
#undef APIENTRY
#undef GLAPIENTRY
#endif

#ifdef GLEW_CALLBACK_DEFINED
#undef GLEW_CALLBACK_DEFINED
#undef CALLBACK
#endif

#ifdef GLEW_WINGDIAPI_DEFINED
#undef GLEW_WINGDIAPI_DEFINED
#undef WINGDIAPI
#endif

#undef GLAPI
/* #undef GLEWAPI */

#endif /* __glew_h__ */
