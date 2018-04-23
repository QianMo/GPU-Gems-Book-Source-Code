/* tess.cpp - GPU tessellation code

Copyright (C) 2005 NVIDIA Corporation

This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.
*/

#include "subdinternal.h"
#include "subd.h"
#include <malloc.h>
#include <math.h>
#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <tchar.h>
#include <GL/gl.h>
#include "../glext.h"
#include "../wglext.h"

extern bool perfTestFlag;

// 2 patch textures are used for subdivision with each buffer split into
// 4 segments. This gives us an 8 segment patch buffer stack to hold intermediate
// tessellation results

// patch texture 0, segment 0 contains patches subdivided 0 times
// patch texture 1, segment 0 contains patches subdivided 1 time
// patch texture 0, segment 1 contains patches subdivided 2 times
// patch texture 1, segment 1 -- 3 times
// patch texture 0, segment 2 -- 4 times
// etc.

#define SEGMENT_SIZE            68
#define PATCH_BUFFER_WIDTH      512
#define PATCH_BUFFER_HEIGHT     (SEGMENT_SIZE*4)

// These values are used to bind the corresponding textures
// offset by 10 to avoid conflicts with the application (we should allocate these instead)
#define PATCH_TEXTURE0          10
#define PATCH_TEXTURE1          11
#define VERTEX_TEXTURE          12
#define OFFSET_TEXTURE          13
#define INDEX_TEXTURE           14
#define FLAT_TEXTURE            15
#define EP_TEXTURE              16
#define TCOORD_TEXTURE          17
#define EPTAN_IN_TEXTURE        18
#define EPTAN_OUT_TEXTURE       19
#define DMAP_TEXTURE            20
#define DMAPMAX_TEXTURE         21

// Shader IDs
#define CREATEPATCH_SHADER      1
#define LIMIT_SHADER            2
#define NORMAL_SHADER           3
#define SUBDIV_SHADER           4
#define EPLIMIT_SHADER          5
#define EPNORMAL_SHADER         6
#define EPSUBDIVE_SHADER        7
#define EPSUBDIVF_SHADER        8
#define FLATTEST_SHADER         9
#define FLATTEST2_SHADER        10
#define DFLATTEST_SHADER        11
#define DFLATTEST2_SHADER       12
#define EPFLAT_SHADER           13
#define PACKFLAT_SHADER         14
#define TANGENT_SHADER          15
#define EPTANGENT_SHADER        16
#define DLIMIT_SHADER           17

extern int init_shader(char *, int id, int shaderType);
extern void set_shader(int id, int shaderType);
extern void set_shader_parameter(int id, char *name, float *value);

float4 TestBuffer[PATCH_BUFFER_WIDTH*SEGMENT_SIZE];
uchar *FlatBuffer[MAXDEPTH+1];

static PFNGLACTIVETEXTUREARBPROC glActiveTexture;
static PFNWGLDESTROYPBUFFERARBPROC wglDestroyPbufferARB;
static PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
static PFNWGLCREATEPBUFFERARBPROC wglCreatePbufferARB;
static PFNWGLGETPBUFFERDCARBPROC wglGetPbufferDCARB;
static PFNWGLQUERYPBUFFERARBPROC wglQueryPbufferARB;
static PFNWGLRELEASETEXIMAGEARBPROC wglReleaseTexImageARB;
static PFNWGLBINDTEXIMAGEARBPROC wglBindTexImageARB;
static PFNGLMULTITEXCOORD4IARBPROC glMultiTexCoord4i;
static PFNGLMULTITEXCOORD4FARBPROC glMultiTexCoord4f;
static PFNGLMULTITEXCOORD4FVARBPROC glMultiTexCoord4fv;

extern PFNGLBINDBUFFERARBPROC glBindBufferARB;

#define GET_PROC_ADDRESS wglGetProcAddress

#define QUERY_EXTENSION_ENTRY_POINT(name, type)               \
    name = (type)GET_PROC_ADDRESS(#name);

typedef struct GPUBuffer {
    int width, height;
    int type;
    float4 *data;
    HPBUFFERARB  hpbuffer;      // Handle to a pbuffer.
    HDC          hdc;           // Handle to a device context.
    HGLRC        hglrc;         // Handle to a GL rendering context.
    struct GPUBuffer *next;
} GPUBuffer;


void setDstGPUBuffer(GPUBuffer *gp);

static GPUBuffer *RenderBuffer, *lastDst;
static int CurrentDepth;
static float4 offset[(PATCH_BUFFER_WIDTH*SEGMENT_SIZE)/4];
static float4 dmapOffset[(PATCH_BUFFER_WIDTH*SEGMENT_SIZE)/4];
static int flattest_shader = FLATTEST_SHADER, flattest2_shader = FLATTEST2_SHADER;

static void wglGetLastError()
{
}

static void
setDstGPUBuffer(GPUBuffer *gp)
{
    if (lastDst != gp) {
        if (gp)
            wglMakeCurrent(gp->hdc, gp->hglrc);
        lastDst = gp;
    }
}


#define MAX_ATTRIBS     256
#define MAX_PFORMATS    256

// create render buffer (pbuffer)
GPUBuffer *
createGPUBuffer(int width, int height, int bitsPerComponent, int numComponents)
{
    GPUBuffer *gp;
    int     format;
    int     pformat[MAX_PFORMATS];
    unsigned int nformats;
    int     iattributes[2*MAX_ATTRIBS];
    float   fattributes[2*MAX_ATTRIBS];
    int     nfattribs = 0;
    int     niattribs = 0;
    HDC hdc = wglGetCurrentDC();
	HGLRC hglrc = wglGetCurrentContext();

    gp = (GPUBuffer *) calloc(sizeof(GPUBuffer), 1);
    gp->width = width;
    gp->height = height;

    wglGetLastError();

    // Attribute arrays must be "0" terminated - for simplicity, first
    // just zero-out the array entire, then fill from left to right.
    memset(iattributes, 0, sizeof(int)*2*MAX_ATTRIBS);
    memset(fattributes, 0, sizeof(float)*2*MAX_ATTRIBS);
    // Since we are trying to create a pbuffer, the pixel format we
    // request (and subsequently use) must be "p-buffer capable".
    iattributes[niattribs  ] = WGL_DRAW_TO_PBUFFER_ARB;
    iattributes[++niattribs] = GL_TRUE;
    // we are asking for a pbuffer that is meant to be bound
    // as an RGBA texture - therefore we need a color plane
    if (bitsPerComponent == 32 || bitsPerComponent == 16) {
        gp->type = GL_FLOAT;
        iattributes[++niattribs] = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV;
        iattributes[++niattribs] = GL_TRUE;
        iattributes[++niattribs] = WGL_FLOAT_COMPONENTS_NV;
        iattributes[++niattribs] = GL_TRUE;
    }
    else {
        gp->type = GL_UNSIGNED_BYTE;
        bitsPerComponent = 8;
        iattributes[++niattribs] = WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV;
        iattributes[++niattribs] = GL_TRUE;
    }
    iattributes[++niattribs] = WGL_RED_BITS_ARB;
    iattributes[++niattribs] = bitsPerComponent;
    iattributes[++niattribs] = WGL_GREEN_BITS_ARB;
    iattributes[++niattribs] = bitsPerComponent;
    iattributes[++niattribs] = WGL_BLUE_BITS_ARB;
    iattributes[++niattribs] = bitsPerComponent;
    iattributes[++niattribs] = WGL_ALPHA_BITS_ARB;
    iattributes[++niattribs] = bitsPerComponent;

    if (!wglChoosePixelFormatARB) {
        QUERY_EXTENSION_ENTRY_POINT(wglChoosePixelFormatARB,
            PFNWGLCHOOSEPIXELFORMATARBPROC);
        QUERY_EXTENSION_ENTRY_POINT(wglCreatePbufferARB,
            PFNWGLCREATEPBUFFERARBPROC);
        QUERY_EXTENSION_ENTRY_POINT(wglGetPbufferDCARB,
            PFNWGLGETPBUFFERDCARBPROC);
        QUERY_EXTENSION_ENTRY_POINT(wglQueryPbufferARB,
            PFNWGLQUERYPBUFFERARBPROC);

        QUERY_EXTENSION_ENTRY_POINT(wglReleaseTexImageARB,
            PFNWGLRELEASETEXIMAGEARBPROC);
        QUERY_EXTENSION_ENTRY_POINT(wglBindTexImageARB,
            PFNWGLBINDTEXIMAGEARBPROC);

        QUERY_EXTENSION_ENTRY_POINT(glActiveTexture,
            PFNGLACTIVETEXTUREARBPROC);
    }

    if ( !wglChoosePixelFormatARB( hdc, iattributes, fattributes, MAX_PFORMATS, pformat, &nformats ) )
    {
        printf("pbuffer creation error:  wglChoosePixelFormatARB() failed.\n");
    }
    wglGetLastError();
	if ( nformats <= 0 )
    {
        printf("pbuffer creation error:  Couldn't find a suitable pixel format.\n");
    }
    format = pformat[0];

    // Set up the pbuffer attributes
    memset(iattributes,0,sizeof(int)*2*MAX_ATTRIBS);
    niattribs = 0;
    // the render texture format is RGBA
    iattributes[niattribs] = WGL_TEXTURE_FORMAT_ARB;
    if (gp->type == GL_FLOAT) {
        iattributes[++niattribs] = WGL_TEXTURE_FLOAT_RGBA_NV;
    }
    else
        iattributes[++niattribs] = WGL_TEXTURE_RGBA_ARB;

    // the render texture target is GL_TEXTURE_RECTANGLE_NV
    iattributes[++niattribs] = WGL_TEXTURE_TARGET_ARB;
    iattributes[++niattribs] = WGL_TEXTURE_RECTANGLE_NV;
    // ask to allocate the largest pbuffer it can, if it is
    // unable to allocate for the width and height
    iattributes[++niattribs] = WGL_PBUFFER_LARGEST_ARB;
    iattributes[++niattribs] = TRUE;
    // Create the p-buffer.
    gp->hpbuffer = (HPBUFFERARB) wglCreatePbufferARB( hdc, format, width, height, iattributes );
    if ( gp->hpbuffer == 0)
    {
        printf("pbuffer creation error:  wglCreatePbufferARB() failed\n");
        return 0;
    }
    wglGetLastError();

    // Get the device context.
    gp->hdc = wglGetPbufferDCARB( gp->hpbuffer );
    if ( gp->hdc == 0)
    {
        printf("pbuffer creation error:  wglGetPbufferDCARB() failed\n");
        wglGetLastError();
    }
    wglGetLastError();

    // Create a gl context for the p-buffer.
    gp->hglrc = wglCreateContext(gp->hdc);
    if ( gp->hglrc == 0)
    {
         printf("pbuffer creation error:  wglCreateContext() failed\n");
        wglGetLastError();
    }
    wglGetLastError();
    wglShareLists(hglrc, gp->hglrc);

    // Determine the actual width and height we were able to create.
    wglQueryPbufferARB( gp->hpbuffer, WGL_PBUFFER_WIDTH_ARB, &gp->width );
    wglQueryPbufferARB( gp->hpbuffer, WGL_PBUFFER_HEIGHT_ARB, &gp->height );

printf("pbuffer created: Width %d height %d\n", gp->width, gp->height);
     wglMakeCurrent(gp->hdc, gp->hglrc);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
     glOrtho(0.0, (double) gp->width, 0.0, (double) gp->height, -1.0, 1.0);

     wglMakeCurrent(hdc, hglrc);

    return gp;
}

void
writeToGPUBuffer(GPUBuffer *gp, float4 *src, int x, int y, int w, int h)
{
    setDstGPUBuffer(gp);
    glRasterPos2i(x, y);
    glDrawPixels(w, h, GL_RGBA, gp->type, (GLvoid *) src);
}

void
readFromGPUBuffer(GPUBuffer *gp, float4 *dst, int x, int y, int w, int h, int bufObj)
{
    setDstGPUBuffer(gp);
    if (bufObj) {   // reading directly into vertex array in vidmem
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, bufObj);
        glReadPixels(x, y, w, h, GL_RGBA, gp->type, (GLvoid *) dst);
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, 0);
        return;
    }
    glReadPixels(x, y, w, h, GL_RGBA, gp->type, (GLvoid *) dst);
}

static void
loadShaders()
{
    init_shader("CreatePatch", CREATEPATCH_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("Limit", LIMIT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("DLimit", DLIMIT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("Normal", NORMAL_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("Subdiv", SUBDIV_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("EPLimit", EPLIMIT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("EPNormal", EPNORMAL_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("EPSubdivE", EPSUBDIVE_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("EPSubdivF", EPSUBDIVF_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("FlatTest", FLATTEST_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("FlatTest2", FLATTEST2_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("EPFlat", EPFLAT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("PackFlat", PACKFLAT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("Tangent", TANGENT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("EPTangent", EPTANGENT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("DFlatTest", DFLATTEST_SHADER, GL_FRAGMENT_PROGRAM_NV);
    init_shader("DFlatTest2", DFLATTEST2_SHADER, GL_FRAGMENT_PROGRAM_NV);
}


// This routine writes the coordinates for the upper-left corner of each
// patch into a texture for the given subdivision depth
// since patches at the same subdivision depth are the same size this
// data is constant

static void
setPatchOffset(int depth)
{
    int size, x, y;
    int rows, cols;
    int i;

    size = (1<<depth)+1;
    x = 0;
    y = 0;
    cols = PATCH_BUFFER_WIDTH/size;
    rows = SEGMENT_SIZE/size;
    i = 0;
    for (y = 0; y < rows; y++) {
        for (x = 0; x < cols; x++) {
            offset[i].x = (float) (x*size) + 0.5f;
            offset[i].y = (float) (y*size) + 0.5f;
            i++;
        }
    }
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, (depth+2)*SEGMENT_SIZE/2, cols, rows,
        GL_RGBA, GL_FLOAT, (void*) offset);
}

// Create the pbuffer for rendering, load shaders, and create the textures
// that will be rendered to. (We are not using render-to-texture yet so the
// data is rendered and then copied to the textures using glTexSubImage)

static void init()
{
    int i;

    QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4i, PFNGLMULTITEXCOORD4IARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4f, PFNGLMULTITEXCOORD4FARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4fv, PFNGLMULTITEXCOORD4FVARBPROC);

    RenderBuffer = createGPUBuffer(PATCH_BUFFER_WIDTH, SEGMENT_SIZE*2, 32, 4);
    setDstGPUBuffer(RenderBuffer);

    glEnable(GL_FRAGMENT_PROGRAM_NV);
    loadShaders();

    FlatBuffer[0] = (uchar *) 
            malloc(PATCH_BUFFER_WIDTH*SEGMENT_SIZE*(MAXDEPTH+1)*sizeof(uchar));
    for (i = 0; i < MAXDEPTH; i++)
        FlatBuffer[i+1] = FlatBuffer[i] + PATCH_BUFFER_WIDTH * SEGMENT_SIZE;

    for (i = 0; i < 2; i++) {
        glActiveTexture(GL_TEXTURE0);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, PATCH_TEXTURE0+i);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
            PATCH_BUFFER_WIDTH, PATCH_BUFFER_HEIGHT, 0, GL_RGBA,
            GL_FLOAT, (GLvoid *) 0);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        PATCH_BUFFER_WIDTH, 2+EP_HEIGHT*(MAXDEPTH+1), 0, GL_RGBA,
        GL_FLOAT, (GLvoid *) 0);
    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, FLAT_TEXTURE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        PATCH_BUFFER_WIDTH, SEGMENT_SIZE, 0, GL_RGBA,
        GL_FLOAT, (GLvoid *) 0);
    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, VERTEX_TEXTURE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        PATCH_BUFFER_WIDTH, SEGMENT_SIZE, 0, GL_RGBA,
        GL_FLOAT, (GLvoid *) 0);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, OFFSET_TEXTURE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        PATCH_BUFFER_WIDTH, (MAXDEPTH+3)*SEGMENT_SIZE/2, 0, GL_RGBA,
        GL_FLOAT, (GLvoid *) 0);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    for (i = 0; i <= MAXDEPTH; i++) 
        setPatchOffset(i);
    
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EPTAN_OUT_TEXTURE);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        PATCH_BUFFER_WIDTH, 4, 0, GL_RGBA,
        GL_FLOAT, (GLvoid *) 0);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

// Load data into textures that define the mesh, texture coordinates, and
// displacement map

void LoadMeshTextures(unsigned char *patchIndexBuffer,
    int patchIndexW, int patchIndexH,
    float4 *texCoordBuffer, int texCoordW, int texCoordH,
    float4 *epTanInBuffer, int epTanInH,
    uchar *dmapTexture, int dmapW, int dmapH,
    uchar *dmapMaxTexture, int dmapMaxW, int dmapMaxH)
{
    if (!RenderBuffer)
        init();
    if (patchIndexBuffer) {
        glActiveTexture(GL_TEXTURE1);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, INDEX_TEXTURE);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA,
            PATCH_BUFFER_WIDTH, patchIndexH, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, (GLvoid *) patchIndexBuffer);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);
    }
    if (texCoordBuffer) {
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, TCOORD_TEXTURE);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
            texCoordW, texCoordH+SEGMENT_SIZE, 0, GL_RGBA,
            GL_FLOAT, (GLvoid *) 0);
        glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, texCoordW, texCoordH,
            GL_RGBA, GL_FLOAT, texCoordBuffer);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);
    }
    if (epTanInBuffer) {
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, EPTAN_IN_TEXTURE);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
            PATCH_BUFFER_WIDTH, epTanInH, 0, GL_RGBA,
            GL_FLOAT, (GLvoid *) epTanInBuffer);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    }
    if (dmapTexture) {
        glActiveTexture(GL_TEXTURE2);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, DMAP_TEXTURE);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_LUMINANCE,
            dmapW, dmapH, 0, GL_LUMINANCE,
            GL_BYTE, (GLvoid *) dmapTexture);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    }
    if (dmapMaxTexture) {
        glActiveTexture(GL_TEXTURE3);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, DMAPMAX_TEXTURE);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_LUMINANCE_ALPHA,
            dmapMaxW, dmapMaxH, 0, GL_LUMINANCE_ALPHA,
            GL_UNSIGNED_BYTE, (GLvoid *) dmapMaxTexture);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    }
}

// Load the vertex data into a texture so that it can be used by createPatches
// to write the patch data into patch texture 0

static void loadVertices(float4 *vlist, int vertices)
{
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, VERTEX_TEXTURE);

    if (vertices >> 8)
        glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 256, vertices>>8,
            GL_RGBA, GL_FLOAT, vlist);
    if (vertices & 255)
        glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, vertices>>8, vertices & 255, 1,
            GL_RGBA, GL_FLOAT, vlist+256*(vertices>>8));
}

// Write out the index data portion of the tessellation result

#define emit_quad(i1, i2, i3, i4) \
            prim_ptr[0] = i1; \
            prim_ptr[1] = i2; \
            prim_ptr[2] = i3; \
            prim_ptr[3] = i4; \
            prim_ptr += 4;

//#define emit_quad(i1, i2, i3, i4) _subd_emit_quad(i1, i2, i3, i4)

// Use the starting vertexIndex, depth and the flatness data computed by the flatTest shader
// to create the index portion of the tessellation result
//
// vertexIndex2 is used in the case a patch needs to be split only vertically or horizontally
// so that vertexIndex points the the patch that corresponds to one edge and
// vertexIndex2 to the other.
static void
tessPatch(Patch *p, int vertexIndex, int vertexIndex2, int depth, int flatmask,
        int n, int w, int flatX, int flatY, int flatX2, int flatY2,
        int v1, int v2, int v3, int v4)
{
    int nv1, nv2, nv3, nv4, nv5;
    int flat;
    extern int *prim_ptr;

    flat = flatmask;
    if (n == 1)
        flat = TOP|BOTTOM|LEFT|RIGHT;
    else if (_subd_adaptive_flag) {
        if (vertexIndex == vertexIndex2)
            flat |=  p->flatPtr[depth][flatY*w + flatX];
        else {
            flat |= p->flatPtr[depth][flatY*w + flatX] & (TOP|LEFT);
            flat |= p->flatPtr[depth][flatY2*w + flatX2] & (BOTTOM|RIGHT);
        }
    }

    if (flat == (TOP|BOTTOM|LEFT|RIGHT)) {
        emit_quad(v1, v2, v4, v3);
        return;
    }

    depth++;
    n >>= 1;
    flatX <<= 1;
    flatY <<= 1;
    nv1 = vertexIndex + n;
    nv2 = vertexIndex + w*n;
    nv3 = nv2 + n;
    nv4 = vertexIndex2 + w*n + n + n;
    nv5 = vertexIndex2 + 2*w*n + n;

    if ((flat & (LEFT|RIGHT)) == (LEFT|RIGHT)) {
        if (flat & TOP) {
            nv1 += w;
            emit_quad(v1, v2, nv1, nv1);
        }
        if (flat & BOTTOM) {
            nv5 -= w;
            emit_quad(v4, v3, nv5, nv5);
        }
        tessPatch(p, vertexIndex, vertexIndex2+w*n, depth, flat, n, w,
                flatX, flatY, flatX, flatY+1, v1, nv1, v3, nv5);
        tessPatch(p, vertexIndex+n, vertexIndex2+w*n+n, depth, flat, n, w,
                flatX+1, flatY, flatX+1, flatY+1, nv1, v2, nv5, v4);
    }
    else if ((flat & (BOTTOM|TOP)) == (BOTTOM|TOP)) {
        if (flat & RIGHT) {
            nv4--;
            emit_quad(v2, v4, nv4, nv4);
        }
        if (flat & LEFT) {
            nv2++;
            emit_quad(v3, v1, nv2, nv2);
        }
        tessPatch(p, vertexIndex, vertexIndex2+n, depth, flat, n, w,
                flatX, flatY, flatX+1, flatY, v1, v2, nv2, nv4);
        tessPatch(p, vertexIndex+w*n, vertexIndex2+w*n+n, depth, flat, n, w,
                flatX, flatY+1, flatX+1, flatY+1, nv2, nv4, v3, v4);
    }
    else {
        if (flat) {
            if (flat & TOP) {
                nv1 += w;
                emit_quad(v1, v2, nv1, nv1);
            }
            if (flat & RIGHT) {
                nv4--;
                emit_quad(v2, v4, nv4, nv4);
            }
            if (flat & BOTTOM) {
                nv5 -= w;
                emit_quad(v4, v3, nv5, nv5);
            }
            if (flat & LEFT) {
                nv2++;
                emit_quad(v3, v1, nv2, nv2);
            }
        }

        tessPatch(p, vertexIndex, vertexIndex, depth,
            flat & (TOP|LEFT), n, w, flatX, flatY, 0, 0, v1, nv1, nv2, nv3);
        tessPatch(p, vertexIndex+n, vertexIndex+n, depth,
            flat & (TOP|RIGHT), n, w, flatX+1, flatY, 0, 0, nv1, v2, nv3, nv4);
        tessPatch(p, vertexIndex+w*n, vertexIndex+w*n, depth,
            flat&(BOTTOM|LEFT), n, w, flatX, flatY+1, 0, 0, nv2, nv3, v3, nv5);
        tessPatch(p, vertexIndex+w*n+n, vertexIndex+w*n+n, depth,
            flat&(BOTTOM|RIGHT),n,w, flatX+1, flatY+1, 0, 0, nv3, nv4, nv5, v4);
    }
}

static void
tessPatches(Patch *p, int depth, int yOffset)
{
    int vertexIndex;
    int stride = PATCH_BUFFER_WIDTH;
    int n;

    n = (1<<depth);
    for (; p; p = p->next) {
        vertexIndex = (p->loc.x*(n+1) + (p->loc.y*(n+1)+yOffset)*stride);
        tessPatch(p, vertexIndex, vertexIndex, 0, 0, n, stride, 0, 0, 0, 0,
            vertexIndex, vertexIndex + n,
            vertexIndex + n*stride, vertexIndex+n*stride+n);
    }
}

/*
    calcEPLimitAndNormal() - calculate the limit position and normal for
    each extra-ordinary point using the EPLimit shader
    The input data was loaded using the CreatePatch shader and is in the
    format:

    v  v  v  v  v  v  v  v ...
    e1 e1 e1 e1 e1 e1 e1 e1 ...
    e2 e2 e2 e2 e2 e2 e2 e2 ...
    .
    .
    .
    e8 e8 e8 e8 e8 e8 e8 e8 ...
    f1 f1 f1 f1 f1 f1 f1 f1 ...
    f2 f2 f2 f2 f2 f2 f2 f2 ...
    .
    .
    .
    f8 f8 f8 f8 f8 f8 f8 f8 ...

    where v is the exraordinary point, edge vertices e1 - e8 are vertices that share an edge
    with v and, face vertices f1-f8 are the points that share a face, but not an edge with v.
    Actually, only n edge and face vertices are really used where n is the valence of v
*/

static void
calcEPLimitAndNormal(GroupInfo *groupInfo)
{
    int w = groupInfo->epTotal;
    int n;
    int v;
    int x;
    float *tp;
    static float weights[MAXVALENCE-2][MAXVALENCE*2+1] = {
        // 3
        9.0f/24.0f,
        4.0f/24.0f, 4.0f/24.0f, 4.0f/24.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f/24.0f, 1.0f/24.0f, 1.0f/24.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // 4
        0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // 5
        25.0f/50.0f,
        4.0f/50.0f, 4.0f/50.0f, 4.0f/50.0f, 4.0f/50.0f, 4.0f/50.0f, 0.0f, 0.0f, 0.0f,
        1.0f/50.0f, 1.0f/50.0f, 1.0f/50.0f, 1.0f/50.0f, 1.0f/50.0f, 0.0f, 0.0f, 0.0f,
        // 6
        36.0f/66.0f,
        4.0f/66.0f, 4.0f/66.0f, 4.0f/66.0f, 4.0f/66.0f, 4.0f/66.0f, 4.0f/66.0f, 0.0f, 0.0f,
        1.0f/66.0f, 1.0f/66.0f, 1.0f/66.0f, 1.0f/66.0f, 1.0f/66.0f, 1.0f/66.0f, 0.0f, 0.0f,
        // 7
        49.0f/84.0f,
        4.0f/84.0f, 4.0f/84.0f, 4.0f/84.0f, 4.0f/84.0f, 4.0f/84.0f, 4.0f/84.0f, 4.0f/84.0f, 0.0f,
        1.0f/84.0f, 1.0f/84.0f, 1.0f/84.0f, 1.0f/84.0f, 1.0f/84.0f, 1.0f/84.0f, 1.0f/84.0f, 0.0f,
        // 8
        64.0f/104.0f,
        4.0f/104.0f, 4.0f/104.0f, 4.0f/104.0f, 4.0f/104.0f, 4.0f/104.0f, 4.0f/104.0f, 4.0f/104.0f, 4.0f/104.0f,
        1.0f/104.0f, 1.0f/104.0f, 1.0f/104.0f, 1.0f/104.0f, 1.0f/104.0f, 1.0f/104.0f, 1.0f/104.0f, 1.0f/104.0f,
    };
    extern float tangent_mask3[], tangent_mask5[], tangent_mask6[], tangent_mask7[],
        tangent_mask8[];
    static float *tangent_masks[MAXVALENCE-2] = {
        tangent_mask3, NULL, tangent_mask5, tangent_mask6, tangent_mask7, tangent_mask8 };


    if (w == 0)
        return;

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);
    set_shader(EPLIMIT_SHADER, GL_FRAGMENT_PROGRAM_NV);

    x = 0;
    glBegin(GL_QUADS);
    for (v = 0; v <= MAXVALENCE-3; v++) {
        n = groupInfo->epCount[v];
        if (!n)
            continue;

        glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.0f, 2.0f, 0.0f, weights[v][0]);
        glMultiTexCoord4fv(GL_TEXTURE1_ARB, &weights[v][1]);
        glMultiTexCoord4fv(GL_TEXTURE2_ARB, &weights[v][5]);
        glMultiTexCoord4fv(GL_TEXTURE3_ARB, &weights[v][9]);
        glMultiTexCoord4fv(GL_TEXTURE4_ARB, &weights[v][13]);
        glVertex2i(x, 0);
        glVertex2i(x+n, 0);
        glVertex2i(x+n, 1);
        glVertex2i(x, 1);
        x += n;
    }
    glEnd();

    set_shader(EPNORMAL_SHADER, GL_FRAGMENT_PROGRAM_NV);

    x = 0;
    glBegin(GL_QUADS);
    for (v = 0; v <= MAXVALENCE-3; v++) {
        n = groupInfo->epCount[v];
        if (!n)
            continue;

        tp = tangent_masks[v];
        glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, 0, v+3, v+3+8);
        glMultiTexCoord4fv(GL_TEXTURE1_ARB, tp);
        glMultiTexCoord4fv(GL_TEXTURE2_ARB, tp+4);
        glMultiTexCoord4fv(GL_TEXTURE3_ARB, tp+8);
        glMultiTexCoord4fv(GL_TEXTURE4_ARB, tp+12);
        glVertex2i(x, 1);
        glVertex2i(x+n, 1);
        glVertex2i(x+n, 2);
        glVertex2i(x, 2);
        x += n;
    }
    glEnd();

    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, w, 2);
}

static Patch *
createPatches(Patch *patchList, GroupInfo *groupInfo)
{
    Patch *p, *lastp;
    int x, y;
    int w, h;
    int epW = groupInfo->epTotal;

    x = 0; y = 0;
    lastp = NULL;
    for (p = patchList; p && p->group == patchList->group; p = p->next) {
        if (x == PATCH_BUFFER_WIDTH/4) {
            if (y + 2 > SEGMENT_SIZE/4)
                break;  // cannot fit any more
            x = 0;
            y++;
        }

        p->loc.x = x;
        p->loc.y = y;
        x++;
        lastp = p;
    }
    lastp->next = NULL;
    w = PATCH_BUFFER_WIDTH;
    h = (y + 1)*4;

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, INDEX_TEXTURE);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, VERTEX_TEXTURE);
    set_shader(CREATEPATCH_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, patchList->indexLoc.y, 0, 0);
    glVertex2i(0, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, PATCH_BUFFER_WIDTH, patchList->indexLoc.y, 0, 0);
    glVertex2i(w, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, PATCH_BUFFER_WIDTH, patchList->indexLoc.y + h, 0, 0);
    glVertex2i(w, h);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, patchList->indexLoc.y+h, 0, 0);
    glVertex2i(0, h);

    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, patchList->indexLoc.y+h, 0, 0);
    glVertex2i(0, h);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, epW, patchList->indexLoc.y+h, 0, 0);
    glVertex2i(epW, h);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, epW, patchList->indexLoc.y+h+EP_HEIGHT, 0, 0);
    glVertex2i(epW, h+EP_HEIGHT);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, patchList->indexLoc.y+h+EP_HEIGHT, 0, 0);
    glVertex2i(0, h+EP_HEIGHT);

    glEnd();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, PATCH_TEXTURE0);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, w, h);

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 2, 0, h, epW, EP_HEIGHT);

    // calc limit and normal for extraordinary points

    calcEPLimitAndNormal(groupInfo);

    return p;
}

static void
calcTangents(Patch *patchList, GroupInfo *groupInfo)
{
    int x, y;
    int size, srcSize;
    int i;
    int maxCol;
    int rows, cols;
    Patch *p;
    float y1, y2, y3;

    // calculate tangents for normal mapping at extra-ordinary points

    if (groupInfo->epTangentH > 0) {
        rows = groupInfo->epTangentH;
        cols = groupInfo->epTangentW;
        set_shader(EPTANGENT_SHADER, GL_FRAGMENT_PROGRAM_NV);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, EPTAN_IN_TEXTURE);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);
        glBegin(GL_QUADS);
        for (i = 0; i < rows; i++) {
            y1 = i*3 + 0.5f;
            y2 = y1 + 1.0f;
            y3 = y1 + 2.0f;
            glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.0f, y1, y2, y3);
            glVertex2i(0, i);
            glMultiTexCoord4f(GL_TEXTURE0_ARB, (float) cols,  y1, y2, y3);
            glVertex2i(cols, i);
            glMultiTexCoord4f(GL_TEXTURE0_ARB, (float) cols,  y1, y2, y3);
            glVertex2i(cols, i+1);
            glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.0f,  y1, y2, y3);
            glVertex2i(0, i+1);
        }
        glEnd();

        glBindTexture(GL_TEXTURE_RECTANGLE_NV, EPTAN_OUT_TEXTURE);
        glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 1, 0, 0, cols, rows);
    }

    size = 2;
    srcSize = 4;
    x = 0;
    y = 0;
    i = 0;
    maxCol = 0;
    cols = PATCH_BUFFER_WIDTH/size;
    for (p = patchList; p; p = p->next) {
        if (x == cols) {
            maxCol = x;
            x = 0;
            y++;
        }

        offset[i].x = (float) (p->loc.x*srcSize - x*size + 1);
        offset[i].y = (float) (p->loc.y*srcSize - y*size + 1);

        x++;
        i++;
    }

    if (x > maxCol)
        maxCol = x;
    cols = maxCol;
    rows = y + 1;

    x = cols*size;
    y = rows*size;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, PATCH_TEXTURE0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, OFFSET_TEXTURE);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, cols, rows,
        GL_RGBA, GL_FLOAT, (void*) offset);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, TCOORD_TEXTURE);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EPTAN_OUT_TEXTURE);

    set_shader(TANGENT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, 0, 0, 0);
    glVertex2i(0, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, 0, cols, 0);
    glVertex2i(x, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, rows, cols, rows);
    glVertex2i(x, y);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, rows, 0, rows);
    glVertex2i(0, y);
    glEnd();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, TCOORD_TEXTURE);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 256, 0, 0, x, y);
}

static int
calcLimit(Patch *patchList, int depth, int shader)
{
    int x, y;
    int size, srcSize;
    int i;
    int maxCol;
    int rows, cols;
    Patch *p;
    int yOffset;
    int patchOffsetY;

    size = (1<<depth)+1;
    srcSize = (1<<depth)+3;
    x = 0;
    y = 0;
    i = 0;
    maxCol = 0;
    cols = PATCH_BUFFER_WIDTH/size;
    yOffset = SEGMENT_SIZE*(depth>>1) + 1;
    for (p = patchList; p; p = p->next) {
        if (x == cols) {
            maxCol = x;
            x = 0;
            y++;
        }

        offset[i].x = (float) (p->loc.x*srcSize - x*size + 1);
        offset[i].y = (float) (p->loc.y*srcSize - y*size + yOffset);
        offset[i].z = (float) p->texCoordLoc.x;
        offset[i].w = (float) p->texCoordLoc.y;

        if (depth <= p->dmapDepth) {
            dmapOffset[i].x = (float) p->dmapLoc[depth].x + 0.5f;
            dmapOffset[i].y = (float) p->dmapLoc[depth].y + 0.5f;
            dmapOffset[i].z = 1.0f;
        }
        else {
            dmapOffset[i].x = (float) p->dmapLoc[p->dmapDepth].x + 0.5f;
            dmapOffset[i].y = (float) p->dmapLoc[p->dmapDepth].y + 0.5f;
            dmapOffset[i].z =(1.0f/(1<<MAXDEPTH))*(1<<(MAXDEPTH-(depth - p->dmapDepth)));
        }
        p->loc.x = x;
        p->loc.y = y;

        x++;
        i++;
    }

    if (x > maxCol)
        maxCol = x;
    cols = maxCol;
    rows = y + 1;

    x = cols*size;
    y = rows*size;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, depth & 1 ? PATCH_TEXTURE1 : PATCH_TEXTURE0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, OFFSET_TEXTURE);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, cols, rows,
        GL_RGBA, GL_FLOAT, (void*) offset);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, SEGMENT_SIZE/2, cols, rows,
        GL_RGBA, GL_FLOAT, (void*) dmapOffset);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, DMAP_TEXTURE);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);

    patchOffsetY = (depth+2)*(SEGMENT_SIZE>>1);
    set_shader(shader, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glMultiTexCoord4f(GL_TEXTURE1_ARB, 1.0f/(1<<depth), 0.0f, 0.0f, 0.0f);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, 0, 0, patchOffsetY);
    glVertex2i(0, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, 0, cols, patchOffsetY);
    glVertex2i(x, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, rows, cols, rows+patchOffsetY);
    glVertex2i(x, y);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, rows, 0, rows+patchOffsetY);
    glVertex2i(0, y);
    glEnd();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, TCOORD_TEXTURE);

    set_shader(NORMAL_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glMultiTexCoord4f(GL_TEXTURE1_ARB, 1.0f/(1<<depth), 0.0f, 0.0f, 0.0f);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, 0, 0, patchOffsetY);
    glVertex2i(0, SEGMENT_SIZE);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, 0, cols, patchOffsetY);
    glVertex2i(x, SEGMENT_SIZE);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, rows, cols, rows+patchOffsetY);
    glVertex2i(x, y+SEGMENT_SIZE);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, rows, 0, rows+patchOffsetY);
    glVertex2i(0, y+SEGMENT_SIZE);
    glEnd();

    return y;
}

static bool
patchIsFlat(Patch *p, int depth)
{
    int i, j;
    int n;
    uchar *fp;

    n = 1<<depth;
    fp = &p->flatPtr[depth][0];
    for (j = 0; j < n; j++, fp += PATCH_BUFFER_WIDTH)
        for (i = 0; i < n; i++) {
            if (fp[i] != 15)
                return false;
        }

    return true;
}

static void
calcEPFlatData(int depth, GroupInfo *groupInfo)
{
    int w = groupInfo->epTotal;
    int n;
    int v;
    int x, y;

    if (w == 0)
        return;

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);
    set_shader(EPFLAT_SHADER, GL_FRAGMENT_PROGRAM_NV);

    x = 0;
    y = depth*EP_HEIGHT + 2;
    glBegin(GL_QUADS);
    for (v = 0; v <= MAXVALENCE-3; v++) {
        n = groupInfo->epCount[v];
        if (!n)
            continue;

        // clockwise
        glMultiTexCoord4i(GL_TEXTURE0_ARB, v+3, y+1, 0, y);
        glMultiTexCoord4i(GL_TEXTURE1_ARB, 0, v-1, 0, 0);
        glVertex2i(x, 0);
        glVertex2i(x+n, 0);
        glVertex2i(x+n, v+3);
        glVertex2i(x, v+3);

        // counter-clockwise
        glMultiTexCoord4i(GL_TEXTURE0_ARB, v+3, y+1 - MAXVALENCE, 0, y);
        glMultiTexCoord4i(GL_TEXTURE1_ARB, 0, 1-v, 0, 0);
        glVertex2i(x, MAXVALENCE);
        glVertex2i(x+n, MAXVALENCE);
        glVertex2i(x+n, MAXVALENCE+v+3);
        glVertex2i(x, MAXVALENCE+v+3);
        x += n;
    }
    glEnd();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, FLAT_TEXTURE);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, w, MAXVALENCE*2);
}

static Patch *
testFlatness(Patch **prev, int depth, GroupInfo *groupInfo)
{
    Patch *p, *flatList;
    int x, y;
    int w;
    int size, srcSize;
    int i;
    int maxCol;
    int rows, cols;
    int yOffset;
    Patch *patchList = *prev;
    uchar *fptr = FlatBuffer[depth];
    int dmapX, dmapY;

    calcEPFlatData(depth, groupInfo);

    size = (1<<depth) + 1;
    srcSize = (1<<depth) + 3;
    x = 0;
    y = 0;
    i = 0;
    maxCol = 0;
    cols = PATCH_BUFFER_WIDTH/size;
    yOffset = SEGMENT_SIZE*(depth>>1) + 1;
    for (p = patchList; p; p = p->next) {
        if (x == cols) {
            maxCol = x;
            x = 0;
            y++;
        }

        offset[i].x = (float) (p->loc.x*srcSize - x*size + 1);
        offset[i].y = (float) (p->loc.y*srcSize - y*size + yOffset);
        if (depth > p->dmapDepth) {
            dmapX = -(PATCH_BUFFER_WIDTH+1);
            dmapY = 0;
        }
        else {
            dmapX = p->dmapLoc[depth].x - x*size;
            dmapY = p->dmapLoc[depth].y - y*size;
        }
        offset[i].z = (float) dmapX;
        offset[i].w = (float) dmapY;

        p->flatPtr[depth] = fptr + (x + y*PATCH_BUFFER_WIDTH)*size;

        x++;
        i++;
    }

    if (x > maxCol)
        maxCol = x;
    cols = maxCol;
    rows = y + 1;

    x = cols*size;
    y = rows*size;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, depth & 1 ? PATCH_TEXTURE1 : PATCH_TEXTURE0);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, DMAPMAX_TEXTURE);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, OFFSET_TEXTURE);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, cols, rows,
        GL_RGBA, GL_FLOAT, (void*) offset);

    set_shader(depth == 0 ? flattest2_shader : flattest_shader, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, 0, 0, 0);
    glMultiTexCoord4f(GL_TEXTURE1_ARB, 0.0f, -1.0f, 0.0f, MAXVALENCE-1.0f);
    glVertex2i(0, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, 0, 0, 0);
    glVertex2i(x, 0);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, cols, rows, 0, 0);
    glVertex2i(x, y);
    glMultiTexCoord4i(GL_TEXTURE0_ARB, 0, rows, 0, 0);
    glVertex2i(0, y);
    glEnd();

    // pack flatness data to a single byte per quad

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, FLAT_TEXTURE);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 0, 0, x, y);

    w = (x + 15) >> 4;
    x = w << 4;
    set_shader(PACKFLAT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glVertex2i(0, 0);
    glVertex2i(w, 0);
    glVertex2i(w, y);
    glVertex2i(0, y);
    glEnd();

    readFromGPUBuffer(RenderBuffer, (float4 *) FlatBuffer[depth],
                          0, 0, PATCH_BUFFER_WIDTH/16, y, 0);

    // remove flat patches from "to do" list and put them on flatList

    flatList = NULL;
    for (p = patchList; p; p = *prev) {
        if (patchIsFlat(p, depth)) {
            *prev = p->next;
            p->next = flatList;
            flatList = p;
        }
        else
            prev = &p->next;
    }

    return flatList;
}

static Patch *
subdividePatches(Patch *patchList, int depth)
{
    Patch *p, *lastp = NULL;
    int x, y;
    int w, h;
    int i;
    int cols, rows;
    int dstSize = (2<<depth) + 3;
    int srcPatchSize = (1<<depth) + 3;
    float srcSize = dstSize * 0.5f;
    int yOffset = SEGMENT_SIZE*(depth>>1);
    float epY;
    
    x = 0;
    y = 0;
    cols = PATCH_BUFFER_WIDTH/dstSize;
    rows = SEGMENT_SIZE/dstSize;
    i = 0;
    for (p = patchList; p; p = p->next) {
        if (x == cols) {
            x = 0;
            if (y+1 == rows)
                break;      // can't fit any more patches
            y++;
        }

        offset[i].x = x*1.5f + (p->loc.x - x)*srcPatchSize;
        offset[i].y = y*1.5f + (p->loc.y - y)*srcPatchSize + yOffset;
        p->loc.x = x;
        p->loc.y = y;

        x++;
        i++;
        lastp = p;
    }

    rows = y + 1;
    h = rows*dstSize;
    w = cols*dstSize;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, depth & 1 ? PATCH_TEXTURE1 : PATCH_TEXTURE0);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, OFFSET_TEXTURE);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, cols, rows,
        GL_RGBA, GL_FLOAT, (void*) offset);

    epY = (depth+1) * EP_HEIGHT + 2.0f;
    set_shader(SUBDIV_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
    glMultiTexCoord4i(GL_TEXTURE1_ARB, 0, 0, 0, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.751f, 0.751f, 0.0f, epY);
    glVertex2i(0, 0);
    glMultiTexCoord4i(GL_TEXTURE1_ARB, cols, 0, 0, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.751f+cols*srcSize, 0.751f, 0.0f, epY);
    glVertex2i(w, 0);
    glMultiTexCoord4i(GL_TEXTURE1_ARB, cols, rows, 0, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.751f+cols*srcSize, 0.751f+rows*srcSize, 0.0f, epY);
    glVertex2i(w, h);
    glMultiTexCoord4i(GL_TEXTURE1_ARB, 0, rows, 0, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.751f, 0.751f+rows*srcSize, 0.0f, epY);
    glVertex2i(0, h);
    glEnd();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, depth & 1 ? PATCH_TEXTURE0 : PATCH_TEXTURE1);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, SEGMENT_SIZE*((depth+1)>>1), 0, 0, w, h);

    lastp->next = NULL;
    return p;
}

static void
subdivideEPoints(int depth, GroupInfo *groupInfo)
{
    int w = groupInfo->epTotal;
    int x, y;
    int v, n;
    static float weights[MAXVALENCE-2][MAXVALENCE*2+1] = {
        // 3
        (36.0f-7.0f*3)/36.0f,
        6.0f/36.0f, 6.0f/36.0f, 6.0f/36.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // 4
        0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // 5
        (100.0f - 7.0f*5)/100.0f,
        6.0f/100.0f, 6.0f/100.0f, 6.0f/100.0f, 6.0f/100.0f, 6.0f/100.0f, 0.0f, 0.0f, 0.0f,
        1.0f/100.0f, 1.0f/100.0f, 1.0f/100.0f, 1.0f/100.0f, 1.0f/100.0f, 0.0f, 0.0f, 0.0f,
        // 6
        (144.0f - 7.0f*6)/144.0f,
        6.0f/144.0f, 6.0f/144.0f, 6.0f/144.0f, 6.0f/144.0f, 6.0f/144.0f, 6.0f/144.0f, 0.0f, 0.0f,
        1.0f/144.0f, 1.0f/144.0f, 1.0f/144.0f, 1.0f/144.0f, 1.0f/144.0f, 1.0f/144.0f, 0.0f, 0.0f,
        // 7
        (196.0f - 7.0f*7)/196.0f,
        6.0f/196.0f, 6.0f/196.0f, 6.0f/196.0f, 6.0f/196.0f, 6.0f/196.0f, 6.0f/196.0f, 6.0f/196.0f, 0.0f,
        1.0f/196.0f, 1.0f/196.0f, 1.0f/196.0f, 1.0f/196.0f, 1.0f/196.0f, 1.0f/196.0f, 1.0f/196.0f, 0.0f,
        // 8
        (256.0f - 7.0f*8)/256.0f,
        6.0f/256.0f, 6.0f/256.0f, 6.0f/256.0f, 6.0f/256.0f, 6.0f/256.0f, 6.0f/256.0f, 6.0f/256.0f, 6.0f/256.0f,
        1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f, 1.0f/256.0f,
    };

    if (!w)
        return;

    y = depth*EP_HEIGHT;

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, EP_TEXTURE);
    set_shader(EPLIMIT_SHADER, GL_FRAGMENT_PROGRAM_NV);

    // v
    x = 0;
    glBegin(GL_QUADS);
    for (v = 0; v <= MAXVALENCE-3; v++) {
        n = groupInfo->epCount[v];
        if (!n)
            continue;

        glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.0f, (float) (y+2), 0.0f, weights[v][0]);
        glMultiTexCoord4fv(GL_TEXTURE1_ARB, &weights[v][1]);
        glMultiTexCoord4fv(GL_TEXTURE2_ARB, &weights[v][5]);
        glMultiTexCoord4fv(GL_TEXTURE3_ARB, &weights[v][9]);
        glMultiTexCoord4fv(GL_TEXTURE4_ARB, &weights[v][13]);
        glVertex2i(x, 0);
        glVertex2i(x+n, 0);
        glVertex2i(x+n, 1);
        glVertex2i(x, 1);
        x += n;
    }
    glEnd();

    // e

    set_shader(EPSUBDIVE_SHADER, GL_FRAGMENT_PROGRAM_NV);
    x = 0;
    glBegin(GL_QUADS);
    for (v = 0; v <= MAXVALENCE-3; v++) {
        n = groupInfo->epCount[v];
        if (!n)
            continue;

        glMultiTexCoord4i(GL_TEXTURE0_ARB, v+3, y+2, 0, 0);
        glVertex2i(x, 1);
        glVertex2i(x+n, 1);
        // render all the way to MAXVALENCE to avoid NANs in buffer
        glVertex2i(x+n, 1+MAXVALENCE);//glVertex2i(x+n, v+4);
        glVertex2i(x, 1+MAXVALENCE);//glVertex2i(x, v+4);
        x += n;
    }
    glEnd();

    // f

    set_shader(EPSUBDIVF_SHADER, GL_FRAGMENT_PROGRAM_NV);
    x = 0;
    glBegin(GL_QUADS);
    for (v = 0; v <= MAXVALENCE-3; v++) {
        n = groupInfo->epCount[v];
        if (!n)
            continue;

        glMultiTexCoord4i(GL_TEXTURE0_ARB, v+3, y+2, 0, 0);
        glVertex2i(x, 1+MAXVALENCE);
        glVertex2i(x+n, 1+MAXVALENCE);
        // render all the way to MAXVALENCE to avoid NANs in buffer
        glVertex2i(x+n, 1+2*MAXVALENCE);//glVertex2i(x+n, v+4+MAXVALENCE);
        glVertex2i(x, 1+2*MAXVALENCE);//glVertex2i(x, v+4+MAXVALENCE);
        x += n;
    }
    glEnd();

    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, y+2+EP_HEIGHT, 0, 0, w, EP_HEIGHT);
}

void
setShaderParameters()
{
    float4 flatInfo, dmapInfo;

    flatInfo.x = _subd_flat_scale2;
    flatInfo.y = _subd_near_len2;
    flatInfo.z = _subd_near_len2 * 2;
    dmapInfo.x = _subd_dmap_scale * (1.0f/128.0f);
    dmapInfo.y = _subd_dmap_scale1;
    dmapInfo.z = _subd_dmap_scale2;
    dmapInfo.w = _subd_dmap_scale_x_2;
    set_shader_parameter(flattest2_shader, "flatScale", &flatInfo.x);
    flatInfo.x = _subd_flat_scale;
    flatInfo.y = _subd_near_len1;
    flatInfo.z = _subd_near_len1 * 2;
    set_shader_parameter(flattest_shader, "flatScale", &flatInfo.x);
    set_shader_parameter(DFLATTEST_SHADER, "dmapScale", &dmapInfo.x);
    set_shader_parameter(DFLATTEST2_SHADER, "dmapScale", &dmapInfo.x);
    set_shader_parameter(DLIMIT_SHADER, "dmapScale", &dmapInfo.x);
}

void Tessellate(Patch *patchList, pvector *vlist, int vertices, SubdEnum state,
        GroupInfo *groupInfo)
{
    HDC hdc = wglGetCurrentDC();
	HGLRC hglrc = wglGetCurrentContext();
    int resultBufferHeight;
    Patch *p;
    static Patch *flatList;
    static Patch *patchesToLoad;
    int resultH = 0;
    static Patch *toDo[MAXDEPTH+1];
    static int h;
    static int maxDepth;
    static int limitShader;

    if (!RenderBuffer)
        init();
    if (!vlist)
        goto done;
    resultBufferHeight = _subd_vertex_list_size / (PATCH_BUFFER_WIDTH * sizeof(float4));
    setDstGPUBuffer(RenderBuffer);
//while (patchList && patchList->group == 0) // MTB
//patchList = patchList->nextPatch;
    if (state == SUBD_START) {
        if (!perfTestFlag)
        loadVertices(vlist, vertices);
        // set temporary next patch pointer
        for (p = patchList; p; p = p->nextPatch)
            p->next = p->nextPatch;

        CurrentDepth = -1;
        maxDepth = 0;
        patchesToLoad = patchList;
        flatList = NULL;

        limitShader = LIMIT_SHADER;
        flattest_shader = FLATTEST_SHADER;
        flattest2_shader = FLATTEST2_SHADER;
        if (_subd_dmap_scale != 0.0f) {
            limitShader = DLIMIT_SHADER;
            flattest_shader = DFLATTEST_SHADER;
            flattest2_shader = DFLATTEST2_SHADER;
        }
        setShaderParameters();
    }

    for (;;) {
        if (!flatList) {    // if not continuing tessellation
            while (CurrentDepth >= 0 && toDo[CurrentDepth] == NULL)
                CurrentDepth--;
            if (CurrentDepth < 0) {   // No patches in patch buffers
                int groupNum;
                if (!patchesToLoad)
                    break;          // all done
                CurrentDepth = 0;
                groupNum = patchesToLoad->group;
                toDo[0] = patchesToLoad;
                patchesToLoad = createPatches(patchesToLoad, groupInfo + groupNum);
                calcTangents(toDo[0], groupInfo + groupNum);
                maxDepth = 0;   // new extraordinary point data
            }

            if (CurrentDepth == _subd_subdiv_level) {
                flatList = toDo[CurrentDepth];
                toDo[CurrentDepth] = NULL;
            }
            else if (_subd_adaptive_flag)
                flatList = testFlatness(&toDo[CurrentDepth], CurrentDepth, groupInfo);
        }

        // write out vertex data of patch on flat patch list

        if (flatList) {
            if (state != SUBD_CONTINUE)
                h = calcLimit(flatList, CurrentDepth, limitShader);
            state = SUBD_DONE;      // anything but SUBD_CONTINUE
            if (resultH + h > resultBufferHeight) {   // not enough room in result buffer
                _subd_stop_tessellating = true;
                break;
            }
            readFromGPUBuffer(RenderBuffer,
                (float4*) _subd_vertex_list[0] + resultH*PATCH_BUFFER_WIDTH, 0, 0,
                PATCH_BUFFER_WIDTH, h, _subd_vertex_bufobj[0]);
            readFromGPUBuffer(RenderBuffer,
                 (float4*) _subd_vertex_list[1] + resultH*PATCH_BUFFER_WIDTH, 0, SEGMENT_SIZE,
                  PATCH_BUFFER_WIDTH, h, _subd_vertex_bufobj[1]);
            tessPatches(flatList, CurrentDepth, resultH);
            resultH += h;
            flatList = NULL;
        }

        if (toDo[CurrentDepth]) {        // subdivide patches on to do list
            p = toDo[CurrentDepth];
            if (CurrentDepth >= maxDepth) {
                subdivideEPoints(CurrentDepth, groupInfo + p->group);
                maxDepth = CurrentDepth+1;
            }
            toDo[CurrentDepth+1] = p;
            toDo[CurrentDepth] = subdividePatches(p, CurrentDepth); 
            CurrentDepth++;
        }
    }
done:
    setDstGPUBuffer(NULL);
    wglMakeCurrent(hdc, hglrc);
    _subd_vertex_ptr = _subd_vertex_list[0] + resultH*PATCH_BUFFER_WIDTH*4;
}
