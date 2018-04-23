//
// render buffer management routines
//

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

extern PFNGLBINDBUFFERARBPROC glBindBufferARB;
extern PFNGLBUFFERDATAARBPROC glBufferDataARB;
extern PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointerARB;
extern PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB;
extern PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArrayARB;
extern PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTextureARB;


#define VERTEX_ARRAY_RANGE 0


extern bool perfTestFlag;

extern int init_shader(char *, int id, int shaderType);
extern void set_shader(int id, int shaderType);
extern void set_shader_parameter(int id, char *name, float *value);
int perftest_flag;
static int FlipBinormal, CurrentDepth;
static int flattest_shader, epflattest_shader;
static int flattest2_shader, epflattest2_shader;
static int Limit_Shader, EPLimit_Shader;

#define SEGMENT_SIZE    68

#define TEX0            0
#define TEX1            1
#define TEX2            2
#define TEX3            3
#define TEX4            4
#define TEX5            5

#define SUBDIV_SHADER       1
#define LIMIT_SHADER        2
#define NORMAL_SHADER       3
#define EPSUBDIV_SHADER     4
#define FLATTEST_SHADER     5
#define FLATTEST2_SHADER    6
#define EPNORMAL_SHADER     7
#define EPLIMIT_SHADER      8
#define CREATEPATCH_SHADER  9
#define EPFLATTEST_SHADER   10
#define EPFLATTEST2_SHADER  11
#define TANGENT_SHADER      12
#define EPTANGENT_SHADER    13

#define DFLATTEST_SHADER    14
#define DFLATTEST2_SHADER   15
#define DEPFLATTEST_SHADER  16
#define DEPFLATTEST2_SHADER 17
#define DLIMIT_SHADER       18
#define DEPLIMIT_SHADER     19

#define MAX_ATTRIBS     256
#define MAX_PFORMATS    256

static PFNGLACTIVETEXTUREARBPROC glActiveTexture;
static PFNWGLDESTROYPBUFFERARBPROC wglDestroyPbufferARB;
static PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
static PFNWGLCREATEPBUFFERARBPROC wglCreatePbufferARB;
static PFNWGLGETPBUFFERDCARBPROC wglGetPbufferDCARB;
static PFNWGLQUERYPBUFFERARBPROC wglQueryPbufferARB;
static PFNWGLRELEASETEXIMAGEARBPROC wglReleaseTexImageARB;
static PFNWGLBINDTEXIMAGEARBPROC wglBindTexImageARB;

static PFNWGLALLOCATEMEMORYNVPROC wglAllocateMemoryNV;
static PFNWGLFREEMEMORYNVPROC wglFreeMemoryNV;
static PFNGLVERTEXARRAYRANGENVPROC glVertexArrayRangeNV;
static PFNGLFLUSHVERTEXARRAYRANGENVPROC glFlushVertexArrayRangeNV;

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

static GPUBuffer *gGPUList;
static GPUBuffer *lastSrc, *lastDst;
static GPUBuffer *lastSrc2;
static int2 PatchBufferSize = { PATCH_BUFFER_WIDTH, SEGMENT_SIZE*5 };
byte4 *FlatBuffer[MAXDEPTH+1];
void setDstGPUBuffer(GPUBuffer *gp);

#define MAX_SHADER_ATTRIBS      6
#define SHADER_VERTS        (10*1024)
static float4 SubdivAttribBuffer[MAX_SHADER_ATTRIBS*SHADER_VERTS];
static int2 SubdivVertexBuffer[SHADER_VERTS];
static int *vertexBuffer;
static float4 *sabPtr[MAX_SHADER_ATTRIBS];
static int2 *svbPtr;
static int ShaderAttribs;

inline static void
SetAttrib(int index, float x, float y, float z, float w)
{
    sabPtr[index]->x = x;
    sabPtr[index]->y = y;
    sabPtr[index]->z = z;
    sabPtr[index]->w = w;
    sabPtr[index] += ShaderAttribs;
}

inline static void
SetAttribi(int index, int x, int y, int z, int w)
{
    sabPtr[index]->x = (float) x;
    sabPtr[index]->y = (float) y;
    sabPtr[index]->z = (float) z;
    sabPtr[index]->w = (float) w;
    sabPtr[index] += ShaderAttribs;
}

inline static void
SetAttrib4(int index, float x, float y, float z, float w)
{
    int i;

    for (i = 0; i < 4; i++) {
        sabPtr[index]->x = x;
        sabPtr[index]->y = y;
        sabPtr[index]->z = z;
        sabPtr[index]->w = w;
        sabPtr[index] += ShaderAttribs;
    }
}

inline static void
SetAttribi4(int index, int x, int y, int z, int w)
{
    int i;

    for (i = 0; i < 4; i++) {
        sabPtr[index]->x = (float) x;
        sabPtr[index]->y = (float) y;
        sabPtr[index]->z = (float) z;
        sabPtr[index]->w = (float) w;
        sabPtr[index] += ShaderAttribs;
    }
}

inline static void
SetVertex(int x, int y)
{
    svbPtr->x = x;
    svbPtr->y = y;
    svbPtr++;
}

static void wglGetLastError()
{
}

// create render/texture buffer
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

    gp->next = gGPUList;
    gGPUList = gp;

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

int
createShader(int shaderID, char *shaderName)
{
    init_shader(shaderName, shaderID, GL_FRAGMENT_PROGRAM_NV);

    return shaderID;
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
    if (bufObj) {
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, bufObj);
        //glReadBuffer(GL_FRONT);
        glReadPixels(x, y, w, h, GL_RGBA, gp->type, (GLvoid *) dst);
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, 0);
        return;
    }
    glReadPixels(x, y, w, h, GL_RGBA, gp->type, (GLvoid *) dst);
}

static float frac(float a)
{
    return a - (float) floor(a);
}

#define MAX_VALENCE 8

#define V0      0
#define E1      1
#define E2      2
#define E3      3
#define E4      4
#define E5      5
#define E6      6
#define E7      7
#define E8      8

#define F1      9
#define F2     10
#define F3     11
#define F4     12
#define F5     13
#define F6     14
#define F7     15
#define F8     16
#define MAX_CPINDEX 17


static void
subdivideEP(float *ep, int n)
{
	float v, e[MAX_VALENCE], f[MAX_VALENCE];
    int i, j;
    float newf;
    float sume, sumf;

	v = ep[V0];

    sume = 0.0f;
    for (i = 0; i < n; i++) {
        e[i] = ep[E1+i];
        f[i] = ep[F1+i];
        sume += e[i];
    }

    // calc new face points
    sumf = 0.0f;
    for (i = 0; i < n; i++) {
        newf = (v + e[i] + f[i] + e[(i+1) % n]) * 0.25f;
        ep[F1+i] = newf;
        sumf += newf;
    }

    // calc new vertex
    ep[V0] = (ep[V0] * n * (n-2) + sume + sumf) / (n*n);

    // calc new edge points
    j = F1+n-1;
    for (i = 0; i < n; i++) {
        ep[i+E1] =  (e[i] + v + ep[i+F1] + ep[j]) * 0.25f;
        j = F1 + i;
    }
}

static float *calcWeightsFor(int cpIndex, int valence)
{
    static float buffer[MAX_CPINDEX];

    memset(buffer, 0, sizeof(buffer));
    if (cpIndex < 0)
        return buffer;

    buffer[cpIndex] = 1.0f;
    subdivideEP(buffer, valence);
    
    return buffer;
}

#define WWIDTH      96
#define WHEIGHT     6
#define WBLOCKSIZE  (WWIDTH*WHEIGHT)

static int dstOffset23[12] = { E4, F4, F5, F6, F7, F8,
                         V0, E1, E5, E6, E7, E8 };
static int srcOffset23[24] = { F3, E4, F4, F5, F6, F7, F8, -1,
                               E3, V0, E1, E5, E6, E7, E8, -1,
                               F2, E2, F1, -1, -1, -1, -1, -1 };
static int dstOffset14[12] = { E1, F4, F5, F6, F7, F8,
                         V0, E4, E5, E6, E7, E8 };
static int srcOffset14[24] = { F1, E1, F4, F5, F6, F7, F8, -1,
                               E2, V0, E4, E5, E6, E7, E8, -1,
                               F2, E3, F3, -1, -1, -1, -1, -1 };
static float weightTable[WBLOCKSIZE*(MAX_VALENCE-3)];

static float *
BuildTangentTables()
{
    int valence;
    int i, j;
    float *wp;
    extern float tangent_mask3[];
    extern float tangent_mask4[];
    extern float tangent_mask5[];
    extern float tangent_mask6[];
    extern float tangent_mask7[];
    extern float tangent_mask8[];
    float *tangentList[6] = {
        tangent_mask3, tangent_mask4, tangent_mask5, tangent_mask6,
        tangent_mask7, tangent_mask8 };
    float w, w2;
    float *tp;
    int cp;

    wp = weightTable;
    memset(weightTable,0,sizeof(weightTable));

    for (valence = 3; valence <= 8; valence++, wp += 3*32) {
        if (valence == 4) {
            wp -= 3*32;
            continue;
        }
        tp = tangentList[valence-3];
        for (j = 0; j < 3; j++) {
            for (i = 0; i < 8; i++) {
                cp = srcOffset14[i+j*8];
                w = 0;
                w2 = 0;
                if (cp >= E1 && cp < E1+valence) {
                    w = tp[cp-E1];
                    if (cp == E1)
                        w2 = tp[valence-1];
                    else
                        w2 = tp[cp-E1-1];
                }
                else if (cp >= F1 && cp < F1+valence) {
                    w = tp[cp-F1 + valence];
                    if (cp == F1)
                        w2 = tp[valence+valence-1];
                    else
                        w2 = tp[cp-F1-1+valence];
                }
                wp[j*32+i] = w;
                wp[j*32+i+8] = w2;

                cp = srcOffset23[i+j*8];
                w = 0;
                w2 = 0;
                if (cp >= E1 && cp < E1+valence) {
                    w = tp[cp-E1];
                    if (cp == E1)
                        w2 = tp[valence-1];
                    else
                        w2 = tp[cp-E1-1];
                }
                else if (cp >= F1 && cp < F1+valence) {
                    w = tp[cp-F1 + valence];
                    if (cp == F1)
                        w2 = tp[valence+valence-1];
                    else
                        w2 = tp[cp-F1-1+valence];
                }
                wp[j*32+i+16] = w;
                wp[j*32+i+24] = w2;
            }
        }
    }
    return weightTable;
}

static float *
BuildLimitTables()
{
    int valence;
    int i, j;
    float *wp;
    float w;
    int cp;
    float vertexWeight, faceWeight, edgeWeight;

    wp = weightTable;
    memset(weightTable, 0, sizeof(weightTable));

    for (valence = 3; valence <= 8; valence++, wp += 3*32) {
        if (valence == 4) {
            wp -= 3*32;
            continue;
        }
        vertexWeight = ((float) valence) / (valence + 5.0f);
        faceWeight = 1.0f/(valence * (valence + 5.0f));
        edgeWeight = faceWeight * 4.0f;

        for (j = 0; j < 3; j++) {
            for (i = 0; i < 8; i++) {
                cp = srcOffset14[i+j*8];
                w = 0;
                if (cp == V0)
                    w = vertexWeight;
                else if (cp >= E1 && cp < E1+valence)
                    w = edgeWeight;
                else if (cp >= F1 && cp < F1+valence)
                    w = faceWeight;

                wp[j*32+i] = w;

                cp = srcOffset23[i+j*8];
                w = 0;
                if (cp == V0)
                    w = vertexWeight;
                else if (cp >= E1 && cp < E1+valence)
                    w = edgeWeight;
                else if (cp >= F1 && cp < F1+valence)
                    w = faceWeight;
                wp[j*32+i+16] = w;
            }
        }
    }
    return weightTable;
}

static float *
BuildWeightTables()
{
    int valence;
    int i, j;
    int k, l;
    float *buffer;
    float *wp;

    wp = weightTable;

    for (valence = 3; valence <= 8; valence++, wp += WBLOCKSIZE) {
        if (valence == 4) {
            wp -= WBLOCKSIZE;
            continue;
        }
        if (valence == 3) {
            dstOffset23[0] = E1;
            dstOffset14[7] = E1;
        }
        else {
            dstOffset23[0] = E4;
            dstOffset14[7] = E4;
        }
        for (j = 0; j < 3; j++) {
            for (i = 0; i < 8; i++) {
                buffer = calcWeightsFor(srcOffset14[i+j*8], valence);
                for (k = 0; k < 2; k++) {
                    for (l = 0; l < 6; l++) {
                        wp[k*96*3+l*8+i+j*96] = buffer[dstOffset14[k*6+l]];
                    }
                }
                buffer = calcWeightsFor(srcOffset23[i+j*8], valence);
                for (k = 0; k < 2; k++) {
                    for (l = 0; l < 6; l++) {
                        wp[48+k*96*3+l*8+i+j*96] = buffer[dstOffset23[k*6+l]];
                    }
                }
            }
        }
    }
    return weightTable;
}

void
setDstGPUBuffer(GPUBuffer *gp)
{
    if (lastSrc && (lastSrc == gp || lastDst != gp)) {
        wglReleaseTexImageARB(lastSrc->hpbuffer, WGL_FRONT_LEFT_ARB);
        lastSrc = 0;
    }
    if (lastDst != gp) {
        if (gp)
            wglMakeCurrent(gp->hdc, gp->hglrc);
        lastDst = gp;
    }
}

void
setSrcGPUBuffer(GPUBuffer *srcgp)
{
    if (lastSrc != srcgp) {
        if (lastSrc)
            wglReleaseTexImageARB(lastSrc->hpbuffer, WGL_FRONT_LEFT_ARB);
        if (srcgp)
            wglBindTexImageARB(srcgp->hpbuffer, WGL_FRONT_LEFT_ARB);
        lastSrc = srcgp;
    }

}

void
setSrcGPUBuffer2(GPUBuffer *srcgp)
{
    if (lastSrc2 != srcgp) {
        if (lastSrc2 && lastSrc2 != lastSrc)
            wglReleaseTexImageARB(lastSrc2->hpbuffer, WGL_FRONT_LEFT_ARB);
        if (srcgp)
            wglBindTexImageARB(srcgp->hpbuffer, WGL_FRONT_LEFT_ARB);
        lastSrc2 = srcgp;
    }
}

void
BeginShade(GPUBuffer *srcgp, GPUBuffer *srcgp2, GPUBuffer *dstgp, int shaderID, int attribs)
{
    int i;

    ShaderAttribs = attribs;
    setDstGPUBuffer(dstgp);
    set_shader(shaderID, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcgp);
    if (srcgp2) {
        glActiveTexture(GL_TEXTURE1);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, 3);
        setSrcGPUBuffer2(srcgp2);
        glActiveTexture(GL_TEXTURE0);
    }
    glEnable(GL_FRAGMENT_PROGRAM_NV);

    for (i = 0; i < ShaderAttribs; i++)
        sabPtr[i] = SubdivAttribBuffer + i;
    svbPtr = SubdivVertexBuffer;
}

void
EndShade(GPUBuffer *srcgp, GPUBuffer *dstgp)
{
    int i;

#if 0
    glEnableVertexAttribArrayARB(0);
    glVertexAttribPointerARB(0, 2, GL_INT, GL_FALSE, sizeof(int2), SubdivVertexBuffer);
    for (i = 0; i < SHADER_ATTRIBS; i++) {
        glEnableVertexAttribArrayARB(8+i); // TEX0 is 8
        glVertexAttribPointerARB(8+i, 4, GL_FLOAT, GL_FALSE,
            sizeof(float4)*SHADER_ATTRIBS, SubdivAttribBuffer + i);
    }

printf("Draw quads %d\n", svbPtr - SubdivVertexBuffer);
    glDrawArrays(GL_QUADS, 0, svbPtr - SubdivVertexBuffer);
    glDisableVertexAttribArrayARB(0);
    for (i = 0; i < SHADER_ATTRIBS; i++)
        glDisableVertexAttribArrayARB(8+i); // TEX0 is 8
#endif
    glEnable(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_INT, sizeof(int2), SubdivVertexBuffer);
    
    for (i = 0; i < ShaderAttribs; i++) {
        glClientActiveTextureARB(GL_TEXTURE0_ARB+i);
		glTexCoordPointer(4, GL_FLOAT, sizeof(float4)*ShaderAttribs,
            SubdivAttribBuffer + i);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if (svbPtr != SubdivVertexBuffer)
        glDrawArrays(GL_QUADS, 0, svbPtr - SubdivVertexBuffer);

    for (i = 0; i < ShaderAttribs; i++) {
        glClientActiveTextureARB(GL_TEXTURE0_ARB+i);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    glDisable(GL_VERTEX_ARRAY);

   // glDisable(GL_TEXTURE_RECTANGLE_NV);
}

void EnableAttrib(int attrib, int size, int type, int stride, char *ptr)
{
    glClientActiveTextureARB(GL_TEXTURE0_ARB+attrib);
    glTexCoordPointer(size, type, stride, ptr);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
}

void
EndShadeI(GPUBuffer *srcgp, GPUBuffer *dstgp, char *vBuf, int indices, int stride, int attribs)
{
    int i;

    glEnable(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_INT, stride, vBuf);
    
    /*for (i = 0; i < attribs; i++) {
        glClientActiveTextureARB(GL_TEXTURE0_ARB+i);
		glTexCoordPointer(4, GL_INT, stride,
            vBuf + 2*sizeof(int) + i*(4*sizeof(int)));
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }*/

    glDrawArrays(GL_QUADS, 0, indices);

    for (i = 0; i < attribs; i++) {
        glClientActiveTextureARB(GL_TEXTURE0_ARB+i);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    glDisable(GL_VERTEX_ARRAY);
}

static Patch *
SubdividePatches(Patch *p, GPUBuffer *srcBuf, int srcY, GPUBuffer *dstBuf, int dstY)
{
    int newSize;
    int left, right;
    int maxPatchHeight = 0;
    float srcSize;
    int x, y;
    Patch *lastp = NULL;
    int *vptr = (int *) vertexBuffer;
    int indices = 0;

#if VERTEX_ARRAY_RANGE
    // glFlushVertexArrayRangeNV();
#endif

    x = y = 0;
    for (; p; p = p->next) {
        left = p->epLeft;
        right = p->epRight;
        newSize = p->size * 2 - 3;
        srcSize = newSize * 0.5f;
        if (x + newSize + left + right > PatchBufferSize.x) {
            x = 0;
            y += maxPatchHeight;
            maxPatchHeight = 0;
        }
        if (y + newSize > SEGMENT_SIZE)
            break;      // can't fit any more patches

        int dX = x + left;
        int dY = y + dstY;
        float sX = p->loc[CURRENT_LOC] + 0.75f;
        float sY = p->loc[CURRENT_LOC+1] + 0.75f + srcY;
        int w = newSize;

        vptr[0] = dX;
        vptr[1] = dY;
        ((float*)vptr)[2] = sX;
        ((float*)vptr)[3] = sY;
        vptr += 4;

        vptr[0] = dX+w;
        vptr[1] = dY;
        ((float*)vptr)[2] = sX+srcSize;
        ((float*)vptr)[3] = sY;
        vptr += 4;

        vptr[0] = dX+w;
        vptr[1] = dY+w;
        ((float*)vptr)[2] = sX+srcSize;
        ((float*)vptr)[3] = sY+srcSize;
        vptr += 4;

        vptr[0] = dX;
        vptr[1] = dY+w;
        ((float*)vptr)[2] = sX;
        ((float*)vptr)[3] = sY+srcSize;
        vptr += 4;

        indices += 4;

        p->size = newSize;
        if (newSize > maxPatchHeight)
            maxPatchHeight = newSize;
        p->loc[LAST_LOC] = p->loc[CURRENT_LOC];
        p->loc[LAST_LOC+1] = p->loc[CURRENT_LOC+1];
        p->loc[CURRENT_LOC] = x + left;
        p->loc[CURRENT_LOC+1] = y;

        x += newSize + left + right;
        lastp = p;
    }

    setDstGPUBuffer(dstBuf);
    set_shader(SUBDIV_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcBuf);
    glEnable(GL_FRAGMENT_PROGRAM_NV);
    EnableAttrib(0, 2, GL_FLOAT, 4*sizeof(int), (char *) vertexBuffer + 2*sizeof(int));

#if VERTEX_ARRAY_RANGE
  //  glVertexArrayRangeNV((vptr - vertexBuffer)*sizeof(*vptr), vertexBuffer);
#endif

    EndShadeI(srcBuf, dstBuf, (char *) vertexBuffer, indices, 4*sizeof(int), 1);

    lastp->next = NULL;
    return p;
}

static int
TestFlatness(Patch *p, GPUBuffer *srcBuf, int srcY, GPUBuffer *dstBuf, int shader)
{
    int maxPatchHeight = 0;
    int x, y;
    int *vptr = (int *) SubdivAttribBuffer;
    int indices = 0;
    int w;

    x = y = 0;
    for (; p; p = p->next) {
        w = p->size - 3;
        if (x + w > PatchBufferSize.x) {
            x = 0;
            y += maxPatchHeight;
            maxPatchHeight = 0;
        }

        if (shader != flattest2_shader || p->ep == 0) {
            int sX = p->loc[CURRENT_LOC] + 1;
            int sY = p->loc[CURRENT_LOC+1] + 1 + srcY;
            int sX2 = p->dmapMaxLoc[CurrentDepth].x;
            int sY2 = p->dmapMaxLoc[CurrentDepth].y;
            int s = w;
            if (CurrentDepth > p->dmapDepth) {
                sX2 = sY2 = 0;
                s = 1;
            }

            vptr[0] = x;
            vptr[1] = y;
            vptr[2] = sX;
            vptr[3] = sY;
            vptr[4] = sX2;
            vptr[5] = sY2;
            vptr += 6;

            vptr[0] = x+w;
            vptr[1] = y;
            vptr[2] = sX+w;
            vptr[3] = sY;
            vptr[4] = sX2+s;
            vptr[5] = sY2;
            vptr += 6;

            vptr[0] = x+w;
            vptr[1] = y+w;
            vptr[2] = sX+w;
            vptr[3] = sY+w;
            vptr[4] = sX2+s;
            vptr[5] = sY2+s;
            vptr += 6;

            vptr[0] = x;
            vptr[1] = y+w;
            vptr[2] = sX;
            vptr[3] = sY+w;
            vptr[4] = sX2;
            vptr[5] = sY2+s;
            vptr += 6;

            indices += 4;
        }

        if (w > maxPatchHeight)
            maxPatchHeight = w;
        p->loc[FLATINFO_LOC] = x;
        p->loc[FLATINFO_LOC+1] = y;
        x += w;
    }
    setDstGPUBuffer(dstBuf);
    set_shader(shader, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcBuf);
    glEnable(GL_FRAGMENT_PROGRAM_NV);
    EnableAttrib(0, 4, GL_INT, 6*sizeof(int), (char *) SubdivAttribBuffer + 2*sizeof(int));

    EndShadeI(srcBuf, dstBuf, (char *) SubdivAttribBuffer, indices, 6*sizeof(int), 1);

    return y + maxPatchHeight;
}

#define emit_quad(i1, i2, i3, i4) \
            prim_ptr[0] = i1; \
            prim_ptr[1] = i2; \
            prim_ptr[2] = i3; \
            prim_ptr[3] = i4; \
            prim_ptr += 4;

//#define emit_quad(i1, i2, i3, i4) _subd_emit_quad(i1, i2, i3, i4)

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
            flat |= p->flatPtr[depth][flatY*w + flatX].x;
        else {
            flat |= p->flatPtr[depth][flatY*w + flatX].x & (TOP|LEFT);
            flat |= p->flatPtr[depth][flatY2*w + flatX2].x & (BOTTOM|RIGHT);
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
    int stride = PatchBufferSize.x;
    int n;

    for (; p; p = p->next) {
        vertexIndex = p->loc[LIMIT_LOC] + (p->loc[LIMIT_LOC+1]+yOffset)*stride;
        n = p->size - 3;
        tessPatch(p, vertexIndex, vertexIndex, 0, 0, n, stride, 0, 0, 0, 0,
            vertexIndex, vertexIndex + n,
            vertexIndex + n*stride, vertexIndex+n*stride+n);
    }
}

static int
patchIsFlat(Patch *p, int depth)
{
    int i, j;
    int n;
    byte4 *fp;

    n = p->size - 3;
    fp = &p->flatPtr[depth][0];
    for (j = 0; j < n; j++, fp += PatchBufferSize.x)
        for (i = 0; i < n; i++) {
            if (fp[i].x != 15.0f)
                return 0;
        }

    return 1;
}

static void
subdivEPoints(Patch *pList, GPUBuffer *srcgp, int srcSeg, GPUBuffer *dstgp, int dstSeg,
        int srcLoc, int dstLoc)
{
    Patch *p;
    int shaderIsSet = 0;
    int dstX, dstY, srcX, srcY;
    int srcW, dstW;
    int wy;
    int x, y;
    int w;

    for (p = pList; p; p = p->next) {
        if (!p->ep)
            continue;
        if (!shaderIsSet) {
            shaderIsSet = 1;
            BeginShade(srcgp, 0, dstgp, EPSUBDIV_SHADER, 2);
        }
        srcX = p->loc[LAST_LOC]; 
        srcY = p->loc[LAST_LOC+1] + srcSeg;
        dstX = p->loc[CURRENT_LOC];
        dstY = p->loc[CURRENT_LOC+1] + dstSeg;
        srcW = ((p->size-3)>>1) + 3;
        dstW = p->size;

        if (p->ep & EP1) {
            x = srcX+2;
            y = srcY;
            wy = (p->epValence[0] - 4) * WHEIGHT;
            w = p->epValence[0] - 2;
            if (wy < 0) {
                wy = 0;
                w = 2;
            }
            SetAttribi(TEX0, x, y, x-1, y);
            SetAttribi4(TEX1, 0, wy + PatchBufferSize.y, -1, 1);
            SetVertex(dstX+2, dstY);
            SetAttribi(TEX0, x-w, y, x-1, y);
            SetVertex(dstX+2-w, dstY);
            SetAttribi(TEX0, x-w, y+2, x-1, y);
            SetVertex(dstX+2-w, dstY + 2);
            SetAttribi(TEX0, x, y+2, x-1, y);
            SetVertex(dstX+2, dstY + 2);
        }
        if (p->ep & EP2) {
            x = srcX+srcW-2;
            y = srcY;
            wy = (p->epValence[1] - 4) * WHEIGHT;
            w = p->epValence[1] - 2;
            if (wy < 0) {
                wy = 0;
                w = 2;
            }
            SetAttribi(TEX0, x, y, x, y);
            SetAttribi4(TEX1, WWIDTH>>3, wy + PatchBufferSize.y, 1, 1);
            SetVertex(dstX+dstW-2, dstY);
            SetAttribi(TEX0, x+w, y, x, y);
            SetVertex(dstX + dstW-2+w, dstY);
            SetAttribi(TEX0, x+w, y+2, x, y);
            SetVertex(dstX + dstW-2+w, dstY + 2);
            SetAttribi(TEX0, x, y+2, x, y);
            SetVertex(dstX+dstW-2, dstY + 2);
        }
        if (p->ep & EP3) {
            x = srcX+2;
            y = srcY+srcW;
            wy = (p->epValence[2] - 4) * WHEIGHT;
            w = p->epValence[2] - 2;
            if (wy < 0) {
                wy = 0;
                w = 2;
            }
            SetAttribi(TEX0, x, y, x-1, y-1);
            SetAttribi4(TEX1, WWIDTH>>3, wy + PatchBufferSize.y, -1, -1);
            SetVertex(dstX+2, dstY+dstW);
            SetAttribi(TEX0, x-w, y, x-1, y-1);
            SetVertex(dstX+2-w, dstY+dstW);
            SetAttribi(TEX0, x-w, y-2, x-1, y-1);
            SetVertex(dstX+2-w, dstY+dstW - 2);
            SetAttribi(TEX0, x, y-2, x-1, y-1);
            SetVertex(dstX+2, dstY+dstW - 2);
        }
        if (p->ep & EP4) {
            x = srcX+srcW-2;
            y = srcY+srcW;
            wy = (p->epValence[3] - 4) * WHEIGHT;
            w = p->epValence[3] - 2;
            if (wy < 0) {
                wy = 0;
                w = 2;
            }
            SetAttribi(TEX0, x, y, x, y-1);
            SetAttribi4(TEX1, 0, wy + PatchBufferSize.y, 1, -1);
            SetVertex(dstX+dstW-2, dstY+dstW);
            SetAttribi(TEX0, x+w, y, x, y-1);
            SetVertex(dstX + dstW-2+w, dstY+dstW);
            SetAttribi(TEX0, x+w, y-2, x, y-1);
            SetVertex(dstX + dstW-2+w, dstY+dstW - 2);
            SetAttribi(TEX0, x, y-2, x, y-1);
            SetVertex(dstX+dstW-2, dstY+dstW - 2);
        }
    }
    if (shaderIsSet) {
        shaderIsSet = 1;
        EndShade(srcgp, dstgp);
    }
}

static void
calcEPValues(Patch *pList, GPUBuffer *srcgp, int srcSeg, GPUBuffer *dstgp, int dstSeg,
        int shader, int srcLoc, int dstLoc, int wTableX, int wTableY, GPUBuffer *srcgp2)
{
    Patch *p;
    int shaderIsSet = 0;
    int dstX, dstY, srcX, srcY;
    int srcW, dstW;
    int wy;

    for (p = pList; p; p = p->next) {
        if (!p->ep)
            continue;
        if (!shaderIsSet) {
            shaderIsSet = 1;
            BeginShade(srcgp, srcgp2, dstgp, shader, 4);
        }
        srcX = p->loc[srcLoc]; 
        srcY = p->loc[srcLoc+1] + srcSeg;
        dstX = p->loc[dstLoc];
        dstY = p->loc[dstLoc+1] + dstSeg;
        srcW = p->size;
        dstW = p->size-2;

        if (p->ep & EP1) {
            wy = (p->epValence[0] - 4) * 3;
            if (wy < 0) 
                wy = 0;
            SetAttribi4(TEX0, srcX+1, srcY, 0, 0);
            SetAttribi4(TEX1, wTableX, wTableY+wy, -1, 1);
            SetAttrib4(TEX2, p->texCoordLoc.x + 0.5f,
                p->texCoordLoc.y + 0.5f, p->tanLoc.x+0.5f, p->tanLoc.y+0.5f);
            if (FlipBinormal)
                SetAttrib4(TEX3, (float) p->flipBinormal, 0.0f,
                    p->dmapLoc[0].x+0.5f, p->dmapLoc[0].y + 0.5f);
            SetVertex(dstX, dstY);
            SetVertex(dstX+1, dstY);
            SetVertex(dstX+1, dstY+1);
            SetVertex(dstX, dstY+1);
        }
        if (p->ep & EP2) {
            wy = (p->epValence[1] - 4) * 3;
            if (wy < 0)
                wy = 0;
            SetAttribi4(TEX0, srcX+srcW-2, srcY, 0, 0);
            SetAttribi4(TEX1, wTableX+4, wTableY+wy, 1, 1);
            SetAttrib4(TEX2, p->texCoordLoc.x + 1.5f,
                p->texCoordLoc.y + 0.5f, p->tanLoc.x+1.5f, p->tanLoc.y+0.5f);
            if (FlipBinormal)
                SetAttrib4(TEX3, (float) p->flipBinormal, 0.0f,
                    p->dmapLoc[0].x+1.5f, p->dmapLoc[0].y+0.5f);
            SetVertex(dstX+dstW-1, dstY);
            SetVertex(dstX+dstW, dstY);
            SetVertex(dstX+dstW, dstY+1);
            SetVertex(dstX+dstW-1, dstY+1);
        }
        if (p->ep & EP3) {
            wy = (p->epValence[2] - 4) * 3;
            if (wy < 0)
                wy = 0;
            SetAttribi4(TEX0, srcX+1, srcY+srcW-1, 0, 0);
            SetAttribi4(TEX1, wTableX+4, wTableY+wy, -1, -1);
            SetAttrib4(TEX2, p->texCoordLoc.x + 0.5f,
                p->texCoordLoc.y + 1.5f, p->tanLoc.x+0.5f, p->tanLoc.y+1.5f);
            if (FlipBinormal)
                SetAttrib4(TEX3, (float) p->flipBinormal, 0.0f,
                    p->dmapLoc[0].x+0.5f, p->dmapLoc[0].y+1.5f);
            SetVertex(dstX, dstY+dstW-1);
            SetVertex(dstX+1, dstY+dstW-1);
            SetVertex(dstX+1, dstY+dstW);
            SetVertex(dstX, dstY+dstW);
        }
        if (p->ep & EP4) {
            wy = (p->epValence[3] - 4) * 3;
            if (wy < 0)
                wy = 0;
            SetAttribi4(TEX0, srcX+srcW-2, srcY+srcW-1, 0, 0);
            SetAttribi4(TEX1, wTableX, wTableY+wy, 1, -1);
            SetAttrib4(TEX2, p->texCoordLoc.x + 1.5f,
                p->texCoordLoc.y + 1.5f, p->tanLoc.x+1.5f, p->tanLoc.y+1.5f);
            if (FlipBinormal)
                SetAttrib4(TEX3, (float) p->flipBinormal, 0.0f,
                    p->dmapLoc[0].x+1.5f, p->dmapLoc[0].y+1.5f);
            SetVertex(dstX+dstW-1, dstY+dstW-1);
            SetVertex(dstX+dstW, dstY+dstW-1);
            SetVertex(dstX+dstW, dstY+dstW);
            SetVertex(dstX+dstW-1, dstY+dstW);
        }
    }
    if (shaderIsSet) {
        shaderIsSet = 1;
        EndShade(srcgp, dstgp);
    }
}

static void
epFlatTest(Patch *pList, GPUBuffer *srcgp, int srcSeg, GPUBuffer *dstgp, int dstSeg,
        int shader, int srcLoc, int dstLoc)
{
    static int offsets[6][4][4] = {
        // valence 3
        1, 0,  0, 1,
        0, 0,  1, 1,
        0, 0,  1, 1,
        1, 0,  0, 1,

        // valence 4
        0, -1,  -1, 0,
        1, -1,  2, 0,
        -1, 1,  0, 2,
        2, 1,  1, 2,

        // valence 5

        -2, 0,  -2, 0,
        3, 0,  3, 0,
        -2, 1,  -2, 1,
        3, 1,  3, 1,

        // valence 6

        -2, 0,  -3, 0,
        4, 0,  3, 0,
        -2, 1,  -3, 1,
        4, 1,  3, 1,

        // valence 7

        -2, 0,  -4, 0,
        5, 0,  3, 0,
        -2, 1,  -4, 1,
        5, 1,  3, 1,

        // valence 8

        -2, 0,  -5, 0,
        6, 0,  3, 0,
        -2, 1,  -5, 1,
        6, 1,  3, 1
    };
    Patch *p;
    int shaderIsSet = 0;
    int dstX, dstY, srcX, srcY;
    int s, w;
    int x, y;
    int *op;
    int2 loc;

    for (p = pList; p; p = p->next) {
        if (!p->ep)
            continue;
        if (!shaderIsSet) {
            shaderIsSet = 1;
            BeginShade(srcgp, 0, dstgp, shader, 5);
        }
        srcX = p->loc[srcLoc]+1; 
        srcY = p->loc[srcLoc+1] + srcSeg + 1;
        dstX = p->loc[dstLoc];
        dstY = p->loc[dstLoc+1] + dstSeg;
        s = p->size-3;
        
        loc = p->dmapMaxLoc[CurrentDepth];
        w = s-1;
        if (CurrentDepth > p->dmapDepth) {
            loc.x = loc.y = 0;
            w = 0;
        }

        if (s == 1) {
            if (p->ep == 0)
                continue;
            x = srcX;
            y = srcY;
            SetAttrib4(TEX0, (float) x, (float) y, loc.x+0.5f, loc.y+0.5f);
            op = &offsets[p->epValence[0]-3][0][0];
            SetAttribi4(TEX1, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[p->epValence[1]-3][1][0];
            SetAttribi4(TEX2, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[p->epValence[2]-3][2][0];
            SetAttribi4(TEX3, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[p->epValence[3]-3][3][0];
            SetAttribi4(TEX4, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            SetVertex(dstX, dstY);
            SetVertex(dstX+1, dstY);
            SetVertex(dstX+1, dstY+1);
            SetVertex(dstX, dstY+1);
            continue;
        }

        if (p->ep & EP1) {
            x = srcX;
            y = srcY;
            SetAttrib4(TEX0, (float) x, (float) y, loc.x+0.5f, loc.y+0.5f);
            op = &offsets[p->epValence[0]-3][0][0];
            SetAttribi4(TEX1, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][1][0];
            SetAttribi4(TEX2, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][2][0];
            SetAttribi4(TEX3, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][3][0];
            SetAttribi4(TEX4, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            SetVertex(dstX, dstY);
            SetVertex(dstX+1, dstY);
            SetVertex(dstX+1, dstY+1);
            SetVertex(dstX, dstY+1);
        }
        if (p->ep & EP2) {
            x = srcX+s-1;
            y = srcY;
            SetAttrib4(TEX0, (float) x, (float) y, loc.x+w+0.5f, loc.y+0.5f);
            op = &offsets[1][0][0];
            SetAttribi4(TEX1, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[p->epValence[1]-3][1][0];
            SetAttribi4(TEX2, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][2][0];
            SetAttribi4(TEX3, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][3][0];
            SetAttribi4(TEX4, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            SetVertex(dstX+s-1, dstY);
            SetVertex(dstX+s, dstY);
            SetVertex(dstX+s, dstY+1);
            SetVertex(dstX+s-1, dstY+1);
        }
        if (p->ep & EP3) {
            x = srcX;
            y = srcY+s-1;
            SetAttrib4(TEX0, (float) x, (float) y, loc.x+0.5f, loc.y+w+0.5f);
            op = &offsets[1][0][0];
            SetAttribi4(TEX1, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][1][0];
            SetAttribi4(TEX2, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[p->epValence[2]-3][2][0];
            SetAttribi4(TEX3, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][3][0];
            SetAttribi4(TEX4, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            SetVertex(dstX, dstY+s-1);
            SetVertex(dstX+1, dstY+s-1);
            SetVertex(dstX+1, dstY+s);
            SetVertex(dstX, dstY+s);
        }
        if (p->ep & EP4) {
            x = srcX+s-1;
            y = srcY+s-1;
            SetAttrib4(TEX0, (float) x, (float) y, loc.x+w+0.5f, loc.y+w+0.5f);
            op = &offsets[1][0][0];
            SetAttribi4(TEX1, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][1][0];
            SetAttribi4(TEX2, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[1][2][0];
            SetAttribi4(TEX3, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            op = &offsets[p->epValence[3]-3][3][0];
            SetAttribi4(TEX4, op[0]+x, op[1]+y, op[2]+x, op[3]+y);
            SetVertex(dstX+s-1, dstY+s-1);
            SetVertex(dstX+s, dstY+s-1);
            SetVertex(dstX+s, dstY+s);
            SetVertex(dstX+s-1, dstY+s);
        }
    }
    if (shaderIsSet) {
        shaderIsSet = 1;
        EndShade(srcgp, dstgp);
    }
}

static Patch *
createPatches(Patch *patchList, GPUBuffer *srcBuf, GPUBuffer *dstBuf, int dstSeg)
{
    Patch *p, *lastp;
    int locX, locY;
    int n;
    int w, h;

    locX = 0; locY = 0;
    lastp = NULL;
    for (p = patchList; p; p = p->next) {
        p->size = 4;

        if (locX + 4 > PATCH_BUFFER_WIDTH) {
            locX = 0;
            locY += 4;
            if (locY + 4 > SEGMENT_SIZE)
                break;  // cannot fit any more
        }

        p->loc[CURRENT_LOC] = locX;
        p->loc[CURRENT_LOC+1] = locY;
        locX += 4;
        lastp = p;
    }
    w = PATCH_BUFFER_WIDTH;
    h = locY;
    if (locX)
        h += 4;
    BeginShade(srcBuf, 0, dstBuf, CREATEPATCH_SHADER, 1);
    SetAttribi(TEX0, 0, patchList->indexLoc.y, 0, 0);
    SetAttribi(TEX0, PATCH_BUFFER_WIDTH, patchList->indexLoc.y, 0, 0);
    SetAttribi(TEX0, PATCH_BUFFER_WIDTH, patchList->indexLoc.y + h, 0, 0);
    SetAttribi(TEX0, 0, patchList->indexLoc.y+h, 0, 0);
    SetVertex(0, dstSeg);
    SetVertex(w, dstSeg);
    SetVertex(w, dstSeg + h);
    SetVertex(0, dstSeg + h);

    EndShade(srcBuf, dstBuf);

    lastp->next = NULL;
    return p;
}

inline static void
shadeTan(int srcX, int srcY, int dstX, int dstY, int tcX, int tcY,
         float e5, float e6, float e7, float e8)
{
    SetAttrib4(TEX0, srcX+1.5f, srcY+0.5f, srcX-0.5f, srcX+e5);
    SetAttrib4(TEX1, srcX+e6, srcY+0.5f, srcX+e7, srcX+e8);
    SetAttrib4(TEX2, srcX+0.5f, srcY+0.5f, srcY-0.5f, srcY+1.5f);
    SetAttrib4(TEX3, tcX+0.5f, tcY+0.5f, tcY+1.5f, 0.0f);

    SetVertex(dstX, dstY);
    SetVertex(dstX+1, dstY);
    SetVertex(dstX+1, dstY+1);
    SetVertex(dstX, dstY+1);
}

static void
calcTangents(Patch *patchList, GPUBuffer *srcBuf, int srcSeg, GPUBuffer *dstBuf, int dstSeg)
{
    Patch *p;
    int2 loc;
    int maxPatchSize = 0;
    int x;
    int *vptr = (int *) SubdivAttribBuffer;
    int indices = 0;

    loc.x = loc.y = 0;

    for (p = patchList; p; p = p->next) {
        if (loc.x + 2 > PatchBufferSize.x) {
            loc.x = 0;
            loc.y += 2;
        }

         if (p->ep != (EP1|EP2|EP3|EP4) || p->size != 4) {
            int srcX = p->loc[CURRENT_LOC]+1;
            int srcY = p->loc[CURRENT_LOC+1]+1+srcSeg;
            int w = 2; int h = 2;
            int dstX = loc.x;
            int dstY = loc.y + dstSeg;
            int srcX2 = p->texCoordLoc.x;
            int srcY2 = p->texCoordLoc.y;

            vptr[0] = dstX;
            vptr[1] = dstY;
            vptr[2] = srcX;
            vptr[3] = srcY;
            vptr[4] = srcX2;
            vptr[5] = srcY2;
            vptr += 6;

            vptr[0] = dstX+w;
            vptr[1] = dstY;
            vptr[2] = srcX+w;
            vptr[3] = srcY;
            vptr[4] = srcX2+w;
            vptr[5] = srcY2;
            vptr += 6;

            vptr[0] = dstX+w;
            vptr[1] = dstY+h;
            vptr[2] = srcX+w;
            vptr[3] = srcY+h;
            vptr[4] = srcX2+w;
            vptr[5] = srcY2+h;
            vptr += 6;

            vptr[0] = dstX;
            vptr[1] = dstY+h;
            vptr[2] = srcX;
            vptr[3] = srcY+h;
            vptr[4] = srcX2;
            vptr[5] = srcY2+h;
            vptr += 6;

            indices += 4;
        }

        p->tanLoc = loc;
        loc.x += 2;
    }
    setDstGPUBuffer(dstBuf);
    set_shader(TANGENT_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcBuf);
    glEnable(GL_FRAGMENT_PROGRAM_NV);
    EnableAttrib(0, 4, GL_INT, 6*sizeof(int), (char *) SubdivAttribBuffer + 2*sizeof(int));

    EndShadeI(srcBuf, dstBuf, (char *) SubdivAttribBuffer, indices, 6*sizeof(int), 1);

    // calc tangents at extra-ordinary points

    BeginShade(srcBuf, 0, dstBuf, EPTANGENT_SHADER, 4);
    for (p = patchList; p; p = p->next) {
        if (!p->ep)
            continue;
        x = 2;
        if (p->ep & EP1) {
            shadeTan(p->loc[CURRENT_LOC]+1, p->loc[CURRENT_LOC+1]+1+srcSeg,
                p->tanLoc.x, p->tanLoc.y, p->texCoordLoc.x+x, p->texCoordLoc.y,
                -1.5f, -2.5f, -3.5f, -4.5f);
            x++;
        }
        if (p->ep & EP2) {
            shadeTan(p->loc[CURRENT_LOC]+2, p->loc[CURRENT_LOC+1]+1+srcSeg,
                p->tanLoc.x+1, p->tanLoc.y, p->texCoordLoc.x+x, p->texCoordLoc.y,
                2.5f, 3.5f, 4.5f, 5.5f);
            x++;
        }
        if (p->ep & EP3) {
            shadeTan(p->loc[CURRENT_LOC]+1, p->loc[CURRENT_LOC+1]+2+srcSeg,
                p->tanLoc.x, p->tanLoc.y+1, p->texCoordLoc.x+x, p->texCoordLoc.y,
                -1.5f, -2.5f, -3.5f, -4.5f);
            x++;
        }
        if (p->ep & EP4) {
            shadeTan(p->loc[CURRENT_LOC]+2, p->loc[CURRENT_LOC+1]+2+srcSeg,
                p->tanLoc.x+1, p->tanLoc.y+1, p->texCoordLoc.x+x, p->texCoordLoc.y,
                2.5f, 3.5f, 4.5f, 5.5f);
            x++;
        }
    }
    EndShade(srcBuf, dstBuf);

    glActiveTexture(GL_TEXTURE0);
}

static int
calcLimit(Patch *patchList, GPUBuffer *srcBuf, int srcSeg, GPUBuffer *dstBuf, int shader)
{
    Patch *p;
    int2 loc;
    int maxPatchSize = 0;
    int y;
    int *vptr = (int *) SubdivAttribBuffer;
    int indices = 0;
    int size;

    loc.x = loc.y = 0;

    for (p = patchList; p; p = p->next) {
        size = p->size - 2;
        if (loc.x + size > PatchBufferSize.x) {
            loc.x = 0;
            loc.y += size;
        }

        y = loc.y + size;
        p->loc[LIMIT_LOC] = loc.x;
        p->loc[LIMIT_LOC+1] = loc.y;
        if (p->ep != (EP1|EP2|EP3|EP4) || p->size != 4) {
            int srcX = p->loc[CURRENT_LOC]+1;
            int srcY = p->loc[CURRENT_LOC+1]+1+srcSeg;
            int w = size; int h = size;
            int dstX = loc.x;
            int dstY = loc.y;
            float srcX2, srcY2;
            int fb = p->flipBinormal;
            int texSampleSize = size;
            float s = (float) size;
            srcX2 = (float) p->dmapLoc[CurrentDepth].x;
            srcY2 = (float) p->dmapLoc[CurrentDepth].y;
            if (CurrentDepth > p->dmapDepth) {
                texSampleSize = (1<<(p->dmapDepth)) + 1;
                float scale = (texSampleSize - 1.0f)/(size - 1.0f);
                float offset = 0.5f - scale*0.5f;
                s = (texSampleSize - 1.0f)+scale;
                srcX2 = p->dmapLoc[p->dmapDepth].x + offset;
                srcY2 = p->dmapLoc[p->dmapDepth].y + offset;
            }

            vptr[0] = dstX;
            vptr[1] = dstY;
            vptr[2] = srcX;
            vptr[3] = srcY;
            vptr[4] = fb;
            ((float*)vptr)[5] = srcX2;
            ((float*)vptr)[6] = srcY2;
            vptr += 7;

            vptr[0] = dstX+w;
            vptr[1] = dstY;
            vptr[2] = srcX+w;
            vptr[3] = srcY;
            vptr[4] = fb;
            ((float*)vptr)[5] = srcX2+s;
            ((float*)vptr)[6] = srcY2;
            vptr += 7;

            vptr[0] = dstX+w;
            vptr[1] = dstY+h;
            vptr[2] = srcX+w;
            vptr[3] = srcY+h;
            vptr[4] = fb;
            ((float*)vptr)[5] = srcX2+s;
            ((float*)vptr)[6] = srcY2+s;
            vptr += 7;

            vptr[0] = dstX;
            vptr[1] = dstY+h;
            vptr[2] = srcX;
            vptr[3] = srcY+h;
            vptr[4] = fb;
            ((float*)vptr)[5] = srcX2;
            ((float*)vptr)[6] = srcY2+s;
            vptr += 7;

            indices += 4;
        }
        loc.x += size;
    }
    setDstGPUBuffer(dstBuf);
    set_shader(shader, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcBuf);
    glEnable(GL_FRAGMENT_PROGRAM_NV);

    EnableAttrib(0, 3, GL_INT, 7*sizeof(int), (char *) SubdivAttribBuffer + 2*sizeof(int));
    EnableAttrib(1, 2, GL_FLOAT, 7*sizeof(int), (char *) SubdivAttribBuffer + 5*sizeof(int));

    EndShadeI(srcBuf, dstBuf, (char *) SubdivAttribBuffer, indices, 7*sizeof(int), 2);

    return y;
}

static void
calcNormals(Patch *patchList, GPUBuffer *srcBuf, int srcSeg, GPUBuffer *src2Buf, GPUBuffer *dstBuf)
{
    Patch *p;
    int *vptr = (int *) SubdivAttribBuffer;
    int indices = 0;
    int size;

    for (p = patchList; p; p = p->next) {
        size = p->size - 2;
        if (p->ep != (EP1|EP2|EP3|EP4) || p->size != 4) {
            int srcX = p->loc[CURRENT_LOC]+1;
            int srcY = p->loc[CURRENT_LOC+1]+1+srcSeg;
            int w = size; int h = size;
            int dstX = p->loc[LIMIT_LOC];
            int dstY = p->loc[LIMIT_LOC+1];
            float scale = 1.0f/(size - 1.0f);
            float offset = 0.5f - scale*0.5f;
            float srcX2 = p->texCoordLoc.x + offset;
            float srcY2 = p->texCoordLoc.y + offset;
            float srcX3 = p->tanLoc.x + offset;
            float srcY3 = p->tanLoc.y + offset;
            float s = scale + 1.0f;


            vptr[0] = dstX;
            vptr[1] = dstY;
            vptr[2] = srcX;
            vptr[3] = srcY;
            ((float*)vptr)[4] = srcX2;
            ((float*)vptr)[5] = srcY2;
            ((float*)vptr)[6] = srcX3;
            ((float*)vptr)[7] = srcY3;
            vptr += 8;

            vptr[0] = dstX+w;
            vptr[1] = dstY;
            vptr[2] = srcX+w;
            vptr[3] = srcY;
            ((float*)vptr)[4] = srcX2+s;
            ((float*)vptr)[5] = srcY2;
            ((float*)vptr)[6] = srcX3+s;
            ((float*)vptr)[7] = srcY3;
            vptr += 8;

            vptr[0] = dstX+w;
            vptr[1] = dstY+h;
            vptr[2] = srcX+w;
            vptr[3] = srcY+h;
            ((float*)vptr)[4] = srcX2+s;
            ((float*)vptr)[5] = srcY2+s;
            ((float*)vptr)[6] = srcX3+s;
            ((float*)vptr)[7] = srcY3+s;
            vptr += 8;

            vptr[0] = dstX;
            vptr[1] = dstY+h;
            vptr[2] = srcX;
            vptr[3] = srcY+h;
            ((float*)vptr)[4] = srcX2;
            ((float*)vptr)[5] = srcY2+s;
            ((float*)vptr)[6] = srcX3;
            ((float*)vptr)[7] = srcY3+s;
            vptr += 8;

            indices += 4;
        }
    }
    setDstGPUBuffer(dstBuf);
    set_shader(NORMAL_SHADER, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcBuf);
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 3);
    setSrcGPUBuffer2(src2Buf);
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_FRAGMENT_PROGRAM_NV);

  
    EnableAttrib(0, 2, GL_INT, 8*sizeof(int), (char *) SubdivAttribBuffer + 2*sizeof(int));
    EnableAttrib(1, 4, GL_FLOAT, 8*sizeof(int), (char *) SubdivAttribBuffer + 4*sizeof(int));

    EndShadeI(srcBuf, dstBuf, (char *) SubdivAttribBuffer, indices, 8*sizeof(int), 2);
}

static void
LoadShaders()
{
    createShader(SUBDIV_SHADER, "Subdiv");
    createShader(TANGENT_SHADER, "Tangent");
    createShader(EPTANGENT_SHADER, "EPTangent");
    createShader(LIMIT_SHADER, "Limit");
    createShader(NORMAL_SHADER, "Normal");
    createShader(EPSUBDIV_SHADER, "EPSubdiv");
    createShader(FLATTEST_SHADER, "FlatTest");
    createShader(FLATTEST2_SHADER, "FlatTest2");
    createShader(EPFLATTEST_SHADER, "EPFlatTest");
    createShader(EPFLATTEST2_SHADER, "EPFlatTest2");
    createShader(EPLIMIT_SHADER, "EPLimit");
    createShader(EPNORMAL_SHADER, "EPNormal");
    createShader(CREATEPATCH_SHADER, "CreatePatch");
    createShader(DFLATTEST_SHADER, "DFlatTest");
    createShader(DFLATTEST2_SHADER, "DFlatTest2");
    createShader(DEPFLATTEST_SHADER, "DEPFlatTest");
    createShader(DEPFLATTEST2_SHADER, "DEPFlatTest2");
    createShader(DLIMIT_SHADER, "DLimit");
    createShader(DEPLIMIT_SHADER, "DEPLimit");
}

void
SetShaderParameters(GPUBuffer *flatTestBuf, GPUBuffer *outBuf,
                    int flattest_shader, int flattest2_shader,
                    int epflattest_shader, int epflattest2_shader)
{
    float4 flatInfo, dmapInfo;

    flatInfo.x = _subd_flat_scale2;
    flatInfo.y = _subd_near_len2;
    flatInfo.z = _subd_near_len2 * 2;
    dmapInfo.x = _subd_dmap_scale * (1.0f/128.0f);
    dmapInfo.y = _subd_dmap_scale1;
    dmapInfo.z = _subd_dmap_scale2;
    dmapInfo.w = _subd_dmap_scale_x_2;
    wglMakeCurrent(flatTestBuf->hdc, flatTestBuf->hglrc);
    set_shader_parameter(flattest2_shader, "flatScale", &flatInfo.x);
    set_shader_parameter(epflattest2_shader, "flatScale", &flatInfo.x);
    set_shader_parameter(DFLATTEST_SHADER, "dmapScale", &dmapInfo.x);
    set_shader_parameter(DFLATTEST2_SHADER, "dmapScale", &dmapInfo.x);
    set_shader_parameter(DEPFLATTEST_SHADER, "dmapScale", &dmapInfo.x);
    set_shader_parameter(DEPFLATTEST2_SHADER, "dmapScale", &dmapInfo.x);
    flatInfo.x = _subd_flat_scale;
    flatInfo.y = _subd_near_len1;
    flatInfo.z = _subd_near_len1 * 2;
    set_shader_parameter(flattest_shader, "flatScale", &flatInfo.x);
    set_shader_parameter(epflattest_shader, "flatScale", &flatInfo.x);
    wglMakeCurrent(outBuf->hdc, outBuf->hglrc);
    set_shader_parameter(DLIMIT_SHADER, "dmapScale", &dmapInfo.x);
    set_shader_parameter(DEPLIMIT_SHADER, "dmapScale", &dmapInfo.x);
}

static void
init() {
    QUERY_EXTENSION_ENTRY_POINT(wglAllocateMemoryNV,
        PFNWGLALLOCATEMEMORYNVPROC);
    QUERY_EXTENSION_ENTRY_POINT(wglFreeMemoryNV, PFNWGLFREEMEMORYNVPROC);
    QUERY_EXTENSION_ENTRY_POINT(glVertexArrayRangeNV,
        PFNGLVERTEXARRAYRANGENVPROC);
    QUERY_EXTENSION_ENTRY_POINT(glFlushVertexArrayRangeNV,
        PFNGLFLUSHVERTEXARRAYRANGENVPROC);

#if VERTEX_ARRAY_RANGE
    vertexBuffer = (int *) wglAllocateMemoryNV(65536*2, 0.0f, 0.0f, 1.0f);
    glVertexArrayRangeNV(65536, vertexBuffer);
#else
    vertexBuffer = (int *) malloc(65536*2);
#endif
}

void tess_GPU(
	Patch *patchList, pvector *vlist, int vertices,
	unsigned char *patchIndexBuffer,
    int patchIndexH, float4 *texCoordBuffer, int texCoordW, int texCoordH,
    uchar *dmapTexture, int2 dmapTexSize, uchar *dmapMaxTexture,
    int2 dmapMaxTexSize, SubdEnum state)
{
    int i;
    static GPUBuffer *Buf0, *Buf1;
    static GPUBuffer *OutBuf;
    static GPUBuffer *buf[8];
    static Patch *toDo[8];
    static int segment[8];
    static int numToDoLists = 7;
    static GPUBuffer *FlatTestBuf;
    static GPUBuffer *TanBuf;
    Patch *p;
    static Patch *flatList;
    static Patch *patchesToDo;
    static int initialized;
    int vertexYOffset;
    HDC hdc = wglGetCurrentDC();
	HGLRC hglrc = wglGetCurrentContext();
    static int y;
    int vertexBufferHeight;

    vertexBufferHeight = _subd_vertex_list_size / (PatchBufferSize.x * sizeof(float4));
    if (!initialized) {
        float *weightTable;

        Buf0 = createGPUBuffer(PatchBufferSize.x,
                PatchBufferSize.y+32, 32, 4);
        Buf1 = createGPUBuffer(PatchBufferSize.x,
                PatchBufferSize.y+32, 32, 4);
        TanBuf = createGPUBuffer(PatchBufferSize.x, SEGMENT_SIZE, 32, 4);
        OutBuf = createGPUBuffer(PatchBufferSize.x, SEGMENT_SIZE, 32, 4);     
        FlatTestBuf = createGPUBuffer(PatchBufferSize.x, SEGMENT_SIZE, 8, 4);
        LoadShaders();

        weightTable = BuildWeightTables();

        writeToGPUBuffer(Buf0, (float4 *) &weightTable[0],
                0, PatchBufferSize.y, WWIDTH>>2, (MAX_VALENCE-3)*WHEIGHT);
        writeToGPUBuffer(Buf1, (float4 *) &weightTable[0],
                0, PatchBufferSize.y, WWIDTH>>2, (MAX_VALENCE-3)*WHEIGHT);

        weightTable = BuildLimitTables();
        writeToGPUBuffer(Buf0, (float4 *) &weightTable[0],
                (WWIDTH>>2), PatchBufferSize.y, 8, (MAX_VALENCE-3)*3);
        writeToGPUBuffer(Buf1, (float4 *) &weightTable[0],
                (WWIDTH>>2), PatchBufferSize.y, 8, (MAX_VALENCE-3)*3);

        weightTable = BuildTangentTables();
        writeToGPUBuffer(Buf0, (float4 *) &weightTable[0],
                (WWIDTH>>2) + 8, PatchBufferSize.y, 8, (MAX_VALENCE-3)*3);
        writeToGPUBuffer(Buf1, (float4 *) &weightTable[0],
                (WWIDTH>>2) + 8, PatchBufferSize.y, 8, (MAX_VALENCE-3)*3);
        FlatBuffer[0] = (byte4 *) malloc(PatchBufferSize.x*SEGMENT_SIZE*(MAXDEPTH+1)*sizeof(byte4));
        for (i = 0; i < MAXDEPTH; i++)
            FlatBuffer[i+1] = FlatBuffer[i] + PatchBufferSize.x * SEGMENT_SIZE;
        buf[0] = buf[2] = buf[4] = buf[6] = Buf0;
        buf[1] = buf[3] = buf[5] = buf[7] = Buf1;
        segment[0] = segment[7] = 0;
        segment[2] = segment[3] = SEGMENT_SIZE;
        segment[4] = segment[5] = SEGMENT_SIZE*2;
        segment[6] = segment[1] = SEGMENT_SIZE*3;

        setDstGPUBuffer(buf[0]);
        glActiveTexture(GL_TEXTURE3);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 13);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
            256, 128, 0, GL_RGBA,
            GL_FLOAT, (GLvoid *) 0);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);

        init(); // other initialization

        initialized = 1;
    }
    if (!vlist) {
        if (perfTestFlag)
         readFromGPUBuffer(OutBuf,
                    (float4*) _subd_vertex_list[1], 0, 0,
                    PatchBufferSize.x, 32, _subd_vertex_bufobj[0]);
        setSrcGPUBuffer(0);
        setSrcGPUBuffer2(0);
        setDstGPUBuffer(0);
        wglMakeCurrent(hdc, hglrc);
        return;
    }

    if (_subd_dmap_scale == 0.0f) {
        flattest_shader = FLATTEST_SHADER;
        epflattest_shader = EPFLATTEST_SHADER;
        flattest2_shader = FLATTEST2_SHADER;
        epflattest2_shader = EPFLATTEST2_SHADER;
        Limit_Shader = LIMIT_SHADER;
        EPLimit_Shader = EPLIMIT_SHADER;
    }
    else {
        flattest_shader = DFLATTEST_SHADER;
        epflattest_shader = DEPFLATTEST_SHADER;
        flattest2_shader = DFLATTEST2_SHADER;
        epflattest2_shader = DEPFLATTEST2_SHADER;
        Limit_Shader = DLIMIT_SHADER;
        EPLimit_Shader = DEPLIMIT_SHADER;
    }
    if (!perfTestFlag)
    SetShaderParameters(FlatTestBuf, OutBuf, flattest_shader, flattest2_shader,
        epflattest_shader, epflattest2_shader);

    if (patchIndexBuffer) {
        setDstGPUBuffer(buf[0]);
        glActiveTexture(GL_TEXTURE1);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 10);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_LUMINANCE_ALPHA,
            PATCH_BUFFER_WIDTH, patchIndexH, 0, GL_LUMINANCE_ALPHA,
            GL_UNSIGNED_BYTE, (GLvoid *) patchIndexBuffer);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);
    }

    if (texCoordBuffer) {
        setDstGPUBuffer(OutBuf);
        glActiveTexture(GL_TEXTURE2);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 11);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
            texCoordW, texCoordH, 0, GL_RGBA,
            GL_FLOAT, (GLvoid *) texCoordBuffer);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);

        setDstGPUBuffer(TanBuf);
        glActiveTexture(GL_TEXTURE2);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 11);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);
    }

    if (dmapTexture) {
        setDstGPUBuffer(OutBuf);
        glActiveTexture(GL_TEXTURE3);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 12);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_LUMINANCE,
            dmapTexSize.x, dmapTexSize.y, 0, GL_LUMINANCE,
            GL_BYTE, (GLvoid *) dmapTexture);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);
    }

    if (dmapMaxTexture) {
        setDstGPUBuffer(FlatTestBuf);
        glActiveTexture(GL_TEXTURE1);
	    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 14);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA,
            dmapMaxTexSize.x, dmapMaxTexSize.y, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, (GLvoid *) dmapMaxTexture);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glActiveTexture(GL_TEXTURE0);
    }

    if (state == SUBD_START) {
        if (!perfTestFlag) {
        setDstGPUBuffer(buf[0]);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, 13);
        if (vertices >> 8)
            glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 256, vertices>>8, 
                GL_RGBA, GL_FLOAT, vlist);
        if (vertices & 255)
            glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, vertices>>8, vertices & 255, 1,
                GL_RGBA, GL_FLOAT, vlist+256*(vertices>>8));
        glActiveTexture(GL_TEXTURE0);
        }
    
        // Set next pointer to create list for tessellation

        for (p = patchList; p; p = p->nextPatch) {
            p->next = p->nextPatch;
        }

        for (i = 0; i < numToDoLists; i++)
            toDo[i] = NULL;
        patchesToDo = patchList;
        flatList = NULL;
    }

    vertexYOffset = 0;
    for (;;) {
        if (flatList) {     // continuing tessellation
            i = CurrentDepth;
        }
        else {
            // find a list of patches to work on
            for (i = numToDoLists-1; i >= 0 && !toDo[i]; i--)
                ;

            if (i < 0) {   // No patches in patch buffers
                if (!patchesToDo)
                    break;          // all done
                toDo[0] = patchesToDo;
                patchesToDo = createPatches(patchesToDo, NULL, buf[0], segment[0]);
                if (0)
                calcTangents(toDo[0], buf[0], segment[0], TanBuf, 0);
                i = 0;
            }
            CurrentDepth = i;

            if (_subd_adaptive_flag && i < _subd_subdiv_level) {
                Patch **prevPtr;
                // test for flatness
                p = toDo[i];
                y = TestFlatness(p, buf[i], segment[i], FlatTestBuf,
                    i == 0 ? flattest2_shader : flattest_shader);
                epFlatTest(p, buf[i], segment[i], FlatTestBuf, 0,
                    i == 0 ? epflattest2_shader : epflattest_shader,
                    CURRENT_LOC, FLATINFO_LOC);
                readFromGPUBuffer(FlatTestBuf, (float4 *) FlatBuffer[i],
                        0, 0, PatchBufferSize.x, y, 0);
                // remove flat patches from "to do" list and put them on flatList
                prevPtr = &toDo[i];
                for (; p; p = *prevPtr) {
                    p->flatPtr[i] = FlatBuffer[i] + p->loc[FLATINFO_LOC] +
                        p->loc[FLATINFO_LOC+1]*PatchBufferSize.x;
                    if (patchIsFlat(p, i)) {
                        *prevPtr = p->next;
                        p->next = flatList;
                        flatList = p;
                    }
                    else
                        prevPtr = &p->next;
                }
            }
            else if (i == _subd_subdiv_level) {
                flatList = toDo[i];
                toDo[i] = 0;
            }
        }

        // write out vertex data of patch on flat patch list
        if (flatList) {
            if (state != SUBD_CONTINUE) {
                y = calcLimit(flatList, buf[i], segment[i], OutBuf, Limit_Shader);
                FlipBinormal = 1;
                if (0)
                calcEPValues(flatList, buf[i], segment[i], OutBuf, 0, EPLimit_Shader, 0, LIMIT_LOC,
                        WWIDTH>>2, PatchBufferSize.y, NULL);
                FlipBinormal = 0;
            }
            state = SUBD_DONE;      // anything but SUBD_CONTINUE
            if (vertexYOffset + y > vertexBufferHeight) {
                setSrcGPUBuffer(0);
                setSrcGPUBuffer2(0);
                setDstGPUBuffer(0);
                wglMakeCurrent(hdc, hglrc);
                _subd_stop_tessellating = true;
                _subd_vertex_ptr = _subd_vertex_list[0] + vertexYOffset*PatchBufferSize.x*4;
                return;
            }
            if (!perfTestFlag)
            readFromGPUBuffer(OutBuf, 
                    (float4*) _subd_vertex_list[0] + vertexYOffset*PatchBufferSize.x, 0, 0,
                    PatchBufferSize.x, y, _subd_vertex_bufobj[0]);
            if (!perfTestFlag)
            calcNormals(flatList, buf[i], segment[i], TanBuf, OutBuf);
            if (0)
            calcEPValues(flatList, buf[i], segment[i], OutBuf, 0, EPNORMAL_SHADER, 0, LIMIT_LOC,
                    (WWIDTH>>2) + 8, PatchBufferSize.y, TanBuf);
            if (!perfTestFlag)
            readFromGPUBuffer(OutBuf,
                    (float4*) _subd_vertex_list[1] + vertexYOffset*PatchBufferSize.x, 0, 0,
                    PatchBufferSize.x, y, _subd_vertex_bufobj[1]);
            if (!perfTestFlag)
            tessPatches(flatList, i, vertexYOffset);
            vertexYOffset += y;
            flatList = NULL;
        }

        if (toDo[i]) {        // subdivide patches on to do list
            p = toDo[i];
            toDo[i+1] = p;
            toDo[i] = SubdividePatches(p, buf[i], segment[i], buf[i+1], segment[i+1]);
            subdivEPoints(p, buf[i], segment[i], buf[i+1], segment[i+1], LAST_LOC, 0);
        }
    }
done:
    setSrcGPUBuffer(0);
    setSrcGPUBuffer2(0);
    setDstGPUBuffer(0);
    if (perfTestFlag)
    glFinish();
    wglMakeCurrent(hdc, hglrc);
    _subd_vertex_ptr = _subd_vertex_list[0] + vertexYOffset*PatchBufferSize.x*4;
}
