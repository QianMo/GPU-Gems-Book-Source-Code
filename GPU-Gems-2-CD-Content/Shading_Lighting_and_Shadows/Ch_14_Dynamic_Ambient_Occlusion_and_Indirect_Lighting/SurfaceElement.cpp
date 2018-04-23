//
//  SurfaceElement.cpp
//
//      Most of this code is used to create a hierarchical list of surface elements used
//      by the GPU to calculate the ambient occlusion or indirect lighting results.
//      The polygon mesh is define using seBeginMesh(), seFace(), seVertex(), seEndMesh()
//
//      CreateElements() is used to create an element hierarchy after the mesh is defined
//      it uses texture coordinates to guide the creation of the hierarchy - so meshes
//      must be uv mapped for it to work properly
//
//      updateElements() is used to change the position, normal, and area information
//      for each surface element.
//     
//      calcAmbientOcclusion() calls updateElements(), writes the element data to texture maps
//      and renders the ambient occlusion results using fragment programs (pixel shaders).
//      It writes the ambient occlusion result for each vertex into the vertex buffer for
//      subdivision and then display
//

#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <tchar.h>
#include <malloc.h>
#include <math.h>
#include "vmath.h"
#include <GL/gl.h>
#include "glext.h"
#include "wglext.h"
#include "SurfaceElement.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern int createShader(int shaderID, char *shaderName);
extern void BeginShade(void *srcgp, void *dstgp, int shaderID);
extern void EndShade(void *srcgp, void *dstgp);

typedef struct vertex {
    float3 *position;
    float2 uv;
    float area;
    float3 normal;
    struct vertex *next;
    int uvSegment;
    float3 ambocc[4];
} vertex;

typedef struct edge {
    vertex *v;
    vertex *vt;
    int index;
    int tindex;
    struct edge *next;
} edge;

typedef struct face {
    edge *e;
    float area;
    float3 normal;
    int vertices;
    struct face *next;
} face;

typedef struct SurfaceElement {
    float3 position;
    float3 normal;
    float  area;
    int height;
    int bufferIndex;
    float   uv[2];
    vertex *v;
    struct SurfaceElement *next;
    struct SurfaceElement *child, *right;
} SurfaceElement;

static SurfaceElement *seList;
static int seHeight;
static face *first_face;
static vertex *first_vertex;
static vertex *first_tcoord;
static float3 vzero;
static edge *LastEdge;
static face *LastFace;
static int MaxTCoordIndex, MaxVertexIndex;

static float distance(float3 *p1, float3 *p2)
{
    float3 v;

    vsub(v, *p1, *p2);
    return (float) sqrt(dot(v, v));
}

static float areaOfTriangle(float3 *p1, float3 *p2, float3 *p3)
{
    float halfp;
    float a, b, c;

    a = distance(p1, p2);
    b = distance(p2, p3);
    c = distance(p3, p1);

    halfp = (a + b + c) * 0.5f;

    return (float) sqrt(halfp * (halfp - a) * (halfp - b) * (halfp - c));
}

static int
normalize(float3 *p)
{
    float len;

	len = dot(*p, *p);
	if (len < 0.000000000001f)
		return 1;

    len = 1.0f / ((float) sqrt(len));

    p->x *= len;
    p->y *= len;
    p->z *= len;

	return 0;
}

static int
calc_normal(float3 *d, float3 *p0, float3 *p1, float3 *p2)
{
    float3 v1, v2;
    float3 n;

	vsub(v1, *p0, *p1);
	vsub(v2, *p1, *p2);
	cross(n, v1, v2);

	*d = n;

    return normalize(d);
}

// calculate an area value and normal for each vertex in the mesh 

static void
calcAreaAndNormal(face *first_face, vertex *first_vertex)
{
    face *f;
    edge *e;
    vertex *v;

    for (f = first_face; f; f = f->next) {
        f->area = 0.0f;
        for (e = f->e->next; e->next != f->e; e = e->next) {
            f->area += areaOfTriangle(f->e->v->position, e->v->position,
                    e->next->v->position);
        }
        e = f->e;
        (void) calc_normal(&f->normal, e->v->position, e->next->v->position, 
                           e->next->next->v->position);
    }

    // calc normal at vertices

    for (v = first_vertex; v; v = v->next) {
        v->normal = vzero;
        v->area = 0.0f;
    }
    for (f = first_face; f; f = f->next) {
        vadd(f->e->v->normal, f->normal);
        f->e->v->area += f->area / f->vertices;
        for (e = f->e->next; e != f->e; e = e->next) {
            vadd(e->v->normal, f->normal);
            e->v->area += f->area / f->vertices;
        }
    }
    for (v = first_vertex; v; v = v->next) {
        normalize(&v->normal);
    }
}

// This routine just calculates the normal at each vertex
static void
calcNormals(face *first_face, vertex *first_vertex)
{
    face *f;
    edge *e;
    vertex *v;

    for (f = first_face; f; f = f->next) {
        e = f->e;
        (void) calc_normal(&f->normal, e->v->position, e->next->v->position, 
                           e->next->next->v->position);
    }

    // calc normal at vertices

    for (v = first_vertex; v; v = v->next)
        v->normal = vzero;

    for (f = first_face; f; f = f->next) {
        vadd(f->e->v->normal, f->normal);
        for (e = f->e->next; e != f->e; e = e->next)
            vadd(e->v->normal, f->normal);
    }
    for (v = first_vertex; v; v = v->next)
        normalize(&v->normal);
}


void
deleteAllElements()
{
    SurfaceElement *se, *seNext;

    for (se = seList; se; se = seNext) {
        seNext = se->child ? se->child : se->next;
        free(se);
    }
    seList = 0;
}

// figure out how many separate uv mapped meshes there are using
// texture coordinate information - assign a different number to each uv mesh
// and save the number for each vertex
// Note that vertices with more than one texture coordinate are arbitrarily assigned
// to one of the uv meshes they are in

static int setUVSegments()
{
    int uvSegment;
    int change, selected;
    vertex *v;
    edge *e;
    vertex *vt;
    face *f;

    for (v = first_vertex; v; v = v->next) {
        v->uvSegment = 0; // no segment
    }
    uvSegment = 0;
    for (vt = first_tcoord; vt; vt = vt->next) {
        vt->uvSegment = 0;
    }
    for (vt = first_tcoord; vt; vt = vt->next) {
        if (vt->uvSegment != 0)
            continue;
        vt->uvSegment = ++uvSegment;

        do {
            change = 0;
            for (f = first_face; f; f = f->next) {
                selected = 0;
                for (e = f->e; ; e = e->next) {
                    if (e->vt && e->vt->uvSegment == uvSegment) {
                        selected = 1;
                    }
                    if (e->next == f->e)
                        break;
                }
                if (!selected)
                    continue;
                for (e = f->e; ; e = e->next) {
                    if (e->vt && (e->vt->uvSegment == 0 || e->v->uvSegment == 0)) {
                        change = 1;
                        e->vt->uvSegment = uvSegment;
                        e->v->uvSegment = uvSegment;
                        e->v->uv.x = e->vt->position->x;
                        e->v->uv.y = e->vt->position->y;
                    }
                    if (e->next == f->e)
                        break;
                }
            }
        } while (change);
    }
    return uvSegment;
}

// change a binary tree into a tree into quadinary (up to 4 children per parent)
static void
combineNodes(SurfaceElement *se)
{
    int i, j;
    SurfaceElement *child[5];

    if (!se)
        return;
    i = 0;
    if (se->child) {
        if (!se->child->child && !se->child->right) // leaf
            child[i++] = se->child;
        else {
            if (se->child->child)
                child[i++] = se->child->child;
            if (se->child->right)
                child[i++] = se->child->right;
            free(se->child);
        }
    }
    if (se->right) {
        if (!se->right->child && !se->right->right) // leaf
            child[i++] = se->right;
        else {
            if (se->right->child)
                child[i++] = se->right->child;
            if (se->right->right)
                child[i++] = se->right->right;
            free(se->right);
        }
    }
    child[i] = 0;

    for (j = 0; j < i; j++) {
        child[j]->next = child[j+1];
        combineNodes(child[j]);
    }
    se->child = child[0];
    se->right = 0;
}

// traverse all children and set the pointers that will be used to create
// indices that the shader program uses for traversal
static void
setPointers(SurfaceElement *se, SurfaceElement *next)
{
    for (;;) {
        if (se->child) {
            setPointers(se->child, se->next ? se->next : next);
        }
        if (!se->next) {
            se->next = next;
            break;
        }
        se = se->next;
    }
}

// create a surface element for each vertex in the polygon model
SurfaceElement *
createElementList(int uvSegment)
{
    vertex *v;
    SurfaceElement *firstse, *lastse, *se;

    firstse = lastse = 0;
    for (v = first_vertex; v; v = v->next) {
        if (v->uvSegment != uvSegment)
            continue;
        se = (SurfaceElement *) malloc(sizeof(*se));
        se->position = *v->position;
        se->area = v->area;
        se->normal = v->normal;
        se->v = v;
        se->uv[0] = v->uv.x;
        se->uv[1] = v->uv.y;
        se->next = 0;
        se->child = se->right = 0;
        if (lastse)
            lastse->next = se;
        else
            firstse = se;
        lastse = se;
    }
    return firstse;
}

// calculate a bounding box in texture coordate space for creating the element hierarchy

static void
calcBoundingBox(float *x1, float *y1, float *x2, float *y2, SurfaceElement *se)
{
    *x1 = *x2 = se->uv[0];
    *y1 = *y2 = se->uv[1];

    for (; se; se = se->next) {
        if (se->uv[0] < *x1)
            *x1 = se->uv[0];
        else if (se->uv[0] > *x2)
            *x2 = se->uv[0];
        if (se->uv[1] < *y1)
            *y1 = se->uv[1];
        else if (se->uv[1] > *y2)
            *y2 = se->uv[1];

    }
}

// This routine is used to split surface elements in two groups based on
// texture coordinate location. It is used recursively to create a binary tree of
// elements
static float
findMedian(SurfaceElement *se, int uOrV, float a, float b, int total)
{
    int i;
    int half = total >> 1;
    int half2 = half | total & 1;
    float m;
    int count;
    SurfaceElement *e;

    for (i = 0; i < 20; i++) {
        m = (a + b) * 0.5f;
        count = 0;
        for (e = se; e; e = e->next)
            if (e->uv[uOrV] < m)
                count++;
        if (count == half || count == half2)
            break;
        if (count < half)
            a = m;
        else
            b = m;

    }
    return m;
}

// convert a list of surface elements into a binary tree based on uv location
static SurfaceElement *
convertListToTree(SurfaceElement *se)
{
    int n;
    int d;
    SurfaceElement *se2, *seNext;
    SurfaceElement *l1, *l2;
    float m;
    float x1, y1, x2, y2;

    if (se == 0 || !se->next)
        return se; // leaf node
    n = 0;
    for (se2 = se; se2; se2 = se2->next)
        n++;
    calcBoundingBox(&x1, &y1, &x2, &y2, se);
    d = 0;
    if (x2 - x1 < y2 - y1) {
        d = 1;
        m = findMedian(se, d, y1, y2, n);
    }
    else
        m = findMedian(se, d, x1, x2, n);

    l1 = 0;
    l2 = 0;
    for (se2 = se; se2; se2 = seNext) {
        seNext = se2->next;
        if (se2->uv[d] < m) {
            se2->next = l1;
            l1 = se2;
        }
        else {
            se2->next = l2;
            l2 = se2;
        }
    }
    if (!l1) {
        l1 = l2;
        l2 = l2->next;
        l1->next = 0;
    }
    else if (!l2) {
        l2 = l1;
        l1 = l1->next;
        l2->next = 0;
    }
    se = (SurfaceElement *) malloc(sizeof(*se));
    se->child = convertListToTree(l1);
    se->right = convertListToTree(l2);
    se->next = 0;
    se->v = 0;
    return se;
}

// create the surface elements and put them in a hierarchy for quick traversal
void
CreateElements()
{
    SurfaceElement *se, *e;
    int uvSegments;
    SurfaceElement *lastse, *firstse;
    int i;
    int bufferIndex;
    int done;

    deleteAllElements();
    calcAreaAndNormal(first_face, first_vertex);
    uvSegments = setUVSegments();
    printf("%d segments\n", uvSegments);
    lastse = 0;
    for (i = 0; i < uvSegments; i++) {
        se = createElementList(i+1);
        se = convertListToTree(se);
       // printTree(se, 0);
        combineNodes(se);
      //  printCombinedTree(se, 0);
        if (lastse)
            lastse->next = se;
        else
            firstse = se;
        lastse = se;
    }
    // add un-uvmapped verts
    se = createElementList(0);
    if (se) {
        if (lastse)
            lastse->next = se;
        else
            firstse = se;
    }
    
    // assign level #s

    seList = firstse;
    setPointers(firstse, 0);
     // assign level #s
    bufferIndex = 0;
    if (seList->child)
        bufferIndex = 1;
    for (se = seList; se; se = se->child ? se->child : se->next) {
        if (se->child)
            se->height = -1;
        else {
            se->height = 0;
            se->bufferIndex = bufferIndex++;
        }
    }
    done = 0;
    for (i = 1; !done; i++) {
        done = 1;
        for (se = seList; se; se = se->child ? se->child : se->next) {
            if (se->height >= 0)
                continue;
            done = 0;
            for (e = se->child; e && e != se->next && e->height >= 0; e = e->next)
                ;
            if (!e || e == se->next) { // all children assigned
                se->height = i;
                if (se == seList)
                    se->bufferIndex = 0;
                else
                    se->bufferIndex = bufferIndex++;
            }
        }
    }
    seHeight = i;
}

// updateElements() recalculates the position, area, and normal value for each surface
// element
static void
updateElements()
{
    SurfaceElement *se, *se2;
    vertex *v;
    int notDone;

    //calcAreaAndNormal(first_face, first_vertex);
    calcNormals(first_face, first_vertex);  // use the precalcuated area for leaf elements

    for (se = seList; se; se = se->child ? se->child : se->next) {
        v = se->v;
        if (!v) {   // non-leaf - could check for se->child
            se->area = -1.0f;
            continue;
        }
        se->position = *v->position;
        se->area = v->area/(float) M_PI;
        se->normal = v->normal;
    }
    do {
        notDone = 0;
        for (se = seList; se; se = se->child ? se->child : se->next) {
            float area;
            int ready;
            int numChildren = 1;
            if (se->area >= 0.0f)
                continue;
            if (!se->child)
                continue;
            if (se->child->area < 0.0f) {
                notDone = 1;
                continue;
            }
            ready = 1;
            area = se->child->area;
            se->position = se->child->position;
            se->normal = se->child->normal;
            for (se2 = se->child->next; se2 && se2 != se->next; se2 = se2->next) {
                numChildren++;
                if (se2->area < 0.0f) {
                    ready = 0;
                    notDone = 1;
                    break;
                }
                area += se2->area;
                vadd(se->position, se2->position);
                vadd(se->normal, se2->normal);
            }
            if (!ready)
                continue;

            se->area = area;
            normalize(&se->normal);
            vmuls(se->position, 1.0f/numChildren);

        }
    } while (notDone);
}

// start a mesh definition
void
seBeginMesh()
{
    face *f, *fnext;
    edge *e, *enext;

    // delete existing mesh data
    
    for (f = first_face; f; f = fnext) {
        fnext = f->next;
        e = f->e;
        do {
            enext = e->next;
            free((void *) e);
            e = enext;
        } while (e != f->e);
        free((void *) f);
    }

    first_face = NULL;
    LastFace = NULL;

    MaxVertexIndex = 0;
    MaxTCoordIndex = 0;

}

// add a face to the mesh definition
void
seFace()
{
    face *f = (face *) calloc(sizeof(face), 1);
    if (LastFace)
        LastFace->next = f;
    else
       first_face = f;

    LastFace = f;
}

// add an edge to the mesh definition
void
seVertex(int vindex, int vtindex)
{
    face *f;
    edge *e;

    f = LastFace;
    if (!f)
        return; // Error

    f->vertices++;
    e = (edge *) calloc(sizeof(edge), 1);
    e->index = vindex;
    e->tindex = vtindex;
    if (vindex > MaxVertexIndex)
        MaxVertexIndex = vindex;
    if (vtindex > MaxTCoordIndex)
        MaxTCoordIndex = vtindex;

    if (!f->e)
        f->e = e;
    else
        LastEdge->next = e;

    LastEdge = e;
    e->next = f->e;
}

// create the vertices for the mesh definition and assign their position pointers and
// uv coordinates
void
seEndMesh(float3 *vertexArray, float2 *tcArray)
{
    int i;
    face *f;
    edge *e;

    if (first_vertex)
        free((void *) first_vertex);
    first_vertex = (vertex *) calloc(sizeof(vertex)*(MaxVertexIndex+1), 1);
    for (i = 0; i < MaxVertexIndex; i++) {
        first_vertex[i].next = first_vertex + i + 1;
        first_vertex[i].position = vertexArray + i*2;
    }
    first_vertex[i].position = vertexArray + i*2;
    
    if (first_tcoord)
        free((void *) first_tcoord);
    first_tcoord = (vertex *) calloc(sizeof(vertex)*(MaxTCoordIndex+1), 1);
    for (i = 0; i < MaxTCoordIndex; i++) {
        first_tcoord[i].next = first_tcoord + i + 1;
        first_tcoord[i].uv = tcArray[i];
        first_tcoord[i].position = (float3 *) (tcArray + i);
    }
    first_tcoord[i].uv = tcArray[i];
    first_tcoord[i].position = (float3 *) (tcArray + i);

    for (f = first_face; f; f = f->next) {
        e = f->e;
        do {
            e->v = first_vertex + e->index;
            e->vt = first_tcoord + e->tindex;
            e = e->next;
        } while (e != f->e);
    }
}

// GPUBuffers are just pbuffers with some extra information
// This data structure and the following routines are used by calcAmbientOcclusion()
// which uses fragment programs to calculate the ambient occlusion results

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
BeginShade(GPUBuffer *srcgp, GPUBuffer *dstgp, int shaderID)
{
    setDstGPUBuffer(dstgp);
    set_shader(shaderID, GL_FRAGMENT_PROGRAM_NV);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 1);
    setSrcGPUBuffer(srcgp);
    glEnable(GL_FRAGMENT_PROGRAM_NV);
    glBegin(GL_QUADS);
}

void
EndShade(GPUBuffer *srcgp, GPUBuffer *dstgp)
{
    glEnd();
    glDisable(GL_TEXTURE_RECTANGLE_NV);
    glDisable(GL_FRAGMENT_PROGRAM_NV);
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
readFromGPUBuffer(GPUBuffer *gp, float4 *dst, int x, int y, int w, int h)
{
    setDstGPUBuffer(gp);
    glReadPixels(x, y, w, h, GL_RGBA, gp->type, (GLvoid *) dst);
}
static void wglGetLastError()
{
}


#define MAX_ATTRIBS     256
#define MAX_PFORMATS    256

static PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
static PFNWGLCREATEPBUFFERARBPROC wglCreatePbufferARB;
static PFNWGLGETPBUFFERDCARBPROC wglGetPbufferDCARB;
static PFNWGLQUERYPBUFFERARBPROC wglQueryPbufferARB;

#define GET_PROC_ADDRESS wglGetProcAddress
#define QUERY_EXTENSION_ENTRY_POINT(name, type)               \
    name = (type)GET_PROC_ADDRESS(#name);

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

        QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4i,
            PFNGLMULTITEXCOORD4IARBPROC);
        QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4f,
            PFNGLMULTITEXCOORD4FARBPROC);
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

     wglMakeCurrent(gp->hdc, gp->hglrc);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
     glOrtho(0.0, (double) gp->width, 0.0, (double) gp->height, -1.0, 1.0);

     wglMakeCurrent(hdc, hglrc);

    return gp;
}

// calcAmbientOcclusion() writes the surface element data - include the indices used
// for traversal to texture maps and renders a quad for each pass of the algorithm
// to calculate the ambient occlusion results - it copies the data into the vertex buffer
// for rendering

// This routine is not very optimal. It reloads a lot of information that does not change
// and it uses glTexImage() when it should use glTexSubImage()

void calcAmbientOcclusion(int passes)
{
    int i, leafVerts;
    static float4 emitterBuffer[1024*64];
    float4 *ebuf;
    static float4 resBuf[1024*64];
    static float4 elementIndex[1024*64];
    int w, h;
    static GPUBuffer *obuf0, *obuf1;
    HDC hdc = wglGetCurrentDC();
	HGLRC hglrc = wglGetCurrentContext();
    SurfaceElement *e;
    int n;
    int totalverts;
    int h2;
 
    updateElements();
    totalverts = 0;
    for (leafVerts = 0, e = seList; e; e = e->child ? e->child : e->next) {
        if (!e->child)
            leafVerts++;
        totalverts++;
    }
    w = 1024;
    h = (leafVerts + (w-1))/w;
    h2 = (totalverts + (w-1))/w;
    ebuf = emitterBuffer + w*h2;

    for (e = seList; e; e = e->child ? e->child : e->next) {
        i = e->bufferIndex;
        emitterBuffer[i].x = e->position.x ;
        emitterBuffer[i].y = e->position.y;
        emitterBuffer[i].z = e->position.z;
        emitterBuffer[i].w = 0.0f;

        ebuf[i].x = e->normal.x;
        ebuf[i].y = e->normal.y;
        ebuf[i].z = e->normal.z;
        ebuf[i].w = e->area;
        if (e->next) {
            n = e->next->bufferIndex;
            elementIndex[i].x = (float) (n & 1023) + 0.5f;
            elementIndex[i].y = (float) (n >> 10) + 0.5f;
        }
        else {
            elementIndex[i].x = 0.0f;
            elementIndex[i].y = 0.0f;
        }

        if (e->child) {
            ebuf[i].w = -ebuf[i].w;
            n = e->child->bufferIndex;
            elementIndex[i].z = (float) (n & 1023) + 0.5f;
            elementIndex[i].w = (float) (n >> 10) + 0.5f;
        }
    }

    if (!obuf0) {   // initialize buffers
        obuf0 = createGPUBuffer(w, 256, 32, 4);
        obuf1 = createGPUBuffer(w, 256, 32, 4);
        createShader(OCCLUSION_SHADER, "occlusion.fp");
        createShader(OCCLUSION2_SHADER, "occlusion2.fp");
    }

    setDstGPUBuffer(obuf1);
    glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, 2);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        w, h2, 0, GL_RGBA, GL_FLOAT, (GLvoid *) emitterBuffer);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, 3);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA_NV,
        w, h2, 0, GL_RGBA, GL_FLOAT, (GLvoid *) (emitterBuffer + w*h2));
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, 4);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA16_NV,
        w, h2, 0, GL_RGBA, GL_FLOAT, (GLvoid *) elementIndex);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glActiveTexture(GL_TEXTURE0);

    setDstGPUBuffer(obuf0);
    glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, 2);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, 3);
    glEnable(GL_TEXTURE_RECTANGLE_NV);
    glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_RECTANGLE_NV, 4);
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    glActiveTexture(GL_TEXTURE0);

    BeginShade(obuf0, obuf1, OCCLUSION_SHADER);
    glMultiTexCoord4i(GL_TEXTURE1_ARB, w, h, 0, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.5f, 0.5f, 0.0f, 0.0f);
    glVertex2i(0, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, (float) w + 0.5f, 0.5f, 0.0f, 0.0f);
    glVertex2i(w, 0);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, (float) w + 0.5f, (float) h2 + 0.5f, 0.0f, 0.0f);
    glVertex2i(w, h2);
    glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.5f, (float) h2 + 0.5f, 0.0f, 0.0f);
    glVertex2i(0, h2);
    EndShade(obuf0, obuf1);

    if (passes > 1) {
        // second pass
        BeginShade(obuf1, obuf0, OCCLUSION2_SHADER);
        glMultiTexCoord4i(GL_TEXTURE1_ARB, w, h, 0, 0);
        glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.5f, 0.5f, 0.0f, 0.0f);
        glVertex2i(0, 0);
        glMultiTexCoord4f(GL_TEXTURE0_ARB, (float) w + 0.5f, 0.5f, 0.0f, 0.0f);
        glVertex2i(w, 0);
        glMultiTexCoord4f(GL_TEXTURE0_ARB, (float) w + 0.5f, (float) h + 0.5f, 0.0f, 0.0f);
        glVertex2i(w, h);
        glMultiTexCoord4f(GL_TEXTURE0_ARB, 0.5f, (float) h + 0.5f, 0.0f, 0.0f);
        glVertex2i(0, h);
        EndShade(obuf1, obuf0);

        readFromGPUBuffer(obuf0, resBuf, 0, 0, w, h);
    }
    else
        readFromGPUBuffer(obuf1, resBuf, 0, 0, w, h);

    // copy the ambient occlusion results to the vertex buffer for rendering later
    for (e = seList; e; e = e->child ? e->child : e->next) {
        if (!e->child) {
            e->v->position[1] = *(float3*)&resBuf[e->bufferIndex];
        }
    }

    setSrcGPUBuffer(NULL);
    setDstGPUBuffer(NULL);
    wglMakeCurrent(hdc, hglrc);
}