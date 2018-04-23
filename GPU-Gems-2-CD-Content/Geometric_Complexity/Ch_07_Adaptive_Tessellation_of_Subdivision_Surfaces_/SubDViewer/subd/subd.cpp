/*
    subd.cpp - Catmull-Clark subdivision surface tessellation code

    Copyright (C) 2005 NVIDIA Corporation

    This file is provided without support, instruction, or implied warranty of any
    kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
    not liable under any circumstances for any damages or loss whatsoever arising
    from the use or inability to use this file or items derived from it.

    This file contains all the functions for interfacing to the tessellator.
    This code assembles the subdivision surface control mesh and creates the data
    that will be written to a texture and used by GPU tessellation code.

    Textures created:

    patch index buffer  - defines each 16-control point bicubic patch in terms
        of the control point indices in the vertex buffer. The createPatch shader
        uses this index data to copy from the vertex buffer into the patch buffer
        and create the patches. This buffer also contains the indices for the
        extraordinary point data used by the createPatch shader to copy from the
        vertex buffer into the extraordinary point buffer.

    texture coordinate buffer - contains the texture coordinates for each of the
        4 corners of every patch. The data is is bilinearly interpolated by the
        "normal" shader to calculate the texture coordinate for each vertex produced
        by the tessellator.

    tangent buffer - This buffer contains data that is used to calculate the tangents
        that are used for orientation for normal mapping. (Note this is not the same as the tangents that are
        calculated to compute the subdivision surface normal). This buffer contains weight values
        and indices for vertices in the vertex buffer used to by the "tangent" shader to
        compute a normal map tangent for each of the 4 corners of the patches. The
        normal map tangents are then bilinearly interpolated by the "normal" shader just
        like the texture map coordinates for each vertex produce by the tessellator.

    displacement map - This buffer contains a version of the displacement map that
        is resampled so that each texel corresponds to a vertex a subdivided patch.
        There is a different set of texels for each subdivision level of the patch, and
        values are averaged when necessary so that patches that share edges will
        match.

    max displacement - This data is used by the testFlat shader when computing
        if an edge is flat. It conains the maximum displacement on either side of
        the top and left edges of a quad produced by the tessellator.

    
*/

#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#define SUBD_EXPORTS
#include "subd.h"
#include "subdinternal.h"

#define MAX_VERTEX_SIZE 64
#define MAX_PRIMITIVE_SIZE 8

// face flags
#define HIDE_FACE   1

float _subd_flip_bump = 1;
int _subd_output_vertex_size = 3;
float _subd_dmap_scale = 0.0f;
float _subd_dmap_scale_x_2;
float _subd_dmap_scale1, _subd_dmap_scale2;
float _subd_flat_scale, _subd_flat_scale2;
float _subd_near_len1, _subd_near_len2;
static float flat_distance = 1.0f;
int _subd_splinemode_flag;
int _subd_cull_face = SUBD_NONE;
int _subd_subdiv_level = 1;
int _subd_adaptive_flag = 0;
int _subd_output_normal;
int _subd_output_texcoord;
int _subd_emit_texcoords;
int _subd_output_tangent;
int _subd_ortho_flag;
static int output_primitive = SUBD_TRIANGLES;
int _subd_test_flag;
float *_subd_vertex_list[2];
int _subd_vertex_bufobj[2];
int _subd_vertex_list_size;
static int prim_list_length;
float *_subd_vertex_ptr;
float *_subd_vertex_end;
int *prim_ptr;
static int *prim_list, *prim_end;
static int vertex_index;
static int surface_type = SUBD_CATMULL_CLARK;
static int texcoord_index;
static int const_texcoord;
static int clip_flag = 0;
static int vp_height = 480;
static int hide_face = 0;
static float z_sign;
struct frustum frustum;
bool _subd_stop_tessellating;

static pvector *vertexarray;
static float2 *texcoordarray;
static float *edgeweightarray;
static int index_size = SUBD_INDEX32;

extern void Tessellate(Patch *patchBuffer, pvector *vlist, int vertices, SubdEnum state,
                       GroupInfo *groupInfo);
extern void LoadMeshTextures(unsigned char *patchIndexBuffer,
    int patchIndexW, int patchIndexH,
    float4 *texCoordBuffer, int texCoordW, int texCoordH,
    float4 *epTanInBuffer, int epTanInW,
    uchar *dmapTexture, int dmapW, int dmapH,
    uchar *dmapMaxTexture, int dmapMaxW, int dmapMaxH);

typedef struct edge {
	int index;
	int tindex;
	pvector *v;
	struct edge *next;
	struct edge *another;
	struct face *f;
    int dmap_ord;
    float2 texCoord, texCoord2;
    float uCoord[8];
    int2 tangentCoord;
    int *epIndex;
    int tangentCol;
} edge;

typedef struct face {
	edge *e;
    int index;
	int valence;
	int size;
	int flags;
    int vertexIndex[16];
    int epIndex[4][MAXVALENCE*2+1];
    int vertexValence[4];
    int ep;
    float4 tcoord[4];
    int dmap_depth;
    int dmap_offset;
	int dmap_raw_offset;
    int dmap_size;
    uchar *dmap_ptr;
	struct face *next;
} face;

typedef struct {
	char *address;
	int length;
	int max_length;
} subdBuffer;

struct vertinfo {
	char valence;
	char hidden_faces;
	char missing_faces;
    char not_shareable;
    int tindex;
    int disp;
    int num_disp;
    int mirror_vertex;
    int group;
    int column;
    int *epIndex;
};

typedef struct SubdMesh {
    int number;
	int *mesh_buffer;
	Patch *patchBuffer;
	subdBuffer le_buffer;
	subdBuffer vv_buffer;
	face *first_face, *last_face;
	edge *laste;
	int vertices;
    int texcoords;
	edge **lastedge;
	struct vertinfo *vert_info;
	face *current_face;
	int faces, patches;
	int surface_type;
	char *vert_flags;
	char *vert_clip_mask;
	float2 *tc_data;
    struct SubdMesh *next;
	float dmap_scale;
    int maxVertexIndex;
    bool newIndexBuffer;
    bool newDisplacement;
    int2 dmapSize;
    uchar *dmapMaxTexture;
    uchar *dmapTexture;
    GroupInfo *groupInfo;
    int2 epTanInLoc;
} SubdMesh;

static SubdMesh mesh0, mesh_clear;
static SubdMesh *current_mesh = &mesh0;
static int current_mesh_number = 0;

#define HASH(x) ((x) & 127)
static SubdMesh *mesh_array[128];

static void delete_mesh(SubdMesh *m);
static edge *rewindEdge(SubdMesh *m, edge *eIn);

static void
error(char *s, int p1)
{
	printf("Error: ");
	printf(s, p1);
}

static void set_mesh(int n)
{
	SubdMesh *p;

    current_mesh_number = n;
    if (n == 0) {
        current_mesh = &mesh0;
        return;
    }

    for (p = mesh_array[HASH(n)]; p && p->number != n; p = p->next)
        ;

    if (!p) {
        p = (SubdMesh *) calloc(sizeof(*p), 1);
        p->number = n;
        p->next = mesh_array[HASH(n)];
        mesh_array[HASH(n)] = p;
    }
    current_mesh = p;

}

SUBD_API void subdSet(SubdEnum e, int value)
{
    switch (e) {
        case SUBD_ADAPTIVE:         _subd_adaptive_flag = value; break;
        case SUBD_OUTPUT_NORMAL:    _subd_output_normal = value; break;
        case SUBD_OUTPUT_TEXCOORD:  _subd_output_texcoord = value; break;
        case SUBD_OUTPUT_TANGENT:   _subd_output_tangent = value; break;
        case SUBD_CULL_FACE: 
            if (value == SUBD_NONE || value == SUBD_FRONT || value == SUBD_BACK)
                _subd_cull_face = value;
            else
                error("subdSet() bad enum (%d) for cull face", value);
            break;
        case SUBD_CLIP:             clip_flag = value; break;
        case SUBD_CURRENT_MESH:     set_mesh(value);
                                    break;
        case SUBD_SURFACE_TYPE:     
            if (surface_type != SUBD_DOO_SABIN &&
                surface_type != SUBD_CATMULL_CLARK) {
                error("subdSet() bad SUBD_SURFACE_TYPE type: %d", value);
            }
            surface_type = value;
            break;
        case SUBD_SUBDIVIDE_DEPTH:
            _subd_subdiv_level = value;
            if (_subd_subdiv_level < 0)
                _subd_subdiv_level = 0;
            break;
        case SUBD_TEXCOORD_INDEX:   texcoord_index = value; break;
        case SUBD_HIDE_FACE:        hide_face = value; break;
        case SUBD_OUTPUT_PRIMITIVE:
            if (value == SUBD_TRIANGLES || value == SUBD_QUADS ||
                value == SUBD_LINES)
                output_primitive = value;
            else
                error("subdSet() bad value for OUTPUT_PRIMITIVE: %d", value);
            break;
        case SUBD_INDEX_SIZE:
            if (value == SUBD_INDEX16 || value == SUBD_INDEX32)
                index_size = value;
            else
                error("subdSet() bad value for INDEX_SIZE: %d", value);
            break;
        default:
            error("Bad enum for subdSet(): %d", e);
            break;
    }
}

SUBD_API void subdSetf(SubdEnum e, float value)
{
    switch (e) {
        case SUBD_FLAT_DISTANCE:
            flat_distance = value;
            break;
        default:
            error("Bad enum for subdSetf(): %d", e);
            break;
    }
}

SUBD_API int subdGet(SubdEnum e)
{
    switch (e) {
        case SUBD_ADAPTIVE:         return _subd_adaptive_flag;
        case SUBD_OUTPUT_NORMAL:    return _subd_output_normal;
        case SUBD_OUTPUT_TEXCOORD:  return _subd_output_texcoord;
        case SUBD_OUTPUT_TANGENT:   return _subd_output_tangent;
        case SUBD_CULL_FACE:        return _subd_cull_face;
        case SUBD_CLIP:             return clip_flag;
        case SUBD_SURFACE_TYPE:     return surface_type;
        case SUBD_SUBDIVIDE_DEPTH:  return _subd_subdiv_level;
        case SUBD_CURRENT_MESH:     return current_mesh_number;
        case SUBD_HIDE_FACE:        return hide_face;
        case SUBD_OUTPUT_VERTEX_SIZE:
            return sizeof(float4);
        case SUBD_OUTPUT_PRIMITIVE: return output_primitive;
        case SUBD_INDEX_SIZE:       return index_size;
        default:
            error("Bad enum for subdGet(): %d", e);
            break;
    }

    return 0;
}

SUBD_API float subdGetf(SubdEnum e)
{
    switch (e) {
        case SUBD_FLAT_DISTANCE:
            return flat_distance;
        default:
            error("Bad enum for subdGetf(): %d", e);
            break;
    }

    return 0.0;
}

SUBD_API void subdFrustum(float l, float r, float b, float t, float n, float f)
{
    z_sign = 1.0f;
    if (f < 0) {
        f = -f;
        n = -n;
        z_sign = -1.0f;
    }
    frustum.l = l;
    frustum.r = r;
    frustum.b = b;
    frustum.t = t;
    frustum.n = n;
    frustum.f = f;
    _subd_ortho_flag = 0;
}

SUBD_API void subdOrtho(float l, float r, float b, float t, float n, float f)
{
    z_sign = 1.0f;
    if (f < 0) {
        f = -f;
        n = -n;
        z_sign = -1.0f;
    }
    frustum.l = l;
    frustum.r = r;
    frustum.b = b;
    frustum.t = t;
    frustum.n = n;
    frustum.f = f;
    _subd_ortho_flag = 1;
}

SUBD_API void subdViewport(int x, int y, int width, int height)
{
    vp_height = height;
}

SUBD_API void subdCtrlMeshVertexPointer(int size, int stride, float *p)
{
    vertexarray = (pvector *) p;
}

SUBD_API void subdCtrlMeshTexCoordPointer(int size, int stride, float *p)
{
    texcoordarray = (float2 *) p;
}

SUBD_API void subdCtrlMeshEdgeWeightPointer(int stride, float *p)
{
    edgeweightarray = p;
}

static void
sizebuffer(subdBuffer *b, int bytes)
{
	int i;

	if (b->max_length < bytes) {
		i = b->max_length;

		if (!b->max_length)
			b->max_length = 16;

		while (b->max_length < bytes)
			b->max_length <<= 1;
		if (b->address)
			b->address = (char *) realloc(b->address, b->max_length);
		else
			b->address = (char *) malloc(b->max_length);
		while (i < b->max_length)
			b->address[i++] = 0;
	}
	if (b->length < bytes)
		b->length = bytes;
}

// make sure there is at least n+1 values in lastedge and vert_info
// tables
static void
size_tables(SubdMesh *m, int n)
{
	sizebuffer(&m->le_buffer, (n+1)*sizeof(*m->lastedge));
	m->lastedge = (edge **) m->le_buffer.address;

	sizebuffer(&m->vv_buffer, (n+1)*sizeof(*m->vert_info));
    m->vert_info = (struct vertinfo *) m->vv_buffer.address;
}

// find the other half-edge that matches the input edge

static edge *
findEdge(SubdMesh *m, edge *eIn)
{
	edge *e;
    int i;

    if (!eIn)
        return 0;

    i = eIn->index;
	for (e = m->lastedge[eIn->next->index]; e; e = e->another)
		if (e->next->index == i)
			return e;

	return 0;
}

SUBD_API void
subdDeleteMesh()
{
    SubdMesh *m, **h;

    m = current_mesh;
	if (!m)
		return;

	delete_mesh(m);

	if (m->mesh_buffer)
		free((void *) m->mesh_buffer);
	if (m->patchBuffer)
		free((void *) m->patchBuffer);
    if (m->groupInfo)
        free((void *) m->groupInfo);

    if (current_mesh_number == 0) {
        mesh0 = mesh_clear;
    }
    else {
        h = &mesh_array[HASH(current_mesh_number)];
        for (; *h; h = &(*h)->next)
            if (*h == m) {
                *h = m->next;
                break;
            }
        free((void *) m);
    }
}

SUBD_API void subdBeginMesh()
{
	subdDeleteMesh();

	set_mesh(current_mesh_number);
}

SUBD_API void
subdFace()
{
	SubdMesh *m = current_mesh;
	face *f;

	m->current_face = f = (face *) calloc(sizeof(face), 1);
	m->laste = 0;
	f->flags = 0;
    if (hide_face)
        f->flags |= HIDE_FACE;

	if (!m->first_face)
		m->first_face = m->last_face = f;
	else {
		m->last_face->next = f;
		m->last_face = f;
	}
}

SUBD_API void
subdVertex(int vindex)
{
	SubdMesh *m = current_mesh;
	face *f = m->current_face;
	edge *e;
    int tindex;

	f->valence++;
    tindex = texcoord_index;
	e = (edge *) calloc(sizeof(edge), 1);
	e->index = vindex;
	e->tindex = tindex;
	e->f = f;
	size_tables(m, e->index);
	e->another = m->lastedge[e->index];
	m->lastedge[e->index] = e;
	m->vert_info[e->index].valence++;
    if (m->vert_info[e->index].valence > 1 && m->vert_info[e->index].tindex !=
        tindex)
        m->vert_info[e->index].not_shareable = 1;
    else
        m->vert_info[e->index].tindex = tindex;
	if (f->flags & HIDE_FACE)
		m->vert_info[e->index].hidden_faces++;
	if (!f->e)
		f->e = e;
	else {
		m->laste->next = e;
	}
	m->laste = e;
	e->next = f->e;

	if (vindex >= m->vertices)
		m->vertices = vindex+1;
    if (tindex >= m->texcoords)
        m->texcoords = tindex+1;
}

static edge *
e_before(edge *e)
{
	edge *e2;

    if (!e)
        return 0;

	for (e2 = e; e2->next != e; e2 = e2->next)
		;

	return e2;
}

static edge *
rewindEdge(SubdMesh *m, edge *eIn)
{
    edge *e, *e2;

    if (!eIn)
        return 0;
    for (e = eIn;;) {
        e2 = findEdge(m, e);
        if (!e2)
            return e;
        e = e2->next;
        if (e == eIn)
            return e;
    }
}

#define PATCH_INDEX_HEIGHT  512
unsigned char patchIndexBuffer[PATCH_INDEX_HEIGHT][PATCH_BUFFER_WIDTH][4];
#define TEXCOORD_WIDTH  512
#define TEXCOORD_HEIGHT 256
#define DMAP_TEXTURE_WIDTH 512
float4 texCoordBuffer[TEXCOORD_HEIGHT][TEXCOORD_WIDTH];
float4 epTanInBuffer[MAXGROUPS*3*4][PATCH_BUFFER_WIDTH];

static void
calcTexCoordTan(SubdMesh *m, edge *eIn)
{
    float2 vtc;
    float2 etc[MAXVALENCE];
    int valence;
    int i;
    static float4 vzero;
    edge *e;
    float scale;

    valence = m->vert_info[eIn->index].valence;
    if (valence < 3 || valence > 8)
        return;

    vtc = eIn->texCoord;
    for (i = 0; i < MAXVALENCE; i++)
        etc[i] = vtc;
    for (i = 0, e = eIn; i < valence; i++) {
        etc[i] = e->next->texCoord;
        if (i + 1 < valence)
            etc[i+1] = e->next->next->next->texCoord;
        e = findEdge(m, e->next->next->next);
        if (!e || e->texCoord.x != vtc.x || e->texCoord.y != vtc.y)
            break;
    }

    if (i < valence) {  // did not finish - go around the other way
        for (i = valence-1, e = eIn; i; i--) {
            e = findEdge(m, e);
            if (!e)
                break;
            e = e->next;
            if (e->texCoord.x != vtc.x || e->texCoord.y != vtc.y)
                break;
            etc[i] = e->next->texCoord;
        }
    }

    // scale these values to -1 to 1 range
    eIn->texCoord2.x = etc[0].x - etc[2].x;
    eIn->texCoord2.y = etc[1].x - etc[3].x;
    scale = (float) (1.0f/sqrt(eIn->texCoord2.x * eIn->texCoord2.x +
        eIn->texCoord2.y * eIn->texCoord2.y));
    eIn->texCoord2.x *= scale;
    eIn->texCoord2.y *= scale;

    for (i = 0; i < valence; i++)
        eIn->uCoord[i] = etc[i].x - vtc.x;
    for (; i < MAXVALENCE; i++)
        eIn->uCoord[i] = 0.0f;
}

static int
calcFlipTri(float2 a, float2 b, float2 c)
{
    if ((b.x - a.x)*(b.y - c.y) > (b.y - a.y)*(b.x - c.x))
        return -1;
    return 1;
}

static int
calcFlip(float2 a, float2 b, float2 c, float2 d)
{
    int count = 0;

    if (calcFlipTri(a, b, c) < 0)
        count++;
    if (calcFlipTri(b, c, d) < 0)
        count++;
    if (calcFlipTri(d, a, b) < 0)
        count++;

    if (count > 1)
        return -1;
    return 1;
}

static void
setPatchBufferEPIndex(int x, int y, int col, int row)
{
    int t;
  
    t = (row << 10 | col) << 1 | 1;
    patchIndexBuffer[y][x][2] = t & 255;
    patchIndexBuffer[y][x][3] = t>>8;
}

static int EPBuffer[EP_BUFFER_WIDTH][MAXVALENCE*2+1];

static int findIndex(int col, int index)
{
    int i;
    int *p;

    p = &EPBuffer[col][0];

    for (i = 1; i <= MAXVALENCE; i++)
        if (p[i] == index)
            return i;
    printf("Error -- findIndex(%d, %d)\n", col, index);
    return 0; // Error
}

static void
clearTangentCoords(SubdMesh *m, int group)
{
    face *f;

    for (f = m->first_face; f; f = f->next) {
        f->e->tangentCol = -1;
        f->e->next->tangentCol = -1;
        f->e->next->next->tangentCol = -1;
        f->e->next->next->next->tangentCol = -1;
    }
    
    m->epTanInLoc.x = 0;
    m->epTanInLoc.y = group * 12;
}

static void
writeEPTangentData(SubdMesh *m, edge *e, int epCol)
{
    int2 loc;
    float w[8];
    int i, n;
    edge *e2;

   for (i = 0; i < 8; i++)
        w[i] = 0.0f;

    for (i = 0; i < 8; i++) {
        if (e->uCoord[i] == 0.0f)
            continue;
        n = findIndex(epCol, e->epIndex[i+1]) - 1;
        if (n >= 0 && n < 8)
            w[n] = e->uCoord[i];
        else
            printf("Error findIndex returned %d\n", n);
    }

    for (i = 0; i < MAXVALENCE; i++)     // save for comparison sake
        e->uCoord[i] = w[i];
    e->tangentCol = epCol;

    for (e2 = m->lastedge[e->index]; e2; e2 = e2->another) {
        if (e2 == e)
            continue;
        if (e2->tangentCol == epCol) {
            for (i = 0; i < MAXVALENCE; i++) {
                if (e->uCoord[i] != e2->uCoord[i])
                    break;
            }
            if (i == MAXVALENCE)
                break;
        }
    }

    if (e2)     // found tangent info to share
        loc = e2->tangentCoord;
    else {
        loc = m->epTanInLoc;
        m->epTanInLoc.x++;
        if (m->epTanInLoc.x == PATCH_BUFFER_WIDTH) {
            m->epTanInLoc.x = 0;
            m->epTanInLoc.y++;
        }
    }
 

    epTanInBuffer[loc.y*3][loc.x] = * (float4 *) &w[0];
    epTanInBuffer[loc.y*3+1][loc.x] = * (float4 *) &w[4];
    epTanInBuffer[loc.y*3+2][loc.x].x = epCol + 0.5f;
    epTanInBuffer[loc.y*3+2][loc.x].y = 2.5f; // first 2 lines are limit surface data


    // set e->tangentCoord

    e->tangentCoord = loc;
}

//#define FtoS(x) ((short) ((x) * 32768))
#define FtoS(x) (x)
// convert mesh to patches
static void
convert_mesh(SubdMesh *m)
{
    face *f;
    int index[4];
    int t;
    int i, j;
    edge *e, *e2, *e3;
    int valence;
    int faces;
    Patch *p, *lastp;
    Patch *groupStart, *q;
    int x, y;
    int groupNumber;
    int epColumn;
    static int epList[EP_BUFFER_WIDTH];
    struct vertinfo * vp;
    int flipBinormal;

    static int indexPos[4][4] = {
        5, 4, 0, 1,
        6, 2, 3, 7,
        10, 11, 15, 14,
        9, 13, 12, 8,
    };

    m->epTanInLoc.x = 0;
    m->epTanInLoc.y = 0;
    faces = 0;
    for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;
        faces++;
        for (i = 0, e = f->e; i < 4; i++, e = e->next) {
            index[0] = e->index;
            index[1] = e->index;
            index[2] = e->index;
            index[3] = e->index;
            e2 = findEdge(m, e);
            if (e2)
                index[3] = e2->next->next->index;
            
            e2 = findEdge(m, e->next->next->next);
            if (e2) {
                e3 = e2 = e_before(e2);
                index[1] = e2->index;
                e2 = findEdge(m, e2);
                if (e2)
                    index[2] = e2->next->next->index;
            }
            else {
                e2 = rewindEdge(m, e)->next;
                index[2] = e2->index;
            }
            for (j = 0; j < 4; j++)
                f->vertexIndex[indexPos[i][j]] = index[j];
        }
        f->ep = 0;
        f->vertexValence[0] = 4;
        f->vertexValence[1] = 4;
        f->vertexValence[2] = 4;
        f->vertexValence[3] = 4;
        for (i = 0, e = f->e; i < 4; i++, e = e->next) {
            valence = m->vert_info[e->index].valence;
            if (m->vert_info[e->index].missing_faces) {
                valence += 2;
            }
            if (valence == 4 || valence < 3 || valence > 8)
                continue;

            f->vertexValence[i] = valence;
            switch (i) {
                case 0: f->ep |= EP1; break;
                case 1: f->ep |= EP2; break;
                case 2: f->ep |= EP4; break;
                case 3: f->ep |= EP3; break;
            }

            e2 = rewindEdge(m, e);
            e->epIndex = &f->epIndex[i][0];
            f->epIndex[i][0] = e->index;
            for (j = 0; j < valence; j++) {
                if (e2) {
                    f->epIndex[i][j+1] = e2->next->index;
                    f->epIndex[i][j+9] = e2->next->next->index;
                    e2 = findEdge(m, e2->next->next->next);
                }
                else {
                    f->epIndex[i][j+1] = e->index;
                    f->epIndex[i][j+9] = e->index;
                }
            }
        }
    }
    // calc texture coordinate information

    for (f = m->first_face; f; f = f->next) {
        for (i = 0, e = f->e; i < 4; i++, e = e->next) {
            if (e->tindex >= 0)
               e->texCoord = m->tc_data[e->tindex];
        }
    }

    // calc texcoord tangent vector for normal mapping

    for (f = m->first_face; f; f = f->next) {
        calcTexCoordTan(m, f->e);
        calcTexCoordTan(m, f->e->next);
        calcTexCoordTan(m, f->e->next->next);
        calcTexCoordTan(m, f->e->next->next->next);
    }

    m->patchBuffer = (Patch *) calloc(faces, sizeof(Patch));
    m->groupInfo = (GroupInfo *) calloc(sizeof(GroupInfo), MAXGROUPS);
    p = m->patchBuffer;
    lastp = NULL;
    x = 0, y = 0;
    m->maxVertexIndex = -1;

    for (i = 0; i < m->vertices; i++)
        m->vert_info[i].group = -1;

    groupNumber = 0;
    epColumn = 0;
    groupStart = p;
    clearTangentCoords(m, 0);
    for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;
       
        p->nextPatch = p + 1;
        p->scratchPtr = (void *) f;
        p->ep = f->ep;
        p->epValence[0] = f->vertexValence[0];
        p->epValence[1] = f->vertexValence[1];
        p->epValence[2] = f->vertexValence[3];
        p->epValence[3] = f->vertexValence[2];

        for (i = 0, e = f->e; i < 4; i++, e = e->next) {
            vp = m->vert_info + e->index;
            if (f->vertexValence[i] == 4)
                continue;
            if (vp->group != groupNumber) {
                vp->group = groupNumber;
                vp->column = epColumn++;
            }
            epList[vp->column] = e->index;
            vp->epIndex = &f->epIndex[i][0];
        }

        p->indexLoc.x = x;
        p->indexLoc.y = y;

        for (j = 0; j < 4; j++) {
            for (i = 0; i < 4; i++) {
                t = f->vertexIndex[j*4+i];
                if (t > m->maxVertexIndex)
                    m->maxVertexIndex = t;
                patchIndexBuffer[y+j][x+i][1] = t>>8;
                patchIndexBuffer[y+j][x+i][0] = t & 255;
                patchIndexBuffer[y+j][x+i][2] = 0;
                patchIndexBuffer[y+j][x+i][3] = 0;
            }
        }
        x += 4;
        if (x + 4 > PATCH_BUFFER_WIDTH) {
            x = 0;
            y += 4;
        }

        p->group = groupNumber;
        if (epColumn + 4 > EP_BUFFER_WIDTH ||
            y + 4 - groupNumber*(SEGMENT_SIZE+EP_HEIGHT) > SEGMENT_SIZE) {
            groupNumber++;
        }

        while (f->next && (f->next->valence != 4 || (f->next->flags & HIDE_FACE)))
            f = f->next;

        if (p->group != groupNumber || !f->next) {
            int col;
            int lastCol = 0;

            if (x != 0) {
                x = 0;
                y += 4;
            }
            // sort extraordinary points in group

            for (j = 3, col = 0; j <= MAXVALENCE; j++) {
                if (j != 4) {
                    for (i = 0; i < epColumn; i++) {
                        vp = m->vert_info + epList[i];
                        if (vp->valence == j)
                            vp->column = col++;
                    }
                }
                m->groupInfo[p->group].epCount[j-3] = col - lastCol;
                lastCol = col;
            }
            m->groupInfo[p->group].epTotal = col;
            // copy ep data to buffer

            for (i = 0; i < epColumn; i++) {
                vp = m->vert_info + epList[i];
                for (j = 0; j < EP_HEIGHT; j++)
                    EPBuffer[vp->column][j] = vp->epIndex[j];
            }

            for (i = 0; i < epColumn; i++) {
                for (j = 0; j < EP_HEIGHT; j++) {
                    t = EPBuffer[i][j];
                    patchIndexBuffer[y+j][i][0] = t & 255;
                    patchIndexBuffer[y+j][i][1] = t>>8;
                    patchIndexBuffer[y+j][i][2] = 0;
                    patchIndexBuffer[y+j][i][3] = 0;
                }
            }
            
            // set ep index values in patch buffer

            for (q = groupStart; q != p->nextPatch; q = q->nextPatch) {
                int x = q->indexLoc.x;
                int y = q->indexLoc.y;
                face *f = (face *) q->scratchPtr;

                if (!f->ep)
                    continue;

                if (f->ep & EP1) { 
                    int c = m->vert_info[f->e->index].column;
                    setPatchBufferEPIndex(x+1, y+1, c, 0);
                    setPatchBufferEPIndex(x+1, y+0, c, findIndex(c, f->vertexIndex[1])+MAXVALENCE);
                    setPatchBufferEPIndex(x+0, y+1, c, findIndex(c, f->vertexIndex[4]));
                    writeEPTangentData(m, f->e, c);
                }

                if (f->ep & EP2) {
                    int c = m->vert_info[f->e->next->index].column;
                    setPatchBufferEPIndex(x+2, y+1, c, 0);
                    setPatchBufferEPIndex(x+2, y+0, c, findIndex(c, f->vertexIndex[2]));
                    setPatchBufferEPIndex(x+3, y+1, c, findIndex(c, f->vertexIndex[7])+MAXVALENCE);
                    writeEPTangentData(m, f->e->next, c);
                }

                if (f->ep & EP4) {
                    int c = m->vert_info[f->e->next->next->index].column;
                    setPatchBufferEPIndex(x+2, y+2, c, 0);
                    setPatchBufferEPIndex(x+3, y+2, c, findIndex(c, f->vertexIndex[11]));
                    setPatchBufferEPIndex(x+2, y+3, c, findIndex(c, f->vertexIndex[14])+MAXVALENCE);
                    writeEPTangentData(m, f->e->next->next, c);
                }

                if (f->ep & EP3) {
                    int c = m->vert_info[f->e->next->next->next->index].column;
                    setPatchBufferEPIndex(x+1, y+2, c, 0);
                    setPatchBufferEPIndex(x+0, y+2, c, findIndex(c, f->vertexIndex[8])+MAXVALENCE);
                    setPatchBufferEPIndex(x+1, y+3, c, findIndex(c, f->vertexIndex[13]));
                    writeEPTangentData(m, f->e->next->next->next, c);
                }
            }

            m->groupInfo[groupStart->group].epTangentH = m->epTanInLoc.y;
            m->groupInfo[groupStart->group].epTangentW = m->epTanInLoc.x;
            if (m->epTanInLoc.x != 0)
                 m->groupInfo[groupStart->group].epTangentH++;
            if (m->epTanInLoc.y > 0)
                m->groupInfo[groupStart->group].epTangentW = PATCH_BUFFER_WIDTH;

            epColumn = 0;
            groupStart = p->nextPatch;
            clearTangentCoords(m, groupNumber);
            y += EP_HEIGHT;
        }
        lastp = p++;
    }
    if (lastp)
        lastp->nextPatch = NULL;

    // create texture coordinate buffer

    x = 0;
    y = 0;
    for (p = m->patchBuffer; p; p = p->nextPatch) {
        f = (face *) p->scratchPtr;
        
        flipBinormal = calcFlip(f->e->texCoord, f->e->next->texCoord,
            f->e->next->next->texCoord, f->e->next->next->next->texCoord);

        p->texCoordLoc.x = (x+1) * flipBinormal;
        p->texCoordLoc.y = y;
        texCoordBuffer[y][x].x = FtoS(f->e->texCoord.x);
        texCoordBuffer[y][x].y = FtoS(f->e->texCoord.y);
        texCoordBuffer[y][x].z = FtoS(f->e->texCoord2.x);
        texCoordBuffer[y][x].w = FtoS(f->e->texCoord2.y);
        if (f->ep & EP1) {
            texCoordBuffer[y][x].z = f->e->tangentCoord.x + 0.5f;
            texCoordBuffer[y][x].w = f->e->tangentCoord.y + 1.5f;
        }

        texCoordBuffer[y][x+1].x = FtoS(f->e->next->texCoord.x);
        texCoordBuffer[y][x+1].y = FtoS(f->e->next->texCoord.y);
        texCoordBuffer[y][x+1].z = FtoS(-f->e->next->texCoord2.y);
        texCoordBuffer[y][x+1].w = FtoS(f->e->next->texCoord2.x);
        if (f->ep & EP2) {
            texCoordBuffer[y][x+1].z = f->e->next->tangentCoord.x + 0.5f;
            texCoordBuffer[y][x+1].w = f->e->next->tangentCoord.y + 1.5f;
        }

        texCoordBuffer[y+1][x].x = FtoS(f->e->next->next->next->texCoord.x);
        texCoordBuffer[y+1][x].y = FtoS(f->e->next->next->next->texCoord.y);
        texCoordBuffer[y+1][x].z = FtoS(f->e->next->next->next->texCoord2.y);
        texCoordBuffer[y+1][x].w = FtoS(-f->e->next->next->next->texCoord2.x);
        if (f->ep & EP3) {
            texCoordBuffer[y+1][x].z = (float) f->e->next->next->next->tangentCoord.x + 0.5f;
            texCoordBuffer[y+1][x].w = (float) f->e->next->next->next->tangentCoord.y + 1.5f;
        }

        texCoordBuffer[y+1][x+1].x = FtoS(f->e->next->next->texCoord.x);
        texCoordBuffer[y+1][x+1].y = FtoS(f->e->next->next->texCoord.y);
        texCoordBuffer[y+1][x+1].z = FtoS(-f->e->next->next->texCoord2.x);
        texCoordBuffer[y+1][x+1].w = FtoS(-f->e->next->next->texCoord2.y);
        if (f->ep & EP4) {
            texCoordBuffer[y+1][x+1].z = (float) f->e->next->next->tangentCoord.x + 0.5f;
            texCoordBuffer[y+1][x+1].w = (float) f->e->next->next->tangentCoord.y + 1.5f;
        }

        x += 2;
        if (x + 2 > TEXCOORD_WIDTH) {
            x = 0;
            y += 2;
        }
    }
    m->newIndexBuffer = true;
}
	
static void
delete_mesh(SubdMesh *m)
{
	face *f, *fnext;
	edge *e, *enext;

	for (f = m->first_face; f; f = fnext) {
		e = f->e;
		do {
			enext = e->next;
			free((void *)e);
		} while ((e = enext) != f->e);
		fnext = f->next;
		free((void *)f);
	}
	m->first_face = 0;

	free(m->lastedge);
	m->lastedge = 0;
	free(m->vert_info);
	m->vert_info = 0;
	if (m->vert_flags)
		free(m->vert_flags);
}

SUBD_API void
subdEndMesh()
{
	SubdMesh *m = current_mesh;

	m->tc_data = (float2 *) texcoordarray;
    convert_mesh(m);
    Tessellate(NULL, NULL, 0, SUBD_START, NULL); // make sure gpu buffer is initialized before loading textures
    LoadMeshTextures(&patchIndexBuffer[0][0][0], PATCH_BUFFER_WIDTH, PATCH_INDEX_HEIGHT,
        &texCoordBuffer[0][0], TEXCOORD_WIDTH, TEXCOORD_HEIGHT,
        &epTanInBuffer[0][0], MAXGROUPS*3*4,
        NULL, 0, 0,
        NULL, 0, 0);
}

SUBD_API void subdOutputVertexBuffer(void *vb, int size, int n, int buffObj)
{
    if (n >= sizeof(_subd_vertex_list)/sizeof(_subd_vertex_list[0]))
        return;
    _subd_vertex_list[n] = (float *) vb;
    _subd_vertex_bufobj[n] = buffObj;
    _subd_vertex_list_size = size;
}

SUBD_API void subdOutputIndexBuffer(void *ib, int size)
{
    prim_list = (int *) ib;
    prim_list_length = size;
}

SUBD_API SubdEnum
subdTessellate(int *vlist_length, int *primlist_length, SubdEnum state)
{
	SubdMesh *m = current_mesh;

	if (!m)
		return SUBD_DONE;

	prim_ptr = prim_list;
    prim_end = prim_ptr + (prim_list_length>>2) - MAX_PRIMITIVE_SIZE;

	vertex_index = 0;

    if (!_subd_ortho_flag)
        _subd_flat_scale = z_sign * (frustum.b - frustum.t)*flat_distance /
            (vp_height*frustum.n);
    else
        _subd_flat_scale = (frustum.b - frustum.t)*flat_distance /
            vp_height;

    _subd_stop_tessellating = false;
	
    // precalculate these values that will be used by the shaders
	_subd_flat_scale *= -8.0f/3.0f;
	_subd_flat_scale *= 3.0f;
    _subd_dmap_scale1 = _subd_dmap_scale * 8.0f;
    _subd_dmap_scale2 = _subd_dmap_scale1 * 8.0f;
    _subd_dmap_scale_x_2 = _subd_dmap_scale * 2.0f * z_sign;
	_subd_flat_scale2 = _subd_flat_scale*0.50f;

    _subd_near_len1 = 2*_subd_flat_scale * frustum.n * z_sign;
    _subd_near_len2 = 2*_subd_flat_scale2 * frustum.n * z_sign;

    Tessellate(m->patchBuffer, vertexarray, m->maxVertexIndex+1, state, m->groupInfo);

    *vlist_length = (_subd_vertex_ptr - _subd_vertex_list[0])/4;
    if (index_size == SUBD_INDEX16)
        *primlist_length = (short *) prim_ptr - (short *) prim_list;
    else
        *primlist_length = prim_ptr - prim_list;

    if (_subd_stop_tessellating) {
        if (*primlist_length == 0)
            return SUBD_DONE;
        return SUBD_CONTINUE;
    }

    return SUBD_DONE;
}

void
_subd_emit_quad(int i1, int i2, int i3, int i4)
{
    short *prim_ptr16;

	if (prim_ptr >= prim_end || _subd_stop_tessellating) {
        _subd_stop_tessellating = 1;
        return;
    }

    if (index_size == SUBD_INDEX16) {
        prim_ptr16 = (short *) prim_ptr;
        if (output_primitive == SUBD_LINES) {
            prim_ptr16[0] = i1;
            prim_ptr16[1] = i2;
            prim_ptr16[2] = i2;
            prim_ptr16[3] = i3;
            prim_ptr16[4] = i3;
            prim_ptr16[5] = i4;
            prim_ptr16[6] = i4;
            prim_ptr16[7] = i1;
            prim_ptr16 += 8;
        }
        else if (output_primitive == SUBD_QUADS) {
            prim_ptr16[0] = i1;
            prim_ptr16[1] = i2;
            prim_ptr16[2] = i3;
            prim_ptr16[3] = i4;
            prim_ptr16 += 4;
        }
        else {  // SUBD_TRIANGLES
            prim_ptr16[0] = i1;
            prim_ptr16[1] = i2;
            prim_ptr16[2] = i3;
            if (i3 != i4) {
                prim_ptr16[3] = i1;
                prim_ptr16[4] = i3;
                prim_ptr16[5] = i4;
                prim_ptr16 += 3;
            }
            prim_ptr16 += 3;
        }
        prim_ptr = (int *) prim_ptr16;
    }
    else {
        if (output_primitive == SUBD_LINES) {
            prim_ptr[0] = i1;
            prim_ptr[1] = i2;
            prim_ptr[2] = i2;
            prim_ptr[3] = i3;
            prim_ptr[4] = i3;
            prim_ptr[5] = i4;
            prim_ptr[6] = i4;
            prim_ptr[7] = i1;
            prim_ptr += 8;
        }
        else if (output_primitive == SUBD_QUADS) {
            prim_ptr[0] = i1;
            prim_ptr[1] = i2;
            prim_ptr[2] = i3;
            prim_ptr[3] = i4;
            prim_ptr += 4;
        }
        else {  // SUBD_TRIANGLES
            prim_ptr[0] = i1;
            prim_ptr[1] = i2;
            prim_ptr[2] = i3;
            if (i3 != i4) {
                prim_ptr[3] = i1;
                prim_ptr[4] = i3;
                prim_ptr[5] = i4;
                prim_ptr += 3;
            }
            prim_ptr += 3;
        }
    }
}

// Displacement map support functions

#define lerp1(a, b, w) (((a) - (b)) * (w) + (b))

static void
fill_disp_info(uchar *dest, char *disp, int w, int rowbytes, int index,
    uchar *dest2)
{
	int c1, c2, c3, c4;
	float a, b, c;
	float h, v;
	int d, maxd;
	int i, j;

	c1 = disp[0];
	c2 = disp[w];
	c3 = disp[rowbytes*w];
	c4 = disp[(rowbytes+1)*w];

	maxd = 0;
	for (j = 0; j <= w; j++) {
		for (i = 0; i <= w; i++) {
			h = (float)i/w;
			v = (float)j/w;
			a = lerp1(c2, c1, h);
			b = lerp1(c4, c3, h);
			c = lerp1(b, a, v);

			d = (int) (disp[j*rowbytes + i] - c + 0.5f);
			if (d < 0)
				d = -d;
			if (d > maxd)
				maxd = d;
		}
	}

    if (maxd < 2)
        maxd = 0;

	dest[index] = maxd;
    dest2[index*8+4] = (uchar) c1;
    dest2[index*8+5] = (uchar) c2;
    dest2[index*8+6] = (uchar) c3;
    dest2[index*8+7] = (uchar) c4;

	w >>= 1;
	if (w > 0) {
		fill_disp_info(dest, disp, w, rowbytes, index*4+1, dest2);
		fill_disp_info(dest, disp + w, w, rowbytes, index*4+2, dest2);
		fill_disp_info(dest, disp + w*rowbytes, w, rowbytes, index*4+3, dest2);
		fill_disp_info(dest, disp + w*(rowbytes+1), w,rowbytes,index*4+4,dest2);
	}
}

#define lerp(dest, a, b, w) { \
		dest.x = (a.x - b.x) * w + b.x; \
		dest.y = (a.y - b.y) * w + b.y; \
	}

#define Max(a, b) ((a)>(b) ? (a) : (b))
#define Min(a, b) ((a)<(b) ? (a) : (b))

static void
set_max_disp(uchar *dst, uchar *src,
    face *center, int centerindex,
	face *top, int topindex, int *top_e,
    face *right, int rightindex, int *right_e,
	face *bottom, int bottomindex, int *bottom_e,
    face *left, int leftindex, int *left_e,
    int depth)
{
	uchar d;
    int dstindex;
    static int ctop[2] = { 3, 4 };
    static int cright[2] = { 1, 3 };
    static int cbottom[2] = { 2, 1 };
    static int cleft[2] = { 4, 2 };

	d = src[center->dmap_offset + centerindex];
	dstindex = centerindex*8;

    if (!top)
        dst[dstindex] = d;
	else if (depth > top->dmap_depth)
		dst[dstindex] = 0;
    else
        dst[dstindex] = Max(d, src[top->dmap_offset+topindex]);

    if (!right)
        dst[dstindex+1] = d;
	else if (depth > right->dmap_depth)
		dst[dstindex+1] = 0;
    else
        dst[dstindex+1] = Max(d, src[right->dmap_offset + rightindex]);

    if (!bottom)
        dst[dstindex+2] = d;
	else if (depth > bottom->dmap_depth)
        dst[dstindex+2] = 0;
    else
        dst[dstindex+2] = Max(d, src[bottom->dmap_offset + bottomindex]);

    if (!left)
        dst[dstindex+3] = d;
	else if (depth > left->dmap_depth)
        dst[dstindex+3] = 0;
    else
        dst[dstindex+3] = Max(d, src[left->dmap_offset + leftindex]);

    depth++;

    if (depth > center->dmap_depth)
        return;

	centerindex *= 4;
	topindex *= 4;
	rightindex *= 4;
	bottomindex *= 4;
	leftindex *= 4;

	set_max_disp(dst, src, center, centerindex+1,
        top, topindex + top_e[0], top_e,
        center, centerindex+2, cright,
        center, centerindex+3, cbottom,
        left, leftindex+left_e[1], left_e,
        depth);

	set_max_disp(dst, src, center, centerindex+2,
        top, topindex + top_e[1], top_e,
        right, rightindex+right_e[0], right_e,
        center, centerindex+4, cbottom,
        center, centerindex+1, cleft,
        depth);

	set_max_disp(dst, src, center, centerindex+3,
        center, centerindex+1, ctop,
        center, centerindex+4, cright,
        bottom, bottomindex+bottom_e[1], bottom_e,
        left, leftindex+left_e[0], left_e,
        depth);

	set_max_disp(dst, src, center, centerindex+4,
        center, centerindex+2, ctop,
        right, rightindex+right_e[1], right_e,
        bottom, bottomindex+bottom_e[0], bottom_e,
        center, centerindex+3, cleft,
        depth);
}

static void
fill_dmap(char *buf, int len, int stride, float val1, float val2)
{
    float val;

    val = (val1 + val2) * 0.5f;
    buf[len*stride] = (int) (val + 0.5f);

    len >>= 1;
    if (len > 0) {
        fill_dmap(buf, len, stride, val1, val);
        fill_dmap(buf + len*stride, len, stride, val, val2);
    }
}

static void
average_dmap(char *buf1, int len1, int stride1, char *buf2, int len2,
    int stride2)
{
    len1 >>= 1;
    len2 >>= 1;
    if (len1 == 0)
        return;
    if (len2 == 0) {
		fill_dmap(buf1, len1, stride1, (float) buf2[stride2], (float) buf2[0]);
        return;
    }

    buf2[len2*stride2] = buf1[len1*stride1] =
        (buf1[len1*stride1] + buf2[len2*stride2]) >> 1;

    average_dmap(buf1, len1, stride1, buf2+len2*stride2, len2, stride2);
    average_dmap(buf1+len1*stride1, len1, stride1, buf2, len2, stride2);
}

SUBD_API void
subdDisplacementScale(float scale)
{
    _subd_dmap_scale = scale;
}

static void
fillDmapTexture(int d, int maxDepth, uchar *buf, int2 *loc, int x, int y,
            int w, uchar *srcBase, int srcIndex)
{
    uchar *dst;
    uchar *src;

    if (d > maxDepth)
        return;

    src = srcBase + srcIndex*8;
    dst = buf + (loc[d].y + y)*w + x + loc[d].x;
    dst[0] = src[4];
    dst[1] = src[5];
    dst[w] = src[6];
    dst[w+1] = src[7];

    srcIndex *= 4;
    x *= 2;
    y *= 2;
    fillDmapTexture(d+1, maxDepth, buf, loc, x, y, w, srcBase, srcIndex+1);
    fillDmapTexture(d+1, maxDepth, buf, loc, x+1, y, w, srcBase, srcIndex+2);
    fillDmapTexture(d+1, maxDepth, buf, loc, x, y+1, w, srcBase, srcIndex+3);
    fillDmapTexture(d+1, maxDepth, buf, loc, x+1, y+1, w, srcBase, srcIndex+4);
}

static void
fillDmapMaxTexture(int d, int maxDepth, uchar *buf, int2 *loc, int x, int y,
            int w, uchar *srcBase, int srcIndex)
{
    uchar *dst;
    uchar *src;

    if (d > maxDepth)
        return;

    src = srcBase + srcIndex*8;
    dst = buf + ((loc[d].y + y)*w + x + loc[d].x)*2;
    dst[0] = src[0];
    dst[1] = src[3];

    dst[2+1] = src[1];  // dst[2+LEFT] = src[RIGHT]
    dst[w*2+0] = src[2];  // dst[w*2+TOP] = src[BOTTOM]

    srcIndex *= 4;
    x *= 2;
    y *= 2;
    fillDmapMaxTexture(d+1, maxDepth, buf, loc, x, y, w, srcBase, srcIndex+1);
    fillDmapMaxTexture(d+1, maxDepth, buf, loc, x+1, y, w, srcBase, srcIndex+2);
    fillDmapMaxTexture(d+1, maxDepth, buf, loc, x, y+1, w, srcBase, srcIndex+3);
    fillDmapMaxTexture(d+1, maxDepth, buf, loc, x+1, y+1, w, srcBase, srcIndex+4);
}

SUBD_API void
subdDisplacementMap(uchar *disp_map, int disp_map_width,
	int disp_map_height, float *texcoordarray)
{
	int i, j;
    SubdMesh *m = current_mesh;
	face *f;
	char *buf, *bufptr, *bufptr2;
    uchar *q;
	int len;
	int len2;
	float2 t1, t2, t3, t4;
	float2 a, b, c;
	float h, v;
	float2 *tca = (float2 *) texcoordarray;
	edge *e, *e2;
	face *top_face, *right_face, *bottom_face, *left_face;
    int total_size;
	float width, height;
	float2 bbox[2];
    int topindex[2], rightindex[2], bottomindex[2], leftindex[2];
    int facenum;
    int faces;
    uchar *dmaph;
    int raw_size;
    int stride1, stride2;
    float pix1, pix2, pix3;
    int ix, iy;
    Patch *p;
    uchar *dmap_edge;

    // calc disp depth for each face

    total_size = 0;
	raw_size = 0;
    faces = 0;
    for (f = m->first_face; f; f = f->next) {
        faces++;
        e = f->e;
        e->dmap_ord = 1;
        e->next->dmap_ord = 2;
        e->next->next->dmap_ord = 4;
        e->next->next->next->dmap_ord = 3;

        if (f->valence != 4) {
            f->dmap_depth = 0;
        }
        else {
            t1 = f->e->texCoord;
            t2 = f->e->next->texCoord;
            t3 = f->e->next->next->texCoord;
            t4 = f->e->next->next->next->texCoord;

            // compute bounding box

            bbox[0].x = Min(Min(t1.x, t2.x), Min(t3.x, t4.x));
            bbox[1].x = Max(Max(t1.x, t2.x), Max(t3.x, t4.x));
            bbox[0].y = Min(Min(t1.y, t2.y), Min(t3.y, t4.y));
            bbox[1].y = Max(Max(t1.y, t2.y), Max(t3.y, t4.y));

            width = (bbox[1].x - bbox[0].x) * disp_map_width;
            height = (bbox[1].y - bbox[0].y) * disp_map_height;
            
            if (width < height)
                width = height;

            f->dmap_depth = 0;
            while ((1<<f->dmap_depth) < width)
                f->dmap_depth++;
            if (f->dmap_depth > MAXDDEPTH)
                f->dmap_depth = MAXDDEPTH;
        }

        f->dmap_offset = total_size;
        f->dmap_size = ((1<<(f->dmap_depth*2 + 1)) - 1) & 0x55555;
        total_size += f->dmap_size;
        f->dmap_raw_offset = raw_size;
        raw_size += ((1<<f->dmap_depth)+1) * ((1<<f->dmap_depth)+1);
    }

	buf = (char *) malloc(raw_size);
	dmaph = (uchar *) malloc(total_size);

	dmap_edge = (uchar *) malloc(total_size*8);

    facenum = 0;
	for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;
        f->dmap_ptr = dmap_edge + f->dmap_offset*8;

		// re-sample displacement texture

        t1 = f->e->texCoord;
        t2 = f->e->next->texCoord;
        t3 = f->e->next->next->texCoord;
        t4 = f->e->next->next->next->texCoord;

		bufptr = buf + f->dmap_raw_offset;
        len = 1<<f->dmap_depth;
		for (j = 0; j <= len; j++) {
			for (i = 0; i <= len; i++) {
				h = (float)i/len;
				if (i == len)
					h = 1.0f;
				v = (float)j/len;
				if (j == len)
					v = 1.0f;
				lerp(a, t2, t1, h);
				lerp(b, t3, t4, h);
				lerp(c, b, a, v);

                c.x *= disp_map_width;
                c.y *= disp_map_height;
                c.x -= 0.5f;
                c.y -= 0.5f;
                ix = (int) c.x;
                iy = (int) c.y;
                ix %= disp_map_width;
                iy %= disp_map_height;
                q = disp_map + iy*disp_map_width + ix;

                pix1 = lerp1((float) q[1], (float) q[0], c.x - (float) ix);
                pix2 = lerp1((float) q[disp_map_width+1],
                    (float) q[disp_map_width], c.x - (float) ix);
                pix3 = lerp1(pix2, pix1, c.y - (float) iy);

                *bufptr++ = (char) (pix3 - 128);
			}
		}
        facenum++;
	}

    // match corners of faces

    for (i = 0; i < m->vertices; i++) {
        m->vert_info[i].disp = 0;
        m->vert_info[i].num_disp = 0;
    }

	for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;

        bufptr = buf + f->dmap_raw_offset;

        len = 1<<f->dmap_depth;
        e = f->e;
        m->vert_info[e->index].num_disp++;
        m->vert_info[e->index].disp += bufptr[0];
        e = e->next;
        m->vert_info[e->index].num_disp++;
        m->vert_info[e->index].disp += bufptr[len];
        e = e->next;
        m->vert_info[e->index].num_disp++;
        m->vert_info[e->index].disp += bufptr[(len+2)*len];
        e = e->next;
        m->vert_info[e->index].num_disp++;
        m->vert_info[e->index].disp += bufptr[(len+1)*len];
    }

    for (i = 0; i < m->vertices; i++) {
        if (m->vert_info[i].num_disp > 1)
            m->vert_info[i].disp /= m->vert_info[i].num_disp;
    }

	for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;

        bufptr = buf + f->dmap_raw_offset;

        len = 1<<f->dmap_depth;
        e = f->e;
        bufptr[0] = m->vert_info[e->index].disp;
        e = e->next;
        bufptr[len] = m->vert_info[e->index].disp;
        e = e->next;
        bufptr[(len+2)*len] = m->vert_info[e->index].disp;
        e = e->next;
        bufptr[(len+1)*len] = m->vert_info[e->index].disp;
    }

    // match edge displacements

	for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;

        len = 1<<f->dmap_depth;
        e = f->e;
        do {
            e2 = findEdge(m, e);
            if (!e2)
                continue;
            len2 = 1<<e2->f->dmap_depth;
            if (len < len2 || e2->f->valence != 4)
                continue;
            bufptr2 = buf + e2->f->dmap_raw_offset;
			bufptr = buf + f->dmap_raw_offset;

            switch (e->dmap_ord) {
                case 1:
                        stride1 = 1;
                        break;
                case 2:
                        bufptr += len;
                        stride1 = len+1;
                        break;
                case 4:
                        bufptr += (len+2)*len;
                        stride1 = -1;
                        break;
                case 3:
                        bufptr += (len+1)*len;
                        stride1 = -(len+1);
                        break;
				default:
				printf("ERROR 1 %d\n", e->dmap_ord);
            }
            switch (e2->dmap_ord) {
                case 1:
                        stride2 = 1;
                        break;
                case 2:
                        bufptr2 += len2;
                        stride2 = len2+1;
                        break;
                case 4:
                        bufptr2 += (len2+2)*len2;
                        stride2 = -1;
                        break;
                case 3:
                        bufptr2 += (len2+1)*len2;
                        stride2 = -(len2+1);
                        break;
				default:
				printf("ERROR 2 %d\n", e->dmap_ord);
            }
            average_dmap((char *) bufptr, len, stride1,
                (char *)bufptr2, len2, stride2);
        } while ((e = e->next) != f->e);
    }

	for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;
        len = 1<<f->dmap_depth;
		// build displacement mipmap
		fill_disp_info(dmaph + f->dmap_offset, buf + f->dmap_raw_offset,
            len, len+1, 0, dmap_edge + f->dmap_offset*8);
    }

	free(buf);

    topindex[0] = topindex[1] = 0;
    rightindex[0] = rightindex[1] = 0;
    bottomindex[0] = bottomindex[1] = 0;
    leftindex[0] = leftindex[1] = 0;
	for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;
        top_face = 0;
        right_face = 0;
        bottom_face = 0;
        left_face = 0;
		e = findEdge(m, f->e);
        if (e) {
            top_face = e->f;
            topindex[0] = e->next->dmap_ord;
            topindex[1] = e->dmap_ord;
        }
		e = findEdge(m, f->e->next);
        if (e) {
            right_face = e->f;
            rightindex[0] = e->next->dmap_ord;
            rightindex[1] = e->dmap_ord;
        }
		e = findEdge(m, f->e->next->next);
        if (e) {
            bottom_face = e->f;
            bottomindex[0] = e->next->dmap_ord;
            bottomindex[1] = e->dmap_ord;
        }
		e = findEdge(m, f->e->next->next->next);
        if (e) {
            left_face = e->f;
            leftindex[0] = e->next->dmap_ord;
            leftindex[1] = e->dmap_ord;
        }

		set_max_disp(dmap_edge + 8*f->dmap_offset, dmaph,
			f, 0,
			top_face, 0, topindex,
			right_face, 0, rightindex,
			bottom_face, 0, bottomindex,
			left_face, 0, leftindex,
            0);
	}

    free(dmaph);

    // create texture maps with max displacement and displacement data

    // allocate displacement info in each patch

    static int2 zero2;
    int2 dmapLoc = zero2;
    int d;
    int size, maxSize;

    for (d = 0; d <= MAXDDEPTH; d++) {
        size = (1<<d) + 1;
        maxSize = 0;
        for (p = m->patchBuffer; p; p = p->nextPatch) {
            f = (face *) p->scratchPtr;
            p->dmapDepth = f->dmap_depth;
            if (d > p->dmapDepth)
                continue;
            if (dmapLoc.x + size > DMAP_TEXTURE_WIDTH) {
                dmapLoc.x = 0;
                dmapLoc.y += maxSize;
            }
            p->dmapLoc[d] = dmapLoc;
            dmapLoc.x += size;
            maxSize = max(maxSize, size);
        }
    }
    m->dmapSize.x = DMAP_TEXTURE_WIDTH;
    m->dmapSize.y = dmapLoc.y + maxSize;
    if (m->dmapTexture)
        free(m->dmapTexture);
    if (m->dmapMaxTexture)
        free(m->dmapMaxTexture);
    m->dmapTexture = (uchar *) malloc(m->dmapSize.x * m->dmapSize.y);
    m->dmapMaxTexture = (uchar *) calloc(m->dmapSize.x * m->dmapSize.y * 2, 1);

    // copy displacement information into buffer for loading texture

    for (p = m->patchBuffer; p; p = p->nextPatch) {
        f = (face *) p->scratchPtr;
        fillDmapTexture(0, p->dmapDepth, m->dmapTexture, p->dmapLoc,
            0, 0, DMAP_TEXTURE_WIDTH, f->dmap_ptr, 0);
        fillDmapMaxTexture(0, p->dmapDepth, m->dmapMaxTexture, p->dmapLoc,
            0, 0, DMAP_TEXTURE_WIDTH, f->dmap_ptr, 0);
    }
    free(dmap_edge);
    m->newDisplacement = true;
    LoadMeshTextures(NULL, 0, 0,
        NULL, 0, 0,
        NULL, 0,
        m->dmapTexture, m->dmapSize.x, m->dmapSize.y,
        m->dmapMaxTexture, m->dmapSize.x, m->dmapSize.y);
}
