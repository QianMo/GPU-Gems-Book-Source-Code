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

typedef unsigned char uchar;

float _subd_flip_bump = 1;
int _subd_output_vertex_size = 3;
float _subd_dmap_scale = 1.10f/128.0f;
float _subd_dmap_scale_x_2;
float _subd_dmap_scale1, _subd_dmap_scale2;
float _subd_flat_scale, _subd_flat_scale2;
float _subd_near_len1, _subd_near_len2;
static float flat_distance = 1.0f;
struct dmapinfo *_subd_dmap_info;
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
float *_subd_vertex_list;
static int vertex_list_length, prim_list_length;
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
int _subd_sse_flag = 0;
int _subd_stop_tessellating;
static struct chkpt {
    uint *data1;
    int data2;
    float *vertex_ptr;
    int *prim_ptr;
} checkpoint;

static pvector *vertexarray;
static float2 *texcoordarray;
static float *edgeweightarray;
static int index_size = SUBD_INDEX32;

extern void tess_GPU(
	Patch *patchBuffer, pvector *vlist, int vertices, int faces,
	unsigned char *patchIndexBuffer, int patchIndexW,
    int patchIndexH, float4 *texCoordBuffer, int texCoordW, int texCoordH);

typedef struct edge {
	int index;
	int tindex;
	pvector *v;
	struct edge *next;
	struct edge *another;
	struct face *f;
    int dmap_ord;
    float2 texCoord, texCoord2;
} edge;

typedef struct face {
	edge *e;
    int index;
	int valence;
	int size;
	int flags;
    int vertexIndex[16];
    int epIndex[4][8];
    int vertexValence[4];
    int ep;
    float4 tcoord[4];
    int dmap_depth;
    int dmap_offset;
	int dmap_raw_offset;
    int dmap_size;
	struct face *next;
} face;

typedef struct {
	char *address;
	int length;
	int max_length;
} subdBuffer;

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
	struct vertinfo {
		char valence;
		char hidden_faces;
		char missing_faces;
        char not_shareable;
        int tindex;
        int disp;
        int num_disp;
        int mirror_vertex;
	}*vert_info;
	face *current_face;
	int faces, patches;
	int surface_type;
	char *vert_flags;
	char *vert_clip_mask;
	float2 *tc_data;
    struct SubdMesh *next;
    uchar *dmap_edge;
	float dmap_scale;
    struct dmapinfo *dmap_info;
    int maxVertexIndex;
    bool newIndexBuffer;
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

// calculate output_vertex_size based on output flags
static void calc_vertex_size()
{
    _subd_output_vertex_size = 3;
    if (_subd_output_normal)
        _subd_output_vertex_size += 3;
    if (_subd_output_texcoord)
        _subd_output_vertex_size += 2;
    if (_subd_output_tangent)
        _subd_output_vertex_size += 4;
    if (_subd_output_vertex_size < 6)
        _subd_output_vertex_size = 6;
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
            calc_vertex_size();
            return _subd_output_vertex_size * sizeof(float);
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
    m->vert_info = (SubdMesh::vertinfo *) m->vv_buffer.address;
}

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
    if (m->dmap_info)
        free(m->dmap_info);
    if (m->dmap_edge)
        free(m->dmap_edge);

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
#define PATCH_INDEX_WIDTH   1024
unsigned char patchIndexBuffer[PATCH_INDEX_HEIGHT][PATCH_INDEX_WIDTH][2];
#define TEXCOORD_WIDTH  512
#define TEXCOORD_HEIGHT 256
float4 texCoordBuffer[TEXCOORD_HEIGHT][TEXCOORD_WIDTH];
extern float tangent_mask3[];
extern float tangent_mask4[];
extern float tangent_mask5[];
extern float tangent_mask6[];
extern float tangent_mask7[];
extern float tangent_mask8[];

static float2
calcTexCoordTan(SubdMesh *m, edge *eIn, int t1, int flip)
{
    float2 vtc;
    float2 etc[MAXVALENCE];
    float2 ftc[MAXVALENCE];
    static float2 vzero;
    int valence;
    int i, j;
    edge *e;
    float *tTable;
    float2 t;
    float len;
    float *tangentList[6] = {
        tangent_mask3, tangent_mask4, tangent_mask5, tangent_mask6,
        tangent_mask7, tangent_mask8 };

    valence = m->vert_info[eIn->index].valence;
    if (valence < 3 || valence > 8)
        return vzero;

    vtc = eIn->texCoord;
    for (i = 0; i < valence; i++)
        etc[i] = ftc[i] = vtc;
    for (i = 0, e = eIn; i < valence; i++) {
        etc[i] = e->next->texCoord;
        ftc[i] = e->next->next->texCoord;
        if (i + 1 < valence)
            ftc[i+1] = etc[i+1] = e->next->next->next->texCoord;
        e = findEdge(m, e->next->next->next);
        if (!e || e->texCoord.x != vtc.x || e->texCoord.y != vtc.y)
            break;
    }

    t.x = etc[0].x - etc[2].x;
    t.y = etc[1].x - etc[3].x;

    if (i < valence) {  // did not finish - go around the other way
        for (i = valence-1, e = eIn; i; i--) {
            e = findEdge(m, e);
            if (!e)
                break;
            e = e->next;
            printf("e %f %f e->next %f %f\n", e->texCoord.x, e->texCoord.y, e->next->texCoord.x, e->next->texCoord.y);
            if (e->texCoord.x != vtc.x || e->texCoord.y != vtc.y)
                break;
            etc[i] = e->next->texCoord;
            ftc[i] = e->next->next->texCoord;
        }
    }
#if 0 
    tTable = tangentList[valence-3];
    t = vzero;
    for (i = 0; i < valence; i++) {
        j = i - t1;
        if (j < 0)
            j += valence;
        t.x += etc[i].x * tTable[j] + ftc[i].x * tTable[j+valence];
        t.y += etc[i].y * tTable[j] + ftc[i].y * tTable[j+valence];
    }
    
e = eIn;

    if (t1) {
        t.x = e->next->next->next->texCoord.x - e->texCoord.x;
    }
    else {
t.x = e->next->texCoord.x - e->texCoord.x;
t.y = e->next->texCoord.y - e->texCoord.y;
}
if (valence == 3) printf("Compare %f %f\n", t.x, t.y);

        len = (float) sqrt(t.x*t.x + t.y*t.y);

    if (len > 0.0f) {
        t.x /= len;
        t.y /= len;
    }

    t.x *= flip;
    t.y *= flip;
#endif
    return t;
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
    int i, j, ii;
    edge *e, *e2, *e3;
    int valence;
    int faces;
    Patch *p, *lastp;
    int x, y;
    int w;
    int n;

    static int indexPos[4][4] = {
        5, 4, 0, 1,
        6, 2, 3, 7,
        10, 11, 15, 14,
        9, 13, 12, 8,
    };
    static int leftEPIndex[4][4] = {
        7, 3, 6, 2, 5, 1, 4, 0,
        11, 15, 10, 14, 9, 13, 8, 12
    };

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
        for (ii = 0, e = f->e; ii < 4; ii++, e = e->next) {
            static int swap23[4] = { 0, 1, 3, 2 };
            valence = m->vert_info[e->index].valence;
            if (m->vert_info[e->index].missing_faces) {
                valence += 2;
            }
            if (valence == 4 || valence < 3 || valence > 8)
                continue;
            i = swap23[ii];

            f->vertexValence[i] = valence;
            switch (i) {
                case 0: f->ep |= EP1; break;
                case 1: f->ep |= EP2; break;
                case 2: f->ep |= EP3; break;
                case 3: f->ep |= EP4; break;
            }
            e3 = e;
            e2 = findEdge(m, e_before(e));
            for (j = 2; j < valence; j++) {
                if (e2) {
                    if (j >= 4) {
                        f->epIndex[i][j-4+4] = e2->next->index;
                        f->epIndex[i][j-4] = e2->next->next->index;
                    }
                    e3 = e2;
                    e2 = findEdge(m, e_before(e2));
                }
                else {
                    if (j >= 4) {
                        e2 = e_before(e3);
                        f->epIndex[i][j-4+4] = f->epIndex[i][j-4] = e2->index;
                    }
                    j++;
                    e2 = rewindEdge(m, e);
                    if (j >= 4 && j < valence) {
                        f->epIndex[i][j-4+4] = e->index;
                        f->epIndex[i][j-4] = e2->next->index;
                    }
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
        e = f->e;
        e->texCoord2 = calcTexCoordTan(m, e, 0, 1);
        e = e->next;
        e->texCoord2 = calcTexCoordTan(m, e, 1, -1);
        e = e->next;
        e->texCoord2 = calcTexCoordTan(m, e, 0, -1);
        e = e->next;
        e->texCoord2 = calcTexCoordTan(m, e, 1, 1);
    }

    m->patchBuffer = (Patch *) calloc(faces * sizeof(Patch), 1);
    p = m->patchBuffer;
    lastp = NULL;
    x = 0, y = 0;
    m->maxVertexIndex = -1;
    for (f = m->first_face; f; f = f->next) {
        if (f->valence != 4 || (f->flags & HIDE_FACE))
            continue;
       
        p->nextPatch = p + 1;
        p->initialSize = 4;
        p->scratchPtr = (void *) f;
        p->ep = f->ep;
        p->epValence[0] = f->vertexValence[0];
        p->epValence[1] = f->vertexValence[1];
        p->epValence[2] = f->vertexValence[2];
        p->epValence[3] = f->vertexValence[3];
        p->epLeft = max(0, max(p->epValence[0], p->epValence[2]) - 4);
        p->epRight = max(0, max(p->epValence[1], p->epValence[3]) - 4);

        if (x + p->epLeft + p->epRight + p->initialSize > PATCH_INDEX_WIDTH) {
            x = 0;
            y += 4;
        }
        x += p->epLeft;
        p->indexLoc.x = x;
        p->indexLoc.y = y;

        for (j = 0; j < 4; j++) {
            for (i = 0; i < 4; i++) {
                t = f->vertexIndex[j*4+i];
                if (t > m->maxVertexIndex)
                    m->maxVertexIndex = t;
                patchIndexBuffer[y+j][x+i][1] = t>>8;
                patchIndexBuffer[y+j][x+i][0] = t & 255;
            }
        }
        n = p->epValence[0]-4;
        for (i = 0; i < n; i++) {
            t = f->epIndex[0][i];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y][x-i-1][1] = t>>8;
            patchIndexBuffer[y][x-i-1][0] = t & 255;
            t = f->epIndex[0][i+4];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y+1][x-i-1][1] = t>>8;
            patchIndexBuffer[y+1][x-i-1][0] = t & 255;
        }

        n = p->epValence[2]-4;
        for (i = 0; i < n; i++) {
            t = f->epIndex[2][i];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y+3][x-i-1][1] = t>>8;
            patchIndexBuffer[y+3][x-i-1][0] = t & 255;
            t = f->epIndex[2][i+4];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y+2][x-i-1][1] = t>>8;
            patchIndexBuffer[y+2][x-i-1][0] = t & 255;
        }

        n = p->epValence[1]-4;
        for (i = 0; i < n; i++) {
            t = f->epIndex[1][i];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y][x+4+i][1] = t>>8;
            patchIndexBuffer[y][x+4+i][0] = t & 255;
            t = f->epIndex[1][i+4];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y+1][x+4+i][1] = t>>8;
            patchIndexBuffer[y+1][x+4+i][0] = t & 255;
        }

        n = p->epValence[3]-4;
        for (i = 0; i < n; i++) {
            t = f->epIndex[3][i];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y+3][x+4+i][1] = t>>8;
            patchIndexBuffer[y+3][x+4+i][0] = t & 255;
            t = f->epIndex[3][i+4];
            if (t > m->maxVertexIndex)
                m->maxVertexIndex = t;
            patchIndexBuffer[y+2][x+4+i][1] = t>>8;
            patchIndexBuffer[y+2][x+4+i][0] = t & 255;
        }
        
        x += p->initialSize + p->epRight;
        lastp = p++;
    }
    if (lastp)
        lastp->nextPatch = NULL;

    // create texture coordinate buffer

    x = 0;
    y = 0;
    for (p = m->patchBuffer; p; p = p->nextPatch) {
        f = (face *) p->scratchPtr;
        w = 2;
        if (p->ep & EP1)
            w++;
        if (p->ep & EP2)
            w++;
        if (p->ep & EP3)
            w++;
        if (p->ep & EP4)
            w++;
        if (x + w > TEXCOORD_WIDTH) {
            x = 0;
            y += 2;
        }
        p->texCoordLoc.x = x;
        p->texCoordLoc.y = y;
        texCoordBuffer[y][x].x = FtoS(f->e->texCoord.x);
        texCoordBuffer[y][x].y = FtoS(f->e->texCoord.y);
        texCoordBuffer[y][x].z = FtoS(f->e->texCoord2.x);
        texCoordBuffer[y][x].w = FtoS(f->e->texCoord2.y);
//printf("TexCoordBuffer %d %d is %f %f %f %f)\n", x, y, f->e->texCoord.x, f->e->texCoord.y, f->e->texCoord2.x, f->e->texCoord2.y);

        texCoordBuffer[y][x+1].x = FtoS(f->e->next->texCoord.x);
        texCoordBuffer[y][x+1].y = FtoS(f->e->next->texCoord.y);
        texCoordBuffer[y][x+1].z = FtoS(-f->e->next->texCoord2.y);
        texCoordBuffer[y][x+1].w = FtoS(f->e->next->texCoord2.x);

        texCoordBuffer[y+1][x].x = FtoS(f->e->next->next->next->texCoord.x);
        texCoordBuffer[y+1][x].y = FtoS(f->e->next->next->next->texCoord.y);
        texCoordBuffer[y+1][x].z = FtoS(f->e->next->next->next->texCoord2.y);
        texCoordBuffer[y+1][x].w = FtoS(-f->e->next->next->next->texCoord2.x);

        texCoordBuffer[y+1][x+1].x = FtoS(f->e->next->next->texCoord.x);
        texCoordBuffer[y+1][x+1].y = FtoS(f->e->next->next->texCoord.y);
        texCoordBuffer[y+1][x+1].z = FtoS(-f->e->next->next->texCoord2.x);
        texCoordBuffer[y+1][x+1].w = FtoS(-f->e->next->next->texCoord2.y);

        x += w;
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
}

SUBD_API void subdOutputVertexBuffer(void *vb, int size)
{
    _subd_vertex_list = (float *) vb;
    vertex_list_length = ((uint)size)/sizeof(*_subd_vertex_list);
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
    uint *mesh_buffer;
    int patches, faces;

	if (!m)
		return SUBD_DONE;

    calc_vertex_size();

	_subd_vertex_ptr = _subd_vertex_list;
	prim_ptr = prim_list;

	_subd_vertex_end = _subd_vertex_ptr + vertex_list_length - MAX_VERTEX_SIZE;
    prim_end = prim_ptr + (prim_list_length>>2) - MAX_PRIMITIVE_SIZE;

	vertex_index = 0;

    if (!_subd_ortho_flag)
        _subd_flat_scale = z_sign * (frustum.b - frustum.t)*flat_distance /
            (vp_height*frustum.n);
    else
        _subd_flat_scale = (frustum.b - frustum.t)*flat_distance /
            vp_height;

    if (state == SUBD_CONTINUE) {
        mesh_buffer = checkpoint.data1;
        patches = faces = checkpoint.data2;
        if (_subd_dmap_scale != 0.0f && m->dmap_info)
            _subd_dmap_info = m->dmap_info + (m->faces - faces);
        else
            _subd_dmap_info = 0;
        if (clip_flag)
            clip_flag = 2;
    }
    else {
        patches = m->patches;
        faces = m->faces;
        mesh_buffer = (uint *) m->mesh_buffer;
        if (_subd_dmap_scale == 0.0f)
            _subd_dmap_info = 0;
        else
            _subd_dmap_info = m->dmap_info;
        if (clip_flag)
            clip_flag = 1;
    }

    _subd_stop_tessellating = 0;
	
	_subd_flat_scale *= -8.0f/3.0f;
	_subd_flat_scale *= 3.0f;
    _subd_dmap_scale1 = _subd_dmap_scale * 8.0f;
    _subd_dmap_scale2 = _subd_dmap_scale1 * 8.0f;
    _subd_dmap_scale_x_2 = _subd_dmap_scale * 2.0f * 128 * z_sign;
	_subd_flat_scale2 = _subd_flat_scale*0.50f;

    _subd_near_len1 = 2*_subd_flat_scale * frustum.n * z_sign;
    _subd_near_len2 = 2*_subd_flat_scale2 * frustum.n * z_sign;

    tess_GPU(m->patchBuffer, vertexarray, m->maxVertexIndex+1, faces,
            m->newIndexBuffer ? &patchIndexBuffer[0][0][0] : 0,
            PATCH_INDEX_WIDTH, PATCH_INDEX_HEIGHT,
            m->newIndexBuffer ? &texCoordBuffer[0][0] : 0,
            TEXCOORD_WIDTH, TEXCOORD_HEIGHT);
    m->newIndexBuffer = false;

    if (_subd_stop_tessellating) {
        _subd_vertex_ptr = checkpoint.vertex_ptr;
        prim_ptr = checkpoint.prim_ptr;
    }

    *vlist_length = (_subd_vertex_ptr - _subd_vertex_list)/_subd_output_vertex_size;
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

int _subd_Checkpoint(uint *data1, int data2)
{
    if (_subd_stop_tessellating)
        return 1;

    checkpoint.data1 = data1;
    checkpoint.data2 = data2;
    checkpoint.vertex_ptr = _subd_vertex_ptr;
    checkpoint.prim_ptr = prim_ptr;

    return 0;
}

int
_subd_emit_vertex(pvector *v, pvector *n, float5 *tc)
{
	if (_subd_vertex_ptr >= _subd_vertex_end || _subd_stop_tessellating) {
        _subd_stop_tessellating = 1;
        return 0;
    }

    _subd_vertex_ptr[0] = v->x;
    _subd_vertex_ptr[1] = v->y;
    _subd_vertex_ptr[2] = v->z;

	_subd_vertex_ptr[3] = n->x;
	_subd_vertex_ptr[4] = n->y;
	_subd_vertex_ptr[5] = n->z;

	if (_subd_output_vertex_size >= 8 && _subd_emit_texcoords) {
		_subd_vertex_ptr[6] = tc->x;
		_subd_vertex_ptr[7] = tc->y;
        if (_subd_output_vertex_size >= 11) {
            _subd_vertex_ptr[8] = tc->z;
			_subd_vertex_ptr[9] = tc->w;
            _subd_vertex_ptr[10] = tc->q;
            _subd_vertex_ptr[11] = _subd_flip_bump;
        }
	}

	_subd_vertex_ptr += _subd_output_vertex_size;

	return vertex_index++;
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

// Displacement map support

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

#define MAXDDEPTH 6

static void
fill_dmap(char *buf, int len, int stride, int val1, int val2)
{
    int val;

    val = (val1 + val2) >> 1;
    buf[len*stride] = val;

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
		fill_dmap(buf1, len1, stride1, buf2[stride2], buf2[0]);
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
    _subd_dmap_scale = scale/128.0f;
}

SUBD_API void
subdDisplacementMap(uchar *disp_map, int disp_map_width,
	int disp_map_height, float *texcoordarray)
{
	int i, j;
    SubdMesh *m = current_mesh;
	face *f;
	char *buf, *bufptr, *bufptr2;
    uchar *p;
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
            if (f->e->tindex < 0) {
                t1.x = 0.0f; t1.y = 0.0f;
                t2 = t3 = t4 = t1;
            }
            else {
                t1 = tca[f->e->tindex];
                t2 = tca[f->e->next->tindex];
                t3 = tca[f->e->next->next->tindex];
                t4 = tca[f->e->next->next->next->tindex];
            }

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

    if (m->dmap_edge)
        free(m->dmap_edge);
	m->dmap_edge = (uchar *) malloc(total_size*8);
    if (m->dmap_info)
        free(m->dmap_info);
    m->dmap_info = (struct dmapinfo *) malloc(faces*sizeof(struct dmapinfo));
		
    facenum = 0;
	for (f = m->first_face; f; f = f->next) {
        if (!f->size || (f->flags & HIDE_FACE))
            continue;
        m->dmap_info[facenum].ptr = m->dmap_edge + f->dmap_offset*8;
        m->dmap_info[facenum].depth = f->dmap_depth;

		// re-sample displacement texture

        if (f->e->tindex < 0) {
            t1.x = 0.0f; t1.y = 0.0f;
            t2 = t3 = t4 = t1;
        }
        else {
            t1 = tca[f->e->tindex];
            t2 = tca[f->e->next->tindex];
            t3 = tca[f->e->next->next->tindex];
            t4 = tca[f->e->next->next->next->tindex];
        }
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
                p = disp_map + iy*disp_map_width + ix;

                pix1 = lerp1((float) p[1], (float) p[0], c.x - (float) ix);
                pix2 = lerp1((float) p[disp_map_width+1],
                    (float) p[disp_map_width], c.x - (float) ix);
                pix3 = lerp1(pix2, pix1, c.y - (float) iy);

                *bufptr++ = (char) (pix3 - 128);
			}
		}
        facenum++;
	}
if (facenum != m->faces)
printf("ERROR: Faces %d facenum %d\n", m->faces, facenum);

    // match corners of faces

    for (i = 0; i < m->vertices; i++) {
        m->vert_info[i].disp = 0;
        m->vert_info[i].num_disp = 0;
    }

	for (f = m->first_face; f; f = f->next) {
        if (!f->size || (f->flags & HIDE_FACE))
            continue;
        if (f->valence != 4)
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
        if (!f->size || (f->flags & HIDE_FACE))
            continue;
        if (f->valence != 4)
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
        if (!f->size || (f->flags & HIDE_FACE))
            continue;
        if (f->valence != 4)
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
        if (!f->size || (f->flags & HIDE_FACE))
            continue;
        len = 1<<f->dmap_depth;
		// build displacement mipmap
		fill_disp_info(dmaph + f->dmap_offset, buf + f->dmap_raw_offset,
            len, len+1, 0, m->dmap_edge + f->dmap_offset*8);
    }

	free(buf);

    topindex[0] = topindex[1] = 0;
    rightindex[0] = rightindex[1] = 0;
    bottomindex[0] = bottomindex[1] = 0;
    leftindex[0] = leftindex[1] = 0;
	for (f = m->first_face; f; f = f->next) {
        if (!f->size || (f->flags & HIDE_FACE))
            continue;
		if (f->valence != 4) {
            uchar *p = m->dmap_edge + 8*f->dmap_offset;
            p[0] = p[1] = p[2] = p[3] = 0;
            p[4] = p[5] = p[6] = p[7] = 0;
			continue;
        }
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

		set_max_disp(m->dmap_edge + 8*f->dmap_offset, dmaph,
			f, 0,
			top_face, 0, topindex,
			right_face, 0, rightindex,
			bottom_face, 0, bottomindex,
			left_face, 0, leftindex,
            0);
	}

    free(dmaph);
}