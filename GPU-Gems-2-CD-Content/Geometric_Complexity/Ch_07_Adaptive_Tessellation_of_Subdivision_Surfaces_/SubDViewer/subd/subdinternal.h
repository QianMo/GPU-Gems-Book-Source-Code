
// subdinternal.h

#define MAXVALENCE  8
#define MAXDDEPTH   5
#define MAXDEPTH    6

#define MAXGROUPS   32

#define PATCH_BUFFER_WIDTH  512
#define EP_BUFFER_WIDTH     PATCH_BUFFER_WIDTH
#define EP_BUFFER_HEIGHT    256
#define SEGMENT_SIZE        68

#define EP_HEIGHT   (2*(MAXVALENCE)+1)

typedef unsigned char uchar;

typedef struct short4 {
    int x, y, z, w;
} short4;

typedef struct {
	float x, y, z, w;
} float4;

typedef struct {
	float x, y, z;
} float3;

typedef struct {
	float x, y;
} float2;

typedef struct {
	float x, y, z, w, q;
} float5;

typedef struct {
	float x, y, z, w, u, v;
} float6;

typedef struct byte4 {
    unsigned char x, y, z, w;
} byte4;

typedef struct int2 {
    int x, y;
} int2;

typedef struct int4 {
    int x, y, z, w;
} int4;

typedef struct Patch {
    int2 loc;
    char ep;
    char group;
    char epValence[4];
    uchar *flatPtr[MAXDEPTH];
    int2 indexLoc;
    int2 texCoordLoc;
    int2 tanLoc;
    char dmapDepth;
    int2 dmapLoc[MAXDDEPTH+1];
    struct Patch *nextPatch;
    struct Patch *next;
    void *scratchPtr;
} Patch;

typedef struct GroupInfo {
    int epCount[6];
    int epTotal;
    int epTangentW;
    int epTangentH;
} GroupInfo;

typedef float4 pvector;

typedef unsigned int uint;

extern float *_subd_vertex_list[2];
extern int _subd_vertex_bufobj[2];
extern float *_subd_vertex_ptr;
extern int _subd_vertex_list_size;
extern float _subd_flip_bump;
extern int _subd_output_vertex_size;
extern int _subd_output_texcoord;
extern int _subd_adaptive_flag;
extern int _subd_subdiv_level;
extern int _subd_cull_face;
extern float _subd_flat_scale, _subd_flat_scale2;
extern float _subd_near_len1, _subd_near_len2;
extern int _subd_splinemode_flag;
extern int _subd_test_flag;
extern int _subd_ortho_flag;
extern int _subd_emit_texcoords;
extern bool _subd_stop_tessellating;
extern float *_subd_vertex_end;
extern struct frustum {
    float l, r, b, t, n, f;
} _subd_frustum;
extern float _subd_dmap_scale, _subd_dmap_scale1, _subd_dmap_scale2, _subd_dmap_scale_x_2;
extern void _subd_emit_quad(int i1, int i2, int i3, int i4);

#define EP1 	1
#define EP2		2
#define EP3		4
#define EP4		8

#define TRIANGLE_FACE 16
#define MIRROR_VERTS    32
#define EP_MIRROR_VERTS 64
#define TEXCOORDS       128

#define TOP_SHIFT       0
#define LEFT_SHIFT     8
#define BOTTOM_SHIFT    16
#define RIGHT_SHIFT      24

#define TOP_MASK        (255<<(TOP_SHIFT))
#define RIGHT_MASK      (255<<(RIGHT_SHIFT))
#define BOTTOM_MASK     (255<<(BOTTOM_SHIFT))
#define LEFT_MASK       (255<<(LEFT_SHIFT))


#define vadd(d, s1) { \
	(d).x += (s1).x; \
	(d).y += (s1).y; \
	(d).z += (s1).z; \
    (d).w += (s1).w; \
}

#define vsub(d, s1, s2) { \
	(d).x = (s1).x - (s2).x; \
	(d).y = (s1).y - (s2).y; \
	(d).z = (s1).z - (s2).z; \
    (d).w = (s1).w - (s2).w; \
}

#define vadd2(d, s1, s2) { \
	(d).x = (s1).x + (s2).x; \
	(d).y = (s1).y + (s2).y; \
	(d).z = (s1).z + (s2).z; \
    (d).w = (s1).w + (s2).w; \
}

#define MID(d, s1, s2) { \
	(d).x = ((s1).x + (s2).x)*0.5f; \
	(d).y = ((s1).y + (s2).y)*0.5f; \
	(d).z = ((s1).z + (s2).z)*0.5f; \
    (d).w = ((s1).w + (s2).w)*0.5f; \
}

#define LERP(d, s1, s2, ww) { \
	(d).x = (s2).x + ((s1).x - (s2).x)*(ww); \
	(d).y = (s2).y + ((s1).y - (s2).y)*(ww); \
	(d).z = (s2).z + ((s1).z - (s2).z)*(ww); \
    (d).w = (s2).w + ((s1).w - (s2).w)*(ww); \
}

#define MID4(d, s1, s2, s3, s4) { \
	(d).x = ((s1).x + (s2).x + (s3).x + (s4).x)*0.25f; \
	(d).y = ((s1).y + (s2).y + (s3).y + (s4).y)*0.25f; \
	(d).z = ((s1).z + (s2).z + (s3).z + (s4).z)*0.25f; \
    (d).w = ((s1).w + (s2).w + (s3).w + (s4).w)*0.25f; \
}

#define vdiv(d, s1) { \
	float tt = 1.0f/(s1); \
	(d).x *= tt; \
	(d).y *= tt; \
	(d).z *= tt; \
    (d).w *= tt; \
}

#define vmuls(d, s1) { \
	(d).x *= (s1); \
	(d).y *= (s1); \
	(d).z *= (s1); \
    (d).w *= (s1); \
}

#define vmul(d, s1) { \
	(d).x *= (s1).x; \
	(d).y *= (s1).y; \
	(d).z *= (s1).z; \
    (d).w *= (s1).w; \
}

#define vmulacc(d, f, s1) { \
	(d).x += (f).x*(s1); \
	(d).y += (f).y*(s1); \
	(d).z += (f).z*(s1); \
    (d).w += (f).w*(s1); \
}

#define vmad(d, s1, s2, s3) { \
	(d).x = (s1).x*(s2) + (s3).x; \
	(d).y = (s1).y*(s2) + (s3).y; \
	(d).z = (s1).z*(s2) + (s3).z; \
    (d).w = (s1).w*(s2) + (s3).w; \
}

#define dot(a, b) ((a).x*(b).x + (a).y*(b).y + (a).z*(b).z)

#define veq(a, b) ((a).x==(b).x && (a).y==(b).y && (a).z==(b).z)

#define cross(d, v1, v2) { \
    (d).x = (v1).y*(v2).z - (v1).z*(v2).y; \
    (d).y = (v1).z*(v2).x - (v1).x*(v2).z; \
    (d).z = (v1).x*(v2).y - (v1).y*(v2).x; \
}

#define ABS(x) (((x) < 0) ? -(x) : (x))

#define TOP 	1
#define LEFT 	2
#define BOTTOM 	4
#define RIGHT	8
#define FRONT	16
#define BACK	32
#define CLIP_ENABLE 128

#ifndef _subd_max
#define _subd_max(a, b) ((a)>(b) ? (a) : (b))
#endif
