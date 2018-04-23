
#if 0
#ifdef SUBD_EXPORTS
#define SUBD_API __declspec(dllexport)
#else
#define SUBD_API __declspec(dllimport)
#endif
#else
#define SUBD_API
#endif

SUBD_API int fnSubd(void);
typedef enum SubdEnum {
    SUBD_ADAPTIVE,
    SUBD_OUTPUT_NORMAL,
    SUBD_OUTPUT_TEXCOORD,
    SUBD_OUTPUT_TANGENT,
    SUBD_EDGES_ONLY,
    SUBD_CULL_FACE,
    SUBD_FRONT,
    SUBD_BACK,
    SUBD_NONE,
    SUBD_CLIP,
    SUBD_DOO_SABIN,
    SUBD_CATMULL_CLARK,
    SUBD_FLAT_DISTANCE,
    SUBD_SUBDIVIDE_DEPTH,
    SUBD_OUTPUT_VERTEX_SIZE,
    SUBD_TEXCOORD_INDEX,
    SUBD_EDGEWEIGHT_INDEX,
    SUBD_CURRENT_MESH,
    SUBD_HIDE_FACE,
    SUBD_SURFACE_TYPE,
    SUBD_OUTPUT_PRIMITIVE,
    SUBD_TRIANGLES,
    SUBD_QUADS,
    SUBD_LINES,
    SUBD_START,
    SUBD_CONTINUE,
    SUBD_DONE,
    SUBD_INDEX_SIZE,
    SUBD_INDEX16,
    SUBD_INDEX32,
	SUBD_PARTIAL_MESH_ID
} SubdEnum;


extern SUBD_API void subdSet(SubdEnum e, int value);
extern SUBD_API int subdGet(SubdEnum e);
extern SUBD_API void subdSetf(SubdEnum e, float v);
extern SUBD_API float subdGetf(SubdEnum e);

extern SUBD_API void subdBeginMesh();
extern SUBD_API void subdFace();
extern SUBD_API void subdVertex(int index);
extern SUBD_API void subdEndMesh();
extern SUBD_API void subdDeleteMesh();

extern SUBD_API void subdCtrlMeshVertexPointer(int size, int stride, float *p);
extern SUBD_API void subdCtrlMeshTexCoordPointer(int size, int stride,float *p);
extern SUBD_API void subdCtrlMeshEdgeWeightPointer(int stride, float *p);

extern SUBD_API SubdEnum subdTessellate(int *vlist_length, int *primlist_length,
    SubdEnum start_or_continue);

extern SUBD_API void subdFrustum(float l, float r, float b, float t, float n, float f);
extern SUBD_API void subdOrtho(float l, float r, float b, float t, float n, float f);
extern SUBD_API void subdViewport(int x, int y, int width, int height);
extern SUBD_API void subdOutputVertexBuffer(void *vb, int size, int n, int bufObj);
extern SUBD_API void subdOutputIndexBuffer(void *ib, int size);
extern SUBD_API void subdDisplacementMap(unsigned char *disp_map,
    int width, int height, float *texcoordarray);
extern SUBD_API void subdDisplacementScale(float scale);
extern SUBD_API void subdExtractMesh(int id);
