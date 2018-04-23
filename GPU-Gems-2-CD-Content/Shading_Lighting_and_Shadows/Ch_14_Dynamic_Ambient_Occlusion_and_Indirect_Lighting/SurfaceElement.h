// SurfaceElement.h

extern void seBeginMesh();
extern void seEndMesh(float3 *vertexArray, float2 *tcArray);
extern void seVertex(int vindex, int vtindex);
extern void seFace();
extern void CreateElements();

#define SURFACE_SHADER              1
#define OCCLUSION_SHADER            2
#define OCCLUSION2_SHADER           3

extern PFNGLACTIVETEXTUREARBPROC glActiveTexture;
extern PFNGLMULTITEXCOORD4IARBPROC glMultiTexCoord4i;
extern PFNGLMULTITEXCOORD4FARBPROC glMultiTexCoord4f;
extern PFNWGLRELEASETEXIMAGEARBPROC wglReleaseTexImageARB;
extern PFNWGLBINDTEXIMAGEARBPROC wglBindTexImageARB;

extern int init_shader(char *name, int id, int shaderType);
extern void set_shader(int id, int shaderType);
extern void calcAmbientOcclusion(int passes);