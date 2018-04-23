//
// DynamicAO.cpp
//
//      Sample code that demonstrates dynamic ambient occlusion calculated using the GPU for
//          polygon meshes. It uses wavefront .obj format object files to describe
//          the mesh. Also it supports reading a list of object files from a text file
//          for doing simple keyframe animation
//
//          The meshes must be uv mapped since that information is used to create the
//          surface element hierarchy used in our ambient occlusion algorithm
//
//          Note that after the ambient occlusion we perform a smoothing step using
//              catmull clark subdivision rules to create better looking results. This step
//              basically quadruples the number of polygons

//
//          When loading the polygon meshes we call routines to define the meshes for
//          the subdivision and surface element code, which need to know the surface topology
//

#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <tchar.h>
#include <malloc.h>
#include "resource.h"
#include <GL/gl.h>
#include "glext.h"
#include "wglext.h"
#include <GL/glu.h>
#include <math.h>
#include <time.h>
#include <commdlg.h>
#include "vmath.h"
#include "SurfaceElement.h"

extern "C" {
#include "subd.h"
}

HWND hwnd;

#define WIN_WIDTH  800
#define WIN_HEIGHT 600

#define MAX_INPUT_VERTS     20000
#define VERTEX_BUFFER_SIZE  (65536*4)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PAN_SCALE   0.1f

bool perfTestFlag;
int faces, triangles;

static float vertexBuffer[VERTEX_BUFFER_SIZE];
static float normalBuffer[VERTEX_BUFFER_SIZE];
float *vertexBufferPtr = vertexBuffer;
float *normalBufferPtr = normalBuffer;
int Passes = 2;
int AnimatedVertices;

// support using the mouse to change the viewpoint
typedef enum {
    NONE,
    DOLLY,
    ROTATE,
    PAN,
} MouseMode;

extern int init_shader(char *name, int id, int shaderType);
extern void set_shader(int id, int shaderType);
extern void set_shader_parameter(int id, char *name, float *value);

#define GET_PROC_ADDRESS wglGetProcAddress

#define QUERY_EXTENSION_ENTRY_POINT(name, type)               \
    name = (type)GET_PROC_ADDRESS(#name);
PFNGLACTIVETEXTUREARBPROC glActiveTexture;
PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTextureARB;
static PFNGLSECONDARYCOLORPOINTEREXTPROC glSecondaryColorPointerEXT;
static PFNGLTRACKMATRIXNVPROC glTrackMatrixNV;
PFNGLBINDBUFFERARBPROC glBindBufferARB;
PFNGLBUFFERDATAARBPROC glBufferDataARB;
PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointerARB;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB;
PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArrayARB;
PFNGLMULTITEXCOORD4IARBPROC glMultiTexCoord4i;
PFNGLMULTITEXCOORD4FARBPROC glMultiTexCoord4f;
PFNWGLRELEASETEXIMAGEARBPROC wglReleaseTexImageARB;
PFNWGLBINDTEXIMAGEARBPROC wglBindTexImageARB;

static int vp_width, vp_height;
static float distance;
static float rotx, roty;
static float posx, posy, posz;
static bool first_move;
MouseMode tracking_mouse;

// we store the per vertex results as a 3 float position and
// 3 float ambient occlusion value - We only use one float for ambient occlusion
// but having three makes changing the code for bent normals or indirect lighting easy.
typedef struct float6 {
    float x, y, z, w, a, b;
} float6;

// buffers for storing the input mesh data
static float3 control_mesh_vertex_buffer[MAX_INPUT_VERTS];
static float3 *control_mesh_vertex = control_mesh_vertex_buffer;
static float6 transformed_control_mesh_vertex[MAX_INPUT_VERTS];
static float2 control_mesh_texcoord[MAX_INPUT_VERTS];
static float3 **animation_vertex_buffer;

// these globals are used to load meshes
static int last_vertex_index;
static int last_texcoord_index;
int VIndexOffset;
int VTIndexOffset;
static bool animationFlag;  // true if animating
static int frame_number, frame_offset; // used for animation

static bool vertex_data_only;   // used when loading keyframes
static int keyframes;

// used for panning
static float ModelViewMatrix[16];

void SetDCPixelFormat(HDC hdc)
{
    int nPixelFormat;
    static PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR),
        1,
        PFD_DRAW_TO_WINDOW |
        PFD_SUPPORT_OPENGL |
        PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA,
        32,
        0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        24,
        8,
        0,
        PFD_MAIN_PLANE,
        0,
        0,0,0 };

    nPixelFormat = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, nPixelFormat, &pfd);
}

static void set_perspective(double a, double n, double f)
{
	double l;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	l = tan(a*M_PI/180.0*0.5) * n;

	glFrustum(-l*vp_width/vp_height, l*vp_width/vp_height,
		-l, l, n, f);
	subdFrustum((float) -l*vp_width/vp_height, (float) l*vp_width/vp_height,
		(float) -l, (float) l, (float) -n, (float) -f);
}

static void set_viewport(int width, int height)
{
	vp_width = width;
	vp_height = height;
    glViewport(0, 0, width, height);
    subdViewport(0, 0, width, height);

	set_perspective(45.0f, 0.5, 500.0);
}

void render_surface()
{
	SubdEnum state;
    int verts, indices;
    int stride;
    static int indexBuffer[4*65536];

    glEnable(GL_FRAGMENT_PROGRAM_NV);
    set_shader(SURFACE_SHADER, GL_FRAGMENT_PROGRAM_NV);

    subdOutputVertexBuffer(vertexBufferPtr, sizeof(vertexBuffer));
    subdOutputIndexBuffer(indexBuffer, sizeof(indexBuffer));
    stride = subdGet(SUBD_OUTPUT_VERTEX_SIZE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, stride, (void *) vertexBufferPtr);
    glClientActiveTextureARB(GL_TEXTURE0_ARB);
	glTexCoordPointer(3, GL_FLOAT, stride, (void *) (vertexBufferPtr+6));
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    triangles = 0;
    state = SUBD_START;
    do {
        state = subdTessellate(&verts, &indices, state);
        glDrawElements(GL_QUADS, indices, GL_UNSIGNED_INT,
            (void *) indexBuffer);
        triangles += indices >> 1;
    } while (state == SUBD_CONTINUE);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

static void copyVertices(float *dst, int dst_stride, float *src,
    int src_stride, int count)
{
    int i;
    
    for (i = 0; i < count; i++, src += src_stride, dst += dst_stride) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }
}

static void
render()
{
    float3 *vptr;

	glClearColor(0.4f, 0.5f, 0.6f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -distance);
    glRotatef(rotx, 1.0f, 0.0f, 0.0f);
    glRotatef(roty, 0.0f, 1.0f, 0.0f);
    glTranslatef(-posx, -posy, -posz);

	// save for panning
    glGetFloatv(GL_MODELVIEW_MATRIX, ModelViewMatrix);

    // point to a keyframe if animating
    vptr = control_mesh_vertex;
    if (frame_number != 0)
        vptr = animation_vertex_buffer[frame_number];

    // copy the vertices from the animated mesh
    if (AnimatedVertices > 0)
        copyVertices((float *) transformed_control_mesh_vertex,
            sizeof(transformed_control_mesh_vertex[0])/sizeof(float),
            (float *) vptr, 
            sizeof(*vptr)/sizeof(float),
            AnimatedVertices);
    // copy vertices of the non-animated mesh (the ground in the sample data)
    if (last_vertex_index > AnimatedVertices)
        copyVertices((float *) (transformed_control_mesh_vertex + AnimatedVertices),
            sizeof(transformed_control_mesh_vertex[0])/sizeof(float),
            (float *) (control_mesh_vertex_buffer + AnimatedVertices), 
            sizeof(*vptr)/sizeof(float),
            last_vertex_index - AnimatedVertices);
    calcAmbientOcclusion(Passes);

    render_surface();

	glPopMatrix();
}

static void
perfTest()
{
	int i;
	int t1, t2;
	int tris;

	perfTestFlag = true;
	t1 = clock();
	tris = 0;
	i = 0;
	do {	// test for 2 seconds
		render();
		t2 = clock();
		tris += triangles;
		i++;
	} while ((t2 - t1) < CLOCKS_PER_SEC*2);
	printf("%d tris, %f faces/sec %f tris/second, %g fps\n", triangles,
		(float) faces*i*CLOCKS_PER_SEC/(t2 - t1),
		(float) tris*CLOCKS_PER_SEC/(t2 - t1), (float) i*CLOCKS_PER_SEC/(t2 - t1));
	perfTestFlag = false;
}

//
// Code to read meshes in Wavefront .obj format and "animation" files
//
static void
read_vertex(char *str)
{
	float3 *loc;
    float *p;

	if (str[1] == ' ') {
        loc = &control_mesh_vertex[last_vertex_index++];
		sscanf(str+1, "%f %f %f", &loc->x, &loc->y, &loc->z);
	}
	else if (str[1] == 'n') { // vertex normal
   	}
	else if (str[1] == 't') {
        if (!vertex_data_only) {
            p = (float *) &control_mesh_texcoord[last_texcoord_index++];
		    sscanf(str+2, "%f %f", p, p + 1);
        }
	}
}

int read_vertex_info(char **s, int *v, int *vn, int *vt)
{
	char *q;
    int fields_read;

	q = *s;
	while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')
		q++;
	
	if (!*q)
		return 0;

	fields_read = sscanf(q, "%d/%d/%d", v, vt, vn);

	while (*q && *q != ' ' && *q != '\t')
		q++;

	*s = q;

	return fields_read;
}

static void
read_face(char *s)
{
    int vindex;
	int vnindex, vtindex;
    int firstindex;
	int n;

    faces++;
	subdFace();
    seFace();
	s++;
	vtindex = 0;
    firstindex = -1;
	while (n = read_vertex_info(&s, &vindex, &vnindex, &vtindex)) {
        vindex += VIndexOffset;
        vtindex += VTIndexOffset;
		if (vindex < 0)
			vindex = last_vertex_index + vindex + 1;
        vindex--;   // .obj index starts at 1 not 0
        if (n >= 2)
            subdSet(SUBD_TEXCOORD_INDEX, vtindex-1);
        subdVertex(vindex);
        seVertex(vindex, n >= 2 ? vtindex-1 : 0);
	}
}

static int
read_float(char **s, float *f)
{
	char *q;

	q = *s;
	while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')
		q++;
	
	if (!*q)
		return 0;

	sscanf(q, "%f", f);

	while (*q && *q != ' ' && *q != '\t')
		q++;

	*s = q;

	return 1;
}

static void
decode_string(char *s)
{
    switch (s[0]) {
        case '#':
                break;
        case 'v':
                read_vertex(s);
                break;
        case 'f':
                if (!vertex_data_only)
                    read_face(s);
                break;
        default:
            break;
    }
}

void
OpenMesh()
{
    faces = 0;
    subdBeginMesh();
    seBeginMesh();
    last_texcoord_index = 0;
    last_vertex_index = 0;
    VIndexOffset = 0;
    VTIndexOffset = 0;
}

void
CloseMesh()
{
    subdCtrlMeshVertexPointer(4, 4, (float*)transformed_control_mesh_vertex);
    subdEndMesh();

    // we copy the vertices here so the surface element code can pre-calculate
    // the area for each element.
    // If the area is being recalculated in updateElements() then this copy is
    /// not needed
    copyVertices((float *) transformed_control_mesh_vertex,
            sizeof(transformed_control_mesh_vertex[0])/sizeof(float),
            (float *) control_mesh_vertex_buffer, 
            sizeof(control_mesh_vertex_buffer[0])/sizeof(float),
            last_vertex_index);
    seEndMesh((float3 *) transformed_control_mesh_vertex, control_mesh_texcoord);
}

static bool
read_model(char *filename)
{
    FILE *f;
    char buf[4096];
    
    f = fopen(filename, "r");

    if (!f)
        return false;

    while (fgets(buf, sizeof(buf), f))
        decode_string(buf);
    fclose(f);

    return true;
}

static char *getExtension(char *s)
{
    int n;

    n = (int) strlen(s);
    if (n >= 4 && s[n-4] == '.')
        return s + n - 3;

    return s + n;
}

static void
readAnimationFile(char *filename)
{
    FILE *f;
    char buf[4096];
    static char buf2[4096];
    static char buf3[4096];
    int frames, i;
    int num_vertices;
   
    f = fopen(filename, "r");
    if (!f)
        return;
    
    OpenMesh();
    while (fgets(buf, sizeof(buf), f)) {
        if (buf[0] == '#')
            ;
        else if (!strncmp(buf, "keyframes", sizeof("keyframes")-1)) {
            frames = 1;
            sscanf(buf, "%s %s %d",  buf2, buf3, &frames);

            sprintf(buf2, buf3, 0);
            if (!read_model(buf2)) {
                CloseMesh();
                fclose(f);
                return;
            }

            if (animation_vertex_buffer) {
                for (i = 1; ; i++) {
                    if (animation_vertex_buffer[i])
                        free(animation_vertex_buffer[i]);
                    else
                        break;
                }
                free(animation_vertex_buffer);
            }
            animation_vertex_buffer = (float3**)
                calloc(sizeof(animation_vertex_buffer[0])*(frames+1), 1);
            animation_vertex_buffer[0] = control_mesh_vertex_buffer;

            AnimatedVertices = num_vertices = last_vertex_index;
            vertex_data_only = true;
            for (i = 1; i < frames; i++) {
                control_mesh_vertex = (float3*)
                    malloc(num_vertices*sizeof(control_mesh_vertex[0]));
                animation_vertex_buffer[i] = control_mesh_vertex;
                sprintf(buf2, buf3, i);
                last_vertex_index = 0;
                if (!read_model(buf2)) {
                    CloseMesh();
                    fclose(f);
                    return;
                }
            }
            vertex_data_only = false;

            control_mesh_vertex = control_mesh_vertex_buffer;
            keyframes = frames;
            animationFlag = true;
            frame_offset = 1;
        }
        else if (!strncmp(buf, "add", sizeof("add")-1)) {
            sscanf(buf, "%s %s",  buf2, buf3);
            VIndexOffset = last_vertex_index;
            VTIndexOffset = last_texcoord_index;
            control_mesh_vertex = control_mesh_vertex_buffer;
            read_model(buf3);
        }
    }
    fclose(f);
    CloseMesh();
}

static void loadControlMesh(HWND hwnd)
{
    OPENFILENAME ofn;
    char *str;
    static char result[257];
   
    AnimatedVertices = 0;
    memset((void *) &ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = "Animation Files\0*.ANI\0Object Files\0*.OBJ\0All Files\0*.*\0\0";
    ofn.lpstrFile = result;
    ofn.nMaxFile = sizeof(result)-1;
    ofn.Flags = OFN_FILEMUSTEXIST;

    if (GetOpenFileName(&ofn)) {
		str = getExtension(ofn.lpstrFile);
        if (!strcmp(str, "ANI") || (!strcmp(str, "ani")))
            readAnimationFile(ofn.lpstrFile);
        else {
            OpenMesh();
            read_model(ofn.lpstrFile);
            CloseMesh();
        }
	}
}

static void
mouse_move(WPARAM button, int x, int y)
{
	static int last_x, last_y;

	if (!button)
		return;
    if (!first_move) {
        if (tracking_mouse == DOLLY)
           distance -= (y-last_y)*0.1f;
        else if (tracking_mouse == ROTATE) {
            roty += x - last_x;
            rotx += (y - last_y)*0.5f;
        }
        else {  // PAN
            float xx, yy, zz;
            float fx, fy;

            fx = (x - last_x) * PAN_SCALE;
            fy = (last_y - y) * PAN_SCALE;

            xx = fx*ModelViewMatrix[0] + fy*ModelViewMatrix[1];
            yy = fx*ModelViewMatrix[4] + fy*ModelViewMatrix[5];
            zz = fx*ModelViewMatrix[8] + fy*ModelViewMatrix[9];

            posx -= xx;
            posy -= yy;
            posz -= zz;
        }
    }

	first_move = false;
	last_x = x;
	last_y = y;
}

static void initialize()
{
    static float lightDirection[4] = { -0.5f, 0.707107f, 0.5f };

    set_viewport(WIN_WIDTH, WIN_HEIGHT);
    distance = 50.0f;
    rotx = 0.0f;
    roty = 0.0f;
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);

    QUERY_EXTENSION_ENTRY_POINT(glActiveTexture, PFNGLACTIVETEXTUREARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glClientActiveTextureARB,
        PFNGLCLIENTACTIVETEXTUREARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glSecondaryColorPointerEXT, PFNGLSECONDARYCOLORPOINTEREXTPROC);
    QUERY_EXTENSION_ENTRY_POINT(glTrackMatrixNV, PFNGLTRACKMATRIXNVPROC);
    QUERY_EXTENSION_ENTRY_POINT(glBindBufferARB, PFNGLBINDBUFFERARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glBufferDataARB, PFNGLBUFFERDATAARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glVertexAttribPointerARB, PFNGLVERTEXATTRIBPOINTERARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glEnableVertexAttribArrayARB, PFNGLENABLEVERTEXATTRIBARRAYARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glDisableVertexAttribArrayARB, PFNGLDISABLEVERTEXATTRIBARRAYARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4i, PFNGLMULTITEXCOORD4IARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glMultiTexCoord4f, PFNGLMULTITEXCOORD4FARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(wglReleaseTexImageARB, PFNWGLRELEASETEXIMAGEARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(wglBindTexImageARB, PFNWGLBINDTEXIMAGEARBPROC);

    subdSet(SUBD_OUTPUT_PRIMITIVE, SUBD_QUADS);
    subdSet(SUBD_SUBDIVIDE_DEPTH, 1);
    subdSet(SUBD_INDEX_SIZE, SUBD_INDEX32);
    subdSet(SUBD_OUTPUT_NORMAL, 1);
    printf("enables %d %d %d\n", subdGet(SUBD_OUTPUT_NORMAL), subdGet(SUBD_OUTPUT_TEXCOORD),
        subdGet(SUBD_OUTPUT_TANGENT));

    init_shader("surface.fp", SURFACE_SHADER, GL_FRAGMENT_PROGRAM_NV);
}

static void checkMenuItem(HMENU hmenu, UINT id, BOOL check)
{
    MENUITEMINFO mif;

    mif.cbSize = sizeof(MENUITEMINFO);
    mif.fMask = MIIM_STATE;
    GetMenuItemInfo(hmenu, id, FALSE, &mif);
    if (check)
        mif.fState |= MFS_CHECKED;
    else
        mif.fState &= ~MFS_CHECKED;

    SetMenuItemInfo(hmenu, id, FALSE, &mif);
}

static LRESULT PASCAL
WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    static HGLRC hrc;
    PAINTSTRUCT ps;
	int width, height;
	int command;
    HDC hdc;

    switch (message) {
        case WM_CREATE:
            hdc = GetDC(hwnd);
            SetDCPixelFormat(hdc);
            hrc = wglCreateContext(hdc);
            wglMakeCurrent(hdc, hrc);
            initialize();
            SetTimer(hwnd, 1, 20, 0);  // get WM_TIMER messages up to 50 times/sec
            readAnimationFile("bigguy.ani");

            break;
        case WM_SIZE:
            width = LOWORD(lParam);
            height = HIWORD(lParam);
            set_viewport(width, height);
            break;
            
        case WM_COMMAND:
			command = LOWORD(wParam);
            switch (command) {
            case ID_FILE_EXIT:
                DestroyWindow(hwnd);
                break;
            case ID_FILEOPEN:
                keyframes = 1;
                frame_number = 0;
                loadControlMesh(hwnd);
                CreateElements();
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            }
			break;
		case WM_CHAR:
            switch(wParam) {
                case 'r':
                    animationFlag = !animationFlag;
                    if (!frame_offset)
                        frame_offset = 1;
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
                case 'y':
                    rotx = 90.0f; roty = 0.0f;
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
                case 'R':
                    frame_number++;
                    if (frame_number >= keyframes)
                        frame_number = 0;
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
                case 'p':
                    perfTest();
                    break;
                case 'P':
                    Passes++;
                    if (Passes > 2)
                        Passes = 1;
                    if (!animationFlag)
                        InvalidateRect(hwnd, NULL, FALSE);
                    break;
			}
			break;
        case WM_LBUTTONDOWN:
			first_move = true;
			if (tracking_mouse == DOLLY) // left+right means pan
				tracking_mouse = PAN;
			else
				tracking_mouse = ROTATE;
			break;
        case WM_RBUTTONDOWN:
			first_move = true;
			if (tracking_mouse == ROTATE) // left+right means pan
				tracking_mouse = PAN;
			else
				tracking_mouse = DOLLY;
			break;
        case WM_MBUTTONDOWN:
            first_move = true;
            tracking_mouse = PAN;
            break;
		case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
			tracking_mouse = NONE;
			break;
        case WM_MOUSEMOVE:
			if (tracking_mouse) {
				mouse_move(wParam, LOWORD(lParam), HIWORD(lParam));
                if (!animationFlag)
				    InvalidateRect(hwnd, NULL, FALSE);
			}
			break;
        case WM_PAINT:
            hdc = BeginPaint(hwnd, &ps);
			render();

            SwapBuffers(hdc);

            EndPaint(hwnd, &ps);
            break;
        case WM_TIMER:
            if (animationFlag) {
                frame_number += frame_offset;
                if (frame_number >= keyframes) {
                    frame_offset = -1;
                    frame_number--;
                }
                else if (frame_number < 0) {
                    frame_number = 0;
                    frame_offset = 1;
                }
                InvalidateRect(hwnd, NULL, FALSE);
            }
            break;
        case WM_DESTROY:
            PostQuitMessage ((int) wParam);
            wglMakeCurrent(0, 0);
			wglDeleteContext(hrc);
            break;
    }
    return DefWindowProc(hwnd, message, wParam, lParam);
}

int WINAPI
WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine,
    int nCmdShow)
{
    WNDCLASS    wc;
    MSG         msg;
    static char ClassName[] = "Window";
    static char Title[] = "DynamicAO";

  	AllocConsole();
	freopen("CON", "w", stdout);

    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = 0;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE+1);
    wc.lpszMenuName = MAKEINTRESOURCE(IDR_MENU1);
    wc.lpszClassName = ClassName;
    RegisterClass(&wc);

    hwnd = CreateWindowEx(0, ClassName, Title, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WIN_WIDTH + 8, WIN_HEIGHT + 46, NULL, NULL, hInstance, NULL);
    if (!hwnd)
        return FALSE;

    ShowWindow(hwnd, nCmdShow);

    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int) msg.wParam;
}