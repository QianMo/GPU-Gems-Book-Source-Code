/*
    Subdivision surface viewer program (OpenGL version)

    Supports .obj file format and a "animation" text file format for specifying
    keyframes, normal, and displacement maps

    There is also a utility to dump tessellated output for use with normal/displacement
    map creation utilities such as Melody. The format of the output is .obj with
    tangent and binormals in comments. Mapping utilities that support the additional
    tangent and binormal information will be able to make normal maps that exactly
    match the math used by the tessellator so that uv seams won't be visible.

    Copyright (C) 2005 NVIDIA Corporation

    This file is provided without support, instruction, or implied warranty of any
    kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
    not liable under any circumstances for any damages or loss whatsoever arising
    from the use or inability to use this file or items derived from it.

*/

#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <tchar.h>
#include <malloc.h>
#include "resource.h"
#include <GL/gl.h>
#include "glext.h"
#include <GL/glu.h>
#include <math.h>
#include <time.h>
#include <commdlg.h>

#include "subd/subd.h"

#define MILLISECONDS_PER_FRAME  20
#define RENDER_TO_VERTEX_ARRAY  1

HWND hwnd;

#define WIN_WIDTH  800
#define WIN_HEIGHT 600

#define VERTEX_BUFFER_SIZE  (65536*5)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PAN_SCALE   0.1f
#define SURFACE_SHADER              30
#define NORMALMAP_SURFACE_SHADER    31

#define SURFACEV_SHADER             32

#define COLOR_TEXTURE           1
#define BUMP_TEXTURE            2

#define MAX_SUBDIVISIONS    6

static int maxDepth = 5;
static int adaptiveFlag = 1;
static GLuint vbo1, vbo2;
bool perfTestFlag;
int faces, triangles;
int ClockTicks, TicksPerFrame = (MILLISECONDS_PER_FRAME*CLOCKS_PER_SEC)/1000;

static float vertexBuffer[VERTEX_BUFFER_SIZE];
static float normalBuffer[VERTEX_BUFFER_SIZE];
float *vertexBufferPtr = vertexBuffer;
float *normalBufferPtr = normalBuffer;

#define cross(d, v1, v2) { \
    (d).x = (v1).y*(v2).z - (v1).z*(v2).y; \
    (d).y = (v1).z*(v2).x - (v1).x*(v2).z; \
    (d).z = (v1).x*(v2).y - (v1).y*(v2).x; \
}

typedef struct {
	float x, y, z;
} float3;


typedef enum {
    NONE,
    DOLLY,
    ROTATE,
    PAN,
} MouseMode;

typedef enum {
    COLOR_MAP,
    NORMAL_MAP,
    DISPLACEMENT_MAP,
} LoadTextureType;

extern int init_shader(char *name, int id, int shaderType);
extern void set_shader(int id, int shaderType);
extern void set_shader_parameter(int id, char *name, float *value);

#define GET_PROC_ADDRESS wglGetProcAddress

#define QUERY_EXTENSION_ENTRY_POINT(name, type)               \
    name = (type)GET_PROC_ADDRESS(#name);
static PFNGLACTIVETEXTUREARBPROC glActiveTexture;
PFNGLCLIENTACTIVETEXTUREARBPROC glClientActiveTextureARB;
static PFNGLSECONDARYCOLORPOINTEREXTPROC glSecondaryColorPointerEXT;
static PFNGLTRACKMATRIXNVPROC glTrackMatrixNV;
PFNGLBINDBUFFERARBPROC glBindBufferARB;
PFNGLBUFFERDATAARBPROC glBufferDataARB;
PFNGLVERTEXATTRIBPOINTERARBPROC glVertexAttribPointerARB;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArrayARB;
PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArrayARB;
static PFNGLGENBUFFERSARBPROC glGenBuffersARB;
static PFNGLDELETEBUFFERSARBPROC glDeleteBuffersARB;

static int vp_width, vp_height;
static float distance;
static float rotx, roty;
//static float posx = -3.0, posy, posz = 5.0;
static float posx = 0.0, posy, posz = 0.0;
static bool first_move;
MouseMode tracking_mouse;
static int wireframe_mode;
static int show_control_mesh = FALSE, show_surface = TRUE;

typedef struct ControlMeshVertex {
    float x, y, z, w;
} ControlMeshVertex;

typedef struct float2 {
    float x, y;
} float2;

static ControlMeshVertex control_mesh_vertex_buffer[20000];
static ControlMeshVertex *control_mesh_vertex = control_mesh_vertex_buffer;
static ControlMeshVertex transformed_control_mesh_vertex[20000];
static float2 control_mesh_texcoord[20000];
static ControlMeshVertex **animation_vertex_buffer;
static int last_vertex_index;
static int last_texcoord_index;
static bool normalMapFlag;
static bool displacementFlag;
static bool animationFlag;
static int frame_number, frame_offset;

static bool vertex_data_only;
static int keyframes;
static bool AnimateReverse = false;

// for drawing wireframe control mesh
static unsigned short control_mesh_indices[160000];
static int num_cm_indices;
static float curr_matrix[16];

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

// Call the tessellator and then render the resulting geometry using
// glDrawElements

void render_surface()
{
	SubdEnum state;
    int verts, indices;
    static int indexBuffer[4*65536*10];

    glEnable(GL_FRAGMENT_PROGRAM_NV);
    set_shader(normalMapFlag ? NORMALMAP_SURFACE_SHADER : SURFACE_SHADER, GL_FRAGMENT_PROGRAM_NV);

    subdOutputVertexBuffer(vertexBufferPtr, sizeof(vertexBuffer), 0, vbo1);
    subdOutputVertexBuffer(normalBufferPtr, sizeof(normalBuffer), 1, vbo2);
    subdOutputIndexBuffer(indexBuffer, sizeof(indexBuffer));

  //  printf("Normal Buffer %f %f %f\n", normalBuffer[0], normalBuffer[1], normalBuffer[2]);

    glBindBufferARB(GL_ARRAY_BUFFER, vbo2);
    glEnableVertexAttribArrayARB(8);   // tex0
    glVertexAttribPointerARB(8, 3, GL_HALF_FLOAT_NV, GL_FALSE, sizeof(float)*4, normalBufferPtr);
    glEnableVertexAttribArrayARB(9);   // tex1
    glVertexAttribPointerARB(9, 3, GL_HALF_FLOAT_NV, GL_FALSE, sizeof(float)*4, (short *) normalBufferPtr + 3);
    glEnableVertexAttribArrayARB(10);   // tex2
    glVertexAttribPointerARB(10, 2, GL_SHORT, GL_FALSE, sizeof(float)*4, (short *) normalBufferPtr + 6);

    glBindBufferARB(GL_ARRAY_BUFFER, vbo1);
    glEnableVertexAttribArrayARB(0);
    glVertexAttribPointerARB(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*4, vertexBufferPtr);
    glEnableVertexAttribArrayARB(11);   // tex3
    glVertexAttribPointerARB(11, 2, GL_HALF_FLOAT_NV, GL_FALSE, sizeof(float)*4, (short *) vertexBufferPtr + 6);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    triangles = 0;
    state = SUBD_START;     // start a new tessellation
    do {
        // tessellate until buffers are filled or surface is finished
        state = subdTessellate(&verts, &indices, state);
        if (!perfTestFlag)
        glDrawElements(GL_QUADS, indices, GL_UNSIGNED_INT,
            (void *) indexBuffer);
        triangles += indices >> 1;
    } while (state == SUBD_CONTINUE);   // keep going if tessellator did not finish

    glDisableVertexAttribArrayARB(0);
    glDisableVertexAttribArrayARB(8);
    glDisableVertexAttribArrayARB(9);
    glDisableVertexAttribArrayARB(10);
    glDisableVertexAttribArrayARB(11);

    glDisable(GL_FRAGMENT_PROGRAM_NV);
}

static void transformVertices(float *dst, int dst_stride, float *src,
    int src_stride, int count, float *m)
{
    int i;
	float x, y, z;
    
    for (i = 0; i < count; i++, src += src_stride, dst += dst_stride) {
        x = src[0];
        y = src[1];
        z = src[2];

        dst[0] = x*m[0] + y*m[4] + z*m[8] + m[12];
        dst[1] = x*m[1] + y*m[5] + z*m[9] + m[13];
        dst[2] = x*m[2] + y*m[6] + z*m[10] + m[14];
    }
}

static void
render()
{
	static float eye_light[] = { 0.0f, 0.0f, 1.0f, 0.0f };
    ControlMeshVertex *vptr;

	glClearColor(0.4f, 0.5f, 0.6f, 0.0f);
    if (!perfTestFlag)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -distance);
    glRotatef(rotx, 1.0f, 0.0f, 0.0f);
    glRotatef(roty, 0.0f, 1.0f, 0.0f);
    glTranslatef(-posx, -posy, -posz);

	// transform vertices
    glGetFloatv(GL_MODELVIEW_MATRIX, curr_matrix);
	glLoadIdentity();

    vptr = control_mesh_vertex;
    if (frame_number != 0)
        vptr = animation_vertex_buffer[frame_number];
    transformVertices((float *) transformed_control_mesh_vertex, 4,
        (float *) vptr, 4,
        last_vertex_index, curr_matrix);

    glPolygonMode(GL_FRONT, wireframe_mode ? GL_LINE : GL_FILL);

    if (show_surface) {
        glActiveTexture(GL_TEXTURE1);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, BUMP_TEXTURE);
        render_surface();
    }

    if (show_control_mesh) {
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 4*sizeof(float),
            (void *) transformed_control_mesh_vertex);
        glDrawElements(GL_LINES, num_cm_indices, GL_UNSIGNED_SHORT,
            (void *) control_mesh_indices);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

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

// The following routine read in a mesh defined in .obj format
static void
read_vertex(char *str)
{
	ControlMeshVertex *loc;
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
	s++;
	vtindex = 0;
    firstindex = -1;
	while (n = read_vertex_info(&s, &vindex, &vnindex, &vtindex)) {
		if (vindex < 0)
			vindex = last_vertex_index + vindex + 1;
        vindex--;   // .obj index starts at 1 not 0
        if (n >= 2)
            subdSet(SUBD_TEXCOORD_INDEX, vtindex-1);
        subdVertex(vindex);

        // This code is used for displaying the wireframe control mesh

        if (firstindex == -1)
            firstindex = vindex;
        else
            control_mesh_indices[num_cm_indices++] = vindex;
        control_mesh_indices[num_cm_indices++] = vindex;
	}
    control_mesh_indices[num_cm_indices++] = firstindex;
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

static bool
read_model(char *filename)
{
    FILE *f;
    char buf[4096];
    
    f = fopen(filename, "r");

    if (!f)
        return false;

    if (!vertex_data_only) {
        faces = 0;
        subdBeginMesh();
	    last_texcoord_index = 0;
    }
    last_vertex_index = 0;
    while (fgets(buf, sizeof(buf), f))
        decode_string(buf);
    fclose(f);

    if (!vertex_data_only) {
        subdCtrlMeshTexCoordPointer(2, 4, (float*)control_mesh_texcoord);
        subdCtrlMeshVertexPointer(4, 4, (float*)transformed_control_mesh_vertex);
    }
    subdEndMesh();

    return true;
}

static char *getExtension(char *s)
{
    int n;

    n = strlen(s);
    if (n >= 4 && s[n-4] == '.')
        return s + n - 3;

    return s + n;
}

// change a displacement map from 3 component to single component
// it is not in single component format already

static unsigned char *
convert_displacement_map(unsigned char *image, int w, int h, int bytespp)
{
	int i, j;
	unsigned char *image2;

    
    if (bytespp == 1)
        return image;

    image2 = (unsigned char *) malloc(w*h);
    for (j = 0; j < h; j++) {
        for (i = 0; i < w; i++) {
            image2[i + j*w] = image[(i + j*w)*bytespp];
        }
    }
    free(image);

    return image2;
}

// round up to the next power of 2

static int
nextpower2(int x)
{
	int y;

	for (y = 1; y < x; y <<= 1)
		;
	return y;
}

// load a .bmp format image file and store it in the appropriate texture

static void
load_image(int command, char *filename)
{
	BITMAPFILEHEADER fileheader;
	BITMAPINFOHEADER infoheader;
    FILE *f;
	int size, size2, i;
	unsigned char *image, *p, c;
	int w, h;
	int extra;
	unsigned char *q;
	int len;
	int tnum;
    int bytesPerPixel;
	unsigned char *disp_map;
    char colorTable[4*256];

    f = fopen(filename, "rb");

    if (!f)
        return;
	
	fread(&fileheader, sizeof(fileheader), 1, f);
	fread(&infoheader, sizeof(infoheader), 1, f);

    bytesPerPixel = infoheader.biBitCount >> 3;
    if (bytesPerPixel == 1) {
        if (infoheader.biClrUsed == 0)
            infoheader.biClrUsed = 256;
        fread(colorTable, 4*infoheader.biClrUsed, 1, f);
    }

	size = infoheader.biWidth * infoheader.biHeight * bytesPerPixel;

	extra = 0;
	if (size != (int) infoheader.biSizeImage) {
		extra = infoheader.biSizeImage/infoheader.biHeight;
		extra -= bytesPerPixel*infoheader.biWidth;
	}
	if (extra < 0)
		extra = 0;

	h = nextpower2(infoheader.biHeight);
	w = nextpower2(infoheader.biWidth);

	size2 = h * w * bytesPerPixel;

	image = (unsigned char *) malloc(size2 + extra);

	p = image;
	len = infoheader.biWidth*bytesPerPixel + extra;
	for (i = 0; i < infoheader.biHeight; i++, p += w*bytesPerPixel) {
		fread(p, len, 1, f);

		// swap red and blue
        if (bytesPerPixel == 3)
            for (q = p; q < p + len; q += 3) {
                c = q[0];
                q[0] = q[2];
                q[2] = c;
            }
	}

	switch (command) {
		case COLOR_MAP:
				tnum = COLOR_TEXTURE; break;
		case NORMAL_MAP:
				tnum = BUMP_TEXTURE;
				break;
        case DISPLACEMENT_MAP:
                disp_map =
                     convert_displacement_map(image, w, h, bytesPerPixel);
                subdDisplacementMap(disp_map, w, h,&control_mesh_texcoord[0].x);
                free(disp_map);
                fclose(f);
                return;
	}

	if (command == NORMAL_MAP)
		glActiveTexture(GL_TEXTURE1);

	glBindTexture(GL_TEXTURE_2D, tnum);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
        w, h, 0, bytesPerPixel == 3 ? GL_RGB :
            GL_LUMINANCE, GL_UNSIGNED_BYTE, (GLvoid *) image);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glActiveTexture(GL_TEXTURE0);

	fclose(f);

	free(image);

}

// read an animation file (extension .ani)

static void
readAnimationFile(char *filename)
{
    FILE *f;
    char buf[4096];
    static char buf2[4096];
    static char buf3[4096];
    int frames, i;
    int num_vertices;
    float displacementScale;
   
    f = fopen(filename, "r");
    if (!f)
        return;
    displacementFlag = false;
    normalMapFlag = false;
    while (fgets(buf, sizeof(buf), f)) {
        if (buf[0] == '#')
            ;
        else if (!strncmp(buf, "keyframes", sizeof("keyframes")-1)) {
            frames = 1;
            sscanf(buf, "%s %s %d",  buf2, buf3, &frames);

            sprintf(buf2, buf3, 0);
            printf("Loading keyframe 0: %s\n", buf2);
            if (!read_model(buf2)) {
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
            animation_vertex_buffer = (ControlMeshVertex**)
                calloc(sizeof(animation_vertex_buffer[0])*(frames+1), 1);
            animation_vertex_buffer[0] = control_mesh_vertex_buffer;

            num_vertices = last_vertex_index;
            vertex_data_only = true;
            for (i = 1; i < frames; i++) {
                control_mesh_vertex = (ControlMeshVertex*)
                    malloc(num_vertices*sizeof(control_mesh_vertex[0]));
                animation_vertex_buffer[i] = control_mesh_vertex;
                sprintf(buf2, buf3, i);
                printf("Loading keyframe %d: %s\n", i, buf2);
                if (!read_model(buf2)) {
                    fclose(f);
                    return;
                }
            }
            vertex_data_only = false;

            control_mesh_vertex = control_mesh_vertex_buffer;
            keyframes = frames;
        }
        else if (!strncmp(buf, "dmapscale", sizeof("dmapscale")-1)) {
            sscanf(buf, "%s %f", buf2, &displacementScale);
            subdDisplacementScale(displacementScale);
        }
        else if (!strncmp(buf, "dmap", sizeof("dmap")-1)) {
            sscanf(buf, "%s %s", buf2, buf3);
            load_image(DISPLACEMENT_MAP, buf3);
            displacementFlag = true;
        }
        else if (!strncmp(buf, "normalmap", sizeof("normalmap")-1)) {
            sscanf(buf, "%s %s", buf2, buf3);
            load_image(NORMAL_MAP, buf3);
            normalMapFlag = true;
        }
        else if (!strncmp(buf, "pingpong", sizeof("pingpong")-1)) {
            AnimateReverse = true;
        }
    }
    fclose(f);

    subdSet(SUBD_OUTPUT_TEXCOORD, normalMapFlag);
    subdSet(SUBD_OUTPUT_TANGENT, normalMapFlag);
    if (!displacementFlag)
        subdDisplacementScale(0.0f);
}

// load a subdivision surface control mesh

static void loadControlMesh(HWND hwnd)
{
    OPENFILENAME ofn;
    char *str;
    static char result[257];
   
    memset((void *) &ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = "Object Files\0*.OBJ\0Animation Files\0*.ANI\0All Files\0*.*\0\0";
    ofn.lpstrFile = result;
    ofn.nMaxFile = sizeof(result)-1;
    ofn.Flags = OFN_FILEMUSTEXIST;

    if (GetOpenFileName(&ofn)) {
		str = getExtension(ofn.lpstrFile);
        if (!strcmp(str, "ANI") || (!strcmp(str, "ani")))
            readAnimationFile(ofn.lpstrFile);
        else {
            read_model(ofn.lpstrFile);
            normalMapFlag = false;
            subdDisplacementScale(0.0f);
        }
	}
}

static void
mouse_move(int button, int x, int y)
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

            xx = fx*curr_matrix[0] + fy*curr_matrix[1];
            yy = fx*curr_matrix[4] + fy*curr_matrix[5];
            zz = fx*curr_matrix[8] + fy*curr_matrix[9];

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
    distance = 75.0f;
    rotx = 0.0f;
    roty = -60.0f;
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
    QUERY_EXTENSION_ENTRY_POINT(glGenBuffersARB, PFNGLGENBUFFERSARBPROC);
    QUERY_EXTENSION_ENTRY_POINT(glDeleteBuffersARB, PFNGLDELETEBUFFERSARBPROC);

    subdSet(SUBD_OUTPUT_PRIMITIVE, SUBD_QUADS);
    subdSet(SUBD_OUTPUT_NORMAL, 1);
    subdSet(SUBD_ADAPTIVE, adaptiveFlag);
    subdSetf(SUBD_FLAT_DISTANCE, 0.5f);
    subdSet(SUBD_SUBDIVIDE_DEPTH, maxDepth);
    subdSet(SUBD_INDEX_SIZE, SUBD_INDEX32);

    init_shader("ShadeSurface", SURFACE_SHADER, GL_FRAGMENT_PROGRAM_NV);
    set_shader_parameter(SURFACE_SHADER, "lightDirection", lightDirection);
    init_shader("ShadeSurfaceN", NORMALMAP_SURFACE_SHADER, GL_FRAGMENT_PROGRAM_NV);
    set_shader_parameter(NORMALMAP_SURFACE_SHADER, "lightDirection", lightDirection);

    init_shader("ShaderSurfaceV", SURFACEV_SHADER, GL_VERTEX_PROGRAM_NV);
    glEnable(GL_VERTEX_PROGRAM_NV);
    glTrackMatrixNV(GL_VERTEX_PROGRAM_NV, 0, GL_MODELVIEW_PROJECTION_NV, GL_IDENTITY_NV);

#if RENDER_TO_VERTEX_ARRAY
    glGenBuffersARB(1, &vbo1);
    printf("vbo1 %d\n", vbo1);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, vbo1);
    glBufferDataARB(GL_PIXEL_PACK_BUFFER_EXT, VERTEX_BUFFER_SIZE*sizeof(float),
        vertexBuffer, GL_STREAM_COPY);
    glGenBuffersARB(1, &vbo2);
    printf("vbo2 %d\n", vbo2);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, vbo2);
    glBufferDataARB(GL_PIXEL_PACK_BUFFER_EXT, VERTEX_BUFFER_SIZE*sizeof(float),
        normalBuffer, GL_STREAM_COPY);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_EXT, 0);
    vertexBufferPtr = NULL;
    normalBufferPtr = NULL;
#endif
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

static void updateSurfaceQuality(HWND hwnd, int command, float value)
{
	static int last_view_quality = ID_SURFACE_QUALITY_05;

	checkMenuItem(GetMenu(hwnd), last_view_quality, FALSE);
	checkMenuItem(GetMenu(hwnd), command, TRUE);
	last_view_quality = command;
    subdSetf(SUBD_FLAT_DISTANCE, value);
}

// convert from 16 bit floating point format to 32 bit

static float
halfToFloat(short h)
{
    int result;

    int sign = (h>>15) & 1;
    int exponent = (h >> 10) & 0x1f;
    int mantissa = h & 0x3ff;

    if (exponent == 0) {
        if (mantissa == 0)
            return 0.0f;

        // denormalized

        while (!(mantissa & 0x400)) {
            exponent--;
            mantissa <<= 1;
        }
        mantissa &= 0x3ff;
    }

    result = sign << 31 | mantissa << 13 | (exponent - 15 + 127)<<23;

    return *(float*) &result;

}

// write out the results of the tessellator to be used with
// normal/displacement mapping programs

static void
DumpMesh(char *fileName)
{
    typedef struct float4 {
        float x, y, z, w;
    } float4;
    int state;
    float4 *TessVerts, *TessNorms;
    int *TessPrims;
    int tvSize, tpSize;
    int numVerts, numIndices;
    FILE *f;
    int i;
    int depth;

    f = fopen(fileName, "w");
    tvSize = faces * 16 * 4 * sizeof(float) * 4 + 8192*64;
    TessVerts = (float4 *) calloc(tvSize, 1);
    TessNorms = (float4 *) calloc(tvSize, 1);
    tpSize = faces * 16 * 4 * sizeof(int) * 10;
    TessPrims = (int *) malloc(tpSize);
    subdSet(SUBD_ADAPTIVE, 0);
    depth = subdGet(SUBD_SUBDIVIDE_DEPTH);
    if (depth > 3)
        subdSet(SUBD_SUBDIVIDE_DEPTH, 2);
    subdOutputVertexBuffer(&TessVerts[0].x, tvSize, 0, 0);
    subdOutputVertexBuffer(&TessNorms[0].x, tvSize, 1, 0);
    subdOutputIndexBuffer((int *)TessPrims, tpSize);
    subdCtrlMeshVertexPointer(4, 4, (float*)control_mesh_vertex);

    state =  subdTessellate(&numVerts, &numIndices, SUBD_START);
    if (state != SUBD_DONE)
        printf("ERROR tessellating\n");
    printf("numVerts %d numIndices %d\n", numVerts, numIndices);
    subdCtrlMeshVertexPointer(4, 4, (float*)transformed_control_mesh_vertex);
    subdSet(SUBD_ADAPTIVE, adaptiveFlag);
    subdSet(SUBD_SUBDIVIDE_DEPTH, depth);

    for (i = 0; i < numVerts; i++)
        fprintf(f, "v %f %f %f\n", TessVerts[i].x, TessVerts[i].y, TessVerts[i].z);

    for (i = 0; i < numVerts; i++) {
        short *p = (short *) &TessNorms[i].w;
        fprintf(f, "vt %f %f\n", p[0] * 2.0f/65535.0f, p[1] * 2.0f/65535.0f);
    }

    for (i = 0; i < numVerts; i++) {
        short *p = (short *) &TessNorms[i].x;
        short *p2 = (short *) &TessVerts[i].w;
        float3 n, t, b;

        n.x = halfToFloat(p[0]);
        n.y = halfToFloat(p[1]);
        n.z = halfToFloat(p[2]);
        t.x = halfToFloat(p[3]);
        t.y = halfToFloat(p[4]);
        t.z = halfToFloat(p[5]);
        fprintf(f, "vn %f %f %f\n", n.x, n.y, n.z);
        fprintf(f, "#_#tangent %f %f %f\n", t.x, t.y, t.z);
        cross(b, n, t);
        if (p2[0] < 10) {
            b.x = -b.x;
            b.y = -b.y;
            b.z = -b.z;
        }
        fprintf(f, "#_#binormal %f %f %f\n", b.x, b.y, b.z);
    }

    for (i = 0; i < numIndices; i += 4) { 
        fprintf(f, "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n",
            TessPrims[i]+1, TessPrims[i]+1, TessPrims[i]+1,
            TessPrims[i+1]+1, TessPrims[i+1]+1, TessPrims[i+1]+1,
            TessPrims[i+2]+1, TessPrims[i+2]+1, TessPrims[i+2]+1,
            TessPrims[i+3]+1, TessPrims[i+3]+1, TessPrims[i+3]+1);

    }

    fclose(f);
    
    free(TessVerts);
    free(TessNorms);
    free(TessPrims);
}

static void
saveMapMesh()
{
	int n;
    OPENFILENAME ofn;
    static char result[257];

    memset((void *) &ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = "OBJ Files\0*.OBJ\0\0";
    ofn.lpstrFile = result;
    ofn.nMaxFile = sizeof(result)-1;
    ofn.Flags = OFN_HIDEREADONLY;

    if (!GetSaveFileName(&ofn))
		return;

		// append .obj to file name if needed
	n = strlen(result);
	if (n < 4 || result[n - 4] != '.')
		strcpy(result + n, ".obj");
    DumpMesh(result);
}

static long PASCAL
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
            SetTimer(hwnd, 1, 1, 0);
            readAnimationFile("MonsterFrog.ani");
            if (keyframes > 1) {
                animationFlag = true;
                frame_offset = 1;
            }
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
                animationFlag = false;
                loadControlMesh(hwnd);
                if (keyframes > 1) {
                    animationFlag = true;
                    frame_offset = 1;
                }
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SAVE_MAPMESH:
                saveMapMesh();
                break;
            case ID_SURFACE_QUALITY_2:
                updateSurfaceQuality(hwnd, command, 2.0f);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SURFACE_QUALITY_1:
                updateSurfaceQuality(hwnd, command, 1.0f);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SURFACE_QUALITY_075:
                updateSurfaceQuality(hwnd, command, 0.75f);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SURFACE_QUALITY_05:
                updateSurfaceQuality(hwnd, command, 0.5f);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SURFACE_QUALITY_025:
                updateSurfaceQuality(hwnd, command, 0.25f);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SURFACE_ADAPTIVETESSELLATION:
                 adaptiveFlag = !adaptiveFlag;
                 subdSet(SUBD_ADAPTIVE, adaptiveFlag);
                 checkMenuItem(GetMenu(hwnd), command, adaptiveFlag);
                 InvalidateRect(hwnd, NULL, FALSE);
                 break;
            case ID_MAXIMUMSUBDIVISION_0:
            case ID_MAXIMUMSUBDIVISION_1:
            case ID_MAXIMUMSUBDIVISION_2:
            case ID_MAXIMUMSUBDIVISION_3:
            case ID_MAXIMUMSUBDIVISION_4:
            case ID_MAXIMUMSUBDIVISION_5:
            case ID_MAXIMUMSUBDIVISION_6:
                checkMenuItem(GetMenu(hwnd), maxDepth + ID_MAXIMUMSUBDIVISION_0, FALSE);
                maxDepth = command - ID_MAXIMUMSUBDIVISION_0;
                checkMenuItem(GetMenu(hwnd), command, TRUE);
                subdSet(SUBD_SUBDIVIDE_DEPTH, maxDepth);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SHOW_SURFACE:
                checkMenuItem(GetMenu(hwnd), command, TRUE);
                checkMenuItem(GetMenu(hwnd), ID_SHOW_CONTROL, FALSE);
                checkMenuItem(GetMenu(hwnd), ID_SHOW_SURFACE_AND_CONTROL,FALSE);
                show_surface = TRUE;
                show_control_mesh = FALSE;
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SHOW_CONTROL:
                checkMenuItem(GetMenu(hwnd), command, TRUE);
                checkMenuItem(GetMenu(hwnd), ID_SHOW_SURFACE_AND_CONTROL,FALSE);
                checkMenuItem(GetMenu(hwnd), ID_SHOW_SURFACE, FALSE);
                show_surface = FALSE;
                show_control_mesh = TRUE;
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_SHOW_SURFACE_AND_CONTROL:
                checkMenuItem(GetMenu(hwnd), command, TRUE);
                checkMenuItem(GetMenu(hwnd), ID_SHOW_SURFACE, FALSE);
                checkMenuItem(GetMenu(hwnd), ID_SHOW_CONTROL, FALSE);
                show_control_mesh = TRUE;
                show_surface = TRUE;
                InvalidateRect(hwnd, NULL, FALSE);
                break;
            case ID_WIREFRAME:
                wireframe_mode = !wireframe_mode;
                checkMenuItem(GetMenu(hwnd), ID_WIREFRAME, wireframe_mode);
                InvalidateRect(hwnd, NULL, FALSE);
                break;
			}
			break;
		case WM_CHAR:
            switch(wParam) {
				case 'w':
                    wireframe_mode = !wireframe_mode;
                    checkMenuItem(GetMenu(hwnd), ID_WIREFRAME, wireframe_mode);
                    InvalidateRect(hwnd, NULL, FALSE);
					break;
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
                case 'q':
                    adaptiveFlag = !adaptiveFlag;
                    printf("Adaptive %d\n", adaptiveFlag);
                    subdSet(SUBD_ADAPTIVE, adaptiveFlag);
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
                case 'a':
                    if (maxDepth < MAX_SUBDIVISIONS)
                        maxDepth++;
                    subdSet(SUBD_SUBDIVIDE_DEPTH, maxDepth);
                    printf("max depth %d\n", maxDepth);
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
                case 'A':
                    if (maxDepth > 0)
                        maxDepth--;
                    subdSet(SUBD_SUBDIVIDE_DEPTH, maxDepth);
                    printf("max depth %d\n", maxDepth);
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
                case 'p':
                    perfTest();
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
                int t = clock();
                if (t - ClockTicks < TicksPerFrame)
                    break;
                ClockTicks = t;
                frame_number += frame_offset;
                if (frame_number >= keyframes) {
                    if (AnimateReverse) {
                        frame_offset = -1;
                        frame_number--;
                    }
                    else
                        frame_number = 0;
                }
                else if (frame_number < 0) {
                    frame_number = 0;
                    frame_offset = 1;
                }
                InvalidateRect(hwnd, NULL, FALSE);
            }
            break;
        case WM_DESTROY:
            PostQuitMessage (wParam);
            glDeleteBuffersARB(1, &vbo1);
            glDeleteBuffersARB(1, &vbo2);
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
    static char Title[] = "SubDViewer";

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

    return msg.wParam;
}