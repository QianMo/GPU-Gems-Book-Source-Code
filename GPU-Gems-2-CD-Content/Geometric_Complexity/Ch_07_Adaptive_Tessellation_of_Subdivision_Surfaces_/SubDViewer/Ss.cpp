
#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <tchar.h>
#include <malloc.h>
#include "resource.h"
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <math.h>
#include <time.h>
#include <commdlg.h>

#include "ss.h"
//#include "convert.h"

#include "subd.h"

#define ADAPTIVE 1

extern void special();
extern int init_shader();
typedef unsigned int uint;
int normalize_flag;

HWND hwnd;
static HDC hdc; // needed for SwapBuffers()

#define WIN_WIDTH  640
#define WIN_HEIGHT 480

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXVERTS 5000

#define INFO_SHIFT		2
#define INFO_MASK		3

#define AQ	1

float flat_distance = 0.5f;
//float flat_scale, flat_scale2;
float view_matrix[16];
float proj_matrix[16];
float curr_matrix[16];

int vp_width, vp_height;
int subdiv_level;
static char model_filename[257];
int triangles;
int first_move, tracking_mouse;

int vindex;
int tindex;
point vertexarray[MAXVERTS];
point2 texcoord_array[MAXVERTS];
point transformed_vertexarray[MAXVERTS];
int faces;

float distance;
float rotx, roty;
int test_flag;
int flag1;
int adaptive_flag = 1;

int vertex_size = 12;

static int cubic_flag = 1;
int bf_cull_flag = 1;
static int cc_flag;

static int nuindex;

void *mesh;

#define NUVAL_0	1.10f
#define NUVAL_1	0.90f
#define NUVAL_2	0.95f
#define NUVAL_3	0.975f

/*
static float nuval[10] = {
	1.1f,
	0.9f,
	0.95f,
	0.975f,
	1.0f,
	1.0f,
	1.0f,
	1.0f,
	1.0f,
	1.0f
};
*/
static float nuval[10] = {
	NUVAL_0,
	NUVAL_1,
	NUVAL_2,
	NUVAL_3,
	1.0f,
	1.0f,
	1.0f,
	1.0f,
	1.0f,
	1.0f
};
point finalpos;

/*
void special()
{
#if 1
	static int adjust;

	adjust++;
	if (adjust == 2)
		adjust = 0;

	if (adjust == 0) {
		nuval[0] = 1.15f;
		nuval[1] = 0.90f;
		nuval[2] = 0.925f;
		nuval[3] = 0.990f;
	}
	else if (adjust == 1) {
		nuval[0] = 1.10f;
		nuval[1] = 0.90f;
		nuval[2] = 0.950f;
		nuval[3] = 0.975f;
	}
	else {
		nuval[0] = 1.0f;
		nuval[1] = 1.0f;
		nuval[2] = 1.0f;
		nuval[3] = 1.0f;
	}
#else
	FILE *f;
	int i;

	f = fopen("C:\\nudata", "r");
	if (!f)
		return;
	for (i = 0; i < 10; i++)
		fscanf(f, "%f", &nuval[i]);
	fclose(f);
#endif
}
*/

void
emit_render()
{
	int n;

}

void
SetDCPixelFormat(HDC hdc)
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
        0,0,
        32,
        0,
        0,
        PFD_MAIN_PLANE,
        0,
        0,0,0 };

    nPixelFormat = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, nPixelFormat, &pfd);
}

static void
set_perspective(double a, double n, double f)
{
	double l;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	l = tan(a*M_PI/180.0*0.5) * n;

	glFrustum(-l*vp_width/vp_height, l*vp_width/vp_height,
		-l, l, n, f);
}

static void
set_viewport(int width, int height)
{
	vp_width = width;
	vp_height = height;
    glViewport(0, 0, width, height);

	set_perspective(45.0f, 0.5, 500.0);
}

int
normalize(point *p)
{
    float len;

	len = dot(*p, *p);
	/*
	if (len < 0.000000000001f)
		return 1;
	*/

    len = 1.0f / ((float) sqrt(len));

    p->x *= len;
    p->y *= len;
    p->z *= len;

	return 0;
}

void
calc_normal(point *n, point *p1, point *p2, point *p3)
{
	point v1, v2;
    float len;

	vsub(v1, *p1, *p2);
	vsub(v2, *p2, *p3);
	cross(*n, v1, v2);
	len = dot(*n, *n);
    len = 1.0f / ((float) sqrt(len));

    n->x *= len;
    n->y *= len;
    n->z *= len;
	//normalize(n);
}

static void
init_lighting()
{
    static float front_mat_diffuse[] = {0.9f, 0.9f, 0.9f, 1.0F};
    static float front_mat_ambient[] = {0.1F, 0.1F, 0.1F, 1.0F};
	static float front_mat_specular[] = {0.2F, 0.2F, 0.2F, 1.0F};

	static float lts_spc[] = {0.9f, 0.9f, 0.9f, 1.0f};
	static float lts_amb[] = {0.8f, 0.8f, 0.8f, 1.0f};
	static float lts_dif[] = {0.5f, 0.5f, 0.5f, 1.0f};
    static float lpos1[] = {1.0, -1.0, 1.0, 0.0};

    glMaterialfv(GL_FRONT, GL_SPECULAR, front_mat_specular);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, front_mat_diffuse);
    glMaterialfv(GL_FRONT, GL_AMBIENT, front_mat_ambient);

	glMaterialf(GL_FRONT, GL_SHININESS, 25.0);

	glLightfv(GL_LIGHT0, GL_AMBIENT, lts_amb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lts_dif);
	glLightfv(GL_LIGHT0, GL_SPECULAR, lts_spc);

	glLightfv(GL_LIGHT1, GL_POSITION, lpos1);
    
    glEnable(GL_LIGHT0);

    glEnable(GL_LIGHTING);
}

#define GET_PROC_ADDRESS wglGetProcAddress

#define QUERY_EXTENSION_ENTRY_POINT(name, type)               \
    name = (type)GET_PROC_ADDRESS(#name);

void
render_model()
{
	int *ip;
	int i;
	float *vertex_list;
	int num_verts;
	int *prim_list;
	int n;
    static PFNGLACTIVETEXTUREARBPROC glClientActiveTexture;

	if (!mesh)
		return;

	subdTessellate(mesh, (point *) transformed_vertexarray,
        texcoord_array,
		&vertex_list, &num_verts, &prim_list, &n, 0, 0);

	if (!test_flag) {
        if (!glClientActiveTexture)
            QUERY_EXTENSION_ENTRY_POINT(glClientActiveTexture,
                PFNGLACTIVETEXTUREARBPROC);
            
		//glNormalPointer(GL_FLOAT, vertex_size*4, (void *) vertex_list);

        glClientActiveTexture(GL_TEXTURE0);
		glTexCoordPointer(3, GL_FLOAT, vertex_size*4, (void *) (vertex_list+6));
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glClientActiveTexture(GL_TEXTURE1);
		glTexCoordPointer(3, GL_FLOAT, vertex_size*4, (void *) (vertex_list+9));
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

		//glColorPointer(3, GL_FLOAT, vertex_size*4, (void *) (vertex_list+3));
		glVertexPointer(3, GL_FLOAT, vertex_size*4, (void *) vertex_list);

		//glEnableClientState(GL_NORMAL_ARRAY);
		//glEnableClientState(GL_COLOR_ARRAY);
		glEnableClientState(GL_VERTEX_ARRAY);

		glDrawElements(GL_QUADS, n, GL_UNSIGNED_INT, (void *) prim_list);
	}

	triangles += (n)>>1;
}

void
psTransform(float *vin, float *vout, float *t)
{
//    float *t = curr_matrix;
	float x, y, z;

	x = vin[0];
	y = vin[1];
	z = vin[2];

	vout[0] = x*t[0] + y*t[4] + z*t[8] + t[12];
	vout[1] = x*t[1] + y*t[5] + z*t[9] + t[13];
	vout[2] = x*t[2] + y*t[6] + z*t[10] + t[14];
}

static void
render()
{
	int i;
	float eye_light[] = { 0.0f, 0.0f, 1.0f, 0.0f };

	glGetFloatv(GL_VIEWPORT, view_matrix);
	glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix);

	// MTB kluge -- get correct ratio
	flat_scale = flat_distance*480/640
		/(view_matrix[3] - view_matrix[1]);
	if (cubic_flag) {
		flat_scale *= cubic_flag ? -8.0f/3.0f : -4.0f;
		flat_scale *= 0.5f;
		flat_scale2 = flat_scale*0.5f;
	}
	else
		flat_scale *= -4.0f;
		
	/*
	if (ortho_flag)
		flat_scale /= proj_matrix[0]*0.5f;
	*/

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef(-45.0f, 0.0f, 1.0f, 0.0f);
	glRotatef(-45.0f, 1.0f, 0.0f, 0.0f);
	glLightfv(GL_LIGHT0, GL_POSITION, eye_light);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -distance);
    glRotatef(rotx, 1.0f, 0.0f, 0.0f);
    glRotatef(roty, 0.0f, 1.0f, 0.0f);

	// transform vertices
    glGetFloatv(GL_MODELVIEW_MATRIX, curr_matrix);
	glLoadIdentity();

	for (i = 0; i < vindex; i++)
		psTransform(&vertexarray[i].x,
			&transformed_vertexarray[i].x, curr_matrix);

	glColor3f(1.0f, 1.0f, 1.0f);

#if 0
{
	face *f;
	edge *e, *elast;
	point n;

	for (f = first_face; f; f = f->next) {
		elast = f->e;
		calc_normal(&n, f->e->v, f->e->next->v,
			f->e->next->next->v);
		glNormal3f(n.x, n.y, n.z);
		glBegin(GL_POLYGON);
		e = f->e;
		do {
			e = elast->next;
			glVertex3f(e->v->x, e->v->y, e->v->z);
		} while ((elast = elast->next) != f->e);

		glEnd();
	}
}
#else
	render_model();
#endif

	glPopMatrix();
}

int
read_vertex_info(char **s, int *v, int *vt, int *vn)
{
	char *q;

	q = *s;
	while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')
		q++;
	
	if (!*q)
		return 0;

	sscanf(q, "%d/%d/%d", v, vn);

	while (*q && *q != ' ' && *q != '\t')
		q++;

	*s = q;

	return 1;
}

static void
read_face(char *s)
{
	int vindex, vtindex, vnindex;

	faces++;
	subdFace(mesh, 0);

	s++;
	vindex = 0;
	while (read_vertex_info(&s, &vindex, &vtindex, &vnindex))
		subdVertex(mesh, vindex, 0, vtindex);
}

static void
read_vertex(char *str)
{
	point *v;

	if (str[1] == ' ') {
		v = (point *) &vertexarray[vindex++];
		sscanf(str+1, "%f %f %f", &v->x, &v->y, &v->z);
	}
	else if (str[1] == 'n') {
		// vertex normal
	}
    else if (str[1] == 't') {   // texture coordinates
		v = (point *) &texcoord_array[tindex++];
		sscanf(str+2, "%f %f", &v->x, &v->y);
    }
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
                read_face(s);
                break;
        default:
            break;
    }
}

static void
read_model(char *filename)
{
    FILE *f;
    char buf[4096];
    
    f = fopen(filename, "r");

    if (!f)
        return;

	vindex = 0;
    tindex = 0;
	mesh = subdBeginMesh(mesh, SUBD_CATMULL_CLARK);
//mesh = subdBeginMesh(mesh, SUBD_DOO_SABIN);
	faces = 0;
    while (fgets(buf, sizeof(buf), f)) {
        decode_string(buf);
    }

    fclose(f);
	subdEndMesh(mesh);
printf("verts %d\n", vindex);
}

static void
perf_test()
{
	int i;
	int t1, t2;

	test_flag = 1;
	t1 = clock();
	i = 0;
	triangles = 0;
	do {	// test for 2 seconds
		render_model();
		t2 = clock();
		i++;
	} while ((t2 - t1) < CLOCKS_PER_SEC*2);
	printf("%f faces/sec %f tris/sec, %d fps triangles %d\n",
		(float) (i*faces)*CLOCKS_PER_SEC/(t2 - t1),
		(float) (triangles)*CLOCKS_PER_SEC/(t2 - t1),
		i/2,
		triangles/i);
	test_flag = 0;
}

static void
open_model()
{
    OPENFILENAME ofn;
    static char result[257];
   
    memset((void *) &ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = "Wavefront OBJ Files\0*.OBJ\0All Files\0*.*\0\0";
    ofn.lpstrFile = model_filename;
    ofn.nMaxFile = sizeof(model_filename)-1;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;

    if (GetOpenFileName(&ofn)) {
		vindex = 0;
        tindex = 0;
		read_model(ofn.lpstrFile);
	}
}

static void
mouse_move(int button, int x, int y)
{
	static int last_x, last_y;

	if (!button)
		return;
	if (!first_move) {
		roty += x - last_x;
		rotx += (y - last_y)*0.5f;
	}

	first_move = 0;
	last_x = x;
	last_y = y;
}

static long PASCAL
WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    static HGLRC hrc;
    PAINTSTRUCT ps;
	int width, height;
	int command;

    switch (message) {
        case WM_CREATE:
            hdc = GetDC(hwnd);
            SetDCPixelFormat(hdc);
            hrc = wglCreateContext(hdc);
            wglMakeCurrent(hdc, hrc);
            set_viewport(WIN_WIDTH, WIN_HEIGHT);
			distance = 10.0f;
			rotx = 0.0f;
			roty = 0.0f;
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_CULL_FACE);
			//init_lighting();
            (void) init_shader();

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
                    open_model();
                    InvalidateRect(hwnd, NULL, FALSE);
                    break;
			}
			break;
		case WM_CHAR:
            switch(wParam) {
				case 'a':
					adaptive_flag = !adaptive_flag;
					InvalidateRect(hwnd, NULL, FALSE);
					break;
				case 'n':
					normalize_flag = !normalize_flag;
					printf("Normalize tangents %d\n", normalize_flag);
					InvalidateRect(hwnd, NULL, FALSE);
					break;
			}
			break;
        case WM_KEYDOWN:
            switch(wParam) {
                case VK_LEFT:
						roty -= 10.0f;
						InvalidateRect(hwnd, NULL, FALSE);
                        break;
                case VK_RIGHT:
						roty += 10.0f;
						InvalidateRect(hwnd, NULL, FALSE);
                        break;
				case VK_UP:
						distance -= 1.0f;
						InvalidateRect(hwnd, NULL, FALSE);
						break;
				case VK_DOWN:
						distance += 1.0f;
						InvalidateRect(hwnd, NULL, FALSE);
						break;
				case VK_SPACE:
						perf_test();
						break;
				case VK_F1:
                        flag1 = !flag1;
printf("flag1 %d\n", flag1);
                        InvalidateRect(hwnd, NULL, FALSE);
						break;
				case VK_F2:
						break;
				case VK_F5:
						subdiv_level++;
						InvalidateRect(hwnd, NULL, FALSE);
						break;
				case VK_F6:
						subdiv_level--;
						if (subdiv_level < 0)
						subdiv_level = 0;
						InvalidateRect(hwnd, NULL, FALSE);
						break;
				case VK_F11:
#if 0
						cubic_flag++;
						if (cubic_flag > 3)
							cubic_flag = 1;
if (cubic_flag == 1) printf("patch catmull_clark\n");
else if (cubic_flag == 2) printf("bspline catmull_clark\n");
else if (cubic_flag == 3) printf("modified simple\n");
						InvalidateRect(hwnd, NULL, FALSE);
#endif
						break;
				case VK_F12:
						//special();
						InvalidateRect(hwnd, NULL, FALSE);
						break;
				case VK_F9:
						bf_cull_flag = !bf_cull_flag;
						printf("backface cull %d\n", bf_cull_flag);
						InvalidateRect(hwnd, NULL, FALSE);
						break;
							
            }
            break;
        case WM_LBUTTONDOWN:
			first_move = 1;
			tracking_mouse = 1;
			break;
		case WM_LBUTTONUP:
			tracking_mouse = 0;
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
        case WM_DESTROY:
            PostQuitMessage (wParam);
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
    static char Title[] = "ss";

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
