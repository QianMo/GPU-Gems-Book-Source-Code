/**
 *
 * fast third order texture filtering
 *
 * This demo shows how to perform fast cubic filtering. It renders an isosurface from
 * synthetic, regularly sampled volume data. Using B-Spline recontruction, it is
 * possible to compute smooth normal and principle curvature of the surface on the fly.
 * For simplicity, no empty-space-skipping acceleration structure has been used for this demo.
 *
 * (c) christian sigg (eth zurich), markus hadwiger (vrvis vienna) 12/2004
 * do not copy code before asking the authors.
 *
 **/

#if defined(WIN32)
#  include <windows.h>
#endif

#define GLH_EXT_SINGLE_FILE
#include <glh/glh_extensions.h>
#include <glh/glh_obs.h>
#include <glh/glh_glut.h>

#include <shared/read_text_file.h>
#include <shared/quitapp.h>

#include "RenderTextureFix.h"
#include "slicing.h"
#include "blobby.h"

using namespace glh;

// forward declarations
void init_opengl();
void init_data(GeneratePtr _fun);
void init_lu_tex();
void init_programs();
void display();
void mouse(int,int,int,int);
void motion(int,int);
void idle();
void menu(int);
void key(unsigned char,int,int);
void resize(int,int);

// key toggle array
bool b[256];

// glh interaction
glut_simple_mouse_interactor camera;

// attributes
GLint positionAttrib = -1;
GLint tangentBasisAttrib = -1;
GLint texcoordAttrib = -1;
GLint modelViewIParam = -1;
GLint lightParam = -1;
GLint halfangleParam = -1;

// programs
GLuint slicing_fp, pos_fp, grad_fp[3], hess_fp[6], curv_fp, shading_fp[4];
GLuint slicing_vp, quad_vp;

// textures
tex_object_3D vol_tex;
tex_object_1D lu_val_tex, lu_der_tex, lu_color_tex;

// render to texture buffer
RenderTextureFix *fbuffer, *dbuffer;

//sizes
GLuint lu_tex_size = 128;
GLuint vol_tex_size = 64;
GLuint pbuffer_size = 256;

// parameters
float num_slices = 512.0f;
float iso_value = 0.15f;
vec4f curv_bias = vec4f(35.0f, 18.0f, 0.35f, -0.6f);
int show_buffer = 4;

void adjust_log(float& x, const float& i) 
{ std::cout << (x *= (20+i)/(20-i)) << std::endl; }
void adjust_lin(float& x, const float& i)
{ std::cout << (x += i*0.05) << std::endl; }

typedef void(*AdjustPtr)(float&,const float&);
AdjustPtr adjust_fun = &adjust_lin;
float*    adjust_val = &iso_value;

// clear colors
const float bg_color[4][3] = {
	0.95, 0.95, 0.70,
	0.24, 0.24, 0.10,
	0.22, 0.32, 0.24,
	0.82, 0.97, 0.82,
};

const float quad_bias[4][4] = {
	0.5, 0.5, -0.5,  0.5,
	0.5, 0.5,  0.5,  0.5,
	0.5, 0.5, -0.5, -0.5,
	0.5, 0.5,  0.5, -0.5,
};

unsigned char lu_color_data[];

/*

context, target and texture unit usage in each pass (* means only set once):

stage      | pass | context | target | vol | depth | pos | grad | hess | curv | lu_v | lu_d | lu_c
--------------------------------------------------------------------------------------------------
slicing    | 1    | dbuffer | depth  |  0* |       |     |      |      |      |  3*  |      | 
back_proj  | 2    | fbuffer | front  |     |   1   |     |      |      |      |      |      | 
derivative | 3- 5 |    "    | aux0   |  0* |       |  1  |      |      |      |  3*  |  4*  | 
hessian    | 6-11 |    "    | aux2+3 |  0* |       |  1  |      |      |      |  3*  |  4*  | 
curvature  | 12   |    "    | aux1   |     |       |     |   0  | 1+2  |      |      |      | 
shading    | 13   | fbuffer | back   |     |       |  0  |   1  |      |  2   |      |      |  3*

*/

/************** MAIN ***************/

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

	bool fullscreen = false;
	for(int i=0; i<argc; ++i)
			fullscreen |= (strcmp(argv[i], "-fullscreen")==0);

	glutGameModeString("1024x768:16");
	fullscreen &= glutGameModeGet(GLUT_GAME_MODE_POSSIBLE) != 0;

	if (fullscreen) 
		glutEnterGameMode();
	else {
		glutInitWindowSize(2*pbuffer_size, 2*pbuffer_size);
		glutInitWindowPosition(100, 100);
		glutCreateWindow("fast third order texture filtering");

		glutCreateMenu(menu);
		glutAddMenuEntry("[1] Display buffer 1    ", '1');
		glutAddMenuEntry("[2] Display buffer 2    ", '2');
		glutAddMenuEntry("[3] Display buffer 3    ", '3');
		glutAddMenuEntry("[4] Display buffer 4    ", '4');
		glutAddMenuEntry("[5] Display all buffers ", '5');
		glutAddMenuEntry("                        ", '~');
						  
		glutAddMenuEntry("[d] Generate dumbell    ", 'd');
		glutAddMenuEntry("[p] Generate pyramid    ", 'p');
		glutAddMenuEntry("[c] Generate cube       ", 'c');
		glutAddMenuEntry("[r] Generate random     ", 'r');
		glutAddMenuEntry("                        ", '~');
							  
		glutAddMenuEntry("[v] Adjust isovalue mode", 'v');
		glutAddMenuEntry("[n] Adjust slices mode  ", 'n');
		glutAddMenuEntry("[b] Adjust k1 bias mode ", 'b');
		glutAddMenuEntry("[B] Adjust k2 bias mode ", 'B');
		glutAddMenuEntry("[s] Adjust k1 scale mode", 's');
		glutAddMenuEntry("[S] Adjust k2 scale mode", 'S');
		glutAddMenuEntry("                        ", '~');
						  
		glutAddMenuEntry("[+] Increase current    ", '+');
		glutAddMenuEntry("[-] Decrease current    ", '-');
		glutAddMenuEntry("                        ", '~');

		glutAddMenuEntry("[q] Quit                ", '\033');
		glutAttachMenu(GLUT_RIGHT_BUTTON);
	}

	init_opengl();

	camera.configure_buttons(1);
	camera.dolly.dolly[2] = -6;
	camera.trackball.incr = rotationf(vec3f(0.5, 0.7, 0.3), 0.03);
	glut_add_interactor(&camera);
	camera.enable();

	glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(idle);
    glutKeyboardFunc(key);
    glutReshapeFunc(resize);

	glutReportErrors();

    b[' '] = true;

    glutMainLoop();

	return 0;
}

/************** OpenGL STUFF ***************/

/** initialize opengl stuff */
void init_opengl()
{
	// initialize extensions
    if (!glh_init_extensions(
		"GL_VERSION_1_4 "
		"GL_ARB_vertex_program "
		"GL_ARB_fragment_program "
		"GL_ATI_texture_float "
		"WGL_ARB_pbuffer "
        "WGL_ARB_pixel_format "
        "WGL_ARB_render_texture "
		"WGL_NV_render_depth_texture "
		"WGL_ATI_pixel_format_float "
		))
    {
		std::cout << "Unable to load the following extension(s):\n"
			<< glh_get_unsupported_extensions() 
			<< "\n\nExiting...\n";
        quitapp(-1);
    }

	slicing_fp =  pos_fp = grad_fp[0] = grad_fp[1] = grad_fp[2] = hess_fp[0] = hess_fp[1] 
		= hess_fp[2] = hess_fp[3] = hess_fp[4] = hess_fp[5] = curv_fp = shading_fp[0] 
		= shading_fp[1] = shading_fp[2] = shading_fp[3] = slicing_vp = quad_vp = 0;

	// load programs
	init_programs();

	// generate data
	init_data(&cube);

	// generate lookup textures
	init_lu_tex();

	vec4f lu_bias(1.0 - 1.0/lu_tex_size, 0.5/lu_tex_size, 0, 0);
	vec4f vol_res(vol_tex_size, vol_tex_size, vol_tex_size, 0);
	vec4f vol_res_inv(1.0f/vol_tex_size, 1.0f/vol_tex_size, 1.0f/vol_tex_size, 0.0f);
	vec4f vol_bias(1.0f - 3.0f/vol_tex_size, 1.5f/vol_tex_size, -0.001f, 0.005f);

	// initialize depth pbuffer
	dbuffer = new RenderTextureFix("rgb depth texture2D=depth", pbuffer_size, pbuffer_size, GL_TEXTURE_2D);
    dbuffer->Activate();
	resize(pbuffer_size, pbuffer_size);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glEnable(GL_VERTEX_PROGRAM_ARB);
	// bind slicing program
    glBindProgramARB(GL_VERTEX_PROGRAM_ARB, slicing_vp);
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, slicing_fp);
	// mask all color channels (just depth)
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_DEPTH_TEST);
	// enable alpha test
	glEnable(GL_ALPHA_TEST);
	// bind textures
	glActiveTexture(GL_TEXTURE0); vol_tex.bind();
	glActiveTexture(GL_TEXTURE3); lu_val_tex.bind();
	// set program parameter
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 0, lu_bias.v);
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 1, vol_res.v);
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 2, vol_res_inv.v);
	glProgramEnvParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 1, vol_bias.v);
	dbuffer->Deactivate();

    // initialize float pbuffer
    fbuffer = new RenderTextureFix("rgba ati_float=16 stencil aux=4 texture2D", pbuffer_size, pbuffer_size, GL_TEXTURE_2D);
    fbuffer->Activate();
	resize(pbuffer_size, pbuffer_size);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glEnable(GL_VERTEX_PROGRAM_ARB);
	// mask depth
	glDepthMask(GL_TRUE);
	// bind textures
	glActiveTexture(GL_TEXTURE0); vol_tex.bind();
	glActiveTexture(GL_TEXTURE3); lu_val_tex.bind();
	glActiveTexture(GL_TEXTURE4); lu_der_tex.bind();
	// set program parameter
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 0, lu_bias.v);
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 1, vol_res.v);
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 2, vol_res_inv.v);
	// set quad_vp to be screen filling
	glProgramEnvParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 0, vec4f(1,1,0,0).v);
	fbuffer->Deactivate();

    // initialize frame buffer
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glEnable(GL_VERTEX_PROGRAM_ARB);
	// mask alpha
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
	// bind textures
	glActiveTexture(GL_TEXTURE3); lu_color_tex.bind();
	// set light position
	glLightfv(GL_LIGHT0, GL_POSITION, vec4f(4, -4, 3, 1).v);
	// set program parameter
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 3, vol_bias.v);

}

// generate blobby data and upload to texture
void init_data(GeneratePtr _fun)
{
	std::cout << "generating data ... ";
	unsigned short* data = _fun(vol_tex_size);
	vol_tex.bind();

	glTexImage3D(vol_tex.target, 0, GL_INTENSITY16, 
		vol_tex_size, vol_tex_size, vol_tex_size, 
		0, GL_LUMINANCE, GL_UNSIGNED_SHORT, data);

	vol_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	vol_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	vol_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	vol_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	vol_tex.parameter(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	delete[] data;

	std::cout << "\n";
}

void init_lu_tex()
{
	// initialize lookup textures
	float *data   = new float[3 * lu_tex_size];
	float *ptr    = data;
	float  h      = 1.0f / (lu_tex_size-1);
	float  c      = 1.0f / 6;
	float  phi[4], x3, x2, x = 0;
	unsigned int i;
	for(i = 0; i < lu_tex_size; ++i, x += h)
	{
		x3 = (x2 = x * x) * x;
		// cubic spline basis functions
		// non-negative partition of one; in [0, 2/3]
		phi[0] = c * (-x3 + 3*x2 - 3*x + 1);
		phi[1] = c * (3*x3 - 6*x2 + 4);
		phi[2] = c * (-3*x3 + 3*x2 + 3*x + 1);
		phi[3] = c * (x3);
		// offsets; first with changed sign; in [0, 1]
		*ptr++ = 1.0 + x - phi[1] / (phi[0] + phi[1]);
		*ptr++ = 1.0 - x + phi[3] / (phi[2] + phi[3]);
		// weight for left sample (lerp: right weight = 1-left weight)
		*ptr++ = phi[0] + phi[1];
	}

	// border not allowed for 16bit formats
	lu_val_tex.bind();
	lu_val_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	lu_val_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	lu_val_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

	ptr = data; x = 0;
	for(i = 0; i < lu_tex_size; ++i, x += h)
	{
		x2 = x * x;
		// cubic spline basis functions derivatives
		// 0,1 negative; 2,3 positive; in [-2/3, 2/3]
		phi[0] = c * (-3*x2 + 6*x - 3);
		phi[1] = c * (9*x2 - 12*x);
		phi[2] = c * (-9*x2 + 6*x + 3);
		phi[3] = c * (3*x2);
		// offsets; first with changed sign; in [0, 1]
		*ptr++ = 1.0 + x - phi[1] / (phi[0] + phi[1]);
		*ptr++ = 1.0 - x + phi[3] / (phi[2] + phi[3]);
		// weight for right sample (central differences: left weight = -right weight)
		*ptr++ = phi[2] + phi[3];
	}

	// border not allowed for 16bit formats
	lu_der_tex.bind();
	glTexImage1D(lu_der_tex.target, 0, GL_RGB16, lu_tex_size, 0, GL_RGB, GL_FLOAT, data);
	lu_der_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	lu_der_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	lu_der_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

	delete[] data;

	lu_color_tex.bind();
	glTexImage1D(lu_color_tex.target, 0, GL_RGB, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, lu_color_data);
	lu_der_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	lu_der_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	lu_der_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
}

// load a vertex or fragment program from a string
bool load_program(GLenum program_type, GLuint& program_id, const char *file_name)
{
	if(program_id == 0)
		glGenProgramsARB(1, &program_id);

	GLcharARB *program_data = read_text_file(file_name);

	if(program_data == 0)
	{
		char buf[10];
		std::cout << "Press <enter> to exit.\n";
		std::cin.getline(buf, 10);
		exit(-1);
	}

	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, 
		(GLsizei)strlen(program_data), (const GLubyte*)program_data);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
	if (error_pos != -1) {
		std::cerr << ((program_type == GL_VERTEX_PROGRAM_ARB) ? "vertex" : "fragment")
			<< " program '" << file_name << "' error at position: " << error_pos << "\n" 
			<< glGetString(GL_PROGRAM_ERROR_STRING_ARB) << "\n";
	}

	delete[] program_data;

	return error_pos == -1;
}


void init_programs()
{
	bool success = true;
    success &= load_program(GL_VERTEX_PROGRAM_ARB, slicing_vp, "fast_third_order/slicing.vp");
    success &= load_program(GL_VERTEX_PROGRAM_ARB, quad_vp, "fast_third_order/quad.vp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, slicing_fp, "fast_third_order/slicing.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, pos_fp, "fast_third_order/pos.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, grad_fp[0], "fast_third_order/grad_x.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, grad_fp[1], "fast_third_order/grad_y.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, grad_fp[2], "fast_third_order/grad_z.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, hess_fp[0], "fast_third_order/hess_xx.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, hess_fp[1], "fast_third_order/hess_yy.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, hess_fp[2], "fast_third_order/hess_zz.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, hess_fp[3], "fast_third_order/hess_xy.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, hess_fp[4], "fast_third_order/hess_xz.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, hess_fp[5], "fast_third_order/hess_yz.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, curv_fp, "fast_third_order/curv.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, shading_fp[0], "fast_third_order/shading_depth.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, shading_fp[1], "fast_third_order/shading_normal.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, shading_fp[2], "fast_third_order/shading_k1.fp");
    success &= load_program(GL_FRAGMENT_PROGRAM_ARB, shading_fp[3], "fast_third_order/shading_k2.fp");

	if(!success)
	{
		char buf[10];
		std::cout << "Press <enter> to exit.\n";
		std::cin.getline(buf, 10);
		exit(-1);
	}
}

/************** GLUT STUFF ***************/

void menu(int i)
{
	key((unsigned char) i, 0, 0);
}

void display()
{
	matrix4f mv = camera.get_transform();

	/***** stage one: slicing *****/

	dbuffer->Activate();

	glClear(GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv.m);
	glTranslatef(-2, -2, -2);
	glScalef(4,4,4);

	// set alpha test value
	glAlphaFunc(GL_LESS, iso_value);
	
	// draw bounding box
	glTexCoord3s(0, 0, 0);
	draw_cube();

	// draw slices
	vec3f dir(0,0,1);
	camera.trackball.r.inverse().mult_vec(dir);
	draw_slices(dir.v,num_slices);

	dbuffer->Deactivate();

	/***** stage two: back projection *****/

	// render to fbuffer
    fbuffer->Activate();

	glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv.m);
	glTranslatef(-2, -2, -2);
	glScalef(4,4,4);

	// bind depth texture, render to front
	glDrawBuffer(GL_FRONT_LEFT);
	glActiveTexture(GL_TEXTURE1);
	dbuffer->Bind(WGL_DEPTH_COMPONENT_NV);

	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	// clear color and stencil (guess it's faster with depth too)
	glClearColor(0, 0, 0, -1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// bind position program
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, pos_fp);
    glBindProgramARB(GL_VERTEX_PROGRAM_ARB, quad_vp);

	// write stencil mask (far depths will be killed in fp)
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 1, 1);
	glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);

	// render depth texture to position buffer
	glRectf(-1, -1, 1, 1);

	// release depth texture
	dbuffer->Release(WGL_DEPTH_COMPONENT_NV);

	/***** stage three: compute gradient *****/

	// test for stencil as set in last stage
	glStencilFunc(GL_EQUAL, 1, 1);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

	// bind position texture, render to aux0
	glDrawBuffer(GL_AUX0);
	glActiveTexture(GL_TEXTURE1);
	fbuffer->Bind(WGL_FRONT_LEFT_ARB);

	// derivative in x direction
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, grad_fp[0]);
	glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	// derivative in y direction
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, grad_fp[1]);
	glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	// derivative in z direction
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, grad_fp[2]);
	glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	/***** stage four: compute hessian *****/

	// render to aux2
	glDrawBuffer(GL_AUX2);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, hess_fp[0]);
	glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, hess_fp[1]);
	glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, hess_fp[2]);
	glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	// render to aux3
	glDrawBuffer(GL_AUX3);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, hess_fp[3]);
	glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, hess_fp[4]);
	glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, hess_fp[5]);
	glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE);
	glRectf(-1, -1, 1, 1);

	// release front
	fbuffer->Release(WGL_FRONT_LEFT_ARB);

	/***** stage five: compute curvature *****/

	// bind aux0+2+3 (grad+hessian), render to aux1
	glDrawBuffer(GL_AUX1);
	glActiveTexture(GL_TEXTURE0);
	fbuffer->Bind(WGL_AUX0_ARB);
	glActiveTexture(GL_TEXTURE1);
	fbuffer->Bind(WGL_AUX2_ARB);
	glActiveTexture(GL_TEXTURE2);
	fbuffer->Bind(WGL_AUX3_ARB);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, curv_fp);
	glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_FALSE);

	glRectf(-1, -1, 1, 1);

	// release aux0+2+3
	fbuffer->Release(WGL_AUX0_ARB);
	fbuffer->Release(WGL_AUX2_ARB);
	fbuffer->Release(WGL_AUX3_ARB);

	glDisable(GL_STENCIL_TEST);

    fbuffer->Deactivate();

	/***** stage six: shading *****/

	glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv.m);
	glTranslatef(-2, -2, -2);
	glScalef(4,4,4);

	// bind front,aux0+1 (pos+grad+curv)
	glDrawBuffer(GL_BACK);
	glActiveTexture(GL_TEXTURE0);
	fbuffer->Bind(WGL_FRONT_LEFT_ARB);
	glActiveTexture(GL_TEXTURE1);
	fbuffer->Bind(WGL_AUX0_ARB);
	glActiveTexture(GL_TEXTURE2);
	fbuffer->Bind(WGL_AUX1_ARB);

	// set curvature bias
	glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 4, curv_bias.v);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    if (show_buffer==4) 
	{	// show all buffers
		for(int i = 0; i<4; ++i)
		{	// set position of quad
			glProgramEnvParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 0, quad_bias[i]);
			// bind shading program
			glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shading_fp[i]);
			// set background color
		    glColor3fv(bg_color[i]);
			// render quad
			glRectf(-1, -1, 1, 1);
		}
	} else {
        // show one buffer
		glProgramEnvParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 0, vec4f(1,1,0,0).v);
		glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shading_fp[show_buffer]);
	    glColor3fv(bg_color[show_buffer]);
		glRectf(-1, -1, 1, 1);
    }

	// release pos, grad and curv
	fbuffer->Release(WGL_FRONT_LEFT_ARB);
	fbuffer->Release(WGL_AUX0_ARB);
	fbuffer->Release(WGL_AUX1_ARB);

	glutSwapBuffers();

	glutReportErrors();

}

void idle()
{
	if (b[' ']) {
        camera.trackball.increment_rotation();
	    glutPostRedisplay();
	}
}

void key(unsigned char k, int x, int y)
{
	b[k] = ! b[k];

	switch(k) {
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
		show_buffer = k - '1';
		break;

	case 'd':
        init_data(&dumbell);
		break;
	case 'p':
        init_data(&pyramid);
		break;
	case 'c':
        init_data(&cube);
		break;
	case 'r':
        init_data(&random);
		break;

	case 'v':
		adjust_fun = &adjust_lin;
		adjust_val = &iso_value;
		break;
	case 'n':
		adjust_fun = &adjust_log;
		adjust_val = &num_slices;
		break;
	case 's':
		adjust_fun = &adjust_log;
		adjust_val = &curv_bias[0];
		break;
	case 'S':
		adjust_fun = &adjust_log;
		adjust_val = &curv_bias[1];
		break;
	case 'b':
		adjust_fun = &adjust_lin;
		adjust_val = &curv_bias[2];
		break;
	case 'B':
		adjust_fun = &adjust_lin;
		adjust_val = &curv_bias[3];
		break;

	case '+':
		adjust_fun(*adjust_val,  1);
		break;
	case '-':
		adjust_fun(*adjust_val, -1);
		break;

	case 'x':
		init_programs();
		break;

	case 27:
	case 'q':
		exit(0);
		break;
	}

	camera.keyboard(k, x, y);
    
	glutPostRedisplay();
}

void resize(int w, int h)
{
   if (h == 0) h = 1;

    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    gluPerspective(45.0, (GLfloat)w/(GLfloat)h, 3.0, 9.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    camera.reshape(w, h);
}

void mouse(int button, int state, int x, int y)
{
	camera.mouse(button, state, x, y);
}

void motion(int x, int y)
{
    camera.motion(x, y);
}


unsigned char lu_color_data[] = {
 255, 170, 255,
 255, 168, 251,
 255, 167, 247,
 255, 166, 243,
 255, 164, 239,
 255, 163, 235,
 255, 162, 231,
 255, 160, 227,
 255, 159, 223,
 255, 158, 219,
 255, 156, 215,
 255, 155, 211,
 255, 154, 207,
 255, 152, 203,
 255, 151, 199,
 255, 150, 195,
 255, 148, 191,
 255, 147, 187,
 255, 146, 183,
 255, 144, 179,
 255, 143, 175,
 255, 142, 171,
 255, 140, 167,
 255, 139, 163,
 255, 138, 159,
 255, 136, 155,
 255, 135, 151,
 255, 134, 147,
 255, 132, 143,
 255, 131, 139,
 255, 130, 135,
 255, 128, 131,
 255, 127, 127,
 255, 126, 123,
 255, 124, 119,
 255, 123, 115,
 255, 122, 111,
 255, 120, 107,
 255, 119, 103,
 255, 118,  99, 
 255, 116,  95, 
 255, 115,  91, 
 255, 114,  87, 
 255, 112,  83, 
 255, 111,  79, 
 255, 110,  75, 
 255, 108,  71, 
 255, 107,  67, 
 255, 106,  63, 
 255, 104,  59, 
 255, 103,  55, 
 255, 102,  51, 
 255, 100,  47, 
 255,  99,  43,
 255,  98,  39,
 255,  96,  35,
 255,  95,  31,
 255,  94,  27,
 255,  92,  23,
 255,  91,  19,
 255,  90,  15,
 255,  88,  11,
 255,  87,   7, 
 255,  86,   3, 
 255,  85,   0, 
 255,  87,   0, 
 255,  90,   0, 
 255,  92,   0, 
 255,  95,   0, 
 255,  98,   0, 
 255, 100,   0, 
 255, 103,   0, 
 255, 106,   0, 
 255, 108,   0, 
 255, 111,   0, 
 255, 114,   0, 
 255, 116,   0, 
 255, 119,   0, 
 255, 122,   0, 
 255, 124,   0, 
 255, 127,   0, 
 255, 130,   0, 
 255, 132,   0, 
 255, 135,   0, 
 255, 138,   0, 
 255, 140,   0, 
 255, 143,   0, 
 255, 146,   0, 
 255, 148,   0, 
 255, 151,   0, 
 255, 154,   0, 
 255, 156,   0, 
 255, 159,   0, 
 255, 162,   0, 
 255, 164,   0, 
 255, 167,   0, 
 255, 170,   0, 
 255, 172,   0, 
 255, 175,   0, 
 255, 177,   0, 
 255, 180,   0, 
 255, 183,   0, 
 255, 185,   0, 
 255, 188,   0, 
 255, 191,   0, 
 255, 193,   0, 
 255, 196,   0, 
 255, 199,   0, 
 255, 201,   0, 
 255, 204,   0, 
 255, 207,   0, 
 255, 209,   0, 
 255, 212,   0, 
 255, 215,   0, 
 255, 217,   0, 
 255, 220,   0, 
 255, 223,   0, 
 255, 225,   0, 
 255, 228,   0, 
 255, 231,   0, 
 255, 233,   0, 
 255, 236,   0, 
 255, 239,   0, 
 255, 241,   0, 
 255, 244,   0, 
 255, 247,   0, 
 255, 249,   0, 
 255, 252,   0, 
 255, 255,   0, 
 252, 251,   3, 
 249, 247,   7, 
 247, 243,  11, 
 244, 239,  15, 
 241, 235,  19, 
 239, 231,  23, 
 236, 227,  27, 
 233, 223,  31, 
 231, 219,  35, 
 228, 215,  39, 
 225, 211,  43, 
 223, 207,  47, 
 220, 203,  51, 
 217, 199,  55, 
 215, 195,  59, 
 212, 191,  63, 
 209, 187,  67, 
 207, 183,  71, 
 204, 179,  75, 
 201, 175,  79, 
 199, 171,  83, 
 196, 167,  87, 
 193, 163,  91, 
 191, 159,  95, 
 188, 155,  99, 
 185, 151, 103, 
 183, 147, 107, 
 180, 143, 111, 
 177, 139, 115, 
 175, 135, 119, 
 172, 131, 123, 
 170, 127, 127, 
 167, 123, 131, 
 164, 119, 135, 
 162, 115, 139, 
 159, 111, 143, 
 156, 107, 147, 
 154, 103, 151, 
 151,  99, 155, 
 148,  95, 159, 
 146,  91, 163, 
 143,  87, 167, 
 140,  83, 171, 
 138,  79, 175, 
 135,  75, 179, 
 132,  71, 183, 
 130,  67, 187, 
 127,  63, 191, 
 124,  59, 195, 
 122,  55, 199, 
 119,  51, 203, 
 116,  47, 207, 
 114,  43, 211, 
 111,  39, 215, 
 108,  35, 219, 
 106,  31, 223, 
 103,  27, 227, 
 100,  23, 231, 
  98,  19, 235, 
  95,  15, 239, 
  92,  11, 243, 
  90,   7, 247, 
  87,   3, 251, 
  85,   0, 255, 
  86,   4, 255, 
  87,   8, 255, 
  89,  12, 255, 
  90,  16, 255, 
  91,  20, 255, 
  93,  24, 255, 
  94,  28, 255, 
  95,  32, 255, 
  97,  36, 255, 
  98,  40, 255, 
  99,  44, 255, 
 101,  48, 255, 
 102,  52, 255, 
 103,  56, 255, 
 105,  60, 255, 
 106,  64, 255, 
 107,  68, 255, 
 109,  72, 255, 
 110,  76, 255, 
 111,  80, 255, 
 113,  85, 255, 
 114,  89, 255, 
 116,  93, 255, 
 117,  97, 255, 
 118, 101, 255, 
 120, 105, 255, 
 121, 109, 255, 
 122, 113, 255, 
 124, 117, 255, 
 125, 121, 255, 
 126, 125, 255, 
 128, 129, 255, 
 129, 133, 255, 
 130, 137, 255, 
 132, 141, 255, 
 133, 145, 255, 
 134, 149, 255, 
 136, 153, 255, 
 137, 157, 255, 
 138, 161, 255, 
 140, 165, 255, 
 141, 170, 255, 
 143, 174, 255, 
 144, 178, 255, 
 145, 182, 255, 
 147, 186, 255, 
 148, 190, 255, 
 149, 194, 255, 
 151, 198, 255, 
 152, 202, 255, 
 153, 206, 255, 
 155, 210, 255, 
 156, 214, 255, 
 157, 218, 255, 
 159, 222, 255, 
 160, 226, 255, 
 161, 230, 255, 
 163, 234, 255, 
 164, 238, 255, 
 165, 242, 255, 
 167, 246, 255, 
 168, 250, 255, 
 170, 255, 255
};

