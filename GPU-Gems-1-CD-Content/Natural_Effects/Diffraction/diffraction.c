/*
  ============================================================================
   diffraction.c --- sample code that demonstrates the Cg implementation of
					 the diffraction shader
  ----------------------------------------------------------------------------
   Author : Jos Stam (jstam@alias.com)

   This implementation uses GLUT for the user interface

  ----------------------------------------------------------------------------
*/

#include <stdlib.h>
#include <stdio.h>
#include <fstream.h>
#include <math.h>

#include <GL/glut.h>

#include <cg/cg.h>
#include <cg/cgGL.h>

#include "load_cubemap.h"
#include "mouse.h"


/*
  ----------------------------------------------------------------------------
	constants
  ----------------------------------------------------------------------------
*/

#define PI (3.1415926535897f)

#define N_DISK 256
#define N_VERT (2*N_DISK)
#define N_QUADS N_DISK

#define NX 256
#define N_VERTS ((NX+1)*(NX+1))
#define N_QUADSS (NX*NX)

#define ROUGHX 50.0f
#define ROUGHY 0.01f
#define SPACINGX 10.0f
#define INDEX 0.25f


/*
  ----------------------------------------------------------------------------
	static variables
  ----------------------------------------------------------------------------
*/

static int WinID;
static int ChangingView;
static int FirstTime;

static GLuint CubemapID;

/* Parameters */

static float RoughX, SpacingX, Reflectivity;

static float LightDir[3] = {0.0f,0.0f,1.0f};
static float EyePos[3] = {0.0f, 0.0f,1.0f}; 
static float MatCol[4] = {1.0f, 0.7f, 0.3f, 1.0f};

static int DrawCD;

static float BumpSpacing;

/* Cg variables */

static CGcontext cg_context;
static CGprofile cg_vp_profile;
static CGprogram cg_vp_program;
static CGprofile cg_fp_profile;
static CGprogram cg_fp_program;

static CGparameter cg_position;
static CGparameter cg_normal;
static CGparameter cg_tangent;

static CGparameter cg_ModelViewProjectionMatrix;
static CGparameter cg_ModelViewMatrix;
static CGparameter cg_ModelViewMatrixIT;
static CGparameter cg_roughX;
static CGparameter cg_spacingX;
static CGparameter cg_hiliteColor;
static CGparameter cg_lightPosition;
static CGparameter cg_eyePosition;

/* Geometry definition */

static int nverts=N_VERT;
static int nindicies=4*N_QUADS;

static unsigned int indicies[4*N_QUADS]; 
static float vertex_data[N_VERT][3];
static float normal_data[N_VERT][3];

static float tangent_data[N_VERT][3];
static float bnormal_data[N_VERT][3];

static float vertex_data2[N_VERT][3];
static float normal_data2[N_VERT][3];

static float tangent_data2[N_VERT][3];
static float bnormal_data2[N_VERT][3];

static int nverts_s=N_VERTS;
static int nindicies_s=4*N_QUADSS;

static unsigned int indicies_s[4*N_QUADSS]; 
static float vertex_data_s[N_VERTS][3];
static float normal_data_s[N_VERTS][3];

static float tangent_data_s[N_VERTS][3];
static float bnormal_data_s[N_VERTS][3];


/*
  ----------------------------------------------------------------------------
	error callback routine for Cg
  ----------------------------------------------------------------------------
*/

static void cgErrorCallback ( void )
{
    CGerror err = cgGetError();

    if ( err != CG_NO_ERROR ) {
        fprintf ( stderr, "cgErrorCallback(): %s\n", cgGetErrorString(err) );
        exit ( 1 );
    }
}


/*
  ----------------------------------------------------------------------------
	Generate data for a simple compact disk
  ----------------------------------------------------------------------------
*/

static void MakeMesh ( void )
{
	float t, dt, x, y;
	int i, ip;

	dt = 2*PI/(N_DISK-1.0f);
	for ( i=0 ; i<N_DISK ; i++ ) {
		t = i*dt;
		x = (float) cos(t);
		y = (float) sin(t);

		vertex_data[i][0] = 0.3f*x; vertex_data[i][1] = 0.3f*y; vertex_data[i][2] = 0.0f;
		vertex_data[i+N_DISK][0] = x; vertex_data[i+N_DISK][1] = y; vertex_data[i+N_DISK][2] = 0.0f;
		tangent_data[i][0] =        -y; tangent_data[i][1] =        x; tangent_data[i][2]        = 0.0f;
		tangent_data[i+N_DISK][0] = -y; tangent_data[i+N_DISK][1] = x; tangent_data[i+N_DISK][2] = 0.0f;
		bnormal_data[i][0] =        x; bnormal_data[i][1] =        y; bnormal_data[i][2]        = 0.0f;
		bnormal_data[i+N_DISK][0] = x; bnormal_data[i+N_DISK][1] = y; bnormal_data[i+N_DISK][2] = 0.0f;
		normal_data[i][0] =        0.0f; normal_data[i][1] =        0.0f; normal_data[i][2]        = 1.0f;
		normal_data[i+N_DISK][0] = 0.0f; normal_data[i+N_DISK][1] = 0.0f; normal_data[i+N_DISK][2] = 1.0f;

	}

	for ( i=0 ; i<N_DISK ; i++ ) {
		ip = i==N_DISK-1 ? 0 : i+N_DISK+1;

		indicies[4*i+0] = i;  indicies[4*i+1] = i+1;
		indicies[4*i+2] = ip; indicies[4*i+3] = i+N_DISK;
	}
}


/*
  ----------------------------------------------------------------------------
	Same mesh as above but a tiny larger to model the transparent layer
  ----------------------------------------------------------------------------
*/

static void MakeMesh2 ( void )
{
	float t, dt, x, y;
	int i;

	dt = 2*PI/(N_DISK-1.0f);
	for ( i=0 ; i<N_DISK ; i++ ) {
		t = i*dt;
		x = 1.02f * (float) cos(t);
		y = 1.02f * (float) sin(t);

		vertex_data2[i][0] = 0.1f*x; vertex_data2[i][1] = 0.1f*y; vertex_data2[i][2] = 0.0f;
		vertex_data2[i+N_DISK][0] = x; vertex_data2[i+N_DISK][1] = y; vertex_data2[i+N_DISK][2] = 0.0f;
		tangent_data2[i][0] =        -y; tangent_data2[i][1] =        x; tangent_data2[i][2]        = 0.0f;
		tangent_data2[i+N_DISK][0] = -y; tangent_data2[i+N_DISK][1] = x; tangent_data2[i+N_DISK][2] = 0.0f;
		bnormal_data2[i][0] =        x; bnormal_data2[i][1] =        y; bnormal_data2[i][2]        = 0.0f;
		bnormal_data2[i+N_DISK][0] = x; bnormal_data2[i+N_DISK][1] = y; bnormal_data2[i+N_DISK][2] = 0.0f;
		normal_data2[i][0] =        0.0f; normal_data2[i][1] =        0.0f; normal_data2[i][2]        = 1.0f;
		normal_data2[i+N_DISK][0] = 0.0f; normal_data2[i+N_DISK][1] = 0.0f; normal_data2[i+N_DISK][2] = 1.0f;

	}
}


/*
  ----------------------------------------------------------------------------
	Generate data for a square
  ----------------------------------------------------------------------------
*/

static void MakeMeshSquare ( void )
{
	float x, y, h;
	int idx, i, j;

	h = 1.0f/NX;

	for ( i=0 ; i<=NX ; i++ ) {
		x = 2*h*i-1;
		for ( j=0 ; j<=NX ; j++ ) {
			y = 2*h*j-1;
			idx = i+(NX+1)*j;

			vertex_data_s[idx][0] = x; vertex_data_s[idx][1] = y; vertex_data_s[idx][2] = 0;
			tangent_data_s[idx][0] = 1; tangent_data_s[idx][1] = 0; tangent_data_s[idx][2] = 0;
			bnormal_data_s[idx][0] = 0; bnormal_data_s[idx][1] = 1; bnormal_data_s[idx][2] = 0;
			normal_data_s[idx][0] = 0; normal_data_s[idx][1] = 0; normal_data_s[idx][2] = 1;
		}
	}

	for ( i=0 ; i<NX ; i++ ) {
		for ( j=0 ; j<NX ; j++ ) {
			idx = i+NX*j;
			indicies_s[4*idx+0] = (i+0)+(NX+1)*(j+0);
			indicies_s[4*idx+1] = (i+1)+(NX+1)*(j+0);
			indicies_s[4*idx+2] = (i+1)+(NX+1)*(j+1);
			indicies_s[4*idx+3] = (i+0)+(NX+1)*(j+1);
		}
	}
}


/*
  ----------------------------------------------------------------------------
	computes an angle value based on the location on the square
  ----------------------------------------------------------------------------
*/

static float angle ( float x, float y )
{
	float a;

	x *= BumpSpacing;
	y *= BumpSpacing;

	a = PI*cos(2*PI*x)*cos(2*PI*y);

	return ( a );
}


/*
  ----------------------------------------------------------------------------
	Add a twist value to the local frame at very vertex
  ----------------------------------------------------------------------------
*/

static void AddTwist ( void )
{
	int i, j, k, idx;
	float x, y, h, a, cos_a, sin_a;
	float U[3], V[3];

	h = 1.0f/NX;

	for ( i=0 ; i<=NX ; i++ ) {
		x = 2*i*h-1;
		for ( j=0 ; j<=NX ; j++ ) {
			y = 2*j*h-1;

			idx = i+(NX+1)*j;

			a = angle ( x, y );
			cos_a = (float) cos(a);
			sin_a = (float) sin(a);

			for ( k=0 ; k<3 ; k++ ) {
				U[k] =  cos_a * tangent_data_s[idx][k] + sin_a * bnormal_data_s[idx][k];
				V[k] = -sin_a * tangent_data_s[idx][k] + cos_a * bnormal_data_s[idx][k];
			}
			for ( k=0 ; k<3 ; k++ ) {
				tangent_data_s[idx][k] = U[k];
				bnormal_data_s[idx][k] = V[k];
			}
		}
	}
}


/*
  ----------------------------------------------------------------------------
	OpenGL drawing routines
  ----------------------------------------------------------------------------
*/

/*
  ----------------------------------------------------------------------------
	draws the cube map on a solid sphere
  ----------------------------------------------------------------------------
*/

static void DrawSkyBox ( void )
{
	glEnable ( GL_TEXTURE_CUBE_MAP_EXT );
	glBindTexture ( GL_TEXTURE_CUBE_MAP_EXT, CubemapID );
	glEnable ( GL_TEXTURE_GEN_S );
	glEnable ( GL_TEXTURE_GEN_T );
	glEnable ( GL_TEXTURE_GEN_R );
	glTexGeni ( GL_S, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP_EXT );
	glTexGeni ( GL_T, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP_EXT );
	glTexGeni ( GL_R, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP_EXT );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );

	glTexEnvi ( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	glMatrixMode ( GL_TEXTURE );
	glPushMatrix ();
	glLoadIdentity ();
	glScalef ( 1.0f, -1.0f, 1.0f );
	glRotatef ( 90.0f, 0.0f, 1.0f, 0.0f );
	glMatrixMode ( GL_MODELVIEW );

	glColor3f ( 0.5f, 0.5f, 0.5f );
	glutSolidSphere ( 100.0, 40, 20 );

	glMatrixMode ( GL_TEXTURE );
	glPopMatrix ();
	glMatrixMode ( GL_MODELVIEW );

	glDisable ( GL_TEXTURE_GEN_S );
	glDisable ( GL_TEXTURE_GEN_T );
	glDisable ( GL_TEXTURE_GEN_R );

	glDisable ( GL_TEXTURE_CUBE_MAP_EXT );
}


/*
  ----------------------------------------------------------------------------
	Draws the outline of the mesh to model the thickness of the coating
  ----------------------------------------------------------------------------
*/

static void DrawMeshOutline ( void )
{
	int i;

	glLineWidth ( 2.0f );

	glColor3f ( 1.0f, 1.0f, 1.0f );

	glBegin ( GL_LINE_LOOP );

		for ( i=0 ; i<N_DISK ; i++ ) {
			glVertex3fv ( vertex_data2[i] );
		}

	glEnd ();

	glBegin ( GL_LINE_LOOP );

		for ( i=N_DISK ; i<2*N_DISK ; i++ ) {
			glVertex3fv ( vertex_data2[i] );
		}

	glEnd ();
}


/*
  ----------------------------------------------------------------------------
	computes dot product...
  ----------------------------------------------------------------------------
*/

static float dot ( float * U, float * V )
{
	int k;
	float sum=0;

	for ( k=0 ; k<3 ; k++ ) sum += U[k]*V[k];

	return ( sum );
}


/*
  ----------------------------------------------------------------------------
	Setup everything Cg needs to render using a vertex program
  ----------------------------------------------------------------------------
*/

static void CgPreRender ( float * cg_p, float * cg_n, float * cg_t )
{

	cgGLEnableClientState ( cg_position );
	cgGLEnableClientState ( cg_normal   );
	cgGLEnableClientState ( cg_tangent  );

	cgGLSetParameterPointer ( cg_position, 3, GL_FLOAT, 0, cg_p );
	cgGLSetParameterPointer ( cg_normal,   3, GL_FLOAT, 0, cg_n );
	cgGLSetParameterPointer ( cg_tangent,  3, GL_FLOAT, 0, cg_t );

	cgGLSetStateMatrixParameter ( cg_ModelViewProjectionMatrix, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY          );
	cgGLSetStateMatrixParameter ( cg_ModelViewMatrix,           CG_GL_MODELVIEW_MATRIX,            CG_GL_MATRIX_IDENTITY          );
	cgGLSetStateMatrixParameter ( cg_ModelViewMatrixIT,         CG_GL_MODELVIEW_MATRIX,            CG_GL_MATRIX_INVERSE_TRANSPOSE );

	cgGLSetParameter1f  ( cg_spacingX,      SpacingX );
	cgGLSetParameter1f  ( cg_roughX,        RoughX   );
	cgGLSetParameter4fv ( cg_hiliteColor,   MatCol   );
	cgGLSetParameter3fv ( cg_lightPosition, LightDir );
	cgGLSetParameter3fv ( cg_eyePosition,   EyePos   );

	cgGLEnableProfile ( cg_vp_profile );
	cgGLEnableProfile ( cg_fp_profile );

	cgGLBindProgram ( cg_vp_program );
	cgGLBindProgram ( cg_fp_program );
}


/*
  ----------------------------------------------------------------------------
	cleanup Cg stuff after a render using a vertex program
  ----------------------------------------------------------------------------
*/

static void CgPostRender ( void )
{
	cgGLDisableProfile ( cg_fp_profile );
	cgGLDisableProfile ( cg_vp_profile );

	cgGLDisableClientState ( cg_tangent  );
	cgGLDisableClientState ( cg_normal   );
	cgGLDisableClientState ( cg_position );
}


/*
  ----------------------------------------------------------------------------
	Draw the CD mesh using a vertex program
  ----------------------------------------------------------------------------
*/

static void DrawMesh ( void )
{
	int i, j;

	CgPreRender ( vertex_data[0], normal_data[0], tangent_data[0] );
	
	glColor3f ( 1, 1, 1 );

	glBegin ( GL_QUADS );

		for ( i=0 ; i<4*N_QUADS ; i++ ) {
			j = indicies[i];
			glTexCoord3fv ( tangent_data[j] );
			glNormal3fv ( normal_data[j] ); glVertex3fv ( vertex_data[j] );
		}

	glEnd ();

	CgPostRender ();
}


/*
  ----------------------------------------------------------------------------
	Draw the square mesh using a vertex program
  ----------------------------------------------------------------------------
*/

static void DrawMeshSquare ( void )
{
	int i, j;

	CgPreRender ( vertex_data_s[0], normal_data_s[0], tangent_data_s[0] );

	glColor3f ( 1, 1, 1 );

	glBegin ( GL_QUADS );

		for ( i=0 ; i<4*N_QUADSS ; i++ ) {
			j = indicies_s[i];
			glTexCoord3fv ( tangent_data_s[j] );
			glNormal3fv ( normal_data_s[j] ); glVertex3fv ( vertex_data_s[j] );
		}

	glEnd ();

	CgPostRender ();
}


/*
  ----------------------------------------------------------------------------
	Draw the coating on the CD mesh taking into account Fresnel reflection of
	the environment
  ----------------------------------------------------------------------------
*/

static void DrawMesh2 ( void )
{
	int i, j;
	float U[3], V[3], W[3];
	float fresnel, cost;

	GetScreenFrame ( U, V, W );

	cost = dot ( W, normal_data[0] );

	fresnel = 0.3f + 0.7f*(1-cost*cost);

	fresnel *= 0.8f;

	glColor3f ( fresnel, fresnel, fresnel );

	glBegin ( GL_QUADS );

		for ( i=0 ; i<4*N_QUADS ; i++ ) {
			j = indicies[i];
			glNormal3fv ( normal_data2[j] ); glVertex3fv ( vertex_data2[j] );
		}

	glEnd ();
}


/*
  ----------------------------------------------------------------------------
	Draw the coating on the square mesh taking into account Fresnel reflection
	of the environment
  ----------------------------------------------------------------------------
*/

static void DrawMeshSquare2 ( void )
{
	float U[3], V[3], W[3];
	float fresnel, cost, F;

	GetScreenFrame ( U, V, W );

	cost = dot ( W, normal_data_s[0] );

	fresnel = 0.3f + 0.7f*(1-cost*cost);

	fresnel *= 0.8f;

	glColor3f ( fresnel, fresnel, fresnel );

	glNormal3f ( 0, 0, 1 );

	F = 1.01f;

	glBegin ( GL_QUADS );

		glVertex2f ( -F, -F );
		glVertex2f (  F, -F );
		glVertex2f (  F,  F );
		glVertex2f ( -F,  F );

	glEnd ();
}


/*
  ----------------------------------------------------------------------------
	Draw the coating on top of mesh
  ----------------------------------------------------------------------------
*/

static void DrawMeshTex ( void )
{
	float col[4] = { 1.0f, 0.0f, 1.0f, 1.0f };

	glEnable ( GL_TEXTURE_CUBE_MAP_EXT ); 

	glTexGeni ( GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP_EXT );
	glTexGeni ( GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP_EXT );
	glTexGeni ( GL_R, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP_EXT );

	glEnable ( GL_TEXTURE_GEN_S );
	glEnable ( GL_TEXTURE_GEN_T );
	glEnable ( GL_TEXTURE_GEN_R );

	glTexEnvi ( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

	glMatrixMode ( GL_TEXTURE );
	glPushMatrix ();
	glLoadIdentity ();
	glScalef ( 1.0f, -1.0f, 1.0f );
	glRotatef ( 90.0f, 0.0f, 1.0f, 0.0f );
	glMatrixMode ( GL_MODELVIEW );

	if ( DrawCD ) {
		DrawMesh2 ();
	} else {
		DrawMeshSquare2 ();
	}

	glMatrixMode ( GL_TEXTURE );
	glPopMatrix ();
	glMatrixMode ( GL_MODELVIEW );

	glDisable ( GL_TEXTURE_GEN_S );
	glDisable ( GL_TEXTURE_GEN_T );
	glDisable ( GL_TEXTURE_GEN_R );

	glDisable ( GL_TEXTURE_CUBE_MAP_EXT );
}


/*
  ----------------------------------------------------------------------------
	Initialize Cg/OpenGL stuff
  ----------------------------------------------------------------------------
*/

static void InitOpenGL ( void )
{
	cgSetErrorCallback ( cgErrorCallback );

	/* set Cg variables */

	cg_context = cgCreateContext ();
	cg_vp_profile = cgGLGetLatestProfile ( CG_GL_VERTEX   );
	cg_fp_profile = cgGLGetLatestProfile ( CG_GL_FRAGMENT );
	cgGLSetOptimalOptions ( cg_vp_profile );
	cgGLSetOptimalOptions ( cg_fp_profile );
	cg_vp_program = cgCreateProgramFromFile ( cg_context, CG_SOURCE, "vp/vp_Diffraction.cg", cg_vp_profile, "vp_Diffraction", NULL );
	cg_fp_program = cgCreateProgramFromFile ( cg_context, CG_SOURCE, "vp/fp_Diffraction.cg", cg_fp_profile, "fp_Diffraction", NULL );
	cgGLLoadProgram ( cg_vp_program );
	cgGLLoadProgram ( cg_fp_program );
	cgGLEnableProfile( cg_vp_profile );
	cgGLEnableProfile( cg_fp_profile );

	cg_position = cgGetNamedParameter ( cg_vp_program, "position" );
	cg_normal	= cgGetNamedParameter ( cg_vp_program, "normal"   );
	cg_tangent	= cgGetNamedParameter ( cg_vp_program, "tangent"  );

	cg_ModelViewProjectionMatrix = cgGetNamedParameter ( cg_vp_program, "ModelViewProjectionMatrix" );
	cg_ModelViewMatrix           = cgGetNamedParameter ( cg_vp_program, "ModelViewMatrix"           );
	cg_ModelViewMatrixIT         = cgGetNamedParameter ( cg_vp_program, "ModelViewMatrixIT"         );

	cg_roughX        = cgGetNamedParameter ( cg_vp_program, "roughX"        );
	cg_spacingX      = cgGetNamedParameter ( cg_vp_program, "spacingX"      );
	cg_hiliteColor   = cgGetNamedParameter ( cg_vp_program, "hiliteColor"   );
	cg_lightPosition = cgGetNamedParameter ( cg_vp_program, "lightPosition" );
	cg_eyePosition   = cgGetNamedParameter ( cg_vp_program, "eyePosition"   );

	/* Disable lighting since we are doing our own. */
	glDisable ( GL_LIGHTING );

	/* Enable depth testing. */
	glEnable ( GL_DEPTH_TEST );

	/* set environment texture map variables */

	glGenTextures ( 1, &CubemapID );

	glEnable ( GL_TEXTURE_CUBE_MAP_EXT );
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
	glBindTexture ( GL_TEXTURE_CUBE_MAP_EXT, CubemapID );

	glTexParameteri ( GL_TEXTURE_CUBE_MAP_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR );
	glTexParameteri ( GL_TEXTURE_CUBE_MAP_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );

	load_bmp_cubemap ( "bmp/uffizi_%s.BMP", 1 );

	glDisable ( GL_TEXTURE_CUBE_MAP_EXT );
}


/*
  ----------------------------------------------------------------------------
	Clean up on exit and save viewing parameters
  ----------------------------------------------------------------------------
*/

static void QuitProgram ( void )
{
	if ( !SaveView() ) {
		fprintf ( stderr, "cannot save viewing parameters\n" );
	}

	cgDestroyContext ( cg_context );

	exit ( 0 );
}


/*
  ----------------------------------------------------------------------------
	GLUT callbacks
  ----------------------------------------------------------------------------
*/

static void KeyFunc ( unsigned char key, int x, int y )
{
	switch ( key )
	{
		case 'q':
		case 'Q':

			QuitProgram ();
			break;

		case 'c':
		case 'C':

			DrawCD = !DrawCD;
			break;

		case '>':

			BumpSpacing *= 1.3f;
			MakeMeshSquare ();
			AddTwist ();
			break;

		case '<':

			BumpSpacing /= 1.3f;
			MakeMeshSquare ();
			AddTwist ();
			break;

	}

	return;
}

static void IdleFunc ( void )
{
	glutSetWindow ( WinID );
	glutPostRedisplay ();
}

static void DisplayFunc ( void )
{

	SetBackGnd ( 0.0f, 0.0f, 0.0f );

	StartDisplay ();

		glDisable ( GL_DEPTH_TEST );
		DrawSkyBox ();
		glEnable ( GL_DEPTH_TEST );

		if ( DrawCD ) {
			DrawMesh ();
		} else {
			DrawMeshSquare ();
		}

		glEnable ( GL_BLEND );
		glBlendFunc ( GL_ONE, GL_ONE );
		glDisable ( GL_DEPTH_TEST );

		DrawMeshTex ();

		glEnable ( GL_DEPTH_TEST );
		glDisable ( GL_BLEND );

		if ( DrawCD ) {
			DrawMeshOutline ();
		}

	EndDisplay ();
}

static void MouseFunc ( int button, int state, int x, int y )
{
	if ( glutGetModifiers () & GLUT_ACTIVE_ALT ) {
		MouseStart ( button, state, x, y );
		ChangingView = TRUE;
		return;
	}
	ChangingView = FALSE;

	MouseStart ( button, state, x, y );
}

static void MotionFunc ( int x, int y )
{
	if ( ChangingView ) {
		MouseHandle ( x, y );
		return;
	}

	MouseHandle ( x, y );
}

	
/*
  ----------------------------------------------------------------------------
	initializes static variables
  ----------------------------------------------------------------------------
*/

static void InitGlobals ()
{
	ChangingView = FALSE;
	FirstTime = TRUE;

	DrawCD = 1;

	BumpSpacing = 1;

	MakeMesh ();
	MakeMesh2 ();

	MakeMeshSquare ();
	AddTwist ();

	RoughX = ROUGHX;
	SpacingX = SPACINGX;
	Reflectivity = INDEX;
}


/*
  ----------------------------------------------------------------------------
	main routine
  ----------------------------------------------------------------------------
*/

int main ( int argc, char ** argv )
{
/* glut stuff */

	glutInit ( &argc, argv );

	if ( argc != 1 ) {
		fprintf ( stderr, "%s : none\n", argv[0] );
		exit ( 1 );
	}

	InitGlobals ();

/* set up the view */

	WinID = InitWindow ( 50, 50, 700, 700, 40.0f );

	glutDisplayFunc ( DisplayFunc );
	glutIdleFunc ( IdleFunc );
	glutKeyboardFunc ( KeyFunc );
	glutMouseFunc ( MouseFunc );
	glutMotionFunc ( MotionFunc );

/* give control to GLUT */

	InitOpenGL ();

	glutMainLoop ();

	return ( 0 );
}
