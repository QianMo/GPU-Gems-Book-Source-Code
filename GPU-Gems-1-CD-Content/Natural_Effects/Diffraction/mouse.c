/*
  =============================================================================
   mouse.c --- handles all mouse events and display events
  -----------------------------------------------------------------------------
   Author : Jos Stam (jstam@alias.com)

  =============================================================================
*/

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <GL/glut.h>

#include <math.h>

#include "mouse.h"

/*
  -----------------------------------------------------------------------------
   constants
  -----------------------------------------------------------------------------
*/

#define TRACK_RADIUS 0.4f
#define MOUSE_SPEEDZ 5.0f
#define MOUSE_SPEEDXY 2.0f

#define UP   0
#define DOWN 1

#ifdef _WIN32
/* math defines from math.h */
#define M_SQRT2 1.41421356237309504880
#define M_SQRT1_2   0.70710678118654752440
#endif


/*
  -----------------------------------------------------------------------------
   static variables
  -----------------------------------------------------------------------------
*/

static float RotVec[4];		/* Euler rotation parameters */
static float RotMat[16];	/* Corresponding rotation matrix */
static float Trans[4];		/* translational part of viewing transformation */
static long OrgX, OrgY;		/* origin of window */
static long SizeX, SizeY;	/* size of window */
static float BackGndR, BackGndG, BackGndB;
							/* background color */

static int already_set_view=0;

static short LeftMouse;		/* status of left mouse button (UP or DOWN) */
static short MiddleMouse;	/* status of middle mouse button */
static short RightMouse;	/* status of right mouse button */
static int MouseX;			/* current X mouse coordinate */
static int MouseY;			/* current Y mouse coordinate */
static float Zoom;			/* zoom factor for the display */

static time_t StartTime;	/* used to compute frames per second */

static int winID;			/* GLUT identifier of the window */


/*
  -----------------------------------------------------------------------------
	simple vector routines
  -----------------------------------------------------------------------------
*/

static void vcopy ( float * v1, float * v2 )
{
	int i;

	for ( i=0 ; i<3 ; i++ ) {
		v2[i] = v1[i];
	}
}

static void vset ( float * v, float x, float y, float z )
{
	v[0] = x;
	v[1] = y;
	v[2] = z;
}

static void vzero ( float * v )
{
	v[0] = 0;
	v[1] = 0;
	v[2] = 0;
}

static float vlength ( float * v )
{
	return ( (float)sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] ) );
}

static void vscale ( float * v, float div )
{
	v[0] *= div;
	v[1] *= div;
	v[2] *= div;
}

static void vnormal ( float * v )
{
	vscale ( v, 1/vlength(v) );
}

static void vmult ( float * src1, float * src2, float * dst )
{
	dst[0] = src1[0] * src2[0];
	dst[1] = src1[1] * src2[1];
	dst[2] = src1[2] * src2[2];
}

static void vadd ( float * src1, float * src2, float * dst )
{
	dst[0] = src1[0] + src2[0];
	dst[1] = src1[1] + src2[1];
	dst[2] = src1[2] + src2[2];
}

static void vsub ( float * src1, float * src2, float * dst )
{
	dst[0] = src1[0] - src2[0];
	dst[1] = src1[1] - src2[1];
	dst[2] = src1[2] - src2[2];
}

static float vdot ( float * v1, float * v2 )
{
	return ( v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] );
}

static void vcross ( float * v1, float * v2, float * cross )
{
	float temp[3];

	temp[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
	temp[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
	temp[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
	vcopy ( temp, cross );
}

/*
  -----------------------------------------------------------------------------
   quaternion/euler angle handling routines
  -----------------------------------------------------------------------------
*/

/*
 * Euler paramaters always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
 * We'll normalize based on this formula.  Also, normalize greatest
 * component, to avoid problems that occur when the component we're
 * normalizing gets close to zero (and the other components may add up
 * to more than 1.0 because of rounding error).
 */
static void normalize_euler ( float * e )
{
	int which, i;
	float gr;

	which = 0;
	gr = e[which];
	for ( i=1 ; i<4 ; i++ ) {
		if ( fabs(e[i]) > fabs(gr) ) {
			gr = e[i];
			which = i;
		}
	}

	e[which] = 0;

	e[which] = (float) sqrt(1 - (e[0]*e[0] + e[1]*e[1] + e[2]*e[2] + e[3]*e[3]));

	/* Check to see if we need negative square root */
	if ( gr < 0.0f ) {
		e[which] = -e[which];
	}
}

/*
 *	Given two rotations, e1 and e2, expressed as Euler paramaters,
 * figure out the equivalent single rotation and stuff it into dest.
 * 
 * This routine also normalizes the result every COUNT times it is
 * called, to keep error from creeping in.
 */
#define COUNT 100
static void add_eulers ( float * e1, float * e2, float * dest )
{
	static int count=0;
	int i;
	float t1[3], t2[3], t3[3];
	float tf[4];

	vcopy ( e1, t1 ); vscale ( t1, e2[3] );
	vcopy ( e2, t2 ); vscale ( t2, e1[3] );
	vcross ( e2, e1, t3 );
	vadd ( t1, t2, tf );
	vadd ( t3, tf, tf );
	tf[3] = e1[3] * e2[3] - vdot(e1,e2);

	for ( i=0 ; i<4 ; i++ ) {
		dest[i] = tf[i];
	}

	if ( ++count > COUNT ) {
		count = 0;
		normalize_euler ( dest );
	}
}


/*
 * Build a rotation matrix, given Euler paramaters.
 */
static void build_rotmatrix ( float * m, float * e )
{
	m[0 ] = 1 - 2*(e[1]*e[1] + e[2]*e[2]);
	m[1 ] =     2*(e[0]*e[1] - e[2]*e[3]);
	m[2 ] =     2*(e[2]*e[0] + e[1]*e[3]);
	m[3 ] = 0;

	m[4 ] =     2*(e[0]*e[1] + e[2]*e[3]);
	m[5 ] = 1 - 2*(e[2]*e[2] + e[0]*e[0]);
	m[6 ] =     2*(e[1]*e[2] - e[0]*e[3]);
	m[7 ] = 0;

	m[8 ] =     2*(e[2]*e[0] - e[1]*e[3]);
	m[9 ] =     2*(e[1]*e[2] + e[0]*e[3]);
	m[10] = 1 - 2*(e[1]*e[1] + e[0]*e[0]);
	m[11] = 0;

	m[12] = 0;
	m[13] = 0;
	m[14] = 0;
	m[15] = 1;
}


/*
  -----------------------------------------------------------------------------
	GetScreenFrame --- get the frame projected into the view
  -----------------------------------------------------------------------------
*/

void GetScreenFrame ( float * S, float * T, float * N )
{
	S[0] = RotMat[0 ];
	S[1] = RotMat[4 ];
	S[2] = RotMat[8 ];

	T[0] = RotMat[1 ];
	T[1] = RotMat[5 ];
	T[2] = RotMat[9 ];

	N[0] = RotMat[2 ];
	N[1] = RotMat[6 ];
	N[2] = RotMat[10];
}


/*
  -----------------------------------------------------------------------------
   InitMouseVars --- Init global variables
  -----------------------------------------------------------------------------
*/

void InitMouseVars ()
{
	int i;
	FILE * f;

	/* all buttons up at the beginning */

	LeftMouse = UP;
	MiddleMouse = UP;
	RightMouse = UP;

	if ( already_set_view ) return;

	/* set default values for viewing parameters */

	for ( i=0 ; i<16 ; i++ ) {
		RotMat[i] = 0;
	}

	for ( i=0 ; i<4 ; i++ ) {
		RotMat[4*i] = 1;
		RotVec[i] = 0;
		Trans[i] = 0;
	}
	RotVec[3] = 1;
	Trans[2] = 60;

	/* try to read view parameters from the view.param file */

	if ( (f=fopen(".view.param","r")) ) {
		fscanf ( f, "%f %f %f ", &Trans[0], &Trans[1], &Trans[2] );
		fscanf ( f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f ",
						&RotMat[0 ], &RotMat[1 ], &RotMat[2 ], &RotMat[3 ],
						&RotMat[4 ], &RotMat[5 ], &RotMat[6 ], &RotMat[7 ],
						&RotMat[8 ], &RotMat[9 ], &RotMat[10], &RotMat[11],
						&RotMat[12], &RotMat[13], &RotMat[14], &RotMat[15] );
		fscanf ( f, "%f %f %f %f ", &RotVec[0],  &RotVec[1],  &RotVec[2],  &RotVec[3] );
		fclose ( f );
	} else {
		fprintf ( stderr, "cannot open view file\n" );
	}
}


/*
  -----------------------------------------------------------------------------
	SaveView  --- save viewing parameters in a file called .view.param
  -----------------------------------------------------------------------------
*/

int SaveView ()
{
	FILE * f;
	if ( !(f=fopen(".view.param","w")) ) {
		return ( FALSE );
	}

	fprintf ( f, "%f %f %f\n", Trans[0], Trans[1], Trans[2] );
	fprintf ( f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
					RotMat[0 ], RotMat[1 ], RotMat[2 ], RotMat[3 ],
					RotMat[4 ], RotMat[5 ], RotMat[6 ], RotMat[7 ],
					RotMat[8 ], RotMat[9 ], RotMat[10], RotMat[11],
					RotMat[12], RotMat[13], RotMat[14], RotMat[15] );
	fprintf ( f, "%f %f %f %f\n", RotVec[0],  RotVec[1],  RotVec[2], RotVec[3] );
	fclose ( f );

	return ( TRUE );
}


/*
  -----------------------------------------------------------------------------
	callback routines for keyboard for mouse events
  -----------------------------------------------------------------------------
*/

void MouseStart ( int button, int state, int x, int y )
{
	int button_status;

	button_status = state == GLUT_DOWN ? DOWN : UP;

	switch ( button )
	{
		case GLUT_LEFT_BUTTON:
			LeftMouse = button_status;
			break;

		case GLUT_MIDDLE_BUTTON:
			MiddleMouse = button_status;
			break;

		case GLUT_RIGHT_BUTTON:
			RightMouse = button_status;
			break;
	}

	MouseX = x;
	MouseY = y;
}




/*
  -----------------------------------------------------------------------------
   UpdateViewing --- well the title says it all...
  -----------------------------------------------------------------------------
*/

static void UpdateViewing ( float * r, float * t )
{
   vadd ( t,Trans, Trans );
   add_eulers ( r, RotVec, RotVec );
   build_rotmatrix ( RotMat, RotVec );
}


/*
  -----------------------------------------------------------------------------
   TranslateXY --- do XY translation from mouse movement
  -----------------------------------------------------------------------------
*/

static void TranslateXY ( int mx, int my )
{
   float r[4], t[3];

   vzero ( r ); r[3] = 1;
   vset( t, (float)(mx-MouseX)/(float)SizeX*MOUSE_SPEEDXY,
            (float)(MouseY-my)/(float)SizeY*MOUSE_SPEEDXY,
			0.0f );

   UpdateViewing ( r, t );

   MouseX = mx;
   MouseY = my;
}


/*
  -----------------------------------------------------------------------------
   TranslateZ -- do Z translation from mouse movement
  -----------------------------------------------------------------------------
*/

static void TranslateZ ( int mx, int my )
{
   float r[4], t[3];

   vzero ( r ); r[3] = 1.0f;
   vzero ( t );

   t[2] = (float)(mx-MouseX)/(float)SizeX + (float)(MouseY-my)/(float)SizeY;
   t[2] *= MOUSE_SPEEDZ;

   UpdateViewing ( r, t );

   MouseX = mx;
   MouseY = my;
}


/*
  -----------------------------------------------------------------------------
   project_to_sphere --- project (x,y) on sphere (or hyperbola)
  -----------------------------------------------------------------------------
*/

static float project_to_sphere ( float x, float y )
{
   float d, t, z;

   d = sqrt ( x*x + y*y );
   if ( d < TRACK_RADIUS*M_SQRT1_2 ) {
      z = (float) sqrt ( TRACK_RADIUS*TRACK_RADIUS - d*d );
   } else {
      t = TRACK_RADIUS / M_SQRT2;
      z = t*t / d;
   }

   return ( z );
}


/*
  -----------------------------------------------------------------------------
   CalcRotate --- calculate rotation from mouse movement
  -----------------------------------------------------------------------------
*/

static void CalcRotate ( float * r, float p1x, float p1y, float p2x, 
float p2y )
{
   float p1[3], p2[3];
   float dp[3];
   float axis[3];
   float phi;

   vzero ( axis );

   if ( p1x == p2x && p1y == p2y ) {
      vzero ( r );
      r[3] = 1;
      return;
   }

   vset ( p1, p1x, p1y, project_to_sphere(p1x,p1y) );
   vset ( p2, p2x, p2y, project_to_sphere(p2x,p2y) );

   vcross ( p2,  p1, axis );

   vsub ( p1, p2, dp );
   phi = (float) sin(vlength(dp) / (2*TRACK_RADIUS));
   vnormal ( axis );
   vcopy ( axis, r );
   vscale ( r, (float)sin(phi/2) );
   r[3] = (float)cos(phi/2);
}
   

/*
  -----------------------------------------------------------------------------
   Rotate --- do rotation from mouse movement
  -----------------------------------------------------------------------------
*/

static void Rotate ( int mx, int my )
{
   float r[4], t[3];
   float p1x, p1y, p2x, p2y;

   vzero ( t );
   
   p1x =  2*(MouseX-OrgX)/(float)SizeX - 1;
   p1y = -2*(MouseY-OrgY)/(float)SizeY + 1;
   p2x =  2*(mx    -OrgX)/(float)SizeX - 1;
   p2y = -2*(my    -OrgY)/(float)SizeY + 1;

   CalcRotate ( r, p1x, p1y, p2x, p2y );

   UpdateViewing ( r, t );

   MouseX = mx;
   MouseY = my;
}


/*
  -----------------------------------------------------------------------------
   MouseHandle --- handles current state of the mouse (motion callback)
  -----------------------------------------------------------------------------
*/

void MouseHandle ( int x, int y )
{
	if ( LeftMouse == UP && RightMouse == DOWN ) {
		TranslateXY ( x, y );
	}
	if ( LeftMouse == DOWN && RightMouse == DOWN ) {
		TranslateZ ( x, y );
	}
	if ( LeftMouse == DOWN && RightMouse == UP ) {
		Rotate ( x, y );
	}
}

/*
  -----------------------------------------------------------------------------
	SetBackGnd -- set the background colour
  -----------------------------------------------------------------------------
*/

void SetBackGnd ( float r, float g, float b )
{
	BackGndR = r;
	BackGndG = g;
	BackGndB = b;
}


/*
  -----------------------------------------------------------------------------
   StartDisplay --- clean up before displaying
  -----------------------------------------------------------------------------
*/

void StartDisplay ( void )
{
	float aspect;

	StartTime = clock ();

	aspect =  (float)SizeX / (float)SizeY;

	glViewport ( 0, 0, SizeX, SizeY );

	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();

	gluPerspective ( Zoom, aspect, 0.01f, 200.0f );

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glTranslatef ( Trans[0], Trans[1], Trans[2] );
	glMultMatrixf ( RotMat );

	glClearColor ( BackGndR, BackGndG, BackGndB,  1.0f );
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
}



/*
  -----------------------------------------------------------------------------
   EndDisplay --- clean up at end of display
  -----------------------------------------------------------------------------
*/

void EndDisplay ( void )
{
	char wname[256];
	time_t t;
	float dt;

	t = clock ();
	dt = (t-StartTime)/(float)CLOCKS_PER_SEC;

	if ( dt > 0.0f ) {
		sprintf ( wname, "Real-Time Diffraction   by Jos Stam            FPS :  %6.2f", 1.0f/dt );
		glutSetWindowTitle( wname );
	}

	glutSwapBuffers ();
}


/*
  -----------------------------------------------------------------------------
	ReshapeFunc -- reshape callback function
  -----------------------------------------------------------------------------
*/

static void ReshapeFunc ( int width, int height )
{
	SizeX = width;
	SizeY = height;
}


/*
  -----------------------------------------------------------------------------
   InitWindow --- open working window and set viewing parameters
  -----------------------------------------------------------------------------
*/

int InitWindow ( long oX, long oY, long sX, long sY, float zoom )
{
	GLfloat lmodel_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };

	OrgX = oX;
	OrgY = oY;
	SizeX = sX;
	SizeY = sY;
	Zoom = zoom;

	InitMouseVars ();

	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );

	glutInitWindowPosition ( OrgX, OrgY );
	glutInitWindowSize ( SizeX, SizeY );
	winID = glutCreateWindow ( "Real-Time Diffraction   by Jos Stam            FPS :" );

	glEnable ( GL_LIGHT0 );
	glEnable ( GL_LIGHTING );
	glShadeModel ( GL_SMOOTH );
	glEnable ( GL_DEPTH_TEST );
	glDisable ( GL_CULL_FACE );
	glCullFace ( GL_BACK );
	glLightModelfv ( GL_LIGHT_MODEL_AMBIENT, lmodel_ambient );

	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glutSwapBuffers ();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glutSwapBuffers ();

	glLineWidth ( 5.0f );

	glutReshapeFunc ( ReshapeFunc );

	SetBackGnd ( 0.0f, 0.0f, 0.0f );

	glEnable ( GL_POLYGON_OFFSET_LINE );

	glPolygonOffset ( 1, 1e-6f );

	return ( winID );
}
