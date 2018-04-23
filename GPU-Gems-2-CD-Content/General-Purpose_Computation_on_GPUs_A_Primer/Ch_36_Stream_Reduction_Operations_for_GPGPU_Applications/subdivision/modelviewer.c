#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/**
 ** Note: this is the only operating system dependency in the whole code.
 **	  With this #ifdef, this code runs without change on UNIX, Windows, 
 **       and Mac.
 **	  GLUT gets most of the credit for this...
 **/

#ifdef WIN32
#define M_PI	3.14159265
#include <windows.h>
#endif


/* include the OpenGL include files:					*/

#include <GL/glut.h>


/**
 **
 **	This is a sample of using the OpenGL and GLUT libraries
 **
 **
 **	The left mouse button does rotation
 **	The middle mouse button does scaling
 **	The right mouse button brings up a pop-up menu that allows:
 **		1. The projection to be changed
 **		2. The colors to be changed
 **		3. Axes on and off
 **		4. The transformations to be reset
 **		5. The program to quit
 **
 **	Author: Mike Houston
 **
 **/


/**
 ** defines:
 **/

/* title of this window:						*/

#define WINDOWTITLE	"Test App - Mike Houston"


/* the escape key:							*/

#define ESCAPE		0x1b



/* upper-left corner of the window:					*/

#define WIN_LEFT	30
#define WIN_TOP		30


/* initial window size:							*/

#define WINDOW_SIZE	1000


/* minimum scale factor allowed:					*/

#define MIN_SCALE	0.01


/* multiplication factors for input interaction:			*/
/*  (these are known from previous experience)				*/

#define ANGFACT		1.0
#define SCLFACT		 0.005


/* active mouse buttons (or them together):				*/

#define LEFT		4
#define MIDDLE		2
#define RIGHT		1


/* values for menu items:						*/

#define  SETCOLORS	1
#define  RESET		2
#define  QUIT		3

#define  ORTHO		1
#define  PERSP		2

#define  RED		1
#define  YELLOW		2
#define  GREEN		3
#define  CYAN		4
#define  BLUE		5
#define  MAGENTA	6
#define  WHITE		7
#define  BLACK		8


/* window background color (rgba):					*/

#define BACKGROUND_COLOR	0.,0.,0.,0.


/* color and line width for the axes:					*/

#define AXES_COLOR	0.,1.,0.
#define AXES_SIZE	18.
#define AXES_WIDTH	5.


/* whether the left button is rotate or scale:				*/

#define ROTATE		0
#define SCALE		1


/* parameters for the torus:						*/

#define RADIUS1		 6.0
#define RADIUS2		10.0
#define SLICES		30
#define STACKS		20
#define TORUS_WIDTH	2.


/* handy to have around:						*/

#ifndef FALSE
#define FALSE		0
#define TRUE		( ! FALSE )
#endif

#define OFF		FALSE
#define ON		TRUE

#define DISABLED	FALSE
#define ENABLED		TRUE



/**
 ** global variables:
 **/
char wire=0;
int togglered;
int	ActiveButton;		/* current button that is down		*/
int	AxesList;		/* list to hold the axes		*/
int	AxesMenu;		/* id of the axes pop-up menu		*/
int	AxesOnOff;		/* ON or OFF				*/
int	ColorMenu;		/* id of the color pop-up menu		*/
int	Debug;			/* non-zero means print debug info	*/
int	GrWindow;		/* window id for top-level window	*/
int	MainMenu;		/* id of the main pop-up menu		*/
int	Projection;		/* ORTHO or PERSP			*/
int	ProjMenu;		/* id of the projection pop-up menu	*/
float	Red, Green, Blue;	/* star colors				*/
float	Scale;			/* scaling factor			*/
int	ObjList[4];		/* list to hold the 3d object		*/
int whichObj=0;
int	TransformMode;		/* ROTATE or SCALE			*/
int	TransformModeMenu;	/* id of the transform mode menu	*/
float	Xrot, Yrot;		/* rotation angles in degrees		*/
int	Xmouse, Ymouse;		/* mouse values				*/
typedef char *str ;
str model[4]={"data","data.cracks","data.liberal","data.conservative"};


/**
 ** function prototypes:
 **/

void	Animate( void );
void	Axes( float length );
void	Display( void );
void	DoAxesMenu( int value );
void	DoColorMenu( int value );
void	DoMainMenu( int value );
void	DoProjMenu( int value );
void	DoRasterString( float x, float y, float z, char *s );
void	DoStrokeString( float x, float y, float z, float ht, char *s );
void	DoTransformModeMenu( int value );
void	InitGraphics( void );
void	InitLists( void );
void	Keyboard( unsigned char, int, int );
void	Keyboard2( int, int, int );
void	MouseButton( int button, int state, int x, int y );
void	MouseMotion( int x, int y );
void	Quit( void );
void	Reset( void );
void	Resize( int width, int height );
void	Visibility( int state );


/**
 ** main program:
 **/

int
main( int argc, char *argv[] )
{
	int i;			/* counter				*/


	/* turn on the glut package:					*/
	/* (do this before checking argc and argv since it might	*/
	/* pull some command line arguments out)			*/

	glutInit( &argc, argv );


	/* set defaults:						*/

	Debug = FALSE;


	/* read the command line:					*/

	for( i=1; i < argc; i++ )
	{
		if( strcmp( argv[i], "-D" )  ==  0 )
		{
			Debug = TRUE;
			continue;
		}else {
		  if (i==1) {
		    model[0]=argv[i];
		  }
		  if (i==2) {
		    model[1]=argv[i];
		  }
		  if (i==3) {
		    model[2]=argv[i];
		  }

		}
/*
		fprintf( stderr, "Unknown argument: '%s'\n", argv[i] );
		fprintf( stderr, "Usage: %s [-D]\n", argv[0] );*/
	}


	/* setup all the graphics stuff, including callbacks:		*/

	InitGraphics();


	/* init the transformations and colors:				*/
	/* (will also post a redisplay)					*/

	Reset();


	/* draw the scene once and wait for some interaction:		*/
	/* (will never return)						*/

	glutMainLoop();


	/* keep lint happy:						*/

	return 0;
}



/**
 ** this is where one would put code that is to be called
 ** everytime the glut main loop has nothing to do, ie,
 ** the glut idle function
 **
 ** this is typically where animation happens
 **/

void
Animate( void )
{
	/* put animation stuff in here -- set some global variables	*/
	/* for Display() to find:					*/

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** draw the complete scene:
 **/

void
Display( void )
{
	int dx, dy, d;		/* viewport dimensions			*/
	int xl, yb;		/* lower-left corner of viewport	*/

        float red[4]={1,0,0,1};
        float white[4]={1,1,1,1};
	/* set which window we want to do the graphics into:		*/

	glutSetWindow( GrWindow );


	/* erase the background:					*/

	glDrawBuffer( GL_BACK );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );


	/* specify shading to be flat:					*/

	glShadeModel( GL_SMOOTH );


	/* set the viewport to a square centered in the window:		*/

	dx = glutGet( GLUT_WINDOW_WIDTH );
	dy = glutGet( GLUT_WINDOW_HEIGHT );
	d = dx < dy ? dx : dy;			/* minimum dimension	*/
	xl = ( dx - d ) / 2;
	yb = ( dy - d ) / 2;
	glViewport( xl, yb,  d, d );


	/* set the viewing volume:					*/
	/* remember that the eye is at the origin looking in -Z		*/
	/* remember that the Znear and Zfar values are actually		*/
	/* given as DISTANCES IN FRONT OF THE EYE			*/

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();

	if( Projection == ORTHO )
		glOrtho( -15., 15.,     -15., 15.,     0.1, 60. );
	else
		gluPerspective( 70., 1.,	0.1, 60. );


	/* viewing transform:						*/

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	gluLookAt( 0., 0., 25.,     0., 0., 0.,     0., 1., 0. );


	/* perform the rotations and scaling about the origin:		*/

	glRotatef( Yrot, 0., 1., 0. );
	glRotatef( Xrot, 1., 0., 0. );
	glScalef( Scale, Scale, Scale );


	/* set the color of the object:					*/

	glColor3f( Red, Green, Blue );




	/* draw the object:						*/

        glLightfv(GL_LIGHT0,GL_DIFFUSE,white);
        glColor4fv(white);
        if (!wire) {
          glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);			  
          glEnable(GL_LIGHTING);
          glEnable(GL_LIGHT0);
        }else {			    
          glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
          glDisable(GL_LIGHTING);
          glDisable(GL_LIGHT0);
          
        }

	glCallList( ObjList[whichObj] );
        glLightfv(GL_LIGHT0,GL_DIFFUSE,red);        
        if (togglered) {
          glColor4fv(red);
          glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);			  
          glDisable(GL_LIGHTING);
          glDisable(GL_LIGHT0);
          
          glCallList( ObjList[0] );
        }
	/* possibly draw the axes:					*/

	if( AxesOnOff == ON )
		glCallList( AxesList );


	/* draw some text that rotates and scales with the scene:	*/

	glEnable( GL_DEPTH_TEST );
	glColor3f( 1., 1., 0. );
/*	DoStrokeString( 0., AXES_SIZE, 0.,  AXES_SIZE/10.,   "Top of Axes" );*/


	/* draw some text that just rotates with the scene:		*/

	glDisable( GL_DEPTH_TEST );
	glColor3f( 0., 1., 1. );
	/*DoRasterString( 0., 0., AXES_SIZE,   "Front of Axes" );
*/

	/* draw some text that is fixed on the screen:			*/

	glDisable( GL_DEPTH_TEST );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D( 0., 100.,     0., 100. );	/* setup "percent units"*/
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	glColor3f( 1., 0., 1. );
/*	DoRasterString( 5., 5., 0., "Sample App - Mike Houston" );*/


	/* swap the double-buffered framebuffers:			*/

	glutSwapBuffers();


	/* be sure the graphics buffer has been sent:			*/

	glFlush();
}




/**
 ** process the axes pop-up menu:
 **/

void
DoAxesMenu( int value )
{
	AxesOnOff = value;

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}




/**
 ** process the color pop-up menu:
 **/

void
DoColorMenu( int value )
{
	Red = Green = Blue = 1.;

	switch( value )
	{
		case RED:
			Green = Blue = 0.;
			break;

		case YELLOW:
			Blue = 0.;
			break;

		case GREEN:
			Red = Blue = 0.;
			break;

		case CYAN:
			Red = 0.;
			break;

		case BLUE:
			Red = Green = 0.;
			break;

		case MAGENTA:
			Green = 0.;
			break;

		case WHITE:
			break;

		case BLACK:
			Red = Green = Blue = 0.;
			break;

		default:
			fprintf( stderr, "Unknown color menu value: %d\n", value );
	}

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}




/**
 ** process the main pop-up menu:
 **/

void
DoMainMenu( int value )
{
	switch( value )
	{
		case RESET:
			Reset();
			break;

		case QUIT:
			Quit();
			break;	/* never returns -- "don't need to do this" */

		default:
			fprintf( stderr, "Unknown main menu value: %d\n", value );
	}
}




/**
 ** process the projection pop-up menu:
 **/

void
DoProjMenu( int value )
{
	Projection = value;

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** use glut to display a string of characters using a raster font:
 ** (raster fonts are made of pixels that get "rubber-stamped" onto the screen)
 **/

void
DoRasterString( float x, float y, float z, char *s )
{
	char c;			/* one character to print		*/

	glRasterPos3f( x, y, z );
	for( ; ( c = *s ) != '\0'; s++ )
	{
		glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, c );
	}
}



/**
 ** use glut to display a string of characters using a stroke font:
 ** (stroke fonts are made of 3D lines that get glBegin-glEnd'ed onto the screen)
 **/

void
DoStrokeString( float x, float y, float z, float ht, char *s )
{
	char c;			/* one character to print		*/
	float sf;		/* the scale factor			*/

	glPushMatrix();
		glTranslatef( x, y, z );
		sf = ht / ( 119.05 + 33.33 );
		glScalef( sf, sf, sf );
		for( ; ( c = *s ) != '\0'; s++ )
		{
			glutStrokeCharacter( GLUT_STROKE_ROMAN, c );
		}
	glPopMatrix();
}




/**
 ** process the transform mode pop-up menu:
 **/

void
DoTransformModeMenu( int value )
{
	TransformMode = value;

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** initialize the glut and OpenGL libraries:
 **	also setup display lists and callback functions
 **/

void
InitGraphics( void )
{
	/* setup the display mode:					*/
	/* ( *must* be done before call to glutCreateWindow() )		*/

	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );

	/* set the initial window configuration:			*/

	glutInitWindowSize( WINDOW_SIZE, WINDOW_SIZE );
	glutInitWindowPosition( WIN_LEFT, WIN_TOP );


	/* open the window and set its title:				*/

	GrWindow = glutCreateWindow( WINDOWTITLE );
	glutSetWindowTitle( WINDOWTITLE );

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	/* setup the clear values:					*/

	glClearColor( BACKGROUND_COLOR );


	/* initialize the pop-up menus:					*/

	ProjMenu = glutCreateMenu( DoProjMenu );
	glutAddMenuEntry( "Orthographic", ORTHO );
	glutAddMenuEntry( "Perspective",  PERSP );

	ColorMenu = glutCreateMenu( DoColorMenu );
	glutAddMenuEntry( "Red",	RED );
	glutAddMenuEntry( "Yellow",	YELLOW );
	glutAddMenuEntry( "Green",	GREEN );
	glutAddMenuEntry( "Cyan",	CYAN );
	glutAddMenuEntry( "Blue",	BLUE );
	glutAddMenuEntry( "Magenta",	MAGENTA );
	glutAddMenuEntry( "White",	WHITE );
	glutAddMenuEntry( "Black",	BLACK );

	AxesMenu = glutCreateMenu( DoAxesMenu );
	glutAddMenuEntry( "Off", OFF );
	glutAddMenuEntry( "On",  ON );

	TransformModeMenu = glutCreateMenu( DoTransformModeMenu );
	glutAddMenuEntry( "Rotate", ROTATE );
	glutAddMenuEntry( "Scale",  SCALE );

	MainMenu = glutCreateMenu( DoMainMenu );
	glutAddSubMenu( "Projection",     ProjMenu );
	glutAddSubMenu( "Transform Mode", TransformModeMenu );
	glutAddSubMenu( "Set Color",      ColorMenu );
	glutAddSubMenu( "Axes",           AxesMenu );
	glutAddMenuEntry( "Reset",        RESET );
	glutAddMenuEntry( "Quit",         QUIT );


	/* attach the pop-up menu to the right mouse button:		*/

	glutAttachMenu( GLUT_RIGHT_BUTTON );


	/* create the display structures that will not change:		*/
	whichObj=0;
	InitLists();
	whichObj=1;
	InitLists();
	whichObj=2;
	InitLists();
	

	/* setup the callback routines:					*/

	/* DisplayFunc -- redraw the window				*/
	/* ReshapeFunc -- handle the user resizing the window		*/
	/* KeyboardFunc -- handle a keyboard input			*/
	/* MouseFunc -- handle the mouse button going down or up	*/
	/* MotionFunc -- handle the mouse moving with a button down	*/
	/* PassiveMotionFunc -- handle the mouse moving with a button up*/
	/* VisibilityFunc -- handle a change in window visibility	*/
	/* EntryFunc	-- handle the cursor entering or leaving the window */
	/* SpecialFunc -- handle special keys on the keyboard		*/
	/* SpaceballMotionFunc -- handle spaceball translation		*/
	/* SpaceballRotateFunc -- handle spaceball rotation		*/
	/* SpaceballButtonFunc -- handle spaceball button hits		*/
	/* ButtonBoxFunc -- handle button box hits			*/
	/* DialsFunc -- handle dial rotations				*/
	/* TabletMotionFunc -- handle digitizing tablet motion		*/
	/* TabletButtonFunc -- handle digitizing tablet button hits	*/
	/* MenuStateFunc -- declare when a pop-up menu is in use	*/
	/* IdleFunc -- what to do when nothing else is going on		*/
	/* TimerFunc -- trigger something to happen every so often	*/

	glutSetWindow( GrWindow );
	glutDisplayFunc( Display );
	glutReshapeFunc( Resize );
	glutKeyboardFunc( Keyboard );
	glutMouseFunc( MouseButton );
	glutMotionFunc( MouseMotion );
	glutPassiveMotionFunc( NULL );
	glutVisibilityFunc( Visibility );
	glutEntryFunc( NULL );
	glutSpecialFunc( Keyboard2 );
	glutSpaceballMotionFunc( NULL );
	glutSpaceballRotateFunc( NULL );
	glutSpaceballButtonFunc( NULL );
	glutButtonBoxFunc( NULL );
	glutDialsFunc( NULL );
	glutTabletMotionFunc( NULL );
	glutTabletButtonFunc( NULL );
	glutMenuStateFunc( NULL );
	glutIdleFunc( Animate );
		/* change "NULL" to "Animate" if doing animation	*/
	glutTimerFunc( 0, NULL, 0 );
}




/**
 ** initialize the display lists that will not change:
 ** Note: glGenLists() is used to generate a unique integer identifier
 **	  for each display list, thus avoiding any possible collision
 **	  with an identifier assigned elsewhere 
 **/

void
InitLists( void )
{
	int num_verts;
	float x, y, z;
	int test;
	/* create the object:						*/
	FILE * datafile = fopen(model[whichObj], "r");
	if(datafile == NULL){
		fprintf(stderr, "Can't open datafile\n");;
		exit(1);
	}
	test = fscanf(datafile, "%d\n", &num_verts);
	if(test == 0){
		fprintf(stderr, "Couldn't get num_verts\n");
		exit(1);
	}
	fprintf(stderr, "Num Verts = %d\n", num_verts);
	ObjList[whichObj] = glGenLists( 1 );
	glNewList( ObjList[whichObj], GL_COMPILE );
	glBegin(GL_TRIANGLES);
	{
		int i;
		for(i=0; i<num_verts; i++){
		  float len=0;
			test = fscanf(datafile, "%f, %f, %f\n", &x, &y, &z);
			if(test == 0){
				fprintf(stderr, "Couldn't get num_verts\n");
				exit(1);
			}
			len=sqrt(x*x+y*y+z*z);
			glNormal3f(x/len,y/len,z/len);
			glVertex3f(x,y,z);
		}
	}
	glEnd();

	fclose(datafile);

	datafile = fopen(model[whichObj], "rb");

	glColor3f(1.0, 0.0, 0.0);
	glPointSize(10.0f);
/*
	glBegin(GL_POINTS);
	{
		int i;
		for(i=0; i<num_verts; i++){
			test = fscanf(datafile, "%f, %f, %f\n", &x, &y, &z);
			if(test == 0){
				fprintf(stderr, "Couldn't get num_verts\n");
				exit(1);
			}	
			glVertex3f(x,y,z);
		}
	}
	glEnd();
	*/	
	glEndList();

	fclose(datafile);

	/* create the axes:						*/

	AxesList = glGenLists( 1 );
	glNewList( AxesList, GL_COMPILE );
		glColor3f( AXES_COLOR );
		glLineWidth( AXES_WIDTH );
			Axes( AXES_SIZE );
		glLineWidth( 1. );
	glEndList();

}



/**
 ** callback to handle keyboard hits:
 ** Note: this callback only handles ketters, numbers, and punctuation.
 **       Special characters, like the arrow keys are handled in Keyboard2()
 **/

void
Keyboard( unsigned char c, int x, int y )
{
	if( Debug )
		fprintf( stderr, "Keyboard: '%c' (0x%0x)\n", c, c );

	switch( c )
	{
		case 'a':
		case 'A':
			AxesOnOff = ! AxesOnOff;
			break;

		case 'd':
		case 'D':
			Debug = ! Debug;
			break;

		case 'o':
		case 'O':
			Projection = ORTHO;
			break;

		case 'p':
		case 'P':
			Projection = PERSP;
			break;

		case 'q':
		case 'Q':
		case ESCAPE:
			Quit();		/* will not return here		*/

		case 'r':
		case 'R':
			TransformMode = ROTATE;
			break;	
        case '1':
          whichObj=0;
          break;
        case '2':
          whichObj=1;
          break;
        case '3':
          whichObj=2;
          break;
        case '4':
          whichObj=3;
          break;
        case 't':
          togglered=!togglered;
          break;
		case 's':
		case 'S':
			TransformMode = SCALE;
			break;
		      case 'm':
		      case 'M':
			{
			  wire=!wire;
			}
			break;
		default:
			fprintf( stderr, "Don't know what to do with keyboard hit: '%c' (0x%0x)\n", c, c );
	}

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** callback to handle special keyboard hits:
 **/

void
Keyboard2( int c, int x, int y )
{
	if( Debug )
		fprintf( stderr, "Keyboard: '%c' (0x%0x)\n", c, c );

	switch( c )
	{
		case GLUT_KEY_LEFT:
		case GLUT_KEY_RIGHT:
		case GLUT_KEY_DOWN:
		case GLUT_KEY_UP:
			fprintf( stderr, "Arrow key hit\n" );
			break;

		default:
			fprintf( stderr, "Don't know what to do with special keyboard hit: 0x%0x\n", c );
	}

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** called when the mouse button transitions down or up:
 **/

void
MouseButton
(
	int button,		/* GLUT_*_BUTTON			*/
	int state,		/* GLUT_UP or GLUT_DOWN			*/
	int x,			/* where mouse was when button was hit	*/
	int y			/* where mouse was when button was hit	*/
)
{
	int b;			/* LEFT, MIDDLE, or RIGHT		*/

	
	/* get the proper button bit mask:				*/

	switch( button )
	{
		case GLUT_LEFT_BUTTON:
			b = LEFT;		break;

		case GLUT_MIDDLE_BUTTON:
			b = MIDDLE;		break;

		case GLUT_RIGHT_BUTTON:
			b = RIGHT;		break;

		default:
			b = 0;
			fprintf( stderr, "Unknown mouse button: %d\n", button );
	}


	/* button down sets the bit, up clears the bit:			*/

	if( state == GLUT_DOWN )
	{
		Xmouse = x;
		Ymouse = y;
		ActiveButton |= b;		/* set the proper bit	*/
	}
	else
		ActiveButton &= ~b;		/* clear the proper bit	*/
}



/**
 ** called when the mouse moves while a button is down:
 **/

void
MouseMotion( int x, int y )
	/* x and y are mouse coords					*/
{
	int dx, dy;		/* change in mouse coordinates		*/



	dx = x - Xmouse;		/* change in mouse coords	*/
	dy = y - Ymouse;

	if( ActiveButton & LEFT )
	{
		if( TransformMode == ROTATE )
		{
			Xrot += ( ANGFACT*dy );
			Yrot += ( ANGFACT*dx );
		}
		else
		{
			Scale += SCLFACT * (float) ( dx - dy );
			if( Scale < MIN_SCALE )
				Scale = MIN_SCALE;	/* do not invert	*/
		}
	}

	if( ActiveButton & MIDDLE )
	{
		Scale += SCLFACT * (float) ( dx - dy );
		if( Scale < MIN_SCALE )
			Scale = MIN_SCALE;	/* do not invert	*/
	}

	Xmouse = x;			/* new current position		*/
	Ymouse = y;

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}




/**
 ** quit the program gracefully:
 **/

void
Quit( )
{
	/* gracefully close out the graphics:				*/

	glFinish();


	/* gracefully close the graphics window:			*/

	glutDestroyWindow( GrWindow );


	/* gracefully exit the program:					*/

	exit( 0 );
}



/**
 ** reset the transformations and the colors:
 **
 ** this only sets the global variables --
 ** the main loop is responsible for redrawing the scene
 **/

void
Reset( void )
{
	ActiveButton = 0;
	AxesOnOff = OFF;
	Projection = ORTHO;
	Red = 1.;	Green = 1.;	Blue = 1.;	/* white	*/
	Scale = .15;
	TransformMode = ROTATE;
	Xrot = 270.; Yrot = 0.;
	//Xrot = 55.; Yrot = 35.;

	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** called when user resizes the window:
 **/

void
Resize
(
	int width,		/* new value for window width		*/
	int height		/* new value for window height		*/
)
{
	glutSetWindow( GrWindow );
	glutPostRedisplay();
}



/**
 ** handle a change to the window's visibility:
 **/

void
Visibility
(
	int state		/* GLUT_VISIBLE or GLUT_NOT_VISIBLE	*/
)
{
	if( state == GLUT_VISIBLE )
	{
		glutSetWindow( GrWindow );
		glutPostRedisplay();
	}
	else
	{
		/* could optimize by keeping track of the fact		*/
		/* that the window is not visible and avoid		*/
		/* redrawing it later ...				*/
	}
}


/* the stroke characters 'X' 'Y' 'Z' :					*/

static float xx[] =
{
	0.f, 1.f, 0.f, 1.f
};

static float xy[] =
{
	-.5f, .5f, .5f, -.5f
};

static int xorder[] =
{
	1, 2, -3, 4
};


static float yx[] =
{
	0.f, 0.f, -.5f, .5f
};

static float yy[] =
{
	0.f, .6f, 1.f, 1.f
};

static int yorder[] =
{
	1, 2, 3, -2, 4
};


static float zx[] =
{
	1.f, 0.f, 1.f, 0.f, .25f, .75f
};

static float zy[] =
{
	.5f, .5f, -.5f, -.5f, 0.f, 0.f
};

static int zorder[] =
{
	1, 2, 3, 4, -5, 6
};


/* fraction of the length to use as height of the characters:		*/

#define LENFRAC		0.10


/* fraction of length to use as start location of the characters:	*/

#define BASEFRAC	1.10


/**
 **	Draw a set of 3D axes:
 **	(length is the axis length in world coordinates)
 **/

void
Axes( float length )
{
	int i, j;			/* counters			*/
	float fact;			/* character scale factor	*/
	float base;			/* character start location	*/


	glBegin( GL_LINE_STRIP );
		glVertex3f( length, 0., 0. );
		glVertex3f( 0., 0., 0. );
		glVertex3f( 0., length, 0. );
	glEnd();
	glBegin( GL_LINE_STRIP );
		glVertex3f( 0., 0., 0. );
		glVertex3f( 0., 0., length );
	glEnd();

	fact = LENFRAC * length;
	base = BASEFRAC * length;

	glBegin( GL_LINE_STRIP );
		for( i = 0; i < 4; i++ )
		{
			j = xorder[i];
			if( j < 0 )
			{
				
				glEnd();
				glBegin( GL_LINE_STRIP );
				j = -j;
			}
			j--;
			glVertex3f( base + fact*xx[j], fact*xy[j], 0.0 );
		}
	glEnd();

	glBegin( GL_LINE_STRIP );
		for( i = 0; i < 5; i++ )
		{
			j = yorder[i];
			if( j < 0 )
			{
				
				glEnd();
				glBegin( GL_LINE_STRIP );
				j = -j;
			}
			j--;
			glVertex3f( fact*yx[j], base + fact*yy[j], 0.0 );
		}
	glEnd();

	glBegin( GL_LINE_STRIP );
		for( i = 0; i < 6; i++ )
		{
			j = zorder[i];
			if( j < 0 )
			{
				
				glEnd();
				glBegin( GL_LINE_STRIP );
				j = -j;
			}
			j--;
			glVertex3f( 0.0, fact*zy[j], base + fact*zx[j] );
		}
	glEnd();

}
