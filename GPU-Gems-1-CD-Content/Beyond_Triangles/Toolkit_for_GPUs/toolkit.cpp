#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "GL/glew.h"
#include "GL/glut.h"
#include "GL/wglew.h"

#include "timer.h"
#include "fragprog.h"
#include "window.h"

#include "defines.h"

void Keyboard( unsigned char key, int x, int y );
void Mouse( int button, int state, int x, int y );
void Motion( int x, int y );
void Idle( void );
void Init( void );
void Display( void );
void DoBinarySearch( void );

Window floatbuffer;
Window colorbuffer;

FragProg binarysearchfp;
GLuint sortedtextureid;
float sortedtexture[(int)(W*H)*4];

FragProg bitonicsortfp;
GLuint randomtextureid;
float randomtexture[(int)(W*H)*4];

FragProg reducefp;

float data[(int)(W*H)*4];

Timer frametime;


void CheckErrors() {
	GLenum error = glGetError();
	if( error != GL_NO_ERROR ) {
		fprintf( stderr, "\nGL Error: %s\n", gluErrorString( error ) );
		assert(0);
	}
}

void Init( void ){
  int i;

  glutInitWindowSize( W, H );
  glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
  glutCreateWindow( "NVToolkit" );

  int err = glewInit();
  if (GLEW_OK != err){
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
  }

  glutDisplayFunc( Display );
  glutMotionFunc( Motion );
  glutKeyboardFunc( Keyboard );
  glutMouseFunc( Mouse );
  glutIdleFunc( Idle );

  colorbuffer.SetContext();

  glDrawBuffer(GL_FRONT);
  glClear(GL_COLOR_BUFFER_BIT);

  floatbuffer.CreateContext(W, H, "floatbuffer");

  binarysearchfp.Load("binarysearch.fp");
  glGenTextures(1, &sortedtextureid);
  for(i=0; i<W*H; i++){
    sortedtexture[4*i+0] = i;
    sortedtexture[4*i+1] = i;
    sortedtexture[4*i+2] = i;
    sortedtexture[4*i+3] = i;
  }
  glActiveTextureARB( GL_TEXTURE1_ARB );
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, sortedtextureid);
	glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA32_NV, W, H, 0, GL_RGBA, GL_FLOAT, sortedtexture);

  bitonicsortfp.Load("bitonicsort.fp");
  glGenTextures(1, &randomtextureid);
  for(i=0; i<W*H; i++){
    randomtexture[4*i+0] = (rand() * (W*H-1)) / RAND_MAX;
    randomtexture[4*i+1] = randomtexture[4*i+0];
    randomtexture[4*i+2] = randomtexture[4*i+0];
    randomtexture[4*i+3] = randomtexture[4*i+0];
  }
  glActiveTextureARB( GL_TEXTURE2_ARB );
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, randomtextureid);
  glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA32_NV, W, H, 0, GL_RGBA, GL_FLOAT, randomtexture);

  reducefp.Load("reduce.fp");

  printf ("Press 1 for Binary Search\n");
  printf ("Press 2 for Bitonic Sort\n");
  printf ("Press 3 for Reduce\n");
  printf ("Press q to quit\n");

  frametime.init();
}

int main(int argc, char **argv)
{
  glutInit( &argc, argv );  
  Init();
  glutMainLoop();
  return 0;
}

void Display( void ){
  char title[128];
  sprintf(title, "%3.3f fps NVToolkit", frametime.fps());
  glutSetWindowTitle(title);
  frametime.reset();
}


void DoBinarySearch( void ){
  floatbuffer.MakeCurrent();

  binarysearchfp.Bind();
  binarysearchfp.SetParameter1f("stride", N/2);
  binarysearchfp.SetParameter1f("pbufinfo", W);
  binarysearchfp.SetParameter1f("sortbufinfo", W);
  glActiveTextureARB( GL_TEXTURE1_ARB );
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, sortedtextureid);

  binarysearchfp.Run(0,0,W,H);
  glFinish();
  char title[128];
  if (frametime.elapsed_ms() < 1000)
    sprintf(title, "%3.3f ms SEARCH NVToolkit", frametime.elapsed_ms());
  else
    sprintf(title, "%3.3f seconds SEARCH NVToolkit", frametime.elapsed_ms()/1000);

  glutSetWindowTitle(title);
  frametime.reset();

#if 0
  glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT, data);
  for(int i=0; i<W*H-1; i++){
    //fprintf(stderr, "%i : %f %f %f %f\n", i, data[4*i+0], data[4*i+1], data[4*i+2], data[4*i+3]);
    assert (data[4*i] <= data[4*(i+1)]);
  }
#endif

  colorbuffer.MakeCurrent();
}


void DoReduce( void ){
  float rval[4];

  floatbuffer.MakeCurrent();

  reducefp.Bind();
  glActiveTextureARB( GL_TEXTURE1_ARB );
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, randomtextureid);

  CheckErrors();

  for (unsigned int i=W/2; i>0; i = i/2) {
	  // Note that we have to subtract -1 from
	  // the texture coordinates since we 
	  // specify the texture coordinates at the 
	  // edge of the pixel but OpenGL samples at 
	  // the pixel centers.
	  reducefp.Run1( 0,  0,     i,     i,
		            -1, -1, i*2-1, i*2-1);
	  
	  glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0,
			              0, 0, 0, 0, i, i);
  }	  

  glFinish();

  char title[128];
  if (frametime.elapsed_ms() < 1000)
    sprintf(title, "%3.3f ms REDUCE NVToolkit", frametime.elapsed_ms());
  else
    sprintf(title, "%3.3f seconds REDUCE NVToolkit", frametime.elapsed_ms()/1000);

  glutSetWindowTitle(title);
  frametime.reset();

  CheckErrors();

  colorbuffer.MakeCurrent();
}

void DoBitonicSort( void ){
  int stepno, offset, stage;
  int i,j;

  floatbuffer.MakeCurrent();

  CheckErrors();

  for(i=0; i<LOGN; i++){
    stepno = 1 << (i+1);//(int)pow(2,i+1);
    for(j=i; j>=0; j--){
      offset = 1 << j;//(int)pow(2,j);
      stage = 2*offset;
      //fprintf(stderr, ".");
      bitonicsortfp.Bind();
      bitonicsortfp.SetParameter3f("pbufinfo", W, 1.0/W, 1.0);
      bitonicsortfp.SetParameter1f("offset", offset);
      bitonicsortfp.SetParameter1f("stage", stage);
      bitonicsortfp.SetParameter1f("stepno", stepno);
      glActiveTextureARB( GL_TEXTURE2_ARB );
      glBindTexture(GL_TEXTURE_RECTANGLE_NV, randomtextureid);

      bitonicsortfp.Run(0,0,W,H);

      glActiveTextureARB( GL_TEXTURE2_ARB );
      glBindTexture(GL_TEXTURE_RECTANGLE_NV, randomtextureid);
	    glCopyTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA32_NV, 0, 0, W, H, 0);
    }
  }

  glFinish();

  char title[128];
  if (frametime.elapsed_ms() < 1000)
    sprintf(title, "%3.3f ms SORT NVToolkit", frametime.elapsed_ms());
  else
    sprintf(title, "%3.3f seconds SORT NVToolkit", frametime.elapsed_ms()/1000);
  glutSetWindowTitle(title);
  frametime.reset();

#if 0
  glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT, data);
  for(i=0; i<W*H-1; i++){
    //fprintf(stderr, "%i : %f %f %f %f\n", i, data[4*i+0], data[4*i+1], data[4*i+2], data[4*i+3]);
    assert (data[4*i] <= data[4*(i+1)]);
  }
#endif

  CheckErrors();

  colorbuffer.MakeCurrent();
}

void Idle(void){
  glutPostRedisplay();
}

void Keyboard( unsigned char key, int x, int y ){
  switch (key) {
  case '0':
    glutDisplayFunc( Display );
    break;
  case '1':
    glutDisplayFunc( DoBinarySearch );
    break;
  case '2':
    glutDisplayFunc( DoBitonicSort );
    break;
  case '3':
    glutDisplayFunc( DoReduce );
    break;
  case 'Q':
  case 'q':
  case 27:
    exit(0);
    break;
  }
}

void Mouse( int button, int state, int x, int y ){
}

void Motion( int x, int y ){
}

