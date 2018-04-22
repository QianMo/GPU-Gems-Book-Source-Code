#ifndef _FRAGPROG_H
#define _FRAGPROG_H

const int maxparams = 8;
const int channels = 4;
const int pnamelen = 16;

class FragProg{
public:
  FragProg(){}
  ~FragProg(){}

  void Load(const char* progname, char* prog){
    glGenProgramsNV( 1, &prog_id );
    glLoadProgramNV( GL_FRAGMENT_PROGRAM_NV, prog_id, strlen(prog), (GLubyte*)prog );
    GLenum error = glGetError();
	  if( error != GL_NO_ERROR )
		  fprintf( stderr, "ERROR\n%s\n", gluErrorString( error ) );
    else
      printf("Loaded program [%s] (id %i) successfully\n", progname, prog_id);
  }

  void Load(char* filename){
    glGenProgramsNV( 1, &prog_id );

    char buf[512];
    char prog[16384*16];
    char *s = prog;
    
    FILE *fp = fopen(filename, "r");

    if (!fp)
    {
        fprintf(stderr, "Couldn't open file [%s].\n", filename);
        exit(1);
    }

    while (!feof(fp))
    {
        if (!fgets(buf, 511, fp))
        {
            break;
        }

        s += sprintf(s, "%s", buf);
    }

    fclose(fp);
    glLoadProgramNV( GL_FRAGMENT_PROGRAM_NV, prog_id, strlen(prog), (GLubyte*)prog );
    GLenum error = glGetError();
	  if( error != GL_NO_ERROR )
		  fprintf( stderr, "ERROR\n%s\n", gluErrorString( error ) );
    else
      printf("Loaded program [%s] (id %i) successfully\n", filename, prog_id);
  }

  void Bind(){
    glEnable( GL_FRAGMENT_PROGRAM_NV );  
    glBindProgramNV( GL_FRAGMENT_PROGRAM_NV, prog_id );
    GLenum error = glGetError();
    if( error != GL_NO_ERROR )
		  fprintf( stderr, "ERROR - Bind()\n%s\n", gluErrorString( error ) );
  }

  void Release(){
    glDisable( GL_FRAGMENT_PROGRAM_NV );  
    GLenum error = glGetError();
    if( error != GL_NO_ERROR )
		  fprintf( stderr, "ERROR - Release()\n%s\n", gluErrorString( error ) );

  }

  void SetParameter1f(const char* pname, float x){
	  glProgramNamedParameter4fNV(prog_id, strlen(pname), (GLubyte*)(pname), x, 0, 0, 0);
  }

  void SetParameter2f(const char* pname, float x, float y){
	  glProgramNamedParameter4fNV(prog_id, strlen(pname), (GLubyte*)(pname), x, y, 0, 0);
  }

  void SetParameter3f(const char* pname, float x, float y, float z){
	  glProgramNamedParameter4fNV(prog_id, strlen(pname), (GLubyte*)(pname), x, y, z, 0);
  }

  void SetParameter4f(const char* pname, float x, float y, float z, float w){
	  glProgramNamedParameter4fNV(prog_id, strlen(pname), (GLubyte*)(pname), x, y, z, w);
  }

  void Run( int minx, int miny, int maxx, int maxy ){
    glViewport(0,0,maxx, maxy);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, maxx, 0.0, maxy);
    glMatrixMode( GL_MODELVIEW );
	  glLoadIdentity();
    glBegin(GL_QUADS);
    glVertex2f((float) 0, (float)0);
    glVertex2f((float) maxx, (float) 0);
    glVertex2f((float) maxx, (float) maxy);
    glVertex2f((float) 0, (float) maxy);
    glEnd();
    glFlush();
  }

  void Run1( int minx, int miny, int maxx, int maxy,
		 	 int mins, int mint, int maxs, int maxt){
    glViewport(0,0,maxx, maxy);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, maxx, 0.0, maxy);
    glMatrixMode( GL_MODELVIEW );
	  glLoadIdentity();
    glBegin(GL_QUADS);
	glTexCoord2f ((float) mins, (float) mint);
    glVertex2f   ((float) minx, (float) miny);
	glTexCoord2f ((float) maxs, (float) mint);
    glVertex2f   ((float) maxx, (float) miny);
	glTexCoord2f ((float) maxs, (float) maxt);
    glVertex2f   ((float) maxx, (float) maxy);
	glTexCoord2f ((float) mins, (float) maxt);
    glVertex2f   ((float) minx, (float) maxy);
    glEnd();
    glFlush();
  }

//private:
  GLuint prog_id;
};

#endif /* _FRAGPROG_H */