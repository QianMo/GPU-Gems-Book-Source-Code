#include <gl/glut.h>
#include <gl/glext.h>
#include <Cg/cg.h>
#include <Cg/cggl.h>
#include <stdio.h>
#include <algorithm>
#include <string>

#include <shared/pug/pug.h>

#define GET_GLERROR(ret)                                          \
    {                                                             \
        GLenum err = glGetError();                                \
        if (err != GL_NO_ERROR) {                                 \
            fprintf(stderr, "[%s line %d] GL Error: %s\n",        \
                    __FILE__, __LINE__, gluErrorString(err));     \
            fflush(stderr);                                       \
            exit(1); \
            return (ret);                                         \
         }                                                        \
    }

std::string path = "../shared/pug/";

CGcontext g_cgContext; 
CGprofile g_profile;

PUGBuffer  *g_rdBuffer = NULL;
PUGProgram *g_rdProgram = NULL;

PUGTarget  g_currentTarget = PUG_BACK;
PUGTarget  g_currentSource = PUG_FRONT;

CGprogram   g_displayProgram = NULL;
CGparameter g_displayTexParam = NULL;

int g_width = 256;
int g_height = 256;

float g_rK = 0.052f;
float g_rF = 0.012f;
float g_rDiffusionU = 0.0004f;
float g_rDiffusionV = 0.0002f;

void cgErrorCallback()
{
    CGerror lastError = cgGetError();

    if(lastError)
    {
        printf("%s\n\n", cgGetErrorString(lastError));
        printf("%s\n", cgGetLastListing(g_cgContext));
        printf("Cg error, exiting...\n");
        exit(1);
    }
}

void reset_rd()
{
    float *pFloatData = new float[4 * g_width * g_height];
    int k = 0;
    int i, j;

    for (i = 0; i < g_width * g_height; ++i)
    {
        pFloatData[k++] = 1.0;
        pFloatData[k++] = 0.0;
        pFloatData[k++] = 0.0;
        pFloatData[k++] = 0.0;
    }

    for (i = (0.48f)*g_height; i < (0.52f)*g_height; ++i)
    {
        for (j = (0.48f)*g_width; j < (0.52f)*g_width; ++j)
        {
            pFloatData[4 * (i * g_width + j)    ] = .5;
            pFloatData[4 * (i * g_width + j) + 1] = .25;
            pFloatData[4 * (i * g_width + j) + 2] = 0;
            pFloatData[4 * (i * g_width + j) + 3] = 0;
        }
    }

    // Now perturb the entire grid. Bound the values by [0,1]
    for (k = 0; k < g_width * g_height * 4; ++k)
    {
        if ( pFloatData[k] < 1.0 )
        {
            float rRand = .02f*(float)rand() / RAND_MAX - .01f;
            pFloatData[k] += rRand * pFloatData[k];
        }
    }

    pugInitBuffer(g_rdBuffer, pFloatData, g_currentSource);
    delete [] pFloatData;
}

int update_rd()
{
    pugBindStream(g_rdProgram, "concentration", g_rdBuffer, g_currentSource);

    PUGRect range(0, g_width, 0, g_height);
    pugRunProgram(g_rdProgram, g_rdBuffer, range, g_currentTarget);

    std::swap(g_currentTarget, g_currentSource);

    GET_GLERROR(0);

    return 0;
}

void init_opengl()
{
    glDisable(GL_DEPTH_TEST);

    // Create cgContext.
    g_cgContext = cgCreateContext();
    cgSetErrorCallback(cgErrorCallback);

    // Setup display program
    g_profile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    cgGLSetOptimalOptions(g_profile);

    const char *args[2];
    args[0] = new char[strlen("-I../shared/pug")+1];
    strcpy(const_cast<char*>(args[0]), "-I../shared/pug");
    args[1] = 0;
    
    g_displayProgram  = cgCreateProgramFromFile(g_cgContext, CG_SOURCE, "rd.cg", 
                                                g_profile, "passthrough", args);
    cgCompileProgram(g_displayProgram);
    cgGLLoadProgram(g_displayProgram);
    g_displayTexParam = cgGetNamedParameter(g_displayProgram, "tex");

    // initialize gpu framework, create a buffer and the r-d program.
    pugInit(path.c_str(), g_cgContext, false);
    g_rdBuffer = pugAllocateBuffer(g_width, g_height, PUG_READWRITE, 4, true);

    g_rdProgram = pugLoadProgram("rd.cg", "rd");   

    // initialize the parameters of the simulation
    pugBindFloat(g_rdProgram, "windowDims", g_width, g_height, 
                 -g_rK - g_rF + (1 - 655.36f * g_rDiffusionV), 0);
    pugBindFloat(g_rdProgram, "rdParams", 655.36f * g_rDiffusionU, 
                 655.36f * g_rDiffusionV, g_rK, g_rF);
   
    reset_rd();              // initialize the r-d input 
    pugMakeWindowCurrent();  // switch back to the graphics window
}

void display()
{
    update_rd();            // update the r-d simulation
    pugMakeWindowCurrent(); // switch back to the graphics window

    glClear(GL_COLOR_BUFFER_BIT);

    pugBindTexture(g_rdBuffer, g_currentSource);
    cgGLBindProgram(g_displayProgram);
    cgGLSetTextureParameter(g_displayTexParam, g_rdBuffer->texHandle);
    cgGLEnableTextureParameter(g_displayTexParam);
    cgGLEnableProfile(g_profile);

    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);              glVertex2f(-1, -1);
        glTexCoord2f(g_width, 0);        glVertex2f( 1, -1);
        glTexCoord2f(g_width, g_height); glVertex2f( 1,  1);
        glTexCoord2f(0, g_height);       glVertex2f(-1,  1);
    glEnd();

    cgGLDisableTextureParameter(g_displayTexParam);
    cgGLDisableProfile(g_profile);

    glutSwapBuffers();
}

void idle()
{
    glutPostRedisplay();
}

void key(unsigned char k, int x, int y)
{
    switch(k)
    {
    case 27:
    case 'q':
        exit(0);
    case 'r':
        reset_rd();
        break;
    }
    
	glutPostRedisplay();
}

void resize(int w, int h)
{
    if (h == 0) h = 1;
    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    gluOrtho2D(-1, 1, -1, 1);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(512, 512);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutCreateWindow("gpgpu_reactiondiffusion");

	init_opengl();

	glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(key);
    glutReshapeFunc(resize);

	glutMainLoop();
	return 0;
}