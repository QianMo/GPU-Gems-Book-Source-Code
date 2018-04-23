/* 
  Copyright (C) 2005, William Donnelly (wdonnelly@uwaterloo.ca)
                  and Stefanus Du Toit (sjdutoit@cgl.uwaterloo.ca)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef WIN32
#define GL_GLEXT_PROTOTYPES
#define GL_GLEXT_LEGACY
#endif

#define _USE_MATH_DEFINES
#define NOMINMAX
#include <limits>
#include <cctype>
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <sh/sh.hpp>
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include "Camera.hpp"
#include "distance.hpp"
#include "gprintf.h"
#include "timer.h"

#ifdef WIN32
PFNGLMULTITEXCOORD3FARBPROC glMultiTexCoord3fARB = 0;
#endif

int iterations = 16;
int dmap_depth = 16;
bool filter = true;

bool show_help = false;
bool changed = false;
bool move_light = false;

float theta = 45, phi=-45;
float light_theta = 45, light_phi = 45;

double elapsed=0.0, fps=0.0;
int frames=0;

int screen_width=512, screen_height=512;

using namespace SH;

ShMatrix4x4f ModelViewProjection, ModelViewInverse;
ShPoint3f lightPos;
Camera camera;
ShProgram vsh, fsh;
ShTexture2D<ShColor3f> colors;
ShTexture2D<ShNormal3f> normals;
ShTexture3D<ShAttrib1f> distmap;
ShAttrib1f bumpdepth = 1.0;

// Glut data
int buttons[5] = {GLUT_UP, GLUT_UP, GLUT_UP, GLUT_UP, GLUT_UP};
int cur_x, cur_y;

void initExtensions()
{
#ifdef WIN32
  glMultiTexCoord3fARB = (PFNGLMULTITEXCOORD3FARBPROC)wglGetProcAddress("glMultiTexCoord3fARB");
#endif
}

void initTextures(const char* height_name,
                  const char* color_name,
                  const char* normal_name)
{

  /* Set up color map */
  ShImage colormap;
  colormap.loadPng(color_name);

  GLuint ctex;
  glGenTextures(1, &ctex);
  glBindTexture(GL_TEXTURE_2D, ctex);
  gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB,
                    colormap.width(), colormap.height(),
                    GL_RGB,
                    GL_FLOAT, colormap.data());

  { std::ostringstream os; os << ctex; colors.meta("opengl:preset", os.str()); }


  /* Set up normal map */
  ShImage normalmap;
  normalmap.loadPng(normal_name);
  GLuint ntex;
  glGenTextures(1, &ntex);
  glBindTexture(GL_TEXTURE_2D, ntex);
  gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB,
                    normalmap.width(), normalmap.height(),
                    GL_RGB,
                    GL_FLOAT, normalmap.data());

  { std::ostringstream os; os << ntex; normals.meta("opengl:preset", os.str()); }
  
  /* Load in height map and generate distance map */
  ShImage heightmap;
  heightmap.loadPng(height_name);

  ShImage3D dmap_img = init_distance_map(heightmap, dmap_depth);
  
  distmap.memory(dmap_img.memory());
  distmap.size(dmap_img.width(), dmap_img.height(), dmap_img.depth());
}

void initShaders()
{
  vsh = SH_BEGIN_VERTEX_PROGRAM {
    ShInputPosition4f SH_DECL(ipos);
    ShOutputPosition4f SH_DECL(opos);

    ShInOutTexCoord3f SH_DECL(tc);
    ShInputNormal3f SH_DECL(normal);
    ShInputNormal3f SH_DECL(tangent);
    ShInputNormal3f SH_DECL(binormal);
    ShOutputVector3f SH_DECL(tan_eye);
    ShOutputVector3f SH_DECL(tan_light);

    // Transform position my modelview projection matrix
    opos = ModelViewProjection | ipos;

    // Set 3rd component of texture co-ordinate, it will be passed through automatically
    tc(2) = 1.0;

    // Map the eye vector into tangent space
    ShVector3f SH_DECL(eye);
    eye = (ModelViewInverse | ShPoint3f(0,0,0)) - ipos(0,1,2);
    tan_eye = -ShVector3f(tangent | eye, binormal | eye,  1.0/bumpdepth * (normal | eye));

    // Map the light vector into tangent space
    ShVector3f SH_DECL(lightVec);
    lightVec = lightPos - ipos(0,1,2);
    tan_light = ShVector3f(tangent | lightVec, binormal | lightVec, normal | lightVec);

  } SH_END;
  
  fsh = SH_BEGIN_FRAGMENT_PROGRAM {

    ShInputPosition4f SH_DECL(pos);
    ShInputTexCoord3f SH_DECL(itc);
    ShInputVector3f SH_DECL(tan_eye);
    ShInputVector3f SH_DECL(tan_light);
 
    // Compute offset vector by normalizing and multiplying with normalization factor
    ShVector3f offset = normalize(tan_eye);
    offset(0,1) *= ((double)dmap_depth)/distmap.width();

    // March a ray 
    ShTexCoord3f tc = itc;    
    for (int i = 0; i < iterations; i++) {
      ShAttrib1f distance = distmap(tc);
      tc = mad(distance, offset, tc);
    }

    // Compute derivatives of unperturbed texture coordinates
    ShAttrib2f ddx = dx(itc(0,1));
    ShAttrib2f ddy = dy(itc(0,1));
    
    // Do tangent-space normal mapping    
    tan_light = normalize(tan_light);
    ShNormal3f tan_normal;
    if(filter) {
      tan_normal = 2.0 * normals(tc(0,1), ddx, ddy) - 1.0;
    } else {
      tan_normal = 2.0 * normals(tc(0,1)) - 1.0;
    }

    // Apply lighting to base color
    ShColor1f diffuse = tan_normal | tan_light;

    ShOutputColor3f c;
    if(filter) {
      c = colors(tc(0,1), ddx, ddy) * diffuse;
    } else {
      c = colors(tc(0,1)) * diffuse;
    }

  } SH_END;
}

void display()
{
  mtimer_t start = mtimer_t::now();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBegin(GL_QUADS); {
    glNormal3f(0.0, 1.0, 0.0);

    // tangent
    glMultiTexCoord3fARB(GL_TEXTURE0 + 1, 1.0, 0.0, 0.0);
    // binormal
    glMultiTexCoord3fARB(GL_TEXTURE0 + 2, 0.0, 0.0, 1.0);

    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, 0.0, -1.0);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 0.0, 1.0);

    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 0.0, 1.0);

    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, 0.0, -1.0);
  } glEnd();

  gprintf(10, screen_height-12, true, "Bump Depth : %3.2f", bumpdepth.getValue(0));
  gprintf(10, screen_height-24, true, "Iterations : %i", iterations);
  gprintf(10, screen_height-36, true, "FPS        : %3.2f", fps);
  gprintf(10, screen_height-48, true, "Filtering  : %s", filter?"on":"off");

  if (show_help) {

    gprintf(10, 70, true, "  L: Toggle light navigator");
    gprintf(10, 58, true, "  F: Toggle special filtering");
    gprintf(10, 46, true, "i/I: Increase/decrease number of iterations");
    gprintf(10, 34, true, "-/+: Increase/decrease displacement");
    gprintf(10, 10, true, "  H: Toggle this help");
  } else {
    gprintf(10, 10, true, "Press H for Help");
  }

  glutSwapBuffers();

  mtimer_t end = mtimer_t::now();
  elapsed += (end - start).value() / 1000.0; 
  frames++;

  if(elapsed >= 1.0f) {
    fps = frames / elapsed;
    frames = 0;
    elapsed = 0.0f;
  }
}

void setupView()
{
  camera.resetRotation();

  camera.rotate(phi, 0, 1, 0);
  camera.rotate(theta, 1, 0, 0);

  lightPos = ShPoint3f(5*cos((M_PI/180)*light_phi)*sin((M_PI/180)*light_theta), 
                       5*cos((M_PI/180)*light_theta),
                       5*sin((M_PI/180)*light_phi)*sin((M_PI/180)*light_theta));

  ModelViewInverse = camera.shInverseModelView();
  ModelViewProjection = camera.shModelViewProjection(ShMatrix4x4f());
}

void reshape(int width, int height)
{
  screen_width = width;
  screen_height = height;
  glViewport(0, 0, width, height);
  camera.glProjection(float(width)/height);
  setupView();
}

void motion(int x, int y)
{
  const double factor = 20.0;
  
  if (buttons[GLUT_LEFT_BUTTON] == GLUT_DOWN) {
        
    if(move_light) {

      light_theta += (y - cur_y)/5;
      light_phi += (x - cur_x)/5;
      if(light_theta > 90) light_theta = 90;
      if(light_theta < 10) light_theta = 10;

    } else {

      theta += (y - cur_y)/5;
      phi += (x - cur_x)/5;
      if(theta > 90) theta = 90;
      if(theta < 10) theta = 10;

    }

    changed = true;
  }
  if (buttons[GLUT_MIDDLE_BUTTON] == GLUT_DOWN) {
    camera.move(0, 0, (y - cur_y)/factor);
    changed = true;
  }
  if (buttons[GLUT_RIGHT_BUTTON] == GLUT_DOWN) {
    camera.move((x - cur_x)/factor, (cur_y - y)/factor, 0);
    changed = true;
  }

  cur_x = x;
  cur_y = y;
}

void mouse(int button, int state, int x, int y)
{
  buttons[button] = state;
  cur_x = x;
  cur_y = y;
}

void keyboard(unsigned char key, int x, int y) {
  if (key == '-') {
    bumpdepth -= 0.1;
    if (bumpdepth.getValue(0) < 0.0) bumpdepth = 0.0;
  }
  if (key == '+') {
    bumpdepth += 0.1;
  }
  if (key == 'I') {
    iterations += 4;
    initShaders();
    shBind(vsh);
    shBind(fsh);
  }
  if (key == 'i') {
    iterations -= 4;
    if (iterations < 0) iterations = 0;
    initShaders();
    shBind(vsh);
    shBind(fsh);
  }
  if (tolower(key) == 'h') {
    show_help = !show_help;
  }
  if(tolower(key) == 'f') {
    filter = !filter;
    initShaders();
    shBind(vsh);
    shBind(fsh);
  }
  if(tolower(key) == 'l') {
    move_light = !move_light;
  }
}

void idle() {  

  if (changed) {
    setupView();
    changed = false;
  }

  glutPostRedisplay();
}

void usage(const char* progname)
{
  std::cerr << "Usage: " << progname << " heightimg colorimg normalimg" << std::endl;
}

int main(int argc, char** argv)
{

  if (argc != 4) {
    usage(argv[0]);
    return 1;
  }

  try {

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(512, 512);
    glutCreateWindow("Per-Pixel Displacement Demo");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);  
    glutIdleFunc(idle);
    shSetBackend("arb");

    initExtensions();
    initTextures(argv[1], argv[2], argv[3]);
    initShaders();
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_ARB);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    setupView();

    // Place the camera at its initial position
    camera.move(0.0, 0.0, -5.0);
    camera.rotate(-45, 0, 1, 0);
    camera.rotate(45, 1, 0, 0);

    // Set up the light position
    lightPos = ShPoint3f(5.0, 5.0, 5.0);
    
    initShaders();

    shBind(vsh);
    shBind(fsh);
    
    glutMainLoop();

  } catch (const ShException& e) {
    std::cerr << "Caught Sh Exception: ";
    std::cerr << e.message() << std::endl;
    throw;
  }
}
