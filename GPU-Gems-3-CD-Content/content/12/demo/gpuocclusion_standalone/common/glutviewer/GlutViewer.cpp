/*! \file GlutViewer.cpp
 *  \author Jared Hoberock, Yuntao Jia
 *  \brief Implementation of GlutViewer class.
 */

#include <string.h>
#include "GlutViewer.h"
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <vector>

bool GlutViewer::instanceFlag = false;
GlutViewer* GlutViewer::viewer= NULL;

/* singleton access */
void GlutViewer
  ::main(int argc, char **argv, const char *title, GlutViewer* pViewer)
{
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100,100);
  glutInitWindowSize(pViewer->width(),pViewer->height());
  glutCreateWindow(title);

  glutReshapeFunc(reshapeFunc);
  glutDisplayFunc(displayFunc);  
  glutKeyboardFunc(keyFunc);
  glutMouseFunc(mouseFunc);
  glutMotionFunc(motionFunc);

  
  viewer = pViewer;
  viewer->init();
  instanceFlag = true;
  
  glutMainLoop();
} // end GlutViewer::main()

GlutViewer* GlutViewer
  ::getInstance(void)
{
  if(!instanceFlag)
  {
    viewer = new GlutViewer();
    viewer->init();
    instanceFlag = true;
    return viewer;
  } // end if
  else
  {
    return viewer;
  } // end else
} // end GlutViewer::getInstance()

void GlutViewer
  ::setDimensions(const int w, const int h)
{
  mWidth = w;
  mHeight = h;
  mBallController.ClientAreaResize(0,0,w,h);
} // endG GlutViewer::setDimensions()

void GlutViewer
  ::resizeGL(int w, int h)
{
  setDimensions(w,h);

  // set the viewport
  glViewport(0, 0, width(), height());
} // end GlutViewer::resizeGL()

GlutViewer
  ::GlutViewer(void)
    :mBallController(0.8f,Quaternion(DegToRad(45.0f),float3(0,1,0))*Quaternion(DegToRad(-30.0f),float3(1,0,0)))
{
  // clear mouse state
  mMouseButton = 0;
  mMouseButtonState = 0;
  m_bMLDown = m_bMMDown = m_bMRDown = false;

  // message display
  m_nMsgShiftX = 0;
  m_nMsgShiftY = 0;
  
  // help
  m_bShowHelp	= false;
  m_nHelpWinPos[0] = 300;
  m_nHelpWinPos[1] = 400;
  m_nHelpWinSize[0] = 600;
  m_nHelpWinSize[1] = 400;
  
  // camera
  m_fCameraX = 0;
  m_fCameraY = 0;
  m_fCameraZ = 50;

  // animation
  mIsAnimating = false;

  // set dimensions
  setDimensions(800,600);
} // end GlutViewer::GlutViewer()

GlutViewer
  ::~GlutViewer(void)
{
  instanceFlag = false;
} // end GlutViewer::~GlutViewer()

/* static call back function for glut */
void GlutViewer
  ::displayFunc(void)
{
  GlutViewer* pViewer = GlutViewer::getInstance();
  pViewer->render();
} // end GlutViewer::render()

void GlutViewer
  ::render(void)
{
  beginDraw();
  draw();
  endDraw();
} // end GlutViewer::render()

void GlutViewer
  ::animate(void)
{
  ;
} // end GlutViewer::animate()

void GlutViewer
  ::idleFunc(void)
{
	GlutViewer* pViewer = GlutViewer::getInstance();
	pViewer->animate();
  pViewer->render();
} // end GlutViewer::idleFunc()

void GlutViewer
  ::mouseFunc(int button, int state, int x, int y)
{
  GlutViewer::getInstance()->mouseEvent(button, state, x, y);
} // end GlutViewer::mouseFunc()

void GlutViewer
  ::motionFunc(int x, int y)
{
  GlutViewer::getInstance()->motionEvent(x, y);
} // end GlutViewer::motionFunc()

void GlutViewer
  ::reshapeFunc(int w, int h)
{
  GlutViewer::getInstance()->resizeGL(w,h);
} // end GlutViewer::reshapeFunc()

void GlutViewer
  ::keyFunc(unsigned char key, int x, int y)
{
  // convert key to uppercase
  unsigned int k = toupper(key);

  // get modifiers
  unsigned int modifiers = glutGetModifiers();

  // convert to Qt-compatible codes
  unsigned int m = 0;
  if(modifiers & GLUT_ACTIVE_SHIFT)
  {
    m |= 0x02000000;
  } // end if
  if(modifiers & GLUT_ACTIVE_CTRL)
  {
    m |= 0x04000000;
  } // end if
  if(modifiers & GLUT_ACTIVE_ALT)
  {
    m |= 0x08000000;
  } // end if

  // create a KeyEvent
  KeyEvent e(k,m);

  GlutViewer::getInstance()->keyPressEvent(&e);
} // end GlutViewer::keyFunc()

void GlutViewer
  ::renderHelp(void)
{
  GlutViewer::getInstance()->drawHelp();
} // end GlutViewer::renderHelp()

void GlutViewer
  ::keyFuncHelp(unsigned char key, int x, int y)
{
  GlutViewer::getInstance()->keyPressEventHelp(key, x, y);
} // end GlutViewer::keyFuncHelp()

/* interface for children classes */
void GlutViewer
  ::init(void)
{
  // fps
  m_nDrawTimes = 0;
  m_nPreTick = glutGet(GLUT_ELAPSED_TIME);
  m_nTick = m_nPreTick;
  m_fElapsedTime = 0.0f;
  m_fTotalTime = 0.0f;
  drawFPS();
  
  // message
  m_fMsgLife = 0.0f;

  // init controller
  mBallController.SetDrawConstraints();
} // end GlutViewer::init()

void GlutViewer
  ::draw(void)
{
  ;
} // GlutViewer::draw()

void GlutViewer
  ::beginDraw(void)
{
  m_nPreTick = m_nTick;
  m_nTick = glutGet(GLUT_ELAPSED_TIME);
  m_fElapsedTime = float(m_nTick-m_nPreTick) / 1000.0f;
  m_fTotalTime += m_fElapsedTime;
  m_nDrawTimes ++;
  
  glClearColor(0,0,0,1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  // Enable depth test
  glEnable(GL_DEPTH_TEST);
  
  // Cull backfacing polygons
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  
  GLfloat gldAspect = GLfloat(width())/ GLfloat(height());
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45, gldAspect, 0.1, 1000.0f);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(m_fCameraX,m_fCameraY,m_fCameraZ,0,0,0,0,1,0);
  
  glPushMatrix();

  mBallController.IssueGLrotation();
} // end GlutViewer::beginDraw()

void GlutViewer
  ::endDraw(void)
{
  glPopMatrix();
  
  drawText();

  if(m_nDrawTimes > 10)
  {
    m_nDrawTimes = 0;
    m_fTotalTime = 0;
  }
  
  if(m_fMsgLife > 0 && m_nTick/1000.f - m_fMsgLife > 10.f)
    m_fMsgLife = 0.f;
  
  glutSwapBuffers();
} // end GlutViewer::endDraw()

void GlutViewer
  ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 0x1b:
    {
      exit(0);
      break;
    } // end case esc

    case 'H':
    {
      m_bShowHelp = !m_bShowHelp;
      if(m_bShowHelp)
        makeHelpWindow();
      else
        killHelpWindow();
      break;
    } // end case H

    // the 'enter' key
    case 13:
    {
      // toggle animation
      if(mIsAnimating) stopAnimation();
      else startAnimation();
      break;
    } // end case 'enter'
  } // end switch
} // end GlutViewer::keyPressEvent()

void GlutViewer
  ::mouseEvent(int button, int state, int x, int y)
{
  mMouseButton = button;
  mMouseButtonState = state;
  
  processClick(x,y);
} // end GlutViewer::mouseEvent()

void GlutViewer
  ::motionEvent(int x, int y)
{
  mBallController.MouseMove(int2(x,y));
  updateGL();
} // end GlutViewer::motionEvent()

static void strokeString(int x, int y, const char *msg)
{
  glRasterPos2f( (GLfloat)x, (GLfloat)y);
  int len = (int) strlen(msg);
  for(int i = 0; i < len; i++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, msg[i]);
} // end strokeString()

std::string GlutViewer
  ::helpString(void) const
{
  std::string helpPtr = 
  	"Keys:                                                          \n\
  	 esc   quit                                                     \n\
  	 h     toggle help                                              ";

  return helpPtr;
} // end GlutViewer::helpString()

void GlutViewer
  ::drawHelp(void)
{
  drawText(m_nHelpWinId, helpString());
} // end GlutViewer::drawHelp()

void GlutViewer
  ::drawText(int wndId, const std::string &text)
{
  glutSetWindow(wndId);
  glClear(GL_COLOR_BUFFER_BIT);
  glLineWidth(1);

  std::string temp = text;
  char *pch = strtok(&temp[0], "\n");

  for(int j = 1; pch != 0; j++)
  {
    strokeString(40,  20 + j * 14, pch);
    pch = strtok(0, "\n");
  } // end for j

	glutSwapBuffers();
} // end GlutViewer::drawText()

void GlutViewer
  ::drawMessage(const char* text, int xshift /* = 0 */, int yshift /* = 0 */)
{
  strcpy(m_Msg, text);
  m_nMsgShiftX = xshift;
  m_nMsgShiftY = yshift;
  m_fMsgLife = glutGet(GLUT_ELAPSED_TIME)/1000.f;
} // end GlutViewer::drawMessage()

void GlutViewer
  ::drawFPS(bool bDraw /* = true */)
{
  m_bDrawFPS = bDraw;
} // end GlutViewer::drawFPS()

void GlutViewer
  ::drawText()
{
  if(!(m_fMsgLife>0) && !m_bDrawFPS)
    return;
  
  glPushMatrix();
  glLoadIdentity();
  
  // set matrix
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, width(), height(), 0);
  glLineWidth(1);
  
  glColor3f(1,1,1);
  
  if(m_fMsgLife>0)
    strokeString(MESSAGE_INDENT+m_nMsgShiftX, height()-MESSAGE_INDENT+m_nMsgShiftY, m_Msg);
  
  if(m_bDrawFPS)
  {
    char buf[128];
    sprintf(buf, "%.2f", m_nDrawTimes/m_fTotalTime);
    strokeString(width()-60, 30, buf);
  } // end if
  
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
} // end GlutViewer::drawText()

void GlutViewer
  ::keyPressEventHelp(unsigned char key, int x, int y)
{
  switch(key)
  {
    case 0x1b:
    case 'h':
    case 'H':
    {
      m_bShowHelp = !m_bShowHelp;
      if(!m_bShowHelp)
      	killHelpWindow();
      break;
    } // end case esc, h
  } // end switch
} // end GlutViewer::keyPressEventHelp()

void GlutViewer
  ::makeHelpWindow()
{
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowPosition(m_nHelpWinPos[0], m_nHelpWinPos[1]);
  glutInitWindowSize(m_nHelpWinSize[0], m_nHelpWinSize[1]);
  m_nHelpWinId = glutCreateWindow("Help.");
  
  glClearColor(0.15f, 0.15f, 0.15f, 1);
  glColor3f(1, 1, 1);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, m_nHelpWinSize[0], m_nHelpWinSize[1], 0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  
  /* addCallbacks();  */
  glutKeyboardFunc(keyFuncHelp);
  glutDisplayFunc(renderHelp);
  glutSetWindowTitle("help");
  glutSetIconTitle("help");
} // end GlutViewer::makeHelpWindow()

void GlutViewer
  ::killHelpWindow()
{
  glutSetWindow(m_nHelpWinId);
  glutDestroyWindow(m_nHelpWinId);
  m_nHelpWinId = 0;
} // end GlutViewer::killHelpWindow()

void GlutViewer
  ::updateGL(void)
{
  glutPostRedisplay();
} // end GlutViewer::updateGL()

int GlutViewer
  ::width(void) const
{
  return mWidth;
} // end GlutViewer::width()

int GlutViewer
  ::height(void) const
{
  return mHeight;
} // end GlutViewer::height()

float3 GlutViewer
  ::viewPosition(void)
{
	float4 vp(m_fCameraX, m_fCameraY, m_fCameraZ, 1);
	float4x4 rm = mBallController.GetRotationMatrix();
	vp = rm.inverse()*vp;
	return float3(vp[0], vp[1], vp[2]);
} // end GlutViewer::viewPosition();

void GlutViewer
  ::processClick(int x, int y)
{
  int2 pt(x,y);
  mBallController.UseConstraints(NO_AXES);
  
  if(mMouseButton == GLUT_LEFT_BUTTON
     && mMouseButtonState == GLUT_DOWN
     && !m_bMMDown && !m_bMRDown) 
  {
    mBallController.MouseRotDown(pt);	 
    m_bMLDown = true;
  } // end if
  else if(mMouseButton == GLUT_LEFT_BUTTON
          && mMouseButtonState == GLUT_UP
          && !m_bMMDown && !m_bMRDown) 
  {
    mBallController.MouseRotUp(pt);
    m_bMLDown = false;		
  } // end else if
  else if(mMouseButton == GLUT_MIDDLE_BUTTON
          && mMouseButtonState == GLUT_DOWN
          && !m_bMLDown && !m_bMRDown) 
  {
    mBallController.MouseZoomDown(pt);
    m_bMMDown = true;
  } // end else if
  else if(mMouseButton == GLUT_MIDDLE_BUTTON
          && mMouseButtonState == GLUT_UP
          && !m_bMLDown && !m_bMRDown)  
  {
    mBallController.MouseZoomUp(pt);
    m_bMMDown = false;
  } // end else if
  else if(mMouseButton == GLUT_RIGHT_BUTTON
          && mMouseButtonState == GLUT_DOWN
          && !m_bMLDown && !m_bMMDown) 
  {
    mBallController.MouseTransDown(pt);
    m_bMRDown = true;
  } // end else if 
  else if(mMouseButton == GLUT_RIGHT_BUTTON
          && mMouseButtonState == GLUT_UP
          && !m_bMLDown && !m_bMMDown)  
  {
    mBallController.MouseTransUp(pt);
    m_bMRDown = false;
  } // end else if
} // end GlutViewer::processClick()

bool GlutViewer
  ::animationIsStarted(void) const
{
  return mIsAnimating;
} // end GlutViewer::animationIsStarted()

void GlutViewer
  ::startAnimation(void)
{
  mIsAnimating = true;
  glutIdleFunc(idleFunc);
} // end GlutViewer::startAnimation()

void GlutViewer
  ::stopAnimation(void)
{
  mIsAnimating = false;
  glutIdleFunc(0);
} // end GlutViewer::stopAnimation()
