/*! \file CommonViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CommonViewer.h.
 */

#include "CommonViewer.h"

template<typename Parent, typename KeyEvent>
  void CommonViewer<Parent,KeyEvent>
    ::init(void)
{
  Parent::init();
  reloadShaders();
} // end CommonViewer::init()

template<typename Parent, typename KeyEvent>
  void CommonViewer<Parent,KeyEvent>
    ::drawTexture(const Texture &t,
                  const Program &p) const
{
  glPushAttrib(GL_DEPTH_BUFFER_BIT |
               GL_LIGHTING_BIT | GL_TRANSFORM_BIT);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  t.bind();

  p.bind();
  printGLError(__FILE__, __LINE__);

  glBegin(GL_QUADS);
  glTexCoord2f(0,0);
  glVertex2f(-1,-1);
  glTexCoord2f(t.getMaxS(), 0);
  glVertex2f(1,-1);
  glTexCoord2f(t.getMaxS(), t.getMaxT());
  glVertex2f(1,1);
  glTexCoord2f(0, t.getMaxT());
  glVertex2f(-1,1);
  glEnd();

  p.unbind();

  t.unbind();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glPopAttrib();
} // end CommonViewer::drawTexture()

template<typename Parent, typename KeyEvent>
  void CommonViewer<Parent,KeyEvent>
    ::drawTexture(const Texture &t) const
{
  const Program &p = (t.getTarget() == GL_TEXTURE_2D_ARRAY_EXT) ? mTexture2DArrayProgram : mTexture2DRectProgram;
  drawTexture(t,p);
} // end CommonViewer::drawTexture()

template<typename Parent, typename KeyEvent>
  void CommonViewer<Parent,KeyEvent>
    ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 'R':
    {
      reloadShaders();
      updateGL();
      break;
    } // end case Qt::Key_R

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end CommonViewer::keyPressEvent()

template<typename Parent, typename KeyEvent>
  void CommonViewer<Parent,KeyEvent>
    ::reloadShaders(void)
{
  ;
} // end CommonViewer::reloadShaders()

template<typename Parent, typename KeyEvent>
  std::string CommonViewer<Parent,KeyEvent>
    ::getOpenFileName(const char *prompt,
                      const char *path,
                      const char *desc)
{
  std::string result;
  std::cout << prompt << std::endl;
  std::cin >> result;

  return result;
} // end CommonViewer::getOpenFileName()

template<typename Parent, typename KeyEvent>
  void CommonViewer<Parent,KeyEvent>
    ::displayMessage(const std::string &message,
                     int delay)
{
  drawMessage(message.c_str());
} // end CommonViewer::displayMessage()

