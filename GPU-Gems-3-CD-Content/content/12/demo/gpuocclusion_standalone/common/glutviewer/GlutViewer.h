/*!	\file	GlutViewer.h
 *	\author Yuntao Jia, Jared Hoberock
 *	\brief	A class wrapper for glut inspired by
 *	        Gilles Debunne's excellent QGLViewer library.
 *          More details here: http://artis.imag.fr/Members/Gilles.Debunne/QGLViewer/index.html.
 */

#ifndef GLUT_VIEWER_H
#define GLUT_VIEWER_H

#include <string>
#include "BallController.h"
#include "KeyEvent.h"

class GlutViewer
{
  public:	
    /*! This method instantiates Glut, binds callbacks, and starts
     *  the Glut main loop.
     *  \param argc The argument count.
     *  \param argv The array of arguments.
     *  \param title The title of the window.
     *  \param pViewer A pointer to a GlutViewer object to bind to Glut's callbacks.
     */
    static void main(int argc, char **argv, const char *title, GlutViewer* pViewer);

    /*! Singleton access */
    static GlutViewer* getInstance(void);
    
    GlutViewer(void);
    virtual ~GlutViewer(void);
    
    /* static call back function for glut */    
    static void displayFunc(void);	
    static void idleFunc(void);
    static void keyFunc(unsigned char key, int x, int y);
    static void mouseFunc(int button, int state, int x, int y);
    static void motionFunc(int x, int y);
    static void reshapeFunc(int w, int h);

    /*! This method is called upon a glut resize event.
     *  \param w The new width of the window.
     *  \param h The new height of the window.
     */
    virtual void resizeGL(int w, int h);

    /*! This method calls glutPostRedisplay().
     */
    virtual void updateGL(void);

    /*! This method returns the width of this GlutViewer's window.
     *  \return mWidth.
     */
    int width(void) const;

    /*! This method returns the height of this GlutViewer's window.
     *  \return mHeight.
     */
    int height(void) const;

    /*! This method starts the animation.
     */
    virtual void startAnimation(void);

    /*! This method stops the animation.
     */
    virtual void stopAnimation(void);

    /*! This method returns true when this GlutViewer is animating.
     *  \return mIsAnimating
     */
    bool animationIsStarted(void) const;

    /*! This method returns the view position.	 
     */
    float3 viewPosition(void);

    // help window
    static void renderHelp(void);
    static void keyFuncHelp(unsigned char key, int x, int y);
    void makeHelpWindow(void);
    void killHelpWindow(void);

  protected:
    /* interface for children classes */
    virtual void init(void);
    virtual void render(void);
    virtual void draw(void);
    virtual void animate(void);
    virtual void beginDraw(void);
    virtual void endDraw(void);
    virtual void keyPressEvent(KeyEvent *e);
    virtual void mouseEvent(int button, int state, int x, int y);
    virtual void processClick(int x, int y);
    virtual void motionEvent(int x, int y);

    // help
    virtual void drawHelp(void);
    virtual std::string helpString(void) const;
    virtual void drawText(int id, const std::string &text);
    virtual void keyPressEventHelp(unsigned char key, int x, int y);

    // message display
    virtual void drawMessage(const char* text, int xshift = 0, int yshift = 0);	

    // fps
    virtual void drawFPS(bool bDraw = true);
    
    // for camera
    float	m_fCameraX;
    float	m_fCameraY;
    float	m_fCameraZ;

  private:
    // internal method, including message and fps
    void drawText(void);

    // This method sets the dimensions of this GlutViewer
    void setDimensions(const int w, const int h);

    /* instance */
    static bool instanceFlag;
    static GlutViewer *viewer;	

    // for fps
    bool m_bDrawFPS;
    int m_nDrawTimes;
    int m_nTick;
    int m_nPreTick;
    float m_fElapsedTime;
    float m_fTotalTime;
    // for window
    int mWidth;
    int mHeight;
	
    // for message display
    static const int MESSAGE_INDENT = 20;
    int m_nMsgShiftX;
    int m_nMsgShiftY;
    char  m_Msg[256];
    float m_fMsgLife;

    // for help window
    bool	m_bShowHelp;
    int		m_nHelpWinId;
    int		m_nHelpWinPos[2];
    int		m_nHelpWinSize[2];
    char**	helpPtr;

    // For mouse/camera ui
    CBallController mBallController;

    int mMouseButton;
    int mMouseButtonState;
    bool m_bMLDown, m_bMMDown, m_bMRDown;

    bool mIsAnimating;
}; // end GlutViewer

#endif // GLUT_VIEWER_H

