/*! \file CommonViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a QGLViewer class
 *         containing common operations.
 */

#ifndef COMMON_VIEWER_H
#define COMMON_VIEWER_H

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#endif // WIN32

#include <GL/glew.h>
#include <gl++/texture/Texture.h>
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>

template<typename Parent, typename KeyEventType>
  class CommonViewer
    : public Parent
{
  public:
    /*! \typedef KeyEvent
     *  \brief Make the type of KeyEvent available to children classes.
     */
    typedef KeyEventType KeyEvent;

    /*! This method displays the given texture as a full screen quad.
     *  \param t The Texture of interest.
     */
    inline void drawTexture(const Texture &t) const;

    /*! This method displays the given texture as a full screen quad.
     *  \param t The Texture of interest.
     *  \param p The Program to use to draw the Texture.
     */
    inline void drawTexture(const Texture &t, const Program &p) const;

    inline virtual void init(void);
    inline virtual void keyPressEvent(KeyEvent *e);

    /*! This method prompts the user for a filename.
     *  \param prompt The prompt to display to the user.
     *  \param path The path to begin.
     *  \param desc A string description of files to filter.
     *  \return The complete path to the file as a string.
     */
    inline std::string getOpenFileName(const char *prompt,
                                       const char *path,
                                       const char *desc);

    /*! This method displays the given string on the screen.
     *  \param message The message to display.
     *  \param int delay
     */
    inline void displayMessage(const std::string &message,
                               int delay = 2000);

  protected:
    /*! This method reloads this CommonViewer's shaders.
     */
    inline virtual void reloadShaders(void);

    // for compatibility with Qt
    static const unsigned int SHIFT_MODIFIER   = 0x02000000;
    static const unsigned int CONTROL_MODIFIER = 0x04000000;
    static const unsigned int ALT_MODIFIER     = 0x08000000;
}; // end CommonViewer

#include "CommonViewer.inl"

#endif // COMMON_VIEWER_H

