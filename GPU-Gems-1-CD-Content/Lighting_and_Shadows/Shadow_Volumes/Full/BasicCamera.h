/**
  @file BasicCamera.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)


*/

#ifndef BasicCamera_H
#define BasicCamera_H

#include <G3DAll.h>


/**
 This class stores information about the current BasicCamera position
 and orientation in world space.  Some information about the near
 clipping plane and field of view is stored in the Viewport class.
*/
class BasicCamera
{
public:
    
    BasicCamera();

    virtual ~BasicCamera() { }

    Vector3 getViewVector() const;
    G3D::CoordinateFrame getWorldToCamera() const;

    // get corners of viewport (on the near clipping plane) in worldspace
    void get3DViewportCorners(
        double              nearX,
        double              nearY,
        double              nearZ,
        Vector3&            ur,
        Vector3&            ul,
        Vector3&            ll,
        Vector3&            lr) const;

    void updateCamera(
        int                 xDirection,
        int                 zDirection,
        int                 mouseX,
        int                 mouseY);

    void orient(
        const Vector3&      eye,
        const Vector3&      look);

	// this is public for sheer ease of use
    G3D::CoordinateFrame m_transformation;

private:

    double                  m_yaw, m_pitch;
    double                  m_maxMoveRate, m_maxTurnRate;
    double                  m_prevMouseX, m_prevMouseY;
    double                  m_centerX, m_centerY;

};

#endif

