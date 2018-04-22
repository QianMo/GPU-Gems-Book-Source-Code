/**
  @file ManualCameraController.h

  @maintainer Morgan McGuire, morgan@cs.brown.edu

  @created 2002-07-28
  @edited  2003-11-11
*/

#ifndef G3D_MANUALCAMERACONTROLLER_H
#define G3D_MANUALCAMERACONTROLLER_H

#include "graphics3D.h"

namespace G3D {

/**
 Uses a Quake-style mapping to translate keyboard and mouse input
 into a flying camera position.  The result is an Euler-angle ManualCameraController
 suitable for games.  To use:

  <OL>
    <LI> Create a G3D::RenderDevice
    <LI> Create a UserInput object (set the keyboard controls when creating it)
    <LI> Create a ManualCameraController
    <LI> Call ManualCameraController::setActive(true)
    <LI> Invoke ManualCameraController::doSimulation every time simulation is invoked (e.g. once per rendering iteration)
    <LI> Use ManualCameraController::getCoordinateFrame() to set the camera's position
  </OL>
 */
class ManualCameraController {
	
	/** m/s */
	double                      maxMoveRate;

	/** rad/s */
	double                      maxTurnRate;

	double                      yaw;
    double                      pitch;
	Vector3                     translation;

    /** Where the first person camera system thinks the mouse is */
    Vector2                     cameraMouse;

    /**
     Position from which we grabbed the mouse (where the
     window system thinks the mouse is);
     */
    Vector2                     guiMouse;

    class RenderDevice*         renderDevice;

    /** Screen center in pixels */
    Vector2                     center;

    bool                        _active;

    /** Whether the app had focus on the previous call to simulate */
    bool                        appHadFocus;

    class UserInput*            userInput;

public:


	ManualCameraController();

    /** Creates and initializes */
	ManualCameraController(class RenderDevice*, class UserInput*);
    
    /** You need to call setActive(true) before the controller will work. */
    void init(class RenderDevice* device, class UserInput* input);

    /** Deactivates the controller */
    virtual ~ManualCameraController();

    /** When active, the ManualCameraController takes over the mouse.  It turns
        off the mouse cursor and switches to first person controller style.
        Use this to toggle between your menu system and first person camera control.
        In release mode, the cursor movement is restricted to the window
        while the controller is active.  This doesn't occur in debug mode because
        you might hit a breakpoint while the controller is active and it
        would be annoying to not be able to move the mouse.*/
    void setActive(bool a);

    bool active() const;

    /** Initial value is 10 */
    void setMoveRate(double metersPerSecond);

    /** Initial value is PI / 2 */
    void setTurnRate(double radiansPerSecond);

    /** Invoke immediately before entering the main game loop. */
    void reset();

	/**
	 Increments the ManualCameraController's orientation and position.
     Invoke once per simulation step.
	 */
	void doSimulation(
        double                  elapsedTime);

	void setPosition(const Vector3& t) {
		translation = t;
	}

    void lookAt(const Vector3& position);

    double getYaw() const {
        return yaw;
    }

    double getPitch() const {
        return pitch;
    }

	const Vector3& getPosition() const {
		return translation;
	}

	Vector3 getLookVector() const {
		return getCoordinateFrame().getLookVector();
	}

    /** Right vector */
	Vector3 getStrafeVector() const {
		return getCoordinateFrame().getRightVector();
	}

	CoordinateFrame getCoordinateFrame() const;

	void getCoordinateFrame(CoordinateFrame& c) const;

    /**
      Sets to the closest legal controller orientation to the coordinate frame.
    */
    void setCoordinateFrame(const CoordinateFrame& c);
};

}
#endif
