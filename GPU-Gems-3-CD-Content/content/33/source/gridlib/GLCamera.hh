/*----------------------------------------------------------------------
|
| $Id: GLCamera.hh,v 1.1 2005/10/21 09:46:05 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef _GLCAMERA_H
#define _GLCAMERA_H

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbVec3.hh"
#include "GbMatrix4.hh"

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/


/*!
  \brief Simple camera model
  \author Peter Kipfer
  $Revision: 1.1 $
  $Date: 2005/10/21 09:46:05 $
*/
class GLCamera
{
public:
    // Possible Camera updates
    typedef enum {
	NONE,
	MOVE_FORWARD,
	MOVE_BACKWARD,
	MOVE_LEFT,
	MOVE_RIGHT,
	MOVE_UP,
	MOVE_DOWN,
	PITCH_UP,
	PITCH_DOWN,
	YAW_LEFT,
	YAW_RIGHT,
	ROLL_LEFT,
	ROLL_RIGHT
    } CameraMovements;

    typedef enum {
	MONO,
	STEREO,
	DUAL_STEREO,
	TILE_UPPER_LEFT,
	TILE_UPPER_RIGHT,
	TILE_LOWER_LEFT,
	TILE_LOWER_RIGHT
    } CameraMode;

    typedef enum {
	LEFT,
	RIGHT
    } EyeMode;

    typedef enum {
		TOP_EDGE,
		BOTTOM_EDGE,
		LEFT_EDGE,
		RIGHT_EDGE
    } Edge;

    GLCamera();
    ~GLCamera();

    INLINE void setMode(CameraMode m);
    INLINE CameraMode getMode() const;
    INLINE void setEyeMode(EyeMode m);
    INLINE EyeMode getEyeMode() const;

    void update(const CameraMovements &);
    void control(const CameraMovements &);
    void interpolate(GLCamera *one, GLCamera *two, float f);

    void finalize();

    INLINE void normalize();
    INLINE GbMatrix4<float> getViewProjectionMatrix() const;
    INLINE GbMatrix4<float> getProjectionMatrix() const;
    INLINE GbMatrix4<float> getViewMatrix() const;

    INLINE float getSpeed() const;
    INLINE void setSpeed(float speed);
    INLINE float getRotationSpeed()  const;
    INLINE void setRotationSpeed(float rspeed);

    INLINE const GbVec3<float> getEye() const;
    INLINE void setEye(const GbVec3<float> &eye);
    INLINE const GbVec3<float> getLookDirection() const;
    INLINE void setLookDirection(const GbVec3<float> &look);
    INLINE const GbVec3<float> getUpDirection() const;
    INLINE void setUpDirection(const GbVec3<float> &up);

    INLINE void setFrustumParam(float fov, float as, float n, float f);
    INLINE void setZoom(float fov);
    INLINE float getZoom() const;
    INLINE void setAspect(float as);
    INLINE float getAspect() const;
    INLINE void setNearFarPlane(float n, float f);
    INLINE float getNearPlane() const;
    INLINE float getFarPlane() const;
    INLINE void setStereoParam(float fl, float es);
    INLINE float getStereoFocalLength() const;
    INLINE float getStereoEyeSeparation() const;

    INLINE void faster();
    INLINE void slower();


    GLCamera operator + ( const GLCamera&) const;
    GLCamera operator * ( float ) const;
    friend GLCamera operator * ( float , const GLCamera& );


    INLINE void moveForward();
    INLINE void moveBackward();
    INLINE void pitch(float theta);
    INLINE void pitchUp();
    INLINE void pitchDown();
    INLINE void moveUp();
    INLINE void moveDown();
    INLINE void yaw(float theta);
    INLINE void yawLeft();
    INLINE void yawRight();
    INLINE void moveRight(int sign = -1);
    INLINE void moveLeft();
    INLINE void roll(float theta);
    INLINE void rollLeft();
    INLINE void rollRight();

    void adjustEdge(Edge t, float f);

private:
    void calculateProjectionMatrix();
    void calculateViewMatrix();

    float fov_;			// field of view in degrees
    float aspect_;		// aspect
    float near_;		// near
    float far_;			// far
    float focalLength_;         // stereo fusion distance
    float eyeSeparation_;       // interocular distance
    EyeMode currentEye_;

    GbVec3<float> eye_;
    GbVec3<float> lookDir_;
    GbVec3<float> up_;

    float linearSpeed_;
    float rotSpeed_;

    GbMatrix4<float> modelViewMatrix_;
    GbMatrix4<float> projectionMatrix_;
    GbMatrix4<float> modelViewProjectionMatrix_;
    float edgeT_,edgeB_,edgeR_,edgeL_;

    CameraMode mode_;
};

#ifndef OUTLINE
#include "GLCamera.in"
#endif  // OUTLINE


#endif
