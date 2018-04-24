/*! \file BallController.h
 *  \author Jared Hoberock & Yuntao Jia
 *  \brief This class implements a virtual trackball rotation controller suitable for
 *         use in 3D graphical applications.
 *         This implementation "takes inspiration" from Alessandro Falappa's trackball implementation
 */

#ifndef BALL_CONTROLLER_H
#define BALL_CONTROLLER_H

#include <gpcpu/Vector.h>
#include <quaternion/Quaternion.h>

enum AxisSet
{
  NO_AXES,
  CAMERA_AXES,
  BODY_AXES,
  OTHER_AXES
}; // end AxisSet

class CBallController  
{
  private:
    bool bDrawConstraints;
    float4x4 bodyorientation;
    int angleKeyIncrement;
    inline void DrawConstraints(void);
    inline float3* GetUsedAxisSet(void);
    bool bProjectionMethod2;
    bool bDrawBallArea;
    int GLdisplayList;
    Quaternion currentQuat;
    Quaternion previousQuat;
    float radius;
    float winWidth;
    float winHeight;
    float xprev;
    float yprev;
    int2 center;
    bool mouseRotDown;	
    AxisSet whichConstraints;
    int currentAxisIndex;
    float3 cameraAxes[3];
    float3 bodyAxes[3];
    float3* otherAxes;
    int otherAxesNum;
    
    bool mouseZoomDown;
    bool mouseTransDown;
    float transZ, transX, transY;
    
    inline void initVars(void);
    inline void ProjectOnSphere(float3& v) const;
    inline Quaternion RotationFromMove(const float3& vfrom,const float3& vto);
    inline float3 ConstrainToAxis(const float3& loose,const float3& axis);
    inline int NearestConstraintAxis(const float3& loose);

  public:
    inline bool GetDrawConstraints(void);
    inline void SetDrawConstraints(bool flag=true);
    inline void DrawBall(void);
    inline int GetAngleKeyIncrement(void);
    inline void SetAngleKeyIncrement(int ang);
    inline void UseConstraints(AxisSet constraints);
    inline void ToggleMethod(void);
    inline void SetAlternateMethod(bool flag=true);
    inline CBallController(void);
    inline CBallController(const float& rad);
    inline CBallController(const float& rad,const Quaternion& initialOrient);
    inline CBallController(const CBallController& other);
    inline virtual ~CBallController(void);
    inline CBallController& operator=(const CBallController& other);
    inline void Resize(const float& newRadius);
    inline void ClientAreaResize(int left, int top, int right, int bottom);
    inline void MouseRotDown(const int2& location);
    inline void MouseRotUp(const int2& location);
    inline void MouseZoomDown(const int2& location);
    inline void MouseZoomUp(const int2& location);
    inline void MouseTransDown(const int2& location);
    inline void MouseTransUp(const int2& location);
    inline void MouseMove(const int2& location);
    inline void IssueGLrotation(void);
	inline const float4x4 GetRotationMatrix(void);
}; // end CBallController

#include "BallController.inl"

#endif // BALL_CONTROLLER_H
