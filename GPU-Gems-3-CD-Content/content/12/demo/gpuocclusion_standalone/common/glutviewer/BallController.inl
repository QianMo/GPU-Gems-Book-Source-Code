/*! \file BallController.inl
 *  \author Jared Hoberock & Yuntao Jia
 *  \brief Inline file for BallController.h.
*/

//-----------------------------------------------------------------------------
// BallController.cpp: implementation of the CBallController class.

#ifndef NOMINMAX
#define NOMINMAX 1
#endif // NOMINMAX

#include "BallController.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>

CBallController
  ::~CBallController()
{
  if(otherAxes) delete[] otherAxes;
} // end CBallController::~CBallController()

CBallController
  ::CBallController(const CBallController& other)
{
  *this=other;
} // end CBallController:CBallController()

void CBallController
  ::Resize(const float& newRadius)
{
  radius=newRadius;
} // end CBallController::Resize()

void CBallController
  ::ClientAreaResize(int left, int top, int right, int bottom)
{
  winWidth=float(right-left);
  winHeight=float(bottom-top);
  center=int2( (right-left)/2 , (bottom-top)/2);
} // end CBallController::ClientAreaResize()

inline CBallController
  ::CBallController()
{
  initVars();
} // end CBallController::CBallController()

void CBallController
  ::SetAlternateMethod(bool flag)
{
  bProjectionMethod2=flag;
} // end CBallController::SetAlternateMethod()

void CBallController
  ::ToggleMethod(void)
{
  if(bProjectionMethod2) bProjectionMethod2=false;
  else bProjectionMethod2=true;
} // end CBallController::ToggleMethod()

void CBallController
  ::UseConstraints(AxisSet constraints)
{
  whichConstraints=constraints;
} // end CBallController::UseConstraints()

int CBallController
  ::GetAngleKeyIncrement(void)
{
  return angleKeyIncrement;
} // end CBallController::GetAngleKeyIncrement()

void CBallController
  ::SetAngleKeyIncrement(int ang)
{
  angleKeyIncrement=abs(ang)%360;
} // end CBallController::SetAngleKeyIncrement()

bool CBallController
  ::GetDrawConstraints(void)
{
  return bDrawConstraints;
} // end CBallController::GetDrawConstraints()

void CBallController
  ::SetDrawConstraints(bool flag)
{
  bDrawConstraints=flag;
} // end CBallController::SetDrawConstraints()

CBallController
  ::CBallController(const float& rad)
{
  initVars();
  radius = std::max(0.1f, std::min(1.0f, rad));
} // end CBallController::CBallController()

CBallController
  ::CBallController(const float& rad,const Quaternion& initialOrient)
{
  initVars();
  radius = std::max(0.1f, std::min(1.0f, rad));
  currentQuat=initialOrient;
} // end CBallController::CBallController()

CBallController& CBallController
  ::operator=(const CBallController& other)
{
  if(this==&other) return *this;
  initVars();
  currentQuat=other.currentQuat;
  previousQuat=other.previousQuat;
  radius=other.radius;
  winWidth=other.winWidth;
  winHeight=other.winHeight;
  otherAxesNum=other.otherAxesNum;
  otherAxes=new float3[otherAxesNum];
  for(int c=0;c<otherAxesNum;c++) otherAxes[c]=other.otherAxes[c];
  return *this;
} // end CBallController::operator=()

void CBallController
  ::MouseRotDown(const int2& location)
{
  xprev=(2*location[0]-winWidth)/winWidth;
  yprev=(winHeight-2*location[1])/winHeight;
  previousQuat=currentQuat;
  mouseRotDown=true;
  bDrawBallArea=bProjectionMethod2;// draw circle only if method 2 active
} // end CBallController::MouseRotDown()

void CBallController
  ::MouseRotUp(const int2& location)
{
  mouseRotDown=false;
  xprev=yprev=0.0;
  bDrawBallArea=false;
  // save current rotation axes for bodyAxes constraint at next rotation
  bodyorientation=currentQuat.getRotationMatrix();
  bodyAxes[0]=float3(bodyorientation(0,0),bodyorientation(1,0),bodyorientation(2,0));
  bodyAxes[1]=float3(bodyorientation(0,1),bodyorientation(1,1),bodyorientation(2,1));
  bodyAxes[2]=float3(bodyorientation(0,2),bodyorientation(1,2),bodyorientation(2,2));
} // end CBallController::MouseRotUp()

void CBallController
  ::MouseZoomDown(const int2& location)
{
  mouseZoomDown = true;
  xprev=(2*location[0]-winWidth)/winWidth;
  yprev=(winHeight-2*location[1])/winHeight;
} // end CBallController::MouseZoomDown()

void CBallController
  ::MouseZoomUp(const int2& location)
{
  mouseZoomDown = false;
  xprev=yprev=0.0;
} // end CBallController::MouseZoomUp()

void CBallController
  ::MouseTransDown(const int2& location)
{
  xprev=(2*location[0]-winWidth)/winWidth;
  yprev=(winHeight-2*location[1])/winHeight;
  mouseTransDown = true;
} // end CBallController::MouseTransDown()

void CBallController
  ::MouseTransUp(const int2& location)
{
  mouseTransDown = false;
  xprev=yprev=0.0;
} // end CBallController::MouseTransUp()

void CBallController
  ::MouseMove(const int2& location)
{
  float xcurr=(2*location[0]-winWidth)/winWidth;
  float ycurr=(winHeight-2*location[1])/winHeight;
  float3 vfrom(xprev,yprev,0);
  float3 vto(xcurr,ycurr,0);
  if(mouseRotDown)
  {
    // find the two points on sphere according to the projection method
    ProjectOnSphere(vfrom);
    ProjectOnSphere(vto);
    // modify the vectors according to the active constraint
    if(whichConstraints != NO_AXES)
    {
      float3* axisSet=GetUsedAxisSet();
      vfrom = ConstrainToAxis(vfrom,axisSet[currentAxisIndex]);
      vto = ConstrainToAxis(vto,axisSet[currentAxisIndex]);
    } // end if

    // get the corresponding unitquaternion
    Quaternion lastQuat=RotationFromMove(vfrom,vto);
    currentQuat*=lastQuat;
    xprev=xcurr;
    yprev=ycurr;
  } // end if
  else if(mouseZoomDown)
  {
    transZ += (ycurr-yprev)*20;
    xprev=xcurr;
    yprev=ycurr;
  } // end else if
  else if(mouseTransDown)
  {
    transX += (xcurr-xprev)*10;
    transY += (ycurr-yprev)*10;
    xprev=xcurr;
    yprev=ycurr;
  } // end else if
  else if(whichConstraints != NO_AXES)
  {
    ProjectOnSphere(vto);
    currentAxisIndex=NearestConstraintAxis(vto);
  } // end else if
}

void CBallController::IssueGLrotation(void)
{
  glTranslatef(transX,transY,transZ);
  glMultTransposeMatrixf(currentQuat.getRotationMatrix());
} // end CBallController::IssueGLrotation()

void CBallController::ProjectOnSphere(float3& v) const
{
  float rsqr=radius*radius;
  float dsqr=v[0]*v[0]+v[1]*v[1];
  if(bProjectionMethod2)
  {
    // if inside sphere project to sphere else on plane
    if(dsqr>rsqr)
    {
      register float scale=(radius-.05)/sqrt(dsqr);
      v[0]*=scale;
      v[1]*=scale;
      v[2]=0;
    } // end if
    else
    {
      v[2]=sqrt(rsqr-dsqr);
    } // end else
  } // end if
  else
  {
    // if relatively "inside" sphere project to sphere else on hyperbolic sheet
    if(dsqr<(rsqr*0.5))	v[2]=sqrt(rsqr-dsqr);
    else v[2]=rsqr/(2*sqrt(dsqr));
  } // end else
} // end CBallController::ProjectOnSphere()

Quaternion CBallController
  ::RotationFromMove(const float3& vfrom,const float3& vto)
{
  if(bProjectionMethod2)
  {
    Quaternion q;
    q[0]=vfrom[2]*vto[1]-vfrom[1]*vto[2];
    q[1]=vfrom[0]*vto[2]-vfrom[2]*vto[0];
    q[2]=vfrom[1]*vto[0]-vfrom[0]*vto[1];
    q[3]=vfrom.dot(vto);
    return Quaternion(q);
  } // end if
  else
  {
    // calculate axis of rotation and correct it to avoid "near zero length" rot axis
    float3 rotaxis=vto.cross(vfrom);
    if(rotaxis.absDot(rotaxis) < 1e-6f) rotaxis = float3(1,0,0);
    // find the amount of rotation
    float3 d(vfrom-vto);
    float t=d.length()/(2.0*radius);
    t = std::max(-1.0f, std::min(1.0f, t));
    float phi=2.0*asin(t);
    return Quaternion(phi,rotaxis);
  } // end else
} // end CBallController::RotationFromMove()

static inline float DegToRad(const float degrees)
{
  return 0.0174532925f * degrees;
} // end DegToRad()

void CBallController
  ::initVars(void)
{
  winWidth=winHeight=0;
  previousQuat=currentQuat=Quaternion(0,float3(1,0,0));
  mouseRotDown=mouseZoomDown=mouseTransDown=bDrawBallArea=bProjectionMethod2=bDrawConstraints=false;
  xprev=yprev=0.0;
  center=int2(0,0),
  radius=0.6;
  GLdisplayList=currentAxisIndex=otherAxesNum=0;
  otherAxes=NULL;
  whichConstraints=NO_AXES;
  cameraAxes[0]=bodyAxes[0]=float3(1,0,0);
  cameraAxes[1]=bodyAxes[1]=float3(0,1,0);
  cameraAxes[2]=bodyAxes[2]=float3(0,0,1);
  bodyorientation = float4x4::identity();
  angleKeyIncrement=5;
  
  transZ = transX = transY = 0.0;
} // end CBallController::initVars()

float3 CBallController
  ::ConstrainToAxis(const float3& loose,const float3& axis)
{
  float3 onPlane;
  register float norm;
  onPlane = loose-axis* axis.dot(loose);
  norm = onPlane.length();
  if(norm > 0)
  {
    if (onPlane[2] < 0.0) onPlane = -onPlane;
    return (onPlane/=sqrt(norm) );
  } // end if

  if(axis[2] == 1)
  {
    onPlane = float3(1,0,0);
  } // end if
  else
  {
    onPlane = float3(-axis[1], axis[0], 0);
    onPlane = onPlane.normalize();
  } // end else
  return onPlane;
} // end CBallController::ConstrainToAxis()

int CBallController
  ::NearestConstraintAxis(const float3& loose)
{
  float3* axisSet=GetUsedAxisSet();
  float3 onPlane;
  register float max, dot;
  register int i, nearest;
  max = -1; 
  nearest = 0;
  if(whichConstraints == OTHER_AXES)
  {
    for(i=0; i<otherAxesNum; i++)
    {
      onPlane = ConstrainToAxis(loose, axisSet[i]);
      dot = onPlane.dot(loose);
      if (dot>max) max = dot; nearest = i;
    } // end for i
  } // end if
  else
  {
    for(i=0; i<3; i++)
    {
      onPlane = ConstrainToAxis(loose, axisSet[i]);
      dot = onPlane.dot(loose);
      if(dot>max)
      {
        max = dot;
        nearest = i;
      } // end if
    } // end for i
  } // end else

  return (nearest);
} // end CBallController::NearestConstraintAxis()

float3* CBallController
  ::GetUsedAxisSet(void)
{
  float3* axes=NULL;
  switch(whichConstraints)
  {
    case CAMERA_AXES:
      axes=cameraAxes;
      break;

    case BODY_AXES:
      axes=bodyAxes;
      break;

    case OTHER_AXES:
      axes=otherAxes;
      break;
  } // end switch

  return axes;
} // end CBallController::GetUsedAxisSet()

const float4x4 CBallController
 ::GetRotationMatrix(void)
{
	return currentQuat.getRotationMatrix();
} // end CBallController::GetRotationMatrix()
