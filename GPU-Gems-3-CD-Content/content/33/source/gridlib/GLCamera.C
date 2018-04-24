#ifdef USE_RCSID
static const char RCSid_GLCamera[] = "$Id: GLCamera.C,v 1.1 2005/10/21 09:46:05 DOMAIN-I15+prkipfer Exp $";
#endif

#include "GLCamera.hh"

#ifdef OUTLINE
#include "GLCamera.in"
#endif


#include "GbMath.hh"
#include <GL/glew.h>

GLCamera::GLCamera()
    : fov_(45.0f)
    , aspect_(1.0f)
    , near_(1.0f)
    , far_(100.0f)
    , focalLength_(50.0f)
    , eyeSeparation_(0.04f)
    , currentEye_(LEFT)
    , eye_(0.0f, 0.0f, 0.0f)
    , lookDir_(0.0f, 0.0f, -1.0f)
    , up_(0.0f, 1.0f, 0.0f)
    , linearSpeed_(1.0f)
    , rotSpeed_(0.01f)
    , modelViewMatrix_()
    , projectionMatrix_()
    , modelViewProjectionMatrix_()
    , edgeT_(0.0f)
	, edgeB_(0.0f)
	, edgeR_(0.0f)
	, edgeL_(0.0f)
    , mode_(MONO)
{
    normalize();
    calculateProjectionMatrix();
    calculateViewMatrix();
}

GLCamera::~GLCamera()
{
}


void 
GLCamera::update(const CameraMovements &cm)
{
    control(cm);

    calculateProjectionMatrix();
    calculateViewMatrix();
    modelViewProjectionMatrix_ = modelViewMatrix_ * projectionMatrix_;

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(&(projectionMatrix_[0][0]));
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(&(modelViewMatrix_[0][0]));
}



void 
GLCamera::control(const CameraMovements &cm)
{
    switch (cm) {
	case NONE:
	    return;
	case MOVE_FORWARD:
	    moveForward();
	    break;
	case MOVE_BACKWARD:
	    moveBackward();
	    break;
	case PITCH_UP:
	    pitchUp();
	    break;
	case PITCH_DOWN:
	    pitchDown();
	    break;
	case MOVE_UP:
	    moveUp();
	    break;
	case MOVE_DOWN:
	    moveDown();
	    break;
	case YAW_LEFT:
	    yawLeft();
	    break;
	case YAW_RIGHT:
	    yawRight();
	    break;
	case MOVE_LEFT:
	    moveLeft();
	    break;
	case MOVE_RIGHT:
	    moveRight();
	    break;
	case ROLL_LEFT:
	    rollLeft();
	    break;
	case ROLL_RIGHT:
	    rollRight();
	    break;
	default:
	    break;
    }
}


void 
GLCamera::calculateProjectionMatrix()
{
    // Set up the projection matrix
    static float m[16];

    // Misc stuff needed for the frustum
    float ratio = aspect_;
    if (mode_ == DUAL_STEREO) ratio *= 0.5f;

    float radians = GbMath<float>::DEG2RAD * fov_ * 0.5f;
    float wd2     = near_ * GbMath<float>::Tan(radians);
    float top     = wd2;
    float bottom  = -wd2;

    // use modelview stack for calculation - it is deeper
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    float left  = -ratio * wd2;
    float right = ratio * wd2;
    float sep   = 0.0f;

    switch (mode_) {

	case STEREO:
	case DUAL_STEREO:
	    sep   = 0.5f * eyeSeparation_ * near_ / focalLength_;
	    if (currentEye_==RIGHT) sep=-sep;
	    left  += sep;
	    right += sep;
	    break;

	case TILE_UPPER_LEFT:
	    right = bottom = 0.0f;
	    break;

	case TILE_UPPER_RIGHT:
	    left = bottom = 0.0f;
	    break;

	case TILE_LOWER_LEFT:
	    right = top = 0.0f;
	    break;

	case TILE_LOWER_RIGHT:
	    left = top = 0.0f;
	    break;

	default:
	    break;
    }

    glFrustum(left+edgeL_,right+edgeR_,bottom+edgeB_,top+edgeT_,near_,far_);
    glGetFloatv(GL_MODELVIEW_MATRIX,m);
    glPopMatrix();
    projectionMatrix_ = GbMatrix4<float>(m);
}

void 
GLCamera::calculateViewMatrix()
{
    static float m[16];

    GbVec3<float> eye(getEye());
    GbVec3<float> lookat(eye + getLookDirection());
    GbVec3<float> up(getUpDirection());

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    if (mode_ == STEREO || mode_ == DUAL_STEREO) {
	// Determine the right eye vector
	GbVec3<float> right(getLookDirection().cross(up));
	right.normalize();
	right *= eyeSeparation_*0.5f;
	if (currentEye_==LEFT) {
	    eye-=right;
	    lookat-=right;
	}
	else {
	    eye+=right;
	    lookat+=right;
	}
    }
    
    gluLookAt(eye[0],eye[1],eye[2],lookat[0],lookat[1],lookat[2],up[0],up[1],up[2]);

    glGetFloatv(GL_MODELVIEW_MATRIX,m);
    glPopMatrix();
    modelViewMatrix_ = GbMatrix4<float>(m);
}


void 
GLCamera::interpolate(GLCamera *one, GLCamera *two, float f)
{
    *this = *one;

    eye_ = one->getEye()*(1.0f-f) + f*two->getEye();
    lookDir_ = one->getLookDirection() * (1.0f-f) + f*two->getLookDirection();
    up_ = one->getUpDirection() * (1.0f-f) + f*two->getUpDirection();

    normalize();

    calculateProjectionMatrix();
    calculateViewMatrix();
	
}


void 
GLCamera::finalize()
{
    normalize();

    calculateProjectionMatrix();
    calculateViewMatrix();
}


GLCamera 
GLCamera::operator + ( const GLCamera& c2) const
{
    GLCamera res;
    res.eye_ = eye_ + c2.eye_;
    res.lookDir_ = lookDir_ + c2.lookDir_;
    res.up_ = up_ + c2.up_;

    return res;
}

GLCamera 
GLCamera::operator * ( float f) const
{
    GLCamera res;
    res.eye_ = eye_ * f;
    res.lookDir_ = lookDir_ * f;
    res.up_ = up_ * f;
    return res;
}

GLCamera 
operator * ( float f, const GLCamera& c )
{
    GLCamera res;
    res.eye_ = c.eye_ * f;
    res.lookDir_ = c.lookDir_ * f;
    res.up_ = c.up_ * f;
    return res;
}

void 
GLCamera::adjustEdge(Edge t, float f)
{
    switch(t) {
	case TOP_EDGE:
	    edgeT_ = f;
	    break;
	case BOTTOM_EDGE:
	    edgeB_ = f;
	    break;
	case LEFT_EDGE:
	    edgeL_ = f;
	    break;
	case RIGHT_EDGE:
	    edgeR_ = f;
	    break;
	default:
	    errormsg("unknown edge : "<<t<<" value: "<<f);
	    break;
    }
    debugmsg("edge adjust "<<edgeT_<<" "<<edgeB_<<" "<<edgeR_<<" "<<edgeL_);

    calculateProjectionMatrix();
}
