/**
  @file GCamera.h

  @maintainer Morgan McGuire, matrix@graphics3d.com

  @created 2001-06-02
  @edited  2004-01-13
*/

#ifndef G3D_GCAMERA_H
#define G3D_GCAMERA_H

#include "G3D/CoordinateFrame.h"
#include "G3D/Vector3.h"
#include "G3D/Plane.h"
#include "G3D/debugAssert.h"

namespace G3D {

/**
  There is a viewport of width x height size in world space that corresponds to
  a screenWidth x screenHeight pixel grid on a
  renderDevice->getWidth() x renderDevice->getHeight()
  window.
 */
class GCamera  {
private:

    /**
    Vertical field of view (in radians)
    */
    double						fieldOfView;

    /** 
     The image plane depth corresponding to a vertical field of 
     view, where the film size is 1x1.  
     */
    double						imagePlaneDepth;

    /**
     Clipping plane, *not* imaging plane.  Positive numbers.
     */
    double						nearPlane;

    /**
     Positive 
     */
    double						farPlane;

    CoordinateFrame                                     cframe;

public:

	GCamera();

    virtual ~GCamera();


	CoordinateFrame getCoordinateFrame() const;
	void getCoordinateFrame(CoordinateFrame& c) const;
	void setCoordinateFrame(const CoordinateFrame& c);
           
   /**
	 Sets the horizontal field of view for a square image.  
	 <UL>
	  <LI> toRadians(50) - Telephoto
	  <LI> toRadians(110) - Normal
	  <LI> toRadians(140) - Wide angle
	 </UL>
 	*/
	void setFieldOfView(double angle);

	/**
	Sets the field of view based on a desired image plane depth
	(<I>s'</I>) and film dimensions in world space.  Depth must be positive.  Width,
	depth, and height are measured in the same units (meters are
	recommended).  The field of view will span the diagonal to the
	image.<P> <I>Note</I>: to simulate a 35mm GCamera, set width =
	0.36 mm and height = 0.24 mm.  The width and height used are
	generally not the pixel dimensions of the image.  
	*/
	void setImagePlaneDepth(
        double                                  depth,
        const class Rect2D&                     viewport);

	inline double getFieldOfView() const {
		return fieldOfView;
	}

    /**
     Projects a world space point onto a width x height screen.  The
     returned coordinate uses pixmap addressing: x = right and y =
     down.  The resulting z value is <I>rhw</I>
     */
    G3D::Vector3 project(
        const G3D::Vector3&                     point,
        const class Rect2D&                     viewport) const;

    /**
     Returns the pixel area covered by a shape of the given
     world space area at the given z value (z must be negative).
     */
    double worldToScreenSpaceArea(double area, double z, const class Rect2D& viewport) const;

    /**
     Returns the world space 3D viewport corners.  These
     are at the near clipping plane.  The corners are constructed
     from the nearPlaneZ, getViewportWidth, and getViewportHeight.
     "left" and "right" are from the GCamera's perspective.
     */
    void get3DViewportCorners(
        const class Rect2D&                     viewport,
        Vector3&                                outUR,
        Vector3&                                outUL,
        Vector3&                                outLL,
        Vector3&                                outLR) const;

    /**
     Returns the image plane depth, <I>s'</I>, given the current field
     of view for film of dimensions width x height.  See
     setImagePlaneDepth for a discussion of worldspace values width and height. 
    */
    double getImagePlaneDepth(
        const class Rect2D&                     viewport) const;


    /**
      Returns the world space ray passing through the center of pixel
      (x, y) on the image plane.  The pixel x and y axes are opposite
      the 3D object space axes: (0,0) is the upper left corner of the screen.
      They are in viewport coordinates, not screen coordinates.


      Integer (x, y) values correspond to
      the upper left corners of pixels.  If you want to cast rays
      through pixel centers, add 0.5 to x and y.        
    */
    Ray worldRay(
        double                                  x,
        double                                  y,
        const class Rect2D&                     viewport) const;


    /**
      Returns a negative z-value.
     */
    inline double getNearPlaneZ() const {
        return -nearPlane;
    }

    /**
     Returns a negative z-value.
     */
    inline double getFarPlaneZ() const {
        return -farPlane;
    }

	inline void setFarPlaneZ(double z) {
		debugAssert(z < 0);
		farPlane = -z;
	}

	inline void setNearPlaneZ(double z) {
		debugAssert(z < 0);
		nearPlane = -z;
	}

    /**
     Returns the GCamera space width of the viewport.
     */
    double getViewportWidth(
        const class Rect2D&                     viewport) const;

    /**
     Returns the GCamera space height of the viewport.
     */
    double getViewportHeight(       
        const class Rect2D&                     viewport) const;

    /**
     Read back a GCamera space z-value at pixel (x, y) from the depth buffer.
    double getZValue(
        double			x,
        double			y,
        const class Rect2D&                     viewport,
        double			polygonOffset = 0) const;
     */

    void setPosition(const Vector3& t);

    void lookAt(const Vector3& position, const Vector3& up = Vector3::UNIT_Y);

   /**
    Returns the clipping planes of the frustum, in world space.  The array
    must have six elements allocated.  The planes have normals facing 
    <B>into</B> the view frustum.

    If the far plane is at infinity, the resulting array will have 
    5 planes, otherwise there will be 6.
    */
   void getClipPlanes(
       const Rect2D& viewport,
       Array<Plane>& outClip) const;
};

}

#endif
