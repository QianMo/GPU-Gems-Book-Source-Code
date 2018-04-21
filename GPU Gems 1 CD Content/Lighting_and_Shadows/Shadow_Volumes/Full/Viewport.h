/**
  @file Viewport.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#ifndef VIEWPORT_H
#define VIEWPORT_H


/**
 This class stores information about the viewport.  More information
 regarding the location and orientation of the camera can be found
 in the camera class.
*/
class Viewport {
public:

	Viewport(
        double                              screenWidth,
        double                              screenHeight, 
        double                              fieldOfView,
        double                              nearPlane);

    virtual ~Viewport() { }

    /**
     * get x, y, z coordinates of viewport (on near clip plane)
     * in camera space
     */
	void getNearXYZ(
        double&                             x,
        double&                             y,
        double&                             z) const;

    /**
     * get an infinite frustum matrix in GL-style column-major order
     */
    void getInfiniteFrustumMatrix(
        double*                             mat) const;


    /**
     * Set the current GL matrix to be a projection matrix with
     * the far clipping plane at inifinity
     */
	void setInfiniteFrustum();


private:
	double                  m_screenWidth, m_screenHeight;
	double                  m_fieldOfView, m_nearPlane;

};

#endif
