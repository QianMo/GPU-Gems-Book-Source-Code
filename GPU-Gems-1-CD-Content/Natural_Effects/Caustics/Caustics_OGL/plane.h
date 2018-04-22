#ifndef _PLANE_INC
#define _PLANE_INC

#include "point.h"


class plane
{
public:
	
	float a,b,c,d;
	point n;

	plane();

	// constructors from a,b,c,d
	plane(float,float,float,float);
	void create(float,float,float,float);

	// constructors from 3 passing points
	plane(point,point,point);
	void create(point,point,point);
	void negate();

	// point comes in xyz. The result indicates the distance
	// to the plane's oriented surface from the point. Values
	// >0 mean point outside (in the normal side) of the plane.
	// <0 mean inside, and =0 mean point on surface.
	double testpoint (point);
	

	// line comes in the form of passing point + director vector. 
	// result is 0 if line parallel to plane. !0 if colliding,
	// and then the third param returns the interesction point
	double testline(point,point,point &);

	
	int magicnumber (point);
	int testtri (point,point,point);
	
	point getpointfromplane();

	double evalxy(double,double);
	double evalxz(double,double);
	double evalyz(double,double);

	void operator=(plane);        // copy
	~plane();

};

#endif