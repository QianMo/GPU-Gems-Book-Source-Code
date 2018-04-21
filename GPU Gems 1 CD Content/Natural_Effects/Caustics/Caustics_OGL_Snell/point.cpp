// Disable warning for loss of data
#pragma warning( disable : 4244 )  

#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "point.h"

// Constructors
point::point()
{
#ifdef DEBUGLOW
printf("point::point1\n");
#endif

x=0.0;
y=0.0;
z=0.0;
}


point::point(double px, double py, double pz)
{
#ifdef DEBUGLOW
printf("point::point2\n");
#endif

x=px;
y=py;
z=pz;
}

point::point(text &t)
{
#ifdef DEBUGLOW
printf("point::point3\n");
#endif

x=t.getfloat();
y=t.getfloat();
z=t.getfloat();
}


point::point(point &p)
{
#ifdef DEBUGLOW
printf("point::point4\n");
#endif

x=p.x;
y=p.y;
z=p.z;
}


// Class methods

void point::create(double px, double py, double pz)
{
#ifdef DEBUGLOW
printf("point::create\n");
#endif

x=px;
y=py;
z=pz;
}


void point::load(text &t)
{
#ifdef DEBUGLOW
printf("point::load\n");
#endif

x=t.getfloat();
y=t.getfloat();
z=t.getfloat();
}

// Operators


double point::modulo()
{
#ifdef DEBUGLOW
printf("point::modulo\n");
#endif

double res;

res=x*x + y*y + z*z;
res=sqrt(res);
return res;
}


double point::modulosq()
{
#ifdef DEBUGLOW
printf("point::modulosq\n");
#endif

double res;

res=x*x + y*y + z*z;
return res;
}


double point::distance(point p)
{
#ifdef DEBUGLOW
printf("point::distance\n");
#endif

double res;

res=sqrt( (p.x-x)*(p.x-x) + (p.y-y)*(p.y-y) + (p.z-z)*(p.z-z) );
return res;
}



double point::distanceman(point p)
{
#ifdef DEBUGLOW
printf("point::distance\n");
#endif

double res;

res=fabs(p.x-x)+ fabs(p.y-y) + fabs(p.z-z);
return res;
}


double point::distancemanxz(point p)
{
double res;

res=fabs(p.x-x)+ fabs(p.z-z);
return res;
}


void point::normalize()
{
#ifdef DEBUGLOW
printf("point::normalize\n");
#endif

double m;

m=modulo();
if (m!=0.0)
	{
	x=x/m;
	y=y/m;
	z=z/m;
	}
else 
	{
	x=0;
	y=0;
	z=0;
	}
}


point point::operator+(point p)
{
#ifdef DEBUGLOW
printf("point::operator+\n");
#endif

point res;


res.x=x+p.x;
res.y=y+p.y;
res.z=z+p.z;
return res;
}

point point::operator-(point p)
{
#ifdef DEBUGLOW
printf("point::operator-\n");
#endif

point res;
res.x=x-p.x;
res.y=y-p.y;
res.z=z-p.z;
return res;
}

int point::operator==(point p)
{
#ifdef DEBUGLOW
printf("point::operator==\n");
#endif

return ((p.x==x) && (p.y==y) && (p.z==z));
}

int point::operator!=(point p)
{
#ifdef DEBUGLOW
printf("point::operator!=\n");
#endif

return ((p.x!=x) || (p.y!=y) || (p.z!=z));
}

point point::operator*(double d)
{
#ifdef DEBUGLOW
printf("point::operator*1\n");
#endif

point res;

res.x=x*d;
res.y=y*d;
res.z=z*d;
return res;
}


point point::operator/(double d)
{
#ifdef DEBUGLOW
printf("point::operator*1\n");
#endif

point res;

res.x=x/d;
res.y=y/d;
res.z=z/d;
return res;
}


double point::operator*(point p)
{
#ifdef DEBUGLOW
printf("point::operator*2\n");
#endif

return (p.x*x + p.y*y + p.z*z);
}


point point::operator^(point p)
{
#ifdef DEBUGLOW
printf("point::operator^\n");
#endif

point res;
res.x=y*p.z - z*p.y;
res.y=z*p.x - x*p.z;
res.z=x*p.y - y*p.x;
return res;
}


void point::operator=(point p)
{
#ifdef DEBUGLOW
printf("point::operator=\n");
#endif

x=p.x;
y=p.y;
z=p.z;
}



void point::negate()
{
#ifdef DEBUGLOW
printf("point::negate\n");
#endif

x=-x;
y=-y;
z=-z;
}


int point::into(point p1,point p2)
{
#ifdef DEBUGLOW
printf("point::into\n");
#endif

int res;

res=(x>=p1.x) && (y>=p1.y) && (z>=p1.z) &&
    (x<=p2.x) && (y<=p2.y) && (z<=p2.z);
return res;
}


point point::interpolate(point p,double fp,point q,double fq)
{
#ifdef DEBUGLOW
printf("point::interpolate\n");
#endif

point res;

res=(p*fp)+(q*fq);
(*this)=res;
return res;
}


float point::distancePointLine(point a,point b){
	float distance = 0;
	point director;
	point p;
	director = b - a;
	p.create(x,y,z);
	distance = (director^(p-a)).modulo()/director.modulo();
	return distance;
}


void point::rotatex(double angle)
{
double sinANG,cosANG,tmpY,tmpZ;
sinANG = sin(angle);
cosANG = cos(angle);
tmpY = y;
tmpZ = z;
y = tmpY*cosANG + tmpZ*sinANG;
z = -tmpY*sinANG + tmpZ*cosANG;
}


void point::rotatey(double angle)
{
double sinANG,cosANG,tmpX,tmpZ;
sinANG = sin(angle);
cosANG = cos(angle);
tmpX = x; 
tmpZ = z;
x = tmpX*cosANG - tmpZ*sinANG;
z = tmpX*sinANG + tmpZ*cosANG;
}


void point::rotatey(float sin,float cos)
{
double tmpX,tmpZ;
tmpX = x; 
tmpZ = z;
x = tmpX*cos - tmpZ*sin;
z = tmpX*sin + tmpZ*cos;
}


void point::rotatez(double angle)
{
double sinANG,cosANG,tmpX,tmpY;
sinANG = sin(angle);
cosANG = cos(angle);
tmpX = x; 
tmpY = y;
x = tmpX*cosANG + tmpY*sinANG;
y = -tmpX*sinANG + tmpY*cosANG;
}


point::~point()
{
#ifdef DEBUGLOW
printf("point::~point\n");
#endif
}


bool point::infrustum(point viewer,double yaw,double fov)
{
point pyaw(cos(yaw),0,sin(yaw));
point view=(*this)-viewer;
view.normalize();
double cangle=pyaw*view;
return (cangle>cos(yaw));
}